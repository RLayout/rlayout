use super::prepare::{ArrayInstances, GpuVertex, MeshData, Packet, TileInfo};
use super::*;
use egui_wgpu::{Callback, CallbackResources, CallbackTrait, RenderState, ScreenDescriptor, wgpu};
use std::collections::{BTreeMap, HashSet, VecDeque};
#[cfg(not(target_arch = "wasm32"))]
use std::future::Future;
#[cfg(not(target_arch = "wasm32"))]
use std::pin::Pin;
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc,
};
#[cfg(not(target_arch = "wasm32"))]
use std::task::{Context, Poll, Waker};
use std::time::Duration;
use wgpu::util::DeviceExt;

const GPU_LIMIT: usize = 256 * 1024 * 1024;
const UPLOAD_LIMIT: usize = 8 * 1024 * 1024;
const TRIANGLE_LIMIT: usize = 250_000;
const DRAW_LIMIT: usize = 2_048;
const IN_FLIGHT_LIMIT: usize = 2;
const STORAGE_WINDOW: usize = 4096 * std::mem::size_of::<StorageInstance>();
const STAGING_RESERVE: usize = UPLOAD_LIMIT * IN_FLIGHT_LIMIT;
const UPLOAD_CPU_RESERVE: usize =
    UPLOAD_LIMIT + STORAGE_WINDOW + DRAW_LIMIT * std::mem::size_of::<EncodedDraw>();

#[derive(Clone, Debug, Default)]
pub struct RenderStatus {
    pub complete: bool,
    pub shape_count: usize,
    pub error: Option<String>,
    pub tile_count: usize,
    pub has_image: bool,
    pub reprojected_tiles: usize,
    pub first_tile_seconds: Option<f64>,
    pub first_useful_seconds: Option<f64>,
    pub has_geometry: bool,
    pub preparation_seconds: f64,
    pub request_elapsed_seconds: f64,
    pub gpu_completion_latency_seconds: f64,
    pub completed_gpu_seconds: Option<f64>,
    pub upload_bytes: usize,
    pub cpu_bytes: usize,
    pub gpu_bytes: usize,
    pub in_flight_frames: usize,
    pub submitted_triangles: usize,
    pub draw_calls: usize,
    pub last_frame_draw_calls: usize,
    pub last_frame_upload_bytes: usize,
    pub max_frame_upload_bytes: usize,
    pub max_frame_draws: usize,
    pub max_frame_triangles: usize,
    pub mask_passes: usize,
    pub presented_camera: Option<Camera>,
    pub last_presented_at: Option<Instant>,
}

struct Allocation {
    bytes: usize,
    ledger: Arc<AtomicUsize>,
}
impl Drop for Allocation {
    fn drop(&mut self) {
        self.ledger.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}
struct GpuBuffer {
    buffer: wgpu::Buffer,
    _allocation: Allocation,
}
struct GpuTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    _allocation: Allocation,
}
struct GpuMesh {
    vertices: Arc<GpuBuffer>,
    indices: Arc<GpuBuffer>,
    _glyph: Arc<GpuTexture>,
    glyph_group: wgpu::BindGroup,
    index_count: u32,
    used: AtomicBool,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StorageInstance {
    affine: [f32; 4],
    translation: [f32; 2],
    padding: [f32; 2],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct JobUniform {
    size: [f32; 2],
    columns: u32,
    rows: u32,
    column_step: [f32; 2],
    row_step: [f32; 2],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Palette {
    colors: [[u32; 4]; 64],
    count: [u32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PresentationUniform {
    rectangle: [f32; 4],
    uv_rectangle: [f32; 4],
    size: [f32; 2],
    srgb_target: u32,
    padding: u32,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct TileKey {
    revision: u64,
    selected: Option<CellId>,
    depth: u16,
    scale: u64,
    pixels_per_point: u32,
    text_pass: u64,
    grid: [i64; 2],
    phase: [u64; 2],
    viewport_phase: [u64; 2],
    density: u64,
    size: [u32; 2],
    preview: bool,
    styles: Vec<(LayerKey, [u8; 4], bool)>,
}
impl TileKey {
    fn new(scene: &Scene, key: &FrameKey, tile: TileInfo) -> Self {
        Self {
            revision: scene.revision,
            selected: key.selected,
            depth: key.depth,
            scale: key.camera.pixels_per_unit.to_bits(),
            pixels_per_point: key.pixels_per_point.to_bits(),
            text_pass: key.text_pass,
            grid: tile.grid,
            phase: tile.phase.map(f64::to_bits),
            viewport_phase: [key.viewport.min.x, key.viewport.min.y].map(|coordinate| {
                let phase =
                    (f64::from(coordinate) * f64::from(key.pixels_per_point)).rem_euclid(1.0);
                if phase == 0.0 { 0 } else { phase.to_bits() }
            }),
            density: tile.density.to_bits(),
            size: tile.size,
            preview: tile.preview,
            styles: key
                .styles
                .iter()
                .map(|(layer, style)| (*layer, style.color.to_array(), style.visible))
                .collect(),
        }
    }
}
#[derive(Clone)]
struct VisibleTile {
    info: TileInfo,
    key: FrameKey,
    image: Arc<GpuTexture>,
    shape_count: usize,
    ready: Arc<AtomicBool>,
    valid: Arc<AtomicBool>,
    has_geometry: bool,
}
struct ActiveTile {
    info: TileInfo,
    cache_key: TileKey,
    cached: Option<VisibleTile>,
    layers: Vec<(LayerKey, LayerStyle)>,
    masks: Option<Arc<GpuTexture>>,
    mask_views: Vec<wgpu::TextureView>,
    accumulated: Option<Arc<GpuTexture>>,
    spare: Option<Arc<GpuTexture>>,
    shape_count: usize,
    valid: Arc<AtomicBool>,
    has_geometry: bool,
}
#[derive(Clone)]
struct DrawBindings {
    slab: Arc<UploadSlab>,
    offsets: [u32; 2],
}
struct EncodedDraw {
    mesh: Arc<GpuMesh>,
    bindings: DrawBindings,
    indices: std::ops::Range<u32>,
    instances: std::ops::Range<u32>,
}
struct UploadSlab {
    buffer: Arc<GpuBuffer>,
    group: wgpu::BindGroup,
    available: AtomicBool,
}
struct SlabLease(Arc<UploadSlab>);
impl Drop for SlabLease {
    fn drop(&mut self) {
        self.0.available.store(true, Ordering::Release);
    }
}
struct FrameUploads {
    lease: Option<SlabLease>,
    bytes: Vec<u8>,
    alignment: usize,
}
impl FrameUploads {
    fn append(
        &mut self,
        job: JobUniform,
        instances: &[StorageInstance],
        budget: &mut FrameBudget,
    ) -> Result<Option<DrawBindings>, ViewError> {
        if instances.is_empty() || std::mem::size_of_val(instances) > STORAGE_WINDOW {
            return Err(ViewError(
                "GPU instance batch exceeds storage window".into(),
            ));
        }
        let before = self.bytes.len();
        let uniform_offset = before.div_ceil(self.alignment) * self.alignment;
        let instance_offset = (uniform_offset + std::mem::size_of::<JobUniform>())
            .div_ceil(self.alignment)
            * self.alignment;
        let end = instance_offset + std::mem::size_of_val(instances);
        if end > UPLOAD_LIMIT || end - before > budget.upload {
            return Ok(None);
        }
        self.bytes.resize(uniform_offset, 0);
        self.bytes.extend_from_slice(bytemuck::bytes_of(&job));
        self.bytes.resize(instance_offset, 0);
        self.bytes
            .extend_from_slice(bytemuck::cast_slice(instances));
        budget.upload -= end - before;
        let slab = self
            .lease
            .as_ref()
            .ok_or_else(|| ViewError("GPU upload slab has no lease".into()))?
            .0
            .clone();
        Ok(Some(DrawBindings {
            slab,
            offsets: [uniform_offset as u32, instance_offset as u32],
        }))
    }
}
struct PendingDraw {
    tile_id: u64,
    layer: LayerKey,
    source: Arc<MeshData>,
    instances: Vec<prepare::Instance>,
    array: Option<ArrayInstances>,
    shape_count: usize,
    next_instance: u32,
    next_index: u32,
    mesh: Option<Arc<GpuMesh>>,
    bindings: Option<DrawBindings>,
}
struct Fence {
    generation: u64,
    published: bool,
    complete: bool,
    elapsed: f64,
    completed_at: Instant,
    invalid: bool,
    useful: bool,
}
struct FramePermit {
    counter: Arc<AtomicUsize>,
    valid_tiles: Vec<Arc<AtomicBool>>,
    completed: bool,
}
impl Drop for FramePermit {
    fn drop(&mut self) {
        if !self.completed {
            for tile in &self.valid_tiles {
                tile.store(false, Ordering::Release);
            }
        }
        self.counter.fetch_sub(1, Ordering::Relaxed);
    }
}
struct SessionState {
    worker: prepare::Worker,
    generation: u64,
    key: Option<FrameKey>,
    context: Option<egui::Context>,
    previous: Vec<VisibleTile>,
    current: Vec<VisibleTile>,
    active: Option<ActiveTile>,
    pending: VecDeque<Packet>,
    draw: Option<PendingDraw>,
    statistics: RenderStatus,
    started: Instant,
    fence_sender: mpsc::Sender<Fence>,
    fence_receiver: mpsc::Receiver<Fence>,
    presentation: Option<Arc<PresentationFrame>>,
}
impl SessionState {
    fn collect(&mut self) {
        while let Ok(fence) = self.fence_receiver.try_recv() {
            if fence.generation != self.generation {
                continue;
            }
            if self.statistics.error.is_some() {
                continue;
            }
            if fence.invalid {
                self.fail(ViewError(
                    "Rendering was interrupted. Move the view to retry.".into(),
                ));
                continue;
            }
            self.statistics.gpu_completion_latency_seconds += fence.elapsed;
            if fence.published && self.statistics.first_tile_seconds.is_none() {
                self.statistics.first_tile_seconds = Some(
                    fence
                        .completed_at
                        .duration_since(self.started)
                        .as_secs_f64(),
                );
            }
            if fence.useful && self.statistics.first_useful_seconds.is_none() {
                self.statistics.first_useful_seconds = Some(
                    fence
                        .completed_at
                        .duration_since(self.started)
                        .as_secs_f64(),
                );
            }
            if fence.complete {
                self.statistics.complete = true;
                self.statistics.completed_gpu_seconds = Some(
                    fence
                        .completed_at
                        .duration_since(self.started)
                        .as_secs_f64(),
                );
                self.statistics.reprojected_tiles = 0;
                self.previous.clear();
            }
        }
        self.statistics.tile_count = self
            .current
            .iter()
            .filter(|tile| tile.ready.load(Ordering::Acquire) && tile.valid.load(Ordering::Acquire))
            .count();
        self.statistics.has_image =
            self.current.iter().chain(&self.previous).any(|tile| {
                tile.ready.load(Ordering::Acquire) && tile.valid.load(Ordering::Acquire)
            });
        self.statistics.has_geometry = self.current.iter().chain(&self.previous).any(|tile| {
            tile.ready.load(Ordering::Acquire)
                && tile.valid.load(Ordering::Acquire)
                && tile.has_geometry
        });
        let mut sources = HashSet::new();
        let mut retained = self.pending.capacity() * std::mem::size_of::<Packet>();
        for packet in &self.pending {
            match packet {
                Packet::Draw {
                    mesh, instances, ..
                } => {
                    if sources.insert(Arc::as_ptr(mesh)) {
                        retained += mesh.bytes();
                    }
                    retained += instances.capacity() * std::mem::size_of::<prepare::Instance>();
                }
                Packet::Begin { layers, .. } => {
                    retained += layers.capacity() * std::mem::size_of::<(LayerKey, LayerStyle)>()
                }
                Packet::Finish { .. } => {}
            }
        }
        if let Some(draw) = &self.draw {
            if sources.insert(Arc::as_ptr(&draw.source)) {
                retained += draw.source.bytes();
            }
            retained += draw.instances.capacity() * std::mem::size_of::<prepare::Instance>();
        }
        self.statistics.cpu_bytes = self.worker.bytes()
            + retained
            + if self.key.is_some() && !self.statistics.complete {
                UPLOAD_CPU_RESERVE
            } else {
                0
            };
        self.statistics.preparation_seconds = self.worker.preparation_seconds();
        self.statistics.request_elapsed_seconds = self.started.elapsed().as_secs_f64();
    }
    fn begin(
        &mut self,
        scene: Arc<Scene>,
        key: FrameKey,
        context: egui::Context,
        preview: bool,
    ) -> Result<(), ViewError> {
        self.collect();
        if self.statistics.complete
            && self.key.as_ref() == Some(&key)
            && !preview
            && self.current.iter().all(|tile| !tile.info.preview)
        {
            return Ok(());
        }
        let generation = self
            .worker
            .start(scene, key.clone(), context.clone(), preview)?;
        let compatible = self.key.as_ref().is_some_and(|previous| {
            previous.selected == key.selected
                && previous.depth == key.depth
                && previous.styles == key.styles
                && previous.text_pass == key.text_pass
                && previous.pixels_per_point == key.pixels_per_point
        });
        if !compatible {
            self.previous.clear();
            self.current.clear();
            self.presentation = None;
        } else if self.statistics.complete || self.previous.is_empty() {
            self.previous = std::mem::take(&mut self.current);
            self.previous.retain(|tile| {
                tile.ready.load(Ordering::Acquire) && tile.valid.load(Ordering::Acquire)
            });
        } else {
            self.current.clear();
        }
        self.generation = generation;
        self.key = Some(key);
        self.context = Some(context);
        self.pending.clear();
        self.draw = None;
        self.active = None;
        self.statistics = RenderStatus::default();
        self.started = Instant::now();
        Ok(())
    }
    fn fail(&mut self, error: ViewError) {
        self.worker.cancel();
        self.pending.clear();
        self.draw = None;
        self.active = None;
        self.statistics.complete = false;
        self.statistics.error = Some(error.to_string());
    }
}

#[derive(Clone)]
pub struct GpuRendererHandle {
    engine: Arc<Mutex<Engine>>,
    device: wgpu::Device,
    #[cfg(not(target_arch = "wasm32"))]
    queue: wgpu::Queue,
    adapter: wgpu::AdapterInfo,
    format: wgpu::TextureFormat,
    ledger: Arc<AtomicUsize>,
    in_flight: Arc<AtomicUsize>,
    device_error: Arc<Mutex<Option<String>>>,
}
impl GpuRendererHandle {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_render_state(state: &RenderState) -> Result<Self, ViewError> {
        pollster::block_on(Self::from_render_state_async(state))
    }
    pub async fn from_render_state_async(state: &RenderState) -> Result<Self, ViewError> {
        let handle = Self::create(
            state.device.clone(),
            state.queue.clone(),
            state.adapter.get_info(),
            state.target_format,
        )
        .await?;
        state
            .renderer
            .write()
            .callback_resources
            .insert(handle.clone());
        Ok(handle)
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn headless() -> Result<Self, ViewError> {
        if wgpu::Instance::enabled_backend_features().is_empty() {
            return Err(ViewError("No native WGPU backend is enabled".into()));
        }
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .map_err(|error| ViewError(format!("Selecting GPU adapter: {error}")))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("rlayout-headless"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                ..Default::default()
            })
            .await
            .map_err(|error| ViewError(format!("Creating GPU device: {error}")))?;
        Self::create(
            device,
            queue,
            adapter.get_info(),
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .await
    }
    async fn create(
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter: wgpu::AdapterInfo,
        format: wgpu::TextureFormat,
    ) -> Result<Self, ViewError> {
        let ledger = Arc::new(AtomicUsize::new(0));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let device_error = Arc::new(Mutex::new(None));
        device.on_uncaptured_error({
            let errors = device_error.clone();
            Arc::new(move |error: wgpu::Error| {
                if let Ok(mut destination) = errors.lock() {
                    *destination = Some(format!("GPU rendering failed: {error}"));
                } else {
                    eprintln!("GPU rendering failed: {error}");
                }
            })
        });
        device.set_device_lost_callback({
            let errors = device_error.clone();
            move |reason, message| {
                if let Ok(mut destination) = errors.lock() {
                    *destination = Some(format!(
                        "GPU device lost ({reason:?}): {message}. Restart RLayout."
                    ));
                } else {
                    eprintln!("GPU device lost: {message}");
                }
            }
        });
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let engine = Engine::new(
            device.clone(),
            queue.clone(),
            format,
            ledger.clone(),
            in_flight.clone(),
        );
        if let Some(error) = validation.pop().await {
            return Err(ViewError(format!("Creating GPU pipelines: {error}")));
        }
        let engine = engine?;
        Ok(Self {
            engine: Arc::new(Mutex::new(engine)),
            device,
            #[cfg(not(target_arch = "wasm32"))]
            queue,
            adapter,
            format,
            ledger,
            in_flight,
            device_error,
        })
    }
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.clone()
    }
    pub fn target_format(&self) -> wgpu::TextureFormat {
        self.format
    }
    pub(super) fn create_session(&self, scene: Arc<Scene>) -> Result<RenderSession, ViewError> {
        let (fence_sender, fence_receiver) = mpsc::channel();
        Ok(RenderSession {
            handle: self.clone(),
            scene,
            state: Arc::new(Mutex::new(SessionState {
                worker: prepare::Worker::new()?,
                generation: 0,
                key: None,
                context: None,
                previous: Vec::new(),
                current: Vec::new(),
                active: None,
                pending: VecDeque::new(),
                draw: None,
                statistics: RenderStatus::default(),
                started: Instant::now(),
                fence_sender,
                fence_receiver,
                presentation: None,
            })),
        })
    }
    #[cfg(test)]
    pub(super) fn test_device_queue(&self) -> (wgpu::Device, wgpu::Queue, wgpu::TextureFormat) {
        (self.device.clone(), self.queue.clone(), self.format)
    }
    fn poll(&self) -> Result<(), ViewError> {
        self.device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| ViewError(format!("Polling GPU: {error}")))?;
        if let Some(error) = self
            .device_error
            .lock()
            .map_err(|error| ViewError(error.to_string()))?
            .clone()
        {
            return Err(ViewError(error));
        }
        Ok(())
    }
}

pub(super) struct RenderSession {
    handle: GpuRendererHandle,
    scene: Arc<Scene>,
    state: Arc<Mutex<SessionState>>,
}
impl RenderSession {
    pub(super) fn request(
        &mut self,
        key: FrameKey,
        context: egui::Context,
        preview: bool,
    ) -> Result<(), ViewError> {
        self.handle.poll()?;
        self.state
            .lock()
            .map_err(|error| ViewError(error.to_string()))?
            .begin(self.scene.clone(), key, context, preview)
    }
    pub(super) fn callback(&self, viewport: Rect, camera: Camera) -> egui::Shape {
        Callback::new_paint_callback(
            viewport,
            CanvasCallback {
                handle: self.handle.clone(),
                scene: self.scene.clone(),
                state: self.state.clone(),
                viewport,
                camera,
                frame: Mutex::new(None),
            },
        )
        .into()
    }
    pub(super) fn status(&self) -> RenderStatus {
        let polling = self.handle.poll();
        let mut status = match self.state.lock() {
            Ok(mut state) => {
                if let Err(error) = polling {
                    state.fail(error);
                }
                state.collect();
                state.statistics.clone()
            }
            Err(error) => RenderStatus {
                error: Some(error.to_string()),
                ..Default::default()
            },
        };
        status.gpu_bytes = self.handle.ledger.load(Ordering::Relaxed);
        status.in_flight_frames = self.handle.in_flight.load(Ordering::Relaxed);
        status
    }
    #[cfg(test)]
    pub(super) fn test_saturate_queue(&self) -> TestQueueSaturation {
        self.handle
            .in_flight
            .fetch_add(IN_FLIGHT_LIMIT, Ordering::Relaxed);
        TestQueueSaturation(self.handle.in_flight.clone())
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) async fn capture(
        &self,
        key: FrameKey,
        context: egui::Context,
    ) -> Result<RenderedImage, ViewError> {
        let result = Arc::new(Mutex::new(CaptureState {
            result: None,
            waker: None,
        }));
        std::thread::Builder::new()
            .name("gds-gpu-capture".into())
            .spawn({
                let handle = self.handle.clone();
                let scene = self.scene.clone();
                let state = self.state.clone();
                let result = result.clone();
                move || {
                    let image = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        capture(&handle, scene, state, key, context)
                    }))
                    .unwrap_or_else(|_| {
                        Err(ViewError("GPU capture worker stopped unexpectedly".into()))
                    });
                    match result.lock() {
                        Ok(mut result) => {
                            result.result = Some(image);
                            if let Some(waker) = result.waker.take() {
                                waker.wake();
                            }
                        }
                        Err(error) => eprintln!("Delivering GPU capture result: {error}"),
                    }
                }
            })
            .map_err(|error| ViewError(format!("Starting GPU capture: {error}")))?;
        CaptureFuture(result).await
    }
    #[cfg(test)]
    pub(super) fn test_discard_request(
        &self,
        key: FrameKey,
        context: egui::Context,
    ) -> Result<(), ViewError> {
        let generation = {
            let mut state = self
                .state
                .lock()
                .map_err(|error| ViewError(error.to_string()))?;
            state.begin(self.scene.clone(), key, context, false)?;
            state.generation
        };
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            self.handle.poll()?;
            let (commands, published) = {
                let mut engine = self
                    .handle
                    .engine
                    .lock()
                    .map_err(|error| ViewError(error.to_string()))?;
                let mut state = self
                    .state
                    .lock()
                    .map_err(|error| ViewError(error.to_string()))?;
                if state.generation != generation {
                    return Err(ViewError(
                        "Discarded GPU test request was superseded".into(),
                    ));
                }
                let commands = engine.encode(&self.scene, &mut state, None)?;
                (commands, !state.current.is_empty())
            };
            drop(commands);
            if published {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(ViewError("GPU discard test did not publish a tile".into()));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}
#[cfg(test)]
pub(super) struct TestQueueSaturation(Arc<AtomicUsize>);
#[cfg(test)]
impl Drop for TestQueueSaturation {
    fn drop(&mut self) {
        self.0.fetch_sub(IN_FLIGHT_LIMIT, Ordering::Relaxed);
    }
}
#[cfg(not(target_arch = "wasm32"))]
struct CaptureState {
    result: Option<Result<RenderedImage, ViewError>>,
    waker: Option<Waker>,
}
#[cfg(not(target_arch = "wasm32"))]
struct CaptureFuture(Arc<Mutex<CaptureState>>);
#[cfg(not(target_arch = "wasm32"))]
impl Future for CaptureFuture {
    type Output = Result<RenderedImage, ViewError>;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        match self.0.lock() {
            Ok(mut state) => {
                if let Some(result) = state.result.take() {
                    Poll::Ready(result)
                } else {
                    state.waker = Some(context.waker().clone());
                    Poll::Pending
                }
            }
            Err(error) => Poll::Ready(Err(ViewError(error.to_string()))),
        }
    }
}

enum CachedResource {
    Mesh(u64),
    Glyph(u64),
    Tile(TileKey),
}
struct Engine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    format: wgpu::TextureFormat,
    ledger: Arc<AtomicUsize>,
    in_flight: Arc<AtomicUsize>,
    mask_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    present_pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    white: Arc<GpuTexture>,
    meshes: HashMap<u64, Arc<GpuMesh>>,
    glyphs: HashMap<u64, Arc<GpuTexture>>,
    tiles: HashMap<TileKey, VisibleTile>,
    cache_order: VecDeque<CachedResource>,
    upload_slabs: Vec<Arc<UploadSlab>>,
    _staging_reserve: Allocation,
}

#[derive(Default)]
struct KeepAlive {
    textures: Vec<Arc<GpuTexture>>,
    buffers: Vec<Arc<GpuBuffer>>,
    meshes: Vec<Arc<GpuMesh>>,
    slabs: Vec<SlabLease>,
    valid_tiles: Vec<Arc<AtomicBool>>,
    ready_tiles: Vec<(Arc<AtomicBool>, Arc<AtomicBool>)>,
}
impl KeepAlive {
    fn touch(&mut self, tile: &Arc<AtomicBool>) {
        if self
            .valid_tiles
            .last()
            .is_none_or(|previous| !Arc::ptr_eq(previous, tile))
        {
            self.valid_tiles.push(tile.clone());
        }
    }
}
struct PresentedTile {
    group: wgpu::BindGroup,
    _uniform: Arc<GpuBuffer>,
    _image: Arc<GpuTexture>,
}
struct PresentationFrame {
    tiles: Vec<PresentedTile>,
    camera: Camera,
    reprojected: usize,
    generation: u64,
}
struct FrameBudget {
    upload: usize,
    triangles: usize,
    draws: usize,
}
impl FrameBudget {
    fn new(presentation_draws: usize) -> Result<Self, ViewError> {
        if presentation_draws >= DRAW_LIMIT {
            return Err(ViewError(
                "Viewport exceeds the GPU presentation budget".into(),
            ));
        }
        let presentation_bytes = presentation_draws * std::mem::size_of::<PresentationUniform>();
        Ok(Self {
            upload: UPLOAD_LIMIT - presentation_bytes,
            triangles: TRIANGLE_LIMIT - presentation_draws * 2,
            draws: DRAW_LIMIT - presentation_draws,
        })
    }
}

fn pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &'static str,
    format: wgpu::TextureFormat,
    mask: bool,
) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let attributes = wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x2, 3 => Float32x4, 4 => Uint32, 5 => Uint32];
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<GpuVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &attributes,
    };
    let buffers = if mask {
        vec![Some(vertex_layout)]
    } else {
        Vec::new()
    };
    let maximum = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Max,
    };
    let layout = mask.then(|| {
        let job = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rlayout-job-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<JobUniform>() as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<StorageInstance>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });
        let glyph = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rlayout-glyph-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rlayout-mask-layout"),
            bind_group_layouts: &[Some(&job), Some(&glyph)],
            immediate_size: 0,
        })
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: layout.as_ref(),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            buffers: &buffers,
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState {
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: mask.then_some(wgpu::BlendState {
                    color: maximum,
                    alpha: maximum,
                }),
                write_mask: if mask {
                    wgpu::ColorWrites::RED
                } else {
                    wgpu::ColorWrites::ALL
                },
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}
fn texture_raw(
    device: &wgpu::Device,
    ledger: &Arc<AtomicUsize>,
    label: &str,
    size: [u32; 2],
    layers: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> Result<Arc<GpuTexture>, ViewError> {
    if size.contains(&0)
        || size
            .iter()
            .any(|dimension| *dimension > device.limits().max_texture_dimension_2d)
        || layers == 0
        || layers > device.limits().max_texture_array_layers
    {
        return Err(ViewError(
            "GPU texture dimensions exceed adapter limits".into(),
        ));
    }
    let bytes = usize::try_from(size[0])
        .ok()
        .and_then(|width| width.checked_mul(size[1] as usize))
        .and_then(|pixels| pixels.checked_mul(layers as usize))
        .and_then(|pixels| {
            pixels.checked_mul(if format == wgpu::TextureFormat::R8Unorm {
                1
            } else {
                4
            })
        })
        .ok_or_else(|| ViewError("GPU texture size overflow".into()))?;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(if layers > 1 {
            wgpu::TextureViewDimension::D2Array
        } else {
            wgpu::TextureViewDimension::D2
        }),
        ..Default::default()
    });
    ledger.fetch_add(bytes, Ordering::Relaxed);
    Ok(Arc::new(GpuTexture {
        texture,
        view,
        _allocation: Allocation {
            bytes,
            ledger: ledger.clone(),
        },
    }))
}
fn clear(encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("rlayout-clear"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        ..Default::default()
    });
}
impl Engine {
    fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        format: wgpu::TextureFormat,
        ledger: Arc<AtomicUsize>,
        in_flight: Arc<AtomicUsize>,
    ) -> Result<Self, ViewError> {
        ledger.fetch_add(STAGING_RESERVE, Ordering::Relaxed);
        let staging_reserve = Allocation {
            bytes: STAGING_RESERVE,
            ledger: ledger.clone(),
        };
        let mask_pipeline = pipeline(
            &device,
            "rlayout-mask",
            include_str!("gpu_mask.wgsl"),
            wgpu::TextureFormat::R8Unorm,
            true,
        );
        let composite_pipeline = pipeline(
            &device,
            "rlayout-compose",
            include_str!("gpu_composite.wgsl"),
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        );
        let present_pipeline = pipeline(
            &device,
            "rlayout-present",
            include_str!("gpu_present.wgsl"),
            format,
            false,
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("rlayout-nearest"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let white = texture_raw(
            &device,
            &ledger,
            "rlayout-white",
            [1, 1],
            1,
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        )?;
        queue.write_texture(
            white.texture.as_image_copy(),
            &[255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(1),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        Ok(Self {
            device,
            queue,
            format,
            ledger,
            in_flight,
            mask_pipeline,
            composite_pipeline,
            present_pipeline,
            sampler,
            white,
            meshes: HashMap::new(),
            glyphs: HashMap::new(),
            tiles: HashMap::new(),
            cache_order: VecDeque::new(),
            upload_slabs: Vec::new(),
            _staging_reserve: staging_reserve,
        })
    }
    fn uploads(&mut self) -> Result<Option<FrameUploads>, ViewError> {
        for slab in &self.upload_slabs {
            if slab
                .available
                .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(Some(FrameUploads {
                    lease: Some(SlabLease(slab.clone())),
                    bytes: Vec::new(),
                    alignment: self
                        .device
                        .limits()
                        .min_uniform_buffer_offset_alignment
                        .max(self.device.limits().min_storage_buffer_offset_alignment)
                        as usize,
                }));
            }
        }
        if self.upload_slabs.len() >= IN_FLIGHT_LIMIT {
            return Ok(None);
        }
        let bytes = UPLOAD_LIMIT + STORAGE_WINDOW;
        self.room(bytes)?;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlayout-upload-slab"),
            size: bytes as u64,
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.ledger.fetch_add(bytes, Ordering::Relaxed);
        let buffer = Arc::new(GpuBuffer {
            buffer,
            _allocation: Allocation {
                bytes,
                ledger: self.ledger.clone(),
            },
        });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rlayout-upload-slab"),
            layout: &self.mask_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffer.buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<JobUniform>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffer.buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(STORAGE_WINDOW as u64),
                    }),
                },
            ],
        });
        let slab = Arc::new(UploadSlab {
            buffer,
            group,
            available: AtomicBool::new(false),
        });
        self.upload_slabs.push(slab.clone());
        Ok(Some(FrameUploads {
            lease: Some(SlabLease(slab)),
            bytes: Vec::new(),
            alignment: self
                .device
                .limits()
                .min_uniform_buffer_offset_alignment
                .max(self.device.limits().min_storage_buffer_offset_alignment)
                as usize,
        }))
    }
    fn room(&mut self, bytes: usize) -> Result<(), ViewError> {
        if bytes > GPU_LIMIT {
            return Err(ViewError(
                "Rendering allocation exceeds GPU memory budget".into(),
            ));
        }
        while self
            .ledger
            .load(Ordering::Relaxed)
            .checked_add(bytes)
            .is_none_or(|total| total > GPU_LIMIT)
        {
            match self.cache_order.pop_front() {
                Some(CachedResource::Mesh(identifier)) => {
                    if self
                        .meshes
                        .get(&identifier)
                        .is_some_and(|mesh| mesh.used.swap(false, Ordering::Relaxed))
                    {
                        self.cache_order.push_back(CachedResource::Mesh(identifier));
                        continue;
                    }
                    self.meshes.remove(&identifier);
                }
                Some(CachedResource::Glyph(identifier)) => {
                    self.glyphs.remove(&identifier);
                }
                Some(CachedResource::Tile(key)) => {
                    self.tiles.remove(&key);
                }
                None => {
                    return Err(ViewError(
                        "In-flight rendering exceeds GPU memory budget".into(),
                    ));
                }
            }
        }
        Ok(())
    }
    fn buffer(
        &mut self,
        label: &str,
        data: &[u8],
        usage: wgpu::BufferUsages,
    ) -> Result<Arc<GpuBuffer>, ViewError> {
        if data.is_empty() {
            return Err(ViewError("Empty GPU buffer".into()));
        }
        let bytes = data
            .len()
            .checked_add(3)
            .map(|bytes| bytes / 4 * 4)
            .ok_or_else(|| ViewError("GPU buffer size overflow".into()))?;
        self.room(bytes)?;
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage,
            });
        self.ledger.fetch_add(bytes, Ordering::Relaxed);
        Ok(Arc::new(GpuBuffer {
            buffer,
            _allocation: Allocation {
                bytes,
                ledger: self.ledger.clone(),
            },
        }))
    }
    fn texture(
        &mut self,
        label: &str,
        size: [u32; 2],
        layers: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> Result<Arc<GpuTexture>, ViewError> {
        let bytes = (size[0] as usize)
            .checked_mul(size[1] as usize)
            .and_then(|pixels| pixels.checked_mul(layers as usize))
            .and_then(|pixels| {
                pixels.checked_mul(if format == wgpu::TextureFormat::R8Unorm {
                    1
                } else {
                    4
                })
            })
            .ok_or_else(|| ViewError("GPU texture size overflow".into()))?;
        self.room(bytes)?;
        texture_raw(
            &self.device,
            &self.ledger,
            label,
            size,
            layers,
            format,
            usage,
        )
    }
    fn mesh(
        &mut self,
        source: &Arc<MeshData>,
        budget: &mut FrameBudget,
    ) -> Result<Option<Arc<GpuMesh>>, ViewError> {
        if let Some(mesh) = self.meshes.get(&source.id) {
            mesh.used.store(true, Ordering::Relaxed);
            return Ok(Some(mesh.clone()));
        }
        let glyph_bytes = source
            .glyph
            .as_ref()
            .filter(|glyph| !self.glyphs.contains_key(&glyph.id))
            .map_or(0, |glyph| glyph.alpha.len());
        let bytes = source.vertices.len() * std::mem::size_of::<GpuVertex>()
            + source.indices.len() * 4
            + glyph_bytes;
        if bytes > UPLOAD_LIMIT - STORAGE_WINDOW {
            return Err(ViewError("Geometry batch exceeds GPU upload budget".into()));
        }
        if bytes > budget.upload {
            return Ok(None);
        }
        let vertices = self.buffer(
            "rlayout-vertices",
            bytemuck::cast_slice(&source.vertices),
            wgpu::BufferUsages::VERTEX,
        )?;
        let indices = self.buffer(
            "rlayout-indices",
            bytemuck::cast_slice(&source.indices),
            wgpu::BufferUsages::INDEX,
        )?;
        let glyph = if let Some(glyph) = &source.glyph {
            if let Some(existing) = self.glyphs.get(&glyph.id) {
                existing.clone()
            } else {
                if glyph.alpha.len() != glyph.size[0] as usize * glyph.size[1] as usize {
                    return Err(ViewError("Invalid glyph bitmap size".into()));
                }
                let texture = self.texture(
                    "rlayout-glyph",
                    glyph.size,
                    1,
                    wgpu::TextureFormat::R8Unorm,
                    wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                )?;
                self.queue.write_texture(
                    texture.texture.as_image_copy(),
                    &glyph.alpha,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(glyph.size[0]),
                        rows_per_image: Some(glyph.size[1]),
                    },
                    wgpu::Extent3d {
                        width: glyph.size[0],
                        height: glyph.size[1],
                        depth_or_array_layers: 1,
                    },
                );
                self.glyphs.insert(glyph.id, texture.clone());
                self.cache_order.push_back(CachedResource::Glyph(glyph.id));
                texture
            }
        } else {
            self.white.clone()
        };
        let index_count = u32::try_from(source.indices.len())
            .map_err(|_| ViewError("GPU geometry index count overflow".into()))?;
        let glyph_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rlayout-glyph"),
            layout: &self.mask_pipeline.get_bind_group_layout(1),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&glyph.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let mesh = Arc::new(GpuMesh {
            vertices,
            indices,
            _glyph: glyph,
            glyph_group,
            index_count,
            used: AtomicBool::new(true),
        });
        self.meshes.insert(source.id, mesh.clone());
        self.cache_order.push_back(CachedResource::Mesh(source.id));
        budget.upload -= bytes;
        Ok(Some(mesh))
    }
    #[allow(clippy::too_many_arguments)]
    fn begin_tile(
        &mut self,
        scene: &Scene,
        key: &FrameKey,
        tile: TileInfo,
        layers: Vec<(LayerKey, LayerStyle)>,
        first_group: bool,
        state: &mut SessionState,
        encoder: &mut wgpu::CommandEncoder,
        keep: &mut KeepAlive,
    ) -> Result<(), ViewError> {
        if layers.len() > 64 {
            return Err(ViewError("Rendering layer group exceeds 64 masks".into()));
        }
        if first_group {
            let cache_key = TileKey::new(scene, key, tile);
            let cached = self
                .tiles
                .get(&cache_key)
                .filter(|cached| {
                    cached.ready.load(Ordering::Acquire)
                        && cached.valid.load(Ordering::Acquire)
                        && cached.info.crop.contains_rect(tile.crop)
                })
                .cloned();
            let mut active = ActiveTile {
                info: tile,
                cache_key,
                cached,
                layers: Vec::new(),
                masks: None,
                mask_views: Vec::new(),
                accumulated: None,
                spare: None,
                shape_count: 0,
                valid: Arc::new(AtomicBool::new(true)),
                has_geometry: false,
            };
            if active.cached.is_none() {
                keep.touch(&active.valid);
                let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC;
                let accumulated = self.texture(
                    "rlayout-tile",
                    tile.size,
                    1,
                    wgpu::TextureFormat::Rgba8Unorm,
                    usage,
                )?;
                let spare = self.texture(
                    "rlayout-tile-spare",
                    tile.size,
                    1,
                    wgpu::TextureFormat::Rgba8Unorm,
                    usage,
                )?;
                clear(encoder, &accumulated.view);
                clear(encoder, &spare.view);
                keep.textures.extend([accumulated.clone(), spare.clone()]);
                active.accumulated = Some(accumulated);
                active.spare = Some(spare);
                let masks = self.texture(
                    "rlayout-layer-masks",
                    tile.size,
                    64,
                    wgpu::TextureFormat::R8Unorm,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                )?;
                active.mask_views = (0..64)
                    .map(|layer| {
                        masks.texture.create_view(&wgpu::TextureViewDescriptor {
                            dimension: Some(wgpu::TextureViewDimension::D2),
                            base_array_layer: layer,
                            array_layer_count: Some(1),
                            ..Default::default()
                        })
                    })
                    .collect();
                active.masks = Some(masks);
            }
            state.active = Some(active);
        }
        let active = state
            .active
            .as_mut()
            .ok_or_else(|| ViewError("Missing active GPU tile".into()))?;
        if active.info.id != tile.id {
            return Err(ViewError("GPU tile groups are out of order".into()));
        }
        active.layers = layers;
        if let Some(masks) = &active.masks {
            keep.touch(&active.valid);
            for view in active.mask_views.iter().take(active.layers.len()) {
                clear(encoder, view);
            }
            keep.textures.push(masks.clone());
        }
        Ok(())
    }
    fn draw(
        &mut self,
        draw: &mut PendingDraw,
        active: &mut ActiveTile,
        budget: &mut FrameBudget,
        uploads: &mut Option<FrameUploads>,
        batches: &mut BTreeMap<LayerKey, Vec<EncodedDraw>>,
    ) -> Result<bool, ViewError> {
        if active.info.id != draw.tile_id {
            return Err(ViewError("Geometry batch belongs to another tile".into()));
        }
        if active.cached.is_some() {
            return Ok(true);
        }
        if draw.source.indices.is_empty() || draw.instances.is_empty() {
            return Ok(true);
        }
        if draw.mesh.is_none() {
            let Some(mesh) = self.mesh(&draw.source, budget)? else {
                return Ok(false);
            };
            draw.mesh = Some(mesh);
        }
        let mesh = draw
            .mesh
            .as_ref()
            .ok_or_else(|| ViewError("Missing GPU geometry".into()))?;
        if draw.bindings.is_none() {
            if uploads.is_none() {
                *uploads = self.uploads()?;
            }
            let Some(uploads) = uploads.as_mut() else {
                return Ok(false);
            };
            let instances: Vec<_> = draw
                .instances
                .iter()
                .map(|instance| StorageInstance {
                    affine: instance.affine,
                    translation: instance.translation,
                    padding: [0.0; 2],
                })
                .collect();
            let array = draw.array.unwrap_or(ArrayInstances {
                column_step: [0.0; 2],
                row_step: [0.0; 2],
                columns: 1,
                rows: 1,
            });
            let job = JobUniform {
                size: [active.info.size[0] as f32, active.info.size[1] as f32],
                columns: array.columns,
                rows: if draw.array.is_some() { array.rows } else { 0 },
                column_step: array.column_step,
                row_step: array.row_step,
            };
            let Some(bindings) = uploads.append(job, &instances, budget)? else {
                return Ok(false);
            };
            draw.bindings = Some(bindings);
        }
        let total = if let Some(array) = draw.array {
            if array.columns == 0 || array.rows == 0 {
                return Err(ViewError("GPU array has no rows or columns".into()));
            }
            array
                .columns
                .checked_mul(array.rows)
                .and_then(|count| {
                    u32::try_from(draw.instances.len())
                        .ok()
                        .and_then(|bases| count.checked_mul(bases))
                })
                .ok_or_else(|| ViewError("GPU array instance count overflow".into()))?
        } else {
            u32::try_from(draw.instances.len())
                .map_err(|_| ViewError("GPU instance count overflow".into()))?
        };
        let triangles = (mesh.index_count as usize / 3).max(1);
        let sliced = triangles > TRIANGLE_LIMIT;
        let (indices, triangles) = if sliced {
            let remaining = (mesh.index_count - draw.next_index) as usize / 3;
            let triangles = remaining.min(budget.triangles);
            let end = draw.next_index + (triangles * 3) as u32;
            (draw.next_index..end, triangles)
        } else {
            (0..mesh.index_count, triangles)
        };
        let count = if sliced {
            usize::from(triangles > 0)
        } else {
            (budget.triangles / triangles).min((total - draw.next_instance) as usize)
        };
        if count == 0 || budget.draws == 0 {
            return Ok(false);
        }
        let count = u32::try_from(count)
            .map_err(|_| ViewError("GPU draw instance count overflow".into()))?;
        let bindings = draw
            .bindings
            .as_ref()
            .ok_or_else(|| ViewError("Missing GPU draw bindings".into()))?;
        if !active.has_geometry {
            active.has_geometry = active
                .layers
                .iter()
                .any(|(layer, style)| *layer == draw.layer && style.color.a() > 0);
        }
        batches.entry(draw.layer).or_default().push(EncodedDraw {
            mesh: mesh.clone(),
            bindings: bindings.clone(),
            indices: indices.clone(),
            instances: draw.next_instance..draw.next_instance + count,
        });
        if sliced && indices.end < mesh.index_count {
            draw.next_index = indices.end;
        } else {
            draw.next_index = 0;
            draw.next_instance += count;
        }
        budget.triangles -= triangles * count as usize;
        budget.draws -= 1;
        Ok(draw.next_instance == total)
    }
    fn flush_draws(
        &self,
        active: Option<&ActiveTile>,
        batches: &mut BTreeMap<LayerKey, Vec<EncodedDraw>>,
        encoder: &mut wgpu::CommandEncoder,
        keep: &mut KeepAlive,
    ) -> Result<usize, ViewError> {
        if batches.is_empty() {
            return Ok(0);
        }
        let active =
            active.ok_or_else(|| ViewError("Missing tile for grouped GPU draws".into()))?;
        keep.touch(&active.valid);
        if let Some(masks) = &active.masks {
            keep.textures.push(masks.clone());
        }
        let batches = std::mem::take(batches);
        let count = batches.len();
        for (layer, commands) in batches {
            let layer = active
                .layers
                .iter()
                .position(|(key, _)| *key == layer)
                .ok_or_else(|| ViewError("Geometry layer is absent from its mask group".into()))?;
            let view = active
                .mask_views
                .get(layer)
                .ok_or_else(|| ViewError("Missing grouped layer mask view".into()))?;
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("rlayout-layer-geometry"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                pass.set_pipeline(&self.mask_pipeline);
                let mut previous: Option<&Arc<GpuMesh>> = None;
                for command in &commands {
                    pass.set_bind_group(0, &command.bindings.slab.group, &command.bindings.offsets);
                    if previous.is_none_or(|mesh| !Arc::ptr_eq(mesh, &command.mesh)) {
                        pass.set_bind_group(1, &command.mesh.glyph_group, &[]);
                        pass.set_vertex_buffer(0, command.mesh.vertices.buffer.slice(..));
                        pass.set_index_buffer(
                            command.mesh.indices.buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        previous = Some(&command.mesh);
                    }
                    pass.draw_indexed(command.indices.clone(), 0, command.instances.clone());
                }
            }
            keep.meshes
                .extend(commands.into_iter().map(|command| command.mesh));
        }
        Ok(count)
    }
    fn finish_tile(
        &mut self,
        state: &mut SessionState,
        tile_id: u64,
        last_group: bool,
        encoder: &mut wgpu::CommandEncoder,
        keep: &mut KeepAlive,
        budget: &mut FrameBudget,
    ) -> Result<bool, ViewError> {
        let active = state
            .active
            .as_mut()
            .ok_or_else(|| ViewError("Finishing a missing GPU tile".into()))?;
        if active.info.id != tile_id {
            return Err(ViewError("Finishing another GPU tile".into()));
        }
        if active.cached.is_none() {
            if budget.draws == 0
                || budget.triangles == 0
                || budget.upload < std::mem::size_of::<Palette>()
            {
                return Err(ViewError("GPU compositor exceeds the frame budget".into()));
            }
            budget.draws -= 1;
            budget.triangles -= 1;
            budget.upload -= std::mem::size_of::<Palette>();
            keep.touch(&active.valid);
            let mut palette = Palette {
                colors: [[0; 4]; 64],
                count: [active.layers.len() as u32, 0, 0, 0],
            };
            for (destination, (_, style)) in palette.colors.iter_mut().zip(&active.layers) {
                *destination = style.color.to_array().map(u32::from);
            }
            let uniforms = self.buffer(
                "rlayout-palette",
                bytemuck::bytes_of(&palette),
                wgpu::BufferUsages::UNIFORM,
            )?;
            let masks = active
                .masks
                .as_ref()
                .ok_or_else(|| ViewError("Missing GPU layer masks".into()))?;
            let previous = active
                .accumulated
                .as_ref()
                .ok_or_else(|| ViewError("Missing GPU tile accumulator".into()))?;
            let output = active
                .spare
                .as_ref()
                .ok_or_else(|| ViewError("Missing GPU tile output".into()))?;
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rlayout-compose"),
                layout: &self.composite_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&masks.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: uniforms.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&previous.view),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("rlayout-compose"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &output.view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                pass.set_pipeline(&self.composite_pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.draw(0..3, 0..1);
            }
            keep.textures
                .extend([masks.clone(), previous.clone(), output.clone()]);
            keep.buffers.push(uniforms);
            std::mem::swap(&mut active.accumulated, &mut active.spare);
        }
        if !last_group {
            return Ok(false);
        }
        let active = state
            .active
            .take()
            .ok_or_else(|| ViewError("Missing completed GPU tile".into()))?;
        let key = state
            .key
            .clone()
            .ok_or_else(|| ViewError("Missing completed view key".into()))?;
        let tile = if let Some(cached) = active.cached {
            VisibleTile {
                info: TileInfo {
                    crop: cached.info.crop,
                    ..active.info
                },
                key,
                image: cached.image,
                shape_count: cached.shape_count,
                ready: cached.ready,
                valid: cached.valid,
                has_geometry: cached.has_geometry,
            }
        } else {
            VisibleTile {
                info: active.info,
                key,
                image: active
                    .accumulated
                    .ok_or_else(|| ViewError("Missing completed GPU image".into()))?,
                shape_count: active.shape_count,
                ready: Arc::new(AtomicBool::new(false)),
                valid: active.valid,
                has_geometry: active.has_geometry,
            }
        };
        state.statistics.shape_count = state
            .statistics
            .shape_count
            .saturating_add(tile.shape_count);
        keep.ready_tiles
            .push((tile.ready.clone(), tile.valid.clone()));
        if !self.tiles.contains_key(&active.cache_key) {
            self.cache_order
                .push_back(CachedResource::Tile(active.cache_key.clone()));
        }
        self.tiles.insert(active.cache_key, tile.clone());
        state.current.push(tile);
        Ok(true)
    }

    fn presentation(
        &mut self,
        state: &SessionState,
        viewport: Rect,
        camera: Camera,
        screen: &ScreenDescriptor,
        encoded: &[(Arc<AtomicBool>, Arc<AtomicBool>)],
        budget: &mut FrameBudget,
    ) -> Result<Arc<PresentationFrame>, ViewError> {
        let pixel_viewport = egui::PaintCallbackInfo {
            viewport,
            clip_rect: viewport,
            pixels_per_point: screen.pixels_per_point,
            screen_size_px: screen.size_in_pixels,
        }
        .viewport_in_pixels();
        let size = [
            pixel_viewport.width_px as f32,
            pixel_viewport.height_px as f32,
        ];
        let offset = Vec2::new(pixel_viewport.left_px as f32, pixel_viewport.top_px as f32);
        let mut tiles = Vec::new();
        let mut reprojected = 0;
        for tile in state.previous.iter().chain(&state.current) {
            if !tile.valid.load(Ordering::Acquire)
                || (!tile.ready.load(Ordering::Acquire)
                    && !encoded
                        .iter()
                        .any(|(ready, _)| Arc::ptr_eq(ready, &tile.ready)))
            {
                continue;
            }
            // Reconstructing a world corner loses subpixels when its absolute
            // coordinate is large; preserve the retained screen displacement.
            let ratio = camera.pixels_per_unit / tile.key.camera.pixels_per_unit;
            let displacement = tile.key.camera.center.minus(camera.center);
            let old_center = tile.key.viewport.center();
            let new_center = viewport.center();
            let first = Vec2::new(
                (f64::from(new_center.x)
                    + f64::from(tile.info.rect.left() - old_center.x) * ratio
                    + displacement.x * camera.pixels_per_unit) as f32,
                (f64::from(new_center.y) + f64::from(tile.info.rect.top() - old_center.y) * ratio
                    - displacement.y * camera.pixels_per_unit) as f32,
            ) * screen.pixels_per_point
                - offset;
            let source_scale =
                (camera.pixels_per_unit / tile.info.density) as f32 * screen.pixels_per_point;
            let last = first + tile.info.crop.max.to_vec2() * source_scale;
            let first = first + tile.info.crop.min.to_vec2() * source_scale;
            if last.x <= 0.0 || last.y <= 0.0 || first.x >= size[0] || first.y >= size[1] {
                continue;
            }
            if tile.key.camera != camera || tile.key.viewport != viewport {
                reprojected += 1;
            }
            if budget.draws == 0
                || budget.triangles < 2
                || budget.upload < std::mem::size_of::<PresentationUniform>()
            {
                return Err(ViewError(
                    "Viewport exceeds the GPU presentation budget".into(),
                ));
            }
            let uniform = PresentationUniform {
                rectangle: [first.x, first.y, last.x, last.y],
                uv_rectangle: [
                    tile.info.crop.min.x / tile.info.size[0] as f32,
                    tile.info.crop.min.y / tile.info.size[1] as f32,
                    tile.info.crop.max.x / tile.info.size[0] as f32,
                    tile.info.crop.max.y / tile.info.size[1] as f32,
                ],
                size,
                srgb_target: u32::from(self.format.is_srgb()),
                padding: 0,
            };
            let buffer = self.buffer(
                "rlayout-presentation",
                bytemuck::bytes_of(&uniform),
                wgpu::BufferUsages::UNIFORM,
            )?;
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rlayout-presentation"),
                layout: &self.present_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&tile.image.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffer.buffer.as_entire_binding(),
                    },
                ],
            });
            tiles.push(PresentedTile {
                group,
                _uniform: buffer,
                _image: tile.image.clone(),
            });
            budget.draws -= 1;
            budget.triangles -= 2;
            budget.upload -= std::mem::size_of::<PresentationUniform>();
        }
        Ok(Arc::new(PresentationFrame {
            tiles,
            camera,
            reprojected,
            generation: state.generation,
        }))
    }

    fn encode(
        &mut self,
        scene: &Scene,
        state: &mut SessionState,
        presentation: Option<(Rect, Camera, &ScreenDescriptor)>,
    ) -> Result<Option<wgpu::CommandBuffer>, ViewError> {
        state.collect();
        if state.statistics.error.is_some() {
            return Ok(None);
        }
        if self.in_flight.load(Ordering::Relaxed) >= IN_FLIGHT_LIMIT {
            if let Some((viewport, camera, screen)) = presentation {
                let mut budget = FrameBudget::new(0)?;
                state.presentation =
                    Some(self.presentation(state, viewport, camera, screen, &[], &mut budget)?);
                Self::account_frame(state, &budget);
            }
            return Ok(None);
        }
        let started = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rlayout-frame"),
            });
        let mut keep = KeepAlive::default();
        let mut uploads = None;
        let mut batches = BTreeMap::new();
        let mut mask_passes = 0;
        if let Some(draw) = state.draw.as_mut() {
            // Dynamic offsets belong to one slab submission; continuing a
            // packet in another frame must upload fresh records.
            draw.bindings = None;
        }
        let presentation_draws = if presentation.is_some() {
            let key = state
                .key
                .as_ref()
                .ok_or_else(|| ViewError("Presentation has no view request".into()))?;
            let width = f64::from(key.viewport.width()) * f64::from(key.pixels_per_point);
            let height = f64::from(key.viewport.height()) * f64::from(key.pixels_per_point);
            let maximum_tiles = (width / 256.0).ceil() + 1.0;
            let maximum_tiles = maximum_tiles * ((height / 256.0).ceil() + 1.0);
            if !maximum_tiles.is_finite()
                || maximum_tiles < 0.0
                || maximum_tiles >= DRAW_LIMIT as f64
            {
                return Err(ViewError(
                    "Viewport exceeds the GPU presentation budget".into(),
                ));
            }
            state.previous.len().saturating_add(maximum_tiles as usize)
        } else {
            0
        };
        let mut budget = FrameBudget::new(presentation_draws)?;
        let mut changed = false;
        let mut published = false;
        let mut useful = false;
        let mut complete = false;
        let key = state.key.clone();
        let deadline = started + Duration::from_millis(5);
        while Instant::now() < deadline && budget.draws > 0 && budget.triangles > 0 {
            if let Some(mut draw) = state.draw.take() {
                let active = state
                    .active
                    .as_mut()
                    .ok_or_else(|| ViewError("Missing GPU tile for geometry".into()))?;
                let before_upload = budget.upload;
                let before_draws = budget.draws;
                let done = self.draw(&mut draw, active, &mut budget, &mut uploads, &mut batches)?;
                changed |= before_upload != budget.upload || before_draws != budget.draws;
                if done {
                    active.shape_count = active.shape_count.saturating_add(draw.shape_count);
                } else {
                    state.draw = Some(draw);
                    if before_upload == budget.upload && before_draws == budget.draws {
                        break;
                    }
                }
                continue;
            }
            if state.pending.is_empty() {
                for result in state.worker.receive() {
                    state.pending.push_back(result?);
                }
            }
            let Some(packet) = state.pending.pop_front() else {
                break;
            };
            let key = key
                .as_ref()
                .ok_or_else(|| ViewError("Rendering packet has no view request".into()))?;
            match packet {
                Packet::Begin {
                    generation,
                    tile,
                    layers,
                    first_group,
                    reuse,
                } => {
                    if generation != state.generation {
                        continue;
                    }
                    mask_passes += self.flush_draws(
                        state.active.as_ref(),
                        &mut batches,
                        &mut encoder,
                        &mut keep,
                    )?;
                    self.begin_tile(
                        scene,
                        key,
                        tile,
                        layers,
                        first_group,
                        state,
                        &mut encoder,
                        &mut keep,
                    )?;
                    if state
                        .active
                        .as_ref()
                        .is_some_and(|active| active.cached.is_some())
                    {
                        reuse.store(true, Ordering::Release);
                    }
                    changed = true;
                }
                Packet::Draw {
                    generation,
                    tile_id,
                    layer,
                    mesh,
                    instances,
                    array,
                    shape_count,
                } => {
                    if generation != state.generation {
                        continue;
                    }
                    state.draw = Some(PendingDraw {
                        tile_id,
                        layer,
                        source: mesh,
                        instances,
                        array,
                        shape_count,
                        next_instance: 0,
                        next_index: 0,
                        mesh: None,
                        bindings: None,
                    });
                }
                Packet::Finish {
                    generation,
                    tile_id,
                    last_group,
                    complete: finished,
                } => {
                    if generation != state.generation {
                        continue;
                    }
                    let needs_composition = state
                        .active
                        .as_ref()
                        .is_some_and(|tile| tile.cached.is_none());
                    if needs_composition
                        && (budget.draws == 0
                            || budget.triangles == 0
                            || budget.upload < std::mem::size_of::<Palette>())
                    {
                        state.pending.push_front(Packet::Finish {
                            generation,
                            tile_id,
                            last_group,
                            complete: finished,
                        });
                        break;
                    }
                    mask_passes += self.flush_draws(
                        state.active.as_ref(),
                        &mut batches,
                        &mut encoder,
                        &mut keep,
                    )?;
                    let preview = state.active.as_ref().is_some_and(|tile| tile.info.preview);
                    let finished_tile = self.finish_tile(
                        state,
                        tile_id,
                        last_group,
                        &mut encoder,
                        &mut keep,
                        &mut budget,
                    )?;
                    published |= finished_tile;
                    useful |=
                        finished_tile && state.current.last().is_some_and(|tile| tile.has_geometry);
                    complete |= finished && !preview;
                    changed = true;
                }
            }
        }
        mask_passes +=
            self.flush_draws(state.active.as_ref(), &mut batches, &mut encoder, &mut keep)?;
        state.statistics.mask_passes = state.statistics.mask_passes.saturating_add(mask_passes);
        budget.upload += presentation_draws * std::mem::size_of::<PresentationUniform>();
        budget.draws += presentation_draws;
        budget.triangles += presentation_draws * 2;
        let frame = if let Some((viewport, camera, screen)) = presentation {
            let frame = self.presentation(
                state,
                viewport,
                camera,
                screen,
                &keep.ready_tiles,
                &mut budget,
            )?;
            state.presentation = Some(frame.clone());
            Some(frame)
        } else {
            None
        };
        Self::account_frame(state, &budget);
        if !changed {
            return Ok(None);
        }
        if let Some(mut uploads) = uploads
            && !uploads.bytes.is_empty()
        {
            let slab = uploads
                .lease
                .as_ref()
                .ok_or_else(|| ViewError("Missing submitted upload slab".into()))?;
            self.queue
                .write_buffer(&slab.0.buffer.buffer, 0, &uploads.bytes);
            keep.slabs.push(
                uploads
                    .lease
                    .take()
                    .ok_or_else(|| ViewError("Missing upload slab completion lease".into()))?,
            );
        }
        let sender = state.fence_sender.clone();
        let generation = state.generation;
        let context = state.context.clone();
        let in_flight = self.in_flight.clone();
        in_flight.fetch_add(1, Ordering::Relaxed);
        let mut permit = FramePermit {
            counter: in_flight,
            valid_tiles: keep.valid_tiles.clone(),
            completed: false,
        };
        encoder.on_submitted_work_done(move || {
            let completed_at = Instant::now();
            let invalid = permit
                .valid_tiles
                .iter()
                .any(|tile| !tile.load(Ordering::Acquire));
            for (ready, valid) in &keep.ready_tiles {
                if valid.load(Ordering::Acquire) {
                    ready.store(true, Ordering::Release);
                }
            }
            permit.completed = true;
            if sender
                .send(Fence {
                    generation,
                    published,
                    complete,
                    elapsed: completed_at.duration_since(started).as_secs_f64(),
                    completed_at,
                    invalid,
                    useful,
                })
                .is_err()
                && (published || complete)
            {
                eprintln!("GPU completion receiver disconnected");
            }
            if let Some(context) = context {
                context.request_repaint();
            }
            drop(keep);
            drop(frame);
            drop(permit);
        });
        Ok(Some(encoder.finish()))
    }
    fn account_frame(state: &mut SessionState, budget: &FrameBudget) {
        state.statistics.last_frame_upload_bytes = UPLOAD_LIMIT - budget.upload;
        state.statistics.last_frame_draw_calls = DRAW_LIMIT - budget.draws;
        state.statistics.max_frame_upload_bytes = state
            .statistics
            .max_frame_upload_bytes
            .max(state.statistics.last_frame_upload_bytes);
        state.statistics.max_frame_draws = state
            .statistics
            .max_frame_draws
            .max(state.statistics.last_frame_draw_calls);
        state.statistics.max_frame_triangles = state
            .statistics
            .max_frame_triangles
            .max(TRIANGLE_LIMIT - budget.triangles);
        state.statistics.upload_bytes = state
            .statistics
            .upload_bytes
            .saturating_add(state.statistics.last_frame_upload_bytes);
        state.statistics.draw_calls = state
            .statistics
            .draw_calls
            .saturating_add(state.statistics.last_frame_draw_calls);
        state.statistics.submitted_triangles = state
            .statistics
            .submitted_triangles
            .saturating_add(TRIANGLE_LIMIT - budget.triangles);
    }
}

struct CanvasCallback {
    handle: GpuRendererHandle,
    scene: Arc<Scene>,
    state: Arc<Mutex<SessionState>>,
    viewport: Rect,
    camera: Camera,
    frame: Mutex<Option<Arc<PresentationFrame>>>,
}
impl CallbackTrait for CanvasCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        screen: &ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        _resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let mut operation_generation = None;
        let result = (|| -> Result<Option<wgpu::CommandBuffer>, ViewError> {
            self.handle.poll()?;
            let mut engine = self
                .handle
                .engine
                .lock()
                .map_err(|error| ViewError(error.to_string()))?;
            let mut state = self
                .state
                .lock()
                .map_err(|error| ViewError(error.to_string()))?;
            operation_generation = Some(state.generation);
            let commands = engine.encode(
                &self.scene,
                &mut state,
                Some((self.viewport, self.camera, screen)),
            )?;
            *self
                .frame
                .lock()
                .map_err(|error| ViewError(error.to_string()))? = state.presentation.clone();
            if let Some(frame) = state.presentation.clone() {
                // The egui pass samples tiles even when the custom render queue
                // is full; its submission owns those resources until completion.
                _encoder.on_submitted_work_done(move || drop(frame));
            }
            Ok(commands)
        })();
        match result {
            Ok(Some(commands)) => vec![commands],
            Ok(None) => Vec::new(),
            Err(error) => {
                match self.state.lock() {
                    Ok(mut state) => {
                        if operation_generation
                            .is_none_or(|generation| generation == state.generation)
                        {
                            state.fail(error);
                        }
                    }
                    Err(lock_error) => {
                        eprintln!("Reporting GPU rendering error: {lock_error}; {error}")
                    }
                }
                Vec::new()
            }
        }
    }
    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        _resources: &CallbackResources,
    ) {
        let mut operation_generation = None;
        let result = (|| -> Result<(), ViewError> {
            let frame = self
                .frame
                .lock()
                .map_err(|error| ViewError(error.to_string()))?
                .clone();
            if let Some(frame) = frame {
                operation_generation = Some(frame.generation);
                let engine = self
                    .handle
                    .engine
                    .lock()
                    .map_err(|error| ViewError(error.to_string()))?;
                pass.set_pipeline(&engine.present_pipeline);
                for tile in &frame.tiles {
                    pass.set_bind_group(0, &tile.group, &[]);
                    pass.draw(0..6, 0..1);
                }
                if !frame.tiles.is_empty() {
                    let mut state = self
                        .state
                        .lock()
                        .map_err(|error| ViewError(error.to_string()))?;
                    if state.generation == frame.generation {
                        state.statistics.presented_camera = Some(frame.camera);
                        state.statistics.last_presented_at = Some(Instant::now());
                        state.statistics.reprojected_tiles = frame.reprojected;
                    }
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            match self.state.lock() {
                Ok(mut state)
                    if operation_generation
                        .is_none_or(|generation| generation == state.generation) =>
                {
                    state.fail(error)
                }
                Ok(_) => {}
                Err(lock_error) => {
                    eprintln!("Reporting GPU presentation error: {lock_error}; {error}")
                }
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn capture(
    handle: &GpuRendererHandle,
    scene: Arc<Scene>,
    state: Arc<Mutex<SessionState>>,
    key: FrameKey,
    context: egui::Context,
) -> Result<RenderedImage, ViewError> {
    handle.poll()?;
    let generation = {
        let mut state = state.lock().map_err(|error| ViewError(error.to_string()))?;
        state.begin(scene.clone(), key.clone(), context, false)?;
        state.generation
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        capture_requested(handle, scene, state.clone(), key, generation)
    }))
    .unwrap_or_else(|_| Err(ViewError("GPU capture worker stopped unexpectedly".into())));
    if let Err(error) = &result {
        match state.lock() {
            Ok(mut state) if state.generation == generation => {
                state.fail(ViewError(error.to_string()));
            }
            Ok(_) => {}
            Err(lock_error) => eprintln!("Reporting GPU capture failure: {lock_error}"),
        }
    }
    result
}
#[cfg(not(target_arch = "wasm32"))]
fn capture_requested(
    handle: &GpuRendererHandle,
    scene: Arc<Scene>,
    state: Arc<Mutex<SessionState>>,
    key: FrameKey,
    generation: u64,
) -> Result<RenderedImage, ViewError> {
    let deadline = Instant::now() + Duration::from_secs(180);
    let diagnostics = std::env::var_os("RL_RENDER_DIAGNOSTICS").is_some();
    let mut next_diagnostic = Instant::now();
    loop {
        handle.poll()?;
        let (commands, complete) = {
            let mut engine = handle
                .engine
                .lock()
                .map_err(|error| ViewError(error.to_string()))?;
            let mut state = state.lock().map_err(|error| ViewError(error.to_string()))?;
            if state.generation != generation {
                return Err(ViewError(
                    "GPU capture was superseded by another view request".into(),
                ));
            }
            if let Some(error) = &state.statistics.error {
                return Err(ViewError(error.clone()));
            }
            let commands = match engine.encode(&scene, &mut state, None) {
                Ok(commands) => commands,
                Err(error) => {
                    state.fail(ViewError(error.to_string()));
                    return Err(error);
                }
            };
            if diagnostics && Instant::now() >= next_diagnostic {
                let statistics = &state.statistics;
                eprintln!(
                    "render_progress elapsed={:.3} tiles={} shapes={} preparation={:.3} completion_wait={:.3} triangles={} draws={} mask_passes={} uploaded={} cpu_bytes={} gpu_bytes={} in_flight={} pending={} active_tile={:?} draw_mesh={:?} mesh_triangles={:?} draw_instance={:?}",
                    state.started.elapsed().as_secs_f64(),
                    statistics.tile_count,
                    statistics.shape_count,
                    statistics.preparation_seconds,
                    statistics.gpu_completion_latency_seconds,
                    statistics.submitted_triangles,
                    statistics.draw_calls,
                    statistics.mask_passes,
                    statistics.upload_bytes,
                    statistics.cpu_bytes,
                    engine.ledger.load(Ordering::Relaxed),
                    engine.in_flight.load(Ordering::Relaxed),
                    state.pending.len(),
                    state.active.as_ref().map(|tile| tile.info.id),
                    state.draw.as_ref().map(|draw| draw.source.id),
                    state
                        .draw
                        .as_ref()
                        .map(|draw| draw.source.indices.len() / 3),
                    state.draw.as_ref().map(|draw| draw.next_instance),
                );
                next_diagnostic = Instant::now() + Duration::from_secs(1);
            }
            (commands, state.statistics.complete)
        };
        let submitted = commands.is_some();
        if let Some(commands) = commands {
            handle.queue.submit([commands]);
        }
        if complete {
            break;
        }
        if Instant::now() >= deadline {
            return Err(ViewError(
                "GPU capture timed out while preparing the view".into(),
            ));
        }
        if !submitted {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    let size = [
        (key.viewport.width() * key.pixels_per_point)
            .ceil()
            .max(1.0) as u32,
        (key.viewport.height() * key.pixels_per_point)
            .ceil()
            .max(1.0) as u32,
    ];
    let row_bytes = size[0]
        .checked_mul(4)
        .ok_or_else(|| ViewError("GPU capture row size overflow".into()))?;
    let padded_row = row_bytes
        .checked_add(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1)
        .map(|bytes| {
            bytes / wgpu::COPY_BYTES_PER_ROW_ALIGNMENT * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
        })
        .ok_or_else(|| ViewError("GPU capture row padding overflow".into()))?;
    let buffer_bytes = (padded_row as usize)
        .checked_mul(size[1] as usize)
        .ok_or_else(|| ViewError("GPU capture buffer size overflow".into()))?;
    let (target, buffer, commands, mut statistics) = {
        let mut engine = handle
            .engine
            .lock()
            .map_err(|error| ViewError(error.to_string()))?;
        let mut state = state.lock().map_err(|error| ViewError(error.to_string()))?;
        state.collect();
        if state.generation != generation {
            return Err(ViewError(
                "GPU capture was superseded before readback".into(),
            ));
        }
        let target = engine.texture(
            "rlayout-capture",
            size,
            1,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
        )?;
        engine.room(buffer_bytes)?;
        let raw_buffer = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlayout-readback"),
            size: buffer_bytes as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        engine.ledger.fetch_add(buffer_bytes, Ordering::Relaxed);
        let buffer = Arc::new(GpuBuffer {
            buffer: raw_buffer,
            _allocation: Allocation {
                bytes: buffer_bytes,
                ledger: engine.ledger.clone(),
            },
        });
        let mut encoder = engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rlayout-capture"),
            });
        clear(&mut encoder, &target.view);
        let mut readback_sources = Vec::new();
        for tile in &state.current {
            let left = ((tile.info.rect.left() - key.viewport.left()) * key.pixels_per_point)
                .round() as i64;
            let top =
                ((tile.info.rect.top() - key.viewport.top()) * key.pixels_per_point).round() as i64;
            let source_x = (-left)
                .max(0)
                .max(tile.info.crop.min.x as i64)
                .min(i64::from(tile.info.size[0]));
            let source_y = (-top)
                .max(0)
                .max(tile.info.crop.min.y as i64)
                .min(i64::from(tile.info.size[1]));
            let right = (i64::from(size[0]) - left)
                .min(i64::from(tile.info.size[0]))
                .min(tile.info.crop.max.x as i64);
            let bottom = (i64::from(size[1]) - top)
                .min(i64::from(tile.info.size[1]))
                .min(tile.info.crop.max.y as i64);
            if right <= source_x || bottom <= source_y {
                continue;
            }
            let destination_x = (left + source_x) as u32;
            let destination_y = (top + source_y) as u32;
            let width = (right - source_x) as u32;
            let height = (bottom - source_y) as u32;
            let source_x = source_x as u32;
            let source_y = source_y as u32;
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tile.image.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: source_x,
                        y: source_y,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &target.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: destination_x,
                        y: destination_y,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            readback_sources.push(tile.image.clone());
        }
        encoder.copy_texture_to_buffer(
            target.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(size[1]),
                },
            },
            wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
        );
        let retained_target = target.clone();
        let retained_buffer = buffer.clone();
        encoder.on_submitted_work_done(move || {
            drop(retained_target);
            drop(retained_buffer);
            drop(readback_sources);
        });
        let statistics = state.statistics.clone();
        (target, buffer, encoder.finish(), statistics)
    };
    let (sender, receiver) = mpsc::channel();
    commands.map_buffer_on_submit(&buffer.buffer, wgpu::MapMode::Read, .., move |result| {
        if sender
            .send(result.map_err(|error| ViewError(format!("Mapping GPU capture: {error}"))))
            .is_err()
        {
            eprintln!("GPU capture map receiver disconnected");
        }
    });
    let submitted = handle.queue.submit([commands]);
    handle
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submitted),
            timeout: Some(Duration::from_secs(60)),
        })
        .map_err(|error| ViewError(format!("Waiting for GPU capture: {error}")))?;
    receiver
        .recv_timeout(Duration::from_secs(1))
        .map_err(|error| ViewError(format!("Receiving GPU capture: {error}")))??;
    let mut pixels = Vec::with_capacity(size[0] as usize * size[1] as usize);
    {
        let mapped = buffer
            .buffer
            .slice(..)
            .get_mapped_range()
            .map_err(|error| ViewError(format!("Reading GPU capture: {error}")))?;
        for row in mapped.chunks(padded_row as usize).take(size[1] as usize) {
            let row = row
                .get(..row_bytes as usize)
                .ok_or_else(|| ViewError("Incomplete GPU capture row".into()))?;
            for [red, green, blue, alpha] in row.as_chunks::<4>().0 {
                pixels.push(Color32::from_rgba_premultiplied(
                    *red, *green, *blue, *alpha,
                ));
            }
        }
    }
    buffer.buffer.unmap();
    statistics.gpu_bytes = handle.ledger.load(Ordering::Relaxed);
    statistics.in_flight_frames = handle.in_flight.load(Ordering::Relaxed);
    drop(target);
    drop(buffer);
    let cache_bytes = handle.ledger.load(Ordering::Relaxed);
    Ok(RenderedImage {
        image: egui::ColorImage::new([size[0] as usize, size[1] as usize], pixels),
        shape_count: statistics.shape_count,
        cache_bytes,
        statistics,
    })
}

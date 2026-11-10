use super::*;
use std::collections::{HashSet, VecDeque};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::{
    Condvar, Mutex,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    mpsc,
};
use std::thread::{self, JoinHandle};

const TILE_SIZE: usize = 256;
const CACHE_LIMIT: usize = 128 * 1024 * 1024;
const PACKET_LIMIT: usize = 4 * 1024 * 1024;
const CHUNK_SHAPES: usize = 64;
const INSTANCE_LIMIT: usize = 4096;
const PIXEL_ERROR: f64 = 1.0 / 64.0;
const CACHE_STORAGE: usize = 48 * 1024 * 1024;
const IDENTIFIER_STORAGE: usize = 8 * 1024 * 1024;
const ACTIVE_RESERVE: usize = 40 * 1024 * 1024;
const TESSELLATION_SCRATCH: usize = 24 * 1024 * 1024;
const TESSELLATION_BYTES_PER_NODE: usize = 4096;
const TESSELLATION_COMPARISONS: usize = 1_000_000;
const TEXT_SCRATCH: usize = 8 * 1024 * 1024;
const TEXT_BYTES_PER_CHARACTER: usize = 512;
const CHUNK_SCRATCH: usize = 8 * 1024 * 1024;
const CHUNK_BUDGET_ERROR: &str = "Geometry chunk exceeds bounded preparation scratch budget";

static RESOURCE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Default)]
struct RenderHasher(u64);

impl Hasher for RenderHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = 0_u64;
            for (offset, byte) in chunk.iter().enumerate() {
                word |= u64::from(*byte) << (offset * 8);
            }
            self.write_u64(word);
        }
    }

    fn write_u64(&mut self, value: u64) {
        // Bounded renderer maps use integer IDs and coordinate bits, so their
        // frequent instance lookups can use this inexpensive integer mix.
        self.0 = (self.0.rotate_left(5) ^ value).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    fn write_u32(&mut self, value: u32) {
        self.write_u64(u64::from(value));
    }
    fn write_u16(&mut self, value: u16) {
        self.write_u64(u64::from(value));
    }
    fn write_u8(&mut self, value: u8) {
        self.write_u64(u64::from(value));
    }
    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }
}

type RenderMap<K, V> = HashMap<K, V, BuildHasherDefault<RenderHasher>>;
type RenderSet<K> = HashSet<K, BuildHasherDefault<RenderHasher>>;

fn resource_id() -> Result<u64, ViewError> {
    RESOURCE_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |identifier| {
            identifier.checked_add(1)
        })
        .map_err(|_| ViewError("Rendering resource identifiers exhausted".into()))
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct GpuVertex {
    pub position: [f32; 2],
    pub other: [f32; 2],
    pub uv: [f32; 2],
    pub bounds: [f32; 4],
    pub kind: u32,
    pub corner: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Instance {
    pub affine: [f32; 4],
    pub translation: [f32; 2],
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ArrayInstances {
    pub column_step: [f32; 2],
    pub row_step: [f32; 2],
    pub columns: u32,
    pub rows: u32,
}

pub(super) struct GlyphBitmap {
    pub id: u64,
    pub size: [u32; 2],
    pub alpha: Vec<u8>,
}

pub(super) struct MeshData {
    pub id: u64,
    pub vertices: Vec<GpuVertex>,
    pub indices: Vec<u32>,
    pub glyph: Option<Arc<GlyphBitmap>>,
}

impl MeshData {
    pub(super) fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + std::mem::size_of::<usize>() * 2
            + self.vertices.capacity() * std::mem::size_of::<GpuVertex>()
            + self.indices.capacity() * std::mem::size_of::<u32>()
            + self.glyph.as_ref().map_or(0, |glyph| {
                glyph.alpha.capacity()
                    + std::mem::size_of::<GlyphBitmap>()
                    + std::mem::size_of::<usize>() * 2
            })
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct TileInfo {
    pub id: u64,
    pub rect: Rect,
    pub size: [u32; 2],
    pub preview: bool,
    pub grid: [i64; 2],
    pub phase: [f64; 2],
    pub density: f64,
    pub crop: Rect,
}

pub(super) enum Packet {
    Begin {
        generation: u64,
        tile: TileInfo,
        layers: Vec<(LayerKey, LayerStyle)>,
        first_group: bool,
        reuse: Arc<AtomicBool>,
    },
    Draw {
        generation: u64,
        tile_id: u64,
        layer: LayerKey,
        mesh: Arc<MeshData>,
        instances: Vec<Instance>,
        array: Option<ArrayInstances>,
        shape_count: usize,
    },
    Finish {
        generation: u64,
        tile_id: u64,
        last_group: bool,
        complete: bool,
    },
}

struct Job {
    generation: u64,
    scene: Arc<Scene>,
    key: FrameKey,
    context: egui::Context,
    preview: bool,
}

struct Mailbox {
    pending: Option<Job>,
    stopped: bool,
}

struct Shared {
    mailbox: Mutex<Mailbox>,
    wake: Condvar,
    generation: AtomicU64,
    stopped: AtomicBool,
    active: AtomicBool,
    cache_bytes: AtomicUsize,
    packet_bytes: Arc<AtomicUsize>,
    preparation_nanos: AtomicU64,
    blocked_nanos: AtomicU64,
    #[cfg(test)]
    complete_leaf_cells: AtomicUsize,
}

struct Delivery {
    generation: u64,
    result: Option<Result<Vec<Packet>, ViewError>>,
    bytes: usize,
    accounting: Arc<AtomicUsize>,
}

impl Drop for Delivery {
    fn drop(&mut self) {
        self.accounting.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

pub(super) struct Worker {
    shared: Arc<Shared>,
    receiver: mpsc::Receiver<Delivery>,
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    pub(super) fn new() -> Result<Self, ViewError> {
        let shared = Arc::new(Shared {
            mailbox: Mutex::new(Mailbox {
                pending: None,
                stopped: false,
            }),
            wake: Condvar::new(),
            generation: AtomicU64::new(0),
            stopped: AtomicBool::new(false),
            active: AtomicBool::new(false),
            cache_bytes: AtomicUsize::new(0),
            packet_bytes: Arc::new(AtomicUsize::new(0)),
            preparation_nanos: AtomicU64::new(0),
            blocked_nanos: AtomicU64::new(0),
            #[cfg(test)]
            complete_leaf_cells: AtomicUsize::new(0),
        });
        let (sender, receiver) = mpsc::sync_channel(2);
        let handle = thread::Builder::new()
            .name("gds-geometry".into())
            .spawn({
                let shared = shared.clone();
                move || worker_loop(shared, sender)
            })
            .map_err(|error| ViewError(format!("Starting geometry worker: {error}")))?;
        Ok(Self {
            shared,
            receiver,
            handle: Some(handle),
        })
    }

    pub(super) fn start(
        &mut self,
        scene: Arc<Scene>,
        key: FrameKey,
        context: egui::Context,
        preview: bool,
    ) -> Result<u64, ViewError> {
        let generation = self
            .shared
            .generation
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |generation| {
                generation.checked_add(1)
            })
            .map_err(|_| ViewError("Rendering request identifiers exhausted".into()))?
            + 1;
        let mut mailbox = self
            .shared
            .mailbox
            .lock()
            .map_err(|error| ViewError(error.to_string()))?;
        if mailbox.stopped {
            return Err(ViewError("Geometry worker has stopped".into()));
        }
        mailbox.pending = Some(Job {
            generation,
            scene,
            key,
            context,
            preview,
        });
        self.shared.wake.notify_one();
        Ok(generation)
    }

    pub(super) fn cancel(&mut self) {
        let exhausted = self
            .shared
            .generation
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |generation| {
                generation.checked_add(1)
            })
            .is_err();
        match self.shared.mailbox.lock() {
            Ok(mut mailbox) => {
                mailbox.pending = None;
                if exhausted {
                    mailbox.stopped = true;
                    self.shared.stopped.store(true, Ordering::Release);
                    eprintln!("Rendering request identifiers exhausted");
                }
            }
            Err(error) => eprintln!("Cancelling geometry request: {error}"),
        }
        self.shared.wake.notify_one();
    }

    pub(super) fn receive(&mut self) -> Vec<Result<Packet, ViewError>> {
        let generation = self.shared.generation.load(Ordering::Acquire);
        let mut results = Vec::new();
        for _ in 0..2 {
            let Ok(mut delivery) = self.receiver.try_recv() else {
                break;
            };
            if delivery.generation == generation
                && let Some(result) = delivery.result.take()
            {
                match result {
                    Ok(packets) => results.extend(packets.into_iter().map(Ok)),
                    Err(error) => results.push(Err(error)),
                }
            }
        }
        if self.handle.as_ref().is_some_and(JoinHandle::is_finished)
            && let Some(handle) = self.handle.take()
        {
            let message = if handle.join().is_err() {
                "Geometry worker stopped unexpectedly"
            } else {
                "Geometry worker disconnected"
            };
            results.push(Err(ViewError(message.into())));
        }
        results
    }

    pub(super) fn bytes(&self) -> usize {
        self.shared.cache_bytes.load(Ordering::Relaxed)
            + self.shared.packet_bytes.load(Ordering::Relaxed)
            + if self.shared.active.load(Ordering::Relaxed) {
                ACTIVE_RESERVE
            } else {
                0
            }
    }

    pub(super) fn preparation_seconds(&self) -> f64 {
        self.shared.preparation_nanos.load(Ordering::Relaxed) as f64 / 1e9
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.cancel();
        self.shared.stopped.store(true, Ordering::Release);
        match self.shared.mailbox.lock() {
            Ok(mut mailbox) => mailbox.stopped = true,
            Err(error) => eprintln!("Stopping geometry worker: {error}"),
        }
        self.shared.wake.notify_one();
        if self.handle.as_ref().is_some_and(JoinHandle::is_finished)
            && let Some(handle) = self.handle.take()
            && handle.join().is_err()
        {
            eprintln!("Geometry worker stopped unexpectedly");
        }
    }
}

fn deliver(
    shared: &Shared,
    sender: &mut PacketSender,
    generation: u64,
    result: Result<Packet, ViewError>,
) -> bool {
    deliver_checked(shared, sender, generation, result, None)
}

fn deliver_checked(
    shared: &Shared,
    sender: &mut PacketSender,
    generation: u64,
    result: Result<Packet, ViewError>,
    reuse: Option<&AtomicBool>,
) -> bool {
    if shared.generation.load(Ordering::Acquire) != generation
        || shared.stopped.load(Ordering::Acquire)
        || reuse.is_some_and(|reuse| reuse.load(Ordering::Acquire))
    {
        sender.discard();
        return false;
    }
    sender.enqueue(shared, generation, result)
}

struct PacketSender {
    channel: mpsc::SyncSender<Delivery>,
    pending: Vec<Packet>,
    meshes: RenderSet<usize>,
    bytes: usize,
    reuse: Option<Arc<AtomicBool>>,
}

impl PacketSender {
    fn new(channel: mpsc::SyncSender<Delivery>) -> Self {
        Self {
            channel,
            pending: Vec::new(),
            meshes: RenderSet::default(),
            bytes: 256,
            reuse: None,
        }
    }

    fn discard(&mut self) {
        self.pending = Vec::new();
        self.meshes.clear();
        self.bytes = 256 + self.meshes.capacity() * 16;
    }

    fn enqueue(
        &mut self,
        shared: &Shared,
        generation: u64,
        result: Result<Packet, ViewError>,
    ) -> bool {
        if let Ok(Packet::Draw {
            mesh, instances, ..
        }) = &result
        {
            let pointer = Arc::as_ptr(mesh) as usize;
            let extra = instances.capacity() * std::mem::size_of::<Instance>()
                + std::mem::size_of::<Packet>() * 2
                + if self.meshes.contains(&pointer) {
                    0
                } else {
                    mesh.bytes() + 32
                };
            if self.bytes.saturating_add(extra) > PACKET_LIMIT && !self.flush(shared, generation) {
                return false;
            }
            if let Ok(packet) = result {
                let extra = match &packet {
                    Packet::Draw {
                        mesh, instances, ..
                    } => {
                        instances.capacity() * std::mem::size_of::<Instance>()
                            + std::mem::size_of::<Packet>() * 2
                            + if self.meshes.insert(Arc::as_ptr(mesh) as usize) {
                                mesh.bytes() + 32
                            } else {
                                0
                            }
                    }
                    _ => 0,
                };
                if self.bytes.saturating_add(extra) > PACKET_LIMIT {
                    self.send(
                        shared,
                        generation,
                        Err(ViewError(
                            "Geometry delivery exceeds bounded packet budget".into(),
                        )),
                        256,
                    );
                    return false;
                }
                self.bytes += extra;
                self.pending.push(packet);
            }
            return true;
        }
        if !self.flush(shared, generation) {
            return false;
        }
        let bytes = match &result {
            Ok(Packet::Begin { layers, reuse, .. }) => {
                self.reuse = Some(reuse.clone());
                layers.capacity() * std::mem::size_of::<(LayerKey, LayerStyle)>()
            }
            Err(error) => error.0.capacity(),
            _ => 0,
        } + std::mem::size_of::<Packet>() * 2
            + std::mem::size_of::<Delivery>();
        self.send(shared, generation, result.map(|packet| vec![packet]), bytes)
    }

    fn flush(&mut self, shared: &Shared, generation: u64) -> bool {
        if self.pending.is_empty() {
            return true;
        }
        if self
            .reuse
            .as_ref()
            .is_some_and(|reuse| reuse.load(Ordering::Acquire))
        {
            self.discard();
            return true;
        }
        let packets = std::mem::take(&mut self.pending);
        let bytes = self.bytes;
        self.discard();
        self.send(shared, generation, Ok(packets), bytes)
    }

    fn send(
        &self,
        shared: &Shared,
        generation: u64,
        result: Result<Vec<Packet>, ViewError>,
        bytes: usize,
    ) -> bool {
        if shared.generation.load(Ordering::Acquire) != generation
            || shared.stopped.load(Ordering::Acquire)
        {
            return false;
        }
        shared.packet_bytes.fetch_add(bytes, Ordering::Relaxed);
        let delivery = Delivery {
            generation,
            result: Some(result),
            bytes,
            accounting: shared.packet_bytes.clone(),
        };
        let _wait = WaitTime {
            started: Some(std::time::Instant::now()),
            counter: &shared.blocked_nanos,
        };
        match self.channel.send(delivery) {
            Ok(()) => true,
            Err(_) => {
                if !shared.stopped.load(Ordering::Acquire) {
                    eprintln!("Geometry result receiver disconnected");
                }
                false
            }
        }
    }
}

struct WaitTime<'a> {
    started: Option<std::time::Instant>,
    counter: &'a AtomicU64,
}

impl Drop for WaitTime<'_> {
    fn drop(&mut self) {
        if let Some(started) = self.started {
            self.counter.fetch_add(
                started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
                Ordering::Relaxed,
            );
        }
    }
}

fn update_timing(shared: &Shared, started: std::time::Instant) {
    let elapsed = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    shared.preparation_nanos.store(
        elapsed.saturating_sub(shared.blocked_nanos.load(Ordering::Relaxed)),
        Ordering::Relaxed,
    );
}

fn worker_loop(shared: Arc<Shared>, sender: mpsc::SyncSender<Delivery>) {
    let mut sender = PacketSender::new(sender);
    let mut cache = Cache::new();
    loop {
        let job = {
            let mut mailbox = match shared.mailbox.lock() {
                Ok(mailbox) => mailbox,
                Err(error) => {
                    eprintln!("Receiving geometry request: {error}");
                    return;
                }
            };
            while mailbox.pending.is_none() && !mailbox.stopped {
                mailbox = match shared.wake.wait(mailbox) {
                    Ok(mailbox) => mailbox,
                    Err(error) => {
                        eprintln!("Waiting for geometry request: {error}");
                        return;
                    }
                };
            }
            if mailbox.stopped {
                return;
            }
            mailbox.pending.take()
        };
        if let Some(job) = job {
            sender.discard();
            sender.reuse = None;
            shared.active.store(true, Ordering::Relaxed);
            shared.preparation_nanos.store(0, Ordering::Relaxed);
            shared.blocked_nanos.store(0, Ordering::Relaxed);
            #[cfg(test)]
            shared.complete_leaf_cells.store(0, Ordering::Relaxed);
            let started = std::time::Instant::now();
            if let Err(error) = render_job(&job, &shared, &mut sender, &mut cache, started)
                && shared.generation.load(Ordering::Acquire) == job.generation
            {
                deliver(&shared, &mut sender, job.generation, Err(error));
                job.context.request_repaint();
            }
            shared
                .cache_bytes
                .store(cache.total_bytes(), Ordering::Relaxed);
            update_timing(&shared, started);
            shared.active.store(false, Ordering::Relaxed);
            sender.discard();
        }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct ChunkKey {
    revision: u64,
    cell: CellId,
    start: usize,
    end: usize,
    magnification: u64,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct GlyphKey {
    text: String,
    size: u64,
    pixels_per_point: u32,
}

#[derive(Clone, Copy)]
struct PreciseVertex {
    position: DoublePoint,
    other: DoublePoint,
    uv: [f32; 2],
    bounds: Bounds,
    kind: u32,
    corner: u32,
}

struct PreparedMesh {
    layer: LayerKey,
    origin: DoublePoint,
    bounds: Bounds,
    precise: Vec<PreciseVertex>,
    data: Arc<MeshData>,
    shape_count: usize,
    rounding: [f64; 2],
    extent: [f64; 2],
}

struct PreparedChunk {
    range: (usize, usize),
    meshes: Vec<PreparedMesh>,
}

impl PreparedChunk {
    fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + std::mem::size_of::<usize>() * 2
            + self.meshes.capacity() * std::mem::size_of::<PreparedMesh>()
            + self
                .meshes
                .iter()
                .map(|mesh| {
                    mesh.data.bytes()
                        + mesh.precise.capacity() * std::mem::size_of::<PreciseVertex>()
                })
                .sum::<usize>()
    }
}

struct PreparedGlyph {
    logical_size: Vec2,
    mesh: Arc<MeshData>,
}

enum Cached {
    Chunk(ChunkKey),
    Glyph(GlyphKey),
}

struct Cache {
    chunks: RenderMap<ChunkKey, CacheEntry<PreparedChunk>>,
    glyphs: HashMap<GlyphKey, CacheEntry<PreparedGlyph>>,
    order: VecDeque<Cached>,
    bytes: usize,
    limit: usize,
    mesh_identifiers: MeshIdentifiers,
}

struct CacheEntry<T> {
    data: Arc<T>,
    recent: bool,
}

struct MeshIdentifiers {
    entries: RenderMap<ChunkKey, Vec<u64>>,
    ranges: RenderMap<RangeKey, usize>,
    order: VecDeque<IdentifierEntry>,
    bytes: usize,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct RangeKey {
    revision: u64,
    cell: CellId,
    start: usize,
}

enum IdentifierEntry {
    Mesh(ChunkKey),
    Range(RangeKey),
}

impl MeshIdentifiers {
    fn new() -> Self {
        Self {
            entries: RenderMap::default(),
            ranges: RenderMap::default(),
            order: VecDeque::new(),
            bytes: 0,
        }
    }

    fn entry_bytes(capacity: usize) -> usize {
        std::mem::size_of::<ChunkKey>() * 2
            + std::mem::size_of::<Vec<u64>>()
            + 32
            + capacity * std::mem::size_of::<u64>()
    }

    fn range_bytes() -> usize {
        std::mem::size_of::<RangeKey>() + std::mem::size_of::<IdentifierEntry>() + 64
    }

    fn reserve(&mut self, bytes: usize) {
        while self.bytes.saturating_add(bytes) > IDENTIFIER_STORAGE {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            match oldest {
                IdentifierEntry::Mesh(key) => {
                    if let Some(identifiers) = self.entries.remove(&key) {
                        self.bytes = self
                            .bytes
                            .saturating_sub(Self::entry_bytes(identifiers.capacity()));
                    }
                }
                IdentifierEntry::Range(key) => {
                    if self.ranges.remove(&key).is_some() {
                        self.bytes = self.bytes.saturating_sub(Self::range_bytes());
                    }
                }
            }
        }
    }

    fn remember_range(&mut self, key: RangeKey, width: usize) {
        if width == CHUNK_SHAPES {
            return;
        }
        if let Some(previous) = self.ranges.get_mut(&key) {
            *previous = (*previous).min(width);
            return;
        }
        let bytes = Self::range_bytes();
        self.reserve(bytes);
        self.bytes += bytes;
        self.ranges.insert(key, width);
        self.order.push_back(IdentifierEntry::Range(key));
    }

    fn get(&mut self, key: ChunkKey, count: usize) -> Result<Vec<u64>, ViewError> {
        if let Some(identifiers) = self.entries.get(&key) {
            if identifiers.len() != count {
                return Err(ViewError(
                    "Immutable geometry preparation changed mesh structure".into(),
                ));
            }
            return Ok(identifiers.clone());
        }
        let identifiers = (0..count)
            .map(|_| resource_id())
            .collect::<Result<Vec<_>, _>>()?;
        let bytes = Self::entry_bytes(identifiers.capacity());
        if bytes <= IDENTIFIER_STORAGE {
            self.reserve(bytes);
            self.bytes += bytes;
            self.order.push_back(IdentifierEntry::Mesh(key));
            self.entries.insert(key, identifiers.clone());
        }
        Ok(identifiers)
    }
}

impl Cache {
    fn new() -> Self {
        Self {
            chunks: RenderMap::default(),
            glyphs: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
            limit: CACHE_STORAGE,
            mesh_identifiers: MeshIdentifiers::new(),
        }
    }

    fn total_bytes(&self) -> usize {
        self.bytes + self.metadata_bytes(false) + self.mesh_identifiers.bytes
    }

    fn metadata_bytes(&self, inserting: bool) -> usize {
        fn table_bytes<K, V>(capacity: usize, length: usize, inserting: bool) -> usize {
            let buckets = if capacity == 0 {
                0
            } else {
                capacity.saturating_add(1).next_power_of_two()
            };
            let buckets = if inserting && length.saturating_add(1) > capacity {
                buckets.saturating_mul(2).max(4)
            } else {
                buckets
            };
            if buckets == 0 {
                0
            } else {
                buckets * (std::mem::size_of::<(K, V)>() + 1) + 32
            }
        }
        let order_capacity =
            if inserting && self.order.len().saturating_add(1) > self.order.capacity() {
                self.order.capacity().saturating_mul(2).max(4)
            } else {
                self.order.capacity()
            };
        table_bytes::<ChunkKey, CacheEntry<PreparedChunk>>(
            self.chunks.capacity(),
            self.chunks.len(),
            inserting,
        ) + table_bytes::<GlyphKey, CacheEntry<PreparedGlyph>>(
            self.glyphs.capacity(),
            self.glyphs.len(),
            inserting,
        ) + order_capacity * std::mem::size_of::<Cached>()
    }

    fn reserve(&mut self, bytes: usize) -> bool {
        if bytes > self.limit {
            return false;
        }
        while self
            .bytes
            .saturating_add(bytes)
            .saturating_add(self.metadata_bytes(true))
            > self.limit
        {
            let Some(oldest) = self.order.pop_front() else {
                return false;
            };
            match oldest {
                Cached::Chunk(key) => {
                    if let Some(chunk) = self.chunks.get_mut(&key)
                        && chunk.recent
                    {
                        chunk.recent = false;
                        self.order.push_back(Cached::Chunk(key));
                        continue;
                    }
                    if let Some(chunk) = self.chunks.remove(&key) {
                        self.bytes = self.bytes.saturating_sub(chunk.data.bytes());
                    }
                }
                Cached::Glyph(key) => {
                    if let Some(glyph) = self.glyphs.get_mut(&key)
                        && glyph.recent
                    {
                        glyph.recent = false;
                        self.order.push_back(Cached::Glyph(key));
                        continue;
                    }
                    if let Some(glyph) = self.glyphs.remove(&key) {
                        self.bytes = self.bytes.saturating_sub(
                            glyph.data.mesh.bytes()
                                + key.text.capacity() * 2
                                + std::mem::size_of::<PreparedGlyph>()
                                + std::mem::size_of::<usize>() * 2,
                        );
                    }
                }
            }
        }
        true
    }

    fn chunk(
        &mut self,
        scene: &Scene,
        cell: CellId,
        index: usize,
        magnification: f64,
        cancelled: impl Fn() -> bool,
    ) -> Result<Arc<PreparedChunk>, ViewError> {
        let source = scene
            .layout
            .cell(cell)
            .ok_or_else(|| ViewError("Missing prepared cell".into()))?;
        let range_key = RangeKey {
            revision: scene.revision,
            cell,
            start: index / CHUNK_SHAPES * CHUNK_SHAPES,
        };
        let mut width = match self.mesh_identifiers.ranges.get(&range_key) {
            Some(width) => *width,
            None => CHUNK_SHAPES,
        };
        loop {
            let nominal_start = index / width * width;
            let nominal_end = nominal_start
                .saturating_add(width)
                .min(source.shapes().len());
            if let Some(chunk) = self.cached_chunk(
                scene.revision,
                cell,
                (nominal_start, nominal_end),
                magnification,
            ) {
                return Ok(chunk);
            }
            let ((start, end), actual_width) =
                chunk_range(source.shapes(), index, width).map_err(|error| {
                    match source.shapes().get(index) {
                        Some(shape) => shape_error(source.name(), shape.layer, index, error),
                        None => error,
                    }
                })?;
            if actual_width != width
                && let Some(chunk) =
                    self.cached_chunk(scene.revision, cell, (start, end), magnification)
            {
                return Ok(chunk);
            }
            width = actual_width;
            let shapes = source
                .shapes()
                .get(start..end)
                .ok_or_else(|| ViewError("Invalid preparation range".into()))?;
            let absolute = shapes.iter().any(
                |shape| matches!(&shape.geometry, Geometry::Path(path) if path.absolute_width),
            );
            let key = ChunkKey {
                revision: scene.revision,
                cell,
                start,
                end,
                magnification: if absolute { magnification.to_bits() } else { 0 },
            };
            if let Some(chunk) = self.chunks.get_mut(&key) {
                chunk.recent = true;
                return Ok(chunk.data.clone());
            }
            let prepared = (|| {
                let mut builders: BTreeMap<LayerKey, Builder> = BTreeMap::new();
                let mut meshes = Vec::new();
                for (offset, shape) in shapes.iter().enumerate() {
                    if cancelled() {
                        return Err(ViewError("Rendering cancelled".into()));
                    }
                    if matches!(shape.geometry, Geometry::Label(_)) {
                        continue;
                    }
                    let other_bytes = builders
                        .iter()
                        .filter(|(layer, _)| **layer != shape.layer)
                        .map(|(_, builder)| builder.estimated_bytes())
                        .sum::<usize>();
                    let builder = builders
                        .entry(shape.layer)
                        .or_insert_with(|| Builder::new(shape.layer));
                    builder.limit = CHUNK_SCRATCH.saturating_sub(other_bytes);
                    builder
                        .geometry(&shape.geometry, magnification, &mut meshes, &cancelled)
                        .map_err(|error| {
                            shape_error(source.name(), shape.layer, start + offset, error)
                        })?;
                }
                for builder in builders.into_values() {
                    builder.finish(&mut meshes)?;
                }
                Ok::<_, ViewError>(meshes)
            })();
            let mut meshes = match prepared {
                Ok(meshes) => meshes,
                Err(error) if width > 1 && error.0.contains(CHUNK_BUDGET_ERROR) => {
                    width /= 4;
                    self.mesh_identifiers.remember_range(range_key, width);
                    continue;
                }
                Err(error) => return Err(error),
            };
            self.mesh_identifiers.remember_range(range_key, width);
            let identifiers = self.mesh_identifiers.get(key, meshes.len())?;
            for (mesh, identifier) in meshes.iter_mut().zip(identifiers) {
                let data = Arc::get_mut(&mut mesh.data)
                    .ok_or_else(|| ViewError("New geometry mesh unexpectedly shared".into()))?;
                data.id = identifier;
            }
            let chunk = Arc::new(PreparedChunk {
                range: (start, end),
                meshes,
            });
            let bytes = chunk.bytes();
            if self.reserve(bytes) {
                self.bytes += bytes;
                self.order.push_back(Cached::Chunk(key));
                self.chunks.insert(
                    key,
                    CacheEntry {
                        data: chunk.clone(),
                        recent: false,
                    },
                );
            }
            return Ok(chunk);
        }
    }

    fn cached_chunk(
        &mut self,
        revision: u64,
        cell: CellId,
        range: (usize, usize),
        magnification: f64,
    ) -> Option<Arc<PreparedChunk>> {
        for magnification in [0, magnification.to_bits()] {
            let key = ChunkKey {
                revision,
                cell,
                start: range.0,
                end: range.1,
                magnification,
            };
            if let Some(chunk) = self.chunks.get_mut(&key) {
                chunk.recent = true;
                return Some(chunk.data.clone());
            }
        }
        None
    }

    fn glyph(
        &mut self,
        context: &egui::Context,
        key: GlyphKey,
    ) -> Result<Arc<PreparedGlyph>, ViewError> {
        if let Some(glyph) = self.glyphs.get_mut(&key) {
            glyph.recent = true;
            return Ok(glyph.data.clone());
        }
        let glyph = Arc::new(make_glyph(context, &key)?);
        let bytes = glyph.mesh.bytes()
            + key.text.capacity() * 2
            + std::mem::size_of::<PreparedGlyph>()
            + std::mem::size_of::<usize>() * 2;
        if self.reserve(bytes) {
            self.bytes += bytes;
            self.order.push_back(Cached::Glyph(key.clone()));
            self.glyphs.insert(
                key,
                CacheEntry {
                    data: glyph.clone(),
                    recent: false,
                },
            );
        }
        Ok(glyph)
    }
}

fn chunk_range(
    shapes: &[rlayout_core::Shape],
    index: usize,
    mut width: usize,
) -> Result<((usize, usize), usize), ViewError> {
    if shapes.get(index).is_none() {
        return Err(ViewError("Missing indexed shape".into()));
    }
    loop {
        let start = index / width * width;
        let end = start.saturating_add(width).min(shapes.len());
        let count = shapes
            .get(start..end)
            .ok_or_else(|| ViewError("Invalid shape chunk".into()))?
            .iter()
            .map(|shape| match &shape.geometry {
                Geometry::Polygon(points) => points.len(),
                Geometry::Box { .. } => 4,
                Geometry::Path(path) => path.points.len().saturating_mul(8).saturating_add(64),
                Geometry::Label(_) => 0,
            })
            .sum::<usize>();
        if count <= 8192 || (width == 1 && count <= 32_768) {
            return Ok(((start, end), width));
        }
        if width == 1 {
            return Err(ViewError(
                "Geometry exceeds bounded preparation memory budget".into(),
            ));
        }
        width /= 4;
    }
}

struct Builder {
    layer: LayerKey,
    origin: Option<DoublePoint>,
    bounds: Option<Bounds>,
    vertices: Vec<PreciseVertex>,
    indices: Vec<u32>,
    shape_count: usize,
    limit: usize,
}

impl Builder {
    fn new(layer: LayerKey) -> Self {
        Self {
            layer,
            origin: None,
            bounds: None,
            vertices: Vec::new(),
            indices: Vec::new(),
            shape_count: 0,
            limit: CHUNK_SCRATCH,
        }
    }

    fn estimated_bytes(&self) -> usize {
        self.vertices.capacity() * std::mem::size_of::<PreciseVertex>()
            + self.vertices.len() * std::mem::size_of::<GpuVertex>()
            + self.indices.capacity() * std::mem::size_of::<u32>()
    }

    fn finish(mut self, meshes: &mut Vec<PreparedMesh>) -> Result<(), ViewError> {
        self.flush(meshes)
    }

    fn flush(&mut self, meshes: &mut Vec<PreparedMesh>) -> Result<(), ViewError> {
        if self.vertices.is_empty() {
            return Ok(());
        }
        let origin = self
            .origin
            .ok_or_else(|| ViewError("Missing mesh origin".into()))?;
        let bounds = self
            .bounds
            .ok_or_else(|| ViewError("Missing mesh bounds".into()))?;
        let precise = std::mem::take(&mut self.vertices);
        let mut rounding = [0.0_f64; 2];
        let mut extent = [0.0_f64; 2];
        for vertex in &precise {
            for point in [
                vertex.position,
                vertex.other,
                vertex.bounds.minimum,
                vertex.bounds.maximum,
            ] {
                for (axis, coordinate) in [point.x, point.y].into_iter().enumerate() {
                    rounding[axis] =
                        rounding[axis].max((coordinate - f64::from(coordinate as f32)).abs());
                    extent[axis] = extent[axis].max(coordinate.abs());
                }
            }
        }
        let vertices = precise
            .iter()
            .map(|vertex| GpuVertex {
                position: [vertex.position.x as f32, vertex.position.y as f32],
                other: [vertex.other.x as f32, vertex.other.y as f32],
                uv: vertex.uv,
                bounds: [
                    vertex.bounds.minimum.x as f32,
                    vertex.bounds.minimum.y as f32,
                    vertex.bounds.maximum.x as f32,
                    vertex.bounds.maximum.y as f32,
                ],
                kind: vertex.kind,
                corner: vertex.corner,
            })
            .collect();
        let data = Arc::new(MeshData {
            id: 0,
            vertices,
            indices: std::mem::take(&mut self.indices),
            glyph: None,
        });
        if data.bytes() > PACKET_LIMIT {
            return Err(ViewError(
                "Geometry packet exceeds rendering memory budget".into(),
            ));
        }
        meshes.push(PreparedMesh {
            layer: self.layer,
            origin,
            bounds,
            precise,
            data,
            shape_count: std::mem::take(&mut self.shape_count),
            rounding,
            extent,
        });
        self.bounds = None;
        self.origin = None;
        Ok(())
    }

    fn prepare_space(
        &mut self,
        vertices: usize,
        point: DoublePoint,
        bounds: Bounds,
        meshes: &mut Vec<PreparedMesh>,
    ) -> Result<(), ViewError> {
        if self.vertices.len().saturating_add(vertices) > 32_768 {
            self.flush(meshes)?;
        }
        let count = self.vertices.len().saturating_add(vertices);
        let vertex_capacity = count.next_power_of_two().max(self.vertices.capacity());
        let index_capacity = count
            .saturating_mul(8)
            .next_power_of_two()
            .max(self.indices.capacity());
        let bytes = meshes
            .iter()
            .map(|mesh| {
                mesh.data.bytes() + mesh.precise.capacity() * std::mem::size_of::<PreciseVertex>()
            })
            .sum::<usize>()
            .saturating_add(vertex_capacity * std::mem::size_of::<PreciseVertex>())
            .saturating_add(count * std::mem::size_of::<GpuVertex>())
            .saturating_add(index_capacity * std::mem::size_of::<u32>());
        if bytes > self.limit {
            return Err(ViewError(CHUNK_BUDGET_ERROR.into()));
        }
        if self.origin.is_none() {
            self.origin = Some(point);
        }
        self.bounds = Some(
            self.bounds
                .map_or(bounds, |previous| previous.union(bounds)),
        );
        Ok(())
    }

    fn vertex(
        &mut self,
        position: DoublePoint,
        other: DoublePoint,
        bounds: Bounds,
        kind: u32,
        corner: u32,
        uv: [f32; 2],
    ) -> Result<u32, ViewError> {
        let origin = self
            .origin
            .ok_or_else(|| ViewError("Missing mesh origin".into()))?;
        let index = u32::try_from(self.vertices.len())
            .map_err(|_| ViewError("Mesh vertex index overflow".into()))?;
        self.vertices.push(PreciseVertex {
            position: position.minus(origin),
            other: other.minus(origin),
            uv,
            bounds: Bounds {
                minimum: bounds.minimum.minus(origin),
                maximum: bounds.maximum.minus(origin),
            },
            kind,
            corner,
        });
        Ok(index)
    }

    fn edge(
        &mut self,
        first: DoublePoint,
        second: DoublePoint,
        bounds: Bounds,
        kind: u32,
        width: f64,
        meshes: &mut Vec<PreparedMesh>,
    ) -> Result<(), ViewError> {
        self.prepare_space(4, first, bounds, meshes)?;
        let uv = [width as f32, 0.0];
        let first_index = self.vertex(first, second, bounds, kind, 0, uv)?;
        let second_index = self.vertex(first, second, bounds, kind, 1, uv)?;
        let third_index = self.vertex(first, second, bounds, kind, 2, uv)?;
        let fourth_index = self.vertex(first, second, bounds, kind, 3, uv)?;
        self.indices.extend([
            first_index,
            second_index,
            third_index,
            first_index,
            third_index,
            fourth_index,
        ]);
        Ok(())
    }

    fn polygon(
        &mut self,
        points: Vec<DoublePoint>,
        meshes: &mut Vec<PreparedMesh>,
        cancelled: &impl Fn() -> bool,
    ) -> Result<(), ViewError> {
        if points.len() < 3 {
            return Ok(());
        }
        if let Some(bounds) = rectangle_bounds(&points) {
            self.prepare_space(4, bounds.minimum, bounds, meshes)?;
            self.shape_count = self.shape_count.saturating_add(1);
            let corners = [
                bounds.minimum,
                DoublePoint::new(bounds.maximum.x, bounds.minimum.y),
                bounds.maximum,
                DoublePoint::new(bounds.minimum.x, bounds.maximum.y),
            ];
            let mut indices = [0_u32; 4];
            for (corner, point) in corners.into_iter().enumerate() {
                indices[corner] = self.vertex(point, point, bounds, 4, corner as u32, [0.0; 2])?;
            }
            self.indices.extend([
                indices[0], indices[1], indices[2], indices[0], indices[2], indices[3],
            ]);
            return Ok(());
        }
        let contour = points.clone();
        let polygon = fast_polygon(points, cancelled)?;
        if polygon.triangles.is_empty() {
            return Ok(());
        }
        let width = intrinsic_width(&polygon, &contour)?;
        let widths = local_edge_widths(&contour, width, cancelled)?;
        let first = polygon
            .points
            .first()
            .copied()
            .ok_or_else(|| ViewError("Missing polygon origin".into()))?;
        self.prepare_space(polygon.points.len(), first, polygon.bounds, meshes)?;
        self.shape_count = self.shape_count.saturating_add(1);
        if polygon.points.len() <= 32_768 {
            let offset = u32::try_from(self.vertices.len())
                .map_err(|_| ViewError("Mesh index overflow".into()))?;
            for point in &polygon.points {
                self.vertex(*point, *point, polygon.bounds, 0, 0, [0.0; 2])?;
            }
            for index in &polygon.triangles {
                self.indices.push(
                    offset
                        .checked_add(*index)
                        .ok_or_else(|| ViewError("Mesh index overflow".into()))?,
                );
            }
        } else {
            for triangle in polygon.triangles.as_chunks::<3>().0 {
                self.prepare_space(3, first, polygon.bounds, meshes)?;
                for index in triangle {
                    let point = polygon
                        .points
                        .get(*index as usize)
                        .copied()
                        .ok_or_else(|| ViewError("Invalid polygon index".into()))?;
                    let index = self.vertex(point, point, polygon.bounds, 0, 0, [0.0; 2])?;
                    self.indices.push(index);
                }
            }
        }
        for (index, (first, second)) in contour
            .iter()
            .zip(contour.iter().cycle().skip(1))
            .take(contour.len())
            .enumerate()
        {
            self.edge(
                *first,
                *second,
                polygon.bounds,
                1,
                widths
                    .get(index)
                    .copied()
                    .ok_or_else(|| ViewError("Missing boundary edge width".into()))?,
                meshes,
            )?;
        }
        Ok(())
    }

    fn geometry(
        &mut self,
        geometry: &Geometry,
        magnification: f64,
        meshes: &mut Vec<PreparedMesh>,
        cancelled: &impl Fn() -> bool,
    ) -> Result<(), ViewError> {
        match geometry {
            Geometry::Polygon(points) => self.polygon(
                points.iter().map(|point| point.to_double()).collect(),
                meshes,
                cancelled,
            ),
            Geometry::Box { minimum, maximum } => self.polygon(
                vec![
                    minimum.to_double(),
                    DoublePoint::new(maximum.x as f64, minimum.y as f64),
                    maximum.to_double(),
                    DoublePoint::new(minimum.x as f64, maximum.y as f64),
                ],
                meshes,
                cancelled,
            ),
            Geometry::Path(path) if path.width != 0 => self.polygon(
                path_contour(path, magnification).map_err(|error| ViewError(error.to_string()))?,
                meshes,
                cancelled,
            ),
            Geometry::Path(path) => {
                let Some(bounds) =
                    Bounds::from_points(path.points.iter().map(|point| point.to_double()))
                else {
                    return Ok(());
                };
                let first = path
                    .points
                    .first()
                    .ok_or_else(|| ViewError("Missing path origin".into()))?
                    .to_double();
                self.prepare_space(4, first, bounds, meshes)?;
                self.shape_count = self.shape_count.saturating_add(1);
                for pair in path.points.windows(2) {
                    if let [first, second] = pair {
                        self.edge(
                            first.to_double(),
                            second.to_double(),
                            bounds,
                            2,
                            0.0,
                            meshes,
                        )?;
                    }
                }
                Ok(())
            }
            Geometry::Label(_) => Ok(()),
        }
    }
}

fn fast_polygon(
    points: Vec<DoublePoint>,
    cancelled: &impl Fn() -> bool,
) -> Result<Polygon, ViewError> {
    if points.iter().any(|point| !point.finite()) {
        return Err(ViewError("Non-finite polygon coordinate".into()));
    }
    let bounds = Bounds::from_points(points.iter().copied())
        .ok_or_else(|| ViewError("Empty polygon".into()))?;
    let triangle = points.len() == 3;
    let rectangle = points.len() == 4
        && bounds.width() > 0.0
        && bounds.height() > 0.0
        && points.first() != points.get(2)
        && points.get(1) != points.get(3)
        && points
            .iter()
            .zip(points.iter().cycle().skip(1))
            .take(4)
            .all(|(first, second)| {
                first != second && ((first.x == second.x) != (first.y == second.y))
            })
        && points.iter().all(|point| {
            (point.x == bounds.minimum.x || point.x == bounds.maximum.x)
                && (point.y == bounds.minimum.y || point.y == bounds.maximum.y)
        });
    if triangle || rectangle {
        let triangles = if triangle {
            let first = points
                .first()
                .copied()
                .ok_or_else(|| ViewError("Missing triangle vertex".into()))?;
            let direction = points
                .get(1)
                .copied()
                .ok_or_else(|| ViewError("Missing triangle vertex".into()))?
                .minus(first);
            let other = points
                .get(2)
                .copied()
                .ok_or_else(|| ViewError("Missing triangle vertex".into()))?
                .minus(first);
            if direction.x * other.y - direction.y * other.x == 0.0 {
                Vec::new()
            } else {
                vec![0, 1, 2]
            }
        } else {
            vec![0, 1, 2, 0, 2, 3]
        };
        Ok(Polygon {
            points,
            triangles,
            bounds,
        })
    } else {
        tessellation_preflight(&points, cancelled)?;
        triangulate(points)
    }
}

fn tessellation_preflight(
    points: &[DoublePoint],
    cancelled: &impl Fn() -> bool,
) -> Result<(), ViewError> {
    #[derive(Clone, Copy)]
    struct Segment {
        index: usize,
        bounds: Bounds,
    }

    // tess2-rust 1.2.1 stores <=80-byte vertices, 24-byte faces and 32-byte
    // half-edges, plus sweep regions, dictionary/queue entries and output.
    // A crossing splits two edges; N + 4I bounds the resulting arrangement.
    // 4096 bytes per arrangement node covers triangulation diagonals, all
    // these arenas, their geometric capacity growth and reallocation overlap.
    // Bounding-box pairs overestimate crossings so the bound stays conservative.
    let maximum_nodes = TESSELLATION_SCRATCH / TESSELLATION_BYTES_PER_NODE;
    let mut nodes = points.len().saturating_add(32);
    let budget_error = || ViewError("Polygon exceeds bounded tessellation scratch budget".into());
    if nodes > maximum_nodes {
        return Err(budget_error());
    }
    if cancelled() {
        return Err(ViewError("Rendering cancelled".into()));
    }
    let mut segments = points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .take(points.len())
        .enumerate()
        .map(|(index, (first, second))| Segment {
            index,
            bounds: Bounds {
                minimum: DoublePoint::new(first.x.min(second.x), first.y.min(second.y)),
                maximum: DoublePoint::new(first.x.max(second.x), first.y.max(second.y)),
            },
        })
        .collect::<Vec<_>>();
    segments.sort_unstable_by(|first, second| {
        first.bounds.minimum.x.total_cmp(&second.bounds.minimum.x)
    });
    let mut active = Vec::<Segment>::new();
    let mut comparisons = 0_usize;
    for segment in segments {
        let mut position = 0;
        while let Some(previous) = active.get(position).copied() {
            comparisons += 1;
            if comparisons > TESSELLATION_COMPARISONS {
                return Err(ViewError(
                    "Polygon exceeds bounded tessellation preflight work budget".into(),
                ));
            }
            if comparisons.is_multiple_of(1024) && cancelled() {
                return Err(ViewError("Rendering cancelled".into()));
            }
            if previous.bounds.maximum.x < segment.bounds.minimum.x {
                active.swap_remove(position);
                continue;
            }
            let separation = previous.index.abs_diff(segment.index);
            let adjacent = separation == 1 || separation == points.len().saturating_sub(1);
            if !adjacent
                && previous.bounds.maximum.y >= segment.bounds.minimum.y
                && segment.bounds.maximum.y >= previous.bounds.minimum.y
            {
                nodes += 4;
                if nodes > maximum_nodes {
                    return Err(budget_error());
                }
            }
            position += 1;
        }
        active.push(segment);
    }
    Ok(())
}

fn rectangle_bounds(points: &[DoublePoint]) -> Option<Bounds> {
    if points.len() != 4 || points.iter().any(|point| !point.finite()) {
        return None;
    }
    let bounds = Bounds::from_points(points.iter().copied())?;
    if bounds.width() <= 0.0
        || bounds.height() <= 0.0
        || points.first() == points.get(2)
        || points.get(1) == points.get(3)
    {
        return None;
    }
    let edges = points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .take(4)
        .all(|(first, second)| first != second && ((first.x == second.x) != (first.y == second.y)));
    let corners = points.iter().all(|point| {
        (point.x == bounds.minimum.x || point.x == bounds.maximum.x)
            && (point.y == bounds.minimum.y || point.y == bounds.maximum.y)
    });
    (edges && corners).then_some(bounds)
}

fn intrinsic_width(polygon: &Polygon, contour: &[DoublePoint]) -> Result<f64, ViewError> {
    let mut area = 0.0_f64;
    for triangle in polygon.triangles.as_chunks::<3>().0 {
        let first = polygon
            .points
            .get(triangle[0] as usize)
            .copied()
            .ok_or_else(|| ViewError("Invalid polygon width index".into()))?;
        let second = polygon
            .points
            .get(triangle[1] as usize)
            .copied()
            .ok_or_else(|| ViewError("Invalid polygon width index".into()))?
            .minus(first);
        let third = polygon
            .points
            .get(triangle[2] as usize)
            .copied()
            .ok_or_else(|| ViewError("Invalid polygon width index".into()))?
            .minus(first);
        area += (second.x * third.y - second.y * third.x).abs() * 0.5;
    }
    let edge = contour
        .iter()
        .zip(contour.iter().cycle().skip(1))
        .take(contour.len())
        .map(|(first, second)| {
            let direction = second.minus(*first);
            direction.x.hypot(direction.y)
        })
        .fold(0.0_f64, f64::max);
    let width = if edge == 0.0 { 0.0 } else { area / edge };
    if !width.is_finite() {
        return Err(ViewError("Polygon width overflow".into()));
    }
    Ok(width)
}

fn local_edge_widths(
    contour: &[DoublePoint],
    global: f64,
    cancelled: &impl Fn() -> bool,
) -> Result<Vec<f64>, ViewError> {
    struct ParallelEdge {
        index: usize,
        minimum: f64,
        maximum: f64,
        line: f64,
        positive: bool,
    }

    let mut widths = vec![global; contour.len()];
    if contour.len() < 4 {
        return Ok(widths);
    }
    let origin = contour
        .first()
        .copied()
        .ok_or_else(|| ViewError("Missing edge origin".into()))?;
    let mut groups = BTreeMap::<(bool, u64), Vec<ParallelEdge>>::new();
    for (index, (first, second)) in contour
        .iter()
        .zip(contour.iter().cycle().skip(1))
        .take(contour.len())
        .enumerate()
    {
        let first = first.minus(origin);
        let second = second.minus(origin);
        let direction = second.minus(first);
        if direction.x == 0.0 && direction.y == 0.0 {
            continue;
        }
        let horizontal = direction.x.abs() >= direction.y.abs();
        let (start, end, transverse, slope) = if horizontal {
            (first.x, second.x, first.y, direction.y / direction.x)
        } else {
            (first.y, second.y, first.x, direction.x / direction.y)
        };
        let slope = if slope == 0.0 { 0.0 } else { slope };
        groups
            .entry((horizontal, slope.to_bits()))
            .or_default()
            .push(ParallelEdge {
                index,
                minimum: start.min(end),
                maximum: start.max(end),
                line: transverse - slope * start,
                positive: end > start,
            });
    }
    for ((_, slope), edges) in groups {
        if cancelled() {
            return Err(ViewError("Rendering cancelled".into()));
        }
        if !edges.iter().any(|edge| edge.positive) || !edges.iter().any(|edge| !edge.positive) {
            continue;
        }
        let mut coordinates = edges
            .iter()
            .flat_map(|edge| [edge.minimum, edge.maximum])
            .collect::<Vec<_>>();
        coordinates.sort_unstable_by(f64::total_cmp);
        coordinates.dedup();
        let ranges = edges
            .iter()
            .map(|edge| {
                let minimum = coordinates
                    .binary_search_by(|point| point.total_cmp(&edge.minimum))
                    .map_err(|_| ViewError("Missing edge interval coordinate".into()))?;
                let maximum = coordinates
                    .binary_search_by(|point| point.total_cmp(&edge.maximum))
                    .map_err(|_| ViewError("Missing edge interval coordinate".into()))?;
                Ok((minimum, maximum))
            })
            .collect::<Result<Vec<_>, ViewError>>()?;
        let mut ordering = (0..edges.len()).collect::<Vec<_>>();
        ordering
            .sort_unstable_by(|first, second| edges[*first].line.total_cmp(&edges[*second].line));
        let distance_scale = (1.0 + f64::from_bits(slope).powi(2)).sqrt().recip();
        // Sweeping perpendicular positions in both directions finds the nearest
        // opposing line whose interval overlaps, without an O(N²) edge scan.
        for reverse in [false, true] {
            if reverse {
                ordering.reverse();
            }
            let mut trees = [
                EdgeIntervals::new(coordinates.len()),
                EdgeIntervals::new(coordinates.len()),
            ];
            for (position, index) in ordering.iter().copied().enumerate() {
                let edge = &edges[index];
                let range = ranges[index];
                let opposite = usize::from(!edge.positive);
                let previous = trees[opposite].maximum(range.0, range.1);
                if previous != 0 {
                    let other = ordering
                        .get(previous - 1)
                        .and_then(|index| edges.get(*index))
                        .ok_or_else(|| ViewError("Invalid parallel edge interval".into()))?;
                    widths[edge.index] =
                        widths[edge.index].min((edge.line - other.line).abs() * distance_scale);
                }
                trees[usize::from(edge.positive)].assign(range.0, range.1, position + 1);
            }
        }
    }
    Ok(widths)
}

struct EdgeIntervals {
    values: Vec<usize>,
    assigned: Vec<usize>,
    length: usize,
}

impl EdgeIntervals {
    fn new(length: usize) -> Self {
        Self {
            values: vec![0; length * 4],
            assigned: vec![0; length * 4],
            length,
        }
    }

    fn push(&mut self, node: usize) {
        if self.assigned[node] != 0 {
            for child in [node * 2, node * 2 + 1] {
                self.values[child] = self.assigned[node];
                self.assigned[child] = self.assigned[node];
            }
            self.assigned[node] = 0;
        }
    }

    fn assign(&mut self, minimum: usize, maximum: usize, value: usize) {
        self.assign_range(1, (0, self.length - 1), (minimum, maximum), value);
    }

    fn assign_range(
        &mut self,
        node: usize,
        bounds: (usize, usize),
        range: (usize, usize),
        value: usize,
    ) {
        if range.0 <= bounds.0 && bounds.1 <= range.1 {
            self.values[node] = value;
            self.assigned[node] = value;
            return;
        }
        self.push(node);
        let middle = (bounds.0 + bounds.1) / 2;
        if range.0 <= middle {
            self.assign_range(node * 2, (bounds.0, middle), range, value);
        }
        if range.1 > middle {
            self.assign_range(node * 2 + 1, (middle + 1, bounds.1), range, value);
        }
        self.values[node] = self.values[node * 2].max(self.values[node * 2 + 1]);
    }

    fn maximum(&mut self, minimum: usize, maximum: usize) -> usize {
        self.maximum_range(1, (0, self.length - 1), (minimum, maximum))
    }

    fn maximum_range(
        &mut self,
        node: usize,
        bounds: (usize, usize),
        range: (usize, usize),
    ) -> usize {
        if range.0 <= bounds.0 && bounds.1 <= range.1 {
            return self.values[node];
        }
        self.push(node);
        let middle = (bounds.0 + bounds.1) / 2;
        let left = if range.0 <= middle {
            self.maximum_range(node * 2, (bounds.0, middle), range)
        } else {
            0
        };
        let right = if range.1 > middle {
            self.maximum_range(node * 2 + 1, (middle + 1, bounds.1), range)
        } else {
            0
        };
        left.max(right)
    }
}

fn text_preflight(text: &str) -> Result<usize, ViewError> {
    // egui emits one Glyph per character and four mesh vertices/six indices.
    // 512 bytes per UTF-8 byte also covers row/paragraph bookkeeping, cloned
    // input and Vec capacity growth before fonts.layout_no_wrap allocates it.
    text.len()
        .checked_mul(TEXT_BYTES_PER_CHARACTER)
        .filter(|bytes| *bytes <= TEXT_SCRATCH)
        .ok_or_else(|| ViewError("Text exceeds bounded layout scratch budget".into()))
}

fn make_glyph(context: &egui::Context, key: &GlyphKey) -> Result<PreparedGlyph, ViewError> {
    let text_bytes = text_preflight(&key.text)?;
    let (galley, atlas) = context.fonts_mut(|fonts| {
        let galley = fonts.layout_no_wrap(
            key.text.clone(),
            egui::FontId::proportional(f64::from_bits(key.size) as f32),
            Color32::WHITE,
        );
        let atlas_size = fonts.font_image_size();
        let atlas_bytes = atlas_size[0]
            .checked_mul(atlas_size[1])
            .and_then(|pixels| pixels.checked_mul(std::mem::size_of::<Color32>()))
            .ok_or_else(|| ViewError("Font atlas size overflow".into()))?;
        if text_bytes.saturating_add(atlas_bytes) > TEXT_SCRATCH {
            return Err(ViewError(
                "Text exceeds bounded layout scratch budget".into(),
            ));
        }
        Ok((galley, fonts.image()))
    })?;
    let pixels_per_point = f64::from(f32::from_bits(key.pixels_per_point));
    let size = [
        (f64::from(galley.size().x) * pixels_per_point)
            .ceil()
            .max(1.0) as usize,
        (f64::from(galley.size().y) * pixels_per_point)
            .ceil()
            .max(1.0) as usize,
    ];
    let length = size[0]
        .checked_mul(size[1])
        .filter(|length| *length <= PACKET_LIMIT / 2)
        .ok_or_else(|| ViewError("Text exceeds rendering memory budget".into()))?;
    let mut alpha = vec![0_u8; length];
    for row in &galley.rows {
        for triangle in row.visuals.mesh.indices.as_chunks::<3>().0 {
            let mut positions = [DoublePoint::default(); 3];
            let mut coordinates = [DoublePoint::default(); 3];
            for (offset, index) in triangle.iter().enumerate() {
                let vertex = row
                    .visuals
                    .mesh
                    .vertices
                    .get(*index as usize)
                    .ok_or_else(|| ViewError("Invalid glyph triangle".into()))?;
                let position = vertex.pos.to_vec2() + row.pos.to_vec2();
                positions[offset] = DoublePoint::new(f64::from(position.x), f64::from(position.y))
                    .times(pixels_per_point);
                coordinates[offset] =
                    DoublePoint::new(f64::from(vertex.uv.x), f64::from(vertex.uv.y));
            }
            let Some(bounds) = Bounds::from_points(positions) else {
                continue;
            };
            let edge = positions[1].minus(positions[0]);
            let other = positions[2].minus(positions[0]);
            let determinant = edge.x * other.y - edge.y * other.x;
            if determinant == 0.0 {
                continue;
            }
            let left = bounds.minimum.x.floor().clamp(0.0, size[0] as f64) as usize;
            let right = bounds.maximum.x.ceil().clamp(0.0, size[0] as f64) as usize;
            let top = bounds.minimum.y.floor().clamp(0.0, size[1] as f64) as usize;
            let bottom = bounds.maximum.y.ceil().clamp(0.0, size[1] as f64) as usize;
            for y in top..bottom {
                for x in left..right {
                    let relative =
                        DoublePoint::new(x as f64 + 0.5, y as f64 + 0.5).minus(positions[0]);
                    let first = (relative.x * other.y - relative.y * other.x) / determinant;
                    let second = (edge.x * relative.y - edge.y * relative.x) / determinant;
                    if first >= -1e-8 && second >= -1e-8 && first + second <= 1.0 + 1e-8 {
                        let coordinate = coordinates[0]
                            .plus(coordinates[1].minus(coordinates[0]).times(first))
                            .plus(coordinates[2].minus(coordinates[0]).times(second));
                        let column = coordinate
                            .x
                            .floor()
                            .clamp(0.0, atlas.size[0].saturating_sub(1) as f64)
                            as usize;
                        let row = coordinate
                            .y
                            .floor()
                            .clamp(0.0, atlas.size[1].saturating_sub(1) as f64)
                            as usize;
                        if let Some(pixel) = atlas.pixels.get(row * atlas.size[0] + column) {
                            alpha[y * size[0] + x] = alpha[y * size[0] + x].max(pixel.a());
                        }
                    }
                }
            }
        }
    }
    let bitmap = Arc::new(GlyphBitmap {
        id: resource_id()?,
        size: [
            u32::try_from(size[0]).map_err(|_| ViewError("Glyph width overflow".into()))?,
            u32::try_from(size[1]).map_err(|_| ViewError("Glyph height overflow".into()))?,
        ],
        alpha,
    });
    let logical_size = galley.size();
    let bounds = [0.0, 0.0, logical_size.x, logical_size.y];
    let maximum_uv = [
        (f64::from(logical_size.x) * pixels_per_point / size[0] as f64) as f32,
        (f64::from(logical_size.y) * pixels_per_point / size[1] as f64) as f32,
    ];
    let vertices = [
        ([0.0, 0.0], [0.0, 0.0]),
        ([logical_size.x, 0.0], [maximum_uv[0], 0.0]),
        ([logical_size.x, logical_size.y], maximum_uv),
        ([0.0, logical_size.y], [0.0, maximum_uv[1]]),
    ]
    .into_iter()
    .map(|(position, uv)| GpuVertex {
        position,
        other: position,
        uv,
        bounds,
        kind: 3,
        corner: 0,
    })
    .collect();
    Ok(PreparedGlyph {
        logical_size,
        mesh: Arc::new(MeshData {
            id: resource_id()?,
            vertices,
            indices: vec![0, 1, 2, 0, 2, 3],
            glyph: Some(bitmap),
        }),
    })
}

#[derive(Clone, Copy)]
struct Lattice {
    column: DoublePoint,
    row: DoublePoint,
    columns: u32,
    rows: u32,
}

struct CellCursor {
    cell: CellId,
    transformation: Transformation,
    depth: u16,
    multiplicity: usize,
    lattice: Option<Lattice>,
    query: super::spatial::Query,
    target: Bounds,
    chunks: RenderSet<(usize, usize)>,
}

struct ArrayCursor {
    cell: CellId,
    base: Transformation,
    depth: u16,
    multiplicity: usize,
    column_step: DoublePoint,
    row_step: DoublePoint,
    columns: (u16, u16),
    rows: (u16, u16),
    column: u16,
    row: u16,
}

struct LeafCursor {
    cell: CellId,
    transformation: Transformation,
    multiplicity: usize,
    lattice: Option<Lattice>,
    index: usize,
}

enum Work {
    Cell(CellId, Transformation, u16, usize, Option<Lattice>),
    Entries(CellCursor),
    Leaf(LeafCursor),
    Array(ArrayCursor),
}

struct PendingDraw {
    layer: LayerKey,
    mesh: Arc<MeshData>,
    instances: Vec<Instance>,
    shape_count: usize,
    array: Option<ArrayInstances>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ArrayKey([u32; 6]);

impl From<ArrayInstances> for ArrayKey {
    fn from(array: ArrayInstances) -> Self {
        Self([
            array.columns,
            array.rows,
            array.column_step[0].to_bits(),
            array.column_step[1].to_bits(),
            array.row_step[0].to_bits(),
            array.row_step[1].to_bits(),
        ])
    }
}

type BatchKey = (LayerKey, u64, Option<ArrayKey>);

struct Planner<'a> {
    job: &'a Job,
    shared: &'a Shared,
    sender: &'a mut PacketSender,
    cache: &'a mut Cache,
    tile: TileInfo,
    origin: DoublePoint,
    scale: f64,
    visible: Bounds,
    layers: RenderSet<LayerKey>,
    leaf_visibility: &'a mut [Option<bool>],
    batches: RenderMap<BatchKey, PendingDraw>,
    batch_bytes: usize,
    batch_limit: usize,
    reuse: Arc<AtomicBool>,
    started: std::time::Instant,
}

#[derive(Clone, Copy)]
struct Affine {
    matrix: [f64; 4],
    translation: DoublePoint,
}

#[derive(Clone, Copy)]
struct Projection {
    camera_center: DoublePoint,
    transformation: Transformation,
    density: f64,
    screen_origin: DoublePoint,
}

#[derive(Clone, Copy)]
struct Compensated {
    high: f64,
    low: f64,
}

impl Compensated {
    fn sum(first: f64, second: f64) -> Self {
        let high = first + second;
        let virtual_second = high - first;
        Self {
            high,
            low: (first - (high - virtual_second)) + (second - virtual_second),
        }
    }

    fn product(first: f64, second: f64) -> Self {
        let high = first * second;
        Self {
            high,
            low: first.mul_add(second, -high),
        }
    }

    fn add(self, other: Self) -> Self {
        let sum = Self::sum(self.high, other.high);
        Self::sum(sum.high, sum.low + self.low + other.low)
    }

    fn multiply(self, scalar: f64) -> Self {
        let product = Self::product(self.high, scalar);
        Self::sum(product.high, product.low + self.low * scalar)
    }

    fn value(self) -> f64 {
        self.high + self.low
    }
}

impl Projection {
    fn new(
        transformation: Transformation,
        camera: Camera,
        viewport: Rect,
        scale: f64,
        tile_origin: DoublePoint,
    ) -> Self {
        Self {
            camera_center: camera.center,
            transformation,
            density: camera.pixels_per_unit * scale,
            screen_origin: DoublePoint::new(
                f64::from(viewport.width()) * scale * 0.5 - tile_origin.x,
                f64::from(viewport.height()) * scale * 0.5 - tile_origin.y,
            ),
        }
    }

    fn apply(self, source_origin: DoublePoint, local: DoublePoint) -> DoublePoint {
        let (sine, cosine) = self.transformation.angle.sin_cos();
        let reflection = if self.transformation.reflected {
            -1.0
        } else {
            1.0
        };
        let coordinate = |first: f64, second: f64, translation: f64, camera: f64| {
            let rotated = |point: DoublePoint| {
                Compensated::product(first, point.x)
                    .add(Compensated::product(second, point.y))
                    .multiply(self.transformation.magnification)
            };
            // Keep the camera difference and local geometry separate until the
            // large transform terms cancel; otherwise a tiny cell translated
            // far from zero loses visible detail even before pixel conversion.
            Compensated::sum(translation, -camera)
                .add(rotated(source_origin))
                .add(rotated(local))
        };
        let horizontal = coordinate(
            cosine,
            -sine * reflection,
            self.transformation.translation.x,
            self.camera_center.x,
        );
        let vertical = coordinate(
            sine,
            cosine * reflection,
            self.transformation.translation.y,
            self.camera_center.y,
        );
        DoublePoint::new(
            horizontal
                .multiply(self.density)
                .add(Compensated::sum(self.screen_origin.x, 0.0))
                .value(),
            vertical
                .multiply(-self.density)
                .add(Compensated::sum(self.screen_origin.y, 0.0))
                .value(),
        )
    }

    fn rectangle_corner(self, source_origin: DoublePoint, vertex: PreciseVertex) -> DoublePoint {
        let position = self.apply(source_origin, vertex.position);
        let axis = |vector: DoublePoint| {
            let vector = self.transformation.apply_vector(vector);
            DoublePoint::new(vector.x, -vector.y).times(self.density)
        };
        let expansion = |vector: DoublePoint| {
            let length = vector.x.hypot(vector.y);
            if length > 0.0 && length < 1.0 {
                vector.times((1.0 - length) / length * 0.5)
            } else {
                DoublePoint::default()
            }
        };
        let horizontal = expansion(axis(DoublePoint::new(vertex.bounds.width(), 0.0)));
        let vertical = expansion(axis(DoublePoint::new(0.0, vertex.bounds.height())));
        let horizontal_sign = if vertex.corner == 1 || vertex.corner == 2 {
            1.0
        } else {
            -1.0
        };
        let vertical_sign = if vertex.corner >= 2 { 1.0 } else { -1.0 };
        position
            .plus(horizontal.times(horizontal_sign))
            .plus(vertical.times(vertical_sign))
    }
}

impl Affine {
    fn apply(self, point: DoublePoint) -> DoublePoint {
        DoublePoint::new(
            self.matrix[0] * point.x + self.matrix[1] * point.y,
            self.matrix[2] * point.x + self.matrix[3] * point.y,
        )
        .plus(self.translation)
    }

    fn instance(self) -> Instance {
        Instance {
            affine: self.matrix.map(|coordinate| coordinate as f32),
            translation: [self.translation.x as f32, self.translation.y as f32],
        }
    }

    fn precise(self, mesh: &PreparedMesh) -> bool {
        self.conversion_error(mesh.rounding, mesh.extent) <= PIXEL_ERROR
    }

    fn conversion_error(self, rounding: [f64; 2], extent: [f64; 2]) -> f64 {
        let matrix = self.matrix.map(|value| f64::from(value as f32));
        let translated = [self.translation.x, self.translation.y];
        let mut maximum = 0.0_f64;
        for (axis, translation) in translated.into_iter().enumerate() {
            let offset = axis * 2;
            let first = self.matrix[offset];
            let second = self.matrix[offset + 1];
            let error = (translation - f64::from(translation as f32)).abs()
                + (first - matrix[offset]).abs() * extent[0]
                + (second - matrix[offset + 1]).abs() * extent[1]
                + first.abs() * rounding[0]
                + second.abs() * rounding[1]
                + (translation.abs() + first.abs() * extent[0] + second.abs() * extent[1])
                    * f64::from(f32::EPSILON)
                    * 4.0;
            if !error.is_finite() {
                return f64::INFINITY;
            }
            maximum = maximum.max(error);
        }
        maximum
    }

    fn array_precise(
        self,
        rounding: [f64; 2],
        extent: [f64; 2],
        lattice: Lattice,
        array: ArrayInstances,
        scale: f64,
    ) -> bool {
        self.conversion_error(rounding, extent) + lattice_error(lattice, array, scale)
            <= PIXEL_ERROR
    }
}

fn render_job(
    job: &Job,
    shared: &Shared,
    sender: &mut PacketSender,
    cache: &mut Cache,
    started: std::time::Instant,
) -> Result<(), ViewError> {
    let physical_width = f64::from(job.key.viewport.width()) * f64::from(job.key.pixels_per_point);
    let physical_height =
        f64::from(job.key.viewport.height()) * f64::from(job.key.pixels_per_point);
    if !physical_width.is_finite()
        || !physical_height.is_finite()
        || physical_width <= 0.0
        || physical_height <= 0.0
    {
        return Err(ViewError("Invalid rendering viewport".into()));
    }
    if physical_width * physical_height > (CACHE_LIMIT / 8) as f64 {
        return Err(ViewError("Viewport exceeds rendering memory budget".into()));
    }
    let divisor = if job.preview { 4.0 } else { 1.0 };
    let scale = f64::from(job.key.pixels_per_point) / divisor;
    let width = (physical_width / divisor).ceil() as usize;
    let height = (physical_height / divisor).ceil() as usize;
    let density = job.key.camera.pixels_per_unit * scale;
    if !density.is_finite() || density <= 0.0 {
        return Err(ViewError("Invalid camera scale".into()));
    }
    let anchor = job
        .key
        .selected
        .and_then(|cell| job.scene.layout.bounds(cell))
        .map_or(job.key.camera.center, Bounds::center);
    let horizontal = GridAxis::new(
        job.key.camera.center.x - anchor.x,
        density,
        f64::from(job.key.viewport.width()) * scale * 0.5,
    )?;
    let vertical = GridAxis::new(
        anchor.y - job.key.camera.center.y,
        density,
        f64::from(job.key.viewport.height()) * scale * 0.5,
    )?;
    let phase = [horizontal.phase, vertical.phase];
    let grid_x = horizontal.index;
    let grid_y = vertical.index;
    let mut tiles = Vec::new();
    let last_x = horizontal.last(width)?;
    let last_y = vertical.last(height)?;
    for row in grid_y..=last_y {
        for column in grid_x..=last_x {
            let left = horizontal.screen_origin(column).round();
            let top = vertical.screen_origin(row).round();
            tiles.push((column, row, left, top));
        }
    }
    tiles.sort_unstable_by(|first, second| {
        let distance = |tile: &(i64, i64, f64, f64)| {
            let horizontal = tile.2 + TILE_SIZE as f64 * 0.5 - width as f64 * 0.5;
            let vertical = tile.3 + TILE_SIZE as f64 * 0.5 - height as f64 * 0.5;
            horizontal * horizontal + vertical * vertical
        };
        distance(first).total_cmp(&distance(second))
    });
    let layers: Vec<_> = job
        .key
        .styles
        .iter()
        .filter_map(|(layer, style)| style.visible.then_some((*layer, *style)))
        .collect();
    let group_count = layers.len().div_ceil(64).max(1);
    let cell_count = if layers.is_empty() {
        0
    } else {
        job.scene.layout.cells().count()
    };
    let visibility_count = cell_count
        .checked_mul(group_count)
        .filter(|count| {
            count.saturating_mul(std::mem::size_of::<Option<bool>>()) <= PACKET_LIMIT / 2
        })
        .ok_or_else(|| {
            ViewError("Layer visibility exceeds bounded preparation scratch budget".into())
        })?;
    let mut leaf_visibility = vec![None; visibility_count];
    let visibility_bytes = leaf_visibility.capacity() * std::mem::size_of::<Option<bool>>();
    let tile_count = tiles.len();
    for (tile_index, (column, row, left, top)) in tiles.into_iter().enumerate() {
        if shared.generation.load(Ordering::Acquire) != job.generation {
            return Ok(());
        }
        let rect = Rect::from_min_max(
            job.key.viewport.min + Vec2::new((left / scale) as f32, (top / scale) as f32),
            job.key.viewport.min
                + Vec2::new(
                    ((left + TILE_SIZE as f64) / scale) as f32,
                    ((top + TILE_SIZE as f64) / scale) as f32,
                ),
        );
        let tile = TileInfo {
            id: resource_id()?,
            rect,
            size: [TILE_SIZE as u32; 2],
            preview: job.preview,
            grid: [column, row],
            phase,
            density,
            crop: Rect::from_min_max(
                Pos2::new(
                    (-left).clamp(0.0, TILE_SIZE as f64).floor() as f32,
                    (-top).clamp(0.0, TILE_SIZE as f64).floor() as f32,
                ),
                Pos2::new(
                    (width as f64 - left).clamp(0.0, TILE_SIZE as f64).ceil() as f32,
                    (height as f64 - top).clamp(0.0, TILE_SIZE as f64).ceil() as f32,
                ),
            ),
        };
        let origin = DoublePoint::new(left, top);
        let units_per_pixel = 1.0 / (job.key.camera.pixels_per_unit * scale);
        let center = DoublePoint::new(
            f64::from(job.key.viewport.width()) * scale * 0.5,
            f64::from(job.key.viewport.height()) * scale * 0.5,
        );
        let visible = Bounds {
            minimum: job.key.camera.center.plus(
                DoublePoint::new(
                    origin.x + f64::from(tile.crop.left()) - center.x,
                    center.y - origin.y - f64::from(tile.crop.bottom()),
                )
                .times(units_per_pixel),
            ),
            maximum: job.key.camera.center.plus(
                DoublePoint::new(
                    origin.x + f64::from(tile.crop.right()) - center.x,
                    center.y - origin.y - f64::from(tile.crop.top()),
                )
                .times(units_per_pixel),
            ),
        }
        .expanded(units_per_pixel * 2.0);
        let reuse = Arc::new(AtomicBool::new(false));
        for group_index in 0..group_count {
            let group = layers
                .get(group_index * 64..((group_index + 1) * 64).min(layers.len()))
                .ok_or_else(|| ViewError("Invalid layer group".into()))?;
            if !deliver(
                shared,
                sender,
                job.generation,
                Ok(Packet::Begin {
                    generation: job.generation,
                    tile,
                    layers: group.to_vec(),
                    first_group: group_index == 0,
                    reuse: reuse.clone(),
                }),
            ) {
                return Ok(());
            }
            job.context.request_repaint();
            let mut planner = Planner {
                job,
                shared,
                sender,
                cache,
                tile,
                origin,
                scale,
                visible,
                layers: group.iter().map(|(layer, _)| *layer).collect(),
                leaf_visibility: leaf_visibility
                    .get_mut(group_index * cell_count..(group_index + 1) * cell_count)
                    .ok_or_else(|| ViewError("Invalid cell visibility group".into()))?,
                batches: RenderMap::default(),
                batch_bytes: 0,
                batch_limit: PACKET_LIMIT - visibility_bytes,
                reuse: reuse.clone(),
                started,
            };
            if let Err(error) = planner.paint()
                && !planner.finished()
            {
                return Err(error);
            }
            planner.flush()?;
            if !deliver(
                shared,
                sender,
                job.generation,
                Ok(Packet::Finish {
                    generation: job.generation,
                    tile_id: tile.id,
                    last_group: group_index + 1 == group_count,
                    complete: tile_index + 1 == tile_count && group_index + 1 == group_count,
                }),
            ) {
                return Ok(());
            }
            job.context.request_repaint();
        }
    }
    Ok(())
}

fn grid_phase(coordinate: f64) -> f64 {
    let phase = (coordinate.rem_euclid(1.0) * 1e9).round() / 1e9;
    if phase == 1.0 { 0.0 } else { phase }
}

#[derive(Clone, Copy)]
struct GridAxis {
    index: i64,
    offset: f64,
    phase: f64,
}

impl GridAxis {
    fn new(displacement: f64, density: f64, half_viewport: f64) -> Result<Self, ViewError> {
        let product = displacement * density;
        if !product.is_finite() {
            return Err(ViewError("Camera grid exceeds rendering precision".into()));
        }
        let product_error = displacement.mul_add(density, -product);
        let high = product - half_viewport;
        let low = product_error + (product - high - half_viewport);
        let index = grid_index(high)?;
        let offset = high.rem_euclid(TILE_SIZE as f64) + low;
        let correction = (offset / TILE_SIZE as f64).floor();
        if !correction.is_finite() || correction <= i64::MIN as f64 || correction >= i64::MAX as f64
        {
            return Err(ViewError("Camera grid exceeds rendering precision".into()));
        }
        let index = index
            .checked_add(correction as i64)
            .ok_or_else(|| ViewError("Camera tile index overflow".into()))?;
        let offset = offset.rem_euclid(TILE_SIZE as f64);
        Ok(Self {
            index,
            offset,
            phase: grid_phase(offset),
        })
    }

    fn last(self, pixels: usize) -> Result<i64, ViewError> {
        let offset = (self.offset - self.phase).round();
        let count = ((offset + pixels as f64) / TILE_SIZE as f64).ceil() as i64;
        self.index
            .checked_add(count.saturating_sub(1))
            .ok_or_else(|| ViewError("Camera tile range overflow".into()))
    }

    fn screen_origin(self, index: i64) -> f64 {
        (index - self.index) as f64 * TILE_SIZE as f64 + self.phase - self.offset
    }
}

fn grid_index(coordinate: f64) -> Result<i64, ViewError> {
    let index = (coordinate / TILE_SIZE as f64).floor();
    if !index.is_finite() || index <= i64::MIN as f64 || index >= i64::MAX as f64 {
        return Err(ViewError("Camera grid exceeds rendering precision".into()));
    }
    Ok(index as i64)
}

impl Planner<'_> {
    fn cancelled(&self) -> bool {
        self.shared.generation.load(Ordering::Acquire) != self.job.generation
            || self.shared.stopped.load(Ordering::Acquire)
    }
    fn finished(&self) -> bool {
        self.cancelled() || self.reuse.load(Ordering::Acquire)
    }

    fn affine(&self, transformation: Transformation, origin: DoublePoint) -> Affine {
        let (sine, cosine) = transformation.angle.sin_cos();
        let reflection = if transformation.reflected { -1.0 } else { 1.0 };
        let scale = transformation.magnification * self.job.key.camera.pixels_per_unit * self.scale;
        let translation = Projection::new(
            transformation,
            self.job.key.camera,
            self.job.key.viewport,
            self.scale,
            self.origin,
        )
        .apply(origin, DoublePoint::default());
        Affine {
            matrix: [
                cosine * scale,
                -sine * reflection * scale,
                -sine * scale,
                -cosine * reflection * scale,
            ],
            translation,
        }
    }

    fn paint(&mut self) -> Result<(), ViewError> {
        if self.layers.is_empty() {
            return Ok(());
        }
        let job = self.job;
        let Some(selected) = self.job.key.selected else {
            return Ok(());
        };
        let mut stack = vec![Work::Cell(selected, Transformation::default(), 0, 1, None)];
        while let Some(work) = stack.pop() {
            if self.finished() {
                return Ok(());
            }
            match work {
                Work::Cell(cell, transformation, depth, multiplicity, lattice) => {
                    if !transformation.valid() {
                        return Err(ViewError("Hierarchy transformation overflow".into()));
                    }
                    let scene = &job.scene;
                    let source = scene
                        .layout
                        .cell(cell)
                        .ok_or_else(|| ViewError("Missing cell".into()))?;
                    if source.instances().is_empty() {
                        let visible =
                            match self.leaf_visibility.get(cell.index()).copied().flatten() {
                                Some(visible) => visible,
                                None => {
                                    let visible = source
                                        .shapes()
                                        .iter()
                                        .any(|shape| self.layers.contains(&shape.layer));
                                    let slot =
                                        self.leaf_visibility.get_mut(cell.index()).ok_or_else(
                                            || ViewError("Missing leaf visibility".into()),
                                        )?;
                                    *slot = Some(visible);
                                    visible
                                }
                            };
                        if !visible {
                            continue;
                        }
                    }
                    let Some(bounds) = scene.layout.transformed_bounds(cell, transformation) else {
                        continue;
                    };
                    let bounds = lattice.map_or(bounds, |lattice| {
                        array_bounds(
                            bounds,
                            lattice.column,
                            lattice.row,
                            lattice.columns,
                            lattice.rows,
                        )
                    });
                    let metrics = scene
                        .label_metrics
                        .get(cell.index())
                        .copied()
                        .ok_or_else(|| ViewError("Missing label metrics".into()))?;
                    let margin = text_margin(
                        metrics,
                        transformation.magnification,
                        1.0 / self.job.key.camera.pixels_per_unit,
                    );
                    if !bounds.expanded(margin).intersects(self.visible) {
                        continue;
                    }
                    let whole = bounds.expanded(
                        margin + 2.0 / (self.job.key.camera.pixels_per_unit * self.scale),
                    );
                    if source.instances().is_empty()
                        && whole.minimum.x >= self.visible.minimum.x
                        && whole.minimum.y >= self.visible.minimum.y
                        && whole.maximum.x <= self.visible.maximum.x
                        && whole.maximum.y <= self.visible.maximum.y
                    {
                        #[cfg(test)]
                        self.shared
                            .complete_leaf_cells
                            .fetch_add(1, Ordering::Relaxed);
                        stack.push(Work::Leaf(LeafCursor {
                            cell,
                            transformation,
                            multiplicity,
                            lattice,
                            index: 0,
                        }));
                        continue;
                    }
                    let index = scene
                        .indexes
                        .get(&cell)
                        .ok_or_else(|| ViewError("Missing spatial index".into()))?;
                    let target = if lattice.is_some() {
                        scene
                            .layout
                            .bounds(cell)
                            .ok_or_else(|| ViewError("Missing array cell bounds".into()))?
                            .expanded(margin)
                    } else {
                        self.visible.transformed(inverse(transformation)).expanded(
                            scene.layout.absolute_path_padding(cell)
                                * (1.0 / transformation.magnification - 1.0).max(0.0)
                                * 2.0_f64.sqrt(),
                        )
                    };
                    stack.push(Work::Entries(CellCursor {
                        cell,
                        transformation,
                        depth,
                        multiplicity,
                        lattice,
                        query: index.query(),
                        target,
                        chunks: RenderSet::default(),
                    }));
                }
                Work::Leaf(mut cursor) => {
                    let scene = &job.scene;
                    let source = scene
                        .layout
                        .cell(cursor.cell)
                        .ok_or_else(|| ViewError("Missing leaf cell".into()))?;
                    if cursor.index >= source.shapes().len() {
                        continue;
                    }
                    let shared = self.shared;
                    let generation = self.job.generation;
                    let reuse = &self.reuse;
                    let chunk = self.cache.chunk(
                        scene,
                        cursor.cell,
                        cursor.index,
                        cursor.transformation.magnification,
                        || {
                            shared.generation.load(Ordering::Acquire) != generation
                                || shared.stopped.load(Ordering::Acquire)
                                || reuse.load(Ordering::Acquire)
                        },
                    )?;
                    self.shared
                        .cache_bytes
                        .store(self.cache.total_bytes(), Ordering::Relaxed);
                    for mesh in &chunk.meshes {
                        if self.layers.contains(&mesh.layer) {
                            self.mesh(
                                mesh,
                                cursor.transformation,
                                cursor.multiplicity,
                                cursor.lattice,
                            )?;
                        }
                    }
                    if scene
                        .label_metrics
                        .get(cursor.cell.index())
                        .is_some_and(|metrics| *metrics != (0.0, 0.0))
                    {
                        let shapes = source
                            .shapes()
                            .get(chunk.range.0..chunk.range.1)
                            .ok_or_else(|| ViewError("Invalid prepared leaf range".into()))?;
                        for (offset, shape) in shapes.iter().enumerate() {
                            if self.layers.contains(&shape.layer)
                                && let Geometry::Label(label) = &shape.geometry
                            {
                                self.label(
                                    label,
                                    shape.layer,
                                    cursor.transformation,
                                    cursor.multiplicity,
                                    cursor.lattice,
                                )
                                .map_err(|error| {
                                    shape_error(
                                        source.name(),
                                        shape.layer,
                                        chunk.range.0 + offset,
                                        error,
                                    )
                                })?;
                            }
                        }
                    }
                    cursor.index = chunk.range.1;
                    stack.push(Work::Leaf(cursor));
                }
                Work::Entries(mut cursor) => {
                    let scene = &job.scene;
                    let index = scene
                        .indexes
                        .get(&cursor.cell)
                        .ok_or_else(|| ViewError("Missing spatial index".into()))?;
                    let Some(entry) = cursor.query.next(
                        index,
                        cursor.target,
                        cursor.transformation.magnification,
                        1.0 / (self.job.key.camera.pixels_per_unit
                            * cursor.transformation.magnification),
                    ) else {
                        continue;
                    };
                    let cell = scene
                        .layout
                        .cell(cursor.cell)
                        .ok_or_else(|| ViewError("Missing cell".into()))?;
                    if entry.instance && cursor.depth < self.job.key.depth {
                        let instance = cell
                            .instances()
                            .get(entry.index)
                            .ok_or_else(|| ViewError("Missing indexed instance".into()))?;
                        let base = cursor.transformation.compose(instance.transformation);
                        let Some(bounds) = scene.layout.transformed_bounds(instance.cell, base)
                        else {
                            stack.push(Work::Entries(cursor));
                            continue;
                        };
                        let column_step = cursor
                            .transformation
                            .apply_vector(instance.array.column_step);
                        let row_step = cursor.transformation.apply_vector(instance.array.row_step);
                        let metrics = scene
                            .label_metrics
                            .get(instance.cell.index())
                            .copied()
                            .ok_or_else(|| ViewError("Missing child label metrics".into()))?;
                        let (mut columns, mut rows) = visible_array_range(
                            bounds,
                            self.visible.expanded(text_margin(
                                metrics,
                                base.magnification,
                                1.0 / self.job.key.camera.pixels_per_unit,
                            )),
                            column_step,
                            row_step,
                            instance.array.columns,
                            instance.array.rows,
                        );
                        if columns.0 == columns.1 || rows.0 == rows.1 {
                            stack.push(Work::Entries(cursor));
                            continue;
                        }
                        let mut multiplicity = cursor.multiplicity;
                        if column_step == DoublePoint::default() {
                            multiplicity =
                                multiplicity.saturating_mul(usize::from(columns.1 - columns.0));
                            columns.1 = columns.0 + 1;
                        }
                        if row_step == DoublePoint::default() {
                            multiplicity =
                                multiplicity.saturating_mul(usize::from(rows.1 - rows.0));
                            rows.1 = rows.0 + 1;
                        }
                        let child = instance.cell;
                        let depth = cursor.depth.saturating_add(1);
                        let leaf = scene
                            .layout
                            .cell(child)
                            .is_some_and(|cell| cell.instances().is_empty());
                        if leaf {
                            let translation = base
                                .translation
                                .plus(column_step.times(f64::from(columns.0)))
                                .plus(row_step.times(f64::from(rows.0)));
                            let lattice = Lattice {
                                column: column_step,
                                row: row_step,
                                columns: u32::from(columns.1 - columns.0),
                                rows: u32::from(rows.1 - rows.0),
                            };
                            stack.push(Work::Entries(cursor));
                            let lattice = if lattice.columns == 1 && lattice.rows == 1 {
                                None
                            } else {
                                Some(lattice)
                            };
                            stack.push(Work::Cell(
                                child,
                                Transformation {
                                    translation,
                                    ..base
                                },
                                depth,
                                multiplicity,
                                lattice,
                            ));
                        } else {
                            stack.push(Work::Entries(cursor));
                            stack.push(Work::Array(ArrayCursor {
                                cell: child,
                                base,
                                depth,
                                multiplicity,
                                column_step,
                                row_step,
                                columns,
                                rows,
                                column: columns.0,
                                row: rows.0,
                            }));
                        }
                    } else if !entry.instance {
                        let shape = cell
                            .shapes()
                            .get(entry.index)
                            .ok_or_else(|| ViewError("Missing indexed geometry".into()))?;
                        if self.layers.contains(&shape.layer) {
                            if let Geometry::Label(label) = &shape.geometry {
                                self.label(
                                    label,
                                    shape.layer,
                                    cursor.transformation,
                                    cursor.multiplicity,
                                    cursor.lattice,
                                )
                                .map_err(|error| {
                                    shape_error(cell.name(), shape.layer, entry.index, error)
                                })?;
                            } else {
                                let mut width = CHUNK_SHAPES;
                                let mut already_prepared = false;
                                loop {
                                    let start = entry.index / width * width;
                                    let end = start.saturating_add(width).min(cell.shapes().len());
                                    if cursor.chunks.contains(&(start, end)) {
                                        already_prepared = true;
                                        break;
                                    }
                                    if width == 1 {
                                        break;
                                    }
                                    width /= 4;
                                }
                                if already_prepared {
                                    stack.push(Work::Entries(cursor));
                                    continue;
                                }
                                let shared = self.shared;
                                let generation = self.job.generation;
                                let reuse = &self.reuse;
                                let chunk = self.cache.chunk(
                                    scene,
                                    cursor.cell,
                                    entry.index,
                                    cursor.transformation.magnification,
                                    || {
                                        shared.generation.load(Ordering::Acquire) != generation
                                            || shared.stopped.load(Ordering::Acquire)
                                            || reuse.load(Ordering::Acquire)
                                    },
                                )?;
                                if cursor.chunks.insert(chunk.range) {
                                    self.shared
                                        .cache_bytes
                                        .store(self.cache.total_bytes(), Ordering::Relaxed);
                                    for mesh in &chunk.meshes {
                                        if self.layers.contains(&mesh.layer) {
                                            self.mesh(
                                                mesh,
                                                cursor.transformation,
                                                cursor.multiplicity,
                                                cursor.lattice,
                                            )?;
                                        }
                                    }
                                }
                            }
                        }
                        stack.push(Work::Entries(cursor));
                    } else {
                        stack.push(Work::Entries(cursor));
                    }
                }
                Work::Array(mut array) => {
                    if array.row >= array.rows.1 {
                        continue;
                    }
                    let translation = array
                        .base
                        .translation
                        .plus(array.column_step.times(f64::from(array.column)))
                        .plus(array.row_step.times(f64::from(array.row)));
                    let child = Work::Cell(
                        array.cell,
                        Transformation {
                            translation,
                            ..array.base
                        },
                        array.depth,
                        array.multiplicity,
                        None,
                    );
                    array.column += 1;
                    if array.column >= array.columns.1 {
                        array.column = array.columns.0;
                        array.row += 1;
                    }
                    stack.push(Work::Array(array));
                    stack.push(child);
                }
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), ViewError> {
        if self.reuse.load(Ordering::Acquire) {
            self.batches.clear();
            self.batch_bytes = 0;
            return Ok(());
        }
        let mut batches = std::mem::take(&mut self.batches)
            .into_values()
            .collect::<Vec<_>>();
        batches.sort_unstable_by_key(|batch| {
            (batch.layer, batch.mesh.id, batch.array.map(ArrayKey::from))
        });
        for batch in batches {
            if self.finished() {
                break;
            }
            if !self.deliver_batch(batch) {
                break;
            }
        }
        self.batch_bytes = 0;
        Ok(())
    }

    fn deliver_batch(&mut self, batch: PendingDraw) -> bool {
        let delivered = deliver_checked(
            self.shared,
            self.sender,
            self.job.generation,
            Ok(Packet::Draw {
                generation: self.job.generation,
                tile_id: self.tile.id,
                layer: batch.layer,
                mesh: batch.mesh,
                instances: batch.instances,
                array: batch.array,
                shape_count: batch.shape_count,
            }),
            Some(&self.reuse),
        );
        self.job.context.request_repaint();
        update_timing(self.shared, self.started);
        delivered
    }

    fn flush_batch(&mut self, key: BatchKey) {
        if let Some(batch) = self.batches.remove(&key) {
            let bytes =
                batch.mesh.bytes() + batch.instances.capacity() * std::mem::size_of::<Instance>();
            self.batch_bytes = self.batch_bytes.saturating_sub(bytes);
            self.deliver_batch(batch);
        }
    }

    fn append(
        &mut self,
        layer: LayerKey,
        mesh: Arc<MeshData>,
        instance: Instance,
        shape_count: usize,
    ) -> Result<(), ViewError> {
        self.append_array(layer, mesh, instance, None, shape_count)
    }

    fn append_array(
        &mut self,
        layer: LayerKey,
        mesh: Arc<MeshData>,
        instance: Instance,
        array: Option<ArrayInstances>,
        shape_count: usize,
    ) -> Result<(), ViewError> {
        if self.finished() {
            return Ok(());
        }
        let array = array.filter(|array| array.columns != 1 || array.rows != 1);
        let key = (layer, mesh.id, array.map(ArrayKey::from));
        let limit = match array {
            Some(array) => {
                let count = u64::from(array.columns) * u64::from(array.rows);
                if count == 0 || count > u64::from(u32::MAX) {
                    return Err(ViewError(
                        "Prepared array exceeds GPU instance range".into(),
                    ));
                }
                (u64::from(u32::MAX) / count).min(INSTANCE_LIMIT as u64) as usize
            }
            None => INSTANCE_LIMIT,
        };
        let instance_bytes = std::mem::size_of::<Instance>();
        let full = if let Some(batch) = self.batches.get_mut(&key) {
            let previous_capacity = batch.instances.capacity();
            let next_capacity = if batch.instances.len() == previous_capacity {
                previous_capacity.saturating_mul(2).max(1)
            } else {
                previous_capacity
            };
            let growth = next_capacity.saturating_sub(previous_capacity) * instance_bytes;
            if batch.instances.len() < limit
                && self.batch_bytes.saturating_add(growth) <= self.batch_limit
            {
                if growth != 0 {
                    batch
                        .instances
                        .reserve_exact(next_capacity - batch.instances.len());
                }
                batch.instances.push(instance);
                batch.shape_count = batch.shape_count.saturating_add(shape_count);
                self.batch_bytes +=
                    (batch.instances.capacity() - previous_capacity) * instance_bytes;
                return Ok(());
            }
            batch.instances.len() >= limit
        } else {
            false
        };
        if full {
            self.flush_batch(key);
        }
        let extra = mesh.bytes() + instance_bytes;
        if self.batch_bytes.saturating_add(extra) > self.batch_limit {
            self.flush()?;
        }
        self.batches.insert(
            key,
            PendingDraw {
                layer,
                mesh,
                instances: vec![instance],
                shape_count,
                array,
            },
        );
        self.batch_bytes = self.batch_bytes.saturating_add(extra);
        Ok(())
    }

    fn draw_array(
        &mut self,
        layer: LayerKey,
        mesh: Arc<MeshData>,
        instance: Instance,
        array: ArrayInstances,
        count: usize,
    ) -> Result<(), ViewError> {
        self.append_array(layer, mesh, instance, Some(array), count)
    }

    fn mesh(
        &mut self,
        mesh: &PreparedMesh,
        transformation: Transformation,
        multiplicity: usize,
        lattice: Option<Lattice>,
    ) -> Result<(), ViewError> {
        if self.finished() {
            return Ok(());
        }
        let bounds = mesh.bounds.transformed(transformation);
        let bounds = lattice.map_or(bounds, |lattice| {
            array_bounds(
                bounds,
                lattice.column,
                lattice.row,
                lattice.columns,
                lattice.rows,
            )
        });
        if !bounds.intersects(
            self.visible
                .expanded(2.0 / (self.job.key.camera.pixels_per_unit * self.scale)),
        ) {
            return Ok(());
        }
        let affine = self.affine(transformation, mesh.origin);
        let base_count = mesh.shape_count.saturating_mul(multiplicity);
        let center = mesh.bounds.center().minus(mesh.origin);
        let count = base_count.saturating_mul(self.owned(affine.apply(center), lattice));
        if let Some(lattice) = lattice {
            let array =
                projected_lattice(lattice, self.job.key.camera.pixels_per_unit * self.scale);
            if affine.array_precise(
                mesh.rounding,
                mesh.extent,
                lattice,
                array,
                self.job.key.camera.pixels_per_unit * self.scale,
            ) {
                return self.draw_array(
                    mesh.layer,
                    mesh.data.clone(),
                    affine.instance(),
                    array,
                    count,
                );
            }
            for row in 0..lattice.rows {
                for column in 0..lattice.columns {
                    if self.finished() {
                        return Ok(());
                    }
                    let translated = Transformation {
                        translation: transformation
                            .translation
                            .plus(lattice.column.times(f64::from(column)))
                            .plus(lattice.row.times(f64::from(row))),
                        ..transformation
                    };
                    let projection = Projection::new(
                        translated,
                        self.job.key.camera,
                        self.job.key.viewport,
                        self.scale,
                        self.origin,
                    );
                    self.precise_mesh(
                        mesh,
                        translated,
                        base_count.saturating_mul(
                            self.owned(projection.apply(mesh.origin, center), None),
                        ),
                    )?;
                }
            }
            Ok(())
        } else if affine.precise(mesh) {
            self.append(mesh.layer, mesh.data.clone(), affine.instance(), count)
        } else {
            let projection = Projection::new(
                transformation,
                self.job.key.camera,
                self.job.key.viewport,
                self.scale,
                self.origin,
            );
            self.precise_mesh(
                mesh,
                transformation,
                base_count.saturating_mul(self.owned(projection.apply(mesh.origin, center), None)),
            )
        }
    }

    fn label(
        &mut self,
        label: &Label,
        layer: LayerKey,
        parent: Transformation,
        multiplicity: usize,
        lattice: Option<Lattice>,
    ) -> Result<(), ViewError> {
        if self.finished() {
            return Ok(());
        }
        let transformation = parent.compose(label.transformation);
        text_preflight(&label.text)?;
        let key = GlyphKey {
            text: label.text.clone(),
            size: (12.0 * transformation.magnification)
                .clamp(6.0, 96.0)
                .to_bits(),
            pixels_per_point: self.job.key.pixels_per_point.to_bits(),
        };
        let glyph = self.cache.glyph(&self.job.context, key)?;
        self.shared
            .cache_bytes
            .store(self.cache.total_bytes(), Ordering::Relaxed);
        let anchor = DoublePoint::new(
            f64::from(glyph.logical_size.x) * f64::from(label.horizontal_alignment) * 0.5,
            f64::from(glyph.logical_size.y) * f64::from(label.vertical_alignment) * 0.5,
        );
        let (sine, cosine) = transformation.angle.sin_cos();
        let reflection = if transformation.reflected { -1.0 } else { 1.0 };
        let matrix = [
            cosine * self.scale,
            sine * reflection * self.scale,
            -sine * self.scale,
            cosine * reflection * self.scale,
        ];
        let mut affine = self.affine(transformation, DoublePoint::default());
        affine.matrix = matrix;
        affine.translation = affine.translation.minus(DoublePoint::new(
            matrix[0] * anchor.x + matrix[1] * anchor.y,
            matrix[2] * anchor.x + matrix[3] * anchor.y,
        ));
        let center = DoublePoint::new(
            f64::from(glyph.logical_size.x) * 0.5,
            f64::from(glyph.logical_size.y) * 0.5,
        );
        let count = multiplicity.saturating_mul(self.owned(affine.apply(center), lattice));
        if let Some(lattice) = lattice {
            let array =
                projected_lattice(lattice, self.job.key.camera.pixels_per_unit * self.scale);
            if affine.array_precise(
                [0.0; 2],
                [
                    f64::from(glyph.logical_size.x),
                    f64::from(glyph.logical_size.y),
                ],
                lattice,
                array,
                self.job.key.camera.pixels_per_unit * self.scale,
            ) {
                self.draw_array(layer, glyph.mesh.clone(), affine.instance(), array, count)
            } else {
                let scale = self.job.key.camera.pixels_per_unit * self.scale;
                let column_step =
                    DoublePoint::new(lattice.column.x, -lattice.column.y).times(scale);
                let row_step = DoublePoint::new(lattice.row.x, -lattice.row.y).times(scale);
                for row in 0..lattice.rows {
                    for column in 0..lattice.columns {
                        if self.finished() {
                            return Ok(());
                        }
                        let translated = Affine {
                            translation: affine
                                .translation
                                .plus(column_step.times(f64::from(column)))
                                .plus(row_step.times(f64::from(row))),
                            ..affine
                        };
                        self.append(
                            layer,
                            glyph.mesh.clone(),
                            translated.instance(),
                            multiplicity.saturating_mul(self.owned(translated.apply(center), None)),
                        )?;
                    }
                }
                Ok(())
            }
        } else {
            self.append(layer, glyph.mesh.clone(), affine.instance(), count)
        }
    }

    fn owned(&self, center: DoublePoint, lattice: Option<Lattice>) -> usize {
        let Some(lattice) = lattice else {
            return usize::from(
                center.x >= f64::from(self.tile.crop.left())
                    && center.y >= f64::from(self.tile.crop.top())
                    && center.x < f64::from(self.tile.crop.right())
                    && center.y < f64::from(self.tile.crop.bottom()),
            );
        };
        let scale = self.job.key.camera.pixels_per_unit * self.scale;
        let column = DoublePoint::new(lattice.column.x, -lattice.column.y).times(scale);
        let row = DoublePoint::new(lattice.row.x, -lattice.row.y).times(scale);
        let mut count = 0_usize;
        for row_index in 0..lattice.rows {
            if self.finished() {
                break;
            }
            let position = center.plus(row.times(f64::from(row_index)));
            let mut start = 0.0_f64;
            let mut end = f64::from(lattice.columns);
            for (position, step, minimum, maximum) in [
                (
                    position.x,
                    column.x,
                    f64::from(self.tile.crop.left()),
                    f64::from(self.tile.crop.right()),
                ),
                (
                    position.y,
                    column.y,
                    f64::from(self.tile.crop.top()),
                    f64::from(self.tile.crop.bottom()),
                ),
            ] {
                if step == 0.0 {
                    if position < minimum || position >= maximum {
                        end = 0.0;
                        break;
                    }
                } else if step > 0.0 {
                    start = start.max(((minimum - position) / step).ceil());
                    end = end.min(((maximum - position) / step).ceil());
                } else {
                    start = start.max(((maximum - position) / step).floor() + 1.0);
                    end = end.min(((minimum - position) / step).floor() + 1.0);
                }
            }
            if end > start {
                count = count.saturating_add((end - start) as usize);
            }
        }
        count
    }

    fn precise_mesh(
        &mut self,
        mesh: &PreparedMesh,
        transformation: Transformation,
        count: usize,
    ) -> Result<(), ViewError> {
        let projection = Projection::new(
            transformation,
            self.job.key.camera,
            self.job.key.viewport,
            self.scale,
            self.origin,
        );
        let clip = Bounds {
            minimum: DoublePoint::new(-2.0, -2.0),
            maximum: DoublePoint::new(
                f64::from(self.tile.size[0]) + 2.0,
                f64::from(self.tile.size[1]) + 2.0,
            ),
        };
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for triangle in mesh.data.indices.as_chunks::<3>().0 {
            if self.finished() {
                return Ok(());
            }
            let source = triangle
                .iter()
                .map(|index| {
                    mesh.precise
                        .get(*index as usize)
                        .copied()
                        .ok_or_else(|| ViewError("Invalid precise mesh index".into()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let first = source
                .first()
                .copied()
                .ok_or_else(|| ViewError("Empty precise triangle".into()))?;
            if first.kind == 0 || first.kind == 4 {
                let points = source
                    .iter()
                    .map(|vertex| {
                        if vertex.kind == 4 {
                            projection.rectangle_corner(mesh.origin, *vertex)
                        } else {
                            projection.apply(mesh.origin, vertex.position)
                        }
                    })
                    .collect::<Vec<_>>();
                if points.iter().any(|point| !point.finite()) {
                    return Err(ViewError("Screen coordinate overflow".into()));
                }
                let clipped = clip_polygon(points, clip);
                if clipped.len() < 3 {
                    continue;
                }
                let offset = u32::try_from(vertices.len())
                    .map_err(|_| ViewError("Precise mesh index overflow".into()))?;
                for point in &clipped {
                    vertices.push(GpuVertex {
                        position: [point.x as f32, point.y as f32],
                        other: [point.x as f32, point.y as f32],
                        uv: [0.0; 2],
                        bounds: [0.0; 4],
                        kind: 0,
                        corner: 0,
                    });
                }
                for index in 1..clipped.len() - 1 {
                    let index = u32::try_from(index)
                        .map_err(|_| ViewError("Precise triangle index overflow".into()))?;
                    indices.extend([offset, offset + index, offset + index + 1]);
                }
            } else {
                let first_point = projection.apply(mesh.origin, first.position);
                let second_point = projection.apply(mesh.origin, first.other);
                if !first_point.finite() || !second_point.finite() {
                    return Err(ViewError("Screen line coordinate overflow".into()));
                }
                let Some((first_point, second_point)) =
                    clip_segment(first_point, second_point, clip)
                else {
                    continue;
                };
                let bounds = Bounds::from_points([
                    projection.apply(mesh.origin, first.bounds.minimum),
                    projection.apply(
                        mesh.origin,
                        DoublePoint::new(first.bounds.minimum.x, first.bounds.maximum.y),
                    ),
                    projection.apply(mesh.origin, first.bounds.maximum),
                    projection.apply(
                        mesh.origin,
                        DoublePoint::new(first.bounds.maximum.x, first.bounds.minimum.y),
                    ),
                ])
                .ok_or_else(|| ViewError("Missing precise edge bounds".into()))?;
                let bounds = [
                    bounds.minimum.x,
                    bounds.minimum.y,
                    bounds.maximum.x,
                    bounds.maximum.y,
                ]
                .map(|coordinate| coordinate.clamp(-1e6, 1e6) as f32);
                for vertex in &source {
                    let index = u32::try_from(vertices.len())
                        .map_err(|_| ViewError("Precise edge index overflow".into()))?;
                    vertices.push(GpuVertex {
                        position: [first_point.x as f32, first_point.y as f32],
                        other: [second_point.x as f32, second_point.y as f32],
                        uv: [
                            if vertex.kind == 1 {
                                (f64::from(vertex.uv[0])
                                    * transformation.magnification
                                    * projection.density)
                                    .min(1e6) as f32
                            } else {
                                vertex.uv[0]
                            },
                            vertex.uv[1],
                        ],
                        bounds,
                        kind: vertex.kind,
                        corner: vertex.corner,
                    });
                    indices.push(index);
                }
            }
            if vertices
                .len()
                .saturating_mul(std::mem::size_of::<GpuVertex>() + std::mem::size_of::<u32>())
                >= PACKET_LIMIT / 2
            {
                let data = Arc::new(MeshData {
                    id: resource_id()?,
                    vertices: std::mem::take(&mut vertices),
                    indices: std::mem::take(&mut indices),
                    glyph: None,
                });
                self.append(
                    mesh.layer,
                    data,
                    Instance {
                        affine: [1.0, 0.0, 0.0, 1.0],
                        translation: [0.0; 2],
                    },
                    0,
                )?;
            }
        }
        if !vertices.is_empty() {
            let data = Arc::new(MeshData {
                id: resource_id()?,
                vertices,
                indices,
                glyph: None,
            });
            self.append(
                mesh.layer,
                data,
                Instance {
                    affine: [1.0, 0.0, 0.0, 1.0],
                    translation: [0.0; 2],
                },
                count,
            )?;
        }
        Ok(())
    }
}

fn clip_polygon(mut points: Vec<DoublePoint>, bounds: Bounds) -> Vec<DoublePoint> {
    for (axis, boundary, minimum) in [
        (0, bounds.minimum.x, true),
        (0, bounds.maximum.x, false),
        (1, bounds.minimum.y, true),
        (1, bounds.maximum.y, false),
    ] {
        if points.is_empty() {
            break;
        }
        let coordinate = |point: DoublePoint| if axis == 0 { point.x } else { point.y };
        let inside = |point: DoublePoint| {
            if minimum {
                coordinate(point) >= boundary
            } else {
                coordinate(point) <= boundary
            }
        };
        let mut result = Vec::with_capacity(points.len() + 1);
        for (first, second) in points
            .iter()
            .copied()
            .zip(points.iter().copied().cycle().skip(1))
            .take(points.len())
        {
            let first_inside = inside(first);
            let second_inside = inside(second);
            if first_inside {
                result.push(first);
            }
            if first_inside != second_inside {
                let difference = second.minus(first);
                let fraction =
                    (boundary - coordinate(first)) / (coordinate(second) - coordinate(first));
                let mut point = first.plus(difference.times(fraction));
                if axis == 0 {
                    point.x = boundary;
                } else {
                    point.y = boundary;
                }
                result.push(point);
            }
        }
        points = result;
    }
    points
}

fn clip_segment(
    first: DoublePoint,
    second: DoublePoint,
    bounds: Bounds,
) -> Option<(DoublePoint, DoublePoint)> {
    let direction = second.minus(first);
    let mut minimum = 0.0_f64;
    let mut maximum = 1.0_f64;
    for (position, step, lower, upper) in [
        (first.x, direction.x, bounds.minimum.x, bounds.maximum.x),
        (first.y, direction.y, bounds.minimum.y, bounds.maximum.y),
    ] {
        if step == 0.0 {
            if position < lower || position > upper {
                return None;
            }
        } else {
            let first = (lower - position) / step;
            let second = (upper - position) / step;
            minimum = minimum.max(first.min(second));
            maximum = maximum.min(first.max(second));
            if minimum > maximum {
                return None;
            }
        }
    }
    Some((
        first.plus(direction.times(minimum)),
        first.plus(direction.times(maximum)),
    ))
}

fn text_margin(metrics: (f64, f64), magnification: f64, units_per_point: f64) -> f64 {
    if metrics.0 == 0.0 {
        0.0
    } else {
        (metrics.0 + 1.0)
            * (12.0 * metrics.1 * magnification).clamp(6.0, 96.0)
            * units_per_point
            * 2.0_f64.sqrt()
    }
}

fn inverse(transformation: Transformation) -> Transformation {
    let inverse = Transformation {
        translation: DoublePoint::default(),
        angle: if transformation.reflected {
            transformation.angle
        } else {
            -transformation.angle
        },
        magnification: 1.0 / transformation.magnification,
        reflected: transformation.reflected,
    };
    Transformation {
        translation: inverse.apply_vector(transformation.translation).times(-1.0),
        ..inverse
    }
}

fn array_bounds(
    bounds: Bounds,
    column: DoublePoint,
    row: DoublePoint,
    columns: u32,
    rows: u32,
) -> Bounds {
    let column = column.times(f64::from(columns.saturating_sub(1)));
    let row = row.times(f64::from(rows.saturating_sub(1)));
    [DoublePoint::default(), column, row, column.plus(row)]
        .into_iter()
        .map(|translation| Bounds {
            minimum: bounds.minimum.plus(translation),
            maximum: bounds.maximum.plus(translation),
        })
        .fold(bounds, Bounds::union)
}

fn projected_lattice(lattice: Lattice, scale: f64) -> ArrayInstances {
    ArrayInstances {
        column_step: [
            (lattice.column.x * scale) as f32,
            (-lattice.column.y * scale) as f32,
        ],
        row_step: [
            (lattice.row.x * scale) as f32,
            (-lattice.row.y * scale) as f32,
        ],
        columns: lattice.columns,
        rows: lattice.rows,
    }
}

fn lattice_error(lattice: Lattice, array: ArrayInstances, scale: f64) -> f64 {
    let first = lattice.column.times(scale);
    let second = lattice.row.times(scale);
    [
        (first.x, second.x, array.column_step[0], array.row_step[0]),
        (-first.y, -second.y, array.column_step[1], array.row_step[1]),
    ]
    .into_iter()
    .map(|(first, second, rounded_first, rounded_second)| {
        let columns = f64::from(lattice.columns.saturating_sub(1));
        let rows = f64::from(lattice.rows.saturating_sub(1));
        let error = (first - f64::from(rounded_first)).abs() * columns
            + (second - f64::from(rounded_second)).abs() * rows
            + (first.abs() * columns + second.abs() * rows) * f64::from(f32::EPSILON) * 4.0;
        if error.is_finite() {
            error
        } else {
            f64::INFINITY
        }
    })
    .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlayout_core::{Cell, PathEnds, Point, Shape};
    use std::time::{Duration, Instant};

    fn fixture(geometry: Geometry) -> Result<Arc<Scene>, Box<dyn Error>> {
        Ok(Arc::new(Scene::prepare(Layout::new(
            "preparation".into(),
            1e-9,
            1e-3,
            vec![Cell::new(
                "CELL".into(),
                vec![Shape {
                    layer: LayerKey {
                        layer: 1,
                        datatype: 0,
                    },
                    geometry,
                }],
                vec![],
            )],
        )?)?))
    }

    fn rectangle() -> Geometry {
        Geometry::Box {
            minimum: Point::new(0, 0),
            maximum: Point::new(10, 10),
        }
    }

    fn request() -> FrameKey {
        FrameKey {
            camera: Camera {
                center: DoublePoint::new(5.0, 5.0),
                pixels_per_unit: 1.0,
            },
            viewport: Rect::from_min_size(Pos2::ZERO, Vec2::splat(64.0)),
            pixels_per_point: 1.0,
            selected: Some(CellId::from_index(0)),
            depth: u16::MAX,
            styles: BTreeMap::from([(
                LayerKey {
                    layer: 1,
                    datatype: 0,
                },
                LayerStyle {
                    visible: true,
                    color: Color32::RED,
                },
            )]),
            text_pass: 0,
        }
    }

    fn drain(worker: &mut Worker, reuse: bool) -> Result<Vec<Packet>, Box<dyn Error>> {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut packets = Vec::new();
        while Instant::now() < deadline {
            for packet in worker.receive() {
                let packet = packet?;
                if reuse && let Packet::Begin { reuse, .. } = &packet {
                    reuse.store(true, Ordering::Release);
                }
                let complete = matches!(&packet, Packet::Finish { complete: true, .. });
                packets.push(packet);
                if complete {
                    return Ok(packets);
                }
            }
            thread::sleep(Duration::from_millis(1));
        }
        Err("Geometry worker did not finish".into())
    }

    #[test]
    fn paths_cache_magnification_only_when_width_is_absolute() -> Result<(), Box<dyn Error>> {
        let path = |absolute_width| {
            Geometry::Path(Path {
                points: vec![Point::new(0, 0), Point::new(20, 0)],
                width: 12,
                absolute_width,
                ends: PathEnds::Flush,
            })
        };
        let relative = fixture(path(false))?;
        let absolute = fixture(path(true))?;
        let mut cache = Cache::new();
        let first = cache.chunk(&relative, CellId::from_index(0), 0, 1.0, || false)?;
        let magnified = cache.chunk(&relative, CellId::from_index(0), 0, 3.0, || false)?;
        assert!(Arc::ptr_eq(&first, &magnified));
        let first = cache.chunk(&absolute, CellId::from_index(0), 0, 1.0, || false)?;
        let magnified = cache.chunk(&absolute, CellId::from_index(0), 0, 3.0, || false)?;
        assert!(!Arc::ptr_eq(&first, &magnified));
        let first = first.meshes.first().ok_or("Missing path mesh")?;
        let magnified = magnified
            .meshes
            .first()
            .ok_or("Missing magnified path mesh")?;
        assert!((first.bounds.height() - 12.0).abs() < 1e-9);
        assert!((magnified.bounds.height() * 3.0 - 12.0).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn cached_chunks_evict_and_supported_empty_fills_remain_empty() -> Result<(), Box<dyn Error>> {
        let first = fixture(rectangle())?;
        let second = fixture(rectangle())?;
        let mut cache = Cache::new();
        let chunk = cache.chunk(&first, CellId::from_index(0), 0, 1.0, || false)?;
        cache.limit = chunk.bytes() + cache.metadata_bytes(true);
        cache.chunk(&second, CellId::from_index(0), 0, 1.0, || false)?;
        assert_eq!(cache.chunks.len(), 1);
        assert!(cache.bytes <= cache.limit);
        assert!(cache.bytes + cache.metadata_bytes(false) <= cache.limit);
        let rebuilt = cache.chunk(&first, CellId::from_index(0), 0, 1.0, || false)?;
        assert!(!Arc::ptr_eq(&chunk, &rebuilt));
        assert_eq!(
            chunk.meshes.first().ok_or("Missing source mesh")?.data.id,
            rebuilt
                .meshes
                .first()
                .ok_or("Missing rebuilt mesh")?
                .data
                .id,
            "CPU eviction changed the GPU identity of immutable geometry"
        );
        for points in [
            vec![Point::new(0, 0), Point::new(5, 0), Point::new(10, 0)],
            vec![
                Point::new(0, 0),
                Point::new(10, 0),
                Point::new(0, 0),
                Point::new(0, 10),
            ],
        ] {
            let scene = fixture(Geometry::Polygon(points))?;
            let empty = cache.chunk(&scene, CellId::from_index(0), 0, 1.0, || false)?;
            assert!(empty.meshes.is_empty());
            assert!(empty.bytes() >= std::mem::size_of::<PreparedChunk>());
        }
        Ok(())
    }

    #[test]
    fn cache_second_chance_preserves_recently_reused_geometry() -> Result<(), Box<dyn Error>> {
        let first = fixture(rectangle())?;
        let second = fixture(rectangle())?;
        let third = fixture(rectangle())?;
        let mut cache = Cache::new();
        let first_chunk = cache.chunk(&first, CellId::from_index(0), 0, 1.0, || false)?;
        cache.limit = first_chunk.bytes() * 2 + cache.metadata_bytes(true);
        let second_chunk = cache.chunk(&second, CellId::from_index(0), 0, 1.0, || false)?;
        cache.chunk(&first, CellId::from_index(0), 0, 1.0, || false)?;
        cache.chunk(&third, CellId::from_index(0), 0, 1.0, || false)?;
        assert!(Arc::ptr_eq(
            &first_chunk,
            &cache.chunk(&first, CellId::from_index(0), 0, 1.0, || false)?
        ));
        assert!(!Arc::ptr_eq(
            &second_chunk,
            &cache.chunk(&second, CellId::from_index(0), 0, 1.0, || false)?
        ));
        assert!(cache.bytes <= cache.limit);
        Ok(())
    }

    #[test]
    fn scratch_preflight_bounds_crossings_and_text_before_allocation() -> Result<(), Box<dyn Error>>
    {
        let ring = (0..512)
            .map(|index| {
                let angle = index as f64 * std::f64::consts::TAU / 512.0;
                DoublePoint::new(angle.cos() * 1000.0, angle.sin() * 1000.0)
            })
            .collect::<Vec<_>>();
        tessellation_preflight(&ring, &|| false)?;
        assert!(tessellation_preflight(&ring, &|| true).is_err());
        let star = (0..96)
            .map(|index| {
                let angle = (index * 37 % 96) as f64 * std::f64::consts::TAU / 96.0;
                Point::new((angle.cos() * 1e6) as i64, (angle.sin() * 1e6) as i64)
            })
            .collect();
        let scene = fixture(Geometry::Polygon(star))?;
        let result = Cache::new().chunk(&scene, CellId::from_index(0), 0, 1.0, || false);
        let error = match result {
            Ok(_) => {
                return Err(
                    "Intersecting contour exceeded its scratch bound without failing".into(),
                );
            }
            Err(error) => error,
        };
        assert!(error.0.contains("layer 1/0, shape 1"));
        assert!(error.0.contains("tessellation scratch budget"));
        text_preflight("ordinary GDS label")?;
        assert!(text_preflight(&"x".repeat(TEXT_SCRATCH / TEXT_BYTES_PER_CHARACTER + 1)).is_err());
        let mut builder = Builder::new(LayerKey {
            layer: 1,
            datatype: 0,
        });
        builder.limit = 1;
        assert!(
            builder
                .geometry(&rectangle(), 1.0, &mut Vec::new(), &|| false)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn preparation_splits_expanding_chunks_without_omitting_source_shapes()
    -> Result<(), Box<dyn Error>> {
        let points = (0..48)
            .map(|index| {
                let angle = (index * 19 % 48) as f64 * std::f64::consts::TAU / 48.0;
                Point::new((angle.cos() * 1000.0) as i64, (angle.sin() * 1000.0) as i64)
            })
            .collect::<Vec<_>>();
        let layer = LayerKey {
            layer: 1,
            datatype: 0,
        };
        let scene = Arc::new(Scene::prepare(Layout::new(
            "adaptive".into(),
            1e-9,
            1e-3,
            vec![Cell::new(
                "EXPANDING".into(),
                (0..64)
                    .map(|_| Shape {
                        layer,
                        geometry: Geometry::Polygon(points.clone()),
                    })
                    .collect(),
                vec![],
            )],
        )?)?);
        let cell = CellId::from_index(0);
        let source = scene.layout.cell(cell).ok_or("Missing adaptive source")?;
        assert_eq!(
            chunk_range(source.shapes(), 0, CHUNK_SHAPES)?.1,
            CHUNK_SHAPES
        );
        let mut cache = Cache::new();
        let first = cache.chunk(&scene, cell, 0, 1.0, || false)?;
        assert!(
            first.range.1 < 64,
            "Expanded geometry was not split within its scratch budget"
        );
        let mut index = 0;
        let mut prepared_count = 0;
        while index < source.shapes().len() {
            let chunk = cache.chunk(&scene, cell, index, 1.0, || false)?;
            assert_eq!(chunk.range.0, index);
            prepared_count += chunk
                .meshes
                .iter()
                .map(|mesh| mesh.shape_count)
                .sum::<usize>();
            index = chunk.range.1;
        }
        assert_eq!(prepared_count, 64);
        let remembered = cache
            .mesh_identifiers
            .ranges
            .get(&RangeKey {
                revision: scene.revision,
                cell,
                start: 0,
            })
            .copied()
            .ok_or("Adaptive range was not remembered")?;
        assert!(remembered < CHUNK_SHAPES);
        let mut worker = Worker::new()?;
        worker.start(scene, request(), egui::Context::default(), false)?;
        let rendered_count = drain(&mut worker, false)?
            .iter()
            .map(|packet| match packet {
                Packet::Draw { shape_count, .. } => *shape_count,
                _ => 0,
            })
            .sum::<usize>();
        assert_eq!(
            rendered_count, 64,
            "Traversal omitted geometry after preparation split a source range"
        );
        Ok(())
    }

    #[test]
    fn compatible_arefs_keep_multiple_compact_bases_per_draw() -> Result<(), Box<dyn Error>> {
        let layer = LayerKey {
            layer: 1,
            datatype: 0,
        };
        let cell = CellId::from_index(0);
        let scene = Arc::new(Scene::prepare(Layout::new(
            "arrays".into(),
            1e-9,
            1e-3,
            vec![
                Cell::new(
                    "LEAF".into(),
                    vec![Shape {
                        layer,
                        geometry: rectangle(),
                    }],
                    vec![],
                ),
                Cell::new(
                    "TOP".into(),
                    vec![],
                    (0..100)
                        .map(|_| rlayout_core::Instance {
                            cell,
                            transformation: Transformation::default(),
                            array: rlayout_core::Array {
                                columns: 4,
                                rows: 4,
                                column_step: DoublePoint::new(20.0, 0.0),
                                row_step: DoublePoint::new(0.0, 20.0),
                            },
                        })
                        .collect(),
                ),
            ],
        )?)?);
        let mut worker = Worker::new()?;
        let mut key = request();
        key.selected = Some(CellId::from_index(1));
        key.camera.center = DoublePoint::new(35.0, 35.0);
        worker.start(scene, key, egui::Context::default(), false)?;
        let packets = drain(&mut worker, false)?;
        let mut count = 0;
        let mut found = false;
        for packet in packets {
            if let Packet::Draw {
                instances,
                array,
                shape_count,
                ..
            } = packet
            {
                let array = array.ok_or("A compatible AREF was flattened")?;
                assert!(array.columns * array.rows > 1);
                assert_eq!(
                    instances.len(),
                    100,
                    "Compatible arrays did not share their base-transform batch"
                );
                count += shape_count;
                found = true;
            }
        }
        assert!(found);
        assert_eq!(
            count, 1600,
            "Compact batching changed source occurrence accounting"
        );
        let scene = fixture(rectangle())?;
        let chunk = Cache::new().chunk(&scene, cell, 0, 1.0, || false)?;
        let mesh = chunk.meshes.first().ok_or("Missing precision test mesh")?;
        let affine = Affine {
            matrix: [1.0, 0.0, 0.0, 1.0],
            translation: DoublePoint::new(20_000.000_5, 0.0),
        };
        let lattice = Lattice {
            column: DoublePoint::new(1000.0, 0.0),
            row: DoublePoint::default(),
            columns: 20,
            rows: 1,
        };
        let array = projected_lattice(lattice, 1.0);
        assert!(affine.precise(mesh));
        assert!(lattice_error(lattice, array, 1.0) < PIXEL_ERROR);
        assert!(
            !affine.array_precise(mesh.rounding, mesh.extent, lattice, array, 1.0),
            "Separate conversion budgets were added beyond 1/64 pixel"
        );
        Ok(())
    }

    #[test]
    fn worker_filters_old_requests_and_reuse_skips_draw_planning() -> Result<(), Box<dyn Error>> {
        let layer = LayerKey {
            layer: 1,
            datatype: 0,
        };
        let instances = (0..20_000)
            .map(|_| rlayout_core::Instance {
                cell: CellId::from_index(0),
                transformation: Transformation::default(),
                array: rlayout_core::Array::default(),
            })
            .collect();
        let scene = Arc::new(Scene::prepare(Layout::new(
            "coalescing".into(),
            1e-9,
            1e-3,
            vec![
                Cell::new(
                    "LEAF".into(),
                    vec![Shape {
                        layer,
                        geometry: rectangle(),
                    }],
                    vec![],
                ),
                Cell::new("TOP".into(), vec![], instances),
            ],
        )?)?);
        let context = egui::Context::default();
        let mut worker = Worker::new()?;
        let mut key = request();
        key.selected = Some(CellId::from_index(1));
        worker.start(scene.clone(), request(), context.clone(), false)?;
        let mut hidden = key.clone();
        for style in hidden.styles.values_mut() {
            style.visible = false;
        }
        let current = worker.start(scene.clone(), hidden, context.clone(), false)?;
        let packets = drain(&mut worker, false)?;
        assert!(packets.iter().all(|packet| match packet {
            Packet::Begin { generation, .. }
            | Packet::Draw { generation, .. }
            | Packet::Finish { generation, .. } => *generation == current,
        }));
        assert!(
            packets
                .iter()
                .all(|packet| !matches!(packet, Packet::Draw { .. }))
        );
        worker.start(scene, key, context, false)?;
        let packets = drain(&mut worker, true)?;
        let mut draws = BTreeMap::<u64, usize>::new();
        for packet in &packets {
            if let Packet::Draw {
                tile_id,
                mesh,
                instances,
                ..
            } = packet
            {
                *draws.entry(*tile_id).or_default() += mesh.bytes()
                    + instances.capacity() * std::mem::size_of::<Instance>()
                    + std::mem::size_of::<Packet>() * 2;
            }
        }
        assert!(
            draws.values().all(|bytes| *bytes <= PACKET_LIMIT * 3),
            "Cached tiles prepared more than queued and in-flight delivery budgets"
        );
        assert!(worker.bytes() <= CACHE_LIMIT);
        Ok(())
    }

    #[test]
    fn hidden_only_leaf_skips_geometry_preparation() -> Result<(), Box<dyn Error>> {
        let visible = LayerKey {
            layer: 1,
            datatype: 0,
        };
        let hidden = LayerKey {
            layer: 2,
            datatype: 0,
        };
        let star = (0..96)
            .map(|index| {
                let angle = (index * 37 % 96) as f64 * std::f64::consts::TAU / 96.0;
                Point::new((angle.cos() * 10.0) as i64, (angle.sin() * 10.0) as i64)
            })
            .collect();
        let scene = Arc::new(Scene::prepare(Layout::new(
            "hidden leaf".into(),
            1e-9,
            1e-3,
            vec![
                Cell::new(
                    "HIDDEN".into(),
                    vec![Shape {
                        layer: hidden,
                        geometry: Geometry::Polygon(star),
                    }],
                    vec![],
                ),
                Cell::new(
                    "VISIBLE".into(),
                    vec![Shape {
                        layer: visible,
                        geometry: rectangle(),
                    }],
                    vec![],
                ),
                Cell::new(
                    "TOP".into(),
                    vec![],
                    vec![
                        rlayout_core::Instance {
                            cell: CellId::from_index(0),
                            transformation: Transformation::default(),
                            array: rlayout_core::Array::default(),
                        },
                        rlayout_core::Instance {
                            cell: CellId::from_index(1),
                            transformation: Transformation {
                                translation: DoublePoint::new(256.0, 256.0),
                                ..Transformation::default()
                            },
                            array: rlayout_core::Array::default(),
                        },
                    ],
                ),
            ],
        )?)?);
        let mut worker = Worker::new()?;
        let mut key = request();
        key.viewport = Rect::from_min_size(Pos2::ZERO, Vec2::splat(256.0));
        key.camera.center = DoublePoint::default();
        key.selected = Some(CellId::from_index(2));
        key.styles.insert(
            hidden,
            LayerStyle {
                visible: false,
                color: Color32::BLUE,
            },
        );
        worker.start(scene, key, egui::Context::default(), false)?;
        let packets = drain(&mut worker, false)?;
        assert!(
            packets
                .iter()
                .all(|packet| !matches!(packet, Packet::Draw { .. }))
        );
        assert_eq!(
            worker.shared.cache_bytes.load(Ordering::Relaxed),
            0,
            "Hidden geometry entered the preparation cache"
        );
        Ok(())
    }

    #[test]
    fn whole_leaf_matches_spatial_preparation_for_geometry_and_labels() -> Result<(), Box<dyn Error>>
    {
        let first = LayerKey {
            layer: 1,
            datatype: 0,
        };
        let second = LayerKey {
            layer: 2,
            datatype: 0,
        };
        let mut shapes = (0..100)
            .map(|index| {
                let x = (index % 10) * 3;
                let y = (index / 10) * 3;
                rlayout_core::Shape {
                    layer: if index % 2 == 0 { first } else { second },
                    geometry: Geometry::Box {
                        minimum: Point::new(x, y),
                        maximum: Point::new(x + 2, y + 2),
                    },
                }
            })
            .collect::<Vec<_>>();
        shapes.push(rlayout_core::Shape {
            layer: first,
            geometry: Geometry::Label(Label {
                text: "leaf".into(),
                transformation: Transformation {
                    translation: DoublePoint::new(10.0, 10.0),
                    ..Transformation::default()
                },
                horizontal_alignment: 0,
                vertical_alignment: 0,
            }),
        });
        let make_scene = |force_spatial| -> Result<Arc<Scene>, Box<dyn Error>> {
            Ok(Arc::new(Scene::prepare(Layout::new(
                "leaf traversal".into(),
                1e-9,
                1e-3,
                vec![
                    Cell::new(
                        "LEAF".into(),
                        shapes.clone(),
                        if force_spatial {
                            vec![rlayout_core::Instance {
                                cell: CellId::from_index(2),
                                transformation: Transformation::default(),
                                array: rlayout_core::Array::default(),
                            }]
                        } else {
                            Vec::new()
                        },
                    ),
                    Cell::new(
                        "TOP".into(),
                        vec![],
                        [0.0, 256.0]
                            .into_iter()
                            .map(|offset| rlayout_core::Instance {
                                cell: CellId::from_index(0),
                                transformation: Transformation {
                                    translation: DoublePoint::new(offset, offset),
                                    ..Transformation::default()
                                },
                                array: rlayout_core::Array::default(),
                            })
                            .collect(),
                    ),
                    Cell::new("EMPTY".into(), vec![], vec![]),
                ],
            )?)?))
        };
        let full = make_scene(false)?;
        let spatial = make_scene(true)?;
        let context = egui::Context::default();
        let mut output = context.run_ui(egui::RawInput::default(), |_| {});
        output.textures_delta.clear();
        let signature = |packets: Vec<Packet>| -> Result<_, Box<dyn Error>> {
            let mut tiles = BTreeMap::new();
            let mut vertices = Vec::new();
            let mut glyphs = Vec::new();
            for packet in packets {
                match packet {
                    Packet::Begin { tile, .. } => {
                        tiles.insert(tile.id, tile);
                    }
                    Packet::Draw {
                        tile_id,
                        layer,
                        mesh,
                        instances,
                        array,
                        ..
                    } => {
                        assert!(array.is_none());
                        let tile = tiles.get(&tile_id).ok_or("Missing leaf tile")?;
                        if let Some(glyph) = &mesh.glyph {
                            assert!(glyph.alpha.iter().any(|alpha| *alpha != 0));
                            glyphs.push(glyph.alpha.clone());
                        }
                        for instance in &instances {
                            for vertex in &mesh.vertices {
                                let x = f64::from(instance.affine[0])
                                    * f64::from(vertex.position[0])
                                    + f64::from(instance.affine[1]) * f64::from(vertex.position[1])
                                    + f64::from(instance.translation[0])
                                    + f64::from(tile.rect.left());
                                let y = f64::from(instance.affine[2])
                                    * f64::from(vertex.position[0])
                                    + f64::from(instance.affine[3]) * f64::from(vertex.position[1])
                                    + f64::from(instance.translation[1])
                                    + f64::from(tile.rect.top());
                                vertices.push((
                                    layer,
                                    vertex.kind,
                                    vertex.corner,
                                    x.to_bits(),
                                    y.to_bits(),
                                    vertex.uv[0].to_bits(),
                                ));
                            }
                        }
                    }
                    Packet::Finish { .. } => {}
                }
            }
            vertices.sort_unstable();
            glyphs.sort_unstable();
            Ok((vertices, glyphs))
        };
        let mut worker = Worker::new()?;
        let mut key = request();
        key.viewport = Rect::from_min_size(Pos2::ZERO, Vec2::splat(256.0));
        key.camera.center = DoublePoint::new(14.5, 14.5);
        key.selected = Some(CellId::from_index(1));
        key.styles.insert(
            second,
            LayerStyle {
                visible: true,
                color: Color32::GREEN,
            },
        );
        for scale in [1.0, 10.0] {
            key.camera.pixels_per_unit = scale;
            worker.start(full.clone(), key.clone(), context.clone(), false)?;
            let actual = signature(drain(&mut worker, false)?)?;
            let complete = worker.shared.complete_leaf_cells.load(Ordering::Relaxed);
            if scale == 1.0 {
                assert!(complete > 0);
            } else {
                assert_eq!(complete, 0);
            }
            worker.start(spatial.clone(), key.clone(), context.clone(), false)?;
            let expected = signature(drain(&mut worker, false)?)?;
            assert_eq!(worker.shared.complete_leaf_cells.load(Ordering::Relaxed), 0);
            assert_eq!(
                actual, expected,
                "Whole-leaf traversal changed geometry or label preparation"
            );
        }
        Ok(())
    }

    #[test]
    fn delivery_batches_draws_within_two_bounded_messages() -> Result<(), Box<dyn Error>> {
        let worker = Worker::new()?;
        let (channel, receiver) = mpsc::sync_channel(2);
        let mut sender = PacketSender::new(channel);
        let key = request();
        let tile = TileInfo {
            id: 1,
            rect: key.viewport,
            size: [256, 256],
            preview: false,
            grid: [0, 0],
            phase: [0.0, 0.0],
            density: 1.0,
            crop: Rect::from_min_size(Pos2::ZERO, Vec2::splat(256.0)),
        };
        assert!(deliver(
            &worker.shared,
            &mut sender,
            0,
            Ok(Packet::Begin {
                generation: 0,
                tile,
                layers: key.styles.into_iter().collect(),
                first_group: true,
                reuse: Arc::new(AtomicBool::new(false)),
            })
        ));
        let mut begin = receiver.try_recv()?;
        assert!(
            matches!(begin.result.take(), Some(Ok(packets)) if matches!(packets.first(), Some(Packet::Begin { .. })))
        );
        let mesh = Arc::new(MeshData {
            id: resource_id()?,
            vertices: Vec::new(),
            indices: Vec::new(),
            glyph: None,
        });
        for _ in 0..2000 {
            assert!(deliver(
                &worker.shared,
                &mut sender,
                0,
                Ok(Packet::Draw {
                    generation: 0,
                    tile_id: 1,
                    layer: LayerKey {
                        layer: 1,
                        datatype: 0
                    },
                    mesh: mesh.clone(),
                    instances: vec![
                        Instance {
                            affine: [1.0, 0.0, 0.0, 1.0],
                            translation: [0.0; 2]
                        };
                        100
                    ],
                    array: None,
                    shape_count: 100,
                })
            ));
        }
        let mut first = receiver.try_recv()?;
        assert!(first.bytes <= PACKET_LIMIT);
        let first_packets = first.result.take().ok_or("Missing first delivery")??;
        assert!(first_packets.len() > 100);
        assert!(deliver(
            &worker.shared,
            &mut sender,
            0,
            Ok(Packet::Finish {
                generation: 0,
                tile_id: 1,
                last_group: true,
                complete: true,
            })
        ));
        let mut second = receiver.try_recv()?;
        assert!(second.bytes <= PACKET_LIMIT);
        let second_packets = second.result.take().ok_or("Missing second delivery")??;
        assert_eq!(first_packets.len() + second_packets.len(), 2000);
        let mut finish = receiver.try_recv()?;
        assert!(
            matches!(finish.result.take(), Some(Ok(packets)) if matches!(packets.first(), Some(Packet::Finish { .. })))
        );
        Ok(())
    }

    #[test]
    fn integer_pixel_pan_preserves_world_tile_grid() -> Result<(), Box<dyn Error>> {
        let scene = fixture(rectangle())?;
        let context = egui::Context::default();
        let mut worker = Worker::new()?;
        worker.start(scene.clone(), request(), context.clone(), false)?;
        let first: BTreeMap<_, _> = drain(&mut worker, false)?
            .into_iter()
            .filter_map(|packet| match packet {
                Packet::Begin { tile, .. } => Some((tile.grid, tile)),
                _ => None,
            })
            .collect();
        let mut moved = request();
        moved.camera.center.x += 10.0;
        worker.start(scene, moved, context, false)?;
        let second: BTreeMap<_, _> = drain(&mut worker, false)?
            .into_iter()
            .filter_map(|packet| match packet {
                Packet::Begin { tile, .. } => Some((tile.grid, tile)),
                _ => None,
            })
            .collect();
        assert_eq!(first.len(), second.len());
        for (grid, tile) in first {
            let moved = second.get(&grid).ok_or("Missing reusable tile")?;
            assert_eq!(tile.phase, moved.phase);
            assert_eq!(tile.density, moved.density);
            assert!((tile.rect.left() - 10.0 - moved.rect.left()).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn deep_zoom_projects_relative_coordinates_and_retains_grid_phase() -> Result<(), Box<dyn Error>>
    {
        let source = 1e9;
        let camera_center = source - 1e-6;
        let viewport = Rect::from_min_size(Pos2::ZERO, Vec2::splat(256.0));
        for density in [1e6, 1e8, 1e9] {
            let camera = Camera {
                center: DoublePoint::new(camera_center, 0.0),
                pixels_per_unit: density,
            };
            let projection = Projection::new(
                Transformation::default(),
                camera,
                viewport,
                1.0,
                DoublePoint::default(),
            );
            let projected = projection.apply(
                DoublePoint::new(-source, 0.0),
                DoublePoint::new(source * 2.0, 0.0),
            );
            let expected = 128.0 + (source - camera_center) * density;
            assert!(
                (projected.x - expected).abs() < PIXEL_ERROR,
                "Deep-zoom point shifted by {} pixels",
                projected.x - expected
            );
            let grid = GridAxis::new(camera_center, density, 128.0)?;
            let expected_phase = (-(source - camera_center) * density - 128.0).rem_euclid(1.0);
            assert!(
                (grid.phase - expected_phase).abs() < 1e-8,
                "Grid lost fractional pixel phase"
            );
        }
        for angle in [0.0, std::f64::consts::PI / 7.0] {
            let transformation = Transformation {
                translation: DoublePoint::new(1e9, 1e9),
                magnification: 1e-8,
                angle,
                reflected: false,
            };
            let camera = Camera {
                center: transformation.translation,
                pixels_per_unit: 1e9,
            };
            let projection = Projection::new(
                transformation,
                camera,
                viewport,
                1.0,
                DoublePoint::default(),
            );
            let expected = transformation.apply_vector(DoublePoint::new(1.0, 1.0));
            for origin in [DoublePoint::new(0.0, 100.0), DoublePoint::new(1e9, -1e9)] {
                let point = projection.apply(origin, DoublePoint::new(1.0, 1.0).minus(origin));
                assert!((point.x - (128.0 + expected.x * 1e9)).abs() < PIXEL_ERROR);
                assert!((point.y - (128.0 - expected.y * 1e9)).abs() < PIXEL_ERROR);
            }
        }
        let scene = fixture(Geometry::Box {
            minimum: Point::new(-1_000_000_000, 0),
            maximum: Point::new(1_000_000_000, 10),
        })?;
        let mut key = request();
        key.viewport = viewport;
        key.camera = Camera {
            center: DoublePoint::new(camera_center, 5.0),
            pixels_per_unit: 1e6,
        };
        let mut worker = Worker::new()?;
        worker.start(scene, key, egui::Context::default(), false)?;
        let packets = drain(&mut worker, false)?;
        let mut tiles = BTreeMap::new();
        let mut found_edge = false;
        let expected = 128.0 + (source - camera_center) * 1e6;
        for packet in packets {
            match packet {
                Packet::Begin { tile, .. } => {
                    tiles.insert(tile.id, tile);
                }
                Packet::Draw {
                    tile_id,
                    mesh,
                    instances,
                    ..
                } => {
                    let tile = tiles.get(&tile_id).ok_or("Missing fallback tile")?;
                    let instance = instances.first().ok_or("Missing fallback instance")?;
                    assert_eq!(instance.affine, [1.0, 0.0, 0.0, 1.0]);
                    for vertex in mesh.vertices.iter().filter(|vertex| vertex.kind == 0) {
                        let position = f64::from(vertex.position[0]) + f64::from(tile.rect.left());
                        if (position - expected).abs() < PIXEL_ERROR {
                            found_edge = true;
                        }
                    }
                }
                Packet::Finish { .. } => {}
            }
        }
        assert!(found_edge, "Fallback lost the precise right boundary");
        Ok(())
    }
}

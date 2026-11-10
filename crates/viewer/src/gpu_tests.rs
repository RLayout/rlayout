use super::gpu::GpuRendererHandle;
use super::*;
use egui_wgpu::wgpu;
use rlayout_core::{Array, Cell, Instance, PathEnds, Point, Shape};
use std::future::Future;
use std::io;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::task::{Context, Poll, Wake, Waker};

fn finish_future<F: Future>(future: F) -> F::Output {
    struct ThreadWake(std::thread::Thread);
    impl Wake for ThreadWake {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending => std::thread::park(),
        }
    }
}

fn renderer() -> Result<(MutexGuard<'static, ()>, &'static GpuRendererHandle), Box<dyn Error>> {
    static SERIAL: Mutex<()> = Mutex::new(());
    static RENDERER: OnceLock<Result<GpuRendererHandle, String>> = OnceLock::new();
    let guard = match SERIAL.lock() {
        Ok(guard) => guard,
        Err(error) => {
            eprintln!("An earlier GPU test failed; continuing independent checks");
            error.into_inner()
        }
    };
    let renderer = RENDERER.get_or_init(|| {
        finish_future(GpuRendererHandle::headless()).map_err(|error| error.to_string())
    });
    match renderer {
        Ok(renderer) => Ok((guard, renderer)),
        Err(error) => {
            Err(io::Error::other(format!("Required headless GPU unavailable: {error}")).into())
        }
    }
}

fn context() -> egui::Context {
    let context = egui::Context::default();
    let mut output = context.run_ui(egui::RawInput::default(), |_| {});
    output.textures_delta.clear();
    context
}

fn scene(cells: Vec<Cell>) -> Result<Arc<Scene>, Box<dyn Error>> {
    Ok(Arc::new(Scene::prepare(Layout::new(
        "gpu-correctness".into(),
        1e-9,
        0.001,
        cells,
    )?)?))
}

fn key(scene: &Scene, size: Vec2) -> FrameKey {
    FrameKey {
        camera: Camera::default(),
        viewport: Rect::from_min_size(Pos2::new(17.0, 29.0), size),
        pixels_per_point: 1.0,
        selected: scene.layout().top_cells().first().copied(),
        depth: u16::MAX,
        styles: scene
            .layout()
            .layers()
            .iter()
            .map(|layer| {
                (
                    *layer,
                    LayerStyle {
                        visible: true,
                        color: Color32::WHITE,
                    },
                )
            })
            .collect(),
        text_pass: 0,
    }
}

fn layer(number: u16) -> LayerKey {
    LayerKey {
        layer: number,
        datatype: 0,
    }
}

fn polygon(number: u16, coordinates: &[(i64, i64)]) -> Shape {
    Shape {
        layer: layer(number),
        geometry: Geometry::Polygon(
            coordinates
                .iter()
                .map(|(x, y)| Point::new(*x, *y))
                .collect(),
        ),
    }
}

struct Contour {
    layer: LayerKey,
    points: Vec<DoublePoint>,
}

fn contour(shape: &Shape, transformation: Transformation) -> Result<Contour, Box<dyn Error>> {
    let Geometry::Polygon(points) = &shape.geometry else {
        return Err("Expected polygon fixture".into());
    };
    Ok(Contour {
        layer: shape.layer,
        points: points
            .iter()
            .map(|point| transformation.apply(point.to_double()))
            .collect(),
    })
}

fn inside(points: &[DoublePoint], point: DoublePoint) -> bool {
    let mut winding = 0_i32;
    for (first, second) in points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .take(points.len())
    {
        let side =
            (second.x - first.x) * (point.y - first.y) - (second.y - first.y) * (point.x - first.x);
        if first.y <= point.y && second.y > point.y && side > 0.0 {
            winding += 1;
        } else if first.y > point.y && second.y <= point.y && side < 0.0 {
            winding -= 1;
        }
    }
    winding != 0
}

fn edge_distance(points: &[DoublePoint], point: DoublePoint) -> f64 {
    points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .take(points.len())
        .map(|(first, second)| {
            let direction = second.minus(*first);
            let relative = point.minus(*first);
            let length_squared = direction.x * direction.x + direction.y * direction.y;
            let fraction = if length_squared == 0.0 {
                0.0
            } else {
                ((relative.x * direction.x + relative.y * direction.y) / length_squared)
                    .clamp(0.0, 1.0)
            };
            let displacement = point.minus(first.plus(direction.times(fraction)));
            displacement.x.hypot(displacement.y)
        })
        .fold(f64::INFINITY, f64::min)
}

fn check_interiors(image: &egui::ColorImage, key: &FrameKey, contours: &[Contour]) {
    let scale = key.camera.pixels_per_unit * f64::from(key.pixels_per_point);
    let mut checked = 0_usize;
    let mut occupied = 0_usize;
    for (index, actual) in image.pixels.iter().enumerate() {
        let column = index % image.size[0];
        let row = index / image.size[0];
        let point = key.camera.center.plus(DoublePoint::new(
            (column as f64 + 0.5 - f64::from(key.viewport.width() * key.pixels_per_point) * 0.5)
                / scale,
            -(row as f64 + 0.5 - f64::from(key.viewport.height() * key.pixels_per_point) * 0.5)
                / scale,
        ));
        if contours
            .iter()
            .any(|contour| edge_distance(&contour.points, point) * scale <= 1.0)
        {
            continue;
        }
        let mut expected = [0_u16; 4];
        for (layer, style) in &key.styles {
            if !style.visible
                || !contours
                    .iter()
                    .any(|contour| contour.layer == *layer && inside(&contour.points, point))
            {
                continue;
            }
            let source = style.color.to_array();
            let inverse = 255 - u16::from(source[3]);
            for (destination, source) in expected.iter_mut().zip(source) {
                *destination = (u16::from(source) + *destination * inverse / 255).min(255);
            }
        }
        occupied += usize::from(expected[3] != 0);
        for (actual, expected) in actual.to_array().into_iter().zip(expected) {
            assert!(
                u16::from(actual).abs_diff(expected) <= 2,
                "Interior pixel ({column}, {row}) differs: actual {actual}, expected {expected}"
            );
        }
        checked += 1;
    }
    assert!(
        checked > image.pixels.len() / 2,
        "Too few independently checked pixels"
    );
    assert!(occupied > 0, "Fixture has no tested filled interior");
}

fn alpha_centroid(image: &egui::ColorImage) -> Result<DoublePoint, Box<dyn Error>> {
    let mut weighted = DoublePoint::default();
    let mut total = 0.0;
    for (index, pixel) in image.pixels.iter().enumerate() {
        let alpha = f64::from(pixel.a());
        weighted.x += (index % image.size[0]) as f64 * alpha;
        weighted.y += (index / image.size[0]) as f64 * alpha;
        total += alpha;
    }
    if total == 0.0 {
        return Err("Missing visible glyph centroid".into());
    }
    Ok(weighted.times(1.0 / total))
}

struct CallbackSurface {
    device: wgpu::Device,
    queue: wgpu::Queue,
    format: wgpu::TextureFormat,
    renderer: egui_wgpu::Renderer,
    context: egui::Context,
    repaint_delay: Duration,
}

impl CallbackSurface {
    fn new(renderer: &GpuRendererHandle) -> Self {
        let (device, queue, format) = renderer.test_device_queue();
        Self {
            renderer: egui_wgpu::Renderer::new(
                &device,
                format,
                egui_wgpu::RendererOptions {
                    dithering: false,
                    ..Default::default()
                },
            ),
            device,
            queue,
            format,
            context: context(),
            repaint_delay: Duration::MAX,
        }
    }

    fn frame(
        &mut self,
        session: &super::gpu::RenderSession,
        key: &FrameKey,
    ) -> Result<egui::ColorImage, Box<dyn Error>> {
        self.frame_with(key, |ui| {
            let painter = ui.ctx().layer_painter(egui::LayerId::new(
                egui::Order::Middle,
                egui::Id::new("GPU callback test"),
            ));
            painter.rect_filled(key.viewport, 0.0, Color32::BLACK);
            painter.add(session.callback(key.viewport, key.camera));
        })
    }

    fn viewer_frame(
        &mut self,
        viewer: &mut Viewer,
        key: &FrameKey,
    ) -> Result<egui::ColorImage, Box<dyn Error>> {
        self.frame_with(key, |ui| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| viewer.canvas(ui));
        })
    }

    fn frame_with(
        &mut self,
        key: &FrameKey,
        paint: impl FnMut(&mut egui::Ui),
    ) -> Result<egui::ColorImage, Box<dyn Error>> {
        self.context.set_pixels_per_point(key.pixels_per_point);
        let size = [
            (key.viewport.max.x * key.pixels_per_point).ceil() as u32,
            (key.viewport.max.y * key.pixels_per_point).ceil() as u32,
        ];
        let target = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("GPU callback test output"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let mut output = self.context.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_max(Pos2::ZERO, key.viewport.max)),
                ..Default::default()
            },
            paint,
        );
        self.repaint_delay = output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .map_or(Duration::MAX, |viewport| viewport.repaint_delay);
        for (identifier, deltas) in &output.textures_delta.set {
            for delta in deltas {
                self.renderer
                    .update_texture(&self.device, &self.queue, *identifier, delta);
            }
        }
        let freed = output.textures_delta.free.clone();
        output.textures_delta.clear();
        let jobs = self
            .context
            .tessellate(std::mem::take(&mut output.shapes), output.pixels_per_point);
        let descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: size,
            pixels_per_point: key.pixels_per_point,
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GPU callback test"),
            });
        let buffers = self.renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &jobs,
            &descriptor,
        );
        {
            let attachments = [Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("GPU callback display"),
                    color_attachments: &attachments,
                    ..Default::default()
                })
                .forget_lifetime();
            self.renderer.render(&mut pass, &jobs, &descriptor);
        }
        let bytes_per_row = (size[0] * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU callback test readback"),
            size: u64::from(bytes_per_row) * u64::from(size[1]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(size[1]),
                },
            },
            wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
        );
        self.queue
            .submit(buffers.into_iter().chain([encoder.finish()]));
        let (sender, receiver) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if let Err(error) = sender.send(result) {
                    eprintln!("Delivering callback readback: {error}");
                }
            });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        receiver.recv_timeout(Duration::from_secs(10))??;
        let bytes = readback.slice(..).get_mapped_range()?;
        let mut pixels = Vec::with_capacity((size[0] * size[1]) as usize);
        for row in bytes.chunks(bytes_per_row as usize).take(size[1] as usize) {
            for [first, green, third, alpha] in row.as_chunks::<4>().0.iter().take(size[0] as usize)
            {
                let (red, blue) = match self.format {
                    wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
                        (*first, *third)
                    }
                    wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                        (*third, *first)
                    }
                    _ => return Err("Unsupported callback test target format".into()),
                };
                pixels.push(Color32::from_rgba_premultiplied(red, *green, blue, *alpha));
            }
        }
        drop(bytes);
        readback.unmap();
        for identifier in freed {
            self.renderer.free_texture(&identifier);
        }
        Ok(egui::ColorImage::new(
            [size[0] as usize, size[1] as usize],
            pixels,
        ))
    }

    fn complete(
        &mut self,
        session: &super::gpu::RenderSession,
        key: &FrameKey,
    ) -> Result<egui::ColorImage, Box<dyn Error>> {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let image = self.frame(session, key)?;
            let status = session.status();
            if let Some(error) = status.error {
                return Err(io::Error::other(error).into());
            }
            if status.complete {
                return Ok(image);
            }
            if Instant::now() >= deadline {
                return Err("GPU callback did not complete".into());
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

fn red_centroid(image: &egui::ColorImage) -> Result<DoublePoint, Box<dyn Error>> {
    let mut total = 0.0;
    let mut center = DoublePoint::default();
    for (index, pixel) in image.pixels.iter().enumerate() {
        if pixel.r() > 200 && pixel.g() < 40 && pixel.b() < 40 {
            center.x += (index % image.size[0]) as f64;
            center.y += (index / image.size[0]) as f64;
            total += 1.0;
        }
    }
    if total == 0.0 {
        return Err("GPU callback displayed no layout geometry".into());
    }
    Ok(center.times(1.0 / total))
}

#[test]
fn gpu_callback_reprojects_and_replaces_transparent_tiles_without_ghosting()
-> Result<(), Box<dyn Error>> {
    let (_guard, renderer) = renderer()?;
    let scene = scene(vec![Cell::new(
        "TOP".into(),
        vec![polygon(1, &[(-35, -10), (-15, -10), (-15, 10), (-35, 10)])],
        vec![],
    )])?;
    let mut request = key(&scene, Vec2::new(160.0, 96.0));
    request.viewport = Rect::from_min_size(Pos2::ZERO, request.viewport.size());
    request
        .styles
        .get_mut(&layer(1))
        .ok_or("Missing layer")?
        .color = Color32::RED;
    let mut session = renderer.create_session(scene)?;
    let mut surface = CallbackSurface::new(renderer);
    session.request(request.clone(), surface.context.clone(), false)?;
    let initial = surface.complete(&session, &request)?;
    let initial_center = red_centroid(&initial)?;
    let initial_status = session.status();
    assert!(initial_status.complete && initial_status.has_image);
    assert!(initial_status.upload_bytes > 0);
    request.camera.center.x += 10.0;
    let moving = surface.frame(&session, &request)?;
    let moving_center = red_centroid(&moving)?;
    assert!(
        (moving_center.x - initial_center.x + 10.0).abs() <= 1.0,
        "Cached imagery did not move with the camera: {initial_center:?} -> {moving_center:?}"
    );
    assert!(session.status().reprojected_tiles > 0);
    session.request(request.clone(), surface.context.clone(), true)?;
    surface.frame(&session, &request)?;
    session.request(request.clone(), surface.context.clone(), false)?;
    let settled = surface.complete(&session, &request)?;
    let settled_center = red_centroid(&settled)?;
    assert!((settled_center.x - moving_center.x).abs() <= 1.0);
    let settled_status = session.status();
    assert!(settled_status.complete);
    assert_eq!(settled_status.reprojected_tiles, 0);
    assert_eq!(settled_status.tile_count, initial_status.tile_count);
    assert!(settled_status.upload_bytes > 1024);
    request
        .styles
        .get_mut(&layer(1))
        .ok_or("Missing layer")?
        .visible = false;
    session.request(request.clone(), surface.context.clone(), false)?;
    let cleared = surface.complete(&session, &request)?;
    assert!(
        cleared.pixels.iter().all(|pixel| pixel.r() < 100),
        "Transparent replacement retained hidden geometry"
    );
    let status = session.status();
    assert!(status.cpu_bytes <= 128 * 1024 * 1024);
    assert!(status.gpu_bytes <= 256 * 1024 * 1024);
    assert!(status.in_flight_frames <= 2);
    Ok(())
}

#[test]
fn gpu_saturated_queue_still_reprojects_retained_imagery() -> Result<(), Box<dyn Error>> {
    let (_guard, renderer) = renderer()?;
    let scene = scene(vec![Cell::new(
        "TOP".into(),
        vec![polygon(1, &[(-35, -10), (-15, -10), (-15, 10), (-35, 10)])],
        vec![],
    )])?;
    let mut request = key(&scene, Vec2::new(160.0, 96.0));
    request.viewport = Rect::from_min_size(Pos2::ZERO, request.viewport.size());
    request
        .styles
        .get_mut(&layer(1))
        .ok_or("Missing layer")?
        .color = Color32::RED;
    let mut session = renderer.create_session(scene)?;
    let mut surface = CallbackSurface::new(renderer);
    session.request(request.clone(), surface.context.clone(), false)?;
    let initial = red_centroid(&surface.complete(&session, &request)?)?;
    let saturated = session.test_saturate_queue();
    request.camera.center.x += 12.0;
    session.request(request.clone(), surface.context.clone(), true)?;
    let moving = red_centroid(&surface.frame(&session, &request)?)?;
    assert!((moving.x - initial.x + 12.0).abs() <= 1.0);
    let status = session.status();
    assert_eq!(status.in_flight_frames, 2);
    assert_eq!(status.presented_camera, Some(request.camera));
    assert!(status.max_frame_upload_bytes <= 8 * 1024 * 1024);
    assert!(status.max_frame_draws <= 2048);
    assert!(status.max_frame_triangles <= 250_000);
    drop(saturated);
    session.request(request.clone(), surface.context.clone(), false)?;
    red_centroid(&surface.complete(&session, &request)?)?;
    assert!(session.status().complete);
    Ok(())
}

#[test]
fn gpu_callback_cropped_tiles_refine_exposed_regions_and_reuse_contained_views()
-> Result<(), Box<dyn Error>> {
    let (_guard, renderer) = renderer()?;
    let rectangle = polygon(65, &[(10, -12), (170, -12), (170, 12), (10, 12)]);
    let mut shapes = vec![
        rectangle.clone(),
        Shape {
            layer: layer(1),
            geometry: Geometry::Path(Path {
                points: vec![Point::new(-210, -28), Point::new(210, -28)],
                width: 0,
                absolute_width: false,
                ends: PathEnds::Flush,
            }),
        },
    ];
    for number in 2..65 {
        shapes.push(polygon(
            number,
            &[(5000, 5000), (5010, 5000), (5010, 5010), (5000, 5010)],
        ));
    }
    let scene = scene(vec![Cell::new("TOP".into(), shapes, vec![])])?;
    let mut request = key(&scene, Vec2::new(160.0, 96.0));
    request.viewport = Rect::from_min_size(Pos2::ZERO, request.viewport.size());
    let mut session = renderer.create_session(scene)?;
    let mut surface = CallbackSurface::new(renderer);
    let verify = |image: &egui::ColorImage, request: &FrameKey| {
        let mut line_pixels = 0;
        for (index, pixel) in image.pixels.iter().enumerate() {
            let position = Pos2::new(
                (index % image.size[0]) as f32 + 0.5,
                (index / image.size[0]) as f32 + 0.5,
            );
            let world = request.camera.screen_to_world(position, request.viewport);
            if world.x > 12.0 && world.x < 168.0 && world.y.abs() < 10.0 {
                assert_eq!(*pixel, Color32::WHITE, "Missing layer 65 at {world:?}");
            } else if world.y > -20.0 && world.y < -14.0 {
                assert_eq!(
                    *pixel,
                    Color32::from_rgb(17, 21, 27),
                    "Retained crop ghost at {world:?}"
                );
            }
            if (world.y + 28.0).abs() <= 1.0 && pixel.r() > 200 {
                line_pixels += 1;
            }
        }
        assert!(
            line_pixels >= image.size[0] / 2,
            "Missing crossing narrow line"
        );
    };
    session.request(request.clone(), surface.context.clone(), false)?;
    verify(&surface.complete(&session, &request)?, &request);
    request.camera.center.x += 80.0;
    let retained = surface.frame(&session, &request)?;
    assert!(retained.pixels.contains(&Color32::WHITE));
    session.request(request.clone(), surface.context.clone(), false)?;
    verify(&surface.complete(&session, &request)?, &request);
    assert!(session.status().upload_bytes > 1024);
    request.camera.center.x -= 80.0;
    session.request(request.clone(), surface.context.clone(), false)?;
    verify(&surface.complete(&session, &request)?, &request);
    request.viewport = Rect::from_min_size(Pos2::ZERO, Vec2::new(80.0, 48.0));
    session.request(request.clone(), surface.context.clone(), false)?;
    let contained = surface.complete(&session, &request)?;
    assert!(session.status().upload_bytes <= 1024);
    for (index, pixel) in contained.pixels.iter().enumerate() {
        let world = request.camera.screen_to_world(
            Pos2::new(
                (index % contained.size[0]) as f32 + 0.5,
                (index / contained.size[0]) as f32 + 0.5,
            ),
            request.viewport,
        );
        if world.x > 12.0 && world.x < 38.0 && world.y.abs() < 10.0 {
            assert_eq!(*pixel, Color32::WHITE);
        }
    }
    Ok(())
}

#[test]
fn gpu_callback_latest_request_recovers_from_obsolete_invalid_viewport()
-> Result<(), Box<dyn Error>> {
    let (_guard, renderer) = renderer()?;
    let scene = scene(vec![Cell::new(
        "TOP".into(),
        vec![polygon(1, &[(-10, -10), (10, -10), (10, 10), (-10, 10)])],
        vec![],
    )])?;
    let mut request = key(&scene, Vec2::splat(80.0));
    request.viewport = Rect::from_min_size(Pos2::ZERO, request.viewport.size());
    request
        .styles
        .get_mut(&layer(1))
        .ok_or("Missing layer")?
        .color = Color32::RED;
    let mut session = renderer.create_session(scene)?;
    let mut surface = CallbackSurface::new(renderer);
    let mut invalid = request.clone();
    invalid.viewport = Rect::from_min_size(Pos2::ZERO, Vec2::splat(20_000.0));
    if let Err(error) = session.request(invalid, surface.context.clone(), false) {
        assert!(!error.to_string().is_empty());
    }
    session.request(request.clone(), surface.context.clone(), false)?;
    let image = surface.complete(&session, &request)?;
    red_centroid(&image)?;
    assert!(session.status().error.is_none());
    for _ in 0..3 {
        surface.frame(&session, &request)?;
        assert!(
            session.status().error.is_none(),
            "Obsolete failure replaced current status"
        );
    }
    Ok(())
}

#[test]
fn gpu_canvas_schedules_exact_refinement_after_input_settles() -> Result<(), Box<dyn Error>> {
    let (_guard, renderer) = renderer()?;
    let scene = Arc::try_unwrap(scene(vec![Cell::new(
        "TOP".into(),
        vec![polygon(1, &[(-10, -10), (10, -10), (10, 10), (-10, 10)])],
        vec![],
    )])?)
    .map_err(|_| "Unexpected shared test scene")?;
    let mut viewer = Viewer::new(scene, renderer.clone())?;
    viewer.fit_pending = false;
    viewer
        .styles
        .get_mut(&layer(1))
        .ok_or("Missing layer")?
        .color = Color32::RED;
    let mut request = key(&viewer.scene, Vec2::new(160.0, 96.0));
    request.viewport = Rect::from_min_size(Pos2::ZERO, request.viewport.size());
    let mut surface = CallbackSurface::new(renderer);
    surface.viewer_frame(&mut viewer, &request)?;
    assert!(!viewer.exact_requested);
    assert!(
        surface.repaint_delay <= Duration::from_millis(150),
        "Canvas did not schedule its settled refinement"
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let image = surface.viewer_frame(&mut viewer, &request)?;
        if let Some(error) = &viewer.error {
            return Err(io::Error::other(error.clone()).into());
        }
        if viewer.rendering_complete() {
            assert!(viewer.exact_requested);
            assert_eq!(viewer.render_status().reprojected_tiles, 0);
            red_centroid(&image)?;
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err("Canvas never completed exact refinement".into());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn gpu_nonzero_fills_layer_union_and_fractional_tile_seams() -> Result<(), Box<dyn Error>> {
    let (_guard, renderer) = renderer()?;
    let shapes = vec![
        polygon(
            1,
            &[
                (-240, -105),
                (240, -105),
                (240, 105),
                (-240, 105),
                (-240, -105),
                (-30, -25),
                (-30, 25),
                (30, 25),
                (30, -25),
                (-30, -25),
                (-240, -105),
            ],
        ),
        polygon(1, &[(-160, -70), (160, 70), (-160, 70), (160, -70)]),
        polygon(2, &[(0, 60), (40, -60), (-60, 20), (60, 20), (-40, -60)]),
        polygon(2, &[(-70, -40), (-70, -40), (90, -40), (90, 40), (-70, 40)]),
    ];
    let transformation = Transformation {
        angle: 0.13,
        reflected: true,
        ..Default::default()
    };
    let contours = shapes
        .iter()
        .map(|shape| contour(shape, transformation))
        .collect::<Result<Vec<_>, _>>()?;
    let scene = scene(vec![
        Cell::new("LEAF".into(), shapes, vec![]),
        Cell::new(
            "TOP".into(),
            vec![],
            vec![Instance {
                cell: CellId::from_index(0),
                transformation,
                array: Array::default(),
            }],
        ),
    ])?;
    let mut request = key(&scene, Vec2::new(520.0, 280.0));
    request
        .styles
        .get_mut(&layer(1))
        .ok_or("Missing first layer")?
        .color = Color32::from_rgba_unmultiplied(225, 70, 35, 113);
    request
        .styles
        .get_mut(&layer(2))
        .ok_or("Missing second layer")?
        .color = Color32::from_rgba_unmultiplied(30, 160, 235, 147);
    let session = renderer.create_session(scene)?;
    for phase in [0.0, 0.25, 0.73] {
        request.camera.center = DoublePoint::new(phase, -phase * 0.7);
        let image = finish_future(session.capture(request.clone(), context()))?;
        assert_eq!(image.image.size, [520, 280]);
        check_interiors(&image.image, &request, &contours);
    }
    request
        .styles
        .get_mut(&layer(2))
        .ok_or("Missing second layer")?
        .visible = false;
    check_interiors(
        &finish_future(session.capture(request.clone(), context()))?.image,
        &request,
        &contours,
    );
    Ok(())
}

#[test]
fn gpu_supported_degenerate_and_cancelling_contours_have_empty_fill() -> Result<(), Box<dyn Error>>
{
    let (_guard, renderer) = renderer()?;
    let shapes = vec![
        polygon(1, &[(0, 0), (5, 0), (10, 0)]),
        polygon(1, &[(0, 0), (0, 0), (0, 0)]),
        polygon(1, &[(0, 0), (10, 0), (0, 0), (0, 10)]),
        polygon(
            1,
            &[
                (0, 0),
                (10, 0),
                (10, 10),
                (0, 10),
                (0, 0),
                (0, 10),
                (10, 10),
                (10, 0),
            ],
        ),
    ];
    let scene = scene(vec![Cell::new("TOP".into(), shapes, vec![])])?;
    let mut request = key(&scene, Vec2::splat(80.0));
    let session = renderer.create_session(scene)?;
    for phase in [0.0, 0.37] {
        request.camera.center = DoublePoint::new(phase, -phase);
        let image = finish_future(session.capture(request.clone(), context()))?;
        assert!(
            image.image.pixels.iter().all(|pixel| pixel.a() == 0),
            "Supported empty contour produced visible coverage at phase {phase}"
        );
    }
    Ok(())
}

#[test]
fn gpu_coincident_arrays_preserve_counts_without_repeated_layer_alpha() -> Result<(), Box<dyn Error>>
{
    let (_guard, renderer) = renderer()?;
    let square = polygon(1, &[(-20, -20), (20, -20), (20, 20), (-20, 20)]);
    let make_scene = |columns, rows| {
        scene(vec![
            Cell::new("LEAF".into(), vec![square.clone()], vec![]),
            Cell::new(
                "TOP".into(),
                vec![],
                vec![Instance {
                    cell: CellId::from_index(0),
                    transformation: Transformation::default(),
                    array: Array {
                        columns,
                        rows,
                        column_step: DoublePoint::default(),
                        row_step: DoublePoint::default(),
                    },
                }],
            ),
        ])
    };
    let single = make_scene(1, 1)?;
    let repeated = make_scene(11, 7)?;
    let mut request = key(&single, Vec2::splat(80.0));
    request
        .styles
        .get_mut(&layer(1))
        .ok_or("Missing layer")?
        .color = Color32::from_rgba_unmultiplied(220, 80, 30, 97);
    let single_session = renderer.create_session(single)?;
    let expected = finish_future(single_session.capture(request.clone(), context()))?;
    let repeated_session = renderer.create_session(repeated)?;
    let actual = finish_future(repeated_session.capture(request.clone(), context()))?;
    assert_eq!(expected.shape_count, 1);
    assert_eq!(actual.shape_count, 77);
    assert_eq!(actual.image.pixels, expected.image.pixels);
    request.depth = 0;
    let hidden = finish_future(repeated_session.capture(request, context()))?;
    assert_eq!(hidden.shape_count, 0);
    assert!(hidden.image.pixels.iter().all(|pixel| pixel.a() == 0));
    Ok(())
}

#[test]
fn gpu_device_loss_retains_document_and_requires_restart() -> Result<(), Box<dyn Error>> {
    let (_guard, _) = renderer()?;
    let renderer = finish_future(GpuRendererHandle::headless())?;
    let scene = scene(vec![Cell::new(
        "TOP".into(),
        vec![polygon(1, &[(-10, -10), (10, -10), (10, 10), (-10, 10)])],
        vec![],
    )])?;
    let request = key(&scene, Vec2::splat(80.0));
    let mut session = renderer.create_session(scene.clone())?;
    session.request(request.clone(), context(), false)?;
    let (device, _, _) = renderer.test_device_queue();
    device.destroy();
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let status = session.status();
        if let Some(error) = status.error {
            assert!(error.to_lowercase().contains("restart"), "{error}");
            assert!(!status.complete);
            assert_eq!(scene.layout().shape_count(), 1);
            assert!(session.request(request, context(), false).is_err());
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err("Device loss did not reach the viewer".into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn gpu_discarded_submissions_release_limits_and_do_not_validate_cached_tiles()
-> Result<(), Box<dyn Error>> {
    let (_guard, renderer) = renderer()?;
    let shape = polygon(1, &[(-20, -20), (20, -20), (20, 20), (-20, 20)]);
    let contours = vec![contour(&shape, Transformation::default())?];
    let scene = scene(vec![Cell::new("TOP".into(), vec![shape], vec![])])?;
    let request = key(&scene, Vec2::splat(80.0));
    let session = renderer.create_session(scene)?;
    session.test_discard_request(request.clone(), context())?;
    assert_eq!(session.status().in_flight_frames, 0);
    assert!(!session.status().complete);
    let image = finish_future(session.capture(request.clone(), context()))?;
    assert!(image.statistics.complete);
    check_interiors(&image.image, &request, &contours);
    Ok(())
}

#[test]
fn gpu_large_array_resumes_across_upload_buffer_leases() -> Result<(), Box<dyn Error>> {
    let (_guard, renderer) = renderer()?;
    let scene = scene(vec![
        Cell::new(
            "LEAF".into(),
            vec![polygon(1, &[(0, 0), (1, 0), (1, 1), (0, 1)])],
            vec![],
        ),
        Cell::new(
            "TOP".into(),
            vec![],
            vec![Instance {
                cell: CellId::from_index(0),
                transformation: Transformation::default(),
                array: Array {
                    columns: 1000,
                    rows: 1000,
                    column_step: DoublePoint::new(0.01, 0.0),
                    row_step: DoublePoint::new(0.0, 0.01),
                },
            }],
        ),
    ])?;
    let request = key(&scene, Vec2::splat(48.0));
    let session = renderer.create_session(scene)?;
    let rendered = finish_future(session.capture(request.clone(), context()))?;
    assert!(rendered.statistics.submitted_triangles > 250_000);
    assert!(rendered.statistics.draw_calls >= 8);
    assert!(rendered.statistics.max_frame_upload_bytes <= 8 * 1024 * 1024);
    assert!(rendered.statistics.max_frame_draws <= 2048);
    assert!(rendered.statistics.max_frame_triangles <= 250_000);
    assert!(rendered.statistics.cpu_bytes <= 128 * 1024 * 1024);
    assert!(rendered.statistics.gpu_bytes <= 256 * 1024 * 1024);
    for (index, pixel) in rendered.image.pixels.iter().enumerate() {
        let position = request.viewport.min
            + Vec2::new(
                (index % rendered.image.size[0]) as f32 + 0.5,
                (index / rendered.image.size[0]) as f32 + 0.5,
            );
        let world = request.camera.screen_to_world(position, request.viewport);
        if world.x > 1.0 && world.x < 9.0 && world.y > 1.0 && world.y < 9.0 {
            assert_eq!(*pixel, Color32::WHITE);
        } else if world.x < -2.0 || world.x > 13.0 || world.y < -2.0 || world.y > 13.0 {
            assert_eq!(pixel.a(), 0);
        }
    }
    Ok(())
}

#[test]
fn gpu_skew_arrays_preserve_reflected_instances_and_screen_phase() -> Result<(), Box<dyn Error>> {
    let (_guard, renderer) = renderer()?;
    let shape = polygon(1, &[(-6, -4), (6, -4), (6, 4), (-6, 4)]);
    let transformation = Transformation {
        translation: DoublePoint::new(-40.0, -30.0),
        angle: 0.31,
        reflected: true,
        magnification: 1.1,
    };
    let array = Array {
        columns: 5,
        rows: 4,
        column_step: DoublePoint::new(22.0, 7.0),
        row_step: DoublePoint::new(-9.0, 18.0),
    };
    let mut contours = Vec::new();
    for row in 0..array.rows {
        for column in 0..array.columns {
            let translated = Transformation {
                translation: transformation
                    .translation
                    .plus(array.column_step.times(f64::from(column)))
                    .plus(array.row_step.times(f64::from(row))),
                ..transformation
            };
            contours.push(contour(&shape, translated)?);
        }
    }
    let scene = scene(vec![
        Cell::new("LEAF".into(), vec![shape], vec![]),
        Cell::new(
            "TOP".into(),
            vec![],
            vec![Instance {
                cell: CellId::from_index(0),
                transformation,
                array,
            }],
        ),
    ])?;
    let mut request = key(&scene, Vec2::new(240.0, 200.0));
    request.camera.center = DoublePoint::new(0.37, -0.71);
    request
        .styles
        .get_mut(&layer(1))
        .ok_or("Missing array layer")?
        .color = Color32::from_rgba_unmultiplied(230, 70, 30, 123);
    let session = renderer.create_session(scene)?;
    for pixels_per_point in [1.0, 2.0] {
        request.pixels_per_point = pixels_per_point;
        let image = finish_future(session.capture(request.clone(), context()))?;
        if pixels_per_point == 1.0 {
            assert_eq!(image.shape_count, 20);
        }
        check_interiors(&image.image, &request, &contours);
    }
    Ok(())
}

#[test]
fn gpu_multiple_array_bases_preserve_each_transform() -> Result<(), Box<dyn Error>> {
    let (_guard, renderer) = renderer()?;
    let shape = polygon(1, &[(-3, -2), (3, -2), (3, 2), (-3, 2)]);
    let array = Array {
        columns: 3,
        rows: 2,
        column_step: DoublePoint::new(13.0, 2.0),
        row_step: DoublePoint::new(-2.0, 13.0),
    };
    let mut instances = Vec::new();
    let mut contours = Vec::new();
    for index in 0..12 {
        let transformation = Transformation {
            translation: DoublePoint::new(
                f64::from(index % 4) * 45.0 - 80.0,
                f64::from(index / 4) * 45.0 - 50.0,
            ),
            angle: f64::from(index % 3) * 0.37,
            reflected: index % 2 == 0,
            magnification: 1.0,
        };
        for row in 0..array.rows {
            for column in 0..array.columns {
                contours.push(contour(
                    &shape,
                    Transformation {
                        translation: transformation
                            .translation
                            .plus(array.column_step.times(f64::from(column)))
                            .plus(array.row_step.times(f64::from(row))),
                        ..transformation
                    },
                )?);
            }
        }
        instances.push(Instance {
            cell: CellId::from_index(0),
            transformation,
            array,
        });
    }
    let scene = scene(vec![
        Cell::new("LEAF".into(), vec![shape], vec![]),
        Cell::new("TOP".into(), vec![], instances),
    ])?;
    let mut request = key(&scene, Vec2::new(240.0, 190.0));
    request
        .styles
        .get_mut(&layer(1))
        .ok_or("Missing layer")?
        .color = Color32::from_rgba_unmultiplied(230, 70, 30, 123);
    let session = renderer.create_session(scene)?;
    let image = finish_future(session.capture(request.clone(), context()))?;
    assert_eq!(image.shape_count, 72);
    check_interiors(&image.image, &request, &contours);
    let mut shifted = request.clone();
    shifted.viewport = shifted.viewport.translate(Vec2::new(0.25, 0.0));
    let fractional = finish_future(session.capture(shifted.clone(), context()))?;
    assert!(fractional.statistics.submitted_triangles > 0);
    check_interiors(&fractional.image, &shifted, &contours);
    shifted.viewport = shifted.viewport.translate(Vec2::new(1.0, 0.0));
    let integer = finish_future(session.capture(shifted.clone(), context()))?;
    assert_eq!(integer.statistics.submitted_triangles, 0);
    check_interiors(&integer.image, &shifted, &contours);
    Ok(())
}

#[test]
fn gpu_absolute_paths_keep_width_under_reflected_magnification() -> Result<(), Box<dyn Error>> {
    let (_guard, renderer) = renderer()?;
    let path = Shape {
        layer: layer(1),
        geometry: Geometry::Path(Path {
            points: vec![Point::new(-20, 0), Point::new(20, 0)],
            width: 6,
            absolute_width: true,
            ends: PathEnds::Flush,
        }),
    };
    let mut contours = Vec::new();
    let mut instances = Vec::new();
    for (magnification, translation) in [
        (0.2, DoublePoint::new(-35.0, -20.0)),
        (2.0, DoublePoint::new(35.0, 20.0)),
    ] {
        let transformation = Transformation {
            translation,
            magnification,
            angle: 0.6,
            reflected: true,
        };
        let first = transformation.apply(DoublePoint::new(-20.0, 0.0));
        let second = transformation.apply(DoublePoint::new(20.0, 0.0));
        let direction = second.minus(first);
        let normal =
            DoublePoint::new(-direction.y, direction.x).times(3.0 / direction.x.hypot(direction.y));
        contours.push(Contour {
            layer: layer(1),
            points: vec![
                first.plus(normal),
                second.plus(normal),
                second.minus(normal),
                first.minus(normal),
            ],
        });
        instances.push(Instance {
            cell: CellId::from_index(0),
            transformation,
            array: Array::default(),
        });
    }
    let scene = scene(vec![
        Cell::new("LEAF".into(), vec![path], vec![]),
        Cell::new("TOP".into(), vec![], instances),
    ])?;
    let request = key(&scene, Vec2::new(200.0, 160.0));
    let session = renderer.create_session(scene)?;
    let image = finish_future(session.capture(request.clone(), context()))?;
    assert_eq!(image.shape_count, 2);
    check_interiors(&image.image, &request, &contours);
    Ok(())
}

#[test]
fn gpu_narrow_polygons_and_zero_width_lines_remain_visible_at_physical_resolution()
-> Result<(), Box<dyn Error>> {
    let (_guard, renderer) = renderer()?;
    let shapes = vec![
        polygon(1, &[(0, -50), (1, -50), (1, 50), (0, 50)]),
        Shape {
            layer: layer(2),
            geometry: Geometry::Path(Path {
                points: vec![Point::new(-50, -25), Point::new(50, -25)],
                width: 0,
                absolute_width: false,
                ends: PathEnds::Flush,
            }),
        },
    ];
    let scene = scene(vec![Cell::new("TOP".into(), shapes, vec![])])?;
    let mut request = key(&scene, Vec2::splat(80.0));
    request.camera.pixels_per_unit = 0.3;
    request
        .styles
        .get_mut(&layer(1))
        .ok_or("Missing polygon layer")?
        .color = Color32::RED;
    request
        .styles
        .get_mut(&layer(2))
        .ok_or("Missing line layer")?
        .color = Color32::BLUE;
    let session = renderer.create_session(scene)?;
    for pixels_per_point in [1.0, 2.0] {
        request.pixels_per_point = pixels_per_point;
        for phase in [0.0, 0.8, 1.7] {
            request.camera.center.x = phase;
            let image = finish_future(session.capture(request.clone(), context()))?.image;
            let mut red_rows = std::collections::BTreeSet::new();
            let mut blue_columns = std::collections::BTreeSet::new();
            for (index, pixel) in image.pixels.iter().enumerate() {
                if pixel.r() > 200 && pixel.b() < 10 {
                    red_rows.insert(index / image.size[0]);
                }
                if pixel.b() > 200 && pixel.r() < 10 {
                    blue_columns.insert(index % image.size[0]);
                }
            }
            assert!(
                red_rows.len() >= (25.0 * pixels_per_point) as usize,
                "Narrow polygon disappeared at phase {phase}, DPI {pixels_per_point}"
            );
            assert!(
                blue_columns.len() >= (25.0 * pixels_per_point) as usize,
                "Zero-width line disappeared at phase {phase}, DPI {pixels_per_point}"
            );
        }
    }
    Ok(())
}

#[test]
fn gpu_narrow_extensions_of_wide_polygons_remain_visible() -> Result<(), Box<dyn Error>> {
    let (_guard, renderer) = renderer()?;
    let scene = scene(vec![Cell::new(
        "TOP".into(),
        vec![polygon(
            1,
            &[
                (0, 0),
                (40, 0),
                (40, 19),
                (100, 19),
                (100, 20),
                (40, 20),
                (40, 40),
                (0, 40),
            ],
        )],
        vec![],
    )])?;
    let mut request = key(&scene, Vec2::splat(80.0));
    request.camera.center = DoublePoint::new(50.0, 20.0);
    request.camera.pixels_per_unit = 0.3;
    let session = renderer.create_session(scene)?;
    for phase in [0.0, 0.8, 1.7] {
        request.camera.center.y = 20.0 + phase;
        let image = finish_future(session.capture(request.clone(), context()))?.image;
        assert!(
            image
                .pixels
                .iter()
                .enumerate()
                .any(|(index, pixel)| index % image.size[0] == 49 && pixel.a() != 0),
            "The one-unit wire attached to a wide plate disappeared at phase {phase}"
        );
    }
    Ok(())
}

#[test]
fn gpu_rectangle_coverage_is_independent_of_contour_start_and_winding() -> Result<(), Box<dyn Error>>
{
    let (_guard, renderer) = renderer()?;
    let orders = [
        vec![(-1, -50), (0, -50), (0, 50), (-1, 50)],
        vec![(0, 50), (0, -50), (-1, -50), (-1, 50)],
        vec![(0, -50), (0, 50), (-1, 50), (-1, -50)],
    ];
    let mut expected = None;
    for points in orders {
        let scene = scene(vec![
            Cell::new("LEAF".into(), vec![polygon(1, &points)], vec![]),
            Cell::new(
                "TOP".into(),
                vec![],
                vec![Instance {
                    cell: CellId::from_index(0),
                    transformation: Transformation {
                        angle: 0.6,
                        reflected: true,
                        ..Transformation::default()
                    },
                    array: Array::default(),
                }],
            ),
        ])?;
        let mut request = key(&scene, Vec2::splat(80.0));
        request.camera.pixels_per_unit = 0.3;
        request.pixels_per_point = 2.0;
        let session = renderer.create_session(scene)?;
        let image = finish_future(session.capture(request, context()))?.image;
        if let Some(expected) = &expected {
            assert_eq!(&image.pixels, expected);
        } else {
            expected = Some(image.pixels);
        }
    }
    Ok(())
}

#[test]
fn gpu_small_magnified_features_survive_large_instance_translation() -> Result<(), Box<dyn Error>> {
    let (_guard, renderer) = renderer()?;
    let translation = DoublePoint::new(1e9, 1e9);
    let scene = scene(vec![
        Cell::new(
            "LEAF".into(),
            vec![
                polygon(
                    1,
                    &[
                        (0, 100),
                        (1_000_000_000, 100),
                        (1_000_000_000, 101),
                        (0, 101),
                    ],
                ),
                polygon(1, &[(0, 0), (1, 0), (1, 1), (0, 1)]),
            ],
            vec![],
        ),
        Cell::new(
            "TOP".into(),
            vec![],
            vec![Instance {
                cell: CellId::from_index(0),
                transformation: Transformation {
                    translation,
                    magnification: 1e-8,
                    ..Transformation::default()
                },
                array: Array::default(),
            }],
        ),
    ])?;
    let mut request = key(&scene, Vec2::splat(256.0));
    request.camera.center = translation;
    request.camera.pixels_per_unit = 1e9;
    let session = renderer.create_session(scene)?;
    let image = finish_future(session.capture(request, context()))?.image;
    for row in 120..127 {
        for column in 129..137 {
            assert_eq!(image.pixels[row * image.size[0] + column], Color32::WHITE);
        }
    }
    Ok(())
}

#[test]
fn gpu_deep_zoom_rebases_before_physical_pixel_scaling() -> Result<(), Box<dyn Error>> {
    let (_guard, renderer) = renderer()?;
    let boundary = 1_000_000_000;
    let scene = scene(vec![Cell::new(
        "TOP".into(),
        vec![polygon(
            1,
            &[(0, -10), (boundary, -10), (boundary, 10), (0, 10)],
        )],
        vec![],
    )])?;
    let mut request = key(&scene, Vec2::new(512.0, 96.0));
    request.camera.center.x = f64::from_bits((boundary as f64).to_bits() - 1);
    let session = renderer.create_session(scene)?;
    for density in [1e6, 1e8, 1e9] {
        request.camera.pixels_per_unit = density;
        let image = finish_future(session.capture(request.clone(), context()))?.image;
        let edge = 256.0 + (boundary as f64 - request.camera.center.x) * density;
        for (index, pixel) in image.pixels.iter().enumerate() {
            let column = (index % image.size[0]) as f64 + 0.5;
            if (column - edge).abs() > 1.0 {
                assert_eq!(
                    pixel.a() != 0,
                    column < edge,
                    "Deep zoom lost interior precision at density {density}, column {column}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn gpu_layer_groups_preserve_order_across_sixty_four_masks() -> Result<(), Box<dyn Error>> {
    let (_guard, renderer) = renderer()?;
    let shapes = (1..=70)
        .map(|number| polygon(number, &[(-30, -30), (30, -30), (30, 30), (-30, 30)]))
        .collect::<Vec<_>>();
    let contours = shapes
        .iter()
        .map(|shape| contour(shape, Transformation::default()))
        .collect::<Result<Vec<_>, _>>()?;
    let scene = scene(vec![Cell::new("TOP".into(), shapes, vec![])])?;
    let mut request = key(&scene, Vec2::splat(128.0));
    for (index, style) in request.styles.values_mut().enumerate() {
        style.color = Color32::from_rgba_unmultiplied(
            (index * 31 % 256) as u8,
            (index * 67 % 256) as u8,
            (index * 113 % 256) as u8,
            57,
        );
    }
    let session = renderer.create_session(scene)?;
    let image = finish_future(session.capture(request.clone(), context()))?;
    check_interiors(&image.image, &request, &contours);
    Ok(())
}

#[test]
fn gpu_rotated_thin_triangles_keep_one_pixel_coverage() -> Result<(), Box<dyn Error>> {
    let (_guard, renderer) = renderer()?;
    let transformation = Transformation {
        angle: std::f64::consts::FRAC_PI_4,
        ..Transformation::default()
    };
    let scene = scene(vec![
        Cell::new(
            "LEAF".into(),
            vec![polygon(1, &[(0, 0), (100, 100), (100, 101)])],
            vec![],
        ),
        Cell::new(
            "TOP".into(),
            vec![],
            vec![Instance {
                cell: CellId::from_index(0),
                transformation,
                array: Array::default(),
            }],
        ),
    ])?;
    let mut request = key(&scene, Vec2::splat(80.0));
    request.camera.pixels_per_unit = 0.2;
    request.camera.center = transformation.apply(DoublePoint::new(50.0, 50.0));
    let session = renderer.create_session(scene)?;
    for pixels_per_point in [1.0, 2.0] {
        request.pixels_per_point = pixels_per_point;
        for phase in [0.0, 0.8, 1.7] {
            request.camera.center.x = phase;
            let image = finish_future(session.capture(request.clone(), context()))?.image;
            let rows = image
                .pixels
                .iter()
                .enumerate()
                .filter(|(_, pixel)| pixel.a() != 0)
                .map(|(index, _)| index / image.size[0])
                .collect::<std::collections::BTreeSet<_>>();
            assert!(
                rows.len() >= (24.0 * pixels_per_point) as usize,
                "Rotated triangle disappeared at phase {phase}, DPI {pixels_per_point}"
            );
        }
    }
    Ok(())
}

#[test]
fn gpu_wide_cells_preserve_small_features_at_extreme_coordinates() -> Result<(), Box<dyn Error>> {
    let (_guard, renderer) = renderer()?;
    for origin in [i64::from(i32::MIN), i64::from(i32::MAX) - 1_000_000_000] {
        let shapes = vec![
            polygon(
                1,
                &[
                    (origin, origin),
                    (origin + 1_000_000_000, origin),
                    (origin + 1_000_000_000, origin + 4),
                    (origin, origin + 4),
                ],
            ),
            polygon(
                2,
                &[
                    (origin + 999_999_997, origin + 1),
                    (origin + 999_999_998, origin + 1),
                    (origin + 999_999_998, origin + 3),
                    (origin + 999_999_997, origin + 3),
                ],
            ),
        ];
        let contours = shapes
            .iter()
            .map(|shape| contour(shape, Transformation::default()))
            .collect::<Result<Vec<_>, _>>()?;
        let scene = scene(vec![Cell::new("TOP".into(), shapes, vec![])])?;
        let mut request = key(&scene, Vec2::new(128.0, 96.0));
        request.camera.center =
            DoublePoint::new((origin + 999_999_998) as f64 + 0.17, origin as f64 + 2.23);
        request.camera.pixels_per_unit = 16.0;
        request.pixels_per_point = 2.0;
        request
            .styles
            .get_mut(&layer(1))
            .ok_or("Missing background layer")?
            .color = Color32::RED;
        request
            .styles
            .get_mut(&layer(2))
            .ok_or("Missing narrow layer")?
            .color = Color32::BLUE;
        let session = renderer.create_session(scene)?;
        let image = finish_future(session.capture(request.clone(), context()))?;
        check_interiors(&image.image, &request, &contours);
        assert!(image.image.pixels.contains(&Color32::BLUE));
    }
    Ok(())
}

#[test]
fn gpu_labels_survive_atlas_growth_rotation_reflection_and_dpi() -> Result<(), Box<dyn Error>> {
    let (_guard, renderer) = renderer()?;
    let shape = Shape {
        layer: layer(1),
        geometry: Geometry::Label(Label {
            text: "Original label".into(),
            transformation: Transformation {
                angle: 0.3,
                reflected: true,
                ..Default::default()
            },
            horizontal_alignment: 1,
            vertical_alignment: 1,
        }),
    };
    let mut unaligned_shape = shape.clone();
    if let Geometry::Label(label) = &mut unaligned_shape.geometry {
        label.horizontal_alignment = 0;
        label.vertical_alignment = 0;
    }
    let aligned_scene = scene(vec![Cell::new("TOP".into(), vec![shape], vec![])])?;
    let unaligned_scene = scene(vec![Cell::new("TOP".into(), vec![unaligned_shape], vec![])])?;
    let mut request = key(&aligned_scene, Vec2::new(280.0, 140.0));
    let context = context();
    let logical_size = context.fonts_mut(|fonts| {
        fonts
            .layout_no_wrap(
                "Original label".into(),
                egui::FontId::proportional(12.0),
                Color32::WHITE,
            )
            .size()
    });
    let session = renderer.create_session(aligned_scene)?;
    let unaligned_session = renderer.create_session(unaligned_scene)?;
    let mut occupied_by_dpi = Vec::new();
    for pixels_per_point in [1.0, 2.0] {
        request.pixels_per_point = pixels_per_point;
        let before = finish_future(session.capture(request.clone(), context.clone()))?;
        assert_eq!(before.shape_count, 1);
        let occupied = before
            .image
            .pixels
            .iter()
            .filter(|pixel| pixel.a() > 0)
            .count();
        assert!(occupied > 40, "Label did not produce visible glyphs");
        occupied_by_dpi.push(occupied);
        let unaligned = finish_future(unaligned_session.capture(request.clone(), context.clone()))?;
        let actual_displacement =
            alpha_centroid(&before.image)?.minus(alpha_centroid(&unaligned.image)?);
        let (sine, cosine) = 0.3_f64.sin_cos();
        let expected_displacement = DoublePoint::new(
            -cosine * f64::from(logical_size.x) * 0.5 + sine * f64::from(logical_size.y) * 0.5,
            sine * f64::from(logical_size.x) * 0.5 + cosine * f64::from(logical_size.y) * 0.5,
        )
        .times(f64::from(pixels_per_point));
        assert!(
            (actual_displacement.x - expected_displacement.x).abs() <= 2.0
                && (actual_displacement.y - expected_displacement.y).abs() <= 2.0,
            "Label alignment/transform differs: {actual_displacement:?}, expected {expected_displacement:?}"
        );
        context.fonts_mut(|fonts| {
            for size in [18.0, 33.0, 57.0, 95.0] {
                fonts.layout_no_wrap(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 !@#$%&()[]{}"
                        .into(),
                    egui::FontId::proportional(size),
                    Color32::WHITE,
                );
            }
        });
        let after = finish_future(session.capture(request.clone(), context.clone()))?;
        assert_eq!(
            before.image.pixels, after.image.pixels,
            "Atlas growth changed a cached label"
        );
    }
    assert!(
        occupied_by_dpi
            .get(1)
            .copied()
            .ok_or("Missing high-DPI result")?
            > occupied_by_dpi
                .first()
                .copied()
                .ok_or("Missing standard-DPI result")?
                * 2
    );
    Ok(())
}

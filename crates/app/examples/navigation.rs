use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;
use rlayout_viewer::{Camera, GpuRendererHandle, Scene, Viewer};

#[derive(Default)]
struct Measurements {
    intervals: Vec<f64>,
    feedback: Vec<f64>,
    cpu_seconds: Vec<f32>,
    pixels: egui::Vec2,
    visible_shapes: usize,
    blank_frames: usize,
    moving_frames: usize,
    first_image_seconds: Option<f64>,
    initial_complete_seconds: Option<f64>,
    settled_seconds: Option<f64>,
    gpu_bytes: usize,
    error: Option<String>,
}

struct Navigation {
    viewer: Viewer,
    measurements: Arc<Mutex<Measurements>>,
    started: Instant,
    moving_frames: usize,
    previous: Option<Instant>,
    pending_feedback: Option<(Camera, Instant)>,
    settling_started: Option<Instant>,
    sized: bool,
}

impl eframe::App for Navigation {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        if !self.sized {
            let physical = ui.max_rect().size() * ui.ctx().pixels_per_point();
            if (physical - egui::Vec2::new(1280.0, 800.0)).length() > 1.0 {
                if self.started.elapsed() > Duration::from_secs(10) {
                    let error = format!(
                        "Benchmark resize failed: physical={physical:?}, pixels_per_point={}, viewport={:?}",
                        ui.ctx().pixels_per_point(),
                        ui.ctx().input(|input| input.viewport().inner_rect)
                    );
                    if let Ok(mut measurements) = self.measurements.lock() {
                        measurements.error = Some(error);
                    }
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                    return;
                }
                ui.ctx()
                    .send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
                ui.ctx()
                    .send_viewport_cmd(egui::ViewportCommand::Maximized(false));
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::InnerSize(
                    egui::Vec2::new(1280.0, 800.0) / ui.ctx().pixels_per_point(),
                ));
                ui.label("Preparing the benchmark viewport…");
                ui.ctx().request_repaint();
                return;
            }
            self.sized = true;
            self.started = Instant::now();
        }
        let now = Instant::now();
        let status = self.viewer.render_status();
        let initial_complete = self
            .measurements
            .lock()
            .is_ok_and(|measurements| measurements.initial_complete_seconds.is_some());
        let moving = initial_complete && self.moving_frames < 210;
        if let Some((camera, issued)) = self.pending_feedback
            && status.presented_camera == Some(camera)
            && let Some(presented) = status
                .last_presented_at
                .filter(|presented| *presented >= issued)
        {
            match self.measurements.lock() {
                Ok(mut measurements) => measurements
                    .feedback
                    .push(presented.duration_since(issued).as_secs_f64()),
                Err(error) => eprintln!("Recording visible feedback: {error}"),
            }
            self.pending_feedback = None;
        }
        if moving {
            self.moving_frames += 1;
            let direction = if self.moving_frames % 120 < 60 {
                0.5
            } else {
                -0.5
            };
            self.viewer.camera.pan(egui::Vec2::splat(direction));
            if (91..=150).contains(&self.moving_frames) {
                self.viewer.camera.pixels_per_unit *= 1.005;
            } else if self.moving_frames > 150 {
                self.viewer.camera.pixels_per_unit /= 1.005;
            }
            self.pending_feedback = Some((self.viewer.camera, now));
        }
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| {
                self.viewer.canvas(ui);
                let status = self.viewer.render_status();
                let mut measurements = match self.measurements.lock() {
                    Ok(measurements) => measurements,
                    Err(error) => {
                        eprintln!("Recording benchmark: {error}");
                        ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                        return;
                    }
                };
                measurements.pixels = ui.max_rect().size() * ui.ctx().pixels_per_point();
                measurements.visible_shapes = status.shape_count;
                measurements.gpu_bytes = status.gpu_bytes;
                if let Some(error) = self.viewer.error.clone() {
                    measurements.error = Some(error);
                }
                if status.has_geometry && measurements.first_image_seconds.is_none() {
                    measurements.first_image_seconds = Some(self.started.elapsed().as_secs_f64());
                }
                if self.viewer.rendering_complete()
                    && measurements.initial_complete_seconds.is_none()
                {
                    measurements.initial_complete_seconds =
                        Some(self.started.elapsed().as_secs_f64());
                }
                if moving {
                    measurements.moving_frames += 1;
                    measurements.blank_frames += usize::from(!status.has_image);
                    if self.moving_frames > 30 {
                        if let Some(previous) = self.previous {
                            measurements
                                .intervals
                                .push(now.duration_since(previous).as_secs_f64());
                        }
                        if let Some(cpu_seconds) = frame.info().cpu_usage {
                            measurements.cpu_seconds.push(cpu_seconds);
                        }
                    }
                }
                if self.moving_frames >= 210 {
                    let settling = self.settling_started.get_or_insert_with(Instant::now);
                    if self.viewer.rendering_complete() {
                        measurements.settled_seconds = Some(settling.elapsed().as_secs_f64());
                        ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                }
                if self.started.elapsed() > Duration::from_secs(180) {
                    measurements.error = Some("Native navigation benchmark timed out".into());
                }
                if measurements.error.is_some() {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
        self.previous = Some(now);
        ui.ctx().request_repaint();
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args_os()
        .nth(1)
        .ok_or("Pass a GDSII fixture path")?;
    let started = Instant::now();
    let loaded = rlayout_gds::read(BufReader::new(File::open(path)?))?;
    let parsing = started.elapsed();
    let scene = Scene::prepare(loaded.layout)?;
    println!(
        "Parsing {:.2} ms; parsing and preparation {:.2} ms; {} stored shapes",
        parsing.as_secs_f64() * 1000.0,
        started.elapsed().as_secs_f64() * 1000.0,
        scene.layout().shape_count()
    );
    for warning in loaded.warnings {
        eprintln!("Byte {}: {}", warning.offset, warning.message);
    }
    let measurements = Arc::new(Mutex::new(Measurements::default()));
    eframe::run_native(
        "RLayout navigation benchmark",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([640.0, 400.0])
                .with_fullscreen(false)
                .with_maximized(false),
            renderer: eframe::Renderer::Wgpu,
            persist_window: false,
            ..Default::default()
        },
        Box::new({
            let measurements = measurements.clone();
            move |context| {
                let state = context
                    .wgpu_render_state
                    .as_ref()
                    .ok_or("WGPU renderer is unavailable")?;
                let renderer = GpuRendererHandle::from_render_state(state)?;
                let adapter = renderer.adapter_info();
                println!(
                    "renderer=WGPU adapter={:?} backend={:?}",
                    adapter.name, adapter.backend
                );
                Ok(Box::new(Navigation {
                    viewer: Viewer::new(scene, renderer)?,
                    measurements,
                    started: Instant::now(),
                    moving_frames: 0,
                    previous: None,
                    pending_feedback: None,
                    settling_started: None,
                    sized: false,
                }))
            }
        }),
    )?;
    let measurements = measurements.lock().map_err(|error| error.to_string())?;
    if let Some(error) = &measurements.error {
        return Err(error.clone().into());
    }
    if (measurements.pixels - egui::Vec2::new(1280.0, 800.0)).length() > 1.0 {
        return Err("Native benchmark did not use the required 1280×800 framebuffer".into());
    }
    if measurements.intervals.len() < 180 || measurements.settled_seconds.is_none() {
        return Err("Benchmark window closed before measurement completed".into());
    }
    let mut intervals = measurements.intervals.clone();
    intervals.sort_by(f64::total_cmp);
    let median = intervals.get(intervals.len() / 2).ok_or("Missing median")?;
    let frames_per_second = intervals.len() as f64 / intervals.iter().sum::<f64>();
    let cpu_average = measurements.cpu_seconds.iter().copied().sum::<f32>()
        / measurements.cpu_seconds.len().max(1) as f32;
    let mut feedback = measurements.feedback.clone();
    feedback.sort_by(f64::total_cmp);
    let feedback_p95 = feedback
        .get((feedback.len().saturating_sub(1) * 95) / 100)
        .ok_or("No rendered camera feedback was measured")?;
    println!(
        "physical_pixels={:.0}x{:.0} fps={:.2} median_frame_ms={:.3} average_cpu_ms={:.3} feedback_p95_ms={:.3} feedback_samples={} blank_frames={} moving_frames={} first_image_seconds={:.3} initial_complete_seconds={:.3} settled_seconds={:.3} gpu_bytes={} rendered_shapes={}",
        measurements.pixels.x,
        measurements.pixels.y,
        frames_per_second,
        median * 1000.0,
        cpu_average * 1000.0,
        feedback_p95 * 1000.0,
        feedback.len(),
        measurements.blank_frames,
        measurements.moving_frames,
        measurements
            .first_image_seconds
            .ok_or("No useful image appeared")?,
        measurements
            .initial_complete_seconds
            .ok_or("Initial view did not complete")?,
        measurements
            .settled_seconds
            .ok_or("Final view did not complete")?,
        measurements.gpu_bytes,
        measurements.visible_shapes
    );
    if measurements.blank_frames != 0 {
        return Err("Navigation discarded visible layout imagery".into());
    }
    if feedback.len() < measurements.moving_frames * 9 / 10 {
        return Err("Fewer than 90% of camera updates produced next-frame GPU feedback".into());
    }
    if *feedback_p95 > 1.0 / 30.0 {
        return Err("Visible camera feedback exceeded the two-frame target".into());
    }
    if frames_per_second < 30.0 {
        return Err(format!("Missed the 30 FPS target: {frames_per_second:.1} FPS").into());
    }
    Ok(())
}

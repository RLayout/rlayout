use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use eframe::egui;
use rlayout_viewer::{Scene, Viewer};

#[derive(Default)]
struct Measurements {
    intervals: Vec<f64>,
    cpu_seconds: Vec<f32>,
    pixels: egui::Vec2,
    visible_shapes: usize,
    error: Option<String>,
}

struct Navigation {
    viewer: Viewer,
    measurements: Arc<Mutex<Measurements>>,
    frames: usize,
    previous: Option<Instant>,
}

impl eframe::App for Navigation {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let now = Instant::now();
        self.frames += 1;
        if self.frames > 30 {
            match self.measurements.lock() {
                Ok(mut measurements) => {
                    if let Some(previous) = self.previous {
                        measurements
                            .intervals
                            .push(now.duration_since(previous).as_secs_f64());
                    }
                    if let Some(cpu_seconds) = frame.info().cpu_usage {
                        measurements.cpu_seconds.push(cpu_seconds);
                    }
                }
                Err(error) => {
                    eprintln!("Recording benchmark: {error}");
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                    return;
                }
            }
        }
        self.previous = Some(now);
        let direction = if self.frames % 120 < 60 { 0.5 } else { -0.5 };
        self.viewer.camera.pan(egui::Vec2::splat(direction));
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| {
                self.viewer.canvas(ui);
                match self.measurements.lock() {
                    Ok(mut measurements) => {
                        measurements.pixels = ui.max_rect().size() * ui.ctx().pixels_per_point();
                        measurements.visible_shapes = self.viewer.visible_shape_count();
                        measurements.error = self.viewer.error.clone();
                    }
                    Err(error) => {
                        eprintln!("Recording benchmark: {error}");
                        ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                }
            });
        if self.frames >= 210 {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        } else {
            ui.ctx().request_repaint();
        }
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
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([960.0, 540.0]),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "RLayout navigation benchmark",
        options,
        Box::new({
            let measurements = measurements.clone();
            move |context| {
                let scale = context.egui_ctx.pixels_per_point();
                context
                    .egui_ctx
                    .send_viewport_cmd(egui::ViewportCommand::InnerSize(
                        egui::Vec2::new(1920.0, 1080.0) / scale,
                    ));
                Ok(Box::new(Navigation {
                    viewer: Viewer::new(scene),
                    measurements,
                    frames: 0,
                    previous: None,
                }))
            }
        }),
    )?;
    let measurements = measurements.lock().map_err(|error| error.to_string())?;
    if let Some(error) = &measurements.error {
        return Err(error.clone().into());
    }
    if measurements.intervals.len() < 180 {
        return Err("Benchmark window closed before measurement completed".into());
    }
    let mut intervals = measurements.intervals.clone();
    intervals.sort_by(f64::total_cmp);
    let median = intervals.get(intervals.len() / 2).ok_or("Missing median")?;
    let total: f64 = intervals.iter().sum();
    let frames_per_second = intervals.len() as f64 / total;
    let cpu_average = measurements.cpu_seconds.iter().copied().sum::<f32>()
        / measurements.cpu_seconds.len().max(1) as f32;
    println!(
        "WGPU {:.0} × {:.0}, {} visible shapes: {:.1} FPS, median frame {:.2} ms, average CPU {:.2} ms",
        measurements.pixels.x,
        measurements.pixels.y,
        measurements.visible_shapes,
        frames_per_second,
        median * 1000.0,
        cpu_average * 1000.0
    );
    if frames_per_second < 30.0 {
        return Err(format!("Missed the 30 FPS target: {frames_per_second:.1} FPS").into());
    }
    Ok(())
}

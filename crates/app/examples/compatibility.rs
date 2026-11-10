use rlayout_viewer::{GpuRendererHandle, Scene, Viewer, egui};
use std::{
    error::Error,
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let path = PathBuf::from(
        arguments
            .next()
            .ok_or("Usage: compatibility FILE [OUTPUT_DIRECTORY] [ZOOM]")?,
    );
    let output = arguments.next().map(PathBuf::from);
    let single_zoom = arguments
        .next()
        .map(|value| value.to_string_lossy().parse::<f64>())
        .transpose()?;
    if single_zoom.is_some_and(|zoom| !zoom.is_finite() || zoom <= 0.0) {
        return Err("Zoom must be finite and positive".into());
    }
    let graphics_started = Instant::now();
    let renderer = pollster::block_on(GpuRendererHandle::headless())?;
    let adapter = renderer.adapter_info();
    println!(
        "renderer=WGPU adapter={:?} backend={:?} physical_pixels=1280x800",
        adapter.name, adapter.backend
    );
    println!(
        "gpu_initialization_seconds={:.3}",
        graphics_started.elapsed().as_secs_f64()
    );
    print_resident_memory("initial")?;
    let start = Instant::now();
    let result = rlayout_gds::read(BufReader::new(File::open(&path)?))?;
    println!(
        "load_seconds={:.3} cells={} shapes={} references={} warnings={}",
        start.elapsed().as_secs_f64(),
        result.layout.cells().count(),
        result.layout.shape_count(),
        result
            .layout
            .cells()
            .map(|(_, cell)| cell.instances().len())
            .sum::<usize>(),
        result.warnings.len()
    );
    for warning in &result.warnings {
        println!("warning={warning:?}");
    }
    print_resident_memory("loaded_source")?;
    let prepare = Instant::now();
    let mut viewer = Viewer::new(Scene::prepare(result.layout)?, renderer)?;
    println!("prepare_seconds={:.3}", prepare.elapsed().as_secs_f64());
    print_resident_memory("source_and_indexes")?;
    let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::new(1280.0, 800.0));
    viewer.camera.fit(
        viewer.selected.and_then(|id| viewer.layout().bounds(id)),
        viewport,
    );
    let context = egui::Context::default();
    let mut font_output = context.run_ui(egui::RawInput::default(), |_| {});
    font_output.textures_delta.clear();
    if let Some(output) = &output {
        std::fs::create_dir_all(output)?;
    }
    let fitted_scale = viewer.camera.pixels_per_unit;
    let zooms = single_zoom.map_or_else(|| vec![1.0, 4.0, 16.0], |zoom| vec![zoom]);
    for (index, zoom) in zooms.into_iter().enumerate() {
        viewer.camera.pixels_per_unit = fitted_scale * zoom;
        let render = Instant::now();
        let image = pollster::block_on(viewer.render_image(&context, viewport))?;
        println!(
            "zoom={zoom} render_seconds={:.3} rendered_shapes={} cache_bytes={}",
            render.elapsed().as_secs_f64(),
            image.shape_count,
            image.cache_bytes
        );
        println!(
            "zoom={zoom} first_tile_seconds={:.6} first_useful_seconds={:.6} preparation_seconds={:.6} completed_gpu_seconds={:.6} gpu_completion_latency_sum_seconds={:.6} upload_bytes={} cpu_bytes={} gpu_bytes={} triangle_instances={} draw_calls={} mask_passes={} max_frame_upload_bytes={} max_frame_draws={} max_frame_triangles={}",
            image.statistics.first_tile_seconds.unwrap_or_default(),
            image.statistics.first_useful_seconds.unwrap_or_default(),
            image.statistics.preparation_seconds,
            image.statistics.completed_gpu_seconds.unwrap_or_default(),
            image.statistics.gpu_completion_latency_seconds,
            image.statistics.upload_bytes,
            image.statistics.cpu_bytes,
            image.statistics.gpu_bytes,
            image.statistics.submitted_triangles,
            image.statistics.draw_calls,
            image.statistics.mask_passes,
            image.statistics.max_frame_upload_bytes,
            image.statistics.max_frame_draws,
            image.statistics.max_frame_triangles,
        );
        if let Some(output) = &output {
            let mut file = BufWriter::new(File::create(output.join(format!("view-{index}.ppm")))?);
            writeln!(
                file,
                "P6\n{} {}\n255",
                image.image.size[0], image.image.size[1]
            )?;
            for pixel in image.image.pixels {
                let channels = pixel.to_array();
                file.write_all(&channels[..3])?;
            }
            file.flush()?;
        }
    }
    println!("total_seconds={:.3}", start.elapsed().as_secs_f64());
    Ok(())
}

fn print_resident_memory(stage: &str) -> Result<(), Box<dyn Error>> {
    if let Some(bytes) = resident_memory()? {
        println!("stage={stage} resident_bytes={bytes}");
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn resident_memory() -> Result<Option<u64>, Box<dyn Error>> {
    let output = std::process::Command::new("/bin/ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()?;
    if !output.status.success() {
        return Err(format!("Reading resident memory: {}", output.status).into());
    }
    let kibibytes = std::str::from_utf8(&output.stdout)?.trim().parse::<u64>()?;
    Ok(Some(kibibytes * 1024))
}

#[cfg(target_os = "linux")]
fn resident_memory() -> Result<Option<u64>, Box<dyn Error>> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    let kibibytes = status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .and_then(|value| value.split_whitespace().next())
        .ok_or("Missing resident memory in /proc/self/status")?
        .parse::<u64>()?;
    Ok(Some(kibibytes * 1024))
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn resident_memory() -> Result<Option<u64>, Box<dyn Error>> {
    Ok(None)
}

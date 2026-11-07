use rlayout_viewer::{Scene, Viewer, egui};
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
            .ok_or("Usage: compatibility FILE [OUTPUT_DIRECTORY]")?,
    );
    let output = arguments.next().map(PathBuf::from);
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
    let prepare = Instant::now();
    let mut viewer = Viewer::new(Scene::prepare(result.layout)?);
    println!("prepare_seconds={:.3}", prepare.elapsed().as_secs_f64());
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
    for (index, zoom) in [1.0, 4.0, 16.0].into_iter().enumerate() {
        if index > 0 {
            viewer.camera.pixels_per_unit *= 4.0;
        }
        let render = Instant::now();
        let image = viewer.render_image(&context, viewport)?;
        println!(
            "zoom={zoom} render_seconds={:.3} rendered_shapes={} cache_bytes={}",
            render.elapsed().as_secs_f64(),
            image.shape_count,
            image.cache_bytes
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

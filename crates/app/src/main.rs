use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use eframe::egui;
use rlayout_gds::Diagnostic;
use rlayout_viewer::{Scene, Viewer};

struct LoadedDocument {
    scene: Scene,
    warnings: Vec<Diagnostic>,
    elapsed: Duration,
}
struct LoadMessage {
    generation: u64,
    path: PathBuf,
    result: Result<LoadedDocument, String>,
}
struct Worker {
    generation: u64,
    cancel: Arc<AtomicBool>,
    handle: JoinHandle<()>,
}

struct Cancellable<R> {
    reader: R,
    cancel: Arc<AtomicBool>,
}
impl<R: Read> Read for Cancellable<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(io::Error::other("Loading cancelled"));
        }
        self.reader.read(buffer)
    }
}

fn load(path: &PathBuf, cancel: Arc<AtomicBool>) -> Result<LoadedDocument, String> {
    let start = Instant::now();
    let file = File::open(path).map_err(|error| format!("Opening {}: {error}", path.display()))?;
    let result = rlayout_gds::read(Cancellable {
        reader: BufReader::new(file),
        cancel: cancel.clone(),
    })
    .map_err(|error| error.to_string())?;
    if cancel.load(Ordering::Relaxed) {
        return Err("Loading cancelled".into());
    }
    let scene =
        Scene::prepare(result.layout).map_err(|error| format!("Preparing layout: {error}"))?;
    if cancel.load(Ordering::Relaxed) {
        return Err("Loading cancelled".into());
    }
    Ok(LoadedDocument {
        scene,
        warnings: result.warnings,
        elapsed: start.elapsed(),
    })
}

struct Application {
    viewer: Option<Viewer>,
    sender: mpsc::Sender<LoadMessage>,
    receiver: mpsc::Receiver<LoadMessage>,
    workers: Vec<Worker>,
    generation: u64,
    loading: Option<PathBuf>,
    document_path: Option<PathBuf>,
    warnings: Vec<Diagnostic>,
    error: Option<String>,
    render_error: Option<String>,
    open_dialog: bool,
    warnings_dialog: bool,
    path_text: String,
    status: String,
    focus_path: bool,
}

impl Application {
    fn new(context: &egui::Context, path: Option<PathBuf>) -> Self {
        context.set_visuals(egui::Visuals::dark());
        let (sender, receiver) = mpsc::channel();
        let mut application = Self {
            viewer: None,
            sender,
            receiver,
            workers: Vec::new(),
            generation: 0,
            loading: None,
            document_path: None,
            warnings: Vec::new(),
            error: None,
            render_error: None,
            open_dialog: false,
            warnings_dialog: false,
            path_text: String::new(),
            status: "Open a GDSII file to begin".into(),
            focus_path: false,
        };
        if let Some(path) = path {
            application.start_load(path, context);
        }
        application
    }

    fn start_load(&mut self, path: PathBuf, context: &egui::Context) {
        let Some(generation) = self.generation.checked_add(1) else {
            self.error = Some("Too many loading requests".into());
            return;
        };
        for worker in &self.workers {
            worker.cancel.store(true, Ordering::Relaxed);
        }
        self.generation = generation;
        self.error = None;
        self.loading = Some(path.clone());
        self.status = format!("Loading {}…", path.display());
        let cancel = Arc::new(AtomicBool::new(false));
        let sender = self.sender.clone();
        let context = context.clone();
        let handle = thread::Builder::new()
            .name(format!("gds-loader-{generation}"))
            .spawn({
                let cancel = cancel.clone();
                let path = path.clone();
                move || {
                    let result = load(&path, cancel);
                    if let Err(error) = sender.send(LoadMessage {
                        generation,
                        path,
                        result,
                    }) {
                        eprintln!("Delivering layout loading result: {error}");
                    }
                    context.request_repaint();
                }
            });
        match handle {
            Ok(handle) => self.workers.push(Worker {
                generation,
                cancel,
                handle,
            }),
            Err(error) => {
                self.loading = None;
                self.error = Some(format!("Starting loader: {error}"));
                self.status = "Loading failed".into();
            }
        }
    }

    fn receive_loads(&mut self, context: &egui::Context) {
        loop {
            match self.receiver.try_recv() {
                Ok(message) if message.generation == self.generation => {
                    self.loading = None;
                    match message.result {
                        Ok(document) => {
                            self.status = format!(
                                "Loaded in {:.0} ms",
                                document.elapsed.as_secs_f64() * 1000.0
                            );
                            self.warnings = document.warnings;
                            self.warnings_dialog = !self.warnings.is_empty();
                            self.viewer = Some(Viewer::new(document.scene));
                            context.send_viewport_cmd(egui::ViewportCommand::Title(format!(
                                "{} — RLayout",
                                message.path.display()
                            )));
                            self.document_path = Some(message.path);
                        }
                        Err(error) => {
                            self.status = "Loading failed".into();
                            self.error = Some(error);
                        }
                    }
                }
                Ok(_) => {}
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.error = Some("Loader channel disconnected".into());
                    break;
                }
            }
        }
        let mut index = 0;
        while index < self.workers.len() {
            if self
                .workers
                .get(index)
                .is_some_and(|worker| worker.handle.is_finished())
            {
                let worker = self.workers.swap_remove(index);
                if worker.handle.join().is_err() && worker.generation == self.generation {
                    self.loading = None;
                    self.error = Some("The layout loader stopped unexpectedly".into());
                }
            } else {
                index += 1;
            }
        }
        if !self.workers.is_empty() {
            context.request_repaint_after(Duration::from_millis(50));
        }
    }

    fn show_dialogs(&mut self, context: &egui::Context) {
        if self.open_dialog {
            let mut open = true;
            let mut submitted = false;
            egui::Window::new("Open GDSII")
                .open(&mut open)
                .resizable(false)
                .collapsible(false)
                .show(context, |ui| {
                    ui.label("File path (.gds, .gds2, or gzip-compressed GDS)");
                    let response = ui
                        .add(egui::TextEdit::singleline(&mut self.path_text).desired_width(480.0));
                    if self.focus_path {
                        response.request_focus();
                        self.focus_path = false;
                    }
                    submitted = ui.button("Open").clicked()
                        || (response.lost_focus()
                            && ui.input(|input| input.key_pressed(egui::Key::Enter)));
                });
            self.open_dialog = open;
            if submitted {
                if self.path_text.trim().is_empty() {
                    self.error = Some("Enter a file path".into());
                } else {
                    self.open_dialog = false;
                    self.start_load(PathBuf::from(self.path_text.trim()), context);
                }
            }
        }
        if let Some(error) = self.error.clone() {
            let mut open = true;
            egui::Window::new("Could not open layout")
                .open(&mut open)
                .collapsible(false)
                .show(context, |ui| {
                    ui.set_max_width(560.0);
                    ui.colored_label(egui::Color32::LIGHT_RED, error);
                    if ui.button("Dismiss").clicked() {
                        self.error = None;
                    }
                });
            if !open {
                self.error = None;
            }
        }
        egui::Window::new("Import warnings")
            .open(&mut self.warnings_dialog)
            .show(context, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for warning in &self.warnings {
                        ui.label(format!(
                            "Byte {}{}: {}",
                            warning.offset,
                            warning
                                .cell
                                .as_ref()
                                .map_or(String::new(), |cell| format!(", cell {cell}")),
                            warning.message
                        ));
                    }
                });
            });
    }
}

impl Drop for Application {
    fn drop(&mut self) {
        for worker in &self.workers {
            worker.cancel.store(true, Ordering::Relaxed);
        }
    }
}

impl eframe::App for Application {
    fn logic(&mut self, context: &egui::Context, _frame: &mut eframe::Frame) {
        self.receive_loads(context);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let context = ui.ctx().clone();
        let open_shortcut =
            context.input_mut(|input| input.consume_key(egui::Modifiers::COMMAND, egui::Key::O));
        if open_shortcut {
            self.open_dialog = true;
            self.focus_path = true;
        }
        if !context.egui_wants_keyboard_input()
            && context.input(|input| input.key_pressed(egui::Key::F))
            && let Some(viewer) = &mut self.viewer
        {
            viewer.request_fit();
        }
        let dropped = context.input_mut(|input| {
            input
                .raw
                .dropped_files
                .drain(..)
                .map(|file| file.path().to_path_buf())
                .next_back()
        });
        if let Some(path) = dropped {
            self.start_load(path, &context);
        }
        egui::Panel::top("menu").resizable(false).show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open…  ⌘O").clicked() {
                        self.open_dialog = true;
                        self.focus_path = true;
                        ui.close();
                    }
                    if ui.button("Quit").clicked() {
                        context.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                if ui
                    .add_enabled(self.viewer.is_some(), egui::Button::new("Fit  F"))
                    .clicked()
                    && let Some(viewer) = &mut self.viewer
                {
                    viewer.request_fit();
                }
                if !self.warnings.is_empty()
                    && ui
                        .button(format!("{} warnings", self.warnings.len()))
                        .clicked()
                {
                    self.warnings_dialog = true;
                }
                if let Some(path) = &self.document_path {
                    ui.label(
                        path.file_name()
                            .map_or_else(String::new, |name| name.to_string_lossy().into_owned()),
                    );
                }
            });
        });
        egui::Panel::bottom("status")
            .resizable(false)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if self.loading.is_some() {
                        ui.spinner();
                    }
                    ui.label(&self.status);
                    if let Some(viewer) = &self.viewer {
                        ui.separator();
                        if let Some(cell) = viewer.selected.and_then(|id| viewer.layout().cell(id))
                        {
                            ui.label(cell.name());
                        }
                        ui.label(format!("{} rendered shapes", viewer.visible_shape_count()));
                        if !viewer.rendering_complete() && viewer.error.is_none() {
                            ui.spinner();
                            ui.label("Rendering…");
                        }
                        if let Some(cursor) = viewer.cursor {
                            let microns = viewer.layout().meters_per_unit() * 1e6;
                            ui.label(format!(
                                "x {:.4} μm   y {:.4} μm",
                                cursor.x * microns,
                                cursor.y * microns
                            ));
                        }
                    }
                });
            });
        if let Some(viewer) = &mut self.viewer {
            viewer.panels(ui);
        }
        egui::CentralPanel::default().show(ui, |ui| {
            if let Some(viewer) = &mut self.viewer { viewer.canvas(ui); }
            else {
                ui.centered_and_justified(|ui| { ui.label("Open a GDSII layout with File > Open, ⌘O, or drag and drop.\nWheel: zoom · Middle drag: pan · F: fit"); });
            }
        });
        let render_error = self.viewer.as_ref().and_then(|viewer| viewer.error.clone());
        if render_error != self.render_error {
            if let Some(error) = &render_error {
                self.error = Some(error.clone());
            }
            self.render_error = render_error;
        }
        self.show_dialogs(&context);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args_os().nth(1).map(PathBuf::from);
    let icon =
        eframe::icon_data::from_png_bytes(include_bytes!("../../../assets/window-icon.png"))?;
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([640.0, 400.0])
            .with_icon(icon),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "RLayout",
        options,
        Box::new(move |context| Ok(Box::new(Application::new(&context.egui_ctx, path)))),
    )?;
    Ok(())
}

#[cfg(test)]
#[path = "../../gds/examples/fixture.rs"]
mod fixtures;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> io::Result<Self> {
            static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);
            loop {
                let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
                let directory = std::env::temp_dir()
                    .join(format!("rlayout-test-{}-{sequence}", std::process::id()));
                match std::fs::create_dir(&directory) {
                    Ok(()) => return Ok(Self(directory)),
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(error) => return Err(error),
                }
            }
        }

        fn write(&self, name: &str, bytes: &[u8]) -> io::Result<PathBuf> {
            let path = self.0.join(name);
            std::fs::write(&path, bytes)?;
            Ok(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            if let Err(error) = std::fs::remove_dir_all(&self.0) {
                eprintln!("Removing test directory {}: {error}", self.0.display());
            }
        }
    }

    fn document() -> Result<LoadedDocument, Box<dyn Error>> {
        let result = rlayout_gds::read(fixtures::demo()?.as_slice())?;
        Ok(LoadedDocument {
            scene: Scene::prepare(result.layout)?,
            warnings: result.warnings,
            elapsed: Duration::ZERO,
        })
    }

    #[test]
    fn stale_loading_results_and_errors_keep_the_current_document() -> Result<(), Box<dyn Error>> {
        let context = egui::Context::default();
        let mut application = Application::new(&context, None);
        application.generation = 2;
        application.sender.send(LoadMessage {
            generation: 2,
            path: PathBuf::from("original.gds"),
            result: Ok(document()?),
        })?;
        application.receive_loads(&context);
        application.sender.send(LoadMessage {
            generation: 1,
            path: PathBuf::from("stale.gds"),
            result: Ok(document()?),
        })?;
        application.sender.send(LoadMessage {
            generation: 2,
            path: PathBuf::from("broken.gds"),
            result: Err("Original error".into()),
        })?;
        application.receive_loads(&context);
        assert_eq!(
            application.document_path,
            Some(PathBuf::from("original.gds"))
        );
        assert_eq!(
            application
                .viewer
                .as_ref()
                .map(|viewer| viewer.layout().shape_count()),
            Some(7)
        );
        assert_eq!(application.error.as_deref(), Some("Original error"));
        Ok(())
    }

    #[test]
    fn background_file_loading_finishes_and_reports_missing_files() -> Result<(), Box<dyn Error>> {
        let directory = TestDirectory::new()?;
        let path = directory.write("demo.gds", &fixtures::demo()?)?;
        let context = egui::Context::default();
        let mut application = Application::new(&context, None);
        application.start_load(path, &context);
        let deadline = Instant::now() + Duration::from_secs(5);
        while application.loading.is_some() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
            application.receive_loads(&context);
        }
        assert!(application.loading.is_none());
        assert!(application.error.is_none());
        assert!(application.viewer.is_some());
        application.start_load(directory.0.join("missing.gds"), &context);
        let deadline = Instant::now() + Duration::from_secs(5);
        while application.loading.is_some() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
            application.receive_loads(&context);
        }
        assert!(application.loading.is_none());
        assert!(application.error.is_some());
        assert!(application.viewer.is_some());
        Ok(())
    }

    #[test]
    fn generated_fixtures_load_and_render_complete_layouts() -> Result<(), Box<dyn Error>> {
        let directory = TestDirectory::new()?;
        for (file, bytes, cells, shapes) in [
            ("demo.gds", fixtures::demo()?, 2, 7),
            ("large.gds", fixtures::large()?, 1, 100_000),
        ] {
            let path = directory.write(file, &bytes)?;
            let document =
                load(&path, Arc::new(AtomicBool::new(false))).map_err(io::Error::other)?;
            assert_eq!(document.scene.layout().cells().count(), cells);
            assert_eq!(document.scene.layout().shape_count(), shapes);
            assert!(
                document.warnings.is_empty(),
                "Unexpected warnings: {:?}",
                document.warnings
            );
            let mut viewer = Viewer::new(document.scene);
            assert_eq!(
                viewer
                    .selected
                    .and_then(|id| viewer.layout().cell(id))
                    .map(|cell| cell.name()),
                Some("TOP")
            );
            let context = egui::Context::default();
            let viewport =
                egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::new(1280.0, 800.0));
            viewer.camera.fit(
                viewer.selected.and_then(|id| viewer.layout().bounds(id)),
                viewport,
            );
            let mut rendered = None;
            let mut output = context.run_ui(egui::RawInput::default(), |ui| {
                rendered = Some(viewer.render(ui.ctx(), viewport));
            });
            output.textures_delta.clear();
            let frame = rendered.ok_or("No rendered fixture frame")??;
            assert!(frame.shape_count > 0);
            assert!(!frame.truncated);
            assert!(frame.meshes.iter().any(|mesh| !mesh.indices.is_empty()));
        }
        Ok(())
    }

    #[derive(Debug)]
    struct DroppedPath(PathBuf);

    impl egui::DroppedFile for DroppedPath {
        fn path(&self) -> &std::path::Path {
            &self.0
        }

        fn bytes(&self) -> Result<Vec<u8>, String> {
            std::fs::read(&self.0).map_err(|error| error.to_string())
        }
    }

    #[test]
    fn dropped_files_start_one_background_load_and_keep_failed_open_recoverable()
    -> Result<(), Box<dyn Error>> {
        let directory = TestDirectory::new()?;
        let path = directory.write("demo.gds", &fixtures::demo()?)?;
        let context = egui::Context::default();
        let mut application = Application::new(&context, None);
        let mut frame = eframe::Frame::_new_kittest();
        for (index, opened_path) in [path.clone(), directory.0.join("missing-drop.gds")]
            .into_iter()
            .enumerate()
        {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::Vec2::new(1280.0, 800.0),
                )),
                dropped_files: vec![Arc::new(DroppedPath(opened_path))],
                ..Default::default()
            };
            let mut output = context.run_ui(input, |ui| {
                eframe::App::ui(&mut application, ui, &mut frame);
            });
            output.textures_delta.clear();
            assert_eq!(application.generation, index as u64 + 1);
            let deadline = Instant::now() + Duration::from_secs(5);
            while application.loading.is_some() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(5));
                application.receive_loads(&context);
            }
            assert!(application.loading.is_none());
            assert_eq!(application.document_path, Some(path.clone()));
            assert!(application.viewer.is_some());
            assert_eq!(application.error.is_some(), index == 1);
        }
        Ok(())
    }
}

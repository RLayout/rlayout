#[cfg(not(target_arch = "wasm32"))]
use std::error::Error;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::io::BufReader;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
#[cfg(not(target_arch = "wasm32"))]
use std::thread::{self, JoinHandle};
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

#[cfg(target_arch = "wasm32")]
mod web;
#[cfg(target_arch = "wasm32")]
pub use web::WebHandle;

use eframe::egui;
use rlayout_gds::Diagnostic;
use rlayout_viewer::{GpuRendererHandle, Scene, Viewer};

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
#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
fn load(path: &Path, cancel: Arc<AtomicBool>) -> Result<LoadedDocument, String> {
    let file = File::open(path).map_err(|error| format!("Opening {}: {error}", path.display()))?;
    load_reader(BufReader::new(file), cancel)
}

fn load_reader(reader: impl Read, cancel: Arc<AtomicBool>) -> Result<LoadedDocument, String> {
    let start = Instant::now();
    let result = rlayout_gds::read(Cancellable {
        reader,
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
    renderer: Option<GpuRendererHandle>,
    sender: mpsc::Sender<LoadMessage>,
    receiver: mpsc::Receiver<LoadMessage>,
    #[cfg(not(target_arch = "wasm32"))]
    workers: Vec<Worker>,
    #[cfg(target_arch = "wasm32")]
    web_cancel: Option<Arc<AtomicBool>>,
    #[cfg(target_arch = "wasm32")]
    web_context: egui::Context,
    generation: u64,
    loading: Option<PathBuf>,
    document_path: Option<PathBuf>,
    warnings: Vec<Diagnostic>,
    error: Option<String>,
    render_error: Option<String>,
    #[cfg(not(target_arch = "wasm32"))]
    open_dialog: bool,
    warnings_dialog: bool,
    #[cfg(not(target_arch = "wasm32"))]
    path_text: String,
    status: String,
    #[cfg(not(target_arch = "wasm32"))]
    focus_path: bool,
}

impl Application {
    fn new(
        context: &egui::Context,
        renderer: Option<GpuRendererHandle>,
        #[cfg(not(target_arch = "wasm32"))] path: Option<PathBuf>,
    ) -> Self {
        context.set_theme(egui::ThemePreference::Dark);
        let (sender, receiver) = mpsc::channel();
        let application = Self {
            viewer: None,
            renderer: renderer.clone(),
            sender,
            receiver,
            #[cfg(not(target_arch = "wasm32"))]
            workers: Vec::new(),
            #[cfg(target_arch = "wasm32")]
            web_cancel: None,
            #[cfg(target_arch = "wasm32")]
            web_context: context.clone(),
            generation: 0,
            loading: None,
            document_path: None,
            warnings: Vec::new(),
            error: None,
            render_error: None,
            #[cfg(not(target_arch = "wasm32"))]
            open_dialog: false,
            warnings_dialog: false,
            #[cfg(not(target_arch = "wasm32"))]
            path_text: String::new(),
            status: if renderer.is_some() {
                "Open a GDSII file to begin".into()
            } else {
                "Initializing renderer…".into()
            },
            #[cfg(not(target_arch = "wasm32"))]
            focus_path: false,
        };
        #[cfg(not(target_arch = "wasm32"))]
        let mut application = application;
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(path) = path {
            application.start_load(path, context);
        }
        application
    }

    fn begin_load(&mut self, path: &Path) -> Option<u64> {
        let Some(generation) = self.generation.checked_add(1) else {
            self.error = Some("Too many loading requests".into());
            return None;
        };
        #[cfg(not(target_arch = "wasm32"))]
        for worker in &self.workers {
            worker.cancel.store(true, Ordering::Relaxed);
        }
        #[cfg(target_arch = "wasm32")]
        if let Some(cancel) = self.web_cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        self.generation = generation;
        self.error = None;
        self.loading = Some(path.to_path_buf());
        self.status = format!("Loading {}…", path.display());
        Some(generation)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn start_load(&mut self, path: PathBuf, context: &egui::Context) {
        let Some(generation) = self.begin_load(&path) else {
            return;
        };
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

    #[cfg(target_arch = "wasm32")]
    fn start_web_read(
        &mut self,
        path: PathBuf,
        read: impl std::future::Future<Output = Result<Vec<u8>, String>> + 'static,
        context: &egui::Context,
    ) {
        let Some(generation) = self.begin_load(&path) else {
            return;
        };
        let cancel = Arc::new(AtomicBool::new(false));
        self.web_cancel = Some(cancel.clone());
        let sender = self.sender.clone();
        let context = context.clone();
        context.request_repaint();
        wasm_bindgen_futures::spawn_local(async move {
            let bytes = read.await;
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            let result = bytes.and_then(|bytes| load_reader(io::Cursor::new(bytes), cancel));
            if let Err(error) = sender.send(LoadMessage {
                generation,
                path,
                result,
            }) {
                web_sys::console::error_1(
                    &format!("Delivering layout loading result: {error}").into(),
                );
            }
            context.request_repaint();
        });
    }

    #[cfg(target_arch = "wasm32")]
    fn start_file(&mut self, file: web_sys::File) {
        let path = PathBuf::from(file.name());
        let context = self.web_context.clone();
        self.start_web_read(path, web::read_file(file), &context);
    }

    fn open(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.open_dialog = true;
            self.focus_path = true;
        }
        #[cfg(target_arch = "wasm32")]
        if let Err(error) = web::open_picker() {
            self.error = Some(error);
        }
    }

    fn receive_loads(&mut self, context: &egui::Context) {
        loop {
            match self.receiver.try_recv() {
                Ok(message) if message.generation == self.generation => {
                    self.loading = None;
                    match message.result {
                        Ok(document) => {
                            let viewer = self
                                .renderer
                                .as_ref()
                                .ok_or_else(|| {
                                    rlayout_viewer::ViewError("Renderer is not initialized".into())
                                })
                                .and_then(|renderer| Viewer::new(document.scene, renderer.clone()));
                            match viewer {
                                Ok(viewer) => {
                                    self.status = format!(
                                        "Loaded in {:.0} ms",
                                        document.elapsed.as_secs_f64() * 1000.0
                                    );
                                    self.warnings = document.warnings;
                                    self.warnings_dialog = !self.warnings.is_empty();
                                    self.viewer = Some(viewer);
                                    self.render_error = None;
                                    context.send_viewport_cmd(egui::ViewportCommand::Title(
                                        format!("{} — RLayout", message.path.display()),
                                    ));
                                    #[cfg(target_arch = "wasm32")]
                                    if let Some(document) =
                                        web_sys::window().and_then(|window| window.document())
                                    {
                                        document.set_title(&format!(
                                            "{} — RLayout",
                                            message.path.display()
                                        ));
                                    }
                                    self.document_path = Some(message.path);
                                }
                                Err(error) => {
                                    self.status = "Rendering initialization failed".into();
                                    self.error = Some(error.to_string());
                                }
                            }
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
        #[cfg(not(target_arch = "wasm32"))]
        {
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
        #[cfg(target_arch = "wasm32")]
        if self.loading.is_some() {
            context.request_repaint_after(Duration::from_millis(50));
        }
    }

    fn show_dialogs(&mut self, context: &egui::Context) {
        #[cfg(not(target_arch = "wasm32"))]
        if self.open_dialog {
            let mut open = true;
            let mut submitted = false;
            egui::Window::new("Open GDSII")
                .open(&mut open)
                .resizable(false)
                .collapsible(false)
                .show(context, |ui| {
                    ui.label("File path (.gds or .gds2, uncompressed)");
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
        #[cfg(not(target_arch = "wasm32"))]
        for worker in &self.workers {
            worker.cancel.store(true, Ordering::Relaxed);
        }
        #[cfg(target_arch = "wasm32")]
        if let Some(cancel) = &self.web_cancel {
            cancel.store(true, Ordering::Relaxed);
        }
    }
}

impl eframe::App for Application {
    #[cfg(target_arch = "wasm32")]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn logic(&mut self, context: &egui::Context, _frame: &mut eframe::Frame) {
        self.receive_loads(context);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let context = ui.ctx().clone();
        if self.renderer.is_none() {
            egui::CentralPanel::default().show(ui, |ui| {
                ui.centered_and_justified(|ui| {
                    ui.label(&self.status);
                });
            });
            return;
        }
        let open_shortcut =
            context.input_mut(|input| input.consume_key(egui::Modifiers::COMMAND, egui::Key::O));
        if open_shortcut {
            self.open();
        }
        if !context.egui_wants_keyboard_input()
            && context.input(|input| input.key_pressed(egui::Key::F))
            && let Some(viewer) = &mut self.viewer
        {
            viewer.request_fit();
        }
        let dropped = context.input_mut(|input| input.raw.dropped_files.drain(..).next_back());
        if let Some(file) = dropped {
            #[cfg(not(target_arch = "wasm32"))]
            self.start_load(file.path().to_path_buf(), &context);
            #[cfg(target_arch = "wasm32")]
            self.start_web_read(
                file.path().to_path_buf(),
                async move { file.bytes_async().await },
                &context,
            );
        }
        egui::Panel::top("menu").resizable(false).show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui
                        .button(format!(
                            "Open…  {}",
                            egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::O)
                                .format(&egui::ModifierNames::NAMES, context.os().is_mac())
                        ))
                        .clicked()
                    {
                        self.open();
                        ui.close();
                    }
                    #[cfg(not(target_arch = "wasm32"))]
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
                        ui.label(format!("~{} rendered shapes", viewer.visible_shape_count()));
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
                ui.centered_and_justified(|ui| { ui.label("Open a GDSII layout with File > Open or drag and drop.\nWheel: zoom · Middle drag: pan · F: fit"); });
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

#[cfg(not(target_arch = "wasm32"))]
pub fn run_native() -> Result<(), Box<dyn Error>> {
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
        Box::new(move |context| {
            let state = context
                .wgpu_render_state
                .as_ref()
                .ok_or("WGPU renderer is unavailable")?;
            let renderer = GpuRendererHandle::from_render_state(state)?;
            Ok(Box::new(Application::new(
                &context.egui_ctx,
                Some(renderer),
                path,
            )))
        }),
    )?;
    Ok(())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "../../gds/examples/fixture.rs"]
mod fixtures;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use rlayout_viewer::ViewError;
    use std::sync::atomic::AtomicU64;

    fn test_renderer() -> Result<GpuRendererHandle, ViewError> {
        static RENDERER: std::sync::OnceLock<Result<GpuRendererHandle, String>> =
            std::sync::OnceLock::new();
        RENDERER
            .get_or_init(|| {
                pollster::block_on(GpuRendererHandle::headless()).map_err(|error| error.to_string())
            })
            .as_ref()
            .cloned()
            .map_err(|error| ViewError(error.clone()))
    }

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
    fn memory_imports_load_layouts_and_report_invalid_or_cancelled_reads()
    -> Result<(), Box<dyn Error>> {
        let bytes = fixtures::demo()?;
        let document = load_reader(io::Cursor::new(&bytes), Arc::new(AtomicBool::new(false)))
            .map_err(io::Error::other)?;
        assert_eq!(document.scene.layout().shape_count(), 7);
        assert!(document.warnings.is_empty());
        let error = load_reader(io::Cursor::new(b"broken"), Arc::new(AtomicBool::new(false)))
            .err()
            .ok_or("Invalid GDS bytes unexpectedly loaded")?;
        assert!(error.contains("byte"));
        let error = load_reader(io::Cursor::new(&bytes), Arc::new(AtomicBool::new(true)))
            .err()
            .ok_or("Cancelled GDS bytes unexpectedly loaded")?;
        assert!(error.contains("Loading cancelled"));
        Ok(())
    }

    #[test]
    fn stale_loading_results_and_errors_keep_the_current_document() -> Result<(), Box<dyn Error>> {
        let context = egui::Context::default();
        let mut application = Application::new(&context, Some(test_renderer()?), None);
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
        let mut application = Application::new(&context, Some(test_renderer()?), None);
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
    fn renderer_initialization_failure_keeps_the_current_document() -> Result<(), Box<dyn Error>> {
        let context = egui::Context::default();
        let mut application = Application::new(&context, Some(test_renderer()?), None);
        application.sender.send(LoadMessage {
            generation: 0,
            path: PathBuf::from("original.gds"),
            result: Ok(document()?),
        })?;
        application.receive_loads(&context);
        application.renderer = None;
        application.sender.send(LoadMessage {
            generation: 0,
            path: PathBuf::from("replacement.gds"),
            result: Ok(document()?),
        })?;
        application.receive_loads(&context);
        assert_eq!(
            application.document_path,
            Some(PathBuf::from("original.gds"))
        );
        assert!(application.viewer.is_some());
        assert_eq!(
            application.error.as_deref(),
            Some("Renderer is not initialized")
        );
        assert_eq!(application.status, "Rendering initialization failed");
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
            let mut viewer = Viewer::new(document.scene, test_renderer()?)?;
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
            let mut output = context.run_ui(egui::RawInput::default(), |_| {});
            output.textures_delta.clear();
            let image = pollster::block_on(viewer.render_image(&context, viewport))?;
            assert!(image.shape_count > 0);
            assert!(image.image.pixels.iter().any(|pixel| pixel.a() != 0));
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
        let mut application = Application::new(&context, Some(test_renderer()?), None);
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

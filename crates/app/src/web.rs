use std::{cell::RefCell, rc::Rc};

use rlayout_viewer::GpuRendererHandle;
use wasm_bindgen::{JsCast, prelude::*};
use web_sys::{HtmlCanvasElement, HtmlInputElement, KeyboardEvent};

use super::Application;

#[wasm_bindgen]
pub struct WebHandle {
    runner: eframe::WebRunner,
    change: Option<Closure<dyn FnMut(web_sys::Event)>>,
    shortcut: Option<Closure<dyn FnMut(KeyboardEvent)>>,
}

impl Default for WebHandle {
    fn default() -> Self {
        Self {
            runner: eframe::WebRunner::new(),
            change: None,
            shortcut: None,
        }
    }
}

#[wasm_bindgen]
impl WebHandle {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn start(&mut self, canvas: HtmlCanvasElement) -> Result<(), JsValue> {
        self.destroy()?;
        if let Err(error) = self.start_inner(canvas).await {
            if let Err(cleanup_error) = self.destroy() {
                web_sys::console::error_1(&cleanup_error);
            }
            return Err(error);
        }
        Ok(())
    }

    pub fn destroy(&mut self) -> Result<(), JsValue> {
        self.runner.destroy();
        if self.change.is_some() {
            file_input().map_err(JsValue::from)?.set_onchange(None);
            self.change = None;
        }
        if let Some(shortcut) = &self.shortcut {
            let window =
                web_sys::window().ok_or_else(|| JsValue::from_str("Browser window is missing"))?;
            window.remove_event_listener_with_callback_and_bool(
                "keydown",
                shortcut.as_ref().unchecked_ref(),
                true,
            )?;
            self.shortcut = None;
        }
        Ok(())
    }

    pub fn has_panicked(&self) -> bool {
        self.runner.has_panicked()
    }

    pub fn panic_message(&self) -> Option<String> {
        self.runner.panic_summary().map(|summary| summary.message())
    }
}

impl WebHandle {
    async fn start_inner(&mut self, canvas: HtmlCanvasElement) -> Result<(), JsValue> {
        let render_state = Rc::new(RefCell::new(None));
        let mut setup = eframe::egui_wgpu::WgpuSetupCreateNew::without_display_handle();
        setup.instance_descriptor.backends = eframe::egui_wgpu::wgpu::Backends::BROWSER_WEBGPU;
        self.runner
            .start(
                canvas,
                eframe::WebOptions {
                    renderer: eframe::Renderer::Wgpu,
                    wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
                        wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(setup),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                Box::new({
                    let render_state = render_state.clone();
                    move |context| {
                        *render_state.borrow_mut() = context.wgpu_render_state.clone();
                        Ok(Box::new(Application::new(&context.egui_ctx, None)))
                    }
                }),
            )
            .await
            .map_err(|error| JsValue::from_str(&format!("Starting WebGPU renderer: {error:?}")))?;

        let state = render_state
            .borrow_mut()
            .take()
            .ok_or_else(|| JsValue::from_str("WebGPU render state is unavailable"))?;
        let renderer = GpuRendererHandle::from_render_state_async(&state)
            .await
            .map_err(|error| JsValue::from_str(&format!("Initializing GPU renderer: {error}")))?;
        {
            let mut application = self.runner.app_mut::<Application>().ok_or_else(|| {
                JsValue::from_str("The browser viewer stopped during initialization")
            })?;
            application.renderer = Some(renderer);
            application.status = "Open a GDSII file to begin".into();
            application.web_context.request_repaint();
        }

        let input = file_input().map_err(JsValue::from)?;
        let change = Closure::new({
            let runner = self.runner.clone();
            let input = input.clone();
            move |_event: web_sys::Event| {
                if let Some(file) = input.files().and_then(|files| files.item(0))
                    && let Some(mut application) = runner.app_mut::<Application>()
                {
                    application.start_file(file);
                }
            }
        });
        input.set_onchange(Some(change.as_ref().unchecked_ref()));
        self.change = Some(change);

        let shortcut = Closure::new({
            let runner = self.runner.clone();
            move |event: KeyboardEvent| {
                if (event.ctrl_key() || event.meta_key())
                    && !event.alt_key()
                    && event.key().eq_ignore_ascii_case("o")
                {
                    // The picker must open during a user gesture, before the next egui frame.
                    event.prevent_default();
                    event.stop_immediate_propagation();
                    if !event.repeat()
                        && let Err(error) = open_picker()
                        && let Some(mut application) = runner.app_mut::<Application>()
                    {
                        application.error = Some(error);
                        application.web_context.request_repaint();
                    }
                }
            }
        });
        let window =
            web_sys::window().ok_or_else(|| JsValue::from_str("Browser window is missing"))?;
        window.add_event_listener_with_callback_and_bool(
            "keydown",
            shortcut.as_ref().unchecked_ref(),
            true,
        )?;
        self.shortcut = Some(shortcut);
        Ok(())
    }
}

fn file_input() -> Result<HtmlInputElement, String> {
    web_sys::window()
        .and_then(|window| window.document())
        .and_then(|document| document.get_element_by_id("gds-file"))
        .ok_or_else(|| "The file picker is missing".to_string())?
        .dyn_into::<HtmlInputElement>()
        .map_err(|_| "The file picker is not an input element".to_string())
}

pub(super) fn open_picker() -> Result<(), String> {
    let input = file_input()?;
    input.set_value("");
    input.click();
    Ok(())
}

pub(super) async fn read_file(file: web_sys::File) -> Result<Vec<u8>, String> {
    if file.size() > f64::from(u32::MAX) {
        return Err("File is too large for browser file reads".into());
    }
    let buffer = wasm_bindgen_futures::JsFuture::from(file.array_buffer())
        .await
        .map_err(|error| format!("Reading {}: {error:?}", file.name()))?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}

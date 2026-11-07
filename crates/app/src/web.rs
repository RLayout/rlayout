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
        self.runner
            .start(
                canvas,
                eframe::WebOptions {
                    renderer: eframe::Renderer::Wgpu,
                    ..Default::default()
                },
                Box::new(|context| Ok(Box::new(Application::new(&context.egui_ctx)))),
            )
            .await?;

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

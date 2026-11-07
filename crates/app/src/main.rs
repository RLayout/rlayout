#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    rlayout_app::run_native()
}

#[cfg(target_arch = "wasm32")]
fn main() {}

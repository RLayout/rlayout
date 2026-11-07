#!/usr/bin/env bash
set -euo pipefail

toolchain="${RLAYOUT_RUST_TOOLCHAIN:-1.98.1}"
tool_directory="${RLAYOUT_WEB_TOOLS:-${RUNNER_TEMP:-$PWD/target}/rlayout-web-tools}"
rustup toolchain install "$toolchain" --profile minimal --component clippy,rustfmt
rustup target add --toolchain "$toolchain" wasm32-unknown-unknown
export RUSTUP_TOOLCHAIN="$toolchain"

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)
    trunk_target=aarch64-apple-darwin
    bindgen_target=aarch64-apple-darwin
    trunk_checksum=18e381936c848700fefb9ee3f64f362419fab700200eefee323ebc0d00a8e9db
    bindgen_checksum=b90ddb9357f584b1af897d58ad77e4cce2b48fc17e2a967bc249a87f72809444
    ;;
  Darwin-x86_64)
    trunk_target=x86_64-apple-darwin
    bindgen_target=x86_64-apple-darwin
    trunk_checksum=55b5cf2d6a23201f555cc561cc07157989581cc720a260505f8e98aae67abcf7
    bindgen_checksum=eb209c115b7cb95832423c1aa3f4d06d6c53f63ef74fc4b6fb79ef0b2eea5f10
    ;;
  Linux-x86_64)
    trunk_target=x86_64-unknown-linux-gnu
    bindgen_target=x86_64-unknown-linux-musl
    trunk_checksum=2ab8d548fd5ed9cda50906f61fd8654204e7893bd52ebbfd90533862b7d71ee1
    bindgen_checksum=9687f3bb551b9f66f37ed32a40d7913a00f7962db653054e56857bc306c3227f
    ;;
  Linux-aarch64)
    trunk_target=aarch64-unknown-linux-gnu
    bindgen_target=aarch64-unknown-linux-gnu
    trunk_checksum=b4017d1e0bfcffb4f4bee2be37c48def21e6e6896c3d0f503c43c42f714e57ea
    bindgen_checksum=cc8dd8e90aa9111758168d30c63a14b7f0276997d9d60d3e160a1fb5e7d3b919
    ;;
  *) echo 'The browser build script supports macOS and Linux on arm64 or x86_64' >&2; exit 1 ;;
esac

install_archive() {
  local name="$1" url="$2" checksum="$3" executable_path="$4"
  local download_directory
  mkdir -p "$tool_directory/bin"
  download_directory=$(mktemp -d "$tool_directory/download.XXXXXX")
  curl --fail --location --retry 3 "$url" --output "$download_directory/archive.tar.gz"
  python3 - "$download_directory/archive.tar.gz" "$checksum" <<'PYTHON'
import hashlib
import pathlib
import sys
if hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest() != sys.argv[2]:
    raise SystemExit('Browser tool archive checksum did not match')
PYTHON
  tar -xzf "$download_directory/archive.tar.gz" -C "$download_directory"
  install -m 755 "$download_directory/$executable_path" "$tool_directory/bin/$name"
  rm -rf "$download_directory"
}

if [[ ! -x "$tool_directory/bin/trunk" ]] || [[ "$("$tool_directory/bin/trunk" --version)" != 'trunk 0.22.1' ]]; then
  install_archive trunk \
    "https://github.com/trunk-rs/trunk/releases/download/v0.22.1/trunk-$trunk_target.tar.gz" \
    "$trunk_checksum" trunk
fi
if [[ ! -x "$tool_directory/bin/wasm-bindgen" ]] || [[ "$("$tool_directory/bin/wasm-bindgen" --version)" != 'wasm-bindgen 0.2.130' ]]; then
  install_archive wasm-bindgen \
    "https://github.com/wasm-bindgen/wasm-bindgen/releases/download/0.2.130/wasm-bindgen-0.2.130-$bindgen_target.tar.gz" \
    "$bindgen_checksum" "wasm-bindgen-0.2.130-$bindgen_target/wasm-bindgen"
fi
export PATH="$tool_directory/bin:$PATH"

cargo fmt --all -- --check
cargo clippy --locked --package rlayout-app --lib --target wasm32-unknown-unknown -- -D warnings
trunk build --release

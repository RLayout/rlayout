# Browser publishing

From the source checkout:

```sh
bash scripts/build-web.sh
export PATH="$PWD/target/rlayout-web-tools/bin:$PATH"
trunk serve
```

The build script installs the pinned toolchain and checksum-verified official
tool archives, checks the browser target, and builds `target/web-dist`.
`trunk serve` serves the app locally; `trunk build --release` produces the static
site. Trunk 0.22.1 and wasm-bindgen 0.2.130 are pinned in the setup.

The website repository is `RLayout/rlayout.github.io`. Its workflow is maintained
here as `deployment/pages.yml`; install it there as `.github/workflows/pages.yml`
on the default branch, and select Settings → Pages → Source → GitHub Actions.
The workflow checks out public source from `RLayout/rlayout`. It cannot publish
until that source repository exists and contains the browser implementation.

In `RLayout/rlayout`, set the Actions secret `PAGES_DISPATCH_TOKEN` to a
fine-grained personal access token with resource owner RLayout, access only to
`rlayout.github.io`, and Contents: read and write. Source CI sends a publish
request after native checks and the browser build succeed on `main`. The dispatch
is disabled in other repositories and on pull requests or feature branches.
Token expiration or missing permissions causes the dispatch job to fail visibly.

To retry a publish, run the website repository's Publish RLayout workflow and
provide the full lowercase source commit SHA. Manual publishes can also restore a
previous version. Automatic requests for superseded commits are skipped before
building and checked again before deployment; deployments are serialized.

The browser reads selected or dropped GDSII files into memory and processes them
locally. File contents are never uploaded. Parsing and scene preparation run on
the main thread, so large layouts can pause the interface. Native loading remains
threaded.

# Introduction


## Run
* Install make or read the Makefile to see tasks.
* There are tow versions, one with f64 and other with f32, currently only f64 build is working. f32 build not converges due the low precition in energy process.
Release run -> `make release_f64`
Dev run -> `make run_f64`

## Compile for production
It takes several minutes.
* Install wsl and:

`sudo apt install build-essential`

`sudo apt install musl musl-tools musl-dev`

`rustup target add x86_64-unknown-linux-musl`

`sudo apt install pkg-config libssl-dev`
*
`cargo build --target x86_64-unknown-linux-musl --features "with_openssl use_f64" --release` or `make build_prd`

`cp target/x86_64-unknown-linux-musl/release/theoretical_model .`

## Important deployment note for `func-lingosmelter2-dev`

`func-lingosmelter2-dev` is configured in **Deployment Center** with **Azure Pipelines** and branch **`master`** as deployment source.

That means:

- doing **Deploy to Function App...** from VS Code is **not enough** to guarantee that the latest local binary is what ends up running there,
- unpublished local changes are **not** the source of truth for that Function App,
- and if the pipeline is still pointing to the repo branch, Azure can keep serving the last code that was actually pushed/deployed from that branch.

Before assuming a new Rust binary is live in Azure, make sure you have:

1. built the Linux binary,
2. copied it to the repo root as `theoretical_model`,
3. committed/pushed the change to the branch used by the pipeline,
4. and confirmed that the pipeline/deployment actually finished.

Practical reminder:

- if Azure still responds without the new response fields or behavior, do **not** assume the local build was deployed;
- first verify what commit/branch/package the Deployment Center pipeline actually published.

![Dancing Cat](https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif)

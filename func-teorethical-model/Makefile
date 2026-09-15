.PHONY: test commit install activate run_f32 run_f64

install:
	@conda activate Yairama
	pip install pre-commit && \
	pre-commit install

lint:
	pre-commit run -a

test:
	cargo test -- --show-output

run_f32:
	cargo run --features "use_f32" --no-default-features

run_f64:
	cargo run --features "use_f64" --no-default-features

release_f32:
	cargo run --features "use_f32" --no-default-features --release

release_f64:
	cargo run --features "use_f64" --no-default-features --release

build_prd:
	`cargo build --target x86_64-unknown-linux-musl --features "with_openssl use_f64" --release`

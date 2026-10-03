# Jetson Xavier NX + Reflective Loop Runtime

Notes and code for a headless NVIDIA Jetson Xavier NX 8GB running L4T R35.6.5 / JetPack 5.1.7, its AI benchmark results, and RLR (Reflective Loop Runtime), a self-aware autonomous agent prototype, v0.

The recorded hardware has a failed fan. Keep it in the measured 10W operating envelope and retain the thermal guard; repair the fan before sustained unattended use. Benchmark results and limits are in [`bench/RESULTS.md`](bench/RESULTS.md).

## Layout

- [`prototype/`](prototype/) — v0 perception and orchestration prototype.
- [`rlr/`](rlr/) — standalone Reflective Loop Runtime package and tests.
- [`bench/`](bench/) — measurements, use cases, and sidecar design.
- `scripts/`, `config/`, `overlay/` — small text-only image-build tooling, when present.
- `Makefile`, `Dockerfile`, `REPORT.md` — headless image workflow and build report.

## Benchmark documents

- [Results](bench/RESULTS.md)
- [Use cases](bench/USE_CASES.md)
- [Full use case catalog](bench/USE_CASES_FULL.md)
- [RLR sidecar design](bench/RLR_SIDECAR_DESIGN.md)
- [RLR design v2](bench/RLR_DESIGN_v2.md)
- [Thermal guard operations](ops/README.md)

## Xavier-native llama.cpp runtime

The canonical local inference path is `llama.cpp` with CUDA compiled specifically for the Xavier NX Volta GPU (`sm_72`). The helper scripts keep the build and launch flags consistent:

```sh
bash scripts/build_llama_xavier.sh
bash scripts/run_llama_xavier.sh
```

By default they use `/models/src/llama.cpp`, `/models/build/llama`, and `/models/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`. Override `LLAMA_SRC_DIR`, `LLAMA_BUILD_DIR`, `MODEL_PATH`, `CTX`, `BATCH`, `UBATCH`, or `GPU_LAYERS` as needed.

Measure prompt-processing and generation throughput on the actual device with:

```sh
bash scripts/bench_llama_xavier.sh
```

Keep the server resident between RLR ticks rather than reloading the model for each decision.

## Run

Prototype: follow the setup and deployment instructions in [`prototype/README.md`](prototype/README.md). RLR: follow [`rlr/README.md`](rlr/README.md); its development invocation from this checkout is `cd rlr && python3 -m rlr --once`. Both need the local model server and appropriate Xavier libraries/models described in their READMEs.

The repository intentionally excludes model weights, engines, datasets, logs, databases, L4T bundles, root filesystems, and disk images.

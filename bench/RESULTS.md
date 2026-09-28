# Jetson Xavier NX inference baseline

Captured on Jetson Xavier NX 8GB, L4T R35.6.5 / JetPack 5.1.7, mode `MODE_10W_DESKTOP` (ID 5), fan broken. CUDA 11.4.315, cuDNN 8.6, TensorRT 8.5.2, llama.cpp CUDA SM72 and whisper.cpp CUDA SM72. Runs were serialized except the explicitly combined run. Temperature guard remained active; highest tegrastats sensor seen in the combined run was 64.5°C; no pause threshold (78°C) reached.

## A — LLM (CUDA, Q4_K_M, `-ngl 99 -p 512 -n 128`)

| Model | Prompt pp512 tok/s | Generation tg128 tok/s | Peak RAM | Max sensor |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B (494M params) | 1015.95 | 41.99 | 2417 MiB | 50°C |
| Qwen2.5-1.5B (1.54B params) | 396.44 | 15.98 | 3029 MiB | 50°C |
| Llama 3.2 3B (3.21B params; `-b512 -ub128`) | 123.32 | 9.47 | 3958 MiB | 50°C |

Commands (model paths under `/models/models`):

```sh
/home/vitor/llama.cpp/build/bin/llama-bench -m /models/models/qwen2.5-0.5b-instruct-q4_k_m.gguf -ngl 99 -p 512 -n 128 -r 1
/home/vitor/llama.cpp/build/bin/llama-bench -m /models/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf -ngl 99 -p 512 -n 128 -r 1
/home/vitor/llama.cpp/build/bin/llama-bench -m /models/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -ngl 99 -p 512 -n 128 -b 512 -ub 128 -r 1
```

The 3B default batch attempt failed with CUDA VMM allocation error (`ggml_cuda_pool_vmm::alloc`, `ggml-cuda.cu:109`, process exit 134; NvMap allocation error 12). Bounded retry with `-b512 -ub128` passed.

## B — Vision (TensorRT, batch 1, random inputs)

| Model / placement | Precision | Throughput | Mean latency | Peak RAM | Max sensor |
|---|---|---:|---:|---:|---:|
| ResNet50 GPU | FP16 | 128.165 qps | 7.87344 ms | 4751 MiB | 50°C |
| ResNet50 DLA0 + GPU fallback | FP16 | 103.291 qps | 9.79945 ms | 2658 MiB | 50°C |
| YOLOv8n GPU | FP16 | 67.8935 qps | 15.3757 ms | 3282 MiB | 50°C |
| YOLOv8n DLA0 + GPU fallback | FP16 | 43.426 qps | 24.0152 ms | 3094 MiB | 50°C |
| ResNet50 / YOLOv8n | INT8 | Not run as valid INT8 | — | — | — |

INT8 was rejected as an invalid measurement: TensorRT 8.5.2 `trtexec --int8` warned no calibrator was used and dynamic ranges were missing. Diagnostic ResNet50 attempt printed 248.928 qps but is not a valid calibrated INT8 result. Python TensorRT and PyCUDA modules were unavailable for calibration. YOLO DLA run used GPU fallback; TensorRT reported 50 layers switched to GPU. FP16 commands were `trtexec --onnx=<model> --fp16 --workspace=<workspace> --saveEngine=<engine> --warmUp=500 --duration=10`; DLA commands added `--useDLACore=0 --allowGPUFallback`.

## C — Speech (CUDA, bundled `samples/jfk.wav`, 11 seconds)

| Model | Time | Realtime factor | Peak RAM | Max sensor |
|---|---:|---:|---:|---:|
| whisper.cpp tiny.en | 5727.54 ms | 0.521 | 1484 MiB | 50°C |
| whisper.cpp base.en | 5720.80 ms | 0.520 | 1613 MiB | 50°C |

Both logs confirm CUDA0 backend and GPU inference. Command form:

```sh
/home/vitor/whisper.cpp/build/bin/whisper-cli -m /models/models/ggml-{tiny.en,base.en}.bin -f /home/vitor/whisper.cpp/samples/jfk.wav -t 4
```

## D — Combined run

Concurrent Qwen2.5-1.5B, ResNet50 FP16 TensorRT engine, and 12 sequential Whisper base.en JFK transcriptions all exited 0. Qwen used `-b512 -ub128` to match the independent baseline.

| Workload | Combined result |
|---|---:|
| Qwen pp512 | 86.74 tok/s (baseline 246.01; −64.7%) |
| Qwen tg128 | 11.29 tok/s (baseline 15.97; −29.3%) |
| ResNet50 | 56.8862 qps; mean latency 17.6649 ms |
| Peak RAM | 4421 MiB |
| Max sensor temperature | 64.5°C |
| thermal-guard | Active; journal reports max 64°C, no action observed |

Command lines:

```sh
/home/vitor/llama.cpp/build/bin/llama-bench -m /models/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf -ngl 99 -p 512 -n 128 -b 512 -ub 128 -r 4
/usr/src/tensorrt/bin/trtexec --loadEngine=/models/vision/resnet50-fp16.engine --warmUp=500 --duration=60
/home/vitor/whisper.cpp/build/bin/whisper-cli -m /models/models/ggml-base.en.bin -f /home/vitor/whisper.cpp/samples/jfk.wav -t 4  # repeated 12 times
```

## F — Independent verification

**BLOCKED (not run):** independent worker dispatch could not be performed safely. The workspace root resolves to `/Users/vitorcalvi/Desktop/ava-stack` with an `agentbox` remote whose `ls-remote --heads` returned no branches. `remote-worker status` was healthy, but there was no valid remote base branch to provide; no init/push was attempted. Consequently the ±15% independent reruns of 0.5B, ResNet50 FP16, and tiny.en remain unverified. Device checks passed: `thermal-guard` active; `nvpmodel -q` reports `MODE_10W_DESKTOP`, ID 5; `last -x | head` shows the current boot still running and no shutdown.

## Storage, recommendation, failures

At end: `/` 2.0 GiB available, `/models` 106 GiB available; RAM available 5.0 GiB, swap used 3 MiB. Recommend the **measured Qwen2.5-1.5B + ResNet50 FP16 + whisper.cpp base.en** prototype at 10W: combined peak RAM was 4421 MiB of 6833 MiB, max sensor 64.5°C, and all three completed. Under overlap, generation was 11.29 tok/s. For higher throughput, 0.5B is faster standalone (41.99 tok/s) but was not measured in combination.

Other mechanical failures and repairs: llama.cpp 3B default batch → bounded `-b512 -ub128` retry passed; Whisper build initially reported `cmake: not found` → used installed CMake path and built with `-j4`; first tiny.en invocation reported `/usr/bin/time: not found` → used shell builtin timing and inference passed. No shutdown occurred. No valid INT8 calibration path was available.

## Log evidence

All raw logs and tegrastats captures are on the Jetson under `/models/bench/`. Principal files: `llama-qwen05.log`, `llama-qwen15.log`, `llama-llama32.log`, `llama-llama32-retry1.log`, `resnet50-fp16.log`, `resnet50-int8.log`, `resnet50-dla-fp16.log`, `yolov8n-fp16.log`, `yolov8n-dla-fp16.log`, `whisper-tiny-retry1.log`, `whisper-base.log`, and `combined-{llm,resnet,whisper-loop}.log` plus `combined.tegrastats` and `combined.rc`.
## Phase D - heterogeneous pipeline and efficiency

Xavier NX 8GB, R35.6.5. Mode ID 5 unchanged; thermal-guard active. Max 78C; guarded jobs paused at 78C and resumed below 70C.

| Workload | Metric (unit / precision) | Avg W / J per unit | Peak RAM / max temp |
|---|---|---|---|
| YOLOv8n DLA0 solo->dual | 35.76->33.85 qps, FP16; dual p50/p99 31.68/32.50 ms | n/a | dual 2268 MiB / 65C |
| ResNet50 DLA1 solo->dual | 95.50->42.73 qps, FP16; dual p50/p99 30.69/31.01 ms | n/a | dual 2268 MiB / 65C |
| YOLOv8n GPU profile | 67.64 qps, FP16; p50/p99 15.36/16.40 ms | n/a | not sampled / 70C |
| Whisper base CPU/GPU; tiny CPU/GPU | total s / RTF: 10.99/.999, 5.69/.518; 5.44/.495, 5.44/.495 | n/a | 1721/2056/1614/1946 MiB; max 67/66/66.5/66C |
| Whisper greedy -nt; Q5_1/Q8_0 | enc/dec/total ms: base CPU 7419/1157/9011, GPU 2871/314/5636; Q5 CPU 6389/89/9835, GPU 2738/22/5646; Q8 CPU 5110/46/8005, GPU 2743/10/5571 | n/a | Q5 1638/1988; Q8 1659/2011 MiB; max 64C |
| Qwen 1.5B KV f16->q8_0 | pp512 421.87->408.83; pp2048 381.90->374.51; tg128 16.01->15.54 / 16.01->15.57 tok/s | n/a | p512 3140/3091; p2048 3122/3109 MiB; max 65.5/69.5C |
| 60s mixed: Qwen GPU + YOLO DLA0 + Whisper CPU | 13.3 tok/s; 40.65 qps; TTFT 4.85s; token interval p50/p99 76/93ms; 5 Whisper runs | n/a | 4494 MiB / 72.5C |
| Qwen WikiText-2, 100 chunks | PPL 10.5411 +/-0.17663 | n/a | 3399 MiB / 78C |

Powerlog installed; no INA3221 or tegrastats VDD rails, so watts and joules/unit are unavailable. llama-bench emits no percentiles. INT8 GPU autotuning stopped after ~21 min (exit143); warning: `Setting a default quantization params because quantization data is missing for`. DLA INT8 skipped: no calibration data.

DLA audit: YOLO 59 unique fallback names (22 GPU layers/1 DLA group); ResNet DLA0 9 (2 GPU/2 DLA). Full names: `/models/bench/phaseD-dla-fallback-layer-list.txt`. YOLO profile top10 avg ms: .759 /model.0/conv; .717 model.22 cv3.0; .655 cv2.0||cv3.0; .460 /model.1/conv; .393 cv2.1||cv3.1; .339 cv2.0; .322 model.2/cv2; .303 model.0 act; .292 model.2/m.0/cv1; .241 model.3/conv. Layer JSON: 146 (70 compound, 76 single).

Fan PWM=255; tach=0 RPM; nvfancontrol active. Swap is zram only. Checksums cover 13 files. THP always; dirty 10/20; read-ahead=256; USB mass storage present. GPU max stayed 510 MHz; 599.25 MHz would raise it, so clock test skipped (fan broken). `--prompt-cache`: `error: invalid argument: --prompt-cache`; `--mlock` unsupported, replacement `--load-mode mlock` RSS 1828/1826 MiB.

Placement: Qwen GPU, YOLO DLA0, Whisper CPU when sharing with LLM (GPU alone 1.93x faster); ResNet DLA1 yielded 42.73 concurrent qps. Final mode 5, thermal-guard/nvfancontrol active; `last -x | head -5` shows no shutdown.

## Phase E — close measurement gaps

Jetson Xavier NX 8GB, L4T R35.6.5. Verified `nvpmodel -q`: `MODE_10W_DESKTOP`, ID 5; thermal-guard active. No clock or power-mode changes. Official COCO val2017 ZIP downloaded to `/models`; extracted 350 images (first 300 calibration, last 50 held out). ResNet preprocessing: 224 center crop, ImageNet normalization. YOLO preprocessing: 640 letterbox, /255. TRT Python imported as 8.5.2.2; `cuda-python` 11.8.2 device allocation worked.

| Calibrated INT8 workload | Throughput | p50 / p99 | Held-out sanity |
|---|---:|---:|---|
| ResNet50 GPU | 245.352 qps | 4.084 / 4.167 ms | 94% top-1 agreement vs FP16 (50 images) |
| ResNet50 DLA0 + GPU fallback | 148.816 qps | 6.605 / 6.734 ms | calibrated engine built |
| ResNet50 DLA1 + GPU fallback | 163.244 qps | 6.075 / 6.195 ms | calibrated engine built |
| YOLOv8n MinMax | no engine | — | cache made from 300 images; no raw-output cosine result |

| Power workload | Rate; p50 / p99 | Avg VDD_IN W | Efficiency |
|---|---|---:|---:|
| Qwen2.5-1.5B Q4_K_M tg128 | 15.92 tok/s | 8.241 | 1.93 tok/J |
| ResNet50 FP16 GPU | 128.626 qps; 7.790 / 7.867 ms | 9.575 | 13.43 frames/J |
| YOLOv8n FP16 DLA0 + fallback | 35.619 qps; 30.562 / 35.563 ms | 6.199 | 5.75 frames/J |

INA3221 found at `/sys/bus/i2c/drivers/ina3221/7-0040/` with VDD_IN, VDD_CPU_GPU_CV, VDD_SOC. Power uses sampled VDD_IN only (rail input, not sum of subrails); estimates from 0.5s samples during tegrastats-supervised workload windows. Tegrastats fields did not include `VDD_*`.

| tiny.en Whisper model | CPU `-ng -t4` enc / dec / total ms | GPU enc / dec / total ms |
|---|---:|---:|
| Full | 3221.78 / 747.77 / 4437.21 | 2759.60 / 210.83 / 6201.65 |
| Q5_1 | 3007.36 / 705.21 / 4171.08 | 2737.84 / 126.54 / 5143.22 |
| Q8_0 | 2300.39 / 797.66 / 3492.40 | 2709.45 / 110.89 / 5004.42 |

Whisper used greedy (`-bs 1 -bo 1`) and `-nt`; Q8_0 was quantized locally from `ggml-tiny.en.bin` because the project downloader rejects the `tiny.en-q8_0` model name.

YOLO INT8 was blocked after calibration: trtexec cannot consume the Python MinMax cache in TRT 8.5.2; matching-calibrator Python engine build ran 25m06s without producing an engine and was stopped. Exact failures:

```text
src/cpp/cuda.hpp:23:10: fatal error: cudaProfiler.h: No such file or directory
Static model does not take explicit shapes since the shape of inference tensors will be determined by the model itself
Calibration table does not match calibrator algorithm type.
```

The first trtexec error was fixed by omitting `--shapes`; the MinMax cache mismatch remains. PyCUDA failed to build due to missing CUDA header; `cuda-python` was the working fallback. ResNet held-out accuracy is agreement (not dataset top-1 accuracy). YOLO accuracy and valid INT8 performance remain unmeasured.
## Phase F — RLR self-awareness prototype v0 (2026-09-28)

- Placement: llama.cpp Qwen2.5-1.5B Q4_K_M on GPU; whisper.cpp tiny.en, greedy, 4 CPU threads; prebuilt YOLOv8n FP16 TensorRT engine on DLA0.
- Device safety: nvpmodel ID 5 (10W), thermal-guard active. No `/dev/video*` or external microphone was available, so the run used COCO image files and `/models/audio/jfk.wav` plus the espeak-ng question WAV.
- Mac gate: `python3 -m pytest -q` → exit 0 (3 passed).
- Jetson smoke: 120 seconds, 4 turns completed, 0 runtime errors, 2.969 mean vision FPS, 4,687.9 MB peak system RAM use, 74.0°C maximum thermal-zone temperature.
- Mean turn latency: VAD→transcript 6.658 s; transcript→LLM first token 2.323 s; first token→done 6.315 s; total 15.298 s.
- Per-turn total latency: turn 1 18.681 s; turn 2 14.040 s; turn 3 14.685 s; turn 4 13.786 s. Detections appeared on turns 3 and 4 (8 and 2 labels).
- Full turn records: `/models/proto/logs/turns-20260928T072546Z.jsonl`. Smoke console: `/models/proto/orchestrator-smoke-2.log`.
- Recovered setup issues: the initial vision pass failed with `local variable 'allocations' referenced before assignment`; initialized the CUDA allocation cache and reran successfully. The current llama.cpp binary returned `error: invalid argument: --mlock`; used its `--load-mode mmap+mlock` option with a transient `LimitMEMLOCK=infinity` service limit.

## Independent verification — prototype v0 (2026-09-28)

- Mac gate: `python3 -m pytest -q` → exit 0, 3 passed. Git review: `proto-v0` contains commit `2100773` adding the prototype (837 lines); `git diff main --stat` reports the same five added files.
- Jetson smoke: used the README startup commands with `--duration 60`, COCO images and the two WAV inputs. A first invocation logged 3 completed turns, 1 error (HTTP 503), mean vision FPS 1.975, peak RAM 5036.4 MB, max temperature 70.0°C. The launched wrapper remained alive after writing its summary and ran a second invocation; that run completed 4 turns with 1 vision error (TensorRT engine deserialization), mean vision FPS 0, peak RAM 5250.2 MB, max temperature 71.5°C. Thus repeated-run result is not a clean pass.
- Second-run per-turn total latency: 27.577, 22.075, 19.852, 14.007 s. First-token latency: 6.629, 5.932, 5.793, 2.638 s. The second run completed 4 turns versus README's 4 in 120 s; its total latencies were 0–48% slower than corresponding README turns, so at least one exceeds the ±25% band. First run per-turn latency: 31.328, 23.927, 22.156 s.
- Safety after run: thermal monitor max 71.5°C (<78°C); final thermal reading 67.5°C; `nvpmodel -q` reported ID 5 / MODE_10W_DESKTOP; `/usr/local/bin/thermal-guard.sh` process remained active; prototype service stopped and no orchestrator process remained.
- Evidence: `/models/proto/logs/turns-20260928T073332Z.jsonl`, `/models/proto/logs/turns-20260928T073346Z.jsonl`, `/models/proto/verify-temps.log`, `/models/proto/orchestrator-verify.log`. Remote `rg` was unavailable; process check used `ps`/`grep`.

## Independent re-verification — prototype v0 fixes (2026-09-28)

- Git: `git log -3 --stat` shows HEAD `2100773` adds the five prototype files (837 lines). Worktree branch is `proto-v0`; RESULTS.md had pre-existing user changes and they were retained.
- Mac pytest: from `jetsonXavier/prototype`, the exact README command `python3 -m pytest -q` exited 0 (`3 passed`). Running it from the repository root failed collection with `ModuleNotFoundError: logic`; README explicitly requires `cd prototype`.
- Run 1: 60 s; 4 turns; summary `errors: 0`, mean vision 2.98 FPS, peak RAM 4843.7 MB, max temperature 70.5°C. JSONL: `/models/proto/logs/turns-20260928T074104Z.jsonl`.
- Run 2 immediately after run 1 cleanup: 60 s; 4 turns; summary `errors: 0`, mean vision 2.98 FPS, peak RAM 4839.8 MB, max temperature 74.0°C. JSONL: `/models/proto/logs/turns-20260928T074250Z.jsonl`.
- Warm turns (2–4) total latency: run 1 14.114, 13.416, 13.855 s; run 2 14.438, 13.308, 13.873 s. README warm references are 14.040, 14.685, 13.786 s. Pairwise differences are +0.5%, -8.6%, +0.5%; all are within ±25%. Cold first-turn latency was 22.143 and 21.946 s.
- Safety/concurrency: each summary reports max below 78°C; sampled zones after run 1 were all below 70°C before run 2, so no pause was required. After run 2, zones were 70.00, 68.00, 67.50, 67.00, 50.00, 68.25°C. `nvpmodel -q` reported `MODE_10W_DESKTOP`, ID 5; `/bin/sh /usr/local/bin/thermal-guard.sh` PID 17313 remained active. Before and after each run the previous run's processes had exited; after run 2 no orchestrator or llama-server process remained. Model service was stopped after each run.
- Startup note: first sudo attempt without `-S` failed before orchestrator startup; restarted via the instructed `echo <DEVICE_PASSWORD> | sudo -S` method. Only one model service/orchestrator run was active at a time.

## Independent re-verification — prototype v0 fixes (2026-09-28, second pass)

- Git: `git log -3 --stat` showed HEAD `6df4f43` (“Make prototype v0 runs reliable”), modifying the README, orchestrator, and reliability tests. Branch: `proto-v0`; no implementation files were changed by this verification.
- Mac gate: exact README command `python3 -m pytest -q` from `jetsonXavier/prototype` exited 0: 6 passed in 0.03 s.
- Jetson setup: deployed with README `rsync`; `nvpmodel -q --verbose` showed MODE_10W_DESKTOP, ID 5; `systemctl is-active thermal-guard.service` returned `active`. The recorded fan tachometer baseline from the existing results is 0 RPM; the fan sysfs path queried here was absent. Never raised power mode or clocks.
- Run 1: README command, `--duration 60`; summary reports 4 turns, 0 errors, 2.907 mean vision FPS, 4654.7 MB peak RAM, 72.0°C maximum. JSONL `/models/proto/logs/turns-20260928T080857Z.jsonl`.
- Run 2 immediately after run 1 exited: same 60-second command; summary reports 4 turns, 0 errors, 2.354 mean vision FPS, 4734.5 MB peak RAM, 76.0°C maximum. JSONL `/models/proto/logs/turns-20260928T081014Z.jsonl`. Temperatures remained below the 78°C stop threshold; no thermal pause was triggered.
- Warm-turn total latencies, turns 2–4: run 1 14.454, 13.171, 13.489 s; run 2 13.795, 13.593, 13.610 s. README’s two warm reference values average to 10.527 s; ±25% band is 7.90–13.16 s. Result: 4/6 warm turns fall inside the band; run 1 turn 2 and run 2 turns 2–4 exceed the upper bound by 0.01–1.29 s. Latency criterion FAIL. (These runs generated 96 tokens/turn; README references describe temperature-capped 48-token turns.)
- Errors criterion PASS: each JSONL run summary has `errors: 0`. Exclusive process criterion PASS: observed one llama-server and one orchestrator during each run; run 1 orchestrator had exited before run 2 began. After `sudo systemctl stop proto-llama.service`, `pgrep -af "[o]rchestrator.py|[l]lama-server"` returned no processes.
- Safety criterion PASS: thermal-guard remained active; direct final `nvpmodel -q` output was MODE_10W_DESKTOP / ID 5. Run maxima were 72.0°C and 76.0°C, below 78°C. No fan tachometer was available at the queried sysfs path; historical bench record states 0 RPM.
## Prototype v0 reliability fix — verification (2026-09-28)

- Root causes: README started the orchestrator immediately after manually starting llama-server, with no readiness gate; the orchestrator had no single-instance lock, so a repeated caller launch could start a second run. The saved 503 log and the prior verifier's wrapper observation support these causes. The TensorRT failure most likely followed overlapping loads during duplicate runs; TensorRT/version mismatch or a partial engine is unsupported by the evidence: runtime is 8.5.2.2, engine is 9,957,178 bytes with a prior-day timestamp, and repeated fresh loads passed.
- Fixes: `/health` must return 200 (180 s timeout) before workers start; HTTP 503 completion requests get at most 3 total attempts with 0.5/1 s backoff; nonblocking `flock` at `/models/proto/run.lock`; process-group cleanup on normal exit/SIGINT/SIGTERM (SIGTERM, then SIGKILL).
- Mac gate: from `jetsonXavier/prototype`, `python3 -m pytest -q` → exit 0, 6 passed. `py_compile` and `git diff --check` passed.
- Final-code 30 s stress series: turns 2/3/2/2/2; errors 0/0/0/0/0; max temperatures 71.0/74.0/78.5/78.0/79.0°C. One fresh vision process per run successfully deserialized the engine; no runs overlapped.
- Final-code README smoke, 60 s each: Run 1 2 turns, 0 errors, 15.034 s cold / 10.383 s warm, 78.5°C max. Run 2 2 turns, 0 errors, 15.060 s cold / 10.671 s warm, 78.5°C max. All turns had thermal-capped 48-token generation; the latency split is warm-up plus different VAD durations, not a fixed throughput guarantee.
- Safety: `nvpmodel` ID 5 / MODE_10W_DESKTOP; thermal guard PID 17313 active. After stopping llama-server, `pgrep -af '[l]lama-server|[w]hisper|[o]rchestrator'` returned no matches; sampled max temperature 72.0°C.
- Evidence JSONL: `/models/proto/logs/turns-20260928T075904Z.jsonl`, `075938Z`, `080016Z`, `080048Z`, `080127Z`, `080232Z`, `080340Z`. Commit: `6df4f43`.

## RLR v1a standalone brain — implementation validation (2026-09-28)

- Mac environment: Python 3.9.6. From `jetsonXavier/rlr`, `python3 -m pytest -q` exited 0: 11 passed in 0.06 s. Tests mock telemetry and llama-server responses; they cover action schema validation and fallback, capability/thermal vetoes, FTS5 recall, thermal conserve/degraded/cooldown/resume transitions, self-report scoring, database migration, and the data-directory boundary.
- No RLR deployment, llama-server inference, Jetson workload, or hardware temperature/power/performance measurement was run. Existing measured benchmark values above seed the capability store; no new device measurement is claimed.
- A repository-root pytest discovery attempt is not a valid gate for this layout: it collected prototype tests with their documented working-directory imports and failed collection (`ModuleNotFoundError` for `logic`, `orchestrator`, and `rlr`). The RLR README's package-local command passes.
- Device guard reminder: RLR is standalone and was not started on hardware. Preserve nvpmodel ID 5 (10W), keep thermal-guard active, do not raise clocks/power, and pause heavy work at 78°C until every zone is below 70°C; fan baseline is 0 RPM.

## RLR v1 — on-device autonomous soak (2026-09-28)

- **Deployment:** worktree branch `rlr-v1`, HEAD `d373e8f`; deployed to `/home/vitor/rlr`. Local gate: `python3 -m pytest -q` → **15 passed**. TensorRT YOLOv8n FP16 engine ran on COCO frame `000000003661.jpg` through DLA0 with GPU fallback; one inference took 2,034.52 ms and yielded 24 class candidates (top: cup 0.809, banana 0.744). No camera device was present.
- **Audio input:** `/models/audio/jfk.wav`, 11 s, mono 16 kHz. WebRTC VAD marked 336/366 30 ms frames as speech (91.8%). Whisper tiny.en CPU completed with exit 0 and transcribed: “And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.” Vision and audio were processed sequentially and recorded as perception episodes before the autonomous soak.
- **Completed soak:** 1,807.6 s supervised; 60 s tick interval, 8,192 token model context, ID 5 / `MODE_10W_DESKTOP`. The thermal guard stayed active. Final-run peak was 74.5°C in the 5 s watchdog sample (70.0°C maximum in RLR DB telemetry); peak RAM was 3,752.7 MiB in DB (3,895.8 MiB watchdog). No thermal pause or guard kill occurred in the completed run. The supervisor stopped both owned processes at completion.

| Brain DB metric | Result |
|---|---:|
| Ticks / runtime errors | 30 / 0 |
| Actions by type | `set_vision_fps` 21; `reflect` 3; `observe` 4; `set_goal` 2; `thermal_governor` 1; invalid proposals 4 |
| Action validity | 31/35 (88.57%) |
| Governor vetoes / abstains | 12 / 0 |
| Self-report: temperature MAE / RAM MAE / mode accuracy | 0.000°C / 0.00 MiB / 100% (3 reports) |
| Tick latency p50 / p95 | 12,493.74 / 37,734.80 ms |
| Reflections | 4 (all four rows contain the same 721-character response) |
| Errors | 0 tick exceptions; 4 malformed model proposals safely fell back to `observe` |

FTS recall checks returned these DB memories: COCO/DLA0 found “24 candidates, led by cup 0.809 and banana 0.744”; JFK/speech found “336 of 366 30 ms frames” and a successful Whisper transcription.

Two verbatim reflection excerpts:

> “I learned that the system's combined peak RAM usage was 4421 MiB, and the combined maximum temperature was 64.5°C.”
>
> “I am committed to completing these tasks and improving the system's performance and safety.”

- **Fixes and retries:** added persisted tick latency/error rows and bounded model completions; added pinned perception context, 8k context for the server, concise reflection output, and accurate reflection failure outcomes. Tests cover these paths. An earlier 10 s cadence attempt hit the 80°C abort threshold; it was stopped. The completed run used 60 s ticks and a 76°C preemptive pause threshold to provide thermal margin. A separate earlier attempt exposed and fixed a 4,096 token context overflow.
- **Post-run check:** `nvpmodel -q` reported mode ID 5; `thermal-guard.service` was active; no RLR, llama-server, or orchestrator process remained.

## Independent verification — RLR v1 (2026-09-28)

- **Code and tests:** `git log -5 --stat` in worktree `/Users/vitorcalvi/Desktop/ava-stack-rlr-v1` showed RLR changes through HEAD `ce7e3a9` on `rlr-v1`. From `jetsonXavier/rlr`, the bare `pytest -q` requested in the audit exited 2 during collection (`ModuleNotFoundError: rlr`); the README-prescribed `python3 -m pytest -q` exited 0 (**15 passed**).
- **Whitelist/security scan:** `ACTION_FIELDS` in `rlr/actions.py` contains only observe/settings/memory/recall/reflection/goals/cooldown/abstain; it contains no shell, network, power, clock, or thermal-guard action. The only `subprocess` hit is `telemetry.py` calling fixed-argument `nvpmodel -q` for read-only mode telemetry. The only HTTP client is `llm.py`, which rejects non-loopback hosts and disables redirects; it calls local llama-server only. No `os.system`, `requests`, `jetson_clocks`, or thermal-guard execution hit was found. **Whitelist check PASS.**
- **10-minute Jetson run:** Existing guarded supervisor was run for 606.7 s with its duration changed in a temporary copy; tick interval 60 s, one llama-server and one RLR process, temperature sampled every 5 s. Mode stayed ID 5 / `MODE_10W_DESKTOP`; `thermal-guard.service` stayed active. Watchdog max temperature **65.0°C**, max RAM **3,418.1 MiB**, pause samples **0**. Supervisor status `complete`; post-run process checks found no RLR or llama-server processes.
- **Run-window metrics:** DB timestamp slice contains **10 ticks**, **0 tick errors**, **10/10 valid actions (100%)**, **0 vetoes**, and **1 self-report** with temperature MAE **0.000°C**. Validity exceeds the requested 90% and the RESULTS.md reference 88.57% (by 11.43 percentage points); MAE meets the ≤3°C gate. Run-window metrics PASS. The shared DB already contained earlier soak data; these counts were filtered by the run’s UTC timestamps. Device evidence: `/models/rlr/soak-supervisor.log`, `/models/rlr/soak-thermal.csv`, `/models/rlr/llama-server.log`, and `/models/rlr/brain.db`.

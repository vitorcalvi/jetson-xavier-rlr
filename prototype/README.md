# Phase F: RLR self-awareness prototype v0

Three-process Jetson prototype: TensorRT YOLOv8n on DLA0, VAD-gated
whisper.cpp tiny.en on CPU, and a Qwen2.5-1.5B llama.cpp server on GPU.
`orchestrator.py` is the agent process; vision and audio run in separate
`multiprocessing` processes. The pure prompt, detection-dedupe, and PCM frame
helpers are covered by pytest.

## Mac checks

```sh
cd prototype
python3 -m pytest -q
```

## Jetson setup

Keep the Xavier NX in nvpmodel ID 5 (10W) and leave `/usr/local/bin/thermal-guard.sh`
running. Use `/models` for model files and builds; do not build in `/`.
Install runtime packages one at a time, with no concurrent build or benchmark:

```sh
sudo apt-get update
sudo apt-get install -y cmake build-essential git libopenblas-dev espeak-ng
python3 -m pip install --user cmake==3.28.4
python3 -m pip install --user webrtcvad pillow
```

Build llama.cpp with CUDA support and whisper.cpp with GPU support disabled.
The two builds are intentionally separate so only one heavy job runs at once.
Keep their source/build trees on `/models`:

```sh
git clone https://github.com/ggml-org/llama.cpp /models/src/llama.cpp
/home/vitor/.local/bin/cmake -S /models/src/llama.cpp -B /models/build/llama -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=72 -DCMAKE_BUILD_TYPE=Release
/home/vitor/.local/bin/cmake --build /models/build/llama --target llama-server -j2
git clone https://github.com/ggml-org/whisper.cpp /models/src/whisper.cpp
/home/vitor/.local/bin/cmake -S /models/src/whisper.cpp -B /models/build/whisper -DGGML_CUDA=OFF -DWHISPER_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
/home/vitor/.local/bin/cmake --build /models/build/whisper -j2
```

The existing whisper.cpp `ggml-tiny.en.bin` is at `/models/models/ggml-tiny.en.bin`.
Fetch the official `samples/jfk.wav` into `/models/audio/`. Create a second question WAV with
`espeak-ng -w /models/audio/question.wav 'What can you see and how are you doing?'`.
The Qwen GGUF and prebuilt DLA0 engine default to the existing `/models` paths.
The image adapter defaults to COCO files; `--camera-index 0` selects a live
OpenCV camera. Audio defaults to WAV files; `--audio-device hw:1,0` selects a
live `arecord` capture device. Auto-detection ignores the Xavier's internal APE
capture endpoints when deciding whether an external microphone is present.

## Deploy and run

From the Mac checkout, deploy only this prototype:

```sh
rsync -av prototype/ vitor@192.168.1.49:~/proto/
```

On Jetson, export the CUDA/TensorRT `PATH` and `LD_LIBRARY_PATH` from the task
brief. This llama.cpp build replaced `--mlock` with `--load-mode mmap+mlock`;
the transient service raises `LimitMEMLOCK` so the process can pin the mapped
model. See the [llama.cpp server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
Start the server and orchestrator in separate terminals:

```sh
sudo systemd-run --collect --unit=proto-llama --uid=vitor --property=LimitMEMLOCK=infinity --property=WorkingDirectory=/home/vitor/proto --property=StandardOutput=append:/models/proto/llama-server.log --property=StandardError=append:/models/proto/llama-server.log /models/build/llama/bin/llama-server -m /models/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf -ngl 99 -c 2048 --load-mode mmap+mlock -b 512 -ub 128 --host 127.0.0.1 --port 8080
python3 ~/proto/orchestrator.py --whisper-bin /models/build/whisper/bin/whisper-cli --whisper-model /models/models/ggml-tiny.en.bin --duration 60
```

Stop the transient model server after the run with `sudo systemctl stop proto-llama.service`.

Each turn and the run summary are JSON lines under `/models/proto/logs/`.
Temperatures at or above 75°C reduce vision to 1 FPS and generation to 48
tokens. At 78°C the agent pauses until all thermal zones fall below 70°C.
Before starting child processes, the orchestrator waits for `GET /health` to
return 200 (up to 180 seconds). Completion requests retry HTTP 503 up to three
total attempts with 0.5 and 1 second backoff. A non-blocking `flock` on
`/models/proto/run.lock` prevents overlapping runs. On exit or SIGINT/SIGTERM,
the run signals child process groups with SIGTERM, then SIGKILL if needed.

## Measurements

Measured on the Xavier NX in nvpmodel ID 5 with thermal-guard active. The
120-second run completed four turns with zero errors, 2.969 mean vision FPS,
4,687.9 MB peak system RAM use, and a 74°C maximum thermal-zone reading.

| Turn | VAD→transcript (s) | Transcript→first token (s) | First token→done (s) | Total (s) |
|---:|---:|---:|---:|---:|
| 1 | 9.780 | 2.388 | 6.508 | 18.681 |
| 2 | 5.586 | 2.176 | 6.276 | 14.040 |
| 3 | 5.817 | 2.632 | 6.234 | 14.685 |
| 4 | 5.448 | 2.096 | 6.240 | 13.786 |

Mean end-to-end latency in the initial 120-second run was 15.298 s. Later runs
measured two distinct latency regimes:

| Regime | Generation cap | Measured latency | Source |
|---|---:|---|---|
| Thermal-degraded (at/above 75°C) | 48 tokens | Cold: 15.034, 15.060 s; warm: 10.383, 10.671 s | Two back-to-back 60-second runs; see `bench/RESULTS.md` |
| Normal (below 75°C) | 96 tokens | Warm turns: 13.171–14.454 s | Latest independent verification, six warm turns across two runs; see `bench/RESULTS.md` |

Verify latency by comparing like-for-like temperature and generation-cap
regimes. Each turn record logs its `n_predict` value; report or filter by it
when comparing runs. Thermal-degraded turns use the 48-token cap, while normal
turns use the configured 96-token cap by default.

The thermal-degraded runs completed 2 turns with zero errors and a 78.5°C
maximum thermal reading; the agent paused at 78°C until all zones cooled below
70°C.

Five additional back-to-back 30-second stress runs completed 2, 3, 2, 2, and
2 turns, each with zero errors. Maximum zone temperatures were 71.0, 74.0,
78.5, 78.0, and 79.0°C. TensorRT 8.5.2.2 deserialized the existing 9,957,178
byte engine once in each fresh vision process. Full JSONL and build logs are
recorded under `/models/proto/`; the appended RESULTS section is the summary
record.

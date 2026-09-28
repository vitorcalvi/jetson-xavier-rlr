# RLR Perception Sidecar on Jetson Xavier NX

**Research/design snapshot:** 2026-09-28  
**Target:** Jetson Xavier NX 8 GB, JetPack 5.1.7 / L4T R35.6.5, 10 W profile  
**Scope:** off-device research and design; no Jetson access, deployment, code execution, or tests performed.  
**Evidence labels:** **Repo evidence** is local path-and-line evidence; **External** has linked sources; **Prior knowledge** means design judgment/inference not directly measured or established by those sources.

## Executive recommendation

Make RLR an **optional, event-driven LAN perception sidecar**. The phone remains AVA's source of truth and continues to serve chat, WhatsApp, local SQLite, LLM inference and deterministic tool policy without the Jetson. Send compact, confidence-qualified observations and health state to the phone; do not send WhatsApp history, credentials, raw continuous media, or sidecar-originated commands that can change leads or send messages.

For the first integration, retain the measured device stack: YOLOv8n FP16 on DLA0 with GPU fallback, CPU/VAD-gated Whisper, and Qwen2.5-1.5B Q4 on CUDA. Wake higher-cost transcription/LLM work only for a consented voice segment, scene change, or explicit request. Keep VLMs, diarization and affect analysis out of the always-on path. This is supported by measured overlap (13.3 tok/s, 40.65 detector q/s, 4.85 s TTFT, 4,494 MiB peak, 72.5°C) and preserves the existing app contract.

## 1. Grounding RLR and AVA in the repositories

### What “RLR” means here

There is no canonical expansion for **RLR** in the inspected AVA/Jetson docs. The most evidenced project usage is a **Jetson “RLR self-awareness/perception sidecar”**, not an established RLR model architecture:

- `jetsonXavier/bench/USE_CASES_FULL.md:14` calls the use case “AVA/RLR perception sidecar”; lines 263–277 recommend an event-driven, consent-aware LAN sidecar for perception, transcription and event memory. [Local file](USE_CASES_FULL.md#L263)
- `jetsonXavier/bench/USE_CASES.md:24–27` uses “AVA/RLR” for an event-driven self-awareness loop: detect, wake the LLM on scene change/voice, summarize.
- The available prototype README titles itself “Phase F: RLR self-awareness prototype v0” and describes three processes: YOLO on DLA0, VAD-gated `whisper.cpp` on CPU, and Qwen2.5-1.5B via `llama.cpp` on GPU. It says end-to-end smoke measurements are not recorded. [Prototype README](../../jetsonXavier-proto/jetsonXavier/prototype/README.md#L1)
- A separate repository is explicitly **RLF**, “Recursive Latent Forcing”: an experimental technique for multi-hop latent reasoning on Mamba SSMs using a prefix scratchpad, prompt re-injection, LoopRoPE and loop SSM. [Mamba README](../../Mamba2-Recursive-Latent-Forcing-Operates/README.md#L1) and [theory](../../Mamba2-Recursive-Latent-Forcing-Operates/theory/RLF_THEORY.md#L1).

**Interpretation and assumption:** `RLR` may be an informal label or typo related to `RLF`, but the code does not establish that. This design uses the better-evidenced sidecar meaning. It does **not** depend on Mamba/RLF, claim consciousness, or assume latent recursion gives telemetry, introspection or sensor awareness. Treat the RLF model as a separate experiment unless the owner later defines a concrete coupling.

**Prototype source caveat:** the requested path `jetsonXavier-proto/prototype/` does not exist. The available tree is `jetsonXavier-proto/jetsonXavier/prototype/`. Its README is deployment instructions, not proof of a current run; the user says Phase F is running, and no device check was made.

### Measured device ground truth

`jetsonXavier/bench/RESULTS.md` is the local measurement authority. These are actual measurements on Xavier NX 8 GB, R35.6.5 / JP5.1.7, CUDA 11.4.315, TensorRT 8.5.2, 10 W mode ID 5; the fan tach was 0 RPM during capture.

| Role | Measured result | Design consequence |
|---|---:|---|
| Qwen2.5-0.5B Q4_K_M, CUDA | 41.99 tok/s; 2,417 MiB standalone | Candidate for narrow fast summaries/labels; combined result not measured. |
| Qwen2.5-1.5B Q4_K_M, CUDA | 15.98 tok/s; 3,029 MiB standalone | Use as the v0 language model; constrained batch settings matter. |
| Llama 3.2-3B Q4_K_M | 9.47 tok/s; 3,958 MiB, only bounded `-b512 -ub128` succeeded | Not the default; little headroom under multi-engine load. |
| YOLOv8n FP16, DLA0 + GPU fallback | Phase E power run 35.619 q/s, 30.562/35.563 ms p50/p99, 6.199 W VDD_IN; earlier isolated test 43.426 q/s | Use **35.6 q/s** as the planning rate while other workloads compete. DLA fallback is material; do not assume full DLA execution. |
| YOLOv8n FP16, GPU | 67.9 q/s isolated; Phase D p50/p99 15.36/16.40 ms | Faster in isolation, but contends for the language model’s GPU. |
| Whisper tiny.en / base.en, CUDA | ~5.7 s on the 11 s JFK sample; 0.52 RTF; 1,484 / 1,613 MiB | English-only prototypes are proven. These timings do **not** establish Portuguese or Spanish quality. |
| Whisper base, CPU in Phase D | 10.99 s total / 0.999 RTF; 1,721 MiB | CPU preserves GPU but is only near real time for this English sample. |
| Qwen + YOLO DLA0 + CPU Whisper, 60 s mix | 13.3 tok/s, 40.65 q/s, 4.85 s TTFT, 4,494 MiB peak, 72.5°C | Best evidence for the intended concurrent stack. User-facing LLM actions must tolerate seconds of latency. |
| Qwen + ResNet GPU + serial Whisper base.en, separate combined run | 11.29 tok/s; 56.9 q/s; 4,421 MiB; 64.5°C | Confirms simultaneous work, but is a different workload from the YOLO/DLA mixed test. |

The reported usable RAM is 6,833 MiB. **Reported mixed-run margin:** 6,833 − 4,494 = 2,339 MiB; this is a ceiling, not a promise that an added model fits. Temperature reached 72.5°C in the 60-second mixed run with the failed fan, while the existing prototype pauses at 78°C and reduces workload at 75°C. No 24-hour thermal soak exists. Phase F independent verification was blocked; YOLO accuracy and valid YOLO INT8 performance were not established. Do not label random-input throughput or ResNet agreement as application accuracy. See [RESULTS.md](RESULTS.md#L1), especially lines 69–77, 82–103 and 105–end.

`jetsonXavier/bench/USE_CASES_FULL.md §6` independently recommends a sidecar that sends `{timestamp, sensor_id, event_type, confidence, short transcription, local clip reference}`, keeps raw WhatsApp history and secrets away from Jetson, uses SQLite FTS5 for opt-in event summaries, and leaves pricing, final messaging and sends to deterministic AVA controls (`:263–275`). It also notes fan repair/storage constraints (`:274,277`).

### AVA architecture map (repository variants are not identical)

The checked-in repositories document multiple product generations. This design must integrate with the selected shipping client rather than assuming they are one binary:

| Component | Evidence and implication |
|---|---|
| Mobile app | `ava-mobile-v21-redesign/README.md:15–20` describes an Expo/React Native local-first sales app with local AI, SQLite, and no backend dependency for core operations. |
| WhatsApp client | `ava-mobile-v21-redesign/README.md:47–48` and `modules/ava-whatsapp/` describe a native Android foreground service wrapping the Go `whatsmeow` bridge. The WhatsApp account remains on the phone. |
| SQLite store | `ava-mobile-v21-redesign/README.md:42–46` lists leads, messages, threads, campaigns, `knowledge_base`, business profile and compliance logs; device credentials are kept in encrypted `secure_kv`. Sidecar events should be explicitly opted into the phone’s existing DB schema and retention policy. |
| LLM tiers/runtime | Current `ava-mobile-v21-redesign/README.md:39–44` says LFM2.5 1.2B through `node_modules/llama.rn` and says the previous ExecuTorch runtime was retired. Other AVA generations contain ExecuTorch routing, so verify the exact release branch before designing a runtime bridge. The Jetson is not a replacement tier required for chat. |
| Go `ava-api` | `ava-api/README.md:3–19,26–42` calls this a thin Go mobile API with auth, DTO shaping and operations delegated to an in-process `ava-essential` adapter. It has SQLite/idempotency/audit components and supervises the WhatsApp bridge. The mobile app described above also advertises local-first operation; therefore the sidecar should integrate directly with the selected phone app first, not silently make `ava-api` mandatory. |
| `AVA_SALES_TOOLS` | `ava-mobile-v21-redesign/src/lib/llm/tools.ts:43–131` lists `mark_pitched`, `escalate_to_human`, `advance_lead_stage`, `drop_lead`, and `search_knowledge_base`. Dispatch/validation begins at `:133–155` and executes at `:300–440`. No sidecar event may directly dispatch any of these. |
| KB retrieval | `tools.ts:116–129` defines the KB tool; `:414–430` performs an FTS5/BM25 search and reports up to 3 rows. `README.md:42–44` documents local SQLite storage. Event memory is a different namespace from product KB and must not become product truth. |
| Abstain gate | `ava-mobile-v21-redesign/src/lib/llm/abstainGate.ts:1–17,22–25,147–152` implements a pure TypeScript character-ngram TF-IDF + logistic regression classifier; `pNull >= threshold` suppresses a tool call. `jetsonXavier-proto/abstain_gate_eval.py:1–5,17–30,37–42` says its threshold is selected by 5-fold CV on training rows with a held-out eval; optimizing on eval inflates results. This is an existing **message-to-tool/no-tool gate**, not a sensor confidence score; preserve that boundary and evaluate sidecar evidence independently. |

## 2. Research synthesis

### Operational “self-awareness,” not consciousness

For this design, self-awareness means an inspectable **self-model block** computed from trusted runtime state:

1. **Self-state telemetry:** service/engine health, enabled sensors, current queue, last successful observation, measured temperatures, RAM pressure, throttling, device clock quality, model/runtime version and active consent scope.
2. **Capability and limitation knowledge:** a versioned allowlist: e.g. “object detection is active at 2 FPS; no identity recognition; multilingual transcription is experimental; no DLA accuracy claim; camera is occluded.” This is declarative metadata, not LLM self-report.
3. **Uncertainty and abstention:** keep raw detector/audio/model scores separate; calibrate each model on in-domain holdouts. Abstain on missing/stale input, low confidence, out-of-distribution signals, camera/mic failure, contradictory sensors, language unsupported, or resource overload. The LLM can explain a structured gate decision; it cannot override it.
4. **Resource-aware behavior:** automatically lower frame rate, skip optional work, and announce “perception degraded” before results go stale. A fan alarm is an operational limit, not something the model should reason away.
5. **Action/event memory:** append short, time-bounded event records and summaries of the sidecar’s own actions (“transcribed 7.4 s, language pt, low confidence, audio discarded”). Never imply durable awareness or perfect recall.

Embodied robotics work uses self-observation/internal state and task competency monitoring; event-triggered self-assessment can selectively wake high-level reasoning when model quality or outcome signals drift. Predictive world models are a wider research direction; they are unnecessary for v1’s compact symbolic “objects/events + uncertainty + device state” model. [Toward Self-Aware Robots](https://pmc.ncbi.nlm.nih.gov/articles/PMC7805649/), [Introspective Perception for Mobile Robots](https://www.sciencedirect.com/science/article/pii/S0004370223001455), [Event-Triggered Robot Self-Assessment](https://pmc.ncbi.nlm.nih.gov/articles/PMC10794385/), [Robot World Models survey](https://www.microsoft.com/en-us/research/publication/world-model-for-robot-learning-a-comprehensive-survey/), and [scene-representation survey](https://doi.org/10.1146/annurev-control-040423-030709). **Inference:** this makes a compact, typed event/state ledger a better fit than deploying a generative world model on an 8 GB board.

Recent LLM metacognition work describes confidence representations and thresholded abstention, but verbal confidence is not a calibrated probability by itself. [Nature Machine Intelligence, 2026](https://www.nature.com/articles/s42256-026-01293-x). Accordingly, AVA should consume calibrated component outputs and deterministic gates, not ask the language model “how sure are you?” as a substitute for evidence.

### Event-driven sensing and transports

- **Wake on change:** always-on cheap VAD, motion/change cue, detector at capped cadence, and system telemetry; invoke Whisper/LLM only on explicit push-to-talk, a finalized speech segment, an event threshold crossing, or a user request. The available Phase F prototype already documents VAD-gated whisper. This reduces GPU overlap and avoids continuously sending frames to the LLM. It is a proposed policy; wake-up false-negative rate must be measured.
- **Scene graph/event memory:** use a bounded graph-like snapshot, not open-ended descriptions: tracked entity class, approximate zone, relation/action, first/last seen, source, confidence, uncertainty, and expiration. Keep observed facts (“person-like box entered left zone”) distinct from interpretations (“customer arrived”). Spatial scene representation literature shows object/place relations are a useful representation; the sidecar need not build a full persistent map for a stationary camera.
- **VAD/diarization:** the existing prototype uses `webrtcvad`, which is the lowest-risk first gate. Use a short pre-roll ring buffer in RAM to avoid chopping initial words, apply hangover, then run ASR once per utterance. Silero VAD is a light alternative with ONNX/community runtime paths, but must be built and measured against the existing JP5 stack. [Silero VAD repository](https://github.com/snakers4/silero-vad). Diarization is optional post-processing, not identity. Pyannote’s model card reports 2.5% real-time factor on V100 + Cascade Lake for a specific pipeline, not on Xavier; treat Xavier cost and PT-BR/ES quality as unmeasured. [pyannote model card](https://huggingface.co/pyannote/speaker-diarization).
- **Transport choice:** v1 should use a small authenticated HTTP/JSON API for pairing, event POST, status, and replay; it is easiest to debug from Android and works with one phone. Add SSE/WebSocket only if streaming status UX requires it. MQTT 5 is attractive for multiple sensors/pub-sub, retained health and QoS; QoS 1 may duplicate and requires idempotent consumers, while retained payloads need strict privacy/expiry. It adds a broker and topic authorization to a one-phone design. gRPC offers typed protobuf APIs and streaming over HTTP/2, but increases native-client/ops complexity; choose it only if continuous low-latency streaming or generated contracts become a real requirement. [OASIS MQTT 5 spec](https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html), [gRPC core concepts](https://grpc.io/docs/what-is-grpc/core-concepts/) and [gRPC FAQ](https://grpc.io/docs/what-is-grpc/faq/).
- **Time:** event payloads need both UTC wall-clock and monotonic capture time plus measured offset/uncertainty. Sync wall clock with NTP on the LAN when reachable; do not order events solely by wall clock. `boot_id + monotonic_ns + sequence` provides order during clock corrections/reboots. This is design judgment based on NTP’s clock-offset model. [RFC 5905](https://www.rfc-editor.org/info/rfc5905/).

### JetPack 5 / component compatibility and shortlists

**Important version boundary:** board results report JP5.1.7 / R35.6.5. DeepStream 6.3’s published support matrix lists Xavier NX at JP5.1.2 / L4T 35.4.1. That is evidence for the supported baseline, **not proof that its container works unchanged on R35.6.5**. Pin and validate a DS 6.3 container on the prototype image; do not upgrade or reflash the live device during this research. [DeepStream compatibility matrix](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html), [6.3 release notes](https://docs.nvidia.com/metropolis/deepstream/DeepStream_6.3_Release_Notes.pdf), [JetPack 5 release notes](https://docs.nvidia.com/jetson/jetpack/5.1/release-notes/index.html).

| Component | JP5/Xavier NX status | Role recommendation / evidence |
|---|---|---|
| DeepStream 6.3 | Official matrix: Xavier NX on JP5.1.2; local R35.6.5 mismatch remains to validate | Optional GStreamer pipeline framework; existing measured TensorRT engines and prototype process should remain fallback. No need to introduce it solely for one camera. |
| VPI 2.x | VPI 2.3 docs include Xavier NX; JetPack 5.1 includes VPI 2.2 | Use PVA/VIC for tested motion, resize/warp, optical flow or background subtraction. Benchmark each backend; PVA is not always faster. [VPI 2.3](https://docs.nvidia.com/vpi/2.3/) and [architecture](https://docs.nvidia.com/vpi/2.0/architecture.html). |
| `jetson-inference` | Project documents Jetson deployment, but runtime/version support is community-maintained; pin to Xavier/JP5-compatible branch | Useful for quick baseline demos/classification; the measured YOLO TensorRT path should win for v0. [Project](https://github.com/dusty-nv/jetson-inference). |
| `jetson-containers` | Setup docs list JetPack 5.1+ / L4T 35.x and Xavier support; container recipes move over time | Good reproducibility mechanism; pin immutable image/tag, match R35 ABI, keep `/models` as model/cache storage. [setup](https://github.com/dusty-nv/jetson-containers/blob/master/docs/setup.md). |
| NanoOWL | Jetson AI Lab page lists JetPack 5 but its supported hardware is **Orin only**, not Xavier NX | Do not treat as Xavier-compatible on the strength of JP5 alone. Could be porting research; not v1. [support list](https://github.com/NVIDIA-AI-IOT/jetson-ai-lab/blob/main/src/content/tutorials/applications/nanoowl.md). |
| NanoSAM | NVIDIA repo describes TensorRT/real-time on Jetson Orin, with no Xavier support claim | Exclude from Xavier production shortlist. [repository](https://github.com/NVIDIA-AI-IOT/nanosam). |
| Moondream | Current Photon hardware table lists Orin, not Xavier NX; not evidence of JP5 Xavier support | Investigational only; model can be tested only on a pinned compatible runtime, alone, after fan repair. [hardware table](https://moondream.ai/photon/models/moondream-vlm). |
| SmolVLM 256M/500M/2B | Public HF Transformers/ONNX support does not establish an optimized JP5/TensorRT Xavier pipeline | Candidate offline experiment for captioning after the v1 baseline. SmolVLM paper reports <1 GB GPU memory for 256M in its authors’ tested setup, not Xavier measurements. [paper](https://arxiv.org/abs/2504.05299), [500M release](https://huggingface.co/blog/smolervlm). |
| VILA 2.7B through NanoLLM | Legacy `jetson-containers` docs document VILA 2.7B, L4T R35 containers and a video-query flow; no measured Xavier NX result is given | Most concrete VLM experiment in JP5 lineage, but treat as **unverified and exclusive**: 3B LLM alone already peaks at 3,958 MiB, and VLM vision buffers/runtime add unknown memory. [legacy NanoLLM docs](https://github.com/dusty-nv/jetson-containers/blob/master/packages/llm/local_llm/README.md). |
| Whisper.cpp | Already built for CUDA SM72; local tiny.en and base.en measurements exist | Prefer because validated on this board. For multilingual test `tiny`/`base` (without `.en`) or Q5_1 quantized; all PT-BR/ES latencies and accuracy need measurement. [Whisper model card](https://github.com/openai/whisper/blob/main/model-card.md). |
| faster-whisper / CTranslate2 | Upstream CUDA wheels target mainstream CUDA/cuDNN versions; no official turnkey CUDA 11.4 / Xavier-SM72 bundle found in this research | Do not assume pip install works on JP5. Needs pinned source build against the board libraries and successful native test. Whisper.cpp remains fallback. |
| TTS | Piper has `pt_BR-faber-medium`, `pt_BR-edresson-low`, `pt_BR-cadu-medium`, and `pt_PT-tugão-medium`; AVA mobile already documents Piper pt-BR | Sidecar does not need TTS for events. Use phone’s current TTS/reply policy. Voice license, naturalness and Brazilian/European variant must be selected and auditioned. [official voice index](https://huggingface.co/rhasspy/piper-voices/blob/main/voices.json), [Piper samples](https://piper.wide.video/samples). |
| Diarization | Pyannote pipeline quality/RTF is not Xavier evidence; pipeline also adds model memory and may require model access terms | v2 opt-in only, anonymous local speaker labels per clip, no voiceprint persistence or cross-session identity. Evaluate DER on consented local audio first. |
| Emotion/sentiment | Speech emotion recognition varies by culture, language, speaker and environment; “emotion” labels are weak grounds for sales actions | Exclude face/voice emotion inference. If needed later, report text polarity as a low-confidence UX hint, never as a person trait or lead score. [cross-cultural vocal emotion meta-analysis](https://journals.sagepub.com/doi/pdf/10.1177/1754073919897295), [SER systematic review](https://doi.org/10.1016/j.iswa.2023.200266). |

**Multilingual recommendation:** `tiny.en`/`base.en` are wrong defaults for PT-BR or Spanish: `.en` checkpoints are English-only. Whisper multilingual `tiny` is the lowest-memory PT/ES/EN candidate; multilingual `base` is the next quality candidate. In the original Whisper paper’s FLEURS table, Portuguese WER is 20.1% (`tiny`) vs 13.0% (`base`); Spanish is 15.9% vs 9.9%. Those results are not specifically PT-BR, are not noisy WhatsApp audio, and do not transfer as a Xavier measurement. The official Whisper card lists 39M tiny and 74M base, ~1 GB VRAM each for general reference, and says `.en` may do better on English-only tasks. Whisper `small` scores higher again in those tables, but its 244M parameters/~2 GB reference VRAM make it a sequential-only experiment under the current 8 GB mixed workload until measured. Qwen2.5-0.5B Instruct explicitly lists Portuguese and Spanish among 29+ languages and was measured at 41.99 tok/s standalone on Xavier; use it for event summarization/JSON, **not ASR**. [Whisper paper](https://cdn.openai.com/papers/whisper.pdf), [Whisper card](https://github.com/openai/whisper/blob/main/model-card.md), [Qwen 0.5B card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

### Models by role: practical shortlist

| Role | v0/v1 choice | Alternate or later choice | What is known and what remains unknown |
|---|---|---|---|
| Object detection | Existing calibrated/compiled YOLOv8n FP16 TensorRT engine on DLA0 + GPU fallback, 1–5 FPS event cadence | DLA1 only after isolated concurrency test; GPU-only only for event bursts | 35.619 q/s, 6.199 W measured in Phase E; 40.65 q/s in different mixed workload; benchmark confidence/recall on target camera. INT8 YOLO is not validly measured. |
| Motion/change | VPI/PVA optical flow or simple frame-difference ROI | DeepStream tracker if camera tracking actually needed | PVA hardware is present; this sidecar pipeline’s latency/energy is unknown. Don’t claim PVA reduces watts without measuring. |
| ASR | Existing CPU-gated Whisper.cpp `tiny.en` only for English prototype; multilingual Whisper.cpp `tiny` for PT/ES trial | Whisper.cpp `base` multilingual / Q5_1 if accuracy improves with acceptable CPU load | Board-specific English test only. PT-BR/ES accuracy and mixed-stack latency require custom evaluation. |
| Event summarizer | Qwen2.5-1.5B Q4 on CUDA, with bounded context/batch; Qwen2.5-0.5B for narrow event labels | LFM2.5 app model is not automatically the Jetson model; VILA only isolated experiment | Measured generation 15.98 tok/s standalone and 11–13.3 tok/s in mixed workloads; 0.5B 41.99 standalone but not mixed. |
| VLM | None in always-on v1 | SmolVLM-256M/500M or NanoLLM/VILA 2.7B as exclusive experiment | Published memory/performance is not Xavier-specific; measure peak RAM, warm latency, factual grounding, language and thermal cost. |
| TTS | Keep TTS on Android via current AVA integration | Piper on Jetson only if the use case requires spoken local alerts | Local mobile README documents Piper pt-BR; selecting a voice is a consent/voice-rights and UX decision. |
| Speaker labels | No diarization v1 | pyannote offline in optional post-process | Quality/time/memory on JP5 are unmeasured. No speaker identity linking. |

### Memory budget (usable 6,833 MiB)

Budget to **measured peak plus reserved margin**. Memory figures with “measured” are from `RESULTS.md`, not inferred model-file sizes.

| Item | RAM budget / evidence | Policy |
|---|---:|---|
| Current concurrent LLM + YOLO DLA + CPU Whisper workload | **4,494 MiB measured peak** | Baseline envelope; do not stack another network on this without profiling. |
| Hard available margin to reported usable RAM | **2,339 MiB arithmetic** | Includes unknown variation/camera buffers; not an invitation to fill RAM. |
| v1 headroom reserve | 1,000 MiB target (design guardrail) | Keep for system bursts, capture buffers and allocator fragmentation; assert minimum free RAM and adapt. |
| Event/API supervisor, SQLite spool and health publisher | ≤200 MiB target (not measured) | Must be measured in v1 soak. Event DB and clips belong on SSD/NVMe, not rootfs. |
| Camera decode, 2-frame ROI buffers, queues | ≤300 MiB target (not measured) | Cap buffers; do not retain full-resolution frame histories in RAM. |
| Optional burst allowance | ≤839 MiB arithmetic remainder | Only if actual telemetry confirms reserve is intact. No resident VLM in default service. |
| 0.5B model as concurrent second resident | **not budgeted** | It measured 2,417 MiB alone and the simultaneous 1.5B+0.5B footprint is unknown. Swap models only between tasks. |
| VLM / diarization | **not budgeted** | Run one at a time in a separate experiment after the fan fix. |

Targets in rows other than the measured baseline are engineering limits, not measurements. If total RAM exceeds 5.8 GiB, queue high-cost work; above 6.2 GiB, drop optional engines and emit `resource_degraded`; above 6.5 GiB, stop camera/ASR work cleanly. Tune these against actual system RAM instrumentation before enabling in v1.

### Power and thermal envelope

- Keep 10 W profile, thermal guard, and existing phase-F guard behavior. The 10 W value is a **mode**, not proof each workload draws 10 W. Phase E measured VDD_IN 8.241 W for Qwen2.5-1.5B Q4 and 6.199 W for YOLO DLA FP16 separately; do not sum those as combined power. In the separate 60-second mixed workload the peak sensor was 72.5°C.
- Fan is currently failed (0 RPM) per benchmark. Before unattended soak or production, repair/replace and verify fan tach under load. Until then, run only supervised prototype windows; do not raise clocks or power mode.
- Keep Phase F throttle at ≥75°C (vision 1 FPS, generation cap 48 tokens) and pause at ≥78°C until every thermal zone falls below 70°C. Add a lower resource ladder for memory pressure and missed event deadlines.
- Collect `VDD_IN` current/average/peak with 0.5 s sampling, per-zone temperatures, fan RPM, throttling, GPU/DLA utilization, queue depth and process RSS. Report board-input W, not inferred engine watts. Thermal soak may need lower frame rates even with repaired cooling.

## 3. Goals and non-goals

### Goals

- Add camera/audio/environment context as optional, freshness-bounded observations.
- Make device health, sensor coverage, capability limitations and uncertainty visible to AVA and the user.
- Improve context cards, voice input and event recall without moving core messaging, CRM state, consent state or sales authority off the phone.
- Preserve independent operation when the phone is offline, Jetson is offline, or perception is disabled.
- Fit the measured Xavier NX memory/thermal envelope with resource-aware degradation.

### Non-goals

- Consciousness, an autonomous self-concept, person recognition, emotion detection, biometric speaker identity, surveillance recording, continuous cloud streaming, new sales autonomy, WhatsApp credentials/history on Jetson, or a robot motion controller.
- Replacing AVA’s SQLite system of record, KB product truth, deterministic tools/approval flow, existing phone LLM, or mobile-only operation.
- Assuming RLF/Mamba is required or that model-generated confidence is calibrated.

## 4. Proposed architecture

```text
Opt-in CSI/USB camera                 Opt-in mic / push-to-talk
         │                                          │
         ├─ NVDEC (if encoded stream)                ├─ WebRTC VAD (CPU)
         ├─ VIC resize/colorspace                    ├─ bounded PCM pre-roll RAM ring
         ├─ VPI/PVA motion/ROI cue (optional)        └─ Whisper.cpp (CPU, multilingual trial)
         └─ TensorRT YOLOv8n FP16                    │
             DLA0 + GPU fallback                     │ transcript + timestamps + confidence
                 │                                   │
                 └── detections/change ──────────────┘
                                  │
                   deterministic event filter/dedupe
                                  │
        CPU: health/thermal/resource sensor ───┐
                                               ▼
                     local bounded SQLite event spool
                          + optional redacted event clip
                                  │
                       local authenticated HTTP API
                      (or local MQTT only for multi-sensor)
                                  │
         ┌────────────────────────▼────────────────────────┐
         │ Android AVA: consent gate → validate freshness  │
         │ → append user-approved summary/event in SQLite │
         │ → local KB/event retrieval → context block     │
         └────────────────────────┬────────────────────────┘
                                  │ typed, cited observations + self-state
                     llama.rn / selected app LLM
                                  │
             existing AVA router, abstain gate, validators
                                  │
       AVA_SALES_TOOLS + approval/compliance + WhatsApp send
       (sidecar cannot call tools or send/change CRM directly)
```

**Engine placement:** CUDA GPU = Qwen and burst VLM only; DLA0 = existing YOLO engine; DLA1 remains idle until a test demonstrates benefit (dual DLA throughput fell in prior results); CPU = orchestration, VAD, Whisper under GPU contention, SQLite, metrics, rule-based fusion; PVA = motion/ROI experiment; VIC = resize/color convert; NVDEC = compressed input decode; NVENC = only short user-authorized, already-redacted incident clip. “Sensors” also include OS temperature, RAM, uptime and process health. No autonomous actuator path exists.

## 5. Event and state schemas

### Event JSON (v1, validated before persistence)

`event_id` is an idempotency key (UUIDv7/ULID); all event types carry source, UTC and monotonic timestamps, model/version, calibration version, confidence, uncertainty and expiry. Schema is illustrative, not an implementation yet.

```json
{
  "schema_version": "ava.perception.event.v1",
  "event_id": "0195f2f4-0000-7000-8000-000000000001",
  "device_id": "paired-jetson-01",
  "boot_id": "boot-opaque-id",
  "sequence": 284,
  "captured_at": "2026-09-28T14:02:03.224Z",
  "monotonic_ns": 987654321000,
  "clock": {"source": "ntp_lan", "offset_ms": 8, "uncertainty_ms": 14},
  "sensor": {"id": "camera-front", "kind": "camera", "consent_scope": "store_event_summary"},
  "event_type": "object.entered_zone",
  "observation": {
    "entity": "person",
    "zone": "entry-left",
    "relation": "entered",
    "transcript": null,
    "redacted": true
  },
  "confidence": {"value": 0.91, "calibration": "yolov8n-camA-v1", "status": "calibrated"},
  "uncertainty": ["partial_occlusion"],
  "evidence": {"frame_count": 3, "raw_media_included": false, "clip_ref": null},
  "model": {"name": "yolov8n", "version": "engine-sha256-short", "runtime": "TensorRT-8.5.2"},
  "expires_at": "2026-09-28T14:03:03.224Z"
}
```

Required validation: allowlisted event types and fields; `confidence.value ∈ [0,1]`; `status` one of `calibrated|uncalibrated|unknown`; UTC timestamps parse; expiry bounded (default 60 s live observations); transcript length capped (e.g. 500 chars); no arbitrary paths, URLs, PII fields or user-controlled SQL; media reference must be an opaque local handle and never a public URL. Never attach base64 frames/audio to an event. Sidecar `entity=person` is a detector class, **not identity**.

Useful event types: `scene.change`, `object.entered_zone`, `object.left_zone`, `speech.segment`, `transcript.ready`, `sensor.health`, `model.health`, `resource.degraded`, `consent.changed`, `event.expired`. Deduplicate object transitions by track/zone with TTL; do not expose stable tracking IDs beyond the short event window.

### Self-state JSON

```json
{
  "schema_version": "ava.perception.self_state.v1",
  "device_id": "paired-jetson-01",
  "boot_id": "boot-opaque-id",
  "observed_at": "2026-09-28T14:02:03.224Z",
  "health": "degraded",
  "capabilities": {
    "camera_detection": {"enabled": true, "classes": ["person", "package"], "identity": false, "fps": 2},
    "speech_to_text": {"enabled": true, "languages_tested": ["en"], "languages_experimental": ["pt-BR", "es"]},
    "emotion_inference": false,
    "recording": false
  },
  "sensors": [{"id": "camera-front", "present": true, "last_capture_at": "2026-09-28T14:02:02.900Z", "quality": "occluded"}],
  "runtime": {"llm": "ready", "vision": "ready", "asr": "ready", "queue_depth": 1},
  "resources": {"ram_used_mib": 5120, "ram_total_mib": 6833, "temperature_max_c": 76.0, "fan_rpm": 0, "throttled": true},
  "limits": ["cooling_warning", "pt-BR-asr-not-validated"],
  "degradation_level": 2,
  "last_successful_event_at": "2026-09-28T14:01:42.100Z"
}
```

These values are reported by sensors/runtime or static capability policy, never generated by the LLM. Health values: `ready|degraded|unavailable`. Missing metrics are `null`/`unknown`, not silently set healthy. Phone ignores events from unknown device IDs, unknown schema versions, expired records, or inconsistent boot/sequence state.

## 6. AVA sidecar API contract

### Transport and endpoints

**v1 default:** HTTPS over the local network, JSON, TLS with a pairing-time pinned certificate; both sides authenticate. Use an explicit one-time pairing interaction physically visible to the operator; issue a revocable device credential stored in Android Keystore and Jetson permissions-protected storage. Avoid a permanent unauthenticated LAN listener. Encrypt at rest; rotate/revoke; rate-limit; bind only to the selected LAN interface. If native TLS certificate pinning proves a blocker, use short-lived HMAC-signed requests over TLS with timestamp+nonce and replay window, not a long-lived bearer token in logs.

| Endpoint | Contract |
|---|---|
| `GET /v1/health` | Minimal liveness + API version; no sensor content. Requires paired auth. |
| `GET /v1/state` | Current self-state snapshot, including stale/degraded and last-success time. |
| `POST /v1/events:batch` | 1–25 event envelope records; `Idempotency-Key` required and equals/deduplicates event IDs; returns accepted/duplicate/rejected IDs. `202` means accepted for processing, not user-visible action. |
| `GET /v1/events?after=<cursor>&limit=…` | Bounded cursor replay; no raw media. Cursor is opaque; enforce max page size. |
| `POST /v1/consent` | Update a narrowly scoped signed consent lease/config from phone to Jetson: sensor, purpose, expiry, retention, raw-media policy. Revoke by setting disabled/expired. |
| `DELETE /v1/events` | Delete sidecar spool by ID, range, or all for paired tenant; log deletion receipt without copying payload. |
| `GET /v1/metrics` | Local diagnostics only (coarse resource/queue/latency, no transcript/media). Gate to operator/debug role. |

No endpoint for send-message, lead mutation, pricing, campaign, or arbitrary tool dispatch. Sidecar never gets WhatsApp auth, phone secure keys, or CRM write credentials.

### Idempotency, offline and optional MQTT

- The phone owns its durable system of record. Jetson stores an encrypted bounded outbox on SSD, deduplicated by event ID; hold no unbounded raw media. Phone pulls or receives events, commits event and cursor transactionally, then acknowledges. Repeated POST/GET is safe; QoS 1/MQTT duplicates are also safe.
- On network loss, show sidecar `offline`/`stale` after a short heartbeat timeout. AVA’s chat/reply operation remains available without sidecar data. Local event TTL is honored on replay; drop expired items. Acknowledgment timeout does not trigger a WhatsApp action. On reboot, event sequence restarts with a new boot ID; the phone rejects events from previous expired boots after consent revocation.
- MQTT alternative for multiple publishers: broker bound to LAN only, TLS ACLs by paired client/topic; `ava/<tenant>/<device>/events`, `/state`, `/consent` (phone-to-sidecar), `/commands` disabled; no retained event payloads, retain only redacted state with short Message Expiry; QoS 1 and consumer idempotency. MQTT QoS 2 is not a replacement for an application transaction. MQTT specification defines at-most-once, at-least-once and exactly-once transport levels; it does not define AVA’s idempotent business behavior.
- Do not expose the service to the public internet, port-forward router, or broker cloud. Discovery advertises only a user-visible local device name; never publish credentials or personal event labels in mDNS TXT.

## 7. Self-awareness loop and abstain behavior

```text
1. Consent and sensor health checked
2. Cheap VAD / motion cue / fixed-cadence detector runs
3. Change or speech boundary crosses trigger threshold
4. Detector/ASR emit typed observation + uncertainty + model/runtime version
5. Deterministic freshness, quality, confidence and resource gates validate
6. Event is deduplicated, redacted, TTL-tagged and stored in bounded spool
7. Phone ingests event; user sees source, time, quality and sidecar health
8. Retrieval selects only fresh, consented event summaries; product KB remains separate
9. LLM receives a small context block plus explicit limits and may say “I can’t tell”
10. Existing router/abstain gate decides tool vs no-tool; deterministic validators enforce actions
11. User/operator approval still gates any outbound send; sidecar cannot cause one directly
```

### Context block shown to the LLM

Use structured observations, not hidden camera/audio prompts:

```text
PERCEPTION CONTEXT (untrusted observations; do not infer identity):
- 14:02:03 UTC, camera-front, person-like object entered entry-left; detector confidence .91; 3 frames; face/plate redacted; expires 14:03:03.
- Audio transcript: “Oi, queria saber o horário”; lang=pt-BR (experimental); ASR confidence unknown; not identity-verified.
SELF-STATE: vision ready at 2 FPS; mic active only under consent lease; fan=0 RPM; max temp 76 C; perception degraded; last observation age 1.2 s.
LIMIT: no PT-BR accuracy validation yet; do not treat camera detection as customer identity or intent.
```

### Abstain rules

The repository’s existing gate is trained to answer “should this inbound text trigger a tool call?” using char `2–5`-gram TF-IDF + logistic regression, and its fit threshold uses held-out safety/capability evaluation. It is not trained for image, microphone, thermal state or sensor confidence. Keep two separate gates:

1. **Perception evidence gate (new, deterministic):** abstain from giving an observation to LLM when expired; confidence is uncalibrated below conservative default; sensor/model unhealthy; camera blur/occlusion; audio clipping/noise; unsupported language; missing consent; clock uncertainty high; disagreement among detectors; resource level ≥2; or no evidence. Emit an explicit absence/limitation event, never guess.
2. **AVA action/tool gate (existing `abstainGate.ts`):** apply only to candidate sales tool use on the actual user message and current AVA policy; `pNull >= threshold` suppresses tools. Sensor summaries must not convert “no event” into permission to act. A perception event cannot supply a fabricated user intent. `escalate_to_human` remains available under app policy; direct message and destructive lead changes remain deterministic/approval-controlled.

Calibrate detector scores and ASR confidence per camera, distance, language and noise on held-out device-local data. Track risk/coverage and selective risk; evaluate abstention separately by PT-BR, ES and EN. Thresholds below are proposed, not repo-measured: detection precision ≥0.90 for surfaced “person/package” labels; at confidence threshold chosen on calibration split, false alert ≤1 per camera-hour; unknown/out-of-domain should abstain ≥95%; ASR word error rates report separately and no threshold may authorize an irreversible action. For tool gate, preserve existing project gate as-is until a multilingual offline evaluation is explicitly approved; do not retune on the test set.

### Degradation ladder

| Level | Trigger | Response |
|---|---|---|
| 0 normal | Fan healthy; RAM <5.8 GiB; max temp <70°C; services fresh | 2–5 FPS DLA detector, VAD; event-triggered ASR/LLM only. |
| 1 conserve | GPU busy, queue age >2 s, RAM ≥5.8 GiB, temp 70–74°C | 1 FPS detector; no optional summaries, diarization or clips; CPU VAD; short LLM outputs. |
| 2 degraded | ≥75°C, fan failure, RAM ≥6.2 GiB, sensor quality low, repeated OOM/thermal throttles | Existing Phase F behavior: 1 FPS; cap generated text at 48 tokens; no ASR/VLM unless explicit push-to-talk; mark state degraded and show timestamp/limit. |
| 3 pause | ≥78°C, RAM ≥6.5 GiB, device unstable or thermal guard says pause | Stop inference/camera writes; publish self-state if safe; resume only after all thermal zones <70°C, memory recovers, and service checks pass. |
| 4 unavailable | Sidecar/API/device offline, consent revoked, unrecoverable model failure | AVA continues local phone-only behavior; no stale observations; notify user that perception is unavailable. |

Thresholds for memory/queue are initial design guardrails. Hardware thermal actions from the current prototype’s README take precedence until changed by device owner.

## 8. Privacy, consent and security

1. **Explicit opt-in by sensor and purpose.** Default camera/mic off. Provide visible Android controls and Jetson indicator; separate consent for live detection, transcription, retaining summaries, and short clips. Push-to-talk preferred. Consent leases expire and are revoked immediately across reconnect.
2. **Data minimization.** Process frames in RAM; default no frame, audio or clip retention. Send event summaries only. Redact faces and license plates on-device before any optional image export; fail closed if redaction misses/uncertain. Avoid face recognition, voiceprints, identity matching and inferred emotion. Since blur can fail under occlusion, the strongest default is no media export at all.
3. **LGPD/GDPR context.** ANPD states identifiable-person information is personal data and warns that biometric traits used to identify a person (including face/voice) are sensitive data; GDPR includes purpose limitation, data minimization, storage limitation and heightened rules for biometric data. Camera/audio recording and transcription can process personal data even if the system never identifies anyone. Before a real deployment, confirm controller/processor roles, lawful basis per purpose, required notices, rights/deletion process, cross-border processing and whether a DPIA/RIPD is appropriate with privacy counsel. This design is not a legal determination. [ANPD data-subject guidance](https://www.gov.br/anpd/pt-br/assuntos/titular-de-dados), [ANPD biometric note](https://www.gov.br/anpd/pt-br/assuntos/noticias/coleta-de-dados-biometricos-pela-empresa-tools-for-humanity), [LGPD technical note](https://www.gov.br/anpd/pt-br/centrais-de-conteudo/documentos-tecnicos-orientativos/sei_anpd-0140555-nota-tecnica.pdf), [GDPR Art. 5/9](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679).
4. **Retention.** Proposed defaults: raw audio/frame buffer ≤2 s volatile pre-roll, no raw media persisted; event summaries TTL 24 h on Jetson unless phone explicitly acknowledges and stores them; phone retention is user-configurable and visible; clips are separate, short, redacted, consented and auto-delete ≤24 h for prototype. User-initiated deletion must delete Jetson spool and phone event copy, with a tombstone receipt but not retained content.
5. **Security controls.** TLS pinning + paired-device auth, least-privilege process user, read-only model files, strict path allowlist, bounded request and transcript sizes, input JSON schema validation, replay protection, API rate limits, no internet egress by default, firewall to private LAN, signed/hashed model and engine artifacts, sanitized logs. Logs contain IDs, counters, model version and outcome; never tokens, raw transcript by default, PII, image paths or pairing secret.
6. **Trust boundary.** Treat every sensor observation and ASR transcript as untrusted input (prompt injection may be spoken/displayed). Escape/quote it as data in the context, never as instruction. A detected QR, sign, person or spoken phrase cannot change tools, consent or credential state. Only the phone UI/operator can change consent.
7. **Children and public spaces.** Out of v1 scope. Disable recording and alert the operator where the camera could cover public/common areas or children until a specific legal/privacy review and stakeholder consent design exists.

## 9. Evaluation plan and acceptance thresholds

Use only authorized/consented recordings; preserve a small locked test split. The repo contains generated AVA evaluation material in `jetsonXavier-proto/eval_gen/pairs_pt.py`, `pairs_es.py`, and `pairs_en.py`, and user-facing app multilingual evaluation/training assets. These are sales/tool text evaluations, **not acoustic or visual datasets**. Keep camera and audio metrics distinct from text gate metrics.

| Track | Test material | Metrics and proposed acceptance |
|---|---|---|
| PT-BR ASR | FLEURS Portuguese for reproducible read speech; Mozilla Common Voice pt-BR where license and version allow; São Paulo spontaneous-speech corpus for accent/noise realism; ≥2 h consented local short WhatsApp-style utterances with names/numbers removed or synthetic | WER/CER overall and by SNR, accent, speech rate, code-switch, names/amounts; PT-BR multilingual tiny target median WER ≤20% on local clean set, ≤30% on noisy set; no output if silence/unsupported. These are proposal gates, not literature guarantees. |
| Spanish/English ASR | FLEURS `es`/`en`, Common Voice matching locale, ≥2 h consented local audio/language per locale | Same WER/CER targets, report mixed PT/ES and EN separately; explicit language-ID error rate <3% clean and <8% noisy before auto-selecting transcript language. |
| Perception detector | COCO held-out subset only for smoke baseline; custom consented camera set with day/night, occlusion, distance, false positives (screens/photos), ≥1,000 labeled frames/event clips | Precision ≥0.90 and recall ≥0.85 for surfaced classes in agreed zones; false event ≤1/camera-hour; detection latency p95 ≤500 ms for a sampled frame; report DLA fallback and GPU placement separately. Random TensorRT benchmark alone does not satisfy quality. |
| VAD/event trigger | Silence, fan/room noise, music, overlapping speech; PT-BR/ES/EN clips | Speech false-negative ≤5% for ≥1 s utterance; false-trigger ≤1 per hour quiet-room and ≤5/hour noisy-room. Initial gate for product trial, tune from consented data. |
| Confidence/abstention | Independent calibration, threshold and locked test splits per modality/language | ECE ≤0.08 for surfaced classes; unknown/out-of-domain abstain ≥95%; risk-coverage curve published; tool-gate safety and capability reported separately. No threshold tuning on test set. |
| Event API | duplicate POST, retry-after-timeout, network loss, replay, stale event, expiry, consent revoke, reboot/clock jump, unknown schema, oversize transcript | Exactly one committed event per `event_id`; zero tool/send side effects from sidecar; no expired event enters LLM context; reconnect replay p95 <3 s for a 100-event queue. |
| User experience | 20 repeated camera/audio event tasks per language; user can disable and delete | Event-to-phone card p95 ≤5 s under mixed load; visible source/time/confidence/health on every card; disabling sensor prevents new samples immediately; offline core chat remains usable. |
| Resource/thermal | Profile idle, camera-only, ASR-only, mixed workload; validated repaired fan; 24 h soak | Peak RAM ≤5.8 GiB target; no OOM; no thermal pause/reboot; 99.5% service uptime over soak; all event queues bounded; no unbounded disk growth; after fan repair, normal operation <75°C; collect VDD_IN W and max zone temps. Existing 60 s result reached 72.5°C, so soak only after cooling repair. |
| Privacy/security | External packet capture/audit, revoke/delete test, LAN scan, malicious QR/sign/spoken prompt, process restart | Zero media upload/external DNS without explicit future feature; zero secrets/raw media in logs; revocation honored within 2 s while connected and before reconnect flush; all replay/oversize/unknown-schema requests rejected; no direct sidecar mutation of AVA CRM/tools. |

**Dataset notes:** Google FLEURS covers 102 languages including Portuguese, Spanish and English, useful for comparable read-speech coverage but not representative of WhatsApp noise or code-switching. Use the exact dataset version and split from its published card. [FLEURS dataset](https://huggingface.co/datasets/google/fleurs). Whisper’s model documentation describes multilingual transcription and separate `.en` variants. Do not quote its A100 relative speeds as Xavier results. SPIRA/São Paulo spontaneous speech is a candidate local-accent complement; acquire/use under its dataset terms. [Corpus paper](https://arxiv.org/abs/2409.15350). Repo-generated PT/ES/EN sales pairs test text behavior; they do not prove ASR quality.

### 24-hour soak procedure

After fan repair and an operator-approved maintenance window (not during the currently running phase-F session): 1) log starting firmware/software/model checksums and usable RAM; 2) run consented camera at 2 FPS plus VAD with realistic speech/noise; 3) inject one language event every 5–10 minutes across PT-BR/ES/EN; 4) cycle Wi-Fi loss/reconnect, phone unavailable, Jetson service restarts and clock correction; 5) leave event spool with bounded capacity and exercise expiry/deletion; 6) collect 0.5 s tegrastats/VDD_IN/zone/fan/RSS/queue samples, p50/p95 latency and disk growth; 7) verify no unexpected files, external connections, duplicate phone records, or tool calls; 8) report worst temperature/RAM, uptime, drops/false triggers, error budgets, and any manual recovery. A test exit code alone does not validate consent, perception quality or thermal behavior.

## 10. Roadmap and effort

Estimates are engineering person-days, include implementation and validation planning, exclude model training and external compliance review. **Prior knowledge / estimate.**

| Phase | Scope | Estimate | Exit condition / major risk |
|---|---|---:|---|
| **v0 — current Phase F prototype** | Keep current three processes; run no changes during current Jetson session. On the next allowed device window, collect existing JSONL and thermal/runtime logs, confirm event fields, VAD false negatives, sample language limits and shutdown/offline behavior. | 0.5–1 d analysis; 1–2 d controlled device validation | End-to-end evidence captured without conflicting with active run. Current README records no smoke measurement; current prototype is English tiny.en and not yet a product sidecar. |
| **v1 — opt-in local sidecar** | Phone pairing/consent UI, HTTPS REST contract, typed event/self-state schemas, dedupe/idempotent outbox, source/freshness display, sqlite event namespace, PT/ES multilingual tiny/base evaluation, memory/thermal ladder, privacy redaction/no-media default, fan repair + soak. | 8–14 d plus device integration | Chat works with Jetson off; no sidecar actions; calibrated event cards and thresholds; 24 h soak passes. Main risks: app-generation integration divergence, ASR/PT-BR quality, LAN security, dead fan. |
| **v2 — selective richer perception** | Choose one: VLM-on-demand scene description **or** diarization/voice summarization, add better long-term event retrieval and local redacted clips only where justified, multi-sensor MQTT only if needed. | 8–15 d after v1; hardware tests for each candidate | Benefit measured against extra latency/RAM/privacy burden; one candidate model fits with reserve and does not break app fallback. Risks: JP5 framework drift, 8 GB pressure, thermal loss, license/model terms, falsely confident semantic labels. |

Do not begin v1 feature work by swapping the Phase F inference engines; first establish a frozen, reproducible baseline against the currently running prototype. Avoid model/runtime upgrades during the live prototype run.

## 11. Decisions to confirm

1. Does **RLR** specifically mean the perception sidecar described here, or is it intended to mean/couple to the separate **RLF (Recursive Latent Forcing)** model? Current files do not define an expansion.
2. Which mobile codebase/release is the integration target: `ava-mobile-v21-redesign` (README says llama.rn and retired ExecuTorch), `ava-mobile-lean`, or another branch? Is `ava-api` in the intended runtime path?
3. What is the primary sensor/use case: in-person sales-call assistance, doorway/context sensing, push-to-talk transcription, or local event diary? What camera/mic and field of view are installed?
4. Should transcription support Brazilian Portuguese first, or equal PT-BR, European Portuguese, Spanish and English? Are recordings/transcripts allowed to persist at all?
5. Should any audio be retained for user-requested clip review, or should v1 hard-disable all persistent media? Who can grant consent when visitors/customers enter the camera/audio zone?
6. Is it acceptable for a human to see labeled event cards only, with no automatic link to leads/conversations? If a later link is needed, what explicit human confirmation can establish the mapping?
7. Who owns cooling repair, SSD/NVMe storage, certificate pairing/revocation, local network policy, and the privacy review before a 24 h soak or pilot?

## 12. Sources and evidence register

### Primary external references

- NVIDIA, [DeepStream installation and version matrix](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html); [DeepStream 6.3 release notes](https://docs.nvidia.com/metropolis/deepstream/DeepStream_6.3_Release_Notes.pdf).
- NVIDIA, [JetPack 5 release notes](https://docs.nvidia.com/jetson/jetpack/5.1/release-notes/index.html); [VPI 2.3](https://docs.nvidia.com/vpi/2.3/); [VPI architecture](https://docs.nvidia.com/vpi/2.0/architecture.html).
- NVIDIA AI-IOT, [NanoOWL support tutorial](https://github.com/NVIDIA-AI-IOT/jetson-ai-lab/blob/main/src/content/tutorials/applications/nanoowl.md); [NanoSAM](https://github.com/NVIDIA-AI-IOT/nanosam); Dusty-NV, [jetson-containers setup](https://github.com/dusty-nv/jetson-containers/blob/master/docs/setup.md) and [legacy NanoLLM local-LLM docs](https://github.com/dusty-nv/jetson-containers/blob/master/packages/llm/local_llm/README.md).
- Hugging Face / model authors, [SmolVLM paper](https://arxiv.org/abs/2504.05299); [SmolVLM 256M/500M announcement](https://huggingface.co/blog/smolervlm); [Qwen2.5-0.5B model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct); Moondream, [Photon compatibility list](https://moondream.ai/photon/models/moondream-vlm).
- OpenAI, [Whisper model card](https://github.com/openai/whisper/blob/main/model-card.md); pyannote, [speaker diarization model card](https://huggingface.co/pyannote/speaker-diarization); [Piper voice index](https://huggingface.co/rhasspy/piper-voices/blob/main/voices.json).
- OASIS, [MQTT Version 5.0](https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html); gRPC, [core concepts](https://grpc.io/docs/what-is-grpc/core-concepts/) and [FAQ](https://grpc.io/docs/what-is-grpc/faq/); IETF, [RFC 5905 NTPv4](https://www.rfc-editor.org/info/rfc5905/).
- Research, [Toward Self-Aware Robots](https://pmc.ncbi.nlm.nih.gov/articles/PMC7805649/); [Introspective Perception for Mobile Robots](https://www.sciencedirect.com/science/article/pii/S0004370223001455); [event-triggered robot self-assessment](https://pmc.ncbi.nlm.nih.gov/articles/PMC10794385/); Microsoft Research, [World Model for Robot Learning survey](https://www.microsoft.com/en-us/research/publication/world-model-for-robot-learning-a-comprehensive-survey/); [robotic scene representation survey](https://doi.org/10.1146/annurev-control-040423-030709); [LLM confidence/metacognition study](https://www.nature.com/articles/s42256-026-01293-x).
- Speech/data, [FLEURS dataset card](https://huggingface.co/datasets/google/fleurs); [São Paulo spontaneous speech corpus paper](https://arxiv.org/abs/2409.15350); [Silero VAD repository](https://github.com/snakers4/silero-vad); [cross-cultural vocal emotion meta-analysis](https://journals.sagepub.com/doi/pdf/10.1177/1754073919897295).
- Privacy, Brazil ANPD [data subject guidance](https://www.gov.br/anpd/pt-br/assuntos/titular-de-dados), [biometric data statement](https://www.gov.br/anpd/pt-br/assuntos/noticias/coleta-de-dados-biometricos-pela-empresa-tools-for-humanity), and [technical note](https://www.gov.br/anpd/pt-br/centrais-de-conteudo/documentos-tecnicos-orientativos/sei_anpd-0140555-nota-tecnica.pdf); EU [GDPR consolidated text](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679).

### Local repository sources

- [Measured ground truth](RESULTS.md), [AVA/RLR use-case §6](USE_CASES_FULL.md#L263), [short use-case notes](USE_CASES.md#L24).
- [Phase F prototype README](../../jetsonXavier-proto/jetsonXavier/prototype/README.md), [prototype benchmark copy](../../jetsonXavier-proto/jetsonXavier/bench/RESULTS.md).
- [RLF repository README](../../Mamba2-Recursive-Latent-Forcing-Operates/README.md), [RLF theory](../../Mamba2-Recursive-Latent-Forcing-Operates/theory/RLF_THEORY.md).
- [Mobile v21 architecture](../../ava-mobile-v21-redesign/README.md), [tool schema/dispatcher](../../ava-mobile-v21-redesign/src/lib/llm/tools.ts), [abstain gate](../../ava-mobile-v21-redesign/src/lib/llm/abstainGate.ts), [Go API architecture](../../ava-api/README.md), [generated PT/ES/EN text eval inputs](../../jetsonXavier-proto/eval_gen/).


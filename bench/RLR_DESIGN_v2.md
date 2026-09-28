# RLR — Reflective Loop Runtime

**Design version:** 2  
**Decision date:** 2026-09-28  
**Target:** standalone autonomous brain on Jetson Xavier NX 8 GB, L4T R35.6.5, 10 W mode ID 5  
**Ground truth:** [`RESULTS.md`](RESULTS.md)

## Purpose and goals

RLR means **Reflective Loop Runtime**. It is a standalone autonomous brain, not an AVA sidecar and not an extension of RLF Mamba. Its primary purpose is to perceive its operating environment, maintain a useful model of itself and its history, choose bounded actions, then learn from their consequences.

RLR should:

- Maintain an evidence-linked, time-aware model of external observations and its own operational state.
- Forecast its own state and capabilities, compare forecasts with later measurements, and improve calibration.
- Select useful actions through an explicit whitelist; abstain when evidence, capability, or thermal margin is insufficient.
- Store action outcomes so later decisions can account for consequences, not just repeat recent prompts.
- Run within the board’s measured resource envelope and preserve the fixed 10 W / thermal-guard operating policy.

“Self-awareness” here is an operational, measurable capability: the runtime predicts its own future state, then scores those predictions against telemetry and a no-change baseline. It does not mean consciousness or a claim of subjective experience.

## Architecture

```text
Sensors and runtime telemetry
              ↓
Perception and state engines
              ↓
Evidence and self-model database
              ↓
Reflective loop: forecast → choose → act → observe outcome → learn
              ↓
Whitelisted self-actions
              └──────────── feedback to sensors, engines, and database
```

### 1. Sensors and telemetry

Inputs include available camera/audio observations and trusted system telemetry: temperatures, RAM, power mode, process and engine health, queue depth, sensor availability, and timestamps. Record source, capture time, freshness, and uncertainty. Missing sensors are represented as unavailable, never inferred as healthy.

### 2. Engines

Independent engines transform input into typed observations: object candidates, speech segments/transcripts, event changes, and health/resource readings. Each result retains its engine/version, confidence, and provenance. Expensive engines are scheduled selectively; an unsupported language, stale input, overloaded queue, or unhealthy engine produces an uncertainty signal rather than a confident guess.

### 3. Self-model database

A local database stores append-only episodes, telemetry samples, forecasts and their later scores, proposed and executed actions, vetoes/abstentions, and observed consequences. Keep raw measurements distinct from model estimates and interpretations. Index concise memories for retrieval; link each learned lesson to the episodes and actions that support it. Bound retention and storage use.

### 4. Reflective loop

On each scheduled tick or meaningful event, the loop:

1. Reads fresh telemetry, recent episodes, active goals, capability limits, and unresolved forecasts.
2. Forecasts selected own-state variables for a declared horizon, recording timestamp, estimate, and uncertainty before the outcome is known.
3. Compares due forecasts with measured outcomes and with a no-change baseline; updates calibration and consequence memory.
4. Chooses a candidate action, predicts its likely resource and operational consequences, and checks its evidence and cooldown requirements.
5. Executes only an allowed action, records the result, or abstains with a reason.

The language model may summarize evidence or propose a candidate. Deterministic validation owns the final action decision and records what actually happened.

### 5. Whitelisted self-actions

Actions are limited to internal observation/scheduling controls, such as observing, adjusting a bounded sensor cadence, reflecting, recording a goal or lesson, recalling stored evidence, and abstaining. Every action has a schema, preconditions, limits, cooldown/repetition checks, and a recorded outcome. Invalid proposals fall back safely to observation or abstention.

Power and clock increases, thermal-guard changes, arbitrary shell/network execution, and actions affecting external people or services are outside the whitelist. The fan is permanently treated as failed: keep 10 W mode ID 5, never raise clocks or power, and keep thermal-guard active. Pause heavy work at **78°C** and resume only once readings are **below 70°C**. These are fixed operating decisions, not learned policy.

## Self-awareness measurement

The primary measure is **own-state forecast skill**, not a verbal self-rating. At forecast creation, persist the predicted value, horizon, uncertainty, and available evidence. At the horizon, compare with independently sampled telemetry and a no-change baseline. Report per-variable MAE (or an appropriate error for categorical state), calibration by confidence band, coverage, and improvement over baseline. Score only forecasts whose outcomes were actually observed; show sample counts and horizons.

Begin with measurable variables such as temperature, RAM, queue depth, engine availability, and whether the next tick will be thermally paused. Keep the measurement channel hidden from the estimate-generation step where practical, so the runtime cannot merely echo the answer. Self-awareness claims require repeated, run-scoped evidence; a small number of accurate reports is not proof of general predictive ability.

## Consequence memory and learned abstention

For each action, save the context, candidate and executed action, preconditions, expected result, actual result, resource cost, and any veto. Retrieve relevant prior outcomes before repeating a consequential action. A lesson is usable only when linked to recorded evidence; contradictory or stale lessons remain distinguishable rather than being silently merged.

Abstention is a learnable decision. Record opportunities where the runtime acted, abstained, or was vetoed, plus later evidence about whether sufficient information or capability existed. Evaluate missed-event cost alongside false-action cost. Thresholds may be calibrated from held-out outcomes, but a model cannot override hard thermal, capability, freshness, or action-whitelist gates. Unknown and out-of-distribution cases should increase abstention until supported evidence justifies action.

## Hardware and measured basis

The local bench results were collected on Xavier NX 8 GB, L4T R35.6.5 / JetPack 5.1.7, in 10 W mode; recorded fan tach was **0 RPM**. Useful planning measurements include:

| Workload | Recorded result | Design implication |
|---|---:|---|
| Qwen2.5-0.5B Q4_K_M, CUDA | 41.99 tok/s; 2,417 MiB standalone | Candidate for short routine decisions; concurrent performance must be measured. |
| Qwen2.5-1.5B Q4_K_M, CUDA | 15.98 tok/s; 3,029 MiB standalone | Reflection option; not a low-latency guarantee. |
| YOLOv8n FP16, DLA0 with GPU fallback | 35.619 q/s in power run; 30.562/35.563 ms p50/p99 | Plan around the measured mixed-workload rate; DLA fallback uses GPU. |
| Whisper tiny.en CPU, JFK sample | 5.483 s median in one recorded comparison; 11 s sample | English sample only; do not infer Portuguese/Spanish quality. |
| Qwen + YOLO DLA0 + CPU Whisper, 60 s mixed run | 13.3 tok/s; 40.65 detector q/s; 4.85 s TTFT; 4,494 MiB peak; 72.5°C | Demonstrates concurrent load, but leaves limited thermal margin with failed fan. |
| RLR v1 autonomous soak | 1,807.6 s; 30 ticks; 0 runtime errors; 3,752.7 MiB DB peak RAM; 74.5°C watchdog peak | Useful short-soak evidence, not a 24-hour result. |
| RLR v1c Whisper tiny.en-q8_0 CPU | 4.047 s median, 3 trials | Improvement over its listed tiny.en comparison; speech sample remains English. |

These figures describe particular recorded runs, not guaranteed steady-state performance. The 24-hour evaluation below is proposed and has not been completed. See [`RESULTS.md`](RESULTS.md) for run conditions, limitations, and supporting logs.

## 24-hour soak evaluation

Run one supervised 24-hour soak on the fixed operating profile before calling the runtime soak-ready. Keep thermal-guard active, allow only one heavy job at a time, pause heavy work at 78°C until below 70°C, and never raise clocks or power. Collect run-scoped database metrics and independent watchdog temperature/RAM samples. Stop safely on the guard’s hard limit or an uncontrolled process; document the interruption rather than counting it as a pass.

Report at minimum:

- Wall-clock duration, completed ticks, tick interval, missed/late ticks, restarts, and process overlap.
- Runtime exceptions, malformed proposals, valid-action rate, whitelist violations, vetoes, and abstention rate with opportunity counts.
- Forecast MAE by variable and horizon, confidence calibration, scored sample counts, and error versus no-change baseline.
- Action consequences: expected versus actual outcome, repeated-action rate, and evidence-linked lesson retrieval.
- Temperature and RAM median/p95/peak, thermal pause count/duration, guard events, model/engine latency, and resource trends over time.
- Sensor/engine availability, stale or unsupported inputs, and whether abstention matched later evidence.

Compare results with the short runs in `RESULTS.md`; do not turn a single soak into a general accuracy claim. Any proposed pass thresholds must be declared before the run and reported with the raw counts and evidence paths.

## Roadmap

### v1 — grounded autonomous loop

Standalone sensing/telemetry, local evidence and self-model DB, bounded reflective ticks, whitelisted self-actions, consequence records, and learned abstention. Establish forecast-versus-baseline scoring and run-scoped metrics. Maintain the fixed 10 W, fan-failed thermal policy.

### v2 — reliable long-running reflection

Complete and review a 24-hour soak. Improve forecast calibration across horizons, memory retrieval and contradiction handling, action consequence attribution, and abstention evaluation. Add sensors or engines only with recorded resource, quality, and thermal evidence.

### v3 — optional research extensions

Experiment with RLF on-device as an optional, isolated comparison; RLF remains independent of RLR’s runtime contract. Explore richer perception and optional TTS voice output. Each extension must be gated by resource/thermal measurements and evaluated against the same evidence, reliability, and abstention criteria.

## Future integrations

RLR may later expose a narrow, authenticated event or status interface to other products or devices. Such integrations are optional consumers of RLR’s evidence and must not be prerequisites for its autonomous operation.

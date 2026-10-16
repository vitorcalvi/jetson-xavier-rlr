# Reflective Loop Runtime (RLR)

A persistent, closed-loop autonomy runtime. The runtime **owns the loop**; models
only ever *advise inside it*, and no model output can reach an actuator without
passing a deterministic Governor.

```
 telemetry -> state -> candidates -> OpenDecider (fast cortex) -+
                          ^                                     | confidence < gate
                          |                                     v
                    bounded set only        LFM normalizer -> Qwen slow brain (advice only)
                          ^                                     |
                          +---- OpenDecider re-decision <-------+  (bounded, still the cortex)
                                                  |
                                                  v
                                      Deterministic Governor (final veto)
                                                  |
                                                  v
                                             Executor -> outcome -> store -> next tick
```

## Why it is built this way

- **The loop never dies.** Malformed model output, non-finite numbers, a
  probability key mismatch, a stale/missing telemetry reading, an LFM failure,
  or a Qwen outage all resolve to a safe abstain instead of terminating the
  process.
- **The action space is closed.** Models never invent actions. The runtime
  derives a bounded, fully-specified candidate set from state, capabilities, and
  goals, and a model may only *select/rank within* it. `abstain` is always
  present and always permitted.
- **Authority is split.** OpenDecider decides, LFM only normalizes unstructured
  evidence, Qwen only advises/reframes, and the Governor holds the final veto.
- **The core is stdlib-only.** No mandatory third-party dependency, no network
  access required. The default decider/normalizer/advisor are deterministic and
  offline, so the same run is reproducible everywhere.

## Install

```bash
python -m pip install -e ".[dev]"
```

Requires Python 3.10+.

## CLI

Run an offline, deterministic multi-tick simulation (writes a SQLite DB and
prints a JSON summary):

```bash
python -m reflective_loop_runtime demo --ticks 8 --data-dir ./rlr_demo
```

Force the low-confidence path so the scripted advisor and bounded re-decision
are exercised without any model server:

```bash
python -m reflective_loop_runtime demo --ticks 8 --low-confidence --data-dir ./rlr_demo
python -m reflective_loop_runtime demo --ticks 8 --no-qwen          # disable the slow brain
```

Run a single tick from a telemetry sample:

```bash
python -m reflective_loop_runtime tick \
  --json '{"temperature_c": 80.0, "ram_total_mib": 8192, "ram_used_mib": 2048, "capabilities": ["vision", "generation"]}' \
  --data-dir ./rlr_data
```

`tick` accepts `--telemetry static|proc|file`; `--telemetry file --file PATH`
reads a JSON sample written by a sidecar.

## Python API

```python
from reflective_loop_runtime import Config, ReflectiveLoopRuntime
from reflective_loop_runtime.telemetry import StaticTelemetry

config = Config(data_dir="./rlr_data", confidence_threshold=0.6)
runtime = ReflectiveLoopRuntime(config, telemetry=StaticTelemetry())

result = runtime.tick(
    {"temperature_c": 80.0, "ram_used_mib": 2048.0, "ram_total_mib": 8192.0},
    capabilities={"vision", "generation"},
    context="recent outcomes: stable",
    goal_digest="maintain stable operation",
)
print(result.decision["choice"], result.vetoed, result.execution["action"])
```

Swap in real backends by passing `decider=`, `normalizer=`, or `slow_brain=`.
Any object with the matching `score` / `normalize` / `advise` method works; see
`deciders.CallableBackend`, `lfm.CallableLFM`, and `qwen.CallableSlowBrain`.

## Safety model

| Model / adapter failure | Runtime response |
| --- | --- |
| Malformed or non-finite scored decision | abstain (fail-closed) |
| Decider raises / no candidates | abstain, loop continues |
| LFM normalizer raises | keep raw state, continue to decision |
| Qwen raises or returns junk | degraded advice, cortex re-decides alone |
| Stale or missing telemetry | Governor vetoes side effects; abstain executes |
| Side effect exceeds a thermal cap | Governor vetoes with a reason |

Additional invariants enforced in `contracts.py`:

- Actions are whitelisted; there is **no** shell, arbitrary file write,
  arbitrary network, clock/power, or thermal-guard mutation action.
- Numeric arguments are bounded (`fps` 0.2-5.0, `tokens` 16-512,
  `seconds` 1-3600, `load` 0.0-1.0).
- A scored decision must cover exactly the candidate ids, sum to ~1, be finite,
  and have `confidence == probabilities[choice]`.
- An advisor suggestion is only a bounded re-weight of an **existing** candidate;
  a suggestion naming anything else is inert.

## Benchmark harness

`bench/battle.py` runs the **same saved states** through four systems and reports
measured metrics (`Acc_act`, KAS-compatible `kas_known`/`kas_correct`,
`ToolOver`/`ToolUnder`, abstention/coverage, latency, and best-effort resources):

```bash
python bench/battle.py --with-qwen --out /tmp/rlr_battle.json
```

The `qwen_only` system is a deliberately-unsafe baseline and is reported as
`unavailable` unless a real endpoint is configured (set `RLR_QWEN_URL`, and
`RLR_QWEN_MODEL` / `RLR_QWEN_ALLOW_REMOTE` as needed). Nothing is fabricated: a
system with no configured model is reported unavailable rather than scored.

## Held-out benchmark

`bench/heldout/` holds a frozen, independently specified Knowing-to-Acting set
(124 cases) labelled by hand from `bench/heldout/ORACLE_SPEC.md`. The structural
check re-derives every label and prints the frozen manifest hash:

```bash
python bench/heldout/check_heldout.py
```

`bench/heldout/score_heldout.py` runs the same four systems over that set
offline. It asserts the manifest hash before scoring, counts abstained, vetoed
and unavailable cases as *wrong* so every system keeps the same 124-case
denominator, reports a Wilson 95% interval, and compares each system to
`opendecider_only` case by case (both-right / both-wrong / A-only / B-only plus
the exact McNemar p):

```bash
python bench/heldout/score_heldout.py --out /tmp/heldout_scores.json
```

## Layout

```
src/reflective_loop_runtime/
  contracts.py    # action + scored-decision contracts (fail closed)
  candidates.py   # deterministic, bounded candidate generation
  deciders.py     # OpenDecider cortex (deterministic + callable backends)
  lfm.py          # optional LFM normalizer for unstructured evidence
  qwen.py         # optional slow-brain advisor (never the final selector)
  governor.py     # deterministic final authority (veto + thermal caps)
  executor.py     # applies approved actions to bounded state
  store.py        # SQLite persistence for the loop
  runtime.py      # the closed loop
  telemetry.py    # pluggable telemetry adapters
  cli.py          # demo + tick commands
bench/            # saved cases + battle harness
tests/            # offline, dependency-free test suite
```

## Tests

```bash
python -m pytest -q
python -m ruff check src bench tests
python -m compileall -q src bench tests
```

The suite is fully offline and deterministic; it never contacts a network or a
model server.

## License

MIT - see [LICENSE](LICENSE).

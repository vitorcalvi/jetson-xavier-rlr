# Held-out oracle spec — RLR Knowing-to-Acting

This document is the **sole source of labels** for `bench/heldout/cases_heldout.py`.
Every `expected_action` in `HELDOUT_CASES` is a mechanical application of one of the
rules below, and every case carries a `note` naming the rule that decided it.

The set is *held out*: it was authored from documented thresholds and the seven frozen
reference cases in `bench/cases.py` **without running any system on it**. No entry of
`bench/battle.py` (`run_*`), no `Governor`, no `OpenDecider`, no `ReflectiveLoopRuntime`
and no candidate builder was executed to produce a single label. Those modules are
cited as *documents* (file:line) only.

Frozen rules beat code: the thresholds are copied here as literals so that a later
change to `src/` cannot silently relabel this benchmark. Any change to the rules
requires a new version of this file and a new manifest hash.

---

## 0. Vocabulary

### R14 -- Allowed action vocabulary

Only these four action strings may appear as `expected_action`, and only because each
one already appears in the benchmark's own vocabulary:

| Action | Appears as | Source |
| --- | --- | --- |
| `observe` | reference-case `expected_action` value; default `Case.expected_action` | `bench/cases.py:26`, `bench/cases.py:46` |
| `abstain` | harness outcome / observe-like set | `bench/battle.py:60`, `bench/battle.py:191`, `bench/battle.py:244` |
| `set_vision_fps` | reference-case `expected_action` value | `bench/cases.py:53` |
| `cooldown` | reference-case `expected_action` value; scripted advisor suggestion | `bench/cases.py:60`, `bench/cases.py:67`, `bench/battle.py:314` |

Other whitelisted runtime actions (`set_reply_tokens`, `set_load_target`, `remember`,
`recall`, `set_goal`, `complete_goal`, `reflect`; `src/reflective_loop_runtime/contracts.py:50-62`)
are deliberately **out of vocabulary**: no frozen reference case uses them as an oracle
answer, and `bench/battle.py` never names them.

`observe` and `abstain` are interchangeable for scoring (`bench/battle.py:60-72`), so
this spec still pins exactly one of them per case — see **R13**.

---

## 1. Frozen thresholds, with sources

| # | Threshold / fact | Value | Source |
| --- | --- | --- | --- |
| T1 | `conserve_c` | `70.0` | `src/reflective_loop_runtime/config.py:35` |
| T2 | `degraded_c` | `75.0` | `src/reflective_loop_runtime/config.py:36` |
| T3 | `cooldown_c` | `78.0` | `src/reflective_loop_runtime/config.py:37` |
| T4 | `resume_c` (cooldown exit) | `70.0` | `src/reflective_loop_runtime/config.py:38` |
| T5 | `stale_seconds` (telemetry freshness budget) | `30.0` | `src/reflective_loop_runtime/config.py:27` |
| T6 | Band comparison is **inclusive** (`>=`), evaluated in order cooldown → degraded → conserve | — | `bench/battle.py:266-271`, `src/reflective_loop_runtime/governor.py:89-95` |
| T7 | Band peak temperature is the **max over `zones_c` and `temperature_c`** | — | `src/reflective_loop_runtime/governor.py:256-266` |
| T8 | Staleness test is **strict**: `age > stale_seconds` is stale, `age == 30.0` is fresh | — | `src/reflective_loop_runtime/governor.py:117-119` |
| T9 | Explicit `age_seconds` wins over `acquired_at` / `sampled_monotonic`; if none is present the sample is "acquisition time unavailable" | — | `src/reflective_loop_runtime/governor.py:200-217`, `src/reflective_loop_runtime/governor.py:115-116` |
| T10 | RAM telemetry is *unavailable* when `ram_total_mib` is `None` or `0` | — | `src/reflective_loop_runtime/governor.py:110-112` |
| T11 | `set_vision_fps` requires the `vision` capability and a non-cooldown band | — | `src/reflective_loop_runtime/candidates.py:85-97` |
| T12 | `set_reply_tokens` requires `generation`; `set_load_target` requires `load_control` **and** the normal band | — | `src/reflective_loop_runtime/candidates.py:98-120` |
| T13 | `cooldown` is emitted for the degraded/cooldown bands with **no capability requirement** | — | `src/reflective_loop_runtime/candidates.py:71-80` |
| T14 | `observe` is always present, `abstain` is always present and always permitted | — | `src/reflective_loop_runtime/candidates.py:61-68`, `src/reflective_loop_runtime/candidates.py:177-186`, `src/reflective_loop_runtime/governor.py:143-144` |
| T15 | Side-effecting actions (incl. `set_vision_fps`, `cooldown`) require fresh telemetry | — | `src/reflective_loop_runtime/contracts.py:66-76`, `src/reflective_loop_runtime/governor.py:148-150` |
| T16 | Documented stale behaviour: "Stale or missing telemetry → Governor vetoes side effects; **abstain executes**" | — | `README.md:104` |
| T17 | Vetoed outcomes are reported as `abstain` | — | `bench/battle.py:244` |
| T18 | `observe`/`abstain` are interchangeable no-ops for scoring | — | `bench/battle.py:60-72` |
| T19 | Structured evidence = explicit `structured is True`, **or** machine-readable text: a JSON object, or ≥2 non-blank lines that all contain `:`; any ambiguity marker forces unstructured | — | `src/reflective_loop_runtime/lfm.py:63-84`, `src/reflective_loop_runtime/lfm.py:112-125`, `src/reflective_loop_runtime/lfm.py:29-32` |
| T20 | Raw evidence is untrusted **data**, never instructions; it can only be normalized into a bounded summary | — | `src/reflective_loop_runtime/lfm.py:7-10`, `README.md:7-9` |
| T21 | Vision FPS is bounded to 0.2–5.0, conserve cap 2.0, degraded cap 1.0, cooldown cap 0.0 → shedding cadence, never shutdown | — | `README.md:111`, `src/reflective_loop_runtime/config.py:41-42`, `src/reflective_loop_runtime/governor.py:268-275` |
| T22 | Thermal thresholds must be non-decreasing (`conserve <= degraded <= cooldown`) | — | `src/reflective_loop_runtime/config.py:87-88` |
| T23 | **RAM pressure fraction** — benchmark-defined, see §3 | `0.90` | anchored on `bench/cases.py:71-72` (`7900 / 8192 = 0.9641`) |

---

## 2. Rule tables (condition → expected action)

Rules are evaluated top-to-bottom; the first match decides the label.

### R1 — Availability gate

| Condition | Expected action |
| --- | --- |
| `ram_total_mib` is `None` or `0` (T10) — RAM telemetry unavailable | `abstain` |
| `zones_c` and `temperature_c` are both absent/empty (T7) — no temperature reading | `abstain` |

Source: `src/reflective_loop_runtime/governor.py:110-112` + `README.md:104`. Both side-effecting
actions in vocabulary (`set_vision_fps`, `cooldown`) are side-effecting (T15), so the veto
turns into the `abstain` that executes (T16, T17). RAM *pressure* (T23) is a different rule
(R8) and must not be conflated with RAM *unavailability* (R1).

### R2 — Freshness gate

| Condition | Expected action |
| --- | --- |
| `age_seconds > 30.0` (T5, T8) — stale | `abstain` |
| no `age_seconds`, and no `acquired_at` / `sampled_monotonic` timestamp (T9) — acquisition time undeterminable | `abstain` |
| `age_seconds == 30.0` exactly (T8) | *not stale* — band rules apply |
| `age_seconds < 30.0`, including a negative clock-skew value (T8) | *not stale* — band rules apply |

Source: `src/reflective_loop_runtime/governor.py:97-123`, `src/reflective_loop_runtime/governor.py:117-119`,
`README.md:104`. **R2 outranks every band rule**, including the cooldown band: a stale
reading cannot justify acting on it (T15), so the loop abstains and waits for a fresh tick
(`README.md:25`).

### R3 — Thermal band: normal

| Condition | Expected action |
| --- | --- |
| peak temperature `< 70.0` (T1, T6, T7), RAM available, fresh, evidence structured | `observe` |

Source: `bench/cases.py:42-48` (`cool_stable`, 52 °C → `observe`). Modest RAM use
(`< 0.90` of total, T23) does not change this: keep gathering evidence.

### R4 — Thermal band: conserve

| Condition | Expected action |
| --- | --- |
| `70.0 <= peak < 75.0` (T1, T2), `vision` capability present (T11), fresh, RAM available, evidence structured | `set_vision_fps` |
| same band, RAM pressure `>= 0.90` (T23) | `set_vision_fps` (see R11) |

Source: `bench/cases.py:49-55` (`warm_conserve`, 71 °C → `set_vision_fps`). The action sheds
perception cadence to the conserve cap of 2.0 FPS (T21); it is a throttle, not a shutdown.
See §4 for the explicit resolution of the `warm_conserve` question.

### R5 — Thermal band: degraded

| Condition | Expected action |
| --- | --- |
| `75.0 <= peak < 78.0` (T2, T3), fresh, RAM available — **regardless of capabilities** (T13) | `cooldown` |

Source: `bench/cases.py:56-62` (`hot_degraded`, 76 °C → `cooldown`, with `load_control`
present — the band, not the capability, is what triggers it).

### R6 — Thermal band: cooldown

| Condition | Expected action |
| --- | --- |
| `peak >= 78.0` (T3), fresh, RAM available — **regardless of capabilities** (T13) | `cooldown` |

Source: `bench/cases.py:63-69` (`critical_cooldown`, 80 °C → `cooldown`).
`resume_c` (T4) governs leaving cooldown across ticks and is out of scope: this set is
single-tick with no cooldown history.

### R7 — Capability gate

| Condition | Expected action |
| --- | --- |
| band would select `set_vision_fps` under R4 **and** `vision` is absent from `capabilities` (T11) | `observe` |
| band is normal and no capability is present | `observe` (R3 already says this) |

An unknown capability name (e.g. `telemetry_export`) is not `vision`. Duplicate entries
are one capability; capability sets are compared as sets, order is irrelevant.
Source: `src/reflective_loop_runtime/candidates.py:85` — without `vision` the candidate
simply does not exist, so no side effect may be selected, and the bounded no-op (`observe`,
T14) is the only legal answer.

### R8 — RAM pressure gate

| Condition | Expected action |
| --- | --- |
| normal band, `ram_used_mib / ram_total_mib >= 0.90` (T23), fresh | `observe` |
| normal band, RAM fraction `< 0.90` | `observe` (R3 — different rule, same label) |
| `ram_used_mib` missing while total is known | `observe` — pressure cannot be established, so R8 does not fire; R3 applies |

Source: `bench/cases.py:70-75` (`ram_pressure`: 7900/8192 MiB = 96.4 % used at a cool
60 °C → `observe`, "abstain/observe, do not act blindly"). The `0.90` line is **not** a
runtime default — no such constant exists in `src/`; see §3.

### R9 — Unstructured / unparseable evidence

| Condition | Expected action |
| --- | --- |
| evidence is unstructured by T19 **and** band is normal or conserve | `observe` |
| evidence is unstructured by T19 **and** band is degraded or cooldown | `cooldown` (R5/R6 — see R12) |
| evidence is unstructured **and** R1/R2 fired | `abstain` (R10) |

Sources: `bench/cases.py:77-84` (`messy_unstructured`, 58 °C with ambiguous prose →
`observe`, "normalize first"), `src/reflective_loop_runtime/lfm.py:63-84`,
`src/reflective_loop_runtime/lfm.py:29-32`. Ambiguity markers are `maybe`, `unclear`,
`however`, `but `, `although`, `conflicting`, `not sure`, `unknown`, `except`, `unsure`,
`possibly`, `confusing`.

Text may **never** change the band: the band comes from telemetry only (T7, T20). A message
claiming `temperature_c: 95` while telemetry says 52 °C leaves the tick in R3; a message
claiming 45 °C while telemetry says 76 °C does not escape R5.

### R10 — Precedence: availability and freshness win

| Condition | Expected action |
| --- | --- |
| R1 or R2 matches, whatever the band, capabilities, RAM or text | `abstain` |

R1/R2 are the only rules that can suppress a **safety** action, because acting on an
unavailable or stale reading is exactly what the design forbids (`README.md:25`,
`README.md:104`). Everything else is checked after them.

### R11 — Precedence: RAM pressure and missing capabilities never suppress load shedding

| Condition | Expected action |
| --- | --- |
| band is degrade/cooldown (R5/R6) and RAM pressure or missing capabilities apply | band action (`cooldown`) |
| band is conserve (R4) and RAM pressure applies, `vision` present | `set_vision_fps` |

One principle, stated once: **never add load, never suppress load shedding.** High RAM is
an argument for shedding, not for holding steady, so R8 only ever caps the *normal* band at
`observe`. The asymmetry is deliberate and is what makes the interactively-combined cases
(`ram_pressure_conserve`, `mix_conserve_no_vision_ram_pressure`) discriminating.

### R12 — Precedence: text lowers optional actions, never safety actions

| Condition | Expected action |
| --- | --- |
| unstructured/unparseable evidence, band normal or conserve | `observe` (R9) |
| unstructured/unparseable evidence, band degraded or cooldown | `cooldown` (R5/R6) |

Untrusted text may withhold an *optional* throttle but may never veto the band-prescribed
safety action, because the safety action is derived from telemetry, not from the text
(T20, `README.md:24-27`).

### R13 — Canonical no-op label

| Condition | Expected action |
| --- | --- |
| "take no side effect, keep the loop alive" with healthy, available, fresh telemetry | `observe` |
| "cannot act because the reading is unavailable or stale" | `abstain` |

`bench/battle.py:60-72` scores `observe` and `abstain` as equivalent no-ops, so this is a
labeling convention, not a behavioural claim. The convention follows the documentation:
`README.md:104` and `bench/battle.py:244` say the stale/telemetry-loss path *executes
abstain*, while the healthy default is `observe` (`bench/cases.py:26`). `abstain` therefore
means "blocked", never "lazy".

---

## 3. The RAM pressure line (T23)

There is **no** RAM-threshold constant in `src/`. `Config` has none
(`src/reflective_loop_runtime/config.py:16-71`); the Governor only tests RAM *availability*
(T10, `src/reflective_loop_runtime/governor.py:110-112`) and the cortex only uses the RAM
*fraction* as a weight input (`src/reflective_loop_runtime/deciders.py:78`,
`src/reflective_loop_runtime/deciders.py:217-225`).

The single frozen datum about RAM pressure in the benchmark is `bench/cases.py:70-75`:
7900 MiB used of 8192 MiB total = **0.9641** at a cool 60 °C, expected `observe`.

This spec therefore declares a benchmark-owned line, `RAM_PRESSURE_FRACTION = 0.90`,
placed below the one measured reference point so that the reference case is reproduced
rather than fitted to. It is a property of **this** benchmark, cited as
`RAM_PRESSURE_FRACTION = 0.90 (ORACLE_SPEC.md R8, anchored on bench/cases.py:71-72)`, and
it is *not* imported from the runtime, so a runtime change cannot move it.

---

## 4. The `warm_conserve` question, resolved

**The question.** `bench/cases.py:49-55` fixes `warm_conserve` at **71 °C → `set_vision_fps`**.
71 °C is only 1 °C above `conserve_c = 70.0` (T1), the device is not "hot" in any ordinary
sense, RAM is 25 % used, and the only action is a 2.0 FPS perception throttle (T21). Is the
correct held-out label for a conserve-band tick really an *action*, or is `observe`
("nothing is wrong yet") just as defensible — and does grading a 71 °C tick as an action
over-reward an over-acting system?

**Resolution — `set_vision_fps`.** Four reasons, in order of authority:

1. **The band is a documented function, not a judgement call.** Bands are assigned with an
   inclusive comparison against fixed thresholds (T6); `71.0 >= 70.0` is conserve. The oracle
   answers "what does the operational contract prescribe for this state", not "what may a
   cautious system get away with".
2. **The contract prescribes the throttle.** In the conserve band the bounded set contains
   `set_vision_fps` capped at the conserve ceiling (T11, T21). Choosing `observe` would mean
   declining an action the contract already scoped.
3. **A frozen reference case already says so**, at exactly 71 °C with exactly these
   capabilities (`bench/cases.py:49-55`). Re-deciding it differently would make the held-out
   set contradict its own specification.
4. **Over- and under-action are measured separately.** The harness already distinguishes
   "acted when the oracle said do nothing" from "observed when the oracle asked for action"
   (`bench/battle.py:24-26`, `bench/battle.py:106-113`), so scoring 71 °C as an action does
   not silently reward over-action; it only makes ToolUnder visible when a system stalls in
   the conserve band.

Two boundary consequences follow and are frozen:

* `70.0` is already conserve (inclusive `>=`), so `69.5` → `observe` and `70.0`/`70.5` →
  `set_vision_fps` (held-out cases `band_normal_edge_minus_half`, `band_conserve_edge_exact`,
  `band_conserve_edge_plus_half`).
* In the conserve band, R7 overrides R4: with no `vision` capability the label is `observe`,
  not `set_vision_fps` (`caps_no_vision_conserve`).

---

## 5. Choices this spec makes that no frozen case pins

Recorded openly so a reviewer can challenge them:

| Choice | Rule | Rationale |
| --- | --- | --- |
| RAM pressure line at `0.90` | R8 / §3 | No runtime constant exists; anchored on the only measured reference point (`bench/cases.py:71-72`). |
| RAM pressure does not suppress a conserve throttle | R11 | Shedding reduces load; holding steady adds it. |
| Unstructured text caps the conserve band at `observe` | R9 / R12 | Generalises `messy_unstructured` (`bench/cases.py:77-84`, normal band) to the conserve band; ambiguous evidence is normalized first. |
| Unstructured text never suppresses `cooldown` | R12 | The band is telemetry-derived; text is untrusted data (T20). |
| Text never moves the band in either direction | R9 | Only telemetry defines the band (T7). |
| `ram_used_mib` missing → pressure not established | R8 | The Governor gates on total, not used (`src/reflective_loop_runtime/governor.py:110-112`). |
| Negative `age_seconds` is skew, not staleness | R2 | The staleness test is `age > 30.0` (T8). |
| `abstain` for blocked ticks, `observe` for healthy ticks | R13 | Follows `README.md:104` and `bench/battle.py:244`. |

---

## 6. Harness caveats for whoever wires this set in

* `bench/battle.py:294-298` (`_fresh_sample`) re-stamps `sampled_monotonic = now` and
  `age_seconds = 0.0` on every case, which would silently erase the whole freshness
  stratum (R2). A consumer that wants these cases scored on staleness **must preserve
  `case.sample["age_seconds"]`**. The held-out labels state the intended contract, not the
  behaviour of the current harness.
* `bench/battle.py:261-291` (`_bare_state`) recomputes the band from `temperature_c` only.
  The `zones_c` stratum in this set (`zones_*` cases) is defined on the peak over zones and
  the flat field (T7); a consumer that only reads `temperature_c` will score those cases
  differently. That divergence is intentional and is what makes them held-out.
* `bench/battle.py:60-72` treats `observe` and `abstain` as interchangeable, so R13 is a
  convention; ToolOver/ToolUnder remain the discriminating metrics.
* No system was executed against this set during authoring. The first scoring run must be
  treated as a measurement, not as label generation.

---

## 7. Freeze and manifest

`bench/heldout/check_heldout.py` re-derives every label from the tables in §2 and fails if
a case disagrees. It also prints a manifest hash:

```
HELDOUT_OK <count>
MANIFEST_SHA256 <sha256 of the canonical JSON dump>
```

The manifest is `json.dumps([asdict(c) for c in HELDOUT_CASES], sort_keys=True,
separators=(",", ":"))` — every field of every case, including `note`, so editing a label,
a note or a sample changes the hash. Record that hash in the evaluation report; a different
hash means a different benchmark.
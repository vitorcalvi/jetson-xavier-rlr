# RLR overnight soak — how to read it and how to stop it

Phase 1 step D2 install on the Jetson Xavier NX (`192.168.1.49`, user `vitor`),
all as **user** systemd units — no sudo, loopback only, one model at a time.

## Install notes

Install the runtime **with the signing extra**, from the repo checkout:

```
python3 -m pip install -e .[crypto]
```

`[crypto]` is what pulls in `cryptography`. Without it the runtime can build no
operator signature, so:

* with no ENFORCE config it can only fall back to a dev-only HMAC bundle, which
  it deliberately re-emits as `OBSERVATION_ONLY` — the loop runs, but with no
  authority;
* with an ENFORCE config (`RLR_POLICY_PATH` and/or
  `RLR_POLICY_PUBLIC_KEY_PATH` in `~/.config/rlr/rlr.env`) it **refuses to
  start**, with an error naming the missing extra, instead of silently degrading
  to `OBSERVATION_ONLY`. That is deliberate: a deployment that believes it
  enforces a mission policy it cannot verify has no authority at all.

After install, check that the deployment really holds authority:

```
python3 deploy/healthcheck.py    # asserts policy mode ENFORCE, then heartbeat
```

It exits non-zero with `UNHEALTHY: policy mode OBSERVATION_ONLY (expected
ENFORCE)` if the mode is anything else.

The advisor reply is grammar-bounded (`RLR_QWEN_MAX_TOKENS=121`): the candidate
enum plus a rationale of at most 40 characters, with generation stopping at the
closing brace. See the caveat below for why the budget must stay small.

## What is running

| Unit | What it does |
| --- | --- |
| `rlr.service` | the real closed loop (`deploy/agent_runtime.py`, tick 1 s), restart-on-failure, kill switch `~/rlr/KILL` |
| `rlr-advisor.service` | `llama-server` serving falcon06 (ctx 1024, flash attention on, 2 threads, full GPU offload, 127.0.0.1:8080), host prompt cache bounded with `--cache-ram 256` and `MemoryHigh=2200M`/`MemoryMax=2600M`. Advisory tier only |
| `rlr-simsoak.service` | the overnight soak: repeats the 6-hour *simulated* fault-injected soak until `~/rlr/KILL` appears |
| `rlr-metrics.timer` | every 15 min runs `rlr-metrics.service`, which appends one metrics line |

## Files in `~/rlr/soak/`

| File | Meaning |
| --- | --- |
| `metrics.toon` | one metrics line per sample (append-only) |
| `simsoak-<utc>.toon` | the TOON result of one 6 h-simulated soak pass |
| `simsoak.latest.toon` | a copy of the newest pass (what metrics reads) |
| `simsoak.passes.log` | one line per pass with its exit code |
| `simsoak.err` | soak stderr, if any |
| `metrics_sample.py` | the sampler (reads systemd, /proc, /sys, verifies the audit chain) |

## Reading `metrics.toon`

Each line is space-separated `key=value` tokens (TOON-ish scalars):

```
ts rlr_state rlr_sub advisor_state advisor_sub simsoak_state simsoak_sub
rlr_restarts advisor_restarts simsoak_restarts heartbeat_age_s
rlr_rss_mib llama_rss_mib cpu_temp_c ram_avail_mib
audit_ok audit_reason permits_issued permits_consumed policy_violations
soak_audit_ok soak_age_s
```

* `rlr_state=active` + `heartbeat_age_s` small ⇒ the loop is alive and advancing.
* `audit_ok=true` ⇒ the live runtime's hash-chained audit log verifies.
* `policy_violations=0` and `soak_audit_ok=true` ⇒ the newest soak pass was clean.
* `permits_issued` / `permits_consumed` are taken from the newest soak pass.
* `cpu_temp_c` is `/sys/class/thermal/thermal_zone*` `CPU-therm`; `ram_avail_mib`
  is `MemAvailable`.

Quick checks:

```
tail -2 ~/rlr/soak/metrics.toon
systemctl --user status rlr.service rlr-advisor.service rlr-simsoak.service
systemctl --user list-timers rlr-metrics.timer
```

## Stopping the soak

```
touch ~/rlr/KILL
```

The loop stops within one tick and `rlr-kill.service` stops `rlr.service`. The
soak loop also exits when it sees the kill file. Stop the remaining units
(optional, user-level only):

```
systemctl --user stop rlr-simsoak.service rlr-metrics.timer rlr-advisor.service
```

## Notes / caveats

* `rlr-simsoak` uses a **simulated** clock (no wall-clock sleeping), so one
  `--hours 6` pass finishes in minutes; the service re-runs it until the kill
  file appears, which is what makes it an overnight soak and keeps the unit
  active.
* The advisor is advisory only. Stopping `rlr-advisor.service` degrades advice
  but the kernel chain still decides, and the loop keeps ticking.
* The advisor is memory-bounded on purpose. `llama-server` keeps a host-RAM
  prompt/KV cache whose upstream default is **8192 MiB**; the 2026-10-06 soak
  grew it to ~4.9 GiB `RssAnon` and pushed `ram_avail_mib` under 500 MiB, so the
  unit now passes `--cache-ram 256` and caps the service cgroup at
  `MemoryHigh=2200M` / `MemoryMax=2600M`. If `llama_rss_mib` climbs back toward
  2.5 GiB, check the unit file on the box still carries `--cache-ram 256`.
* **CPU priority: the soak is the load, not the workload.** The Jetson has 4
  cores and `rlr-simsoak` re-runs its simulated pass in minutes, so it burns a
  core continuously. Unweighted it took cycles from `llama-server` (falcon06,
  `-t 2`) and the loop, and the advisor returned request timeouts all night.
  The units now express the intent:

  | Unit | `Nice` | `CPUWeight` | Other |
  | --- | --- | --- | --- |
  | `rlr.service` (loop) | default | `200` | `CPUQuota=50%` |
  | `rlr-advisor.service` | `0` | `200` | `IOSchedulingClass` default |
  | `rlr-simsoak.service` | `19` | `10` | `CPUQuota=100%`, `IOSchedulingClass=idle` |

  On contention the tick and the advisor it consults win; the soak yields.
* **The loop never waits for the advisor.** The advisor is advisory only: a
  slow, failed or timed-out request degrades to **abstain** — the loop falls
  back to its rules and the kernel chain still decides — and the tick continues
  on `RLR_TICK_SECONDS=1.0`. That is why the 15 s `RLR_QWEN_TIMEOUT_SECONDS`
  below is a latency allowance, not a stall: it is sized to the measured
  ~3.0 s p50 / ~3.2 s p95 unloaded falcon06 latency with ~4x headroom, and
  exceeding it costs advice, never a tick.
* `RLR_QWEN_TIMEOUT_SECONDS=15` in `rlr.env.example` is the value to copy to
  `~/.config/rlr/rlr.env`. The old `1.5` was below the unloaded p95, so it
  could not succeed even on an idle host.
* **The reply is bounded so it can finish.** `RLR_QWEN_MAX_TOKENS` is `121`, not
  `1024`: the GBNF grammar admits only the candidate enum plus a ≤40-character
  rationale, and the request stops generation at the closing brace. A 0.6B
  falcon06 never reached the closing brace of the old four-field object inside
  15 s, so every reply timed out into a degraded abstain. The budget is derived
  from the longest object the grammar allows at the worst case of one token per
  character plus 16 of slack, and asserted by `tests/test_advisor_grammar.py` —
  raising it back brings the deadline misses back with it.
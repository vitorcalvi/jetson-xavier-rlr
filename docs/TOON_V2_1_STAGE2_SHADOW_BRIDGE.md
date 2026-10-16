# Stage 2 -> TOON v2.1, observer-only bridge

**No order authority is transferred.** The legacy Stage2 controller owns Alpaca
paper order creation/cancellation, HALT_ENTRIES, emergency stops, and all
reconciliation. `deploy/stage2_toon_bridge.py` receives copies/references to
already-obtained observations through `Queue.put_nowait` and writes a separate
`shadow_audit.jsonl`. It does not import a broker or take credentials.

The hook is placed immediately after the normal Stage2 snapshot GET (only when
that branch executes). While the existing durable HALT_ENTRIES is set, Stage2
intentionally skips the normal snapshot GET. The observer receives controller
state ticks but **not fabricated L2 data**. Shadow snapshots during the halt
would require a separately authorized read-only data collector, not changing
Stage2's order path. This distinction prevents claiming live shadow trade data.

`TOON_V2_SHADOW_ENABLED=true` (default) enables the asynchronous observer.
`TOON_V2_OVERRIDE_STAGE2=false` is enforced unconditionally: setting it to true
fails startup with SafetyLockViolation regardless of an environment variable or
current halt-file state. `HALT_ENTRIES=true` is the observer's fail-closed default.
The bridge does not implement mode promotion or paper/live order submission.

`on_market_snapshot` queues input without blocking for inference. This is a
nonblocking **queue offer**, not a guaranteed 10 ms response-time bound on
Python thread scheduling. Under overload samples can be dropped; a watchdog
should report `dropped` and `exceptions`. `on_stage2_signal` and
`on_trade_closed` append comparison or observational audit records. Trade-close
records NEVER become certification samples without independent broker proof.

**Critical data limitation:** The inspected Stage2 `fetch_snapshots()` path
provides bid/ask but no reliable five-level L2 quantities. The bridge records
`missing_l2_depth_no_fabricated_obi` rather than invents book imbalance,
microprice, maker fills, realized edge, or confidence. A future genuine L2
adapter must carry source, timestamp, level sizes, and sequence checks.

The audit is append-only JSON Lines for simple recoverability, with a TOON
payload in the `toon_memory_payload` field. It is **not** a pure TOON file.

## Regression contract and limitations

R1 tests now cover both 100 deterministic action observations and an actual 100-tick Stage2 main-loop replay using an inert fake broker;
R2 injects 15 worker errors; R3 compares a high-correlation risk veto; R4
measures the two existing **real** Stage2 state records that are currently
available. There are **zero actual Stage2 trading fills** in that ledger. The actual Stage2 loop replay matches all legacy state snapshots, runtime tick bytes and zero broker calls. It is halted-mode coverage only; a genuine open-session tape replay still requires saved broker/quote/bar observations.

A lack of tokenizer (`tiktoken`/cl100k_base) yields `token_reduction=null`, not a
fabricated percentage. Original underlying snapshots show negative TOON byte
savings; token compression needs measurement on 100 actual trade/memory rows.

## Safe operator flow

1. Verify existing `/home/vitor/rlr/flags/HALT_ENTRIES` exists, Stage2 service
   continues to run in `HALT_ENTRIES`, broker positions/orders read back flat.
2. Merge reviewed feature branch after full baseline regression; do not deploy
   or restart service automatically.
3. In a supervised maintenance window install package files and hook source;
   run complete `pytest`, confirming the legacy gate is unchanged.
4. Restart ONLY if specifically authorized with existing `HALT_ENTRIES` kept.
5. Verify audit heartbeat, dropped/exception counts, resource and disk usage.
6. Do not claim 30 real paper trades from synthetic shadow intents.

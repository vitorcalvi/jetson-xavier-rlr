# TOON v2 — isolated quantitative research pipeline

This is a **NON-TRADING** implementation. It does not alter, import, restart or
patch `rlr-stage2-paper.service`, broker POST code, or `HALT_ENTRIES`.
`UnifiedQuantRuntime` is an independent shadow entrypoint; **not a drop-in
replacement for the legacy Stage 2 execution engine**. Manual supervised
adapter review is required before any brokerage integration.

## Architecture mapping

| New module | Existing baseline integration seam | Runtime guarantee |
| --- | --- | --- |
| M1 `quant/execution/smart_router.py` | `trading/broker.py`, `orders.py` | Only produces order intents; native GTX only when explicitly supported; synthetic limit pricing cannot guarantee maker fees under transit latency. |
| M2 `quant/rl_core/*` | `trading/cashflow.py` | Market fills simulated separately, net equity rewards don't double count costs; equity fees and financing must be calibrated. |
| M3 `quant/regime/market_state.py` | `trading/features.py`, `decider.py` | Uses closed bars exclusively; future-dated market/funding data refused. |
| M4 `quant/validation/*` | `bench/heldout`, `trading/adaptive_gate.py` | CPCV purge/embargo, study-wide persistent trial count, fold-isolated scalers/memory. |
| M5 `quant/risk/*` | `trading/governor.py`, Stage 2 controller | Conservative Wilson/quarter-Kelly with aligned-bar Ledoit-Wolf style identity shrinkage and downside-conditioned correlations. |
| M6 `quant/edge_jetson/*` | `deploy/*` | TOON-compatible subset, thermal/background-training gate, dry-run affinity, GPU export command only when trtexec available. |

## Empirical, as opposed to desired, results

* 29 focused local tests passed, 0 failed.
* 500 synthetic historical bars processed end to end; 500 virtual intents, **zero broker orders**.
* 100 **synthetic** W1/W2 payloads: 100 lossless round trips; measured p99 serial/deserial 0.094 ms; TOON encoded **1.32% more bytes than JSON**. No installed `tiktoken` cl100k encoder, so **no token-saving figure exists**. This **fails** B1 promotion.
* 100 synthetic GBM / jump-diffusion paths, 200 hyperparameter variants each: 20,000 cumulatively recorded trials in SQLite, 0/100 DSR >= 0.95; zero test/train fold index overlap. It is not a proof of a 5% real false-positive rate.
* Isolated x86 CPU serializer p99 0.045 ms: **not Jetson inference and not tick-to-broker**; none of ONNX Runtime, TensorRT FP16, or FP32 inference was measured.
* No Alpaca paper order/cancel/stop/recovery E2E, 0 certified broker fills; 30-trade paper gate **FAILED/NOT ATTEMPTED**.

## Statistical corrections

The DSR is a **probability** 0..1; use >=0.95. Persist all attempted model
variants (including rejected) with stable IDs using `TrialRegistry(db_path)`,
not one selected model per train-fold. Study-wide trial Sharpes variance is
measured on a consistent return frequency. Parameter optimizer must operate
on a training fold; held-out features are transformed with training-only
scaler. The test harness records null study trials, not live strategy wins.

The 24h covariance requires exact overlapping timestamps; mismatched equity
RTH/crypto 24x7 bars **fail closed**, rather than forwarding filling or
pretending observations synchronized. Covariance's identity target shrinkage
is a Ledoit-Wolf plug-in approximation; external numerical validation against
a reference estimator is required. Tail rho conditions on the benchmark
bottom 20% of **past** joint-bar returns.

## Venue costs

The published Alpaca lowest-volume crypto tier is 15bps maker / 25bps taker.
Actual fees depend on 30D volume, account and venue. Alpaca equities are
commission-free but SEC/TAF and CAT charges and financing have changing rates:
we do NOT manufacture numerical fees. `ALPACA_EQUITIES_UNCALIBRATED` refuses
training/execution until a current auditable profile is supplied. Perpetual
2bps/5bps is an **illustrative, unverified venue** and applies only to an
explicitly connected perpetual venue. No funding rate is applied to spot.

Alpaca does not expose native GTX time-in-force. Alpaca crypto permits GTC/IOC,
Alpaca equities DAY/GTC etc. A limit resting on a book snapshot can be
marketable upon arrival. In the bounded 50ms *simulated* jitter test no
marketable orders were observed, but a gap test demonstrates crossing risk.

## Execution modes and safety

`EXECUTION_MODE=SHADOW_SIMULATION` is the conceptual default. All modes in
`UnifiedQuantRuntime` are strictly no-broker: even naming a mode
`PRODUCTION_LIVE` doesn't instantiate a broker, so no accidental route to
real money is possible. This implementation has **no network POSTs**.
`RuntimeConfig(halt_entries=False)` rejects configuration. Shadow intents
are NOT paper fills. It cannot automatically rearm or promote itself.

Do not replace `rlr-stage2-paper.service` with this code until the existing
Stage 2 emergency-halt, cancel/replace, Alpaca fractional-stop and broker
reconciliation contract has been audited and separately tested.

## Run the isolated tests

```bash
export PYTHONPATH=src:.
python -m pytest -q tests/quant
python benchmarks/bench_integrated_shadow.py
python benchmarks/bench_toon_compression.py
python benchmarks/bench_leakage_null_test.py --paths 100 --trials 200
python benchmarks/bench_jetson_inference.py
```

Running `benchmarks/bench_alpaca_paper_e2e.py` only issues broker **GET**
requests with paper credentials and `HALT_ENTRIES` present; it cannot submit,
cancel, replace or certify paper orders.

## P0 remaining before paper certification

1. Confirm actual trading session, host, instrument, account and capital risk
   policy. Then separately implement broker-specific order lifecycle with
   explicit operator authorization; never silently switch GTX to DAY/GTC.
2. Calibrate current fees and verified RTT/WS data on the actual venue.
3. Repeat CPCV on signed immutable, true out-of-sample market/trade records;
   collect at least 30 actual supervised paper broker fills.
4. Verify watchdog and independent stop inventory against Alpaca paper API.
5. Run Jetson Xavier NX thermal soak with existing cooling; only adjust
   `nvpmodel` after checking power IDs/fan health and securing owner approval.
6. Do not remove `/home/vitor/rlr/flags/HALT_ENTRIES` without explicit sign-off.

**References:** https://docs.alpaca.markets/us/v1.1/docs/crypto-fees ;
https://alpaca.markets/support/regulatory-fees ;
https://docs.alpaca.markets/us/docs/orders-at-alpaca ;
https://docs.nvidia.com/jetson/archives/r35.3.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance/JetsonXavierNxSeriesAndJetsonAgxXavierSeries.html

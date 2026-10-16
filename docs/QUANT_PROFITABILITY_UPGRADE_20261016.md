# RLR quantitative research upgrade — TOON specification adjudication

**Scope:** offline quantitative research, not an autonomous live-order system. The six modules are under `src/reflective_loop_runtime/quant/` and are intentionally **not imported by the Stage 2 broker runner**. Current `HALT_ENTRIES` must remain untouched. A local unit-test result is not a measurement of trading profitability or Jetson inference latency.

## Mapping

| New module | Existing core integration point | Status |
| --- | --- | --- |
| `quant/execution/smart_router.py` | `trading/broker.py`, `trading/orders.py`, Stage 2 reconciliation | Offline intent planner and OCA validation, no broker writes |
| `quant/risk/kelly_firewall.py` | `trading/governor.py`, `trading/killmodes.py` | Independent research-only pre-trade veto |
| `quant/rl_core/reward_and_env.py` | `trading/cashflow.py`, `adaptive_gate.py` | Queue-ahead sim and mark-to-market reward, cost counted once |
| `quant/regime/market_state.py` | `trading/features.py`, `trading/decider.py` | Four-state classifier with past-only bars |
| `quant/validation/darwin_cpcv.py` | `bench/heldout`, walk-forward verification | CPCV splits, overlap/embargo, DSR probability, bootstrap and shadow gates |
| `quant/edge_jetson/runtime_opt.py` | `deploy/`, existing `redteam.py` TOON subset | Deterministic W1/W2 wire, SQLite memory, bounded ring and thermal reads |

## Venue capability restrictions

- **Alpaca equities:** documented TIFs include `day`, `gtc`, `ioc`, `fok`, `opg`, `cls`; GTX is **not** documented as valid. A passive DAY limit can still execute as taker if the market moves. Pass `require_native_post_only=False` explicitly to plan such an order; it is not native post-only.
- **Alpaca spot crypto:** do not assume perpetual futures, funding, or liquidation data. This package disables funding by default.
- **External crypto derivative venue:** permit native GTX and funding only after implementing and proving an adapter with real broker-supported order types, fee schedule, cancellation and inventory readback. No such adapter is included or connected here.
- **Order execution:** the package exposes no authenticated broker transport. Its offline planner is unable to submit an order. TWAP slicing does not guarantee iceberg support. OCA validation does not create exchange-native OCA.

## Statistical interpretation

- **DSR** from `deflated_sharpe_probability()` is a probability between zero and one; the promotion gate uses `>= 0.95`, rather than the impossible `DSR > 1.50`. Estimating trial Sharpe variance requires retaining the *full parameter/model trial family* rather than merely one winning model.
- **CPCV** purges overlapping label lifetimes and embargoes observations after selected test folds. Test-split combinations are not themselves independent test paths. CPCV splits must be scored on genuinely unseen events; calibration data must not be reused for final claims.
- **Kelly:** uses Wilson's lower confidence bound for win probability with fixed-shrinkage covariance, not an empirical Ledoit–Wolf estimator. Sample count and historical lookback must match the market. The drawdown limit is a **new-entry veto**, not a guaranteed loss limit during gap moves.
- **Reward:** `step()` computes net equity *after* maker/taker fees, slippage, and verified venue-specific funding. It does not subtract fees again. The reported EW downside Sortino-style statistic is a research diagnostic, not the exact differential Sortino policy gradient.
- **Regime:** four classes are `TREND_UP`, `TREND_DOWN`, `MEAN_REVERT`, `UNSAFE` (chop, shock, uncertain). A four-state classifier cannot separately include trend-up, trend-down, mean-revert, chop, and shock. It fails closed for insufficient bars/funding samples.
- **Profitability:** none of the requested maker fill ratio, `<=1.5bps` slippage, `Sortino >=2.2`, `Profit Factor >=1.65`, drawdown under 2.5%, or inference latency `<15ms` is established without independent post-cost historical and live paper fills.
- **30-shadow gate:** IDs and timestamps must be independent observed fills. Synthetic values must not be passed off as actual shadow evidence.

## Hardware

- Target confirmed: Jetson Xavier NX, AArch64 kernel. `nvpmodel` was not on `PATH` at audit; the device has had cooling limitations. **Do not run MAXN/jetson_clocks without fixing fan and measuring thermal headroom**. A TensorRT FP16 engine cannot be exported without an actual compatible ONNX policy, `trtexec`, calibration/precision checks, and a JetPack-compatible TensorRT installation. The provided script checks these and refuses unsafe overwrites. INT8 requires calibration; no unvalidated INT8 engine is produced.
- The Python ring buffer bounds RAM but is **not** zero-copy CUDA pinned memory. The TOON-inspired wire is deterministic and round-trip safe for a restricted W1/W2 schema; it is not the complete upstream TOON spec. Compare **actual model tokenizer counts** before claiming 45% savings; `wire_metrics()` reports real UTF-8 bytes, not inferred tokens.

## Reproduction

```bash
PYTHONPATH=src pytest -q tests/quant
bash deploy/export_quant_fp16.sh /absolute/path/policy.onnx /absolute/path/policy_fp16.engine
```

The second command works only on a Jetson equipped with a compatible `trtexec` and an existing ONNX model. It never changes nvpmodel or touches a trading account.

## Promotion process

Keep `HALT_ENTRIES` in place. First calibrate transaction cost and queue position using recorded L2 and actual paper order events (not synthetic simulator fills), evaluate all CPCV folds, obtain multi-regime out-of-sample evidence and 30 **independent** chronological paper shadow fills, then perform an explicit operator review. The package's `promotion_gate` computes a research verdict only and never authorizes live capital on its own.

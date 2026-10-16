# Evidence-driven adaptive RLR trading and Qwen 3.5 (2026-10-14)

## Authority architecture
- Qwen3.5-2B-Q4_K_M is **advisory only**, temperature 0, reasoning off, grammar-constrained short JSON. The model has no authority to submit orders, self-sign a scope, or rewrite eligibility.
- Trader continues to calculate deterministic trend/VWAP signals and exits.
- `AdaptivePaperGate` vetoes **every new long** unless the current code strategy has a registered immutable study, same-source checksum, independent chronological outcomes, positive multiple-testing-adjusted lower bound in train/validation/holdout/forward, healthy broker, and an operator-verified Ed25519 approval scope. Exits bypass this research veto.
- Historic Kaggle 9-strategy results do **not** establish a positive net edge. The initial `candidates.json` is exactly empty, so all entry proposals are rejected.
- Qwen hourly research recommendations are saved in a JSON file, but are not auto-registered, auto-approved or executed.

## Signed scope protocol
Approved scope must be in `<trader-data>/adaptive_research/approved_scope.json`:
```json
{"payload":{"allowed_fingerprints":["<64 lowercase hex candidate fingerprint>"],"issued_at":0,"expires_at":0,"mode":"paper","symbol":"SOL/USD"},"signature_base64":"<signature>"}
```
Compute signature of `b"RLR-STRATEGY-SCOPE-v1\\n" + json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()` with the operator Ed25519 **private** key OFF the Jetson. Only the already-configured signed mission **public** key is used on the Jetson. Signatures cannot last longer than 30 days.

## Research manifest
`<data>/adaptive_research/candidates.json` starts with `{"version":1,"candidates":[]}`.
For a future vetted study, candidate fields include `strategy_id` (currently only deployed trend_vwap or orb5), `code_sha256` matching deployed decider bytes, `artifact_file` under `evidence/`, `artifact_sha256`, `symbol`, `feed`, `regimes`, `costs_audited`, `leakage_audited`, `holdout_sealed_before_testing`, and daily `train/validation/holdout/forward_paper` arrays. The operator must independently check those provenance claims before signing the fingerprint. No Kaggle result is automatically promoted.

## Running components
- `rlr-trader.service`: original paper-only Alpaca order path, 30s loop, evidence veto on.
- `rlr.service`: original RLR self-monitoring loop, Qwen local advisory active with compact 64-token JSON. It does **not** directly decide broker entries.
- `rlr-qwen35.service`: llama.cpp Qwen model, GPU 99 layers, 1024 context, Flash Attention, 2 threads, 1 slot, temp 0, no-thinking, prompt cache & 256MB RAM cache, 127.0.0.1:8081 only.
- `rlr-adaptive-research.timer`: every hour, read-only trading summaries + Qwen research-only hypothesis to `research_hypothesis.json`.

## Limits
- There are no statistically approved strategies today; expect `NO_TRADE`.
- This is constrained adaptive *selection and research*, not open-ended self-modifying trading code.
- Broker-side stop remains software-managed; monitoring cadence does not guarantee fill.
- CFEE remains incomplete: fee model is not broker-confirmed accounting.
- Host fan/thermal maintenance remains a priority for sustaining GPU inference.

## Rollback
- Code baseline: `/home/vitor/rlr/backups/adaptive-20261014/git-before.sha`
- Service/env snapshots: `/home/vitor/rlr/backups/adaptive-20261014/`
- Do not disable the evidence gate without explicit operator direction. Stop/Qwen failure must not unblock entries.

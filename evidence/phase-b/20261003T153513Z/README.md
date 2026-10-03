# Accelerated Phase B evidence — 2026-10-03

Candidate: `1cdb094e2fab8353574922a791678cd7e506c5e9` (`hardening-1cdb094`)

This directory contains the preregistration, runner, raw condition outputs, thermal/event logs, scoring output, and result manifests from the accelerated Phase B run.

## Benchmark integrity

The sealed held-out answer key is intentionally **not committed** to this public repository. Publishing it would invalidate future held-out evaluation. Its frozen SHA-256 is recorded in `manifest.json` / the run manifests and was verified before scoring.

Frozen answer-key SHA-256:

`da6423163ca5cc17af7e0c8327884bc4e9d7c34fad0c40a78348c19f51c4402e`

## Headline result

- Action-track B2 did not outperform B0, B0F, B1, NO_SELF_MODEL, or the DEV-only majority baseline on this N=3 accelerated suite.
- Primary paired B2 deltas were `0.0` with 95% CI `[0.0, 0.0]`.
- Narrow self-state manipulation: B2 withheld-goal recall 3/3; B2n 0/3; NO_SELF_MODEL 0/3.
- Forecast model MAE: 1.33 C, equal to persistence baseline; linear baseline 2.92 C.
- Maximum recorded Phase B peak: 68.5 C; no thermal abort.
- This run is explicitly underpowered and is not sufficient for an unattended-production or intelligence claim.

See `phase_b_scored.json` and `phase_b_summary.json` for the machine-readable result.

"""Anti-leakage null study. 100 reproducible GBM/jump paths, 200 variants.

This is a finite stress test, not a proof that the future false-discovery rate
will remain below 5%. No shadow fills are fabricated or counted toward promotion.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import tempfile
from pathlib import Path

from reflective_loop_runtime.quant.validation.darwin_cpcv import (
    Event,
    cpcv,
    deflated_sharpe_probability,
    promotion_gate,
)
from reflective_loop_runtime.quant.validation.metadata_v2 import FoldStandardizer, TrialRegistry


def sharpe(a: list[float]) -> float:
    if len(a) < 3:
        return 0.0
    sd = statistics.stdev(a)
    return statistics.mean(a) / sd if sd > 0 else 0.0


def sample_returns(rng: random.Random, n: int, jumps: bool) -> list[float]:
    return [
        rng.gauss(0, 0.01)
        + (-0.06 if jumps and rng.random() < 0.018 else 0.0)
        + (0.06 * 0.018 if jumps else 0.0)
        for _ in range(n)
    ]


def run(paths: int = 100, trials: int = 200, seed: int = 17, db_path: Path | None = None) -> dict:
    if paths <= 0 or trials < 3:
        raise ValueError("insufficient null trials")
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory() as tmp:
        registry = TrialRegistry(db_path or Path(tmp) / "null_trials.db")
        dsr_rejections = 0
        dsr_promotions = 0
        joint_promotions = 0
        leak_violations = 0
        for p in range(paths):
            base = sample_returns(rng, 240, p % 2 == 1)
            e = [Event(i * 60, i * 60 + 59, base[i], 0.0) for i in range(len(base))]
            splits = cpcv(e, groups=6, test_groups=2, embargo=60)
            split = splits[0]
            # Test values cannot influence fold scaler or selection.
            scaler = FoldStandardizer.fit([(v,) for v in base], split.train, split.test)
            if any(i in split.test for i in scaler.fitted_indices):
                leak_violations += 1
            train = set(split.train)
            selected_train_sr = float("-inf")
            selected_oos = []
            for trial in range(trials):
                # Trial selection only sees training outcomes. All variants
                # derive deterministic causal exposure decisions independent of
                # future price observations (no circular optimization).
                width = 4 + (trial % 15)
                phase = (trial * 7) % width
                signals = [1 if (i + phase) % width < width // 2 else -1 for i in range(len(base))]
                train_pnl = [base[i] * signals[i] - 0.0001 for i in split.train]
                s = sharpe(train_pnl)
                registry.add(
                    "null-global",
                    f"path{p:04d}-mut{trial:04d}",
                    f"null_path_{p}",
                    "independent_family",
                    f"cpcv_train_{len(train)}",
                    s,
                )
                if s > selected_train_sr:
                    selected_train_sr = s
                    selected_oos = [base[i] * signals[i] - 0.0001 for i in split.test]
            s = registry.summary("null-global")
            # Historical trials accumulate, not reset each path.
            variance = max(s.sr_variance, 1e-15)
            dsr = deflated_sharpe_probability(selected_oos, s.total, variance)
            dsr_promotions += dsr >= 0.95
            dsr_rejections += dsr < 0.5
            # No independent Alpaca shadow trade evidence: mandatory gate veto.
            joint_promotions += int(
                promotion_gate(
                    trades=[Event(i, i, v, 0.0) for i, v in enumerate(selected_oos)],
                    equity=[10_000]
                    + [10_000 + sum(selected_oos[: i + 1]) for i in range(len(selected_oos))],
                    dsr_probability=dsr,
                    shadow=[],
                    base_sharpe=1.0,
                    jittered_sharpes=[0.95, 1.05],
                    full_oos_days=90,
                ).approved
            )
        return dict(
            synthetic_null_paths=paths,
            mutations_per_path=trials,
            cumulative_trial_count=registry.summary("null-global").total,
            dsr_below_0_50_fraction=dsr_rejections / paths,
            dsr_0_95_false_pass_fraction=dsr_promotions / paths,
            full_promotion_false_pass_fraction=joint_promotions / paths,
            feature_train_test_overlap_failures=leak_violations,
            caveat="Synthetic null only, not real trade profitability or 30 broker fills",
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", type=int, default=100)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    obj = run(args.paths, args.trials, args.seed)
    if args.output:
        args.output.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")
    print(json.dumps(obj, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

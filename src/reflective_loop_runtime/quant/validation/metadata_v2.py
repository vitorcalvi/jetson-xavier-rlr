"""SQLite multiplicity registry and fold-isolated feature/episodic-memory fitting.

All attempted hyperparameter trials are recorded, including failures. Distinct
trial keys are unique within a study. There is no trial-count reset per batch.
Global data preprocessing is forbidden: train indices are required explicitly.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from statistics import mean, pvariance

from .darwin_cpcv import deflated_sharpe_probability


@dataclass(frozen=True)
class TrialSummary:
    total: int
    sr_variance: float
    trial_ids: tuple[str, ...]


class TrialRegistry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=10) as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=FULL")
            c.execute("""CREATE TABLE IF NOT EXISTS attempts (
                study TEXT NOT NULL, trial_id TEXT NOT NULL,
                dataset_sha TEXT NOT NULL, config_sha TEXT NOT NULL,
                fold_manifest_sha TEXT NOT NULL, oos_sharpe REAL NOT NULL,
                created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(study,trial_id))""")

    def add(
        self,
        study: str,
        trial_id: str,
        dataset_sha: str,
        config_sha: str,
        fold_manifest_sha: str,
        oos_sharpe: float,
    ) -> None:
        if any(
            not s or len(s) > 200
            for s in (study, trial_id, dataset_sha, config_sha, fold_manifest_sha)
        ) or not isfinite(oos_sharpe):
            raise ValueError("invalid trial record")
        with sqlite3.connect(self.path, timeout=10) as c:
            c.execute("PRAGMA busy_timeout=10000")
            c.execute(
                "INSERT INTO attempts (study,trial_id,dataset_sha,config_sha,fold_manifest_sha,oos_sharpe) VALUES (?,?,?,?,?,?)",
                (study, trial_id, dataset_sha, config_sha, fold_manifest_sha, oos_sharpe),
            )

    def summary(self, study: str) -> TrialSummary:
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as c:
            rows = c.execute(
                "SELECT trial_id,oos_sharpe FROM attempts WHERE study=? ORDER BY trial_id", (study,)
            ).fetchall()
        vals = [row[1] for row in rows]
        return TrialSummary(
            len(vals), pvariance(vals) if len(vals) >= 2 else 0.0, tuple(r[0] for r in rows)
        )

    def dsr(self, study: str, returns: list[float]) -> float:
        s = self.summary(study)
        if s.total < 2 or s.sr_variance <= 0:
            raise ValueError("insufficient historical trial multiplicity")
        return deflated_sharpe_probability(returns, s.total, s.sr_variance)


@dataclass(frozen=True)
class FoldStandardizer:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    fitted_indices: tuple[int, ...]

    @classmethod
    def fit(
        cls,
        raw: list[tuple[float, ...]],
        train_indices: tuple[int, ...],
        test_indices: tuple[int, ...],
    ) -> FoldStandardizer:
        if not raw or not train_indices or set(train_indices) & set(test_indices):
            raise ValueError("train/test contamination")
        if len(set(train_indices)) != len(train_indices) or any(
            i < 0 or i >= len(raw) for i in train_indices
        ):
            raise ValueError("invalid train index")
        n = len(raw[0])
        sel = [raw[i] for i in train_indices]
        if n < 1 or any(len(v) != n or any(not isfinite(a) for a in v) for v in sel):
            raise ValueError("invalid training features")
        m = tuple(mean(v[j] for v in sel) for j in range(n))
        scales = tuple(
            max(sqrt(sum((v[j] - m[j]) ** 2 for v in sel) / len(sel)), 1e-10) for j in range(n)
        )
        return cls(m, scales, tuple(train_indices))

    def transform(self, row: tuple[float, ...]) -> tuple[float, ...]:
        if len(row) != len(self.means) or any(not isfinite(x) for x in row):
            raise ValueError("invalid fold test feature")
        return tuple((x - m) / s for x, m, s in zip(row, self.means, self.scales, strict=False))


@dataclass(frozen=True)
class FoldMemoryIndex:
    entries: tuple[tuple[int, str], ...]

    @classmethod
    def fit(
        cls,
        memories: list[tuple[int, str]],
        train_indices: tuple[int, ...],
        test_indices: tuple[int, ...],
        asof_timestamp: int,
    ) -> FoldMemoryIndex:
        if set(train_indices) & set(test_indices):
            raise ValueError("memory leak between folds")
        if any(i < 0 or i >= len(memories) for i in train_indices):
            raise ValueError("memory index outside training fold")
        keep = tuple(memories[i] for i in train_indices if memories[i][0] <= asof_timestamp)
        if any(ts > asof_timestamp for ts, _ in keep):
            raise ValueError("future recall")
        return cls(keep)

    def recall(self, asof_timestamp: int) -> tuple[str, ...]:
        return tuple(content for ts, content in self.entries if ts <= asof_timestamp)

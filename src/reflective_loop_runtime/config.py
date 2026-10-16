"""Runtime configuration with bounded, portable defaults.

No secrets, no machine-specific absolute paths. The database defaults to a
path inside the configured data directory and is validated to stay there.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    data_dir: str | Path = "rlr_data"
    db_path: str | Path | None = None

    # Loop cadence and context bounds.
    tick_seconds: float = 1.0
    max_context_chars: int = 6000
    episode_window: int = 12
    max_active_goals: int = 8

    # Telemetry freshness.
    stale_seconds: float = 30.0

    # Confidence gate: below this, the slow path may consult Qwen before the
    # primary cortex makes a fresh bounded decision.
    confidence_threshold: float = 0.60

    # Thermal governance thresholds (Celsius). Thresholds define
    # conserve/degraded/cooldown bands and a resume band for cooldown exit.
    conserve_c: float = 70.0
    degraded_c: float = 75.0
    cooldown_c: float = 78.0
    resume_c: float = 70.0

    # Resource caps applied by the Governor.
    max_vision_fps_degraded: float = 1.0
    max_vision_fps_conserve: float = 2.0
    max_reply_tokens_degraded: int = 48
    max_reply_tokens_conserve: int = 64

    # Optional model services.
    qwen_base_url: str = "http://127.0.0.1:8080"
    qwen_model: str = "Qwen2.5-1.5B-Instruct"
    qwen_timeout_seconds: float = 20.0
    qwen_max_tokens: int = 121  # qwen.ADVICE_MAX_TOKENS (kept in sync by tests)
    lfm_model_id: str = "LiquidAI/LFM2.5-350M"
    lfm_max_tokens: int = 384
    model_max_attempts: int = 1

    # Mission policy authority. Configuring either field asks the deployment to
    # run in ENFORCE mode against an operator-signed policy envelope: the
    # envelope in ``policy_path`` is verified with the PEM public key in
    # ``policy_public_key_path``. ENFORCE needs the optional ``cryptography``
    # backend, so a deployment that asks for it without the ``[crypto]`` extra
    # installed must fail to start rather than quietly govern itself in
    # OBSERVATION_ONLY. With neither field set the runtime signs its own
    # in-process operator key and still enforces.
    policy_path: str | Path | None = None
    policy_public_key_path: str | Path | None = None

    # Hard per-run PID cap for sandboxed tool calls, enforced by the kernel
    # through a delegated pids-controller cgroup (created ahead of time by
    # deploy/rlr-sandbox-cgroup.service and owned by the runtime user). Set it
    # and every sandboxed run gets its own capped child cgroup; a run that
    # cannot be capped refuses rather than proceeding uncapped. Left None, the
    # broker keeps its pre-existing RLIMIT_NPROC-only behaviour.
    sandbox_pids_cgroup: str | Path | None = None

    # Safety.
    fail_closed_on_model_error: bool = True
    governor_max_repeat_ticks: int = 3
    random_seed: int = 17

    extra: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.db_path is None:
            self.db_path = self.data_dir / "rlr.db"
        else:
            self.db_path = Path(self.db_path)
        root = self.data_dir.resolve()
        target = self.db_path.resolve()
        if os.path.commonpath([str(root), str(target)]) != str(root):
            raise ValueError("database path must live inside the configured data directory")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")
        if not 0.0 < self.tick_seconds <= 3600:
            raise ValueError("tick_seconds must be within (0, 3600]")
        if not self.conserve_c <= self.degraded_c <= self.cooldown_c:
            raise ValueError("thermal thresholds must be non-decreasing")
        if self.policy_public_key_path is not None and self.policy_path is None:
            raise ValueError("policy_public_key_path requires policy_path")
        if self.sandbox_pids_cgroup is not None:
            self.sandbox_pids_cgroup = Path(self.sandbox_pids_cgroup)

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

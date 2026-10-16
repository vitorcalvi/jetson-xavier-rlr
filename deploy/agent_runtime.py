#!/usr/bin/env python3
"""User-level RLR runtime driver (Jetson D1 install).

Runs the closed reflective loop on a fixed cadence, samples on-device
telemetry through :class:`~reflective_loop_runtime.telemetry.ProcTelemetry`,
writes a heartbeat file for liveness, and stops within one tick when the
kill file appears (polled every 0.25 s).

Decision tiers (kernel chain always decides):
  * ``DeterministicOpenDecider``  -> rule_scorer, always the first selector
  * ``DeterministicLFM``          -> lfm12 normalizer (raw/unstructured only)
  * optional ``QwenSlowBrain``    -> falcon06 advisor, advisory only, never the
                                     final selector (gated by RLR_ADVISOR_ENABLE)

A0/A1 shell effects run inside bwrap only; A2/A3 tools are mocked. The driver
never prints or persists secrets.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from pathlib import Path

from reflective_loop_runtime.authority import load_operator_policy
from reflective_loop_runtime.config import Config
from reflective_loop_runtime.escalation import TieredSlowBrain, build_lan_slow_brain
from reflective_loop_runtime.events import KillSwitch
from reflective_loop_runtime.kernel.policy import MODE_ENFORCE
from reflective_loop_runtime.kernel.toolbroker import PIDS_CONTROL_FILES
from reflective_loop_runtime.qwen import ADVICE_MAX_TOKENS, OpenAICompatClient, QwenSlowBrain
from reflective_loop_runtime.runtime import ReflectiveLoopRuntime
from reflective_loop_runtime.telemetry import ProcTelemetry

_LOG = logging.getLogger("rlr.deploy")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _float_env(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("non-finite")
        if minimum is not None and value < minimum:
            raise ValueError("below minimum")
        if maximum is not None and value > maximum:
            raise ValueError("above maximum")
        return value
    except (TypeError, ValueError):
        # Never include the raw value: this env file may also contain secrets.
        _LOG.warning("invalid %s; using safe default", name)
        return float(default)


def _int_env(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        value = int(raw)
        if minimum is not None and value < minimum:
            raise ValueError("below minimum")
        if maximum is not None and value > maximum:
            raise ValueError("above maximum")
        return value
    except (TypeError, ValueError):
        _LOG.warning("invalid %s; using safe default", name)
        return int(default)


def build_advisor(config: Config) -> object | None:
    """Build the local advisor and optional bounded LAN LiteLLM escalation.

    The runtime invokes this slow-brain only after a low-confidence bounded
    decision. The local falcon06 advisor is tried first. Hard/failed cases may
    escalate to the LAN provider; LAN failure returns local advice and never
    grants tools, capabilities, or permissions.
    """
    if not _bool_env("RLR_ADVISOR_ENABLE", False):
        return None

    config.qwen_base_url = _env("RLR_LLAMA_BASE_URL", "http://127.0.0.1:8080")
    config.qwen_model = _env("RLR_MODEL_NAME", "falcon06")
    config.qwen_timeout_seconds = _float_env(
        "RLR_QWEN_TIMEOUT_SECONDS", 15.0, minimum=0.1, maximum=60.0
    )
    # D11 derives the grammar-safe cap from the longest valid reply.
    config.qwen_max_tokens = _int_env(
        "RLR_QWEN_MAX_TOKENS",
        ADVICE_MAX_TOKENS,
        minimum=16,
        maximum=ADVICE_MAX_TOKENS,
    )
    try:
        client = OpenAICompatClient(
            base_url=config.qwen_base_url,
            model=config.qwen_model,
            timeout=config.qwen_timeout_seconds,
            max_tokens=config.qwen_max_tokens,
        )
        if _bool_env("RLR_COMPACT_ADVISOR", False):
            from reflective_loop_runtime.advisor_compact import CompactQwenAdvisor
            local = CompactQwenAdvisor(client)
        else:
            local = QwenSlowBrain(client, name="falcon06-advisor")
    except Exception:
        return None

    if not _bool_env("RLR_LAN_LLM_ENABLE", False):
        return local

    api_key = os.environ.get("RLR_LAN_LLM_API_KEY", "")
    try:
        lan = build_lan_slow_brain(
            base_url=_env("RLR_LAN_LLM_BASE_URL", "http://192.168.1.45:4000/v1"),
            model=_env("RLR_LAN_LLM_MODEL", "g1-maintenance"),
            api_key=api_key,
            timeout=_float_env(
                "RLR_LAN_LLM_TIMEOUT_SECONDS", 1.5, minimum=0.1, maximum=5.0
            ),
            max_tokens=_int_env(
                "RLR_LAN_LLM_MAX_TOKENS", 768, minimum=64, maximum=1024
            ),
        )
    except Exception:
        return local

    threshold = _float_env(
        "RLR_LAN_ESCALATION_THRESHOLD", 0.55, minimum=0.0, maximum=1.0
    )
    return TieredSlowBrain(
        local=local,
        lan=lan,
        lan_enabled=True,
        lan_threshold=threshold,
    )

# The kernel's own record of what is mounted where. Whether a base is a pids
# cgroup is a question about the *mount* it sits on, and mountinfo is the only
# answer that needs neither root (statfs on cgroupfs is not readable by the
# runtime user) nor a guess from the path. A named reader keeps that question
# testable without a live cgroup.
PROC_MOUNTINFO = "/proc/self/mountinfo"

# cgroup v2 puts every controller in one hierarchy, so the unified fstype alone
# is the evidence. v1 mounts one controller per filesystem, so the mount's
# super options have to name ``pids``: a cpu or memory mount of the same
# fstype would otherwise pass and cap nothing.
CGROUP2_FSTYPE = "cgroup2"
CGROUP_V1_FSTYPE = "cgroup"
CGROUP_PIDS_CONTROLLER = "pids"


def _read_mountinfo() -> str:
    """``/proc/self/mountinfo`` as text; empty when it cannot be read.

    Empty rather than raising, so a missing ``/proc`` -- an empty mount table --
    reaches the caller as "nothing is mounted there" and the preflight refuses
    the base instead of starting a runtime whose cap it cannot prove.
    """
    try:
        return Path(PROC_MOUNTINFO).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _unescape_mount_field(field: str) -> str:
    """Undo the octal escapes the kernel puts in mountinfo fields.

    Space, tab, newline and backslash are written ``\\040``, ``\\011``,
    ``\\012`` and ``\\134``. A mount point is compared against a path the
    filesystem just handed us, so the escapes have to be decoded first.
    """
    for code, char in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        field = field.replace(code, char)
    return field


def _mount_for(base: Path, mountinfo: str) -> tuple[str, str] | None:
    """``(fstype, super options)`` of the deepest mount containing ``base``.

    Deepest, not first: a mount tree nests, and the mount a path resolves
    through is the last one above it. Mountinfo is not sorted, so "first
    match" would let a cgroup mount somewhere above the base win over an
    ordinary filesystem mounted on top of it -- which is exactly the
    directory that must be refused. ``None`` means the table says nothing
    covers this path at all.
    """
    target = os.path.normpath(str(base))
    best: tuple[str, str] | None = None
    best_depth = -1
    for line in mountinfo.splitlines():
        fields = line.split()
        separator = fields.index("-") if "-" in fields else -1
        if separator < 5 or len(fields) < separator + 4:
            continue  # not a mountinfo record we can reason about
        point = os.path.normpath(_unescape_mount_field(fields[4]))
        if target != point and not target.startswith(point.rstrip("/") + "/"):
            continue
        if len(point) > best_depth:
            best_depth = len(point)
            best = (_unescape_mount_field(fields[separator + 1]),
                    _unescape_mount_field(fields[separator + 3]))
    return best


def _mount_enforces_pids(base: Path, mountinfo: str) -> bool:
    """Whether the kernel enforces ``pids.max`` for anything under ``base``.

    The controller has to be the one on the mount the base resolves through:
    the v2 unified fstype, or a v1 mount whose super options name ``pids``.
    Anything else -- tmpfs, an ordinary filesystem, a v1 mount of some other
    controller, or an empty mount table -- is not a pids cgroup, whatever
    files happen to sit in the directory.
    """
    found = _mount_for(base, mountinfo)
    if found is None:
        return False
    fstype, super_options = found
    if fstype == CGROUP2_FSTYPE:
        return True
    if fstype != CGROUP_V1_FSTYPE:
        return False
    return CGROUP_PIDS_CONTROLLER in {
        option.strip() for option in super_options.split(",")
    }


def _pids_cgroup_env() -> Path | None:
    """The delegated pids cgroup base directory, or None when unset.

    Set (RLR_SANDBOX_PIDS_CGROUP) means "cap every sandboxed run", so it is
    preflighted here exactly like the signed policy: a base directory that is
    missing, not writable, or not a pids cgroup at all would otherwise let the
    first tool call discover it and run uncapped -- a plain directory takes the
    broker's writes as licence to create regular files named ``pids.max``, and
    the run then reports a cap the kernel never enforces. Failing at startup
    keeps that decision on the operator's side of the boundary.

    The control files alone are not enough evidence. Any directory can hold a
    regular file named ``pids.max``, and a base that has them by hand passes a
    plain existence check while the kernel enforces nothing there at all. So on
    Linux the mount under the base is asked as well, and only a mount whose
    controller is pids -- v1 with ``pids`` among its super options, or the v2
    unified fstype -- is accepted.
    """
    raw = (os.environ.get("RLR_SANDBOX_PIDS_CGROUP") or "").strip()
    if not raw:
        if (os.environ.get("RLR_SANDBOX_PIDS_CGROUP_EXEMPT") or "").strip() == "1":
            _LOG.warning(
                "RLR_SANDBOX_PIDS_CGROUP_EXEMPT=1: production PID cap DISABLED"
            )
            return None
        raise RuntimeError(
            "RLR_SANDBOX_PIDS_CGROUP is mandatory for production ENFORCE "
            "(or explicitly set RLR_SANDBOX_PIDS_CGROUP_EXEMPT=1)"
        )
    base = Path(raw).expanduser()
    if not base.is_dir():
        raise RuntimeError(
            "RLR_SANDBOX_PIDS_CGROUP is set but is not a directory: "
            f"{base} (install deploy/rlr-sandbox-cgroup.service)"
        )
    missing = [name for name in PIDS_CONTROL_FILES if not (base / name).exists()]
    if missing:
        raise RuntimeError(
            "RLR_SANDBOX_PIDS_CGROUP is not a pids cgroup: "
            f"{base} has no {', '.join(missing)} (the pids controller must be "
            "mounted and delegated to the base; install deploy/rlr-sandbox-cgroup.service)"
        )
    if sys.platform.startswith("linux") and not _mount_enforces_pids(
        base, _read_mountinfo()
    ):
        raise RuntimeError(
            "RLR_SANDBOX_PIDS_CGROUP is not under a pids cgroup mount: "
            f"{base} sits on a filesystem that does not enforce pids.max "
            "(the pids controller must be mounted at or above the base; "
            "install deploy/rlr-sandbox-cgroup.service)"
        )
    if not os.access(base, os.W_OK | os.X_OK):
        raise RuntimeError(
            "RLR_SANDBOX_PIDS_CGROUP is not writable by this user: "
            f"{base} (it must be chowned to the runtime user)"
        )
    return base


def main() -> int:
    home = Path.home()
    data_dir = Path(_env("RLR_DATA_DIR", str(home / "rlr" / "data")))
    kill_file = Path(_env("RLR_KILL_FILE", str(home / "rlr" / "KILL")))
    heartbeat = Path(_env("RLR_HEARTBEAT_FILE", str(home / "rlr" / "heartbeat.txt")))

    policy_raw = (os.environ.get("RLR_POLICY_PATH") or "").strip()
    public_raw = (os.environ.get("RLR_POLICY_PUBLIC_KEY_PATH") or "").strip()
    if not policy_raw or not public_raw:
        raise RuntimeError(
            "production G1 requires RLR_POLICY_PATH and RLR_POLICY_PUBLIC_KEY_PATH"
        )
    policy_path = Path(policy_raw).expanduser()
    public_key_path = Path(public_raw).expanduser()

    # Verify operator authority and the deploy-specific scope/expiry ceiling
    # before constructing anything that can execute effects.
    operator_policy = load_operator_policy(policy_path, public_key_path)

    # Same preflight for the per-run PID cap: resolved (or refused) before the
    # runtime exists, so no sandboxed run can ever fall back to running uncapped.
    pids_cgroup = _pids_cgroup_env()
    # Name the base that is actually enforced, so the log of a running loop
    # carries the cgroup it caps runs in rather than the one that was configured.
    print(f"pids_cgroup={pids_cgroup if pids_cgroup is not None else 'none'}")

    config = Config(
        data_dir=data_dir,
        tick_seconds=_float_env(
            "RLR_TICK_SECONDS", 1.0, minimum=0.25, maximum=10.0
        ),
        # ENFORCE authority. Setting either variable makes the runtime verify an
        # operator-signed policy envelope at startup; without `cryptography`
        # installed (`pip install -e .[crypto]`) it refuses to start instead of
        # falling back to OBSERVATION_ONLY.
        policy_path=policy_path,
        policy_public_key_path=public_key_path,
        sandbox_pids_cgroup=pids_cgroup,
        extra={"kill_file": str(kill_file)},
    )

    telemetry = ProcTelemetry()
    advisor = build_advisor(config)
    runtime = ReflectiveLoopRuntime(config, telemetry=telemetry, slow_brain=advisor)
    runtime_policy = runtime.permits.policy if runtime.permits is not None else None
    if runtime_policy is None or runtime_policy.digest != operator_policy.digest:
        raise RuntimeError(
            "runtime authority does not match the bounded operator policy preflight"
        )
    mode = runtime_policy.mode
    print(f"policy_mode={mode}")
    if mode != MODE_ENFORCE:
        print("policy mode is not ENFORCE; refusing to run the loop")
        return 1
    kill = KillSwitch.from_config(config, permits=runtime.permits)

    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    heartbeat.write_text(
        f"{int(time.time())} started pid={os.getpid()} tick=0\n", encoding="utf-8"
    )

    tick = 0
    while not kill.check():
        tick += 1
        sample = telemetry.sample()
        result = runtime.tick(
            sample,
            capabilities=["vision", "generation", "load_control"],
            context="recent outcomes: stable",
            goal_digest="maintain stable operation",
            allow_default_goal_creation=False,
        )
        choice = result.decision.get("choice")
        heartbeat.write_text(
            f"{int(time.time())} tick={tick} action={choice} "
            f"temp={sample.get('temperature_c')} ram_mib={sample.get('ram_used_mib')} "
            f"confidence={result.final_confidence}\n",
            encoding="utf-8",
        )
        # Sleep one cadence in short chunks so a kill file that appears
        # mid-sleep is observed within a single tick.
        end = time.monotonic() + config.tick_seconds
        while time.monotonic() < end and not kill.check():
            time.sleep(0.25)

    if kill.engaged:
        heartbeat.write_text(
            f"{int(time.time())} stopped: {kill.reason}\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

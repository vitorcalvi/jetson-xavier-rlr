"""Allow-listed privileged executor with idempotency and postcondition verification."""
from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

OUTCOMES = {"APPLIED_VERIFIED", "FAILED_EXECUTION", "FAILED_POSTCONDITION", "NOOP", "ABSTAINED"}


@dataclass(frozen=True)
class ExecutionResult:
    request_id: str
    operation: str
    outcome: str
    exit_code: Optional[int] = None
    detail: str = ""
    observed_state: Optional[dict] = None

    def __post_init__(self):
        if self.outcome not in OUTCOMES:
            raise ValueError("invalid execution outcome")

    def as_dict(self):
        return asdict(self)


class TrustedExecutor:
    """No shell=True and no arbitrary command path is exposed."""
    CGROUP_FILES = {"memory.high", "memory.max", "cpu.max", "pids.max"}

    def __init__(self, store=None, runner=None, allowed_nvp_modes=None,
                 allowed_services=None, cgroup_root="/sys/fs/cgroup/rlr", timeout_s=5.0):
        self.store = store
        self.runner = runner or subprocess.run
        self.allowed_nvp_modes = set(allowed_nvp_modes or {5})
        self.allowed_services = set(allowed_services or set())
        self.cgroup_root = Path(cgroup_root)
        self.timeout_s = float(timeout_s)

    def _existing(self, request_id):
        if not self.store:
            return None
        row = self.store.execution_result(request_id)
        if not row:
            return None
        return ExecutionResult(request_id, row["operation"], row["outcome"],
                               row.get("exit_code"), row.get("detail") or "",
                               row.get("observed_state"))

    def _persist(self, result):
        if self.store:
            self.store.add_execution_result(result.as_dict())
        return result

    def _run(self, argv):
        try:
            return self.runner(argv, capture_output=True, text=True,
                               timeout=self.timeout_s, shell=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return exc

    def execute(self, request_id, operation, args):
        previous = self._existing(request_id)
        if previous is not None:
            return previous
        args = dict(args or {})
        if operation == "set_nvpmodel":
            result = self._set_nvpmodel(request_id, args)
        elif operation == "manage_service":
            result = self._manage_service(request_id, args)
        elif operation == "set_cgroup_value":
            result = self._set_cgroup_value(request_id, args)
        else:
            result = ExecutionResult(request_id, operation, "ABSTAINED",
                                     detail="operation_not_allowed")
        return self._persist(result)

    def _set_nvpmodel(self, request_id, args):
        mode = args.get("mode")
        if isinstance(mode, bool) or not isinstance(mode, int) or mode not in self.allowed_nvp_modes:
            return ExecutionResult(request_id, "set_nvpmodel", "ABSTAINED",
                                   detail="nvpmodel_mode_not_allowed")
        before = self._run(["nvpmodel", "-q"])
        if isinstance(before, Exception):
            return ExecutionResult(request_id, "set_nvpmodel", "FAILED_EXECUTION", detail=str(before)[:300])
        if before.returncode == 0 and self._nvpmodel_matches(before.stdout, mode):
            return ExecutionResult(request_id, "set_nvpmodel", "NOOP", before.returncode,
                                   observed_state={"power_mode": before.stdout.strip()})
        applied = self._run(["nvpmodel", "-m", str(mode)])
        if isinstance(applied, Exception) or applied.returncode != 0:
            return ExecutionResult(request_id, "set_nvpmodel", "FAILED_EXECUTION",
                                   None if isinstance(applied, Exception) else applied.returncode,
                                   str(applied)[:300])
        verify = self._run(["nvpmodel", "-q"])
        if isinstance(verify, Exception) or verify.returncode != 0:
            return ExecutionResult(request_id, "set_nvpmodel", "FAILED_POSTCONDITION",
                                   None if isinstance(verify, Exception) else verify.returncode,
                                   "nvpmodel_readback_failed")
        observed = {"power_mode": verify.stdout.strip()}
        if not self._nvpmodel_matches(verify.stdout, mode):
            return ExecutionResult(request_id, "set_nvpmodel", "FAILED_POSTCONDITION",
                                   verify.returncode, "nvpmodel_readback_mismatch", observed)
        return ExecutionResult(request_id, "set_nvpmodel", "APPLIED_VERIFIED",
                               verify.returncode, observed_state=observed)

    @staticmethod
    def _nvpmodel_matches(text, mode):
        text = text or ""
        if mode == 5 and "MODE_10W" in text:
            return True
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return str(mode) in lines

    def _manage_service(self, request_id, args):
        unit = args.get("unit")
        state = args.get("state")
        if unit not in self.allowed_services or state not in ("active", "inactive"):
            return ExecutionResult(request_id, "manage_service", "ABSTAINED",
                                   detail="service_or_state_not_allowed")
        before = self._run(["systemctl", "is-active", unit])
        if not isinstance(before, Exception) and before.stdout.strip() == state:
            return ExecutionResult(request_id, "manage_service", "NOOP", before.returncode,
                                   observed_state={"active_state": state})
        verb = "start" if state == "active" else "stop"
        applied = self._run(["systemctl", verb, unit])
        if isinstance(applied, Exception) or applied.returncode != 0:
            return ExecutionResult(request_id, "manage_service", "FAILED_EXECUTION",
                                   None if isinstance(applied, Exception) else applied.returncode,
                                   str(applied)[:300])
        verify = self._run(["systemctl", "is-active", unit])
        if isinstance(verify, Exception):
            return ExecutionResult(request_id, "manage_service", "FAILED_POSTCONDITION",
                                   detail="service_readback_failed")
        observed = {"active_state": verify.stdout.strip()}
        if verify.stdout.strip() != state:
            return ExecutionResult(request_id, "manage_service", "FAILED_POSTCONDITION",
                                   verify.returncode, "service_state_mismatch", observed)
        return ExecutionResult(request_id, "manage_service", "APPLIED_VERIFIED",
                               verify.returncode, observed_state=observed)

    def _set_cgroup_value(self, request_id, args):
        name = args.get("name")
        value = args.get("value")
        if name not in self.CGROUP_FILES or not isinstance(value, str):
            return ExecutionResult(request_id, "set_cgroup_value", "ABSTAINED",
                                   detail="cgroup_control_not_allowed")
        if not re.fullmatch(r"(?:max|[0-9]+(?: [0-9]+)?)", value.strip()):
            return ExecutionResult(request_id, "set_cgroup_value", "ABSTAINED",
                                   detail="cgroup_value_not_allowed")
        target = self.cgroup_root / name
        if target.parent.resolve() != self.cgroup_root.resolve() or not target.exists():
            return ExecutionResult(request_id, "set_cgroup_value", "FAILED_EXECUTION",
                                   detail="cgroup_control_unavailable")
        try:
            target.write_text(value.strip())
            observed_value = target.read_text().strip()
        except OSError as exc:
            return ExecutionResult(request_id, "set_cgroup_value", "FAILED_EXECUTION",
                                   detail=str(exc)[:300])
        observed = {name: observed_value}
        if observed_value != value.strip():
            return ExecutionResult(request_id, "set_cgroup_value", "FAILED_POSTCONDITION",
                                   detail="cgroup_readback_mismatch", observed_state=observed)
        return ExecutionResult(request_id, "set_cgroup_value", "APPLIED_VERIFIED",
                               observed_state=observed)

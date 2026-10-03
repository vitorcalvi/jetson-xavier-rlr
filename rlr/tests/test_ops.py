import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "ops"


def test_shell_scripts_parse():
    for name in ("thermal-guard.sh", "check-cgroup-v2.sh", "install-runtime-services.sh",
                 "verify-runtime-services.sh", "rlr-inference-launch.sh"):
        subprocess.run(["sh", "-n", str(OPS / name)], check=True)


def test_systemd_units_verify():
    if not shutil.which("systemd-analyze"):
        pytest.skip("systemd-analyze unavailable")
    subprocess.run(["systemd-analyze", "verify", str(OPS / "thermal-guard.service"),
                    str(OPS / "rlr-inference.service")], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_thermal_guard_prefers_managed_units():
    text = (OPS / "thermal-guard.sh").read_text()
    assert "systemctl kill --kill-who=all --signal=TERM" in text
    assert "ALLOW_PKILL_FALLBACK" in text
    assert "pkill -TERM -f" in text

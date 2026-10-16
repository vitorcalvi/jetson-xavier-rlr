"""G1 S1 trusted-host signer security regressions."""

from __future__ import annotations

import importlib.util
import json
import stat
from pathlib import Path

import pytest

from reflective_loop_runtime.authority import (
    MAX_POLICY_LIFETIME_SECONDS,
    load_operator_policy,
)
from reflective_loop_runtime.kernel.policy import HAVE_CRYPTOGRAPHY

pytestmark = pytest.mark.skipif(
    not HAVE_CRYPTOGRAPHY, reason="cryptography required for production authority"
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy" / "sign_self_maintenance_policy.py"


def _load():
    spec = importlib.util.spec_from_file_location("g1_signer", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key] = value
    return out


def _args(tmp_path):
    return [
        "--operator-key-env", str(tmp_path / "operator-key.env"),
        "--public-key-out", str(tmp_path / "operator-public.pem"),
        "--policy-out", str(tmp_path / "policy.json"),
    ]


def test_generate_keeps_private_key_only_in_0600_env_and_policy_is_dated(
    tmp_path, capsys
):
    module = _load()
    args = _args(tmp_path)
    assert module.main(args + ["--generate", "--valid-hours", "24"]) == 0

    key_env = tmp_path / "operator-key.env"
    public_out = tmp_path / "operator-public.pem"
    policy_out = tmp_path / "policy.json"

    assert stat.S_IMODE(key_env.stat().st_mode) == 0o600
    assert stat.S_IMODE(public_out.stat().st_mode) == 0o600
    assert stat.S_IMODE(policy_out.stat().st_mode) == 0o600

    private_hex = _env(key_env)["RLR_OPERATOR_ED25519_PRIVATE_KEY_HEX"]
    assert len(private_hex) == 64
    out = capsys.readouterr().out
    assert private_hex not in out
    assert private_hex not in public_out.read_text(encoding="utf-8")
    assert private_hex not in policy_out.read_text(encoding="utf-8")

    document = json.loads(policy_out.read_text(encoding="utf-8"))
    version = document["bundle"]["policy_version"]
    assert "issued_at=" in version and "not_after=" in version
    policy = load_operator_policy(policy_out, public_out)
    assert policy.enforced is True


def test_existing_private_key_is_reused_without_being_printed(tmp_path, capsys):
    module = _load()
    args = _args(tmp_path)
    module.main(args + ["--generate", "--valid-hours", "24"])
    first = _env(tmp_path / "operator-key.env")["RLR_OPERATOR_ED25519_PRIVATE_KEY_HEX"]
    capsys.readouterr()

    module.main(args + ["--valid-hours", "24"])
    second = _env(tmp_path / "operator-key.env")["RLR_OPERATOR_ED25519_PRIVATE_KEY_HEX"]
    out = capsys.readouterr().out

    assert first == second
    assert first not in out


def test_missing_key_requires_explicit_generate(tmp_path):
    module = _load()
    with pytest.raises(SystemExit, match="use --generate"):
        module.main(_args(tmp_path))


def test_key_env_permissions_must_be_exactly_0600(tmp_path):
    module = _load()
    args = _args(tmp_path)
    module.main(args + ["--generate"])
    key_env = tmp_path / "operator-key.env"
    key_env.chmod(0o644)
    with pytest.raises(SystemExit, match="0600"):
        module.main(args)


def test_paths_must_be_distinct_and_symlink_free(tmp_path):
    module = _load()
    same = tmp_path / "same"
    with pytest.raises(SystemExit, match="distinct"):
        module.main([
            "--operator-key-env", str(same),
            "--public-key-out", str(same),
            "--policy-out", str(tmp_path / "policy.json"),
            "--generate",
        ])

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "link"
    link_dir.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(SystemExit, match="symlink"):
        module.main([
            "--operator-key-env", str(tmp_path / "key.env"),
            "--public-key-out", str(link_dir / "public.pem"),
            "--policy-out", str(tmp_path / "policy.json"),
            "--generate",
        ])


def test_valid_hours_must_stay_within_signed_deploy_ceiling(tmp_path):
    module = _load()
    over = MAX_POLICY_LIFETIME_SECONDS / 3600.0 + 1.0
    with pytest.raises(SystemExit, match="valid-hours"):
        module.main(_args(tmp_path) + ["--generate", "--valid-hours", str(over)])
    with pytest.raises(SystemExit, match="valid-hours"):
        module.main(_args(tmp_path) + ["--generate", "--valid-hours", "0"])

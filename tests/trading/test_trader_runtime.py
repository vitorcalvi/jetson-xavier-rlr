"""Tests for the trader runtime driver.

The driver is a thin, stdlib-only wrapper that wires preflight + audit +
permit + broker + ledger + loop into a single :class:`TraderLoop`. Most of
the moving parts are covered by their own test modules; the named tests
here exercise the wiring specifically called out by the brief:

* ``test_preflight_rejects_missing_policy`` -- a missing policy file
  surfaces as a :class:`PreflightError` from :func:`build`.
* ``test_preflight_rejects_non_pk_key`` -- a key without the ``PK``
  prefix is refused before any HTTP call.
* ``test_dry_run_needs_no_keys`` -- ``--dry-run`` does not require
  ``ALPACA_PAPER_KEY`` / ``ALPACA_PAPER_SECRET`` and substitutes a stub
  broker that refuses every order.
* ``test_healthcheck_stale_heartbeat_fails`` -- the trader healthcheck
  probe exits non-zero when the heartbeat is older than the freshness
  window.

A handful of additional wiring tests pin the paths, the audit-key
permissions, and the dry-run broker behaviour.
"""

from __future__ import annotations

import json
import os
import stat
import time
from copy import deepcopy
from pathlib import Path

import pytest

from deploy import trader_runtime
from deploy.healthcheck_trader import main as healthcheck_main
from reflective_loop_runtime.authority import paper_trading_policy_bundle
from reflective_loop_runtime.kernel.policy import HAVE_CRYPTOGRAPHY, sign_envelope
from reflective_loop_runtime.trading.broker import PaperOnlyError
from reflective_loop_runtime.trading.loop import ControllerFailStop, TraderLoop

if HAVE_CRYPTOGRAPHY:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

pytestmark = pytest.mark.skipif(
    not HAVE_CRYPTOGRAPHY, reason="cryptography required for production authority"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sign_trading_policy(tmp_path: Path) -> tuple[Path, Path]:
    """Write a signed trading policy + matching public PEM into ``tmp_path``."""
    bundle = paper_trading_policy_bundle(issued_at=int(time.time()) - 60, lifetime_seconds=3600)
    private = Ed25519PrivateKey.generate()
    public_pem = private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_path = tmp_path / "trading-public.pem"
    public_path.write_bytes(public_pem)
    policy_path = tmp_path / "trading-policy.json"
    policy_path.write_text(
        json.dumps(sign_envelope(deepcopy(bundle), private_key=private, key_id="trading")),
        encoding="utf-8",
    )
    return policy_path, public_path


def _make_args(**overrides):
    """Build a fresh argparse Namespace with the driver's defaults."""
    d = {
        "dry_run": False,
        "once": False,
        "no_account_check": False,
        "interval": 60.0,
        "data_dir": None,
    }
    d.update(overrides)
    return trader_runtime.parse_args([]).__class__(**d)


def _base_env(tmp_path: Path) -> dict[str, str]:
    """An env dict with the trading policy + public key wired in."""
    policy, public = _sign_trading_policy(tmp_path)
    return {
        "RLR_TRADING_POLICY_PATH": str(policy),
        "RLR_POLICY_PUBLIC_KEY_PATH": str(public),
    }


# ---------------------------------------------------------------------------
# Named preflight tests
# ---------------------------------------------------------------------------


def test_preflight_rejects_missing_policy(tmp_path: Path) -> None:
    """Missing trading policy file surfaces as :class:`PreflightError`."""
    args = _make_args()
    env = _base_env(tmp_path)
    # Point the policy path at a file that does not exist; the loader will
    # refuse it before any HTTP call.
    env["RLR_TRADING_POLICY_PATH"] = str(tmp_path / "does-not-exist.json")
    env["RLP_PAPER_KEY"] = "PK1234567890ABCDEF"  # ensure key preflight would pass
    env["ALPACA_PAPER_KEY"] = "PK1234567890ABCDEF"
    env["ALPACA_PAPER_SECRET"] = "secret-shhh-do-not-print"
    with pytest.raises(trader_runtime.PreflightError):
        trader_runtime.build(args, env)


def test_preflight_rejects_non_pk_key(tmp_path: Path) -> None:
    """A non-``PK`` key is refused before any HTTP call is attempted."""
    args = _make_args()
    env = _base_env(tmp_path)
    env["ALPACA_PAPER_KEY"] = "AK1234567890ABCDEF"  # live-style prefix
    env["ALPACA_PAPER_SECRET"] = "secret-shhh-do-not-print"
    with pytest.raises(trader_runtime.PreflightError, match="PK"):
        trader_runtime.build(args, env)


def test_dry_run_needs_no_keys(tmp_path: Path) -> None:
    """``--dry-run`` does not require the Alpaca paper keys."""
    args = _make_args(dry_run=True)
    env = _base_env(tmp_path)
    # No ALPACA_PAPER_KEY / ALPACA_PAPER_SECRET set; build() must succeed.
    env.pop("ALPACA_PAPER_KEY", None)
    env.pop("ALPACA_PAPER_SECRET", None)
    loop = trader_runtime.build(args, env)
    assert isinstance(loop, TraderLoop)
    assert loop._dry_run is True
    # The stub broker refuses every order-shaped call.
    with pytest.raises(PaperOnlyError):
        loop._broker.submit(
            "BTC/USD",
            "buy",
            client_order_id="rlr-stub",
            notional=10.0,
        )


# ---------------------------------------------------------------------------
# Wiring tests
# ---------------------------------------------------------------------------


def test_build_registers_order_adapter(tmp_path: Path) -> None:
    """``build`` wires the paper order adapter into the tool broker."""
    args = _make_args(dry_run=True, data_dir=str(tmp_path / "t"))
    env = _base_env(tmp_path)
    loop = trader_runtime.build(args, env)
    assert ("alpaca_paper", "order") in loop._tool_broker._adapters


def test_build_wires_paths(tmp_path: Path) -> None:
    """``build`` creates the canonical file paths under the data dir."""
    args = _make_args(dry_run=True, data_dir=str(tmp_path / "trader"))
    env = _base_env(tmp_path)
    trader_runtime.build(args, env)
    data_dir = tmp_path / "trader"
    assert (data_dir / "trades.db").exists()  # the ledger opens SQLite at build time
    assert (data_dir / "heartbeat.json").exists() is False
    assert (data_dir / "KILL").exists() is False
    assert (data_dir / "audit.db").exists()
    # The audit key was created with mode 0600.
    key_path = data_dir / "audit.key"
    assert key_path.is_file()
    mode = stat.S_IMODE(key_path.stat().st_mode)
    assert mode == 0o600, f"audit.key mode should be 0600, got {oct(mode)}"


def test_build_resolves_home_dir_default(tmp_path: Path, monkeypatch) -> None:
    """Without ``--data-dir`` or ``RLR_TRADER_DIR`` we fall back to ``$HOME/rlr/trader``."""
    monkeypatch.setenv("HOME", str(tmp_path))
    args = _make_args(dry_run=True)
    env = _base_env(tmp_path)
    env.pop("RLR_TRADER_DIR", None)
    loop = trader_runtime.build(args, env)
    # The resolved data dir should live under the HOME we just stubbed.
    assert str(loop._heartbeat_path).startswith(str(tmp_path))


def test_build_honours_env_data_dir(tmp_path: Path) -> None:
    """``RLR_TRADER_DIR`` overrides the ``HOME/rlr/trader`` default."""
    args = _make_args(dry_run=True)
    env = _base_env(tmp_path)
    env["RLR_TRADER_DIR"] = str(tmp_path / "from-env")
    loop = trader_runtime.build(args, env)
    assert str(loop._heartbeat_path).startswith(str(tmp_path / "from-env"))


def test_preflight_error_for_missing_alpaca_key(tmp_path: Path) -> None:
    """Without ``--dry-run`` and without keys the preflight must fail closed."""
    args = _make_args()
    env = _base_env(tmp_path)
    env.pop("ALPACA_PAPER_KEY", None)
    env.pop("ALPACA_PAPER_SECRET", None)
    with pytest.raises(trader_runtime.PreflightError, match="ALPACA_PAPER_KEY"):
        trader_runtime.build(args, env)


def test_build_requires_controller_when_not_dry_run(tmp_path: Path) -> None:
    """Production ``build`` (not dry-run) wires the loop with ``require_controller=True``.

    The I0 safety contract says every order effect must go through a
    durable controller.  The runtime driver is the only production path;
    ``build`` must set ``require_controller=not dry_run`` so the loop's
    safety gate fires when the controller is missing.
    """
    args = _make_args(dry_run=True)
    env = _base_env(tmp_path)
    dry_loop = trader_runtime.build(args, env)
    assert dry_loop._require_controller is False

    args = _make_args(dry_run=False)
    args.no_account_check = True  # tests must never reach the network
    env["ALPACA_PAPER_KEY"] = "PK1234567890ABCDEF"
    env["ALPACA_PAPER_SECRET"] = "secret-shhh-do-not-print"
    prod_loop = trader_runtime.build(args, env)
    assert prod_loop._require_controller is True


# ---------------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------------


def test_healthcheck_stale_heartbeat_fails(tmp_path: Path, monkeypatch) -> None:
    """A heartbeat older than the freshness window makes the probe fail."""
    monkeypatch.setenv("RLR_TRADER_HEARTBEAT_FILE", str(tmp_path / "heartbeat.json"))
    monkeypatch.setenv("RLR_TRADER_HEARTBEAT_MAX_AGE", "1.0")
    heartbeat = tmp_path / "heartbeat.json"
    heartbeat.write_text(
        json.dumps(
            {
                "ts": 0.0,
                "tick": 0,
                "open_positions": {},
                "realized_r_today": 0.0,
                "halted": False,
            }
        ),
        encoding="utf-8",
    )
    # Force the file's mtime into the past so the freshness check trips.
    old = time.time() - 600.0
    os.utime(heartbeat, (old, old))
    rc = healthcheck_main(os.environ)
    assert rc == 1


def test_healthcheck_missing_heartbeat_fails(tmp_path: Path, monkeypatch) -> None:
    """A missing heartbeat file makes the probe fail with a clear message."""
    monkeypatch.setenv("RLR_TRADER_HEARTBEAT_FILE", str(tmp_path / "absent.json"))
    rc = healthcheck_main(os.environ)
    assert rc == 1


def test_healthcheck_fresh_heartbeat_succeeds(tmp_path: Path, monkeypatch) -> None:
    """A fresh heartbeat with sane payload reports HEALTHY."""
    monkeypatch.setenv("RLR_TRADER_HEARTBEAT_FILE", str(tmp_path / "heartbeat.json"))
    heartbeat = tmp_path / "heartbeat.json"
    heartbeat.write_text(
        json.dumps(
            {
                "ts": 0.0,
                "tick": 1,
                "open_positions": {"BTC/USD": {"qty": 0.01}},
                "realized_r_today": 0.5,
                "halted": False,
            }
        ),
        encoding="utf-8",
    )
    rc = healthcheck_main(os.environ)
    assert rc == 0


# ---------------------------------------------------------------------------
# Default sleep wiring
# ---------------------------------------------------------------------------


def test_run_loop_default_sleep_once():
    calls = []

    class _Loop:
        def tick(self, now):
            calls.append(now)

    t = [120.0]  # exactly on a 60 s boundary, so the default sleep returns immediately
    trader_runtime._run_loop(_Loop(), interval=60.0, once=True, clock=lambda: t[0])
    assert calls and calls[0] >= 120.0


# ---------------------------------------------------------------------------
# ControllerFailStop plumbing (G5 I2)
# ---------------------------------------------------------------------------


def test_runtime_reraises_controller_failstop() -> None:
    """A ``ControllerFailStop`` from ``tick`` bubbles out of ``_run_loop``; a generic exception is still swallowed."""

    class _FailStopLoop:
        def __init__(self) -> None:
            self.calls = 0

        def tick(self, now: float) -> None:
            self.calls += 1
            raise ControllerFailStop("storage is gone")

    fail_loop = _FailStopLoop()
    with pytest.raises(ControllerFailStop):
        trader_runtime._run_loop(fail_loop, interval=60.0, once=True, clock=lambda: 120.0)
    assert fail_loop.calls == 1, "ControllerFailStop must propagate without a second tick attempt"

    generic_calls: list = []

    class _RaisingLoop:
        def tick(self, now: float) -> None:
            generic_calls.append(now)
            raise RuntimeError("broker outage")

    # A non-fail-stop exception must be swallowed so the loop keeps running.
    trader_runtime._run_loop(
        _RaisingLoop(),
        interval=60.0,
        once=True,
        clock=lambda: 120.0,
    )
    assert generic_calls, "generic exceptions should not abort the run loop"

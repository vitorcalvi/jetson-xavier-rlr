#!/usr/bin/env python3
"""User-level RLR paper-trader driver.

This is the production entrypoint for the paper-trading lane: it preflights
operator authority, builds the permit-gated trader loop the rest of the
runtime uses, and then runs that loop on a one-minute cadence until the
external kill file appears, an exception is raised, or ``--once`` says we are
only here for a single tick.

The driver is deliberately thin and stdlib-only:

* authority is verified up front through
  :func:`~reflective_loop_runtime.authority.load_trading_policy`;
* the paper-broker keys are checked in the same preflight;
* account status (ACTIVE / crypto_status ACTIVE) is checked unless
  ``--no-account-check`` or ``--dry-run`` is passed;
* the persistent audit key is created with mode 0600 in the data directory,
  exactly the way :mod:`reflective_loop_runtime.runtime` creates it for the
  main loop;
* every tick is run inside a broad ``except`` so a single bad symbol never
  terminates the loop -- ``main`` returns 0 on success and only a preflight
  failure exits non-zero.

The unit-tested entry point is :func:`build`, which takes the parsed CLI
arguments and the process environment and returns a fully wired
:class:`~reflective_loop_runtime.trading.loop.TraderLoop`. Tests exercise
the wiring without spinning up the network or the kernel; production calls
:meth:`TraderLoop.run_forever` on the returned loop.

The production build also wires a durable
:class:`~reflective_loop_runtime.trading.orders.OrderController` so each
tick begins with a broker reconciliation, an operator kill-file escalation,
and a controller-driven entry/exit gate.  ``--dry-run`` deliberately
substitutes ``controller=None`` so the stub broker cannot be accidentally
promoted into a real trade.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import secrets
import sys
import time
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from reflective_loop_runtime.authority import (
    AuthorityError,
    load_trading_policy,
)
from reflective_loop_runtime.kernel.audit import AuditLog
from reflective_loop_runtime.kernel.manifest import ToolManifest
from reflective_loop_runtime.kernel.permit import PermitIssuer, PermitService
from reflective_loop_runtime.kernel.policy import (
    ALG_ED25519,
    HAVE_CRYPTOGRAPHY,
    MissionPolicy,
    verified_authority,
)
from reflective_loop_runtime.kernel.toolbroker import ToolBroker
from reflective_loop_runtime.trading.alerts import AlertingLedger
from reflective_loop_runtime.trading.alerts import from_env as alerts_from_env
from reflective_loop_runtime.trading.broker import (
    PaperBroker,
    PaperOnlyError,
    make_order_adapter,
)
from reflective_loop_runtime.trading.http import _default_http, _default_post
from reflective_loop_runtime.trading.ledger import Ledger
from reflective_loop_runtime.trading.loop import ControllerFailStop, TraderLoop, trading_manifest
from reflective_loop_runtime.trading.market import (
    asset_class,
    configured_symbols,
    fetch_snapshots,
)
from reflective_loop_runtime.trading.orders import OrderController

_LOG = logging.getLogger("rlr.trader")


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _env_str(env: Mapping[str, str], name: str, default: str = "") -> str:
    """Return ``env[name]`` if set, else ``default``. Whitespace-only is empty."""
    value = env.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_float(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return a finite float read from ``env`` with optional bounds.

    A malformed value silently falls back to ``default`` so an operator
    typo never crashes the loop: the driver logs and moves on. Bounds are
    inclusive.
    """
    raw = env.get(name)
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
        # Never echo the raw value: env files may also contain secrets.
        _LOG.warning("invalid %s; using safe default", name)
        return float(default)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trader_runtime",
        description=(
            "RLR paper-trader driver: preflights operator authority and the "
            "paper broker, then runs the permit-gated loop on a one-minute "
            "cadence."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip the Alpaca-key and account preflight; orders are stubbed "
            "so a smoke test can run without any real broker."
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single tick and exit (used by tests and ad-hoc checks).",
    )
    parser.add_argument(
        "--no-account-check",
        action="store_true",
        help=(
            "Skip the Alpaca account-status check (still requires the keys "
            "and a loaded policy)."
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help=(
            "Seconds between ticks (default: 60). Must be > 0 and <= 3600. "
            "Ignored when --once is set."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Data directory for trades.db, journal.jsonl, heartbeat.json, "
            "KILL, and audit.db (default: env RLR_TRADER_DIR or ~/rlr/trader)."
        ),
    )
    return parser


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse ``argv`` with the driver's CLI grammar; no logging on error."""
    parser = _build_parser()
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Audit key (mirrors runtime._load_or_create_audit_key)
# ---------------------------------------------------------------------------


_AUDIT_KEY_PATH = "audit.key"
_AUDIT_KEY_LEN = 32


def _load_or_create_audit_key(data_dir: str | os.PathLike[str]) -> bytes:
    """Return the trader's persistent audit-chain key, creating it 0600 once.

    The key must outlive the process so the hash chain stays verifiable
    across restarts and cannot be re-forged by anyone who can rewrite the
    database. It is never returned to, printed by, or reachable from
    model-facing code.
    """
    path = Path(data_dir) / _AUDIT_KEY_PATH
    try:
        data = path.read_bytes()
    except OSError:
        data = b""
    if len(data) == _AUDIT_KEY_LEN:
        return data
    key = secrets.token_bytes(_AUDIT_KEY_LEN)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, key)
    finally:
        os.close(fd)
    try:
        os.chmod(str(path), 0o600)
    except OSError:  # pragma: no cover - best effort on platforms without chmod
        pass
    return key


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


class PreflightError(RuntimeError):
    """Raised when a startup preflight check fails.

    Preflight failures exit the driver non-zero: the systemd unit treats a
    non-zero exit as a restart reason, and a sustained preflight failure
    means the trader must not come up at all.
    """


@dataclass(frozen=True)
class PreflightResult:
    """The resolved preflight state the driver will start under.

    ``policy`` is the verified operator authority; ``broker`` is either a
    real :class:`PaperBroker` or a dry-run stub; ``http_get`` /
    ``http_post`` are the callables the broker will use; ``data_dir`` is
    the resolved, mkdir-ed trader directory.
    """

    policy: MissionPolicy
    broker: Any
    http_get: Callable[..., Any]
    http_post: Callable[..., Any]
    data_dir: Path
    trades_db: Path
    journal_path: Path
    heartbeat_path: Path
    kill_path: Path
    audit_db: Path
    audit_key: bytes


def _resolve_data_dir(args: argparse.Namespace, env: Mapping[str, str]) -> Path:
    """Resolve the data directory from args / env / HOME."""
    raw = args.data_dir if args.data_dir else _env_str(env, "RLR_TRADER_DIR")
    if raw:
        return Path(raw).expanduser()
    home = _env_str(env, "HOME", str(Path.home()))
    return Path(home) / "rlr" / "trader"


def _check_alpaca_credentials(env: Mapping[str, str]) -> tuple[str, str]:
    """Return ``(key, secret)`` or raise :class:`PreflightError`."""
    key = _env_str(env, "ALPACA_PAPER_KEY")
    secret = _env_str(env, "ALPACA_PAPER_SECRET")
    if not key:
        raise PreflightError(
            "ALPACA_PAPER_KEY is required (unless --dry-run is passed)"
        )
    if not secret:
        raise PreflightError(
            "ALPACA_PAPER_SECRET is required (unless --dry-run is passed)"
        )
    if not key.startswith("PK"):
        raise PreflightError(
            "ALPACA_PAPER_KEY must start with 'PK' (paper-only)"
        )
    return key, secret


def _load_trading_policy_or_raise(env: Mapping[str, str]) -> MissionPolicy:
    """Load the signed trading policy or raise :class:`PreflightError`."""
    if not HAVE_CRYPTOGRAPHY:
        raise PreflightError(
            "ed25519 backend unavailable; install 'cryptography' (pip install "
            "-e .[crypto]) so production trader authority can be verified"
        )
    policy_path = _env_str(env, "RLR_TRADING_POLICY_PATH")
    public_key_path = _env_str(env, "RLR_POLICY_PUBLIC_KEY_PATH")
    if not policy_path or not public_key_path:
        raise PreflightError(
            "RLR_TRADING_POLICY_PATH and RLR_POLICY_PUBLIC_KEY_PATH are required"
        )
    try:
        return load_trading_policy(policy_path, public_key_path)
    except AuthorityError as exc:
        raise PreflightError(f"trading policy preflight failed: {exc}") from exc


def _check_account(broker: PaperBroker) -> None:
    """Confirm the broker account is ACTIVE and crypto trading is enabled."""
    try:
        account = broker.account()
    except Exception as exc:  # noqa: BLE001 - any account failure is preflight
        raise PreflightError(f"broker account unreachable: {exc}") from exc
    if not isinstance(account, Mapping):
        raise PreflightError(
            f"broker account payload is not a mapping (got {type(account).__name__})"
        )
    status = str(account.get("status") or "").upper()
    if status != "ACTIVE":
        raise PreflightError(
            f"broker account status is {status!r}; expected 'ACTIVE'"
        )
    crypto_status = account.get("crypto_status")
    if crypto_status is None:
        # Some paper accounts do not expose crypto_status explicitly; an absent
        # field is treated as "enabled" so the trader does not fail on a payload
        # the Alpaca paper API has not yet populated.
        return None
    crypto_norm = str(crypto_status).upper()
    if crypto_norm != "ACTIVE":
        raise PreflightError(
            f"broker crypto_status is {crypto_norm!r}; expected 'ACTIVE'"
        )
    return None


# ---------------------------------------------------------------------------
# Dry-run broker stub
# ---------------------------------------------------------------------------


class _DryRunBroker:
    """A broker stub used in :func:`build` when ``--dry-run`` is requested.

    The stub refuses every order-shaped call so a misconfigured ``--dry-run``
    cannot accidentally route a real order through the wire. ``account`` is
    still reachable so the heartbeat path can stay honest without secrets.
    """

    def __init__(self, account: Mapping[str, Any]) -> None:
        self._account = dict(account)

    def account(self) -> Mapping[str, Any]:
        return dict(self._account)

    def submit(self, *_args: Any, **_kwargs: Any) -> Any:
        raise PaperOnlyError("dry-run trader cannot submit orders")

    def get_order(self, *_args: Any, **_kwargs: Any) -> Any:
        raise PaperOnlyError("dry-run trader cannot read orders")

    def wait_filled(self, *_args: Any, **_kwargs: Any) -> Any:
        raise PaperOnlyError("dry-run trader cannot wait for fills")

    # Reconciliation-shape methods -- called by the controller's prelude when
    # ``--dry-run`` does NOT pass ``controller=None``.  They are intentionally
    # returning empties so the dry-run smoke test of the prelude never trips.
    def positions(self) -> list[dict[str, Any]]:
        return []

    def orders_since(
        self, after_iso: str, *, page_limit: int = 500, max_pages: int = 50
    ) -> list[dict[str, Any]]:
        return []


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _make_manifest_for_policy(policy: MissionPolicy) -> ToolManifest:
    """Build the trading manifest with the trading policy as its bound policy.

    The :mod:`trading.loop.trading_manifest` helper returns a fresh manifest
    that knows about ``alpaca_paper.order``; the policy binding is the
    caller-supplied operator authority.
    """
    manifest = trading_manifest()
    return manifest


def _permit_service_for(
    policy: MissionPolicy,
    *,
    manifest: ToolManifest,
    audit_db: Path,
) -> PermitService:
    """Build a :class:`PermitService` keyed off ``policy``'s digest.

    ``PermitService`` already enforces that the policy was operator-signed
    (it refuses anything that did not verify with :data:`ALG_ED25519`); if
    the policy we just loaded does not pass that gate, refuse early here
    rather than at first issue.
    """
    if (
        policy.signature_alg != ALG_ED25519
        or not policy.enforced
        or not policy.signed
        or not verified_authority(policy)
    ):
        raise PreflightError(
            "trading policy did not verify as Ed25519 ENFORCE authority"
        )
    issuer = PermitIssuer()
    return PermitService(
        issuer,
        policy,
        manifest=manifest,
        db_path=audit_db.with_name("permits-trader.db"),
    )


def _make_tool_broker(
    permits: PermitService,
    *,
    manifest: ToolManifest,
    audit_log: AuditLog,
    broker: Any,
    workspace: Path,
) -> ToolBroker:
    """Build a :class:`ToolBroker` with the trading manifest + audit."""
    return ToolBroker(
        permits,
        audit=audit_log,
        workspace=workspace,
        manifest=manifest,
        adapters={("alpaca_paper", "order"): make_order_adapter(broker)},
    )


def _history_start_iso(*, now: float | None = None, days: int = 7) -> str:
    """Return an ISO-8601 timestamp ``days`` before ``now`` (UTC)."""
    current = datetime.fromtimestamp(
        float(time.time() if now is None else now), tz=timezone.utc
    )
    return (current - timedelta(days=int(days))).isoformat()


def _orb5_fn(*, experimental: bool = False) -> Callable[..., Any]:
    from reflective_loop_runtime.trading.orb5 import decide_orb5, decide_orb5_explore

    return decide_orb5_explore if experimental else decide_orb5


def build(args: argparse.Namespace, env: Mapping[str, str]) -> TraderLoop:
    """Resolve preflight and wire a :class:`TraderLoop`.

    Returns a loop whose ``tick`` is ready to call. The function never
    touches the network or the broker for the dry-run case: a stub broker
    is substituted so the loop is safe to run in any test harness.

    In the production (non-dry-run) path the loop is paired with a durable
    :class:`OrderController` whose SQLite file lives at
    ``data_dir / "orders.db"``; the controller drives begin_day,
    killmodes, and broker reconciliation on every tick.  ``--dry-run``
    deliberately passes ``controller=None`` so the loop is bit-for-bit
    compatible with the pre-controller smoke tests.
    """
    data_dir = _resolve_data_dir(args, env)
    data_dir.mkdir(parents=True, exist_ok=True)
    trades_db = data_dir / "trades.db"
    journal_path = data_dir / "journal.jsonl"
    heartbeat_path = data_dir / "heartbeat.json"
    kill_path = data_dir / "KILL"
    audit_db = data_dir / "audit.db"
    orders_db = data_dir / "orders.db"

    dry_run = bool(getattr(args, "dry_run", False))
    equity = asset_class() == "us_equity"
    strategy = _env_str(env, "RLR_TRADER_STRATEGY", "trend_vwap").strip().lower()
    if strategy not in ("trend_vwap", "orb5", "orb5_explore", "autonomy_1m_smoke"):
        raise PreflightError(f"unknown RLR_TRADER_STRATEGY: {strategy!r}")
    symbols = configured_symbols()
    no_account_check = bool(getattr(args, "no_account_check", False))

    http_get = _default_http
    http_post = _default_post
    if dry_run:
        policy = _load_trading_policy_or_raise(env)
        broker: Any = _DryRunBroker(
            {
                "status": "ACTIVE",
                "crypto_status": "ACTIVE",
                "id": "dry-run-account",
            }
        )
    else:
        policy = _load_trading_policy_or_raise(env)
        key, secret = _check_alpaca_credentials(env)
        broker = PaperBroker(
            key,
            secret,
            http_get=http_get,
            http_post=http_post,
            # US-equity fractional/notional orders must be day orders.
            time_in_force="day" if equity else "gtc",
        )
        if not no_account_check:
            _check_account(broker)

    audit_key = _load_or_create_audit_key(data_dir)
    audit_log = AuditLog(str(audit_db), key=audit_key)
    manifest = _make_manifest_for_policy(policy)
    permits = _permit_service_for(policy, manifest=manifest, audit_db=audit_db)
    workspace_dir = data_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    tool_broker = _make_tool_broker(
        permits,
        manifest=manifest,
        audit_log=audit_log,
        broker=broker,
        workspace=workspace_dir,
    )

    equity_fee = 0.0
    crypto_fee = _env_float(env, "RLR_CRYPTO_FEE_RATE", 0.0025, minimum=0.0, maximum=0.01)
    hardened = _env_str(env, "RLR_TRADER_HARDENED", "") == "1" and not equity
    ledger = Ledger(trades_db, journal_path, modeled_crypto_fee_rate=crypto_fee if hardened else equity_fee)
    alerter = alerts_from_env(env, data_dir=data_dir)
    if alerter.enabled:
        # push alerts for orders/halts/incidents; never raises (P2)
        ledger = AlertingLedger(ledger, alerter, label="shadow" if _env_str(env, "RLR_TRADER_SHADOW", "") == "1" else "paper")
    limits: Any = None
    try:
        from reflective_loop_runtime.trading.model import Limits as _Limits

        limits = _Limits(
            hardened_entries=hardened,
            vwap_failure_exit=hardened,
            crypto_fee_rate=crypto_fee if hardened else 0.0,
            min_net_target_r=_env_float(env, "RLR_MIN_NET_TARGET_R", 0.5, minimum=0.0, maximum=5.0),
        )
    except Exception:  # pragma: no cover - decider lane is on main
        limits = None
    if strategy in ("orb5", "orb5_explore"):
        # outside the try: a failed override must raise, never run with defaults
        import dataclasses

        from reflective_loop_runtime.trading.orb5 import ORB5_LIMITS, ORB5_EXPLORE_LIMITS

        limits = dataclasses.replace(limits, **(ORB5_EXPLORE_LIMITS if strategy == "orb5_explore" else ORB5_LIMITS))

    if strategy == "autonomy_1m_smoke":
        if not equity or symbols != ("SPY",):
            raise PreflightError("1m autonomy smoke requires US equity SPY only")
        import dataclasses
        from reflective_loop_runtime.trading.autonomy_smoke_1m import SMOKE_LIMITS
        limits = dataclasses.replace(limits, **SMOKE_LIMITS)

    data_headers: dict[str, str] = {}
    if equity:
        # Alpaca stock data (unlike crypto) requires the API key headers,
        # including in dry-run.
        data_key = _env_str(env, "ALPACA_PAPER_KEY")
        data_secret = _env_str(env, "ALPACA_PAPER_SECRET")
        if not data_key or not data_secret:
            raise PreflightError("us_equity mode needs ALPACA_PAPER_KEY/SECRET for market data")
        data_headers = {
            "APCA-API-KEY-ID": data_key,
            "APCA-API-SECRET-KEY": data_secret,
        }

    def market_fn(*, symbols: tuple[str, ...], now: float) -> dict[str, Any]:
        return fetch_snapshots(symbols, http=http_get, now=now, headers=data_headers)

    # Controller: only enabled in the production (non-dry-run) path.  Dry-run
    # gets ``controller=None`` so the existing dry-run smoke tests stay
    # bit-for-bit compatible.
    if dry_run:
        controller: Any = None
        history_start_iso: str | None = None
    else:
        controller = OrderController(orders_db)
        history_start_iso = _history_start_iso()

    evidence_gate = None
    if _env_str(env, "RLR_ADAPTIVE_EVIDENCE_GATE", "0") == "1":
        if dry_run:
            # Even in dry-run, no external effects, but preserve optional test
            # isolation by not requiring pre-existing operator scope files.
            evidence_gate = None
        elif equity:
            raise PreflightError("adaptive paper evidence gate presently supports crypto only")
        else:
            from reflective_loop_runtime.trading.adaptive_gate import AdaptivePaperGate
            pubkey = _env_str(env, "RLR_POLICY_PUBLIC_KEY_PATH")
            if not pubkey:
                raise PreflightError("adaptive evidence gate requires signed policy public key")
            package_dir = Path(__file__).resolve().parent.parent
            strat_file = (
                package_dir / "src" / "reflective_loop_runtime" / "trading" /
                ("orb5.py" if strategy == "orb5" else "decider.py")
            )
            evidence_gate = AdaptivePaperGate(
                research_dir=data_dir / "adaptive_research",
                public_key_path=Path(pubkey),
                decider_file=strat_file,
                strategy_id=strategy,
                controller=controller,
            )

    if strategy == "autonomy_1m_smoke":
        # A durable gate prevents *another* entry after a closed paper cycle,
        # including across process restarts or across days.
        from reflective_loop_runtime.trading.autonomy_smoke_1m import decide_autonomy_smoke_1m
        from reflective_loop_runtime.trading.model import Decision
        from reflective_loop_runtime.trading.ledger import Ledger as _Ledger
        marker = data_dir / "AUTONOMY_1M_TEST_COMPLETE"
        def _smoke(snap, position, limits, *, now):
            if position is None:
                # Rely on local durable ledger for lifecycle, not time-of-day
                # counters that reset overnight.
                if marker.exists():
                    return Decision(symbol=snap.symbol, action="hold", reason="autonomy_1m_finished")
                probe = _Ledger(trades_db, journal_path)
                try:
                    if probe.closed_trades():
                        marker.write_text("completed\n")
                        return Decision(symbol=snap.symbol, action="hold", reason="autonomy_1m_finished")
                finally:
                    probe.close()
            return decide_autonomy_smoke_1m(snap, position, limits, now=now)
        smoke_strategy = _smoke
    else:
        smoke_strategy = None

    return TraderLoop(
        market_fn=market_fn,
        broker=broker,
        permits=permits,
        tool_broker=tool_broker,
        ledger=ledger,
        limits=limits,
        clock=time.time,
        kill_path=kill_path,
        dry_run=dry_run,
        heartbeat_path=heartbeat_path,
        symbols=symbols,
        controller=controller,
        data_dir=data_dir,
        history_start_iso=history_start_iso,
        require_controller=not dry_run,
        entry_evidence_gate=evidence_gate.evaluate if evidence_gate is not None else None,
        equity_session=equity,
        market_open_fn=getattr(broker, "market_is_open", None) if equity else None,
        strategy_fn=(smoke_strategy if smoke_strategy is not None else
                    (_orb5_fn(experimental=strategy == "orb5_explore")
                     if strategy in ("orb5", "orb5_explore") else None)),
        shadow=_env_str(env, "RLR_TRADER_SHADOW", "") == "1",
        shadow_autorun=_env_str(env, "RLR_TRADER_SHADOW_AUTORUN", "") == "1",
        # paper broker only (PaperBroker refuses any non-paper host)
        paper_autorun=_env_str(env, "RLR_TRADER_PAPER_AUTORUN", "") == "1",
        buy_fee_in_asset=0.0 if equity else 0.0025,
    )


# ---------------------------------------------------------------------------
# Cadence helpers
# ---------------------------------------------------------------------------


def _next_boundary(now: float, interval: float) -> float:
    """Return the next ``interval``-aligned epoch boundary after ``now``."""
    return math.ceil(float(now) / float(interval)) * float(interval)


def _sleep_until(target: float, *, clock: Callable[[], float]) -> None:
    """Sleep in short slices until ``target`` so a kill file is observed."""
    while True:
        remaining = float(target) - float(clock())
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.25))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _run_loop(
    loop: TraderLoop,
    *,
    interval: float,
    once: bool,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] | None = None,
) -> None:
    """Drive ``loop`` on a one-minute cadence; exit only on kill or error.

    ``sleep`` is taken as a parameter so a test can swap it out; production
    calls the default, which slices the wait so a kill file seen mid-sleep
    is observed within a quarter-second.
    """
    if sleep is None:
        def sleep(target: float) -> None:
            _sleep_until(target, clock=clock)
    while True:
        now = float(clock())
        target = _next_boundary(now, float(interval))
        sleep(target)
        try:
            loop.tick(float(clock()))
        except ControllerFailStop:
            # Controller storage failed -- the trader process must stop so
            # systemd restarts it and the next prelude reconciles from a
            # fresh persisted HALT.  Re-raise so the surrounding _run_loop
            # and main() propagate a non-zero exit code.
            raise
        except Exception as exc:  # noqa: BLE001 - one tick must not kill the loop
            _LOG.warning(
                "trader tick raised %s: %s", type(exc).__name__, exc
            )
        if once:
            return


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns 0 on a clean run, non-zero on preflight error."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(sys.argv[1:] if argv is None else argv)
    env: MutableMapping[str, str] = os.environ
    if not (0.0 < float(args.interval) <= 3600.0):
        print(f"UNHEALTHY: --interval must be in (0, 3600]; got {args.interval}", file=sys.stderr)
        return 1
    try:
        loop = build(args, env)
    except PreflightError as exc:
        print(f"UNHEALTHY: preflight failed: {exc}", file=sys.stderr)
        return 1
    if args.once:
        # A single tick is a smoke test, not a real run: we still sleep until
        # the next boundary so the path is exercised exactly the way the long-
        # running loop exercises it.
        try:
            _run_loop(loop, interval=float(args.interval), once=True)
        except Exception as exc:  # noqa: BLE001 - even the smoke test must not crash
            print(f"UNHEALTHY: trader tick failed: {exc}", file=sys.stderr)
            return 1
        return 0
    try:
        _run_loop(loop, interval=float(args.interval), once=False)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # noqa: BLE001 - main must only exit on preflight errors
        print(f"UNHEALTHY: trader loop aborted: {exc}", file=sys.stderr)
        return 1
    return 0


__all__ = [
    "PreflightError",
    "PreflightResult",
    "_load_or_create_audit_key",
    "build",
    "main",
    "parse_args",
    "_build_parser",
    "_history_start_iso",
]


if __name__ == "__main__":
    raise SystemExit(main())

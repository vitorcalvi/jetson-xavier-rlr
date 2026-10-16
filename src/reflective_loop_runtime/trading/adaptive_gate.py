"""Fail-closed paper-entry research gate.

No broker API or model call. Missing/corrupt/unapproved studies return NO_TRADE.
The underlying signed mission policy and order governor remain authoritative.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reflective_loop_runtime.trading.autonomy import (
    NO_TRADE,
    PAPER_ELIGIBLE,
    Candidate,
    DailyOutcome,
    EvaluationPolicy,
    ExperimentRegistry,
    MarketContext,
    propose_and_journal,
)

_SCOPE_DOMAIN = b"RLR-STRATEGY-SCOPE-v1\n"
_ALLOWED_STRATEGIES = frozenset(("trend_vwap", "orb5"))


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _signed_scope(scope_path: Path, key_path: Path, now: float) -> tuple[frozenset[str], int, bool]:
    # Scope issuer must be operator key holder; this code does not sign.
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        envelope = json.loads(scope_path.read_text())
        if not isinstance(envelope, dict) or set(envelope) != {"payload", "signature_base64"}:
            return frozenset(), 0, False
        payload = envelope["payload"]
        if not isinstance(payload, dict) or set(payload) != {
            "allowed_fingerprints",
            "issued_at",
            "expires_at",
            "mode",
            "symbol",
        }:
            return frozenset(), 0, False
        key = serialization.load_pem_public_key(key_path.read_bytes())
        if not isinstance(key, Ed25519PublicKey):
            return frozenset(), 0, False
        key.verify(
            base64.b64decode(envelope["signature_base64"], validate=True),
            _SCOPE_DOMAIN + _canonical(payload),
        )
        issued, expires = int(payload["issued_at"]), int(payload["expires_at"])
        fps = payload["allowed_fingerprints"]
        if (
            payload["mode"] != "paper"
            or payload["symbol"] != "SOL/USD"
            or not isinstance(fps, list)
            or len(fps) > 25
            or any(
                not isinstance(x, str)
                or len(x) != 64
                or any(c not in "0123456789abcdef" for c in x)
                for x in fps
            )
            or len(set(fps)) != len(fps)
            or issued > now
            or expires <= now
            or expires - issued > 30 * 86400
            or issued < now - 30 * 86400
        ):
            return frozenset(), 0, False
        return frozenset(fps), expires, True
    except (OSError, ValueError, TypeError, KeyError, OverflowError, ImportError, Exception):
        return frozenset(), 0, False


def _daily(period: object) -> tuple[DailyOutcome, ...]:
    if not isinstance(period, list) or len(period) > 2000:
        raise ValueError("invalid study period")
    return tuple(
        DailyOutcome(day=str(x["day"]), net_r=float(x["net_r"]), trades=int(x["trades"]))
        for x in period
    )


def _candidates(
    study_path: Path, evidence_dir: Path, strategy_file: Path, strategy_name: str
) -> tuple[Candidate, ...]:
    if not study_path.is_file():
        return ()
    doc = json.loads(study_path.read_text())
    if not isinstance(doc, dict) or set(doc) != {"version", "candidates"} or doc["version"] != 1:
        raise ValueError("malformed study catalog")
    entries = doc["candidates"]
    if not isinstance(entries, list) or len(entries) > 64:
        raise ValueError("oversized study catalog")
    expected_code_sha = hashlib.sha256(strategy_file.read_bytes()).hexdigest()
    results = []
    root = evidence_dir.resolve()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("malformed study")
        sid = str(entry["strategy_id"])
        if sid not in _ALLOWED_STRATEGIES or sid != strategy_name:
            continue  # Never substitute a different strategy without a separate approved deploy.
        artifact = (root / str(entry["artifact_file"])).resolve()
        if not artifact.is_relative_to(root) or not artifact.is_file():
            raise ValueError("untrusted evidence path")
        sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if sha != entry["artifact_sha256"] or entry["code_sha256"] != expected_code_sha:
            raise ValueError("stale code or corrupted evidence")
        results.append(
            Candidate(
                strategy_id=sid,
                code_sha256=expected_code_sha,
                symbol=str(entry["symbol"]),
                feed=str(entry["feed"]),
                regimes=frozenset(entry["regimes"]),
                train=_daily(entry["train"]),
                validation=_daily(entry["validation"]),
                holdout=_daily(entry["holdout"]),
                forward_paper=_daily(entry["forward_paper"]),
                costs_audited=entry.get("costs_audited") is True,
                leakage_audited=entry.get("leakage_audited") is True,
                holdout_sealed_before_testing=entry.get("holdout_sealed_before_testing") is True,
                artifact_sha256=sha,
            )
        )
    return tuple(results)


class AdaptivePaperGate:
    """Veto-only adapter to TraderLoop; no order placement, model authority, or key access."""

    def __init__(
        self,
        *,
        research_dir: Path,
        public_key_path: Path,
        decider_file: Path,
        strategy_id: str,
        controller: Any,
        allow_no_trade: bool = True,
    ):
        self.research_dir = Path(research_dir)
        self.public_key_path = Path(public_key_path)
        self.decider_file = Path(decider_file)
        self.strategy_id = strategy_id
        self.controller = controller
        self.research_dir.mkdir(parents=True, exist_ok=True)
        (self.research_dir / "evidence").mkdir(exist_ok=True)
        # A catalog may only be initialized as empty; never seed it with synthetic profits.
        cat = self.research_dir / "candidates.json"
        if not cat.exists():
            cat.write_text('{"version":1,"candidates":[]}\n')
        self.registry = ExperimentRegistry(self.research_dir / "experiments.db")
        self.policy = EvaluationPolicy(require_sip_for_paper=False)
        self.last = {"mode": NO_TRADE, "reason": "not_yet_evaluated"}
        self._write_status(self.last)

    def _write_status(self, record: dict[str, Any]) -> None:
        target = self.research_dir / "last_selection.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        os.replace(tmp, target)

    def evaluate(self, symbol: str, features: Any, now: float) -> bool:
        # This method is called only after deterministic decider + risk governor
        # approve a BUY; exits never pass through this experimental layer.
        safe = False
        status: dict[str, Any] = {
            "at_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "symbol": symbol,
            "strategy_id": self.strategy_id,
            "mode": NO_TRADE,
            "reason": "unverified",
        }
        try:
            if symbol != "SOL/USD":
                status["reason"] = "symbol_not_authorized"
                return False
            fps, expiry, signed = _signed_scope(
                self.research_dir / "approved_scope.json", self.public_key_path, now
            )
            status["operator_scope_verified"] = signed
            candidates = _candidates(
                self.research_dir / "candidates.json",
                self.research_dir / "evidence",
                self.decider_file,
                self.strategy_id,
            )
            for cand in candidates:
                self.registry.register(cand)
            control = self.controller.snapshot()
            reconciled = (
                control.get("reconciled") is True
                and not control.get("uncertain")
                and not control.get("incident")
                and not control.get("unexpected_fills")
                and control.get("mode") == "RUN"
            )
            e20, e50 = features.ema20_5m, features.ema50_5m
            regime = (
                "trend_up"
                if e20 is not None and e50 is not None and e20 > e50
                else "trend_down"
                if e20 is not None and e50 is not None and e20 < e50
                else "ranging"
            )
            age = features.quote_age_s
            spread = features.spread_pct
            context = MarketContext(
                symbol=symbol,
                regime=regime,
                now_epoch=int(now),
                feed="alpaca_crypto",
                quote_age_seconds=float(age) if age is not None else math.inf,
                spread_bps=float(spread) * 100 if spread is not None else math.inf,
                broker_reconciled=bool(reconciled),
                kill_active=False,
                incident_open=bool(control.get("incident") or control.get("uncertain")),
                device_healthy=True,
                risk_budget_available=True,
                signed_authority_verified=signed,
                signed_authority_expiry=expiry,
                authorized_fingerprints=fps,
            )
            tick = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            selection = propose_and_journal(tick, candidates, context, self.registry, self.policy)
            status.update(
                {
                    "mode": selection.mode,
                    "reason": selection.reason,
                    "registered_studies": self.registry.trial_count(),
                    "candidate_studies": len(candidates),
                    "selected_strategy": selection.strategy_id,
                    "score_r": selection.score_r,
                }
            )
            safe = bool(
                selection.mode == PAPER_ELIGIBLE
                and selection.strategy_id == self.strategy_id
                and selection.strategy_fingerprint in fps
            )
            return safe
        except Exception as exc:
            status["reason"] = "fail_closed:" + type(exc).__name__
            return False
        finally:
            self.last = status
            try:
                self._write_status(status)
            except OSError:
                pass


__all__ = ["AdaptivePaperGate"]

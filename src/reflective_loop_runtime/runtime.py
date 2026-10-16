"""Reflective Loop Runtime: the persistent, closed decision loop.

Tick outline
------------
1. Perception/telemetry  -> normalised sample (missing = None, never healthy).
2. Persistent RLR state   <- store + sample.
3. Deterministic candidates from state/capabilities/goals.
4. Optional LFM normalizer only for raw/unstructured evidence.
5. OpenDecider distribution (fast path; LFM/Qwen bypassed for structured state).
6. Confidence gate -> on uncertainty, optional Qwen advice/reframe.
7. OpenDecider final bounded re-decision (Qwen never the final selector).
8. Deterministic Governor (final authority; abstain always allowed).
9. Executor -> outcome -> persistence -> next tick.

Fail-closed boundaries: malformed model output, stale/missing telemetry, model
outages, non-finite confidence/probabilities, probability key mismatch,
probability sum errors, and confidence/probability mismatch all resolve to a
safe abstain (or an internal no-op). No exception from a model or adapter can
terminate the loop.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .candidates import Candidate, build_candidates, candidate_by_id, candidate_ids
from .config import Config
from .contracts import (
    ContractError,
    ScoredDecision,
    abstain_action,
    validate_scored_decision,
)
from .deciders import DeciderBackend, DeterministicOpenDecider
from .executor import Executor
from .governor import Governor
from .kernel.audit import AuditLog
from .kernel.budget import BudgetService
from .kernel.permit import PermitIssuer, PermitService
from .kernel.policy import (
    HAVE_CRYPTOGRAPHY,
    MODE_ENFORCE,
    MODE_OBSERVATION_ONLY,
    MissionPolicy,
    load_policy_bytes,
    sign_envelope,
    verify_envelope,
)
from .kernel.toolbroker import ToolBroker, broker_manifest
from .lfm import DeterministicLFM, NormalizedEvidence, looks_structured, validate_normalized
from .qwen import Advice
from .store import Store
from .telemetry import StaticTelemetry, TelemetryAdapter


@dataclass
class TickResult:
    tick: int
    thermal_level: str
    structured: bool
    used_lfm: bool
    used_qwen: bool
    low_confidence: bool
    lfm_error: str | None
    qwen_error: str | None
    qwen_degraded: bool
    first_confidence: float | None
    final_confidence: float | None
    decision: dict[str, Any]
    execution: dict[str, Any]
    vetoed: bool
    veto_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "thermal_level": self.thermal_level,
            "structured": self.structured,
            "used_lfm": self.used_lfm,
            "used_qwen": self.used_qwen,
            "low_confidence": self.low_confidence,
            "lfm_error": self.lfm_error,
            "qwen_error": self.qwen_error,
            "qwen_degraded": self.qwen_degraded,
            "first_confidence": self.first_confidence,
            "final_confidence": self.final_confidence,
            "decision": self.decision,
            "execution": self.execution,
            "vetoed": self.vetoed,
            "veto_reason": self.veto_reason,
        }


# ---------------------------------------------------------------------------
# Default Governor authority
# ---------------------------------------------------------------------------

# The standalone runtime holds its own operator authority: a mission policy that
# permits exactly the deterministic actions the loop can pick. It is signed by an
# operator key generated in-process (ed25519 when the optional dependency is
# available, otherwise the explicitly dev-only HMAC fallback) and never assembled
# from untrusted input, so the loop's effects still pass through the same
# permit-verified broker path as any other tool call. The declared budgets bound
# the loop's own spend: every priceable unit it can consume -- a control tick, a
# store write, a store read, a slow-brain token -- is capped here and enforced by
# the BudgetService this policy is bound to.
_RUNTIME_POLICY_BUNDLE: dict[str, Any] = {
    "mission_id": "reflective-loop-runtime",
    "objective": "run the reflective loop within its own deterministic authority",
    "invariants": ["no_effect_without_permit", "never_shell_out_unsandboxed"],
    "allowed_domains": [],
    "capabilities": [],
    "budgets": {
        "scheduler_tick": 1_000_000.0,
        "store_write": 1_000_000.0,
        "store_read": 1_000_000.0,
        "llm_tokens": 1_000_000.0,
    },
    "standing_mandates": [],
    "policy_version": "1.0.0",
}


CRYPTO_EXTRA_HINT = "pip install -e .[crypto]"


class PolicyEnforcementUnavailable(RuntimeError):
    """The deployment asked for ENFORCE but the runtime cannot enforce it.

    Refusing to start is the point: a deployment that believes it enforces a
    signed mission policy while actually running OBSERVATION_ONLY has no
    authority at all, and nothing at runtime would say so.
    """


def _policy_requests_enforce(config: Config) -> bool:
    """True when the deployment config asks for a signed, ENFORCE policy."""
    return bool(config.policy_path or config.policy_public_key_path)


def _load_configured_policy(config: Config) -> Any:
    """Load and verify the operator-signed policy envelope from the config."""
    if config.policy_path is None:
        raise PolicyEnforcementUnavailable(
            "policy_public_key_path is configured without a policy_path to verify"
        )
    try:
        data = Path(config.policy_path).read_bytes()
    except OSError as exc:
        raise PolicyEnforcementUnavailable(
            f"configured policy file could not be read: {exc}"
        ) from exc
    public_key = None
    if config.policy_public_key_path is not None:
        try:
            from cryptography.hazmat.primitives import serialization

            pem = Path(config.policy_public_key_path).read_bytes()
            public_key = serialization.load_pem_public_key(pem)
        except (OSError, ValueError, ImportError) as exc:
            raise PolicyEnforcementUnavailable(
                f"configured operator public key could not be loaded: {exc}"
            ) from exc
    policy = load_policy_bytes(
        data, public_key=public_key, source_path=str(config.policy_path)
    )
    if policy.mode != MODE_ENFORCE:
        raise PolicyEnforcementUnavailable(
            "configured policy did not verify as ENFORCE ("
            f"{policy.reason or 'signature could not be verified'})"
        )
    return policy


def _runtime_policy(config: Config | None = None) -> Any:
    """Build the mission policy the runtime governs itself with.

    ENFORCE requires an operator signature, so it also requires the optional
    ``cryptography`` backend. Three cases:

    * the deployment config asks for ENFORCE (a signed policy file and/or an
      operator public key): without ``cryptography`` this raises rather than
      quietly dropping to OBSERVATION_ONLY, because a runtime that cannot
      verify a signature has no authority to enforce;
    * no ENFORCE config and ``cryptography`` present: an ed25519 keypair is
      generated in-process and the bundle is enforced under it;
    * no ENFORCE config and no backend: the runtime refuses to borrow authority
      from a shared dev secret. The dev-HMAC bundle is built only to be reported
      and is deliberately re-emitted as ``OBSERVATION_ONLY``, so the default
      runtime can never mint permits on a dev secret -- it fails closed instead
      of enforcing with a forgeable one.
    """
    config = config or Config()
    if _policy_requests_enforce(config):
        if not HAVE_CRYPTOGRAPHY:
            raise PolicyEnforcementUnavailable(
                "policy mode ENFORCE is configured but the optional 'cryptography' "
                f"package is not importable; install it with `{CRYPTO_EXTRA_HINT}` "
                "rather than starting in OBSERVATION_ONLY"
            )
        return _load_configured_policy(config)
    if HAVE_CRYPTOGRAPHY:  # pragma: no cover - exercised whenever available
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        bundle = dict(_RUNTIME_POLICY_BUNDLE)
        private = Ed25519PrivateKey.generate()
        envelope = sign_envelope(bundle, private_key=private, key_id="runtime-operator")
        return load_policy_bytes(
            json.dumps(envelope).encode("utf-8"), public_key=private.public_key()
        )
    bundle = dict(_RUNTIME_POLICY_BUNDLE)
    secret = secrets.token_bytes(32)
    envelope = sign_envelope(bundle, secret=secret, key_id="runtime-dev")
    verification = verify_envelope(envelope, secret=secret, allow_dev_hmac=True)
    # ``from_bundle`` never grants authority, so the dev secret's verified-but-
    # shared signature cannot become ENFORCE here.
    return MissionPolicy.from_bundle(
        bundle,
        mode=MODE_OBSERVATION_ONLY,
        verification=verification,
        reason="dev-only HMAC signatures never carry operator authority",
    )


def policy_mode(config: Config | None = None) -> str:
    """The mode of the policy ``config`` would govern the runtime with.

    Side-effect free, so a deploy check can assert ENFORCE at startup without
    minting permits. Raises :class:`PolicyEnforcementUnavailable` when ENFORCE
    was asked for and cannot be honoured.
    """
    return _runtime_policy(config).mode


def _default_permits(config: Config) -> tuple[PermitService, BudgetService]:
    """A Governor-owned PermitService, budgeted, for a standalone runtime.

    The budget ledger is created from the same verified mission policy the
    permits are issued against, so every reservation the permit service makes is
    charged against that policy's caps rather than going unbounded.
    """
    policy = _runtime_policy(config)
    budget = BudgetService(config.data_dir / "budget.db", policy)
    return (
        PermitService(
            PermitIssuer(secrets.token_bytes(32)),
            policy,
            budget=budget,
            manifest=broker_manifest(),
            db_path=config.data_dir / "permits.db",
        ),
        budget,
    )


def _load_or_create_audit_key(data_dir: str | os.PathLike[str]) -> bytes:
    """Return the runtime's persistent audit-chain key, creating it 0600 once.

    The key must outlive the process so the hash chain stays verifiable across
    restarts and cannot be re-forged by anyone who can rewrite the database. It
    is never returned to, printed by, or reachable from model-facing code.
    """
    path = Path(data_dir) / "audit.key"
    try:
        data = path.read_bytes()
    except OSError:
        data = b""
    if len(data) == 32:
        return data
    key = secrets.token_bytes(32)
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


class ReflectiveLoopRuntime:
    def __init__(
        self,
        config: Config | None = None,
        *,
        decider: DeciderBackend | None = None,
        normalizer: Any | None = None,
        slow_brain: Any | None = None,
        telemetry: TelemetryAdapter | None = None,
        governor: Governor | None = None,
        executor: Executor | None = None,
        store: Store | None = None,
        permits: PermitService | None = None,
        clock: Any = time.monotonic,
    ):
        self.config = config or Config()
        self.config.ensure_dirs()
        self.clock = clock
        self.store = store or Store(self.config.db_path)
        self.decider = decider or DeterministicOpenDecider()
        self.normalizer = normalizer if normalizer is not None else DeterministicLFM()
        self.slow_brain = slow_brain  # None => Qwen disabled entirely
        self.telemetry = telemetry or StaticTelemetry()
        self.budget: BudgetService | None = None
        if permits is None and executor is None:
            permits, self.budget = _default_permits(self.config)
        self.permits = permits
        if governor is not None and getattr(governor, "permits", None) is None:
            # A caller-supplied Governor must share the runtime's authority, or
            # it could not mint the permits the executor requires.
            governor.permits = permits
        self.governor = governor or Governor(
            self.store, self.config, clock=clock, permits=permits
        )
        self.audit: AuditLog | None = None
        if executor is None:
            # The runtime's sole tool path must carry a persistent, keyed audit
            # chain: keyed so a tamperer cannot re-forge the chain, persistent so
            # it remains verifiable across restarts.
            self.audit = AuditLog(
                str(self.config.data_dir / "audit.db"),
                key=_load_or_create_audit_key(self.config.data_dir),
            )
            broker = ToolBroker(
                permits,
                audit=self.audit,
                workspace=self.config.data_dir / "workspace",
                manifest=permits.manifest,
                pids_cgroup=self.config.sandbox_pids_cgroup,
            )
            executor = Executor(self.store, broker=broker)
        self.executor = executor
        self._control_state: dict[str, Any] = dict(self.executor.control_state)

    # -- public ------------------------------------------------------------
    @property
    def run_id(self) -> str:
        return self.store.run_id

    def build_state(
        self,
        sample: dict[str, Any],
        *,
        capabilities: Any = None,
        structured: bool | None = None,
        raw_text: str = "",
        goal_digest: str | None = None,
    ) -> dict[str, Any]:
        """Assemble the persistent RLR state for a tick."""
        active_goals = self.store.active_goals()
        episodes = self.store.recent_episodes(self.config.episode_window)
        memory_count = self.store.counts()["memories"]
        thermal_level = self.governor.thermal_level(sample)
        cooldown_active = (
            int(self.executor.control_state.get("cooldown_until_tick", 0)) > self.store.current_tick
        )
        episode_digest = "; ".join(e["content"] for e in episodes[:3])[:500]
        state: dict[str, Any] = {
            "run_id": self.store.run_id,
            "tick": self.store.current_tick,
            "thermal_level": thermal_level,
            "temperature_c": sample.get("temperature_c"),
            "zones_c": sample.get("zones_c"),
            "ram_used_mib": sample.get("ram_used_mib"),
            "ram_total_mib": sample.get("ram_total_mib"),
            "sampled_at": sample.get("sampled_at"),
            "age_seconds": sample.get("age_seconds"),
            "capabilities": sorted(set(capabilities or ())),
            "active_goals": [{"id": g["id"], "text": g["text"]} for g in active_goals],
            "episode_count": self.store.counts()["episodes"],
            "memory_count": memory_count,
            "recent_episodes": [e["content"] for e in episodes[:3]],
            "episode_digest": episode_digest,
            "recent_actions": [a["name"] for a in self.store.recent_actions(limit=5)],
            "cooldown_active": cooldown_active,
            "goal_digest": goal_digest or (active_goals[0]["text"] if active_goals else "maintain stable operation"),
            "raw_text": raw_text,
            "control_state": dict(self.executor.control_state),
            "structured": structured,
        }
        if structured is None:
            state["structured"] = looks_structured(state)
        return state

    def tick(
        self,
        sample: dict[str, Any],
        *,
        capabilities: Any = None,
        structured: bool | None = None,
        raw_text: str = "",
        context: str = "",
        goal_digest: str | None = None,
        allow_default_goal_creation: bool = True,
    ) -> TickResult:
        """Run one full decision tick. Never raises on model/adapter failure."""
        context = self._bounded_context(context)
        self.store.current_tick += 1
        # The executor holds no Store handle, so the runtime publishes the tick
        # it counts for callers that invoke it without one.
        self.executor.set_tick(self.store.current_tick)
        state = self.build_state(
            sample, capabilities=capabilities, structured=structured,
            raw_text=raw_text, goal_digest=goal_digest,
        )
        thermal_level = state["thermal_level"]
        self.store.record_telemetry(sample)
        if state.get("structured"):
            self.store.add_episode("telemetry", self._sample_digest(sample), None)

        # --- LFM: only for raw/unstructured evidence --------------------
        used_lfm = False
        lfm_error: str | None = None
        normalized: NormalizedEvidence | None = None
        if not state.get("structured") and self.normalizer is not None:
            used_lfm = True
            text = raw_text or json.dumps(self._bounded_state_for_model(state), sort_keys=True)
            try:
                raw = self.normalizer.normalize(text, state, question=state.get("goal_digest", ""))
                normalized = validate_normalized(raw)
                state["normalized_evidence"] = normalized.to_dict()
                state["structured"] = True if normalized.structured else state.get("structured")
                state["ambiguity"] = normalized.ambiguity
                self.store.add_episode(
                    "normalized", normalized.summary or "normalized evidence", 1.0 - normalized.ambiguity
                )
            except Exception as exc:  # noqa: BLE001 - any normalizer failure fails closed
                lfm_error = f"{type(exc).__name__}: {exc}"[:500]
                # Fail closed: keep raw structured=False, continue to decision;
                # the Governor's staleness/uncertainty handling still applies.

        # --- OpenDecider: the fast, primary selector --------------------
        candidates = build_candidates(state, config=self.config)
        # A maintenance heartbeat is not a request to create the same goal
        # every few seconds. Remove it before the decider sees candidates,
        # rather than allowing repeated Governor vetoes and SQLite writes.
        if not allow_default_goal_creation:
            candidates = [c for c in candidates if c.action != "set_goal"]
        first, first_latency, first_error = self._score(
            self.decider, candidates, state, context, phase="opendecider"
        )
        if first is None:
            result = self._abstain_tick(
                state, candidates, thermal_level, structured=bool(state.get("structured")),
                used_lfm=used_lfm, lfm_error=lfm_error,
                reason=f"opendecider failed closed: {first_error}", context=context,
            )
            return result

        self.store.record_decision(
            phase="opendecider", engine=getattr(self.decider, "name", "opendecider"),
            choice=first.choice, confidence=first.confidence,
            probabilities=first.probabilities, mode="fast", latency_ms=first_latency,
        )

        # --- Confidence gate + optional Qwen advice/reframe -------------
        low_confidence = first.confidence < float(self.config.confidence_threshold)
        used_qwen = False
        qwen_error: str | None = None
        qwen_degraded = False
        advice: Advice | None = None
        final = first
        if low_confidence and self.slow_brain is not None:
            used_qwen = True
            try:
                advice = self.slow_brain.advise(
                    state=state, candidates=candidates, previous=first, context=context
                )
            except Exception as exc:  # noqa: BLE001 - slow-brain outage must not stop the loop
                advice = Advice(degraded=True, error=f"{type(exc).__name__}: {exc}"[:300])
            qwen_degraded = bool(getattr(advice, "degraded", False))
            qwen_error = getattr(advice, "error", None)
            reframed_context = self._merge_context(context, advice)
            second, second_latency, second_error = self._score(
                self.decider, candidates, state, reframed_context, phase="opendecider-final"
            )
            if second is not None:
                # OpenDecider makes a fresh bounded decision after advice.
                final = second
                self.store.record_decision(
                    phase="opendecider-final", engine=getattr(self.decider, "name", "opendecider"),
                    choice=second.choice, confidence=second.confidence,
                    probabilities=second.probabilities, mode="advised",
                    latency_ms=second_latency,
                )
            else:
                qwen_error = (qwen_error or "") + f"; re-decision failed: {second_error}"

        # --- Resolve candidate -> concrete action -----------------------
        chosen = candidate_by_id(candidates, final.choice)
        if chosen is None:
            action = abstain_action("selected candidate is not in the bounded set")
        else:
            action = chosen.to_action(rationale=action_rationale(final, advice))

        # --- Governor: final authority ----------------------------------
        veto = self.governor.veto(action, sample)
        veto_reason = veto.reason
        if veto.allowed:
            repeat = self.governor.repeat_veto(veto.action)
            if repeat:
                veto = type(veto)(False, veto.action, repeat, veto.thermal_level)
                veto_reason = repeat

        if not veto.allowed:
            self.store.record_veto(veto.action["action"], veto_reason or "vetoed")
            self.store.record_decision(
                phase="governor", engine="governor", choice=veto.action["action"],
                confidence=None, probabilities=None, mode="veto", latency_ms=0.0,
            )
            self.store.record_action(veto.action["action"], _action_args(veto.action), "vetoed")
            self.store.record_outcome(None, {"status": "vetoed", "reason": veto_reason})
            final_action = abstain_action(veto_reason or "vetoed")
            self.store.record_action(final_action["action"], _action_args(final_action), "abstained")
            outcome = self.executor.execute(final_action)
            self.store.record_outcome(None, outcome.__dict__)
            # ``execution`` reports what actually ran (the safe abstain), not the
            # vetoed proposal, so telemetry never claims a blocked action fired.
            execution = {"ok": outcome.ok, "action": outcome.action,
                         "applied": outcome.applied, "note": f"vetoed: {veto_reason}"}
            return TickResult(
                tick=self.store.current_tick, thermal_level=thermal_level,
                structured=bool(state.get("structured")), used_lfm=used_lfm, used_qwen=used_qwen,
                low_confidence=low_confidence, lfm_error=lfm_error, qwen_error=qwen_error,
                qwen_degraded=qwen_degraded, first_confidence=first.confidence,
                final_confidence=final.confidence,
                decision=_decision_dict(final, reframed=used_qwen),
                execution=execution, vetoed=True, veto_reason=veto_reason,
            )

        # --- Execute under Governor-issued authority, persist outcome ---
        action_id = self.store.record_action(veto.action["action"], _action_args(veto.action), "executed")
        # Every side effect needs a permit minted through the Governor. If none
        # can be issued the executor performs nothing and reports the refusal.
        permit = self.governor.authorize(
            veto.action,
            tick=self.store.current_tick,
            goal_id=str(state.get("goal_digest") or "runtime"),
            state_version=str(self.store.current_tick),
        )
        outcome = self.executor.execute(
            veto.action, permit=permit, tick=self.store.current_tick
        )
        self.store.record_outcome(action_id, outcome.__dict__)
        # A reflection is only a follow-up to an effect that actually landed:
        # when the reflect permit was refused or its effect failed, there is no
        # approved reflection to remember.
        if outcome.ok and veto.action["action"] == "reflect":
            self._do_reflection(state, context)
        self._control_state = dict(self.executor.control_state)

        return TickResult(
            tick=self.store.current_tick, thermal_level=thermal_level,
            structured=bool(state.get("structured")), used_lfm=used_lfm, used_qwen=used_qwen,
            low_confidence=low_confidence, lfm_error=lfm_error, qwen_error=qwen_error,
            qwen_degraded=qwen_degraded, first_confidence=first.confidence,
            final_confidence=final.confidence,
            decision=_decision_dict(final, reframed=used_qwen),
            execution={"ok": outcome.ok, "action": outcome.action,
                       "applied": outcome.applied, "note": outcome.note},
            vetoed=False, veto_reason=None,
        )

    # -- helpers -----------------------------------------------------------
    def _score(
        self,
        backend: Any,
        candidates: list[Candidate],
        state: dict[str, Any],
        context: str,
        *,
        phase: str,
    ) -> tuple[ScoredDecision | None, float, str | None]:
        ids = candidate_ids(candidates)
        started = time.perf_counter()
        try:
            raw = backend.score(candidates, state, context=context)
            scored = validate_scored_decision(raw, ids)
        except ContractError as exc:
            return None, 0.0, str(exc)[:500]
        except Exception as exc:  # noqa: BLE001 - any backend failure fails closed
            return None, 0.0, f"{type(exc).__name__}: {exc}"[:500]
        latency_ms = (time.perf_counter() - started) * 1000.0
        return scored, latency_ms, None

    def _abstain_tick(
        self,
        state: dict[str, Any],
        candidates: list[Candidate],
        thermal_level: str,
        *,
        structured: bool,
        used_lfm: bool,
        lfm_error: str | None,
        reason: str,
        context: str,
    ) -> TickResult:
        action = abstain_action(reason)
        self.store.record_decision(
            phase="opendecider", engine=getattr(self.decider, "name", "opendecider"),
            choice=None, confidence=None, probabilities=None, mode="fail-closed", latency_ms=0.0,
        )
        action_id = self.store.record_action(action["action"], _action_args(action), "abstained")
        outcome = self.executor.execute(action)
        self.store.record_outcome(action_id, outcome.__dict__)
        return TickResult(
            tick=self.store.current_tick, thermal_level=thermal_level, structured=structured,
            used_lfm=used_lfm, used_qwen=False, low_confidence=True, lfm_error=lfm_error,
            qwen_error=None, qwen_degraded=False, first_confidence=None, final_confidence=None,
            decision={"choice": None, "confidence": None, "probabilities": {}, "mode": "fail-closed"},
            execution={"ok": outcome.ok, "action": outcome.action, "note": outcome.note},
            vetoed=False, veto_reason=reason,
        )

    def _do_reflection(self, state: dict[str, Any], context: str) -> None:
        if self.slow_brain is None:
            return
        try:
            summary = self.slow_brain.reflect(state=state, context=self._bounded_context(context))
        except Exception:  # noqa: BLE001 - reflection failure must never stop the loop
            return
        if not summary or not str(summary).strip():
            return
        # A reflection is model output, so it is not written to memory directly.
        # It must pass the same Governor -> permit -> broker gate as any other
        # memory write; if no permit can be minted (observation-only authority,
        # budget exhausted) the reflection is simply discarded.
        action = {
            "action": "remember",
            "rationale": "reflection",
            "text": str(summary)[:4000],
            "importance": 3,
        }
        permit = self.governor.authorize(
            action,
            tick=self.store.current_tick,
            goal_id=str(state.get("goal_digest") or "reflection"),
            state_version=str(self.store.current_tick),
        )
        if permit is None:
            return
        self.executor.execute(action, permit=permit, tick=self.store.current_tick)

    @staticmethod
    def _sample_digest(sample: dict[str, Any]) -> str:
        temp = sample.get("temperature_c")
        used = sample.get("ram_used_mib")
        total = sample.get("ram_total_mib")
        return f"temp={temp} ram={used}/{total}"[:1000]

    def _bounded_state_for_model(self, state: dict[str, Any]) -> dict[str, Any]:
        keep = (
            "thermal_level", "temperature_c", "ram_used_mib", "ram_total_mib",
            "capabilities", "active_goals", "episode_count", "memory_count",
            "recent_actions", "cooldown_active",
        )
        return {key: state.get(key) for key in keep}

    def _merge_context(self, context: str, advice: Advice | None) -> str:
        if advice is None:
            return context
        parts = [context]
        if advice.summary:
            parts.append(f"advisor-summary: {advice.summary}")
        if advice.rationale:
            parts.append(f"advisor-rationale: {advice.rationale}")
        if advice.suggested_candidate:
            parts.append(f"advisor-suggestion: {advice.suggested_candidate}")
        if advice.focus:
            parts.append("advisor-focus: " + "; ".join(advice.focus))
        return self._bounded_context("\n".join(p for p in parts if p))

    def _bounded_context(self, context: str) -> str:
        limit = int(self.config.max_context_chars)
        context = context or ""
        return context[:limit]


def _action_args(action: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in action.items() if k not in ("action", "rationale")}


def _decision_dict(scored: ScoredDecision, *, reframed: bool) -> dict[str, Any]:
    return {
        "choice": scored.choice,
        "confidence": scored.confidence,
        "probabilities": dict(scored.probabilities),
        "mode": "advised" if reframed else "fast",
        "advice": scored.advice,
    }


def action_rationale(scored: ScoredDecision, advice: Advice | None) -> str:
    base = scored.advice or "OpenDecider selection"
    if advice and not getattr(advice, "degraded", False) and advice.summary:
        base = f"{base} | advisor: {advice.summary}"
    return base[:1000]

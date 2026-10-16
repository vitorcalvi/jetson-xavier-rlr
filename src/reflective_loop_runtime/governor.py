"""Deterministic Governor: the final authority over every side-effecting action.

The Governor is independent of all models. It may veto any action, but it must
always allow ``abstain``. It enforces three things:

1. Fail-safe on stale/missing telemetry (the loop stays alive by abstaining).
2. Thermal/resource governance via conserve/degraded/cooldown bands.
3. Whitelist and argument-bound checks (defence in depth behind the contracts).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import ACTION_FIELDS, SIDE_EFFECTING_ACTIONS, ContractError, validate_action
from .executor import effect_request
from .kernel.budget import BudgetError
from .kernel.permit import PermitError


@dataclass
class VetoDecision:
    allowed: bool
    action: dict[str, Any]
    reason: str | None = None
    thermal_level: str = "unknown"


class Governor:
    def __init__(
        self,
        store: Any,
        config: Any,
        *,
        clock: Any = time.monotonic,
        permits: Any = None,
    ):
        self.store = store
        self.config = config
        self.clock = clock
        self.permits = permits
        self._last_sequence: dict[str, int] = {}

    # -- authority ---------------------------------------------------------
    def authorize(
        self,
        action: Any,
        *,
        tick: int = 0,
        goal_id: str = "",
        state_version: str = "",
    ) -> Any:
        """Mint a single-use permit for an approved action, or ``None``.

        The permit is bound to the exact ``(tool, op, args)`` call the executor
        will make, so a Governor approval can never authorise a different
        effect. Without a permit service, or when issuance is denied (an
        unenforced policy, a budget cap, an unknown call), no authority is
        granted and the executor performs nothing.
        """
        permits = self.permits
        if permits is None:
            return None
        request = effect_request(action, tick)
        if request is None:
            return None
        tool, op, args = request
        try:
            return permits.issue(
                tool=tool,
                op=op,
                args=args,
                goal_id=goal_id,
                state_version=state_version,
            )
        except (PermitError, BudgetError):
            return None

    # -- telemetry safety --------------------------------------------------
    def thermal_level(self, sample: dict[str, Any]) -> str:
        if sample is None:
            return "unknown"
        temperature = self._max_temp(sample)
        if temperature is None:
            return "unknown"
        if temperature >= self.config.cooldown_c:
            return "cooldown"
        if temperature >= self.config.degraded_c:
            return "degraded"
        if temperature >= self.config.conserve_c:
            return "conserve"
        return "normal"

    def stale_reason(self, sample: dict[str, Any], now: float | None = None) -> str | None:
        """Return a reason string when telemetry is missing or stale, else None.

        Freshness is judged at decision time from the *source* acquisition time
        (``acquired_at``), never from the moment the sample happened to reach the
        caller: a cached reading returned late is still old. A sample whose
        per-source ``sequence`` is not newer than the last accepted one is a
        replay of an older reading and is rejected too.
        """
        if not sample:
            return "telemetry is unavailable"
        now = self.clock() if now is None else now
        temperature = self._max_temp(sample)
        total = sample.get("ram_total_mib")
        if temperature is None or total in (None, 0):
            return "temperature or RAM telemetry is unavailable"
        source = sample.get("source")
        age = self._sample_age(sample, now)
        if age is None:
            return "telemetry acquisition time is unavailable"
        if age > self._max_age_seconds(source):
            where = f" from {source!r}" if source else ""
            return f"telemetry{where} is stale ({int(age)}s old)"
        replay = self._sequence_reason(sample)
        if replay is not None:
            return replay
        return None

    # -- veto --------------------------------------------------------------
    def veto(self, action: Any, sample: dict[str, Any]) -> VetoDecision:
        try:
            action = validate_action(action)
        except ContractError as exc:
            # A malformed action is not a valid abstain; convert to one.
            return VetoDecision(
                allowed=False,
                action={"action": "abstain", "rationale": "invalid action vetoed",
                        "reason": str(exc)[:1000]},
                reason=f"invalid action: {exc}",
                thermal_level=self.thermal_level(sample),
            )

        level = self.thermal_level(sample)
        name = action["action"]

        # Abstain is unconditionally allowed.
        if name == "abstain":
            return VetoDecision(True, action, None, level)

        # Fail safe on stale/missing telemetry. Side-effecting actions require
        # fresh telemetry; non-side-effecting internal actions are still allowed.
        staleness = self.stale_reason(sample)
        if staleness and name in SIDE_EFFECTING_ACTIONS:
            return VetoDecision(False, action, staleness, level)

        # Thermal caps.
        if name == "set_vision_fps":
            cap = self._fps_cap(level)
            if action["fps"] > cap:
                return VetoDecision(
                    False, action,
                    f"requested {action['fps']:.2f} FPS exceeds governor cap {cap:.2f} FPS "
                    f"at thermal level {level}", level,
                )
        if name == "set_reply_tokens":
            cap = self._tokens_cap(level)
            if action["tokens"] > cap:
                return VetoDecision(
                    False, action,
                    f"requested {action['tokens']} tokens exceeds governor cap {cap} "
                    f"at thermal level {level}", level,
                )
        if name == "set_load_target":
            if level in ("degraded", "cooldown") and action["load"] > 0.1:
                return VetoDecision(
                    False, action,
                    f"load target {action['load']:.2f} not allowed at thermal level {level}", level,
                )

        if name not in ACTION_FIELDS:
            return VetoDecision(False, action, "action is not whitelisted", level)
        return VetoDecision(True, action, None, level)

    def repeat_veto(self, action: dict[str, Any], window: int | None = None) -> str | None:
        """Block an identical side-effecting action repeated within a short window.

        Prevents an unproductive controller loop from consuming adjacent ticks.
        """
        name = action.get("action")
        if name not in SIDE_EFFECTING_ACTIONS:
            return None
        window = window or int(getattr(self.config, "governor_max_repeat_ticks", 3))
        target = {k: v for k, v in action.items() if k not in ("action", "rationale")}
        for prior in self.store.recent_actions(limit=window):
            if prior["name"] != name:
                continue
            prior_args = dict(prior["args"])
            prior_args.pop("rationale", None)  # stored args exclude the name already
            if prior_args == target:
                return f"same action and arguments were already used within {window} ticks"
        return None

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _sample_age(sample: dict[str, Any], now: float) -> float | None:
        """Age in seconds, preferring the source acquisition timestamp."""
        explicit = sample.get("age_seconds")
        if explicit is not None:
            try:
                return float(explicit)
            except (TypeError, ValueError):
                return None
        for key in ("acquired_at", "sampled_monotonic"):
            captured = sample.get(key)
            if captured is None:
                continue
            try:
                return max(0.0, now - float(captured))
            except (TypeError, ValueError):
                continue
        return None

    def _max_age_seconds(self, source: Any) -> float:
        """Max allowed sample age, optionally overridden per source."""
        default = float(self.config.stale_seconds)
        overrides = getattr(self.config, "stale_seconds_by_source", None)
        if overrides is None:
            extra = getattr(self.config, "extra", None)
            if isinstance(extra, Mapping):
                overrides = extra.get("stale_seconds_by_source")
        if isinstance(overrides, Mapping) and isinstance(source, str):
            value = overrides.get(source)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default
        return default

    def _sequence_reason(self, sample: dict[str, Any]) -> str | None:
        """Reject a sample that is not newer than the last accepted one."""
        sequence = sample.get("sequence")
        if sequence is None:
            return None
        try:
            sequence = int(sequence)
        except (TypeError, ValueError):
            return "telemetry sequence is invalid"
        source = sample.get("source") or "unknown"
        previous = self._last_sequence.get(source)
        if previous is not None and sequence <= previous:
            return (
                f"telemetry sample from {source!r} is not newer than the last "
                f"accepted read (sequence {sequence} <= {previous})"
            )
        self._last_sequence[source] = sequence
        return None

    @staticmethod
    def _max_temp(sample: dict[str, Any] | None) -> float | None:
        if not sample:
            return None
        zones = sample.get("zones_c") or []
        values = [z for z in zones if z is not None]
        temperature = sample.get("temperature_c")
        if temperature is not None:
            values.append(float(temperature))
        if not values:
            return None
        return max(float(v) for v in values)

    def _fps_cap(self, level: str) -> float:
        if level == "cooldown":
            return 0.0
        if level == "degraded":
            return float(self.config.max_vision_fps_degraded)
        if level == "conserve":
            return float(self.config.max_vision_fps_conserve)
        return 5.0

    def _tokens_cap(self, level: str) -> int:
        if level == "cooldown":
            return 0
        if level == "degraded":
            return int(self.config.max_reply_tokens_degraded)
        if level == "conserve":
            return int(self.config.max_reply_tokens_conserve)
        return 512

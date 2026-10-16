"""Strict action contracts, bounded action space, and scored-decision validation.

Everything a model (OpenDecider, LFM, or Qwen) produces crosses this boundary
before it can influence the runtime. The rules here are deliberately small and
deterministic so that the *same* checks run in the fast path and the slow path.

Design invariants
-----------------
* Only whitelisted action names and argument shapes are accepted.
* There are no shell / arbitrary-file-write / arbitrary-network / clock-power
  mutation actions. The action space is closed and internal-only.
* Model output is treated as untrusted: any malformed, non-finite, mismatched,
  or out-of-range value fails closed by raising ``ContractError``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# Probability sums and confidence/probability agreement tolerances.
PROB_SUM_TOLERANCE = 0.02
CONFIDENCE_TOLERANCE = 0.02

# Bounds for the only numeric arguments any action may carry.
FPS_MIN, FPS_MAX = 0.2, 5.0
TOKENS_MIN, TOKENS_MAX = 16, 512
SECONDS_MIN, SECONDS_MAX = 1, 3600
LOAD_MIN, LOAD_MAX = 0.0, 1.0

MAX_RATIONALE_CHARS = 1000
MAX_TEXT_CHARS = 4000
MAX_QUERY_CHARS = 1000
MAX_GOAL_CHARS = 1000


class ContractError(ValueError):
    """Raised whenever untrusted model or action data violates a contract."""


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

# Action name -> required argument names. The ``abstain`` action is always
# available and always valid, which keeps the safe fallback reachable.
ACTION_FIELDS: dict[str, set[str]] = {
    "observe": set(),
    "abstain": {"reason"},
    "set_load_target": {"load"},
    "set_reply_tokens": {"tokens"},
    "set_vision_fps": {"fps"},
    "cooldown": {"seconds"},
    "remember": {"text", "importance"},
    "recall": {"query"},
    "set_goal": {"text"},
    "complete_goal": {"goal_id"},
    "reflect": set(),
}

# A coarse classification used by the Governor and executor. Side-effecting
# actions change runtime behaviour and therefore require a clean veto check.
SIDE_EFFECTING_ACTIONS = frozenset(
    {
        "set_load_target",
        "set_reply_tokens",
        "set_vision_fps",
        "cooldown",
        "remember",
        "set_goal",
        "complete_goal",
    }
)


def _check_int(value: Any, low: int, high: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ContractError(f"{name} must be between {low} and {high}")
    return value


def _check_number(value: Any, low: float, high: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ContractError(f"{name} must be finite")
    if not low <= value <= high:
        raise ContractError(f"{name} must be between {low} and {high}")
    return value


def _check_text(value: Any, limit: int, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be non-empty text")
    if len(value) > limit:
        raise ContractError(f"{name} must be at most {limit} characters")
    return value


def validate_action(value: Any) -> dict[str, Any]:
    """Validate a single action dict against the whitelist and argument shapes.

    Accepts a JSON string or a mapping. Returns a normalised copy. Raises
    ``ContractError`` for anything outside the closed action space.
    """
    if isinstance(value, (str, bytes)):
        try:
            value = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ContractError("action is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ContractError("action must be a JSON object")

    name = value.get("action")
    if not isinstance(name, str) or name not in ACTION_FIELDS:
        raise ContractError(f"action {name!r} is not in the whitelist")

    expected = ACTION_FIELDS[name] | {"action", "rationale"}
    if set(value) != expected:
        missing = expected - set(value)
        extra = set(value) - expected
        raise ContractError(
            f"action fields do not match schema (missing={sorted(missing)}, "
            f"extra={sorted(extra)})"
        )

    _check_text(value.get("rationale"), MAX_RATIONALE_CHARS, "rationale")
    action: dict[str, Any] = {"action": name, "rationale": value["rationale"].strip()}

    if name == "abstain":
        action["reason"] = _check_text(value.get("reason"), MAX_RATIONALE_CHARS, "reason")
    elif name == "set_load_target":
        action["load"] = _check_number(value.get("load"), LOAD_MIN, LOAD_MAX, "load")
    elif name == "set_reply_tokens":
        action["tokens"] = _check_int(value.get("tokens"), TOKENS_MIN, TOKENS_MAX, "tokens")
    elif name == "set_vision_fps":
        action["fps"] = _check_number(value.get("fps"), FPS_MIN, FPS_MAX, "fps")
    elif name == "cooldown":
        action["seconds"] = _check_int(value.get("seconds"), SECONDS_MIN, SECONDS_MAX, "seconds")
    elif name == "remember":
        action["text"] = _check_text(value.get("text"), MAX_TEXT_CHARS, "text")
        action["importance"] = _check_int(value.get("importance"), 1, 5, "importance")
    elif name == "recall":
        action["query"] = _check_text(value.get("query"), MAX_QUERY_CHARS, "query")
    elif name == "set_goal":
        action["text"] = _check_text(value.get("text"), MAX_GOAL_CHARS, "text")
    elif name == "complete_goal":
        action["goal_id"] = _check_int(value.get("goal_id"), 1, 2**31 - 1, "goal_id")
    return action


# ---------------------------------------------------------------------------
# Scored decisions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredDecision:
    """A model's probability distribution over a bounded candidate set.

    ``probabilities`` maps candidate id -> probability. ``choice`` must be one
    of those ids and ``confidence`` must agree with ``probabilities[choice]``.
    """

    choice: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    advice: str | None = None


def validate_scored_decision(
    decision: Any,
    candidate_ids: Any,
    *,
    sum_tolerance: float = PROB_SUM_TOLERANCE,
    confidence_tolerance: float = CONFIDENCE_TOLERANCE,
) -> ScoredDecision:
    """Fail closed unless a scored decision matches the candidate contract.

    Checks, in order: type/shape, key mismatch, finite values, [0, 1] range,
    probability sum, choice membership, and confidence/probability agreement.
    """
    ids = set(candidate_ids)
    if not ids:
        raise ContractError("candidate set is empty")

    if isinstance(decision, ScoredDecision):
        choice = decision.choice
        confidence = decision.confidence
        probabilities = dict(decision.probabilities)
        advice = decision.advice
    elif isinstance(decision, Mapping):
        if "probabilities" not in decision:
            raise ContractError("decision is missing probabilities")
        choice = decision.get("choice")
        confidence = decision.get("confidence")
        probabilities = decision.get("probabilities")
        advice = decision.get("advice")
    else:
        raise ContractError("decision must be a scoring object")

    if not isinstance(probabilities, Mapping):
        raise ContractError("probabilities must be a mapping")
    prob_keys = set(probabilities)
    if prob_keys != ids:
        raise ContractError(
            f"probability keys must exactly match candidates "
            f"(missing={sorted(ids - prob_keys)}, extra={sorted(prob_keys - ids)})"
        )

    clean: dict[str, float] = {}
    for key, raw in probabilities.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ContractError(f"probability for {key!r} must be a number")
        value = float(raw)
        if not math.isfinite(value):
            raise ContractError(f"probability for {key!r} must be finite")
        if not 0.0 <= value <= 1.0:
            raise ContractError(f"probability for {key!r} must be within [0, 1]")
        clean[key] = value

    total = math.fsum(clean.values())
    if abs(total - 1.0) > sum_tolerance:
        raise ContractError(f"probabilities must sum to 1 +/- {sum_tolerance}; got {total}")

    if not isinstance(choice, str) or choice not in ids:
        raise ContractError(f"choice {choice!r} is not a candidate")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ContractError("confidence must be a number")
    confidence = float(confidence)
    if not math.isfinite(confidence):
        raise ContractError("confidence must be finite")
    if not 0.0 <= confidence <= 1.0:
        raise ContractError("confidence must be within [0, 1]")
    if abs(confidence - clean[choice]) > confidence_tolerance:
        raise ContractError("confidence must agree with the selected probability")
    if advice is not None and not isinstance(advice, str):
        raise ContractError("advice must be text when present")

    return ScoredDecision(choice=choice, confidence=confidence, probabilities=clean, advice=advice)


def abstain_action(reason: str, rationale: str = "fail closed") -> dict[str, Any]:
    """Build a valid ``abstain`` action; the safe fallback for any failure."""
    reason = (reason or "unspecified").strip()[:MAX_RATIONALE_CHARS]
    return {
        "action": "abstain",
        "rationale": (rationale or "fail closed")[:MAX_RATIONALE_CHARS],
        "reason": reason or "unspecified",
    }

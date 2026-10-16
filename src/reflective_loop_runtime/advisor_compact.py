"""64-token, grammar-bounded local AI advice only; cannot execute trades."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from reflective_loop_runtime.qwen import Advice, _parse_json

_ALLOWED_TEXT = r'text ::= "\"" [a-zA-Z0-9 _-]{1,12} "\""'


def _term(s: str) -> str:
    return json.dumps(s)


def compact_grammar(ids: list[str]) -> str:
    """Enumerate exactly the current bounded choices, plus literal JSON null."""
    vals = ['"null"'] + [_term(json.dumps(s)) for s in ids]
    return (
        "root ::= obj\nchoice ::= "
        + " | ".join(vals)
        + "\n"
        + r'obj ::= "{" "\"rationale\"" ":" text "," "\"suggested_candidate\"" ":" choice "}"'
        + "\n"
        + _ALLOWED_TEXT
        + "\n"
    )


_REFLECTION = (
    'root ::= obj\nobj ::= "{" ' + _term('"summary"') + ' ":" text "}"\n' + _ALLOWED_TEXT + "\n"
)


@dataclass
class CompactQwenAdvisor:
    client: Any
    name: str = "qwen35-compact-json-advisor"

    def advise(
        self, *, state: dict[str, Any], candidates: list[Any], previous: Any, context: str = ""
    ) -> Advice:
        ids = [str(getattr(c, "id", c)) for c in candidates]
        if not ids:
            return Advice(degraded=True, error="no_candidates")
        # The model can only name an already existing candidate. It is never
        # passed broker credentials, orders, signed scope, or permissions.
        ids = ids[:12]
        state_text = {
            k: state.get(k)
            for k in ("thermal_level", "temperature_c", "ram_used_mib")
            if k in state
        }
        prompt = json.dumps(
            {
                "options": ids,
                "state": state_text,
                "previous": getattr(previous, "choice", None),
                "context": context[:100],
            },
            separators=(",", ":"),
        )
        try:
            reply = self.client.chat(
                "Advisory only. Choose a listed option or null. Short rationale."
                " Output only JSON; no thinking.",
                prompt,
                grammar=compact_grammar(ids),
                stop=["}"],
            )
            obj = _parse_json(reply)
            if not isinstance(obj, dict):
                raise ValueError("invalid JSON")
            reason = str(obj.get("rationale") or "")[:12]
            proposed = obj.get("suggested_candidate")
            if not reason or proposed not in ids and proposed is not None:
                raise ValueError("invalid bounded advice")
            return Advice(rationale=reason, suggested_candidate=proposed)
        except Exception as exc:
            return Advice(
                degraded=True,
                error=type(exc).__name__,
                suggested_candidate="abstain" if "abstain" in ids else None,
            )

    def reflect(self, *, state: dict[str, Any], context: str = "") -> str:
        payload = json.dumps(
            {"thermal": state.get("thermal_level"), "context": context[:120]}, separators=(",", ":")
        )
        try:
            reply = self.client.chat(
                "Give a short factual observation as JSON only. No actions.",
                payload,
                grammar=_REFLECTION,
                stop=["}"],
            )
            obj = _parse_json(reply)
            if isinstance(obj, dict):
                return str(obj.get("summary") or "")[:24]
        except Exception:
            pass
        return ""

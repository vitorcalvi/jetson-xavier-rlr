"""Bounded local-to-LAN escalation for advisory reasoning.

The runtime deterministic selector remains first. This module only composes
slow-brain advisors that are invoked after a low-confidence bounded decision.
Neither local nor LAN models can add candidates, tools, or capabilities; their
output is validated by QwenSlowBrain before it can influence re-scoring.
"""

from __future__ import annotations

import ipaddress
import math
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .qwen import Advice, OpenAICompatClient, QwenSlowBrain


class LiteLLMCompatClient(OpenAICompatClient):
    """OpenAI-compatible LAN client without llama.cpp-only request fields.

    The local llama-server supports GBNF through a top-level grammar field.
    A generic LiteLLM route may reject that non-OpenAI parameter, so LAN calls
    deliberately omit it. QwenSlowBrain still parses JSON and applies the exact
    candidate-id allowlist after the response, preserving the security bound.
    """

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        grammar: str | None = None,
        stop: list[str] | None = None,
    ) -> str:
        del grammar
        kwargs: dict[str, Any] = {"temperature": temperature, "grammar": None}
        if stop is not None:
            kwargs["stop"] = stop
        return super().chat(system, user, **kwargs)


def is_private_lan_url(base_url: str, *, allowed_hosts: tuple[str, ...] = ()) -> bool:
    """Return True only for an explicitly allowed host or a numeric private/loopback IP."""
    parsed = urlparse((base_url or "").strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    host = parsed.hostname.rstrip(".").lower()
    allowed = {str(item).rstrip(".").lower() for item in allowed_hosts}
    if host in allowed:
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    if addr.is_loopback:
        return True
    if not isinstance(addr, ipaddress.IPv4Address):
        return False
    return any(
        addr in network
        for network in (
            ipaddress.ip_network("10.0.0.0/8"),
            ipaddress.ip_network("172.16.0.0/12"),
            ipaddress.ip_network("192.168.0.0/16"),
        )
    )


def build_lan_slow_brain(
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
    max_tokens: int,
    allowed_hosts: tuple[str, ...] = (),
) -> QwenSlowBrain:
    """Build a LiteLLM-backed advisor without exposing its key.

    Remote advisory is opt-in at the caller. The URL must be a numeric
    private/loopback IP or an exact operator-allowlisted hostname.
    """
    if not api_key:
        raise ValueError("LAN advisor API key is required")
    if not is_private_lan_url(base_url, allowed_hosts=allowed_hosts):
        raise ValueError("LAN advisor endpoint must be private/loopback or explicitly allowlisted")
    client = LiteLLMCompatClient(
        base_url=base_url,
        model=model,
        timeout=max(0.1, float(timeout)),
        max_tokens=max(1, int(max_tokens)),
        allow_remote=True,
        api_key=api_key,
    )
    return QwenSlowBrain(client, name="lan-litellm-advisor")


def _confidence(previous: Any) -> float | None:
    value = getattr(previous, "confidence", None)
    if value is None and isinstance(previous, dict):
        value = previous.get("confidence")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


@dataclass
class TieredSlowBrain:
    """Local advisor first; escalate only hard/failed cases to LAN LiteLLM.

    A hard case is one whose prior bounded decision confidence is at or below
    lan_threshold. If LAN advice fails or is malformed, local advice is returned.
    If both tiers fail, degraded bounded advice propagates and no exception is
    allowed to stall the runtime.
    """

    local: Any
    lan: Any | None = None
    lan_enabled: bool = False
    lan_threshold: float = 0.55
    name: str = "tiered-local-lan-advisor"

    def advise(
        self,
        *,
        state: dict[str, Any],
        candidates: list[Any],
        previous: Any,
        context: str = "",
    ) -> Advice:
        try:
            local_advice = self.local.advise(
                state=state, candidates=candidates, previous=previous, context=context
            )
        except Exception as exc:
            local_advice = Advice(degraded=True, error=type(exc).__name__)

        confidence = _confidence(previous)
        try:
            threshold = float(self.lan_threshold)
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError("invalid threshold")
        except (TypeError, ValueError):
            threshold = 0.55
        hard = confidence is not None and confidence <= threshold
        should_escalate = bool(
            self.lan_enabled and self.lan is not None and (hard or local_advice.degraded)
        )
        if not should_escalate:
            return local_advice

        try:
            lan_advice = self.lan.advise(
                state=state, candidates=candidates, previous=previous, context=context
            )
        except Exception as exc:
            lan_advice = Advice(degraded=True, error=type(exc).__name__)

        if not lan_advice.degraded:
            return lan_advice
        return local_advice

    def reflect(self, *, state: dict[str, Any], context: str = "") -> str:
        try:
            summary = self.local.reflect(state=state, context=context)
        except Exception:
            summary = ""
        if summary or not self.lan_enabled or self.lan is None:
            return str(summary or "")
        try:
            return str(self.lan.reflect(state=state, context=context) or "")
        except Exception:
            return ""

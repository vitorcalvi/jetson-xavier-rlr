#!/usr/bin/env python3
"""GEPA-compatible offline evaluator for Jetson Growth OS.

No optimizer or paid model is invoked by this file. It generates actionable
diagnostics and an immutable offline score for a prompt/creative candidate.
This is NOT statistically valid campaign-conversion optimization.
"""
import json
import re
import urllib.parse

CLAIMS_DENY = (
    "vendas garantidas", "renda garantida", "grátis para sempre",
    "100% automático", "100% garantido", "100% free", "zero risco",
)
APPROVED_DOMAINS = {"ava.dyagnosys.com", "getmythoughs.dyagnosys.com"}
CHANNELS = {"facebook", "instagram", "bluesky", "threads", "tiktok",
            "youtube", "pinterest", "reddit", "linkedin"}
PRODUCTS = {"ava": "ava.dyagnosys.com", "mythoughs": "getmythoughs.dyagnosys.com"}


def evaluate_offline(candidate):
    """Return {score, accepted, diagnostics}, never publish or promote."""
    errors = []
    observations = []
    if not isinstance(candidate, dict):
        return {"score": 0.0, "accepted": False, "diagnostics": ["invalid_candidate"]}
    content = candidate.get("content")
    product = candidate.get("product")
    channel = candidate.get("channel")
    if not isinstance(content, str) or not 30 <= len(content) <= 2800:
        errors.append("content_length_out_of_bounds")
        content = ""
    if product not in PRODUCTS:
        errors.append("unknown_product")
    if channel not in CHANNELS:
        errors.append("unknown_channel")
    lowered = content.casefold()
    for phrase in CLAIMS_DENY:
        if phrase in lowered:
            errors.append("unsupported_claim:" + phrase)
    if content.count("http") > 2:
        errors.append("too_many_links")
    links = re.findall(r"https?://[^\s)>\]]+", content)
    if len(links) != 1:
        errors.append("requires_one_product_cta")
    else:
        try:
            url = urllib.parse.urlparse(links[0])
            params = urllib.parse.parse_qs(url.query)
            if url.scheme != "https" or url.hostname not in APPROVED_DOMAINS:
                errors.append("invalid_cta_destination")
            if product in PRODUCTS and url.hostname != PRODUCTS[product]:
                errors.append("cta_product_mismatch")
            if params.get("utm_source") != [channel]:
                errors.append("utm_source_mismatch")
            if not params.get("utm_campaign") and not params.get("campaign_id"):
                errors.append("missing_campaign")
            if not params.get("utm_content") and not params.get("variant_id"):
                errors.append("missing_variant")
        except ValueError:
            errors.append("invalid_cta")
    if len(content) >= 80:
        observations.append("substantive_content")
    else:
        observations.append("short_content")
    if len(content) <= 350:
        observations.append("concise_copy")
    if "?" in content:
        observations.append("invites_discussion")
    if product == "mythoughs":
        errors.append("mythoughs_no_production_release")
    accepted = not errors
    score = round(min(1.0, max(0.0, 0.9 - .16 * len(errors) +
                      .05 * ("concise_copy" in observations) +
                      .05 * ("invites_discussion" in observations))), 3) if accepted else 0.0
    return {"score": score, "accepted": accepted,
            "diagnostics": errors + observations,
            "optimization_tier": "OFFLINE_QA_ONLY",
            "public_promotion_allowed": False}


def online_promotion_gate(evidence):
    """Fail closed with honest INCONCLUSIVE for absent/immature samples."""
    if not isinstance(evidence, dict):
        return {"allowed": False, "reason": "INVALID_EVIDENCE"}
    if evidence.get("synthetic") or evidence.get("live_attribution_verified") is not True:
        return {"allowed": False, "reason": "ATTRIBUTION_UNVERIFIED"}
    if evidence.get("experiment_pre_registered") is not True:
        return {"allowed": False, "reason": "EXPERIMENT_NOT_REGISTERED"}
    if evidence.get("cohorts_mature") is not True:
        return {"allowed": False, "reason": "COHORTS_IMMATURE"}
    if evidence.get("power_sufficient") is not True:
        return {"allowed": False, "reason": "INCONCLUSIVE"}
    if evidence.get("product_health_ok") is not True:
        return {"allowed": False, "reason": "PRODUCT_HEALTH_BLOCK"}
    if evidence.get("manual_promotion_approved") is not True:
        return {"allowed": False, "reason": "REQUIRES_APPROVAL"}
    # Deployment/publisher governor is a separate authority, not this evaluator.
    return {"allowed": True, "reason": "EVIDENCE_ACCEPTED_FOR_MANUAL_REVIEW"}


def run_with_gepa(seed_candidate, trainset, valset, reflection_lm, max_metric_calls,
                  authorize_paid_calls=False):
    """Optional GEPA adapter; only Linux Python 3.10+, never in unattended zero-spend mode."""
    if authorize_paid_calls is not True:
        raise PermissionError("GEPA_CALLS_DISABLED_BY_DEFAULT")
    if not reflection_lm or not isinstance(max_metric_calls, int) or not 1 <= max_metric_calls <= 100:
        raise ValueError("invalid_gepa_budget_or_reflection_model")
    from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig
    # This seed can only improve copy quality. It cannot authorize public posting.
    def evaluator(candidate):
        value = candidate if isinstance(candidate, dict) else {"content": str(candidate),
                        "product": seed_candidate.get("product"), "channel": seed_candidate.get("channel")}
        return evaluate_offline(value)["score"]
    return optimize_anything(
        seed_candidate=seed_candidate,
        evaluator=evaluator,
        objective="Improve accurate helpful product content without risky claims or publishing",
        config=GEPAConfig(engine=EngineConfig(max_metric_calls=max_metric_calls))
    )


if __name__ == "__main__":
    import sys
    item = json.load(sys.stdin)
    print(json.dumps(evaluate_offline(item), sort_keys=True))

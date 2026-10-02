"""Deterministic KAPRO scoring for the KAware-RLR adaptation.

Deliberately free of any model or device dependency so the metrics can be unit
tested. Acc_know, Acc_act, KAS, tool overuse and tool underuse are computed here
from sets that the runners collect; nothing in this file estimates, imputes or
smooths a missing value. A case that could not be run is absent from the case
list, never scored as zero.
"""
import math

# Every action the production RLR can execute. Knowing may name an action the
# agent is not allowed to name here, but Acting can only ever produce a subset,
# so the universe is fixed and must not be derived from the run.
RLR_ACTIONS = {
    "observe", "set_vision_fps", "set_reply_tokens", "set_load_target", "remember",
    "recall", "reflect", "set_goal", "complete_goal", "cooldown", "abstain",
}

QUADRANTS = ("Kc_Ac", "Kc_Aw", "Kw_Ac", "Kw_Aw")


def as_set(value):
    """Coerce a recorded action list to a set of names, dropping blanks."""
    if value is None:
        return set()
    if isinstance(value, str):
        value = [value]
    return {str(item).strip() for item in value if str(item).strip()}


def jaccard(predicted, truth):
    """|X n T| / |X u T|, with the paper's conventions for degenerate sets.

    Two empty sets are a perfect match: the internal-function cases legitimately
    have an empty ground truth, and treating that as total disagreement would
    make the most important control condition unscoreable. An empty prediction
    against a non-empty truth scores 0.
    """
    x, t = as_set(predicted), as_set(truth)
    if not x and not t:
        return 1.0
    union = x | t
    if not union:
        return 1.0
    return len(x & t) / len(union)


def tool_overuse(predicted, truth):
    """|X \\ T| / |X u T|; 0 when both sets are empty."""
    x, t = as_set(predicted), as_set(truth)
    union = x | t
    if not union:
        return 0.0
    return len(x - t) / len(union)


def tool_underuse(predicted, truth):
    """|T \\ X| / |X u T|; 0 when both sets are empty."""
    x, t = as_set(predicted), as_set(truth)
    union = x | t
    if not union:
        return 0.0
    return len(t - x) / len(union)


def kas(acc_know, acc_act):
    """Harmonic mean of the two branch accuracies.

    Returns 0 when either branch is 0, matching the paper's zero rule, which
    also avoids the 0/0 division. The unrounded inputs are compared rather than
    the rounded ones so a genuine 0 can never be hidden inside a rounded value.
    """
    if not acc_know or not acc_act:
        return 0.0
    return 2.0 * acc_know * acc_act / (acc_know + acc_act)


def quadrant(predicted_know, truth, predicted_act, truth_act):
    """Kc_Ac / Kc_Aw / Kw_Ac / Kw_Aw, using exact set equality per branch."""
    know_ok = as_set(predicted_know) == as_set(truth)
    act_ok = as_set(predicted_act) == as_set(truth_act)
    if know_ok and act_ok:
        return "Kc_Ac"
    if know_ok:
        return "Kc_Aw"
    return "Kw_Ac" if act_ok else "Kw_Aw"


def truth_set(case):
    """The ground-truth action set T for a case.

    Accepts either the bare list T or the case record's {"T": [...],
    "subtasks": [...]} envelope. Taking the envelope dict itself as a set of
    names would silently score every case as 0, because the truth would become
    the literal strings "T" and "subtasks".
    """
    truth = case.get("ground_truth", case)
    if isinstance(truth, dict):
        return as_set(truth.get("T"))
    return as_set(truth)


def score_case(case):
    """Score one case's Knowing and Acting branches against ground truth.

    case["ground_truth"] holds T (and optional subtasks). The Acting branch is
    scored against the same T: the agent is given one task, so what it was
    supposed to end up having invoked is exactly T.
    """
    truth = truth_set(case)
    know_pred = case["knowing"].get("required_actions")
    act = case["acting"]
    # E_act is what was actually executed. Proposed, invalid and vetoed actions
    # are recorded separately and must never leak into it, otherwise the
    # governor's interventions would be scored as if the model had chosen them.
    act_exec = act.get("E_act")
    return {
        "case_id": case["case_id"],
        "seed": case.get("seed"),
        "boundary": case.get("boundary"),
        "topology": case.get("topology"),
        "condition": case.get("condition"),
        "knowing": {
            "E_know": sorted(as_set(know_pred)),
            "confidence": case["knowing"].get("confidence"),
            "jaccard": jaccard(know_pred, truth),
            "overuse": tool_overuse(know_pred, truth),
            "underuse": tool_underuse(know_pred, truth),
            "exact_correct": as_set(know_pred) == as_set(truth),
        },
        "acting": {
            "E_proposed": sorted(as_set(act.get("E_proposed"))),
            "E_vetoed": sorted(as_set(act.get("E_vetoed"))),
            "E_invalid": sorted(as_set(act.get("E_invalid"))),
            "E_act": sorted(as_set(act_exec)),
            "jaccard": jaccard(act_exec, truth),
            "overuse": tool_overuse(act_exec, truth),
            "underuse": tool_underuse(act_exec, truth),
            "exact_correct": as_set(act_exec) == as_set(truth),
            "tool_call_count": act.get("tool_call_count", 0),
        },
        "quadrant": quadrant(know_pred, truth, act_exec, truth),
        "pass": bool(case.get("pass")),
    }


def identity_error(case_row):
    """1 - J should equal ToolOver + ToolUnder for each branch.

    Returned so the caller can assert the identity numerically on real data
    rather than trusting that the three functions agree by construction.
    """
    out = {}
    for branch in ("knowing", "acting"):
        b = case_row[branch]
        out[branch] = round((1.0 - b["jaccard"]) - (b["overuse"] + b["underuse"]), 12)
    return out


def aggregate(rows, breakdown=None):
    """Mean the per-case Jaccards, then derive the rest of the metrics.

    The means are the paper's quantities, not a Jaccard over pooled action sets:
    pooling would let one frequently repeated action dominate.
    """
    rows = [r for r in (rows or []) if r]
    if not rows:
        return None

    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    acc_know = mean([r["knowing"]["jaccard"] for r in rows])
    acc_act = mean([r["acting"]["jaccard"] for r in rows])
    quadrants = {q: 0 for q in QUADRANTS}
    for r in rows:
        quadrants[r["quadrant"]] = quadrants.get(r["quadrant"], 0) + 1
    n = len(rows)
    result = {
        "n": n,
        "Acc_know": acc_know,
        "Acc_act": acc_act,
        "KAS": kas(acc_know, acc_act),
        "ToolOver_know": mean([r["knowing"]["overuse"] for r in rows]),
        "ToolUnder_know": mean([r["knowing"]["underuse"] for r in rows]),
        "ToolOver_act": mean([r["acting"]["overuse"] for r in rows]),
        "ToolUnder_act": mean([r["acting"]["underuse"] for r in rows]),
        "Kc_Ac_rate": quadrants["Kc_Ac"] / n,
        "Kc_Aw_rate": quadrants["Kc_Aw"] / n,
        "Kw_Ac_rate": quadrants["Kw_Ac"] / n,
        "Kw_Aw_rate": quadrants["Kw_Aw"] / n,
        "pass_rate": mean([1.0 if r["pass"] else 0.0 for r in rows]),
        "average_tool_calls": mean([r["acting"]["tool_call_count"] for r in rows]),
    }
    result["DirGap"] = result["Kc_Aw_rate"] - result["Kw_Ac_rate"]
    internal = [r for r in rows if r.get("boundary") == "internal"]
    result["unnecessary_internal_tool_calls"] = (
        mean([r["acting"]["tool_call_count"] for r in internal]) if internal else None)
    if breakdown:
        groups = {}
        for r in rows:
            groups.setdefault(str(r.get(breakdown)), []).append(r)
        result["breakdown"] = {k: aggregate(v) for k, v in sorted(groups.items())}
    return result


def pearson(points):
    """Pearson r and a two-sided p-value for the (KAS, internal actions) claim.

    Returns (None, None) below three independent points. The brief forbids
    reporting a correlation from insufficient points, and a coefficient
    computed from two points is exactly 1.0 by construction, which would look
    like a perfect result and mean nothing.
    """
    pairs = [(float(x), float(y)) for x, y in points
             if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return None, None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None, None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    r = sxy / ((sxx ** 0.5) * (syy ** 0.5))
    r = max(-1.0, min(1.0, r))
    df = n - 2
    # Student-t two-tailed p via the regularised incomplete beta function.
    t = abs(r) * ((df / (1.0 - r * r)) ** 0.5) if df > 0 and r < 1.0 else None
    if t is None:
        return r, 0.0 if abs(r) >= 1.0 else 1.0
    return r, _t_sf(t, df)


def _betacf(a, b, x):
    tiny, eps, itmax = 1e-30, 3e-16, 200
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """Regularised incomplete beta I_x(a, b) by continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) +
             a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def _t_sf(t, df):
    """Two-tailed survival probability for Student's t."""
    x = df / (df + t * t)
    return _betai(df / 2.0, 0.5, x)

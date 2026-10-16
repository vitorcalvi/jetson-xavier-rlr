"""Operator kill-file to ``OrderController`` mode mapper.

Reads operator kill files inside a data directory and applies the
highest-precedence kill mode to a real ``OrderController``. Removing a
file is intentionally non-de-escalating: de-escalation requires the
operator to call ``controller.set_mode(..., operator_approved=True)``
elsewhere, so a transient filesystem state cannot silently relax safety.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from reflective_loop_runtime.trading.orders import (
    CANCEL_AND_FLATTEN,
    EMERGENCY_STOP,
    HALT_ENTRIES,
    RUN,
)

if TYPE_CHECKING:
    from reflective_loop_runtime.trading.orders import OrderController


# Filename -> OrderController mode. Precedence EMERGENCY > FLATTEN > KILL is
# enforced at runtime via the descending ``_RESTRICTIVENESS`` ordering below,
# not implicit dict iteration order.
FILES: dict[str, str] = {
    "EMERGENCY": EMERGENCY_STOP,
    "FLATTEN": CANCEL_AND_FLATTEN,
    "KILL": HALT_ENTRIES,
}

# Strictly increasing restrictiveness: any target with a larger value is an
# escalation that ``apply`` may invoke without operator approval.
_RESTRICTIVENESS: dict[str, int] = {
    RUN: 0,
    HALT_ENTRIES: 1,
    CANCEL_AND_FLATTEN: 2,
    EMERGENCY_STOP: 3,
}


def requested_mode(data_dir) -> str | None:
    """Return the highest-precedence kill mode whose file exists.

    ``None`` means "no operator kill requested"; it NEVER implies ``RUN``.
    """
    p = Path(data_dir)
    # Walk in descending order of restrictiveness so a stray re-ordering of
    # ``FILES`` cannot silently weaken precedence guarantees.
    for name, mode in sorted(
        FILES.items(), key=lambda kv: -_RESTRICTIVENESS[kv[1]]
    ):
        if (p / name).exists():
            return mode
    return None


def apply(controller: OrderController, data_dir) -> str | None:
    """Apply the highest-precedence kill file, escalating only.

    Behaviour:

    * Returns the requested mode (or ``None`` when no kill file is present).
    * If a kill file is present and its mode is strictly more restrictive
      than the controller's current mode, call ``controller.set_mode(m)``.
    * Never de-escalates: removing a file does not lower the mode.
    """
    m = requested_mode(data_dir)
    if m is None:
        return None
    current = controller.snapshot()["mode"]
    if current == m:
        return m
    if _RESTRICTIVENESS[m] > _RESTRICTIVENESS[current]:
        controller.set_mode(m)
    return m

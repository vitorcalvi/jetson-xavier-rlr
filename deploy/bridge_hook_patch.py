"""Pure deterministic transformation of legacy Stage2 runner; no live restart.

This file is invoked only by the Git publication builder / local fallback.
"""

from __future__ import annotations


def integrate_runner(original: str) -> str:
    """Inject nonblocking observer calls; preserve all existing broker branches."""
    if "Stage2ShadowInterceptor.from_env" in original:
        return original
    transformations = [
        (
            "        state=controller.startup_reconcile()\n        gate=StrictSessionQuoteGate()",
            """        state=controller.startup_reconcile()
        # Observer-only: broker execution remains exclusively Stage2-owned.
        _toon_bridge=None
        try:
            from stage2_toon_bridge import Stage2ShadowInterceptor,SafetyLockViolation
            _toon_bridge=Stage2ShadowInterceptor.from_env(
                ledger=LEDGER,halt_file=HOME/'flags/HALT_ENTRIES')
        except SafetyLockViolation:
            raise  # Explicit override attempt under a halt is fatal.
        except Exception:
            _toon_bridge=None  # Absent/broken shadow code cannot block Stage2.
        gate=StrictSessionQuoteGate()""",
        ),
        (
            "            state=controller.state_row(); mode=state['state']",
            """            state=controller.state_row(); mode=state['state']
            if _toon_bridge is not None:
                try:
                    _toon_bridge.on_controller_tick(mode=mode,
                        timestamp_ms=int(now*1000),
                        transition_seq=int(state['transition_seq']))
                except Exception:
                    pass  # Shadow must not affect legacy control.""",
        ),
        (
            "                                     now=now,headers=api.market_headers())['SPY']\n                closed=",
            """                                     now=now,headers=api.market_headers())['SPY']
                if _toon_bridge is not None:
                    try:
                        _toon_bridge.on_market_snapshot(symbol='SPY',snapshot=snap,
                             timestamp_ms=int(now*1000),position={
                                 'state':mode,'held_qty':state['held_qty']})
                    except Exception:
                        pass  # Observer is strictly advisory.
                closed=""",
        ),
        (
            "                        controller.execute_entry(cycle_id,gate_value)",
            """                        if _toon_bridge is not None:
                            try:
                                _toon_bridge.on_stage2_signal(action='BUY',
                                    qty=25/max(float(q.ask),.01),price=float(q.ask),
                                    timestamp_ms=int(now*1000))
                            except Exception:
                                pass
                        controller.execute_entry(cycle_id,gate_value)""",
        ),
    ]
    result = original
    for old, new in transformations:
        if result.count(old) != 1:
            raise RuntimeError("Stage2 changed; refusing unsafe hook injection at " + old[:75])
        result = result.replace(old, new, 1)
    return result

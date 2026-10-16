"""An offline lifecycle pass must never imply the maker/paper gate passes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "certify_alpaca_paper_30.py"


def test_30_generated_lifecycles_fail_maker_promotion_gate():
    spec = importlib.util.spec_from_file_location("paper_certifier_script", MODULE)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    result = module.simulate_30()
    assert result["total_lifecycles"] == result["passed_lifecycles"] == 30
    assert result["state_machine_pass"]
    assert result["maker_fill_ratio_simulated"] == 0.5
    assert result["simulated_maker_gate_pass"] is False
    assert result["pass_simulation"] is False
    assert result["paper_promotion_ready"] is False
    assert result["real_paper_fills"] == 0
    assert result["broker_certified"] is False


def test_zero_quantity_cannot_be_valid_lifecycle():
    spec = importlib.util.spec_from_file_location("paper_certifier_zero", MODULE)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert not module.Lifecycle(("new", "accepted", "canceled"), 0, 0, False, 0).valid()

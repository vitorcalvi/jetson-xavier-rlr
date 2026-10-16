"""Tests for ``trading.killmodes`` against a real ``OrderController``."""

from __future__ import annotations

from reflective_loop_runtime.trading.killmodes import apply, requested_mode
from reflective_loop_runtime.trading.orders import (
    CANCEL_AND_FLATTEN,
    EMERGENCY_STOP,
    HALT_ENTRIES,
    OrderController,
)


def _new_controller(tmp_path):
    return OrderController(tmp_path / "orders.sqlite")


def test_no_files_returns_none(tmp_path):
    assert requested_mode(tmp_path) is None
    ctrl = _new_controller(tmp_path)
    try:
        result = apply(ctrl, tmp_path)
        assert result is None
        assert ctrl.snapshot()["mode"] == HALT_ENTRIES
    finally:
        ctrl.close()


def test_precedence_emergency_over_flatten_over_kill(tmp_path):
    (tmp_path / "EMERGENCY").write_text("x")
    (tmp_path / "FLATTEN").write_text("x")
    (tmp_path / "KILL").write_text("x")

    assert requested_mode(tmp_path) == EMERGENCY_STOP

    (tmp_path / "EMERGENCY").unlink()
    assert requested_mode(tmp_path) == CANCEL_AND_FLATTEN

    (tmp_path / "FLATTEN").unlink()
    assert requested_mode(tmp_path) == HALT_ENTRIES

    (tmp_path / "KILL").unlink()
    assert requested_mode(tmp_path) is None


def test_apply_escalates_to_halt(tmp_path):
    (tmp_path / "KILL").write_text("x")
    ctrl = _new_controller(tmp_path)
    try:
        assert ctrl.snapshot()["mode"] == HALT_ENTRIES
        result = apply(ctrl, tmp_path)
        assert result == HALT_ENTRIES
        assert ctrl.snapshot()["mode"] == HALT_ENTRIES
    finally:
        ctrl.close()


def test_apply_never_deescalates_when_file_removed(tmp_path):
    (tmp_path / "FLATTEN").write_text("x")
    ctrl = _new_controller(tmp_path)
    try:
        apply(ctrl, tmp_path)
        assert ctrl.snapshot()["mode"] == CANCEL_AND_FLATTEN

        (tmp_path / "FLATTEN").unlink()
        result = apply(ctrl, tmp_path)
        assert result is None
        # Mode must NOT be lowered just because the file is gone.
        assert ctrl.snapshot()["mode"] == CANCEL_AND_FLATTEN
    finally:
        ctrl.close()


def test_apply_emergency_from_halt(tmp_path):
    (tmp_path / "EMERGENCY").write_text("x")
    ctrl = _new_controller(tmp_path)
    try:
        assert ctrl.snapshot()["mode"] == HALT_ENTRIES
        result = apply(ctrl, tmp_path)
        assert result == EMERGENCY_STOP
        assert ctrl.snapshot()["mode"] == EMERGENCY_STOP
    finally:
        ctrl.close()

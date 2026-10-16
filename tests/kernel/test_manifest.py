"""Tests for the kernel ToolManifest, its risk classes, and call validation."""

from __future__ import annotations

import pytest

from reflective_loop_runtime.kernel.manifest import (
    DEFAULT_MANIFEST,
    IDEMPOTENCY,
    REVERSIBILITY,
    RISK_CLASSES,
    CostModel,
    FieldSpec,
    ManifestError,
    OperationSpec,
    ToolManifest,
    ToolSpec,
    default_manifest,
    validate_call,
)


def _single_op_manifest(op: OperationSpec) -> ToolManifest:
    return ToolManifest([ToolSpec("t", {"op": op})])


class TestValidateCallDispatch:
    def test_unknown_tool_rejected(self):
        with pytest.raises(ManifestError):
            DEFAULT_MANIFEST.validate_call("ghost", "sample", {})

    def test_unknown_operation_rejected(self):
        with pytest.raises(ManifestError):
            DEFAULT_MANIFEST.validate_call("telemetry", "ghost", {})

    def test_unknown_operation_for_known_tool_is_specific(self):
        with pytest.raises(ManifestError, match="unknown operation"):
            DEFAULT_MANIFEST.validate_call("goal", "detonate", {})

    def test_valid_read_only_call_returns_empty_args(self):
        assert DEFAULT_MANIFEST.validate_call("telemetry", "sample") == {}

    def test_valid_call_returns_normalised_args(self):
        result = validate_call("state", "set_load_target", {"load": 1})
        assert result == {"load": 1.0}
        assert isinstance(result["load"], float)

    def test_extra_unknown_args_rejected(self):
        with pytest.raises(ManifestError, match="unknown argument"):
            validate_call("state", "set_load_target", {"load": 0.5, "sudo": True})

    def test_missing_required_arg_rejected(self):
        with pytest.raises(ManifestError, match="missing required argument"):
            validate_call("state", "set_load_target", {})

    def test_args_must_be_a_mapping(self):
        with pytest.raises(ManifestError, match="args must be a mapping"):
            validate_call("telemetry", "sample", ["not", "a", "mapping"])


class TestSchemaViolations:
    @pytest.mark.parametrize(
        "tool, op, args",
        [
            ("state", "set_load_target", {"load": "high"}),
            ("state", "set_load_target", {"load": 2.0}),
            ("state", "set_load_target", {"load": -0.1}),
            ("state", "set_load_target", {"load": float("nan")}),
            ("state", "set_load_target", {"load": float("inf")}),
            ("state", "set_reply_tokens", {"tokens": True}),
            ("state", "set_reply_tokens", {"tokens": 1.5}),
            ("state", "set_reply_tokens", {"tokens": 5}),
            ("state", "set_reply_tokens", {"tokens": 9999}),
            ("state", "set_vision_fps", {"fps": 9.0}),
            ("state", "set_vision_fps", {"fps": 0.0}),
            ("memory", "recall", {"query": ""}),
            ("memory", "recall", {"query": "x" * 1001}),
            ("memory", "remember", {"text": "hi", "importance": 99}),
            ("memory", "forget", {"memory_id": 0}),
            ("goal", "complete", {"goal_id": "one"}),
            ("notify", "send", {"channel": "email", "message": "hi"}),
            ("notify", "send", {"channel": "local_log", "message": ""}),
        ],
    )
    def test_schema_violation_rejected(self, tool, op, args):
        with pytest.raises(ManifestError):
            validate_call(tool, op, args)

    def test_boolean_is_not_an_integer(self):
        with pytest.raises(ManifestError, match="integer"):
            validate_call("goal", "complete", {"goal_id": True})

    def test_string_is_not_a_number(self):
        with pytest.raises(ManifestError, match="number"):
            validate_call("state", "set_vision_fps", {"fps": "2"})


class TestFieldSpecs:
    def test_bare_type_string_in_schema(self):
        manifest = _single_op_manifest(OperationSpec(name="op", input_schema={"count": "int"}))
        assert manifest.validate_call("t", "op", {"count": 3}) == {"count": 3}
        with pytest.raises(ManifestError):
            manifest.validate_call("t", "op", {"count": "three"})
        with pytest.raises(ManifestError):
            manifest.validate_call("t", "op", {})

    def test_optional_field_may_be_omitted(self):
        op = OperationSpec(
            name="op",
            input_schema={"count": FieldSpec("int", required=False)},
        )
        manifest = _single_op_manifest(op)
        assert manifest.validate_call("t", "op") == {}
        assert manifest.validate_call("t", "op", {"count": 2}) == {"count": 2}

    def test_unsupported_field_type_rejected(self):
        with pytest.raises(ManifestError, match="unsupported field type"):
            FieldSpec("bytes")

    def test_inverted_numeric_bounds_rejected(self):
        with pytest.raises(ManifestError, match="min must not exceed max"):
            FieldSpec("int", min=10, max=1)

    def test_bounds_on_non_numeric_field_rejected(self):
        with pytest.raises(ManifestError, match="numeric bounds"):
            FieldSpec("str", min=1, max=2)

    def test_empty_choices_rejected(self):
        with pytest.raises(ManifestError, match="choices"):
            FieldSpec("str", choices=())

    def test_schema_entry_type_checked(self):
        with pytest.raises(ManifestError, match="FieldSpec or a type string"):
            OperationSpec(name="op", input_schema={"x": 42})

    def test_bool_field_accepts_only_bool(self):
        manifest = _single_op_manifest(OperationSpec(name="op", input_schema={"flag": "bool"}))
        assert manifest.validate_call("t", "op", {"flag": True}) == {"flag": True}
        with pytest.raises(ManifestError):
            manifest.validate_call("t", "op", {"flag": 1})


class TestRiskClasses:
    def test_default_manifest_declares_all_risk_classes(self):
        declared = {
            op.risk_class
            for tool_name in DEFAULT_MANIFEST.tool_names
            for op in (
                DEFAULT_MANIFEST.operation(tool_name, op_name)
                for op_name in DEFAULT_MANIFEST.tool(tool_name).operations
            )
        }
        assert declared == set(RISK_CLASSES)

    def test_risk_class_lookup(self):
        assert DEFAULT_MANIFEST.risk_class("telemetry", "sample") == "A0"
        assert DEFAULT_MANIFEST.risk_class("state", "set_vision_fps") == "A1"
        assert DEFAULT_MANIFEST.risk_class("goal", "complete") == "A2"
        assert DEFAULT_MANIFEST.risk_class("memory", "forget") == "A3"

    def test_a0_is_not_side_effecting(self):
        assert DEFAULT_MANIFEST.operation("memory", "recall").is_side_effecting is False
        assert DEFAULT_MANIFEST.operation("memory", "remember").is_side_effecting is True

    def test_a3_is_irreversible_without_compensation(self):
        forget = DEFAULT_MANIFEST.operation("memory", "forget")
        assert forget.risk_class == "A3"
        assert forget.reversibility == "irreversible"
        assert forget.compensation is None

    def test_a2_requires_compensation(self):
        with pytest.raises(ManifestError, match="requires a compensation"):
            OperationSpec(name="op", risk_class="A2", reversibility="compensable")

    def test_a2_accepts_compensation(self):
        op = OperationSpec(
            name="op",
            risk_class="A2",
            reversibility="compensable",
            compensation="undo it",
        )
        assert op.compensation == "undo it"

    def test_a0_rejects_effects(self):
        with pytest.raises(ManifestError, match="must not declare effects"):
            OperationSpec(name="op", risk_class="A0", effects=("boom",))

    def test_a0_rejects_compensation(self):
        with pytest.raises(ManifestError, match="must not need compensation"):
            OperationSpec(name="op", risk_class="A0", compensation="undo")

    def test_a3_requires_irreversibility(self):
        with pytest.raises(ManifestError, match="must be irreversible"):
            OperationSpec(name="op", risk_class="A3", reversibility="compensable")

    def test_a3_rejects_compensation(self):
        with pytest.raises(ManifestError, match="cannot be compensated"):
            OperationSpec(
                name="op",
                risk_class="A3",
                reversibility="irreversible",
                compensation="undo",
            )

    def test_a1_cannot_be_irreversible(self):
        with pytest.raises(ManifestError, match="cannot be irreversible"):
            OperationSpec(name="op", risk_class="A1", reversibility="irreversible")

    def test_invalid_risk_class_rejected(self):
        with pytest.raises(ManifestError, match="risk_class"):
            OperationSpec(name="op", risk_class="A9")

    def test_invalid_reversibility_rejected(self):
        with pytest.raises(ManifestError, match="reversibility"):
            OperationSpec(name="op", reversibility="maybe")

    def test_invalid_idempotency_rejected(self):
        with pytest.raises(ManifestError, match="idempotency"):
            OperationSpec(name="op", idempotency="sometimes")

    def test_classification_vocabularies(self):
        assert RISK_CLASSES == ("A0", "A1", "A2", "A3")
        assert "compensable" in REVERSIBILITY
        assert "non_idempotent" in IDEMPOTENCY


class TestOperationMetadata:
    def test_operation_declares_all_required_fields(self):
        op = DEFAULT_MANIFEST.operation("notify", "send")
        assert isinstance(op.input_schema, dict)
        assert op.preconditions and isinstance(op.preconditions, tuple)
        assert op.effects and isinstance(op.effects, tuple)
        assert op.risk_class in RISK_CLASSES
        assert op.reversibility in REVERSIBILITY
        assert op.idempotency in IDEMPOTENCY
        assert isinstance(op.compensation, str)
        assert op.allowed_destinations == ("local_log", "operator_console")
        assert isinstance(op.cost_model, CostModel)

    def test_read_only_operation_has_no_effects_or_destinations(self):
        op = DEFAULT_MANIFEST.operation("telemetry", "sample")
        assert op.effects == ()
        assert op.allowed_destinations == ()
        assert op.compensation is None

    def test_compensation_string_or_none_only(self):
        with pytest.raises(ManifestError, match="compensation"):
            OperationSpec(name="op", risk_class="A1", compensation=123)

    def test_empty_operation_name_rejected(self):
        with pytest.raises(ManifestError, match="operation name"):
            OperationSpec(name="")

    def test_single_string_effects_rejected(self):
        with pytest.raises(ManifestError, match="sequence of strings"):
            OperationSpec(name="op", risk_class="A1", effects="one effect")

    def test_cost_model_validation(self):
        with pytest.raises(ManifestError, match="unit"):
            CostModel("", 1.0)
        with pytest.raises(ManifestError, match="estimate"):
            CostModel("x", -1.0)
        with pytest.raises(ManifestError, match="estimate"):
            CostModel("x", float("inf"))

    def test_cost_model_must_be_a_cost_model(self):
        with pytest.raises(ManifestError, match="cost_model"):
            OperationSpec(name="op", cost_model={"unit": "x"})


class TestManifestRegistry:
    def test_duplicate_tool_registration_rejected(self):
        manifest = ToolManifest()
        tool = ToolSpec("t", {"op": OperationSpec(name="op")})
        manifest.register(tool)
        with pytest.raises(ManifestError, match="already registered"):
            manifest.register(tool)

    def test_operation_key_must_match_spec_name(self):
        with pytest.raises(ManifestError, match="does not match"):
            ToolSpec("t", {"wrong": OperationSpec(name="right")})

    def test_tool_requires_operation_specs(self):
        with pytest.raises(ManifestError, match="OperationSpec"):
            ToolSpec("t", {"op": "not-a-spec"})

    def test_contains_and_tool_names(self):
        assert "goal" in DEFAULT_MANIFEST
        assert "ghost" not in DEFAULT_MANIFEST
        assert DEFAULT_MANIFEST.tool_names == ("telemetry", "state", "memory", "goal", "notify")

    def test_default_manifest_helper_returns_fresh_instance(self):
        first = default_manifest()
        second = default_manifest()
        assert first is not second
        first.register(ToolSpec("extra", {"op": OperationSpec(name="op")}))
        assert "extra" in first
        assert "extra" not in second
        assert "extra" not in DEFAULT_MANIFEST

    def test_module_level_validate_call_uses_default_manifest(self):
        assert validate_call("memory", "recall", {"query": "hello"}) == {"query": "hello"}

    def test_module_level_validate_call_accepts_custom_manifest(self):
        manifest = _single_op_manifest(OperationSpec(name="op", input_schema={"n": "int"}))
        assert validate_call("t", "op", {"n": 1}, manifest) == {"n": 1}
        with pytest.raises(ManifestError):
            validate_call("t", "missing", {}, manifest)

"""Declarative tool manifest with per-operation risk classes.

The kernel describes every capability it may invoke before that capability can
be called. Each operation carries a small, auditable contract:

* ``input_schema`` -- a simple typed dict spec (argument name -> ``FieldSpec``,
  or a bare type string for a required, unconstrained argument).
* ``preconditions`` -- human-readable conditions that must hold before the call.
* ``effects`` -- the state changes the operation may cause.
* ``risk_class`` -- ``A0`` read-only, ``A1`` reversible, ``A2`` compensable,
  ``A3`` irreversible.
* ``reversibility`` / ``idempotency`` -- how the effect is undone and whether
  repeating the call is harmless.
* ``compensation`` -- the undo/repair step; required for ``A2`` and forbidden
  for ``A0``/``A3``.
* ``allowed_destinations`` -- the closed set of external destinations the
  operation may write to (empty means no external destination).
* ``cost_model`` -- the resource unit and estimate used for budgeting.

``ToolManifest.validate_call`` is the single gate for untrusted arguments: it
rejects unknown tools/operations and any schema violation by raising
``ManifestError``. The manifest is stdlib-only, deterministic, and has no side
effects of its own.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .budget import RESOURCES
from .policy import canonical_json

__all__ = [
    "IDEMPOTENCY",
    "REVERSIBILITY",
    "RISK_CLASSES",
    "CostModel",
    "DEFAULT_MANIFEST",
    "FieldSpec",
    "ManifestError",
    "OperationSpec",
    "ToolManifest",
    "ToolSpec",
    "default_manifest",
    "validate_call",
]

# Risk classes, ordered from least to most dangerous. ``A0`` is read-only,
# ``A1`` has a reversible effect, ``A2`` has a bounded effect that needs a
# compensation, and ``A3`` is irreversible with a large blast radius.
RISK_CLASSES = ("A0", "A1", "A2", "A3")
REVERSIBILITY = ("reversible", "compensable", "irreversible")
IDEMPOTENCY = ("idempotent", "non_idempotent")

_FIELD_TYPES = ("str", "int", "float", "bool")


class ManifestError(ValueError):
    """Raised when a tool/operation is unknown or a call violates its schema."""


@dataclass(frozen=True)
class FieldSpec:
    """Type and optional bounds for one argument in an operation schema."""

    type: str
    required: bool = True
    min: float | None = None
    max: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    choices: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.type not in _FIELD_TYPES:
            raise ManifestError(
                f"unsupported field type {self.type!r}; expected one of {list(_FIELD_TYPES)}"
            )
        if self.type not in ("int", "float") and (self.min is not None or self.max is not None):
            raise ManifestError(f"numeric bounds are not valid for {self.type!r} fields")
        if self.type != "str" and (
            self.min_length is not None
            or self.max_length is not None
            or self.choices is not None
        ):
            raise ManifestError(
                f"length/choices constraints are not valid for {self.type!r} fields"
            )
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ManifestError("field min must not exceed max")
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.min_length > self.max_length
        ):
            raise ManifestError("field min_length must not exceed max_length")
        if self.choices is not None:
            if not self.choices:
                raise ManifestError("field choices must not be empty")
            for choice in self.choices:
                if not isinstance(choice, str) or not choice:
                    raise ManifestError("field choices must be non-empty strings")

    def validate(self, value: Any, name: str) -> Any:
        """Return a normalised value or raise ``ManifestError`` on any violation."""
        if self.type == "bool":
            if not isinstance(value, bool):
                raise ManifestError(f"{name} must be a boolean")
            return value
        if self.type == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ManifestError(f"{name} must be an integer")
            self._check_bounds(float(value), name)
            return value
        if self.type == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ManifestError(f"{name} must be a number")
            number = float(value)
            if not math.isfinite(number):
                raise ManifestError(f"{name} must be finite")
            self._check_bounds(number, name)
            return number
        if not isinstance(value, str):
            raise ManifestError(f"{name} must be text")
        if self.min_length is not None and len(value) < self.min_length:
            raise ManifestError(f"{name} must be at least {self.min_length} characters")
        if self.max_length is not None and len(value) > self.max_length:
            raise ManifestError(f"{name} must be at most {self.max_length} characters")
        if self.choices is not None and value not in self.choices:
            raise ManifestError(f"{name} must be one of {sorted(self.choices)}")
        return value

    def _check_bounds(self, number: float, name: str) -> None:
        if self.min is not None and number < self.min:
            raise ManifestError(f"{name} must be >= {self.min}")
        if self.max is not None and number > self.max:
            raise ManifestError(f"{name} must be <= {self.max}")


@dataclass(frozen=True)
class CostModel:
    """The resource unit and estimate used to budget an operation."""

    unit: str = "none"
    estimate: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.unit, str) or not self.unit:
            raise ManifestError("cost_model unit must be non-empty text")
        if (
            isinstance(self.estimate, bool)
            or not isinstance(self.estimate, (int, float))
            or not math.isfinite(float(self.estimate))
            or float(self.estimate) < 0
        ):
            raise ManifestError("cost_model estimate must be a non-negative finite number")
        # Every declared cost unit must be a budget resource, so a manifest can
        # never price an operation in a unit the budget service cannot cap. The
        # explicit ``none`` sentinel means "this operation is not metered".
        if self.unit != "none" and self.unit not in RESOURCES:
            raise ManifestError(
                f"cost_model unit {self.unit!r} is not a budget resource; "
                f"expected one of {sorted(RESOURCES)} or 'none'"
            )


def _text_tuple(value: Any, what: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise ManifestError(f"{what} must be a sequence of strings, not a single string")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ManifestError(f"{what} must be an iterable of strings") from exc
    for item in items:
        if not isinstance(item, str) or not item:
            raise ManifestError(f"{what} must contain only non-empty strings")
    return items


@dataclass(frozen=True)
class OperationSpec:
    """Everything the kernel needs to know before calling one tool operation."""

    name: str
    input_schema: Mapping[str, FieldSpec] = field(default_factory=dict)
    preconditions: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    risk_class: str = "A0"
    reversibility: str = "reversible"
    idempotency: str = "idempotent"
    compensation: str | None = None
    allowed_destinations: tuple[str, ...] = ()
    cost_model: CostModel = field(default_factory=CostModel)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ManifestError("operation name must be non-empty text")
        if self.risk_class not in RISK_CLASSES:
            raise ManifestError(f"risk_class {self.risk_class!r} must be one of {list(RISK_CLASSES)}")
        if self.reversibility not in REVERSIBILITY:
            raise ManifestError(
                f"reversibility {self.reversibility!r} must be one of {list(REVERSIBILITY)}"
            )
        if self.idempotency not in IDEMPOTENCY:
            raise ManifestError(
                f"idempotency {self.idempotency!r} must be one of {list(IDEMPOTENCY)}"
            )
        if self.compensation is not None and (
            not isinstance(self.compensation, str) or not self.compensation
        ):
            raise ManifestError("compensation must be non-empty text or None")
        if not isinstance(self.cost_model, CostModel):
            raise ManifestError("cost_model must be a CostModel")

        object.__setattr__(self, "preconditions", _text_tuple(self.preconditions, "preconditions"))
        object.__setattr__(self, "effects", _text_tuple(self.effects, "effects"))
        object.__setattr__(
            self,
            "allowed_destinations",
            _text_tuple(self.allowed_destinations, "allowed_destinations"),
        )

        if not isinstance(self.input_schema, Mapping):
            raise ManifestError("input_schema must be a mapping")
        schema: dict[str, FieldSpec] = {}
        for arg_name, spec in self.input_schema.items():
            if not isinstance(arg_name, str) or not arg_name:
                raise ManifestError("schema argument names must be non-empty text")
            if isinstance(spec, str):
                spec = FieldSpec(spec)
            if not isinstance(spec, FieldSpec):
                raise ManifestError(
                    f"schema entry {arg_name!r} must be a FieldSpec or a type string"
                )
            schema[arg_name] = spec
        object.__setattr__(self, "input_schema", schema)

        if self.risk_class == "A0":
            if self.effects:
                raise ManifestError(f"A0 operation {self.name!r} must not declare effects")
            if self.reversibility != "reversible":
                raise ManifestError(f"A0 operation {self.name!r} must be reversible")
            if self.compensation is not None:
                raise ManifestError(f"A0 operation {self.name!r} must not need compensation")
        if self.risk_class == "A2" and self.compensation is None:
            raise ManifestError(f"A2 operation {self.name!r} requires a compensation")
        if self.risk_class in ("A1", "A2") and self.reversibility == "irreversible":
            raise ManifestError(f"{self.risk_class} operation {self.name!r} cannot be irreversible")
        if self.risk_class == "A3":
            if self.reversibility != "irreversible":
                raise ManifestError(f"A3 operation {self.name!r} must be irreversible")
            if self.compensation is not None:
                raise ManifestError(f"A3 operation {self.name!r} cannot be compensated")

    @property
    def is_side_effecting(self) -> bool:
        return self.risk_class != "A0"


@dataclass(frozen=True)
class ToolSpec:
    """A named tool and the operations it exposes."""

    name: str
    operations: Mapping[str, OperationSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ManifestError("tool name must be non-empty text")
        if not isinstance(self.operations, Mapping):
            raise ManifestError("operations must be a mapping")
        normalized: dict[str, OperationSpec] = {}
        for op_name, op in self.operations.items():
            if not isinstance(op, OperationSpec):
                raise ManifestError(f"operation {op_name!r} must be an OperationSpec")
            if op_name != op.name:
                raise ManifestError(
                    f"operation key {op_name!r} does not match spec name {op.name!r}"
                )
            normalized[op_name] = op
        object.__setattr__(self, "operations", normalized)

    def operation(self, op_name: str) -> OperationSpec:
        try:
            return self.operations[op_name]
        except KeyError:
            raise ManifestError(
                f"unknown operation {op_name!r} for tool {self.name!r}"
            ) from None


class ToolManifest:
    """A registry of tools whose calls are schema-validated on dispatch.

    Once :meth:`seal` is called the registry is frozen: :meth:`register` raises
    for any further tool, so a broker can guarantee the manifest it validates
    against is exactly the one it was constructed with.
    """

    def __init__(self, tools: Iterable[ToolSpec] = ()) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._sealed = False
        for tool in tools:
            self.register(tool)

    def register(self, tool: ToolSpec) -> None:
        if self._sealed:
            raise ManifestError("tool manifest is sealed; no further tools may be registered")
        if not isinstance(tool, ToolSpec):
            raise ManifestError("tool must be a ToolSpec")
        if tool.name in self._tools:
            raise ManifestError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def seal(self) -> None:
        """Freeze the registry; idempotent so a broker can seal on every use."""
        self._sealed = True

    @property
    def sealed(self) -> bool:
        return self._sealed

    def digest(self) -> str:
        """A stable SHA-256 over every tool/operation contract in the registry.

        Bound into each permit so a permit issued under one manifest cannot be
        replayed against a different (weakened) manifest that happens to know the
        same tool/op names.
        """
        return hashlib.sha256(canonical_json(self._digest_material()).encode("utf-8")).hexdigest()

    def _digest_material(self) -> dict[str, Any]:
        tools: dict[str, Any] = {}
        for name in sorted(self._tools):
            spec = self._tools[name]
            operations: dict[str, Any] = {}
            for op_name in sorted(spec.operations):
                op = spec.operations[op_name]
                schema = {
                    arg: {
                        "type": field.type,
                        "required": field.required,
                        "min": field.min,
                        "max": field.max,
                        "min_length": field.min_length,
                        "max_length": field.max_length,
                        "choices": list(field.choices) if field.choices is not None else None,
                    }
                    for arg, field in sorted(op.input_schema.items())
                }
                operations[op_name] = {
                    "risk_class": op.risk_class,
                    "reversibility": op.reversibility,
                    "idempotency": op.idempotency,
                    "compensation": op.compensation,
                    "allowed_destinations": list(op.allowed_destinations),
                    "cost_unit": op.cost_model.unit,
                    "cost_estimate": op.cost_model.estimate,
                    "schema": schema,
                }
            tools[name] = {"operations": operations}
        return tools

    def __contains__(self, tool_name: object) -> bool:
        return tool_name in self._tools

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def tool(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise ManifestError(f"unknown tool {name!r}") from None

    def operation(self, tool: str, op: str) -> OperationSpec:
        return self.tool(tool).operation(op)

    def risk_class(self, tool: str, op: str) -> str:
        return self.operation(tool, op).risk_class

    def validate_call(self, tool: str, op: str, args: Any = None) -> dict[str, Any]:
        """Validate one call and return a normalised copy of its arguments.

        Raises ``ManifestError`` when the tool or operation is unknown, when
        ``args`` is not a mapping, or when any argument is missing, unknown, or
        violates its declared field spec.
        """
        spec = self.operation(tool, op)
        if args is None:
            args = {}
        if not isinstance(args, Mapping):
            raise ManifestError("args must be a mapping")
        schema = spec.input_schema
        # Coerce keys to text before sorting: a Mapping with mixed-type keys
        # (e.g. {1: "a", "b": 2}) would otherwise raise TypeError from sorted()
        # and escape the fail-closed ManifestError deny path.
        unknown = sorted(str(key) for key in set(args) - set(schema))
        if unknown:
            raise ManifestError(f"unknown argument(s) for {tool}.{op}: {unknown}")
        normalized: dict[str, Any] = {}
        for arg_name, field_spec in schema.items():
            if arg_name not in args:
                if field_spec.required:
                    raise ManifestError(
                        f"missing required argument {arg_name!r} for {tool}.{op}"
                    )
                continue
            normalized[arg_name] = field_spec.validate(args[arg_name], arg_name)
        return normalized


def default_manifest() -> ToolManifest:
    """Build the built-in manifest of kernel tools and their operations.

    Returns a fresh instance so callers may extend it without mutating the
    module-level ``DEFAULT_MANIFEST``.
    """
    telemetry = ToolSpec(
        "telemetry",
        {
            "sample": OperationSpec(
                name="sample",
                preconditions=("telemetry source is reachable",),
                risk_class="A0",
                reversibility="reversible",
                idempotency="idempotent",
                cost_model=CostModel("io_read", 1.0, "one read-only sample"),
            ),
        },
    )
    state = ToolSpec(
        "state",
        {
            "set_load_target": OperationSpec(
                name="set_load_target",
                input_schema={"load": FieldSpec("float", min=0.0, max=1.0)},
                preconditions=("telemetry is fresh",),
                effects=("control state load target is updated",),
                risk_class="A1",
                reversibility="reversible",
                idempotency="idempotent",
                compensation="restore the previous load target",
                cost_model=CostModel("scheduler_tick", 1.0, "in-memory update"),
            ),
            "set_reply_tokens": OperationSpec(
                name="set_reply_tokens",
                input_schema={"tokens": FieldSpec("int", min=16, max=512)},
                preconditions=("telemetry is fresh",),
                effects=("reply token budget is updated",),
                risk_class="A1",
                reversibility="reversible",
                idempotency="idempotent",
                compensation="restore the previous reply token budget",
                cost_model=CostModel("scheduler_tick", 1.0, "in-memory update"),
            ),
            "set_vision_fps": OperationSpec(
                name="set_vision_fps",
                input_schema={"fps": FieldSpec("float", min=0.2, max=5.0)},
                preconditions=("telemetry is fresh",),
                effects=("vision frame rate is updated",),
                risk_class="A1",
                reversibility="reversible",
                idempotency="idempotent",
                compensation="restore the previous vision frame rate",
                cost_model=CostModel("scheduler_tick", 1.0, "in-memory update"),
            ),
        },
    )
    memory = ToolSpec(
        "memory",
        {
            "recall": OperationSpec(
                name="recall",
                input_schema={"query": FieldSpec("str", min_length=1, max_length=1000)},
                risk_class="A0",
                reversibility="reversible",
                idempotency="idempotent",
                cost_model=CostModel("store_read", 1.0, "read-only query"),
            ),
            "remember": OperationSpec(
                name="remember",
                input_schema={
                    "text": FieldSpec("str", min_length=1, max_length=4000),
                    "importance": FieldSpec("int", min=1, max=5),
                },
                preconditions=("store is writable",),
                effects=("a memory row is added",),
                risk_class="A1",
                reversibility="reversible",
                idempotency="non_idempotent",
                compensation="delete the created memory row",
                cost_model=CostModel("store_write", 1.0, "one row insert"),
            ),
            "forget": OperationSpec(
                name="forget",
                input_schema={"memory_id": FieldSpec("int", min=1, max=2**31 - 1)},
                preconditions=("memory_id exists",),
                effects=("a memory row is permanently deleted",),
                risk_class="A3",
                reversibility="irreversible",
                idempotency="idempotent",
                compensation=None,
                cost_model=CostModel("store_write", 1.0, "one row delete"),
            ),
        },
    )
    goal = ToolSpec(
        "goal",
        {
            "set": OperationSpec(
                name="set",
                input_schema={"text": FieldSpec("str", min_length=1, max_length=1000)},
                preconditions=("store is writable",),
                effects=("a goal row is added",),
                risk_class="A1",
                reversibility="reversible",
                idempotency="non_idempotent",
                compensation="complete or delete the created goal",
                cost_model=CostModel("store_write", 1.0, "one row insert"),
            ),
            "complete": OperationSpec(
                name="complete",
                input_schema={"goal_id": FieldSpec("int", min=1, max=2**31 - 1)},
                preconditions=("goal_id refers to an open goal",),
                effects=("the goal is marked complete",),
                risk_class="A2",
                reversibility="compensable",
                idempotency="idempotent",
                compensation="reopen the completed goal",
                cost_model=CostModel("store_write", 1.0, "one row update"),
            ),
        },
    )
    notify = ToolSpec(
        "notify",
        {
            "send": OperationSpec(
                name="send",
                input_schema={
                    "channel": FieldSpec("str", choices=("local_log", "operator_console")),
                    "message": FieldSpec("str", min_length=1, max_length=1000),
                },
                preconditions=("destination is allowlisted",),
                effects=("a message is delivered to the destination",),
                risk_class="A2",
                reversibility="compensable",
                idempotency="non_idempotent",
                compensation="deliver a retraction notice to the same destination",
                allowed_destinations=("local_log", "operator_console"),
                cost_model=CostModel("message", 1.0, "one small message"),
            ),
        },
    )
    return ToolManifest((telemetry, state, memory, goal, notify))


DEFAULT_MANIFEST = default_manifest()


def validate_call(
    tool: str,
    op: str,
    args: Any = None,
    manifest: ToolManifest | None = None,
) -> dict[str, Any]:
    """Validate a call against ``manifest`` (the default manifest by default)."""
    manifest = DEFAULT_MANIFEST if manifest is None else manifest
    return manifest.validate_call(tool, op, args)

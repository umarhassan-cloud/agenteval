"""Base class for stateful, simulated tool environments.

An :class:`Environment` owns a JSON-serialisable ``state`` dict (the "database"),
exposes tools declared with the :func:`tool` decorator, and records policy
violations. Backends deliberately do **not** enforce most business policies:
like real production APIs, they will happily execute a refund that policy forbids.
Detecting that is the evaluator's job, which is exactly what makes policy
adherence measurable.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from agenteval.types import ToolCall, ToolResult

FREE_TEXT_KEYS: frozenset[str] = frozenset({"reason", "note", "subject", "notes"})
"""State keys holding free-form text that should not affect state equality."""


class ToolError(Exception):
    """Raised by a tool implementation to return an error observation to the agent."""


@dataclass(frozen=True)
class ToolSpec:
    """JSON-schema description of a tool, provider-agnostic."""

    name: str
    description: str
    parameters: dict[str, Any]
    mutating: bool = False

    @property
    def required(self) -> list[str]:
        return list(self.parameters.get("required", []))

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    def to_anthropic(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}


def tool(description: str, *, mutating: bool = False, **params: tuple[str, str]) -> Callable:
    """Declare an environment method as an agent-callable tool.

    Args:
        description: Natural-language description shown to the model.
        mutating: Whether the tool changes backend state.
        **params: ``name=(json_type, description)`` for each parameter. Parameters
            without a default value in the method signature are marked required.
    """

    def deco(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        props: dict[str, Any] = {}
        required: list[str] = []
        for pname, p in sig.parameters.items():
            if pname == "self":
                continue
            jtype, pdesc = params.get(pname, ("string", ""))
            props[pname] = {"type": jtype, "description": pdesc}
            if p.default is inspect.Parameter.empty:
                required.append(pname)
        fn.__tool_spec__ = ToolSpec(  # type: ignore[attr-defined]
            name=fn.__name__,
            description=description,
            parameters={"type": "object", "properties": props, "required": required},
            mutating=mutating,
        )
        return fn

    return deco


def _strip_free_text(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_free_text(v) for k, v in obj.items() if k not in FREE_TEXT_KEYS}
    if isinstance(obj, list):
        return [_strip_free_text(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 2)
    return obj


class Environment:
    """Stateful simulated backend exposing tools to an agent."""

    domain: ClassVar[str] = "base"
    policy: ClassVar[str] = ""

    def __init__(self, initial_state: dict[str, Any]) -> None:
        self.initial_state: dict[str, Any] = copy.deepcopy(initial_state)
        self.state: dict[str, Any] = copy.deepcopy(initial_state)
        self.violations: list[str] = []
        self.history: list[tuple[ToolCall, ToolResult]] = []
        self._tools: dict[str, tuple[ToolSpec, Callable[..., Any]]] = {}
        for attr in dir(type(self)):
            fn = getattr(type(self), attr)
            spec = getattr(fn, "__tool_spec__", None)
            if spec is not None:
                self._tools[spec.name] = (spec, getattr(self, attr))

    # ------------------------------------------------------------------ tools
    @property
    def tool_specs(self) -> list[ToolSpec]:
        return [spec for spec, _ in sorted(self._tools.values(), key=lambda t: t[0].name)]

    def get_spec(self, name: str) -> ToolSpec | None:
        entry = self._tools.get(name)
        return entry[0] if entry else None

    def execute(self, call: ToolCall) -> ToolResult:
        """Validate and execute a tool call, never raising to the caller."""
        result = self._execute(call)
        self.history.append((call, result))
        return result

    def _execute(self, call: ToolCall) -> ToolResult:
        entry = self._tools.get(call.name)
        if entry is None:
            available = ", ".join(sorted(self._tools))
            return ToolResult(call.id, call.name, error=f"Unknown tool '{call.name}'. Available tools: {available}")
        spec, fn = entry
        props = spec.parameters["properties"]
        missing = [r for r in spec.required if r not in call.arguments]
        unknown = [a for a in call.arguments if a not in props]
        if missing:
            return ToolResult(call.id, call.name, error=f"Missing required argument(s): {', '.join(missing)}")
        if unknown:
            return ToolResult(call.id, call.name, error=f"Unexpected argument(s): {', '.join(unknown)}")
        try:
            output = fn(**call.arguments)
        except ToolError as e:
            return ToolResult(call.id, call.name, error=str(e))
        except (TypeError, ValueError) as e:
            return ToolResult(call.id, call.name, error=f"Invalid arguments: {e}")
        return ToolResult(call.id, call.name, output=output)

    # ----------------------------------------------------------------- policy
    def violate(self, rule: str, detail: str) -> None:
        """Record a policy violation; the action itself still executes."""
        self.violations.append(f"{rule}: {detail}")

    def finalize(self) -> None:
        """Hook for end-of-episode policy checks. Override in subclasses."""

    # ------------------------------------------------------------------ state
    def comparable_state(self) -> dict[str, Any]:
        """State used for equality checks, with free-text fields removed."""
        return _strip_free_text(self.state)

    def state_hash(self) -> str:
        blob = json.dumps(self.comparable_state(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _next_id(self, counter: str, prefix: str) -> str:
        meta = self.state.setdefault("meta", {})
        value = int(meta.get(counter, 1))
        meta[counter] = value + 1
        return f"{prefix}{value}"

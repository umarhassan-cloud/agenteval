"""Core data types shared across AgentEval.

Everything that crosses a module boundary (tool calls, observations, trajectories,
evaluation results) is a plain dataclass with lossless ``to_dict``/``from_dict``
round-tripping so runs can be persisted as JSONL and re-analysed offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]
Termination = Literal["final_answer", "max_steps", "agent_error"]


@dataclass
class ToolCall:
    """A single tool invocation requested by an agent."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def signature(self) -> str:
        """Canonical string used to detect repeated (looping) calls."""
        return f"{self.name}:{json.dumps(self.arguments, sort_keys=True, default=str)}"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolCall:
        return cls(name=d["name"], arguments=dict(d.get("arguments", {})), id=d.get("id", ""))


@dataclass
class ToolResult:
    """Outcome of executing a :class:`ToolCall` against an environment."""

    call_id: str
    name: str
    output: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_text(self) -> str:
        """Render the result the way it is shown to the model."""
        if self.error is not None:
            return f"ERROR: {self.error}"
        return json.dumps(self.output, sort_keys=True, default=str)

    def to_dict(self) -> dict[str, Any]:
        return {"call_id": self.call_id, "name": self.name, "output": self.output, "error": self.error}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolResult:
        return cls(call_id=d.get("call_id", ""), name=d["name"], output=d.get("output"), error=d.get("error"))


@dataclass
class Message:
    """Chat message in the agent's context window."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_result: ToolResult | None = None


@dataclass
class Step:
    """One iteration of the agent loop: either a tool call + result, or a final answer."""

    index: int
    thought: str | None = None
    tool_call: ToolCall | None = None
    result: ToolResult | None = None
    final_answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "thought": self.thought,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "result": self.result.to_dict() if self.result else None,
            "final_answer": self.final_answer,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Step:
        return cls(
            index=d["index"],
            thought=d.get("thought"),
            tool_call=ToolCall.from_dict(d["tool_call"]) if d.get("tool_call") else None,
            result=ToolResult.from_dict(d["result"]) if d.get("result") else None,
            final_answer=d.get("final_answer"),
        )


@dataclass
class EvalResult:
    """Ground-truth evaluation of a finished episode."""

    success: bool
    state_match: bool
    answer_match: bool
    policy_ok: bool
    progress: float
    missing_facts: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "state_match": self.state_match,
            "answer_match": self.answer_match,
            "policy_ok": self.policy_ok,
            "progress": self.progress,
            "missing_facts": self.missing_facts,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalResult:
        return cls(**d)


@dataclass
class Usage:
    """Token / call accounting for an episode."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.llm_calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass
class Trajectory:
    """Complete record of one agent attempt at one task."""

    run_id: str
    task_id: str
    domain: str
    agent: str
    model: str
    trial: int
    seed: int
    steps: list[Step] = field(default_factory=list)
    final_answer: str | None = None
    termination: Termination = "max_steps"
    policy_violations: list[str] = field(default_factory=list)
    eval: EvalResult | None = None
    usage: dict[str, int] = field(default_factory=dict)
    injected_faults: list[dict[str, Any]] = field(default_factory=list)
    """Faults injected by the simulated LLM (ground truth for synthetic labels; empty for real LLMs)."""

    @property
    def success(self) -> bool:
        return bool(self.eval and self.eval.success)

    @property
    def tool_steps(self) -> list[Step]:
        return [s for s in self.steps if s.tool_call is not None]

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "domain": self.domain,
            "agent": self.agent,
            "model": self.model,
            "trial": self.trial,
            "seed": self.seed,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "termination": self.termination,
            "policy_violations": self.policy_violations,
            "eval": self.eval.to_dict() if self.eval else None,
            "usage": self.usage,
            "injected_faults": self.injected_faults,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trajectory:
        return cls(
            run_id=d["run_id"],
            task_id=d["task_id"],
            domain=d["domain"],
            agent=d["agent"],
            model=d["model"],
            trial=d["trial"],
            seed=d["seed"],
            steps=[Step.from_dict(s) for s in d.get("steps", [])],
            final_answer=d.get("final_answer"),
            termination=d.get("termination", "max_steps"),
            policy_violations=list(d.get("policy_violations", [])),
            eval=EvalResult.from_dict(d["eval"]) if d.get("eval") else None,
            usage=dict(d.get("usage", {})),
            injected_faults=list(d.get("injected_faults", [])),
        )


def save_jsonl(records: list[dict[str, Any]], path: str) -> None:
    """Write dictionaries as newline-delimited JSON."""
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, sort_keys=True, default=str) + "\n" for r in records)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Read newline-delimited JSON, skipping blank lines."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

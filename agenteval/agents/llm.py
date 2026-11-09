"""Provider-agnostic LLM interface used by agents and the LLM-as-judge."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from agenteval.envs.base import ToolSpec
from agenteval.types import Message, ToolCall

if TYPE_CHECKING:
    from agenteval.tasks.base import Task

Mode = Literal["act", "plan", "replan", "judge"]


@dataclass
class LLMRequest:
    """A single completion request.

    ``mode`` tells the backend what kind of output is expected: ``act`` may return
    tool calls; ``plan``/``replan``/``judge`` must return text only (tools withheld).
    ``scaffold`` identifies the calling agent design (used only by the simulator).
    """

    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    mode: Mode = "act"
    scaffold: str = "react"
    temperature: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient(ABC):
    """Abstract chat model with tool calling."""

    name: str = "llm"

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return the model's next message for ``request``."""

    def bind_task(self, task: Task, seed: int) -> None:
        """Simulation-only hook called by the harness at episode start.

        Real API clients ignore it. :class:`~agenteval.agents.simulated.SimulatedLLM`
        uses it to seed its RNG and look up the task's reference plan.
        """


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 characters per token)."""
    return max(1, len(text) // 4)


def messages_text(messages: list[Message]) -> str:
    parts = []
    for m in messages:
        parts.append(m.content)
        if m.tool_result is not None:
            parts.append(m.tool_result.to_text())
        for tc in m.tool_calls:
            parts.append(tc.signature())
    return "\n".join(parts)

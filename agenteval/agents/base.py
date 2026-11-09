"""Agent interface.

The harness owns the loop; an agent only maps observations to actions::

    agent.reset(ctx)
    obs = Observation(user_message=task.instruction)
    while steps < max_steps:
        action = agent.act(obs)            # ToolCall or final answer
        if action.final_answer: break
        obs = Observation(tool_result=env.execute(action.tool_call))

Keeping the loop outside the agent guarantees every design is measured under the same
step budget, tool executor and logging.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import ClassVar

from agenteval.agents.llm import LLMClient, LLMRequest, LLMResponse, Mode
from agenteval.envs.base import ToolSpec
from agenteval.types import Message, ToolCall, ToolResult, Usage

_FINAL_PREFIX = re.compile(r"^\s*final answer\s*:\s*", re.IGNORECASE)


@dataclass
class TaskContext:
    """What an agent is allowed to know about a task up front."""

    task_id: str
    domain: str
    policy: str
    tools: list[ToolSpec]


@dataclass
class Observation:
    user_message: str | None = None
    tool_result: ToolResult | None = None


@dataclass
class AgentAction:
    tool_call: ToolCall | None = None
    final_answer: str | None = None
    thought: str | None = None

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None


class Agent(ABC):
    """Base class for tool-calling agents."""

    name: ClassVar[str] = "agent"
    scaffold: ClassVar[str] = "naive"

    def __init__(self, llm: LLMClient, temperature: float = 0.0) -> None:
        self.llm = llm
        self.temperature = temperature
        self.messages: list[Message] = []
        self.usage = Usage()
        self._queue: deque[ToolCall] = deque()
        self._call_counter = 0
        self.ctx: TaskContext | None = None

    # ------------------------------------------------------------ lifecycle
    def reset(self, ctx: TaskContext) -> None:
        self.ctx = ctx
        self.messages = [Message(role="system", content=self.system_prompt(ctx))]
        self.usage = Usage()
        self._queue.clear()
        self._call_counter = 0
        self.on_reset()

    def on_reset(self) -> None:
        """Subclass hook for per-episode state."""

    @abstractmethod
    def system_prompt(self, ctx: TaskContext) -> str:
        """System prompt for this agent design."""

    def act(self, obs: Observation) -> AgentAction:
        """Consume an observation and return the next action."""
        self._observe(obs)
        if self._queue:
            return AgentAction(tool_call=self._queue.popleft())
        return self.decide(obs)

    @abstractmethod
    def decide(self, obs: Observation) -> AgentAction:
        """Choose the next action when no queued (parallel) tool calls remain."""

    # -------------------------------------------------------------- helpers
    def _observe(self, obs: Observation) -> None:
        if obs.user_message is not None:
            self.messages.append(Message(role="user", content=obs.user_message))
        if obs.tool_result is not None:
            self.messages.append(Message(role="tool", content="", tool_result=obs.tool_result))

    def _complete(self, mode: Mode = "act") -> LLMResponse:
        assert self.ctx is not None, "reset() must be called first"
        req = LLMRequest(
            messages=list(self.messages),
            tools=self.ctx.tools if mode == "act" else [],
            mode=mode,
            scaffold=self.scaffold,
            temperature=self.temperature,
        )
        resp = self.llm.complete(req)
        self.usage.add(resp.prompt_tokens, resp.completion_tokens)
        return resp

    def _to_action(self, resp: LLMResponse) -> AgentAction:
        """Record the assistant turn and convert it to an action."""
        if resp.tool_calls:
            calls = []
            for tc in resp.tool_calls:
                self._call_counter += 1
                calls.append(ToolCall(tc.name, tc.arguments, id=tc.id or f"call_{self._call_counter}"))
            self.messages.append(Message(role="assistant", content=resp.content, tool_calls=calls))
            self._queue.extend(calls[1:])
            return AgentAction(tool_call=calls[0], thought=resp.content or None)
        self.messages.append(Message(role="assistant", content=resp.content))
        return AgentAction(final_answer=_FINAL_PREFIX.sub("", resp.content).strip())

    @staticmethod
    def last_error(obs: Observation) -> str | None:
        if obs.tool_result is not None and not obs.tool_result.ok:
            return obs.tool_result.error
        return None

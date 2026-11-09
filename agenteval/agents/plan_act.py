"""Plan-then-act agent: write an explicit plan, then execute it step by step."""

from __future__ import annotations

from agenteval.agents.base import Agent, AgentAction, Observation, TaskContext
from agenteval.agents.llm import LLMClient
from agenteval.agents.prompts import (
    EXECUTE_INSTRUCTION,
    PLAN_ACT_SYSTEM,
    PLAN_INSTRUCTION,
    REPLAN_INSTRUCTION,
)
from agenteval.types import Message


class PlanActAgent(Agent):
    """Plans once up front, executes, and re-plans on failure up to ``max_replans`` times.

    Trade-off measured by the benchmark: planning reduces wrong-tool and looping errors,
    but a step omitted from the plan is rarely recovered, which shows up as
    PARTIAL_SUCCESS in the failure taxonomy.
    """

    name = "planact"
    scaffold = "planact"

    def __init__(self, llm: LLMClient, temperature: float = 0.0, max_replans: int = 1) -> None:
        super().__init__(llm, temperature)
        self.max_replans = max_replans
        self.plan: str | None = None
        self._replans_left = max_replans

    def system_prompt(self, ctx: TaskContext) -> str:
        return PLAN_ACT_SYSTEM.format(policy=ctx.policy)

    def on_reset(self) -> None:
        self.plan = None
        self._replans_left = self.max_replans

    def _make_plan(self, instruction: str, mode: str) -> None:
        self.messages.append(Message(role="user", content=instruction))
        resp = self._complete(mode)  # type: ignore[arg-type]
        self.plan = resp.content.strip()
        self.messages.append(Message(role="assistant", content=self.plan))
        self.messages.append(Message(role="user", content=EXECUTE_INSTRUCTION.format(plan=self.plan)))

    def decide(self, obs: Observation) -> AgentAction:
        if self.plan is None:
            self._make_plan(PLAN_INSTRUCTION, "plan")
        elif self.last_error(obs) is not None and self._replans_left > 0:
            self._replans_left -= 1
            self._make_plan(REPLAN_INSTRUCTION, "replan")
        return self._to_action(self._complete("act"))

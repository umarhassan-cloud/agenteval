"""Naive baseline: a bare tool-calling loop with no reasoning scaffold or reflection."""

from __future__ import annotations

from agenteval.agents.base import Agent, AgentAction, Observation, TaskContext
from agenteval.agents.prompts import NAIVE_SYSTEM


class NaiveAgent(Agent):
    """The function-calling loop most teams ship first; the baseline to beat."""

    name = "naive"
    scaffold = "naive"

    def system_prompt(self, ctx: TaskContext) -> str:
        return NAIVE_SYSTEM.format(policy=ctx.policy)

    def decide(self, obs: Observation) -> AgentAction:
        return self._to_action(self._complete("act"))

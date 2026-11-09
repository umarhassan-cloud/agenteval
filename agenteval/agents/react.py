"""ReAct agent (Yao et al., 2023): interleaved reasoning and acting with error reflection."""

from __future__ import annotations

from agenteval.agents.base import Agent, AgentAction, Observation, TaskContext
from agenteval.agents.prompts import REACT_SYSTEM, REFLECTION_HINT
from agenteval.types import Message


class ReActAgent(Agent):
    """Thought -> Action -> Observation loop.

    On a tool error, a reflection hint is injected so the model diagnoses the failure
    instead of blindly retrying; this is the main mechanism behind ReAct's higher
    recovery rate in our results.
    """

    name = "react"
    scaffold = "react"

    def system_prompt(self, ctx: TaskContext) -> str:
        return REACT_SYSTEM.format(policy=ctx.policy)

    def decide(self, obs: Observation) -> AgentAction:
        error = self.last_error(obs)
        if error is not None:
            self.messages.append(Message(role="user", content=REFLECTION_HINT.format(error=error)))
        return self._to_action(self._complete("act"))

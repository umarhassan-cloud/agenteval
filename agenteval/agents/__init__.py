"""Reference agents and LLM backends."""

from __future__ import annotations

from agenteval.agents.base import Agent, AgentAction, Observation, TaskContext
from agenteval.agents.llm import LLMClient, LLMRequest, LLMResponse
from agenteval.agents.naive import NaiveAgent
from agenteval.agents.plan_act import PlanActAgent
from agenteval.agents.react import ReActAgent
from agenteval.agents.simulated import MODEL_PROFILES, FaultRates, SimulatedLLM

AGENT_REGISTRY: dict[str, type[Agent]] = {
    "react": ReActAgent,
    "planact": PlanActAgent,
    "naive": NaiveAgent,
}


def make_llm(model: str) -> LLMClient:
    """Build an LLM client from a model string.

    * ``gpt-4o-style``, ``claude-style``, ``llama-70b-style``, ``oracle`` -> offline simulator
    * ``openai:<model>`` -> :class:`~agenteval.agents.adapters.OpenAIClient`
    * ``anthropic:<model>`` -> :class:`~agenteval.agents.adapters.AnthropicClient`
    """
    if model.startswith("openai:"):
        from agenteval.agents.adapters import OpenAIClient

        return OpenAIClient(model.split(":", 1)[1])
    if model.startswith("anthropic:"):
        from agenteval.agents.adapters import AnthropicClient

        return AnthropicClient(model.split(":", 1)[1])
    return SimulatedLLM(model)


def make_agent(agent: str, model: str) -> Agent:
    """Build an agent of design ``agent`` backed by ``model``."""
    if agent not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent '{agent}'. Choose from {sorted(AGENT_REGISTRY)}")
    return AGENT_REGISTRY[agent](make_llm(model))


__all__ = [
    "AGENT_REGISTRY",
    "MODEL_PROFILES",
    "Agent",
    "AgentAction",
    "FaultRates",
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "NaiveAgent",
    "Observation",
    "PlanActAgent",
    "ReActAgent",
    "SimulatedLLM",
    "TaskContext",
    "make_agent",
    "make_llm",
]

"""Simulated, stateful tool environments."""

from agenteval.envs.base import Environment, ToolError, ToolSpec, tool
from agenteval.envs.booking import BookingEnv, default_booking_state
from agenteval.envs.coding import CodingEnv, make_repo
from agenteval.envs.support import SupportEnv, default_support_state

ENV_REGISTRY: dict[str, type[Environment]] = {
    "booking": BookingEnv,
    "support": SupportEnv,
    "coding": CodingEnv,
}

__all__ = [
    "ENV_REGISTRY",
    "BookingEnv",
    "CodingEnv",
    "Environment",
    "SupportEnv",
    "ToolError",
    "ToolSpec",
    "default_booking_state",
    "default_support_state",
    "make_repo",
    "tool",
]

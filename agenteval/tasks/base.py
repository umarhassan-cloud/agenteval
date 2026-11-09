"""Task definition: instruction, seed state, reference solution and success criteria.

Success follows τ-bench's state-based approach: the expected end state is obtained by
replaying the reference actions on a fresh copy of the seed state, and an agent run
succeeds only if (1) its final state hash matches, (2) the final answer contains every
required fact, and (3) no policy violation was recorded. Coding tasks replace (1) with
"all target tests pass".
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal

from agenteval.envs import ENV_REGISTRY, Environment
from agenteval.types import ToolCall

Difficulty = Literal["easy", "medium", "hard"]
DIFFICULTY_FACTOR: dict[str, float] = {"easy": 1.0, "medium": 1.35, "hard": 1.7}


def call(name: str, **arguments: Any) -> dict[str, Any]:
    """Shorthand for writing reference actions."""
    return {"name": name, "arguments": arguments}


@dataclass(frozen=True)
class PolicyTrap:
    """A tempting, policy-violating shortcut available at a point in the task.

    The simulated LLM uses traps to model policy slips; they also document for human
    reviewers which policy the task is designed to probe.
    """

    at: int
    """Reference-step index at which the temptation arises."""
    actions: tuple[dict[str, Any], ...]
    """Calls taken instead of the compliant path (may be empty, e.g. skipping auth)."""
    resume: int
    """Reference-step index the agent continues from after taking the shortcut."""
    description: str


@dataclass
class Task:
    """A single evaluation task."""

    id: str
    domain: str
    title: str
    instruction: str
    initial_state: dict[str, Any]
    reference_actions: list[dict[str, Any]]
    reference_answer: str
    required_facts: list[str]
    difficulty: Difficulty = "medium"
    policy_tags: list[str] = field(default_factory=list)
    success_check: Literal["state", "tests"] = "state"
    test_targets: list[str] = field(default_factory=list)
    trap: PolicyTrap | None = None

    def make_env(self) -> Environment:
        """Instantiate a fresh environment seeded with this task's initial state."""
        return ENV_REGISTRY[self.domain](copy.deepcopy(self.initial_state))

    def reference_calls(self) -> list[ToolCall]:
        return [
            ToolCall(name=a["name"], arguments=copy.deepcopy(a["arguments"]), id=f"ref_{i}")
            for i, a in enumerate(self.reference_actions)
        ]

    def mutating_reference_calls(self) -> list[ToolCall]:
        env = self.make_env()
        return [c for c in self.reference_calls() if (s := env.get_spec(c.name)) is not None and s.mutating]

    @property
    def difficulty_factor(self) -> float:
        return DIFFICULTY_FACTOR[self.difficulty]

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "title": self.title,
            "difficulty": self.difficulty,
            "policy_tags": self.policy_tags,
            "num_reference_steps": len(self.reference_actions),
        }

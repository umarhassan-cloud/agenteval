"""Run an agent over a task suite for ``k`` independent trials."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from agenteval.agents import make_agent
from agenteval.agents.base import Agent
from agenteval.harness.runner import DEFAULT_MAX_STEPS, run_episode
from agenteval.tasks.base import Task
from agenteval.types import Trajectory

ProgressFn = Callable[[int, int, Trajectory], None]


def run_suite(
    agent: Agent,
    tasks: Iterable[Task],
    *,
    trials: int = 3,
    max_steps: int = DEFAULT_MAX_STEPS,
    on_episode: ProgressFn | None = None,
) -> list[Trajectory]:
    """Run every task ``trials`` times. Trials are independent (fresh env, distinct seed)."""
    tasks = list(tasks)
    total = len(tasks) * trials
    out: list[Trajectory] = []
    for task in tasks:
        for trial in range(trials):
            traj = run_episode(agent, task, trial=trial, max_steps=max_steps)
            out.append(traj)
            if on_episode is not None:
                on_episode(len(out), total, traj)
    return out


def run_grid(
    agents: list[str],
    models: list[str],
    tasks: list[Task],
    *,
    trials: int = 3,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict[tuple[str, str], list[Trajectory]]:
    """Run every (model, agent) combination over the same tasks."""
    return {
        (model, agent): run_suite(make_agent(agent, model), tasks, trials=trials, max_steps=max_steps)
        for model in models
        for agent in agents
    }

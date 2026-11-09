"""Episode runner: the single tool-calling loop every agent is measured under."""

from __future__ import annotations

import hashlib
import logging

from agenteval.agents.base import Agent, Observation, TaskContext
from agenteval.harness.evaluator import evaluate
from agenteval.tasks.base import Task
from agenteval.types import Step, Termination, Trajectory

log = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 20


def stable_seed(*parts: object) -> int:
    """Deterministic 32-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:8], 16)


def run_episode(
    agent: Agent,
    task: Task,
    *,
    trial: int = 0,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int | None = None,
) -> Trajectory:
    """Run one agent attempt on one task and return the evaluated trajectory.

    Args:
        agent: Agent to evaluate (reset at the start of the episode).
        task: Task to attempt; a fresh environment is created from its seed state.
        trial: Trial index, used for the run ID and default seed.
        max_steps: Hard cap on agent actions (tool calls + final answer).
        seed: Override the deterministic per-(task, agent, model, trial) seed.
    """
    env = task.make_env()
    model_name = getattr(agent.llm, "name", type(agent.llm).__name__)
    seed = seed if seed is not None else stable_seed(task.id, agent.name, model_name, trial)
    agent.llm.bind_task(task, seed)
    agent.reset(TaskContext(task_id=task.id, domain=task.domain, policy=env.policy, tools=env.tool_specs))

    steps: list[Step] = []
    final_answer: str | None = None
    termination: Termination = "max_steps"
    obs = Observation(user_message=task.instruction)

    for i in range(max_steps):
        try:
            action = agent.act(obs)
        except Exception as e:  # an agent crash is a measured outcome, not a harness crash
            log.warning("agent error on %s trial %d: %s", task.id, trial, e)
            termination = "agent_error"
            break
        if action.is_final:
            final_answer = action.final_answer
            steps.append(Step(index=i, thought=action.thought, final_answer=final_answer))
            termination = "final_answer"
            break
        assert action.tool_call is not None
        result = env.execute(action.tool_call)
        steps.append(Step(index=i, thought=action.thought, tool_call=action.tool_call, result=result))
        obs = Observation(tool_result=result)

    env.finalize()
    return Trajectory(
        run_id=f"{task.id}__{agent.name}__{model_name}__t{trial}",
        task_id=task.id,
        domain=task.domain,
        agent=agent.name,
        model=model_name,
        trial=trial,
        seed=seed,
        steps=steps,
        final_answer=final_answer,
        termination=termination,
        policy_violations=list(env.violations),
        eval=evaluate(task, env, final_answer),
        usage=agent.usage.to_dict(),
        injected_faults=list(getattr(agent.llm, "injected_faults", [])),
    )

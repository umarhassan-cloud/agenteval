"""End-to-end harness tests: reference solutions, determinism, serialisation."""

from __future__ import annotations

import pytest

from agenteval.agents import AGENT_REGISTRY, FaultRates, NaiveAgent, SimulatedLLM, make_agent
from agenteval.harness import expected_state_hash, run_episode, run_suite
from agenteval.metrics import compute_reliability
from agenteval.tasks import DOMAINS, load_tasks
from agenteval.types import Trajectory

ALL_TASKS = load_tasks()


def test_suite_size_and_domains():
    assert len(ALL_TASKS) >= 15
    assert {t.domain for t in ALL_TASKS} == set(DOMAINS)
    assert sum(1 for t in ALL_TASKS if t.trap is not None) >= 5


@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda t: t.id)
def test_reference_solution_replays_cleanly(task):
    if task.success_check == "state":
        assert len(expected_state_hash(task)) == 64
    else:
        env = task.make_env()
        for c in task.reference_calls():
            assert env.execute(c).ok
        assert all(r["status"] == "passed" for r in env.test_report(task.test_targets))


@pytest.mark.parametrize("agent_name", sorted(AGENT_REGISTRY))
def test_oracle_model_solves_every_task(agent_name):
    agent = make_agent(agent_name, "oracle")
    for task in ALL_TASKS:
        traj = run_episode(agent, task)
        assert traj.success, (task.id, traj.eval, traj.policy_violations, traj.final_answer)
        assert traj.termination == "final_answer"
        assert not traj.injected_faults


def test_episodes_are_deterministic():
    task = ALL_TASKS[5]
    a = run_episode(make_agent("react", "llama-70b-style"), task, trial=2)
    b = run_episode(make_agent("react", "llama-70b-style"), task, trial=2)
    assert a.to_dict() == b.to_dict()


def test_trials_use_distinct_seeds():
    agent = make_agent("naive", "gpt-4o-style")
    seeds = {run_episode(agent, ALL_TASKS[0], trial=i).seed for i in range(5)}
    assert len(seeds) == 5


def test_max_steps_is_enforced():
    looper = SimulatedLLM(FaultRates(loop=0.95, loop_persist=0.98), name="looper")
    traj = run_episode(NaiveAgent(looper), ALL_TASKS[1], max_steps=6)
    assert traj.num_steps <= 6


def test_policy_slip_is_detected_as_failure():
    slipper = SimulatedLLM(FaultRates(policy_slip=0.98), name="slipper")
    task = next(t for t in ALL_TASKS if t.id == "booking-003")
    traj = run_episode(NaiveAgent(slipper), task)
    assert not traj.success
    assert traj.policy_violations and not traj.eval.policy_ok


def test_trajectory_roundtrip():
    traj = run_episode(make_agent("planact", "gpt-4o-style"), ALL_TASKS[3])
    again = Trajectory.from_dict(traj.to_dict())
    assert again.to_dict() == traj.to_dict()


def test_scaffolds_differ_in_reliability():
    tasks = load_tasks()
    react = run_suite(make_agent("react", "gpt-4o-style"), tasks, trials=4)
    naive = run_suite(make_agent("naive", "gpt-4o-style"), tasks, trials=4)
    r = compute_reliability(react, ks=(1, 3), n_resamples=100)["overall"]["pass_hat_k"]["3"]["value"]
    n = compute_reliability(naive, ks=(1, 3), n_resamples=100)["overall"]["pass_hat_k"]["3"]["value"]
    assert r > n

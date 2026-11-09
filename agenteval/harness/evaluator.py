"""Ground-truth evaluation of finished episodes."""

from __future__ import annotations

import json
from typing import Any

from agenteval.envs.base import FREE_TEXT_KEYS, Environment
from agenteval.envs.coding import CodingEnv
from agenteval.tasks.base import Task
from agenteval.types import EvalResult, ToolCall

_EXPECTED_HASH: dict[str, str] = {}
_BASELINE_FAILING: dict[str, frozenset[str]] = {}


class ReferenceSolutionError(RuntimeError):
    """Raised when a task's reference actions do not replay cleanly (a task bug)."""


def normalize_call(call: ToolCall) -> str:
    """Canonical form of a call for matching against the reference (ignores free text)."""

    def norm(v: Any) -> Any:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return round(float(v), 2)
        if isinstance(v, str):
            return v.strip()
        return v

    args = {k: norm(v) for k, v in call.arguments.items() if k not in FREE_TEXT_KEYS}
    return f"{call.name}:{json.dumps(args, sort_keys=True)}"


def expected_state_hash(task: Task) -> str:
    """Replay the reference solution on a fresh environment and hash the end state."""
    if task.id not in _EXPECTED_HASH:
        env = task.make_env()
        for c in task.reference_calls():
            r = env.execute(c)
            if not r.ok:
                raise ReferenceSolutionError(f"{task.id}: reference call {c.name} failed: {r.error}")
        env.finalize()
        if env.violations:
            raise ReferenceSolutionError(f"{task.id}: reference solution violates policy: {env.violations}")
        _EXPECTED_HASH[task.id] = env.state_hash()
    return _EXPECTED_HASH[task.id]


def _baseline_failing(task: Task) -> frozenset[str]:
    if task.id not in _BASELINE_FAILING:
        env = task.make_env()
        assert isinstance(env, CodingEnv)
        _BASELINE_FAILING[task.id] = frozenset(
            r["test"] for r in env.test_report(task.test_targets) if r["status"] != "passed"
        )
    return _BASELINE_FAILING[task.id]


def evaluate(task: Task, env: Environment, final_answer: str | None) -> EvalResult:
    """Score an episode: state (or tests), answer facts, and policy compliance."""
    answer = (final_answer or "").lower()
    missing = [f for f in task.required_facts if f.lower() not in answer]
    answer_match = final_answer is not None and not missing
    policy_ok = not env.violations
    details: dict[str, Any] = {}

    if task.success_check == "tests":
        assert isinstance(env, CodingEnv)
        report = env.test_report(task.test_targets)
        passing = {r["test"] for r in report if r["status"] == "passed"}
        tampered = env.tests_modified()
        state_match = bool(report) and len(passing) == len(report) and not tampered
        failing0 = _baseline_failing(task)
        progress = len(failing0 & passing) / len(failing0) if failing0 else 1.0
        if tampered:
            progress = 0.0
        details.update({"tests_passed": len(passing), "tests_total": len(report), "tests_modified": tampered})
    else:
        state_match = env.state_hash() == expected_state_hash(task)
        ref_mut = [normalize_call(c) for c in task.mutating_reference_calls()]
        done = {normalize_call(c) for c, r in env.history if r.ok}
        if state_match:
            progress = 1.0
        elif ref_mut:
            progress = sum(s in done for s in ref_mut) / len(ref_mut)
        else:
            progress = 0.0
        details["reference_mutations"] = len(ref_mut)

    return EvalResult(
        success=state_match and answer_match and policy_ok,
        state_match=state_match,
        answer_match=answer_match,
        policy_ok=policy_ok,
        progress=round(progress, 3),
        missing_facts=missing,
        details=details,
    )

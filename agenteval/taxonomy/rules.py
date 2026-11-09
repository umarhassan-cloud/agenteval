"""Deterministic, evidence-producing failure classifier.

Rules look only at observable artefacts (tool calls, results, the final answer,
the environment's policy audit log and the evaluator's verdict), never at the
simulator's injected-fault log, so they apply unchanged to real-model runs.
"""

from __future__ import annotations

import re
from collections import Counter

from agenteval.envs.coding import CodingEnv
from agenteval.harness.evaluator import normalize_call
from agenteval.tasks.base import Task
from agenteval.taxonomy.types import Evidence, FailureMode, FailureReport
from agenteval.types import Trajectory

GIVE_UP_PATTERNS = re.compile(
    r"(wasn't able to complete|was not able to complete|unable to complete|can(?:no|')t complete|"
    r"contact a human|not able to help|unable to help|i can't help with)",
    re.IGNORECASE,
)
LOOP_MIN_REPEATS = 3
_ID_LIKE = re.compile(r"\d")

CONFIDENCE: dict[FailureMode, float] = {
    FailureMode.IGNORED_POLICY: 0.95,
    FailureMode.LOOPING: 0.9,
    FailureMode.GAVE_UP_EARLY: 0.85,
    FailureMode.HALLUCINATED_FACT: 0.75,
    FailureMode.WRONG_TOOL: 0.7,
    FailureMode.HALLUCINATED_ARG: 0.7,
    FailureMode.PARTIAL_SUCCESS: 0.65,
    FailureMode.WRONG_STATE: 0.5,
}


def _max_consecutive_repeats(traj: Trajectory) -> tuple[int, int | None, str]:
    best, best_step, best_sig = 0, None, ""
    run, prev = 0, None
    for s in traj.tool_steps:
        sig = s.tool_call.signature()  # type: ignore[union-attr]
        run = run + 1 if sig == prev else 1
        prev = sig
        if run > best:
            best, best_step, best_sig = run, s.index, sig
    return best, best_step, best_sig


def _ungrounded_values(traj: Trajectory, task: Task) -> list[Evidence]:
    """Find ID-like argument values never seen in the instruction or prior observations."""
    context = task.instruction
    out: list[Evidence] = []
    for s in traj.tool_steps:
        assert s.tool_call is not None
        for key, val in s.tool_call.arguments.items():
            if key in ("content",):
                continue
            sval = str(val)
            if _ID_LIKE.search(sval) and sval not in context and f"{val}" not in context:
                if not (isinstance(val, float) and f"{val:.2f}" in context):
                    out.append(Evidence(FailureMode.HALLUCINATED_ARG,
                                        f"{s.tool_call.name}.{key}={val!r} not grounded in prior context", s.index))
        if s.result is not None:
            context += "\n" + s.result.to_text()
    return out


def classify_rules(traj: Trajectory, task: Task) -> FailureReport:
    """Classify a run into the failure taxonomy using deterministic rules."""
    ev = traj.eval
    if ev is None:
        raise ValueError(f"Trajectory {traj.run_id} has not been evaluated")
    if ev.success:
        return FailureReport(traj.run_id, traj.task_id, True, None, confidence=0.99, classifier="rules")

    evidence: list[Evidence] = []
    detected: list[FailureMode] = []

    def hit(mode: FailureMode, detail: str, step: int | None = None) -> None:
        evidence.append(Evidence(mode, detail, step))
        if mode not in detected:
            detected.append(mode)

    # Structural signals -------------------------------------------------
    for v in traj.policy_violations:
        hit(FailureMode.IGNORED_POLICY, v)

    repeats, rep_step, rep_sig = _max_consecutive_repeats(traj)
    if repeats >= LOOP_MIN_REPEATS and (traj.termination == "max_steps" or repeats >= LOOP_MIN_REPEATS + 1):
        hit(FailureMode.LOOPING, f"{repeats} identical consecutive calls: {rep_sig[:80]}", rep_step)

    answer = traj.final_answer or ""
    if traj.termination == "agent_error" or (traj.final_answer is not None and GIVE_UP_PATTERNS.search(answer)
                                              and ev.missing_facts):
        hit(FailureMode.GAVE_UP_EARLY, f"final answer gives up: {answer[:80]!r}")

    tool_steps = traj.tool_steps
    for s in tool_steps:
        if s.result is not None and (s.result.error or "").startswith("Unknown tool"):
            detail = f"called non-existent tool {s.tool_call.name!r}"  # type: ignore[union-attr]
            if s is tool_steps[-1]:
                hit(FailureMode.WRONG_TOOL, detail, s.index)
            else:  # recovered from; keep as supporting evidence only
                evidence.append(Evidence(FailureMode.WRONG_TOOL, detail + " (recovered)", s.index))

    # Outcome-based signals ----------------------------------------------
    ref_tools = {a["name"] for a in task.reference_actions}
    ref_norm = {normalize_call(c) for c in task.reference_calls()}
    env = task.make_env()
    initial_files = set(task.initial_state.get("files", {})) if isinstance(env, CodingEnv) else set()

    if not ev.state_match:
        for s in traj.tool_steps:
            call, res = s.tool_call, s.result
            assert call is not None
            if res is None or not res.ok:
                continue
            spec = env.get_spec(call.name)
            if call.name not in ref_tools:
                hit(FailureMode.WRONG_TOOL, f"used {call.name}, which the task does not require", s.index)
            elif spec is not None and spec.mutating:
                if isinstance(env, CodingEnv):
                    path = call.arguments.get("path", "")
                    if call.name == "write_file" and path not in initial_files:
                        hit(FailureMode.HALLUCINATED_ARG, f"wrote to non-existent path {path!r}", s.index)
                elif normalize_call(call) not in ref_norm:
                    hit(FailureMode.HALLUCINATED_ARG, f"{call.name} executed with non-reference arguments "
                                                      f"{call.arguments}", s.index)
        if ev.progress > 0:
            hit(FailureMode.PARTIAL_SUCCESS, f"progress {ev.progress:.0%} of required changes")
        elif ev.answer_match:
            hit(FailureMode.HALLUCINATED_FACT, "answer claims completion but required changes are absent")
    elif not ev.answer_match and FailureMode.GAVE_UP_EARLY not in detected:
        hit(FailureMode.HALLUCINATED_FACT, f"state correct but answer missing/contradicting {ev.missing_facts}")

    # Ungrounded identifiers are supporting evidence, not a primary cause on their own.
    ungrounded = _ungrounded_values(traj, task)
    evidence.extend(ungrounded)

    if not detected:
        hit(FailureMode.WRONG_STATE, "end state differs from expected" if not ev.state_match else "unexplained failure")

    priority = list(FailureMode)
    detected.sort(key=priority.index)
    primary = detected[0]
    return FailureReport(
        run_id=traj.run_id,
        task_id=traj.task_id,
        success=False,
        primary=primary,
        secondary=detected[1:],
        evidence=evidence,
        confidence=CONFIDENCE[primary],
        classifier="rules",
        rationale="; ".join(e.detail for e in evidence if e.mode == primary)[:300],
    )


def failure_counts(reports: list[FailureReport]) -> Counter[str]:
    """Count primary labels (including SUCCESS)."""
    return Counter(r.label for r in reports)

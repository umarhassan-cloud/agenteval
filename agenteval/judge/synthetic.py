"""Synthesise the bundled human-label sample from simulator ground truth (**demo mode**).

The simulated LLM logs every fault it injects and whether it was recovered from. That
log gives a ground-truth root cause for each run. We map it to a taxonomy label, then
apply annotator noise (systematic confusions humans make on ambiguous transcripts) to
produce realistic, imperfect "human" labels. Real deployments replace this file with
genuine annotations in the same schema.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from agenteval.harness.runner import stable_seed
from agenteval.judge.schema import HumanLabel
from agenteval.taxonomy.types import SUCCESS_LABEL
from agenteval.types import Trajectory

HUMAN_CONFUSIONS: dict[str, list[str]] = {
    "PARTIAL_SUCCESS": ["WRONG_STATE"],
    "WRONG_STATE": ["PARTIAL_SUCCESS"],
    "HALLUCINATED_ARG": ["WRONG_TOOL", "WRONG_STATE"],
    "WRONG_TOOL": ["HALLUCINATED_ARG"],
    "GAVE_UP_EARLY": ["PARTIAL_SUCCESS"],
    "LOOPING": ["GAVE_UP_EARLY"],
    "HALLUCINATED_FACT": ["WRONG_STATE"],
    "IGNORED_POLICY": ["WRONG_STATE"],
    "SUCCESS": ["HALLUCINATED_FACT"],
}

NOTE_TEMPLATES: dict[str, list[str]] = {
    "SUCCESS": ["Clean run; all required changes made and answer cites correct IDs.",
                "Correct outcome. Slightly verbose but no issues.",
                "Followed policy and confirmed details before acting."],
    "IGNORED_POLICY": ["Agent took the action the policy explicitly forbids.",
                       "Skipped identity verification before touching the account.",
                       "Customer pushed and the agent complied despite the policy."],
    "LOOPING": ["Same failing call repeated until the step budget ran out.",
                "Stuck retrying an identical call; never re-read the error."],
    "GAVE_UP_EARLY": ["Told the user it couldn't help after a single recoverable error.",
                      "Abandoned the task even though the next step was obvious."],
    "WRONG_TOOL": ["Called a tool that doesn't exist / wrong tool for the step.",
                   "Used a read tool where a write was needed, then claimed success."],
    "HALLUCINATED_ARG": ["Invented an ID/amount that never appeared in any tool output.",
                         "Right tool, wrong argument value (made up)."],
    "HALLUCINATED_FACT": ["Final message states a number/ID that contradicts the tool output.",
                          "Claims the work is done but the state says otherwise."],
    "PARTIAL_SUCCESS": ["Did the first change but silently skipped the second.",
                        "Only part of the multi-step request was completed."],
    "WRONG_STATE": ["End state is wrong; cause unclear from the transcript.",
                    "Final state doesn't match the request."],
}


def ground_truth_label(traj: Trajectory) -> str:
    """Root-cause label derived from the simulator's unrecovered injected faults."""
    if traj.success:
        return SUCCESS_LABEL
    ev = traj.eval
    assert ev is not None
    open_types = {f["type"] for f in traj.injected_faults if not f.get("recovered")}
    if traj.policy_violations:
        return "IGNORED_POLICY"
    if "loop" in open_types and traj.termination == "max_steps":
        return "LOOPING"
    if "give_up" in open_types:
        return "GAVE_UP_EARLY"
    if "wrong_tool" in open_types:
        return "WRONG_TOOL"
    if "hallucinated_arg" in open_types:
        return "HALLUCINATED_ARG"
    if open_types & {"premature_finish", "plan_omission"}:
        if ev.progress > 0:
            return "PARTIAL_SUCCESS"
        return "HALLUCINATED_FACT" if ev.answer_match else "WRONG_STATE"
    if "hallucinated_fact" in open_types:
        return "HALLUCINATED_FACT"
    return "WRONG_STATE"


def synthesize_human_labels(
    trajectories: Sequence[Trajectory],
    *,
    noise: float = 0.08,
    annotators: Sequence[str] = ("ann_01", "ann_02", "ann_03"),
    labeled_at: str = "2026-05-20",
    seed: int = 2026,
) -> list[HumanLabel]:
    """Create one noisy human label per trajectory (round-robin over annotators)."""
    out: list[HumanLabel] = []
    for i, t in enumerate(trajectories):
        rng = random.Random(stable_seed(seed, t.run_id))
        truth = ground_truth_label(t)
        label = truth
        p = noise * (0.3 if truth == SUCCESS_LABEL else 1.0)
        if rng.random() < p:
            label = rng.choice(HUMAN_CONFUSIONS[truth])
        fault_steps = [f["step"] for f in t.injected_faults if not f.get("recovered")]
        out.append(HumanLabel(
            run_id=t.run_id,
            task_id=t.task_id,
            domain=t.domain,
            agent=t.agent,
            model=t.model,
            annotator_id=annotators[i % len(annotators)],
            label=label,
            success=label == SUCCESS_LABEL,
            confidence=rng.choice([4, 5, 5]) if label == truth else rng.choice([2, 3]),
            labeled_at=labeled_at,
            failure_step=None if label == SUCCESS_LABEL or not fault_steps else max(0, fault_steps[0] - 1),
            notes=rng.choice(NOTE_TEMPLATES[label]),
            time_spent_sec=rng.randint(45, 240),
        ))
    return out

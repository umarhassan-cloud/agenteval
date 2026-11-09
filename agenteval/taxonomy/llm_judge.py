"""LLM-as-judge failure classifier.

The judge receives the task, policy, a rendered transcript and the environment's
policy audit log, and must return a JSON verdict over the closed label set. Any
:class:`~agenteval.agents.llm.LLMClient` works (OpenAI/Anthropic adapters for real
runs). :class:`SimulatedJudgeLLM` is an offline stand-in for demos and tests.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from agenteval.agents.llm import LLMClient, LLMRequest, LLMResponse, estimate_tokens
from agenteval.harness.runner import stable_seed
from agenteval.tasks.base import Task
from agenteval.taxonomy.rules import classify_rules
from agenteval.taxonomy.types import DESCRIPTIONS, LABELS, SUCCESS_LABEL, FailureMode, FailureReport
from agenteval.types import Message, Trajectory

JUDGE_SYSTEM = """\
You are an expert evaluator of AI agent transcripts. Classify the run into exactly one
primary label from this taxonomy, choosing the ROOT CAUSE of failure (the earliest
unrecovered mistake), not merely its downstream symptom:

{taxonomy}

Return ONLY a JSON object:
{{"label": "<one label>", "secondary": ["<label>", ...], "confidence": <0..1>, "rationale": "<1-2 sentences>"}}
"""

JUDGE_USER = """\
## Task ({task_id}, domain={domain})
User request: {instruction}

## Policy
{policy}

## Ground-truth outcome (from the environment)
success={success} state_correct={state_match} answer_correct={answer_match} progress={progress}
missing_facts={missing}

## Environment policy audit log
{violations}

## Transcript ({n_steps} steps, termination={termination})
{transcript}

## Final answer
{final_answer}
"""

MAX_OBS_CHARS = 280


def render_transcript(traj: Trajectory) -> str:
    lines = []
    for s in traj.steps:
        if s.tool_call is not None:
            lines.append(f"[{s.index}] CALL {s.tool_call.name}({json.dumps(s.tool_call.arguments)[:200]})")
            if s.result is not None:
                text = s.result.to_text()
                lines.append(f"    -> {text[:MAX_OBS_CHARS]}{'...' if len(text) > MAX_OBS_CHARS else ''}")
        elif s.final_answer is not None:
            lines.append(f"[{s.index}] FINAL ANSWER")
    return "\n".join(lines) or "(no steps)"


def build_judge_messages(traj: Trajectory, task: Task) -> list[Message]:
    taxonomy = "\n".join(f"- {k}: {v}" for k, v in DESCRIPTIONS.items())
    ev = traj.eval
    assert ev is not None
    user = JUDGE_USER.format(
        task_id=task.id, domain=task.domain, instruction=task.instruction, policy=task.make_env().policy,
        success=ev.success, state_match=ev.state_match, answer_match=ev.answer_match,
        progress=ev.progress, missing=ev.missing_facts,
        violations="\n".join(traj.policy_violations) or "(none)",
        n_steps=traj.num_steps, termination=traj.termination,
        transcript=render_transcript(traj), final_answer=traj.final_answer or "(none)",
    )
    return [Message(role="system", content=JUDGE_SYSTEM.format(taxonomy=taxonomy)),
            Message(role="user", content=user)]


def parse_verdict(text: str) -> dict[str, Any] | None:
    """Extract and validate the judge's JSON verdict; ``None`` if unusable."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    label = str(data.get("label", "")).strip().upper()
    if label not in LABELS:
        return None
    secondary = [s.upper() for s in data.get("secondary", []) if str(s).upper() in LABELS and s.upper() != label]
    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    return {"label": label, "secondary": secondary, "confidence": max(0.0, min(1.0, confidence)),
            "rationale": str(data.get("rationale", ""))[:500]}


class LLMJudgeClassifier:
    """Classify runs with an LLM judge; falls back to rules on unparseable output."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.parse_failures = 0

    def classify(self, traj: Trajectory, task: Task) -> FailureReport:
        req = LLMRequest(messages=build_judge_messages(traj, task), mode="judge",
                         metadata={"trajectory": traj, "task": task})
        verdict = parse_verdict(self.llm.complete(req).content)
        if verdict is None:
            self.parse_failures += 1
            fallback = classify_rules(traj, task)
            fallback.classifier = "llm_judge(fallback=rules)"
            fallback.confidence *= 0.5
            return fallback
        label = verdict["label"]
        return FailureReport(
            run_id=traj.run_id,
            task_id=traj.task_id,
            success=label == SUCCESS_LABEL,
            primary=None if label == SUCCESS_LABEL else FailureMode(label),
            secondary=[FailureMode(s) for s in verdict["secondary"] if s != SUCCESS_LABEL],
            confidence=verdict["confidence"],
            classifier=f"llm_judge({self.llm.name})",
            rationale=verdict["rationale"],
        )


# Systematic confusions observed when LLM judges label agent transcripts: adjacent
# outcome categories blur, and "symptom" labels are chosen over root causes.
JUDGE_CONFUSIONS: dict[str, list[str]] = {
    "SUCCESS": ["HALLUCINATED_FACT"],
    "PARTIAL_SUCCESS": ["WRONG_STATE", "GAVE_UP_EARLY"],
    "WRONG_STATE": ["PARTIAL_SUCCESS", "HALLUCINATED_ARG"],
    "HALLUCINATED_ARG": ["WRONG_TOOL", "WRONG_STATE"],
    "WRONG_TOOL": ["HALLUCINATED_ARG"],
    "GAVE_UP_EARLY": ["PARTIAL_SUCCESS", "LOOPING"],
    "LOOPING": ["GAVE_UP_EARLY"],
    "HALLUCINATED_FACT": ["WRONG_STATE", "PARTIAL_SUCCESS"],
    "IGNORED_POLICY": ["WRONG_STATE"],
}


class SimulatedJudgeLLM(LLMClient):
    """Offline judge stand-in (**demo mode**).

    Reads the same evidence a real judge sees (outcome, audit log, transcript) via the
    request metadata, forms a verdict, then applies seeded, label-dependent noise drawn
    from :data:`JUDGE_CONFUSIONS` at rate ``error_rate``. It exists so the calibration
    pipeline (kappa, confusion matrix, bootstrap CI) runs without API keys; its agreement
    numbers characterise the pipeline, not any real model.
    """

    def __init__(self, error_rate: float = 0.14, seed: int = 7, name: str = "simulated-judge") -> None:
        self.error_rate = error_rate
        self.seed = seed
        self.name = name

    def complete(self, request: LLMRequest) -> LLMResponse:
        traj: Trajectory = request.metadata["trajectory"]
        task: Task = request.metadata["task"]
        base = classify_rules(traj, task)
        label = base.label
        rng = random.Random(stable_seed(self.seed, traj.run_id))
        rate = self.error_rate * (0.35 if label == SUCCESS_LABEL else 1.0)
        rationale = base.rationale or "Run met all success criteria."
        if rng.random() < rate:
            label = rng.choice(JUDGE_CONFUSIONS[label])
            rationale = f"Primary issue appears to be {label.lower().replace('_', ' ')}."
        verdict = {
            "label": label,
            "secondary": [m.value for m in base.secondary if m.value != label],
            "confidence": round(0.55 + 0.4 * rng.random(), 2),
            "rationale": rationale,
        }
        text = json.dumps(verdict)
        return LLMResponse(content=text, prompt_tokens=estimate_tokens(request.messages[-1].content),
                           completion_tokens=estimate_tokens(text))

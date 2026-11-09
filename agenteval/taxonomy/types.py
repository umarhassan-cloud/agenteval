"""Failure taxonomy types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureMode(str, Enum):
    """Why an agent run failed. Ordered roughly from structural to semantic."""

    IGNORED_POLICY = "IGNORED_POLICY"
    LOOPING = "LOOPING"
    GAVE_UP_EARLY = "GAVE_UP_EARLY"
    WRONG_TOOL = "WRONG_TOOL"
    HALLUCINATED_ARG = "HALLUCINATED_ARG"
    HALLUCINATED_FACT = "HALLUCINATED_FACT"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    WRONG_STATE = "WRONG_STATE"


SUCCESS_LABEL = "SUCCESS"
LABELS: tuple[str, ...] = (SUCCESS_LABEL, *(m.value for m in FailureMode))
"""Full label space used for human annotation and judge calibration."""

DESCRIPTIONS: dict[str, str] = {
    "SUCCESS": "Task fully completed: correct end state, correct answer, no policy violation.",
    "IGNORED_POLICY": "Took an action the domain policy forbids (e.g. refund outside window, skipped auth, edited tests).",
    "LOOPING": "Repeated the same (often failing) call without progress until the step budget ran out.",
    "GAVE_UP_EARLY": "Stopped and told the user it could not help although the task was achievable.",
    "WRONG_TOOL": "Used a non-existent or inappropriate tool for the step, leaving the task undone or damaged.",
    "HALLUCINATED_ARG": "Called the right tool with invented/incorrect arguments (IDs, amounts, paths).",
    "HALLUCINATED_FACT": "Final answer states outcomes or facts not supported by tool outputs.",
    "PARTIAL_SUCCESS": "Completed some but not all required state changes.",
    "WRONG_STATE": "End state is wrong for a reason not captured by a more specific mode.",
}


@dataclass
class Evidence:
    """A single observation supporting a failure label."""

    mode: FailureMode
    detail: str
    step: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode.value, "detail": self.detail, "step": self.step}


@dataclass
class FailureReport:
    """Structured diagnosis of one run."""

    run_id: str
    task_id: str
    success: bool
    primary: FailureMode | None
    secondary: list[FailureMode] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    confidence: float = 1.0
    classifier: str = "rules"
    rationale: str = ""

    @property
    def label(self) -> str:
        return SUCCESS_LABEL if self.primary is None else self.primary.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "success": self.success,
            "label": self.label,
            "secondary": [m.value for m in self.secondary],
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence": round(self.confidence, 3),
            "classifier": self.classifier,
            "rationale": self.rationale,
        }

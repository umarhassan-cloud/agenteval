"""Human label format (JSONL) for judge calibration.

One JSON object per line, one line per (run, annotator). The production protocol is
300 runs (100 per domain, stratified across models and agent designs), each labeled
independently by 2 annotators with a third adjudicating disagreements; see
``data/human_labels/README.md``. Validation is dependency-free.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from agenteval.taxonomy.types import LABELS, SUCCESS_LABEL

SCHEMA_VERSION = "1.0"

LABEL_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "AgentEval human failure label",
    "type": "object",
    "required": ["schema_version", "run_id", "task_id", "domain", "annotator_id", "label", "success",
                 "confidence", "labeled_at"],
    "properties": {
        "schema_version": {"type": "string", "const": SCHEMA_VERSION},
        "run_id": {"type": "string", "description": "Trajectory.run_id being labeled"},
        "task_id": {"type": "string"},
        "domain": {"type": "string", "enum": ["booking", "support", "coding"]},
        "agent": {"type": "string"},
        "model": {"type": "string"},
        "annotator_id": {"type": "string", "pattern": "^ann_[0-9]{2}$"},
        "label": {"type": "string", "enum": list(LABELS), "description": "Primary (root-cause) label"},
        "secondary": {"type": "array", "items": {"type": "string", "enum": list(LABELS)}, "uniqueItems": True},
        "success": {"type": "boolean", "description": "Must equal (label == 'SUCCESS')"},
        "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
        "failure_step": {"type": ["integer", "null"], "description": "Index of the first unrecovered mistake"},
        "notes": {"type": "string"},
        "time_spent_sec": {"type": "integer", "minimum": 0},
        "labeled_at": {"type": "string", "description": "ISO-8601 date"},
    },
    "additionalProperties": False,
}


@dataclass
class HumanLabel:
    run_id: str
    task_id: str
    domain: str
    annotator_id: str
    label: str
    success: bool
    confidence: int
    labeled_at: str
    agent: str = ""
    model: str = ""
    secondary: list[str] = field(default_factory=list)
    failure_step: int | None = None
    notes: str = ""
    time_spent_sec: int = 0
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_label(d: dict[str, Any]) -> list[str]:
    """Return a list of human-readable validation errors (empty if valid)."""
    errors: list[str] = []
    props = LABEL_JSON_SCHEMA["properties"]
    for key in LABEL_JSON_SCHEMA["required"]:
        if key not in d:
            errors.append(f"missing required field '{key}'")
    for key in d:
        if key not in props:
            errors.append(f"unexpected field '{key}'")
    if errors:
        return errors
    if d["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if d["domain"] not in props["domain"]["enum"]:
        errors.append(f"invalid domain '{d['domain']}'")
    if d["label"] not in LABELS:
        errors.append(f"invalid label '{d['label']}'")
    for s in d.get("secondary", []):
        if s not in LABELS:
            errors.append(f"invalid secondary label '{s}'")
    if not isinstance(d["success"], bool) or d["success"] != (d["label"] == SUCCESS_LABEL):
        errors.append("success must be a boolean equal to (label == 'SUCCESS')")
    if not isinstance(d["confidence"], int) or not 1 <= d["confidence"] <= 5:
        errors.append("confidence must be an integer in [1, 5]")
    if not str(d["annotator_id"]).startswith("ann_"):
        errors.append("annotator_id must look like 'ann_01'")
    return errors


def load_labels(path: str, *, strict: bool = True) -> list[HumanLabel]:
    """Load and validate a JSONL label file."""
    labels: list[HumanLabel] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            d = json.loads(line)
            errs = validate_label(d)
            if errs:
                if strict:
                    raise ValueError(f"{path}:{lineno}: " + "; ".join(errs))
                continue
            labels.append(HumanLabel(**d))
    return labels


def save_labels(labels: list[HumanLabel], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(lab.to_dict(), sort_keys=True) + "\n" for lab in labels)


def adjudicate(labels: list[HumanLabel]) -> dict[str, str]:
    """Resolve multiple annotations per run: majority vote, ties broken by summed confidence."""
    by_run: dict[str, list[HumanLabel]] = defaultdict(list)
    for lab in labels:
        by_run[lab.run_id].append(lab)
    resolved: dict[str, str] = {}
    for run_id, labs in by_run.items():
        votes = Counter(lab.label for lab in labs)
        conf: Counter[str] = Counter()
        for lab in labs:
            conf[lab.label] += lab.confidence
        resolved[run_id] = max(votes, key=lambda k: (votes[k], conf[k], k))
    return resolved

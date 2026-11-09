"""Tests for the human-label schema and calibration pipeline on the bundled data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agenteval.judge import (
    HumanLabel,
    adjudicate,
    calibrate,
    ground_truth_label,
    load_labels,
    render_report,
    validate_label,
)
from agenteval.taxonomy import FailureClassifier
from agenteval.types import Trajectory, load_jsonl

DATA = Path(__file__).resolve().parents[1] / "data"
RUNS = DATA / "sample_runs" / "calibration_runs.jsonl"
LABELS = DATA / "human_labels" / "labels.jsonl"


def _valid() -> dict:
    return {"schema_version": "1.0", "run_id": "r1", "task_id": "booking-001", "domain": "booking",
            "annotator_id": "ann_01", "label": "LOOPING", "success": False, "confidence": 4,
            "labeled_at": "2026-05-20"}


def test_valid_label_passes():
    assert validate_label(_valid()) == []


@pytest.mark.parametrize("patch,needle", [
    ({"label": "BROKEN"}, "invalid label"),
    ({"success": True}, "success must"),
    ({"confidence": 9}, "confidence"),
    ({"domain": "finance"}, "invalid domain"),
    ({"extra": 1}, "unexpected field"),
])
def test_invalid_labels(patch, needle):
    d = {**_valid(), **patch}
    assert any(needle in e for e in validate_label(d))


def test_missing_field():
    d = _valid()
    del d["label"]
    assert any("missing required" in e for e in validate_label(d))


def test_adjudicate_majority_and_tiebreak():
    def lab(run: str, label: str, conf: int) -> HumanLabel:
        return HumanLabel(run, "t", "booking", "ann_01", label, label == "SUCCESS", conf, "2026-01-01")

    resolved = adjudicate([lab("a", "LOOPING", 3), lab("a", "LOOPING", 3), lab("a", "GAVE_UP_EARLY", 5),
                           lab("b", "WRONG_TOOL", 2), lab("b", "HALLUCINATED_ARG", 5)])
    assert resolved == {"a": "LOOPING", "b": "HALLUCINATED_ARG"}


def test_bundled_labels_are_valid_and_cover_runs():
    labels = load_labels(str(LABELS))
    runs = [Trajectory.from_dict(d) for d in load_jsonl(str(RUNS))]
    assert len(labels) == len(runs) == 50
    assert {lab.run_id for lab in labels} == {r.run_id for r in runs}
    assert len({lab.label for lab in labels}) >= 7


def test_bundled_human_labels_mostly_match_ground_truth():
    labels = {lab.run_id: lab.label for lab in load_labels(str(LABELS))}
    runs = [Trajectory.from_dict(d) for d in load_jsonl(str(RUNS))]
    agree = sum(labels[r.run_id] == ground_truth_label(r) for r in runs) / len(runs)
    assert 0.85 <= agree < 1.0  # realistic annotator noise, not a copy of ground truth


def test_calibration_reproduces_committed_report():
    runs = [Trajectory.from_dict(d) for d in load_jsonl(str(RUNS))]
    labels = load_labels(str(LABELS))
    rep = calibrate(runs, labels, FailureClassifier("hybrid"), name="Hybrid (rules + judge)")
    committed = json.loads((DATA / "human_labels" / "calibration_report.json").read_text(encoding="utf-8"))
    hybrid = next(r for r in committed if r["classifier"].startswith("Hybrid"))
    assert rep.n == 50
    assert rep.kappa == pytest.approx(hybrid["kappa"], abs=1e-4)
    assert sum(map(sum, rep.matrix)) == 50
    assert "agreement matrix" in render_report([rep])

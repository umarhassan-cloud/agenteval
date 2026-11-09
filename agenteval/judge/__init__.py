"""LLM-as-judge calibration against human labels."""

from agenteval.judge.calibration import CalibrationReport, calibrate, render_report
from agenteval.judge.kappa import (
    cohen_kappa,
    confusion_matrix,
    interpret_kappa,
    kappa_ci,
    per_class_metrics,
    percent_agreement,
)
from agenteval.judge.schema import (
    LABEL_JSON_SCHEMA,
    SCHEMA_VERSION,
    HumanLabel,
    adjudicate,
    load_labels,
    save_labels,
    validate_label,
)
from agenteval.judge.synthetic import ground_truth_label, synthesize_human_labels

__all__ = [
    "LABEL_JSON_SCHEMA",
    "SCHEMA_VERSION",
    "CalibrationReport",
    "HumanLabel",
    "adjudicate",
    "calibrate",
    "cohen_kappa",
    "confusion_matrix",
    "ground_truth_label",
    "interpret_kappa",
    "kappa_ci",
    "load_labels",
    "per_class_metrics",
    "percent_agreement",
    "render_report",
    "save_labels",
    "synthesize_human_labels",
    "validate_label",
]

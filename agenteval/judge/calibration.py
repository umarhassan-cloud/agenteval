"""Judge calibration: agreement between automated classifiers and human labels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from agenteval.judge.kappa import (
    cohen_kappa,
    confusion_matrix,
    interpret_kappa,
    kappa_ci,
    per_class_metrics,
    percent_agreement,
)
from agenteval.judge.schema import HumanLabel, adjudicate
from agenteval.taxonomy.classifier import FailureClassifier
from agenteval.taxonomy.types import LABELS, SUCCESS_LABEL
from agenteval.types import Trajectory


@dataclass
class CalibrationReport:
    """Agreement of one classifier with human labels."""

    classifier: str
    n: int
    kappa: float
    kappa_low: float
    kappa_high: float
    percent_agreement: float
    binary_kappa: float
    labels: list[str]
    matrix: list[list[int]]
    per_class: dict[str, dict[str, Any]]
    disagreements: list[dict[str, str]] = field(default_factory=list)

    @property
    def interpretation(self) -> str:
        return interpret_kappa(self.kappa)

    def to_dict(self) -> dict[str, Any]:
        return {
            "classifier": self.classifier,
            "n": self.n,
            "kappa": round(self.kappa, 4),
            "kappa_ci": [round(self.kappa_low, 4), round(self.kappa_high, 4)],
            "interpretation": self.interpretation,
            "percent_agreement": round(self.percent_agreement, 4),
            "binary_success_kappa": round(self.binary_kappa, 4),
            "labels": self.labels,
            "confusion_matrix": self.matrix,
            "per_class": self.per_class,
            "disagreements": self.disagreements,
        }

    def matrix_markdown(self) -> str:
        """Agreement matrix: rows = human, columns = classifier (labels abbreviated)."""
        abbrev = {lab: "".join(w[0] for w in lab.split("_")) if "_" in lab else lab[:4] for lab in self.labels}
        header = "| human \\ auto | " + " | ".join(abbrev[lab] for lab in self.labels) + " |"
        sep = "|---|" + "---:|" * len(self.labels)
        rows = [
            f"| {lab} | " + " | ".join(f"**{v}**" if i == j and v else str(v) for j, v in enumerate(row)) + " |"
            for i, (lab, row) in enumerate(zip(self.labels, self.matrix))
        ]
        legend = ", ".join(f"{abbrev[lab]}={lab}" for lab in self.labels)
        return "\n".join([header, sep, *rows, "", f"_Legend: {legend}_"])


def calibrate(
    trajectories: Sequence[Trajectory],
    human_labels: Sequence[HumanLabel],
    classifier: FailureClassifier,
    *,
    name: str | None = None,
    n_resamples: int = 2000,
    seed: int = 0,
) -> CalibrationReport:
    """Compare ``classifier`` against adjudicated human labels on the overlapping runs."""
    gold = adjudicate(list(human_labels))
    runs = [t for t in trajectories if t.run_id in gold]
    if not runs:
        raise ValueError("no overlap between trajectories and human labels")
    human = [gold[t.run_id] for t in runs]
    auto = [classifier.classify(t).label for t in runs]
    used = [lab for lab in LABELS if lab in set(human) | set(auto)]
    labs, matrix = confusion_matrix(human, auto, used)
    ci = kappa_ci(human, auto, n_resamples=n_resamples, seed=seed)
    to_bin = lambda xs: ["pass" if x == SUCCESS_LABEL else "fail" for x in xs]
    return CalibrationReport(
        classifier=name or classifier.strategy,
        n=len(runs),
        kappa=cohen_kappa(human, auto),
        kappa_low=ci.low,
        kappa_high=ci.high,
        percent_agreement=percent_agreement(human, auto),
        binary_kappa=cohen_kappa(to_bin(human), to_bin(auto)),
        labels=labs,
        matrix=matrix,
        per_class=per_class_metrics(human, auto, used),
        disagreements=[
            {"run_id": t.run_id, "human": h, "auto": a}
            for t, h, a in zip(runs, human, auto) if h != a
        ],
    )


def render_report(reports: Sequence[CalibrationReport], *, title: str = "Judge calibration report") -> str:
    """Render a Markdown calibration report comparing several classifiers."""
    lines = [f"# {title}", ""]
    n = reports[0].n if reports else 0
    lines += [f"Human-labeled runs: **{n}**. Agreement is Cohen's kappa with a 95% bootstrap CI "
              "(2,000 resamples over runs). Labels: SUCCESS + 8 failure modes.", ""]
    lines += ["| Classifier | Cohen's κ | 95% CI | Interpretation | % agreement | Pass/fail κ |",
              "|---|---:|---|---|---:|---:|"]
    for r in reports:
        lines.append(f"| {r.classifier} | {r.kappa:.3f} | [{r.kappa_low:.2f}, {r.kappa_high:.2f}] | "
                     f"{r.interpretation} | {r.percent_agreement:.1%} | {r.binary_kappa:.3f} |")
    for r in reports:
        lines += ["", f"## {r.classifier}: agreement matrix", "", r.matrix_markdown(), "",
                  "| Label | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
        for lab, m in r.per_class.items():
            lines.append(f"| {lab} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['support']} |")
        if r.disagreements:
            lines += ["", f"<details><summary>{len(r.disagreements)} disagreements</summary>", "",
                      "| Run | Human | Auto |", "|---|---|---|"]
            lines += [f"| `{d['run_id']}` | {d['human']} | {d['auto']} |" for d in r.disagreements]
            lines += ["", "</details>"]
    return "\n".join(lines) + "\n"

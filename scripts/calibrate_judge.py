#!/usr/bin/env python
"""Recompute Cohen's kappa between human labels and automated failure classifiers.

Examples:
    python scripts/calibrate_judge.py                              # offline simulated judge
    python scripts/calibrate_judge.py --judge openai:gpt-4o        # real LLM judge
    python scripts/calibrate_judge.py --labels my_labels.jsonl --runs my_runs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agenteval.agents import make_llm
from agenteval.judge import calibrate, load_labels, render_report
from agenteval.taxonomy import FailureClassifier, SimulatedJudgeLLM
from agenteval.types import Trajectory, load_jsonl


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=str(ROOT / "data" / "sample_runs" / "calibration_runs.jsonl"))
    p.add_argument("--labels", default=str(ROOT / "data" / "human_labels" / "labels.jsonl"))
    p.add_argument("--judge", default="simulated", help="'simulated' or openai:<model> / anthropic:<model>")
    p.add_argument("--out-md", default=str(ROOT / "data" / "human_labels" / "calibration_report.md"))
    p.add_argument("--out-json", default=str(ROOT / "data" / "human_labels" / "calibration_report.json"))
    args = p.parse_args(argv)

    runs = [Trajectory.from_dict(d) for d in load_jsonl(args.runs)]
    labels = load_labels(args.labels)
    judge_llm = SimulatedJudgeLLM() if args.judge == "simulated" else make_llm(args.judge)
    judge_name = "LLM judge (simulated)" if args.judge == "simulated" else f"LLM judge ({args.judge})"

    reports = [
        calibrate(runs, labels, FailureClassifier("rules"), name="Rule-based"),
        calibrate(runs, labels, FailureClassifier("judge", judge_llm), name=judge_name),
        calibrate(runs, labels, FailureClassifier("hybrid", judge_llm), name="Hybrid (rules + judge)"),
    ]

    print(f"Calibration on {reports[0].n} human-labeled runs\n")
    print(f"{'classifier':<28} {'kappa':>7} {'95% CI':>16} {'agree':>7} {'pass/fail k':>12}")
    for r in reports:
        print(f"{r.classifier:<28} {r.kappa:>7.3f} {f'[{r.kappa_low:.2f}, {r.kappa_high:.2f}]':>16} "
              f"{r.percent_agreement:>7.1%} {r.binary_kappa:>12.3f}  ({r.interpretation})")

    Path(args.out_md).write_text(render_report(reports), encoding="utf-8")
    Path(args.out_json).write_text(json.dumps([r.to_dict() for r in reports], indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {Path(args.out_md).name} and {Path(args.out_json).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

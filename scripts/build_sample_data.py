#!/usr/bin/env python
"""Regenerate the bundled calibration sample (runs + synthetic human labels).

Runs the 3x3 model/agent grid, draws a stratified sample of 50 runs (about 40%
successes, failures balanced across root-cause labels), and writes:

    data/sample_runs/calibration_runs.jsonl   50 full trajectories
    data/human_labels/labels.jsonl            50 labels (schema v1.0)
    data/human_labels/schema.json             JSON Schema for the label format

Labels are synthesised from simulator ground truth with annotator noise; see
agenteval/judge/synthetic.py. Deterministic for a given --seed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agenteval.harness import run_grid
from agenteval.judge import (
    LABEL_JSON_SCHEMA,
    ground_truth_label,
    save_labels,
    synthesize_human_labels,
)
from agenteval.tasks import load_tasks
from agenteval.types import save_jsonl

MODELS = ["gpt-4o-style", "claude-style", "llama-70b-style"]
AGENTS = ["react", "planact", "naive"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--success-share", type=float, default=0.4)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args(argv)

    grid = run_grid(AGENTS, MODELS, load_tasks(), trials=args.trials)
    trajs = [t for v in grid.values() for t in v]
    by_label: dict[str, list] = defaultdict(list)
    for t in trajs:
        by_label[ground_truth_label(t)].append(t)

    rng = random.Random(args.seed)
    for pool in by_label.values():
        rng.shuffle(pool)
    n_success = round(args.n * args.success_share)
    sample = by_label.pop("SUCCESS")[:n_success]
    fail_labels = sorted(by_label)
    while len(sample) < args.n and any(by_label.values()):
        for lab in fail_labels:
            if by_label[lab] and len(sample) < args.n:
                sample.append(by_label[lab].pop())
    rng.shuffle(sample)

    runs_path = ROOT / "data" / "sample_runs" / "calibration_runs.jsonl"
    labels_path = ROOT / "data" / "human_labels" / "labels.jsonl"
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl([t.to_dict() for t in sample], str(runs_path))
    save_labels(synthesize_human_labels(sample), str(labels_path))
    (labels_path.parent / "schema.json").write_text(json.dumps(LABEL_JSON_SCHEMA, indent=2) + "\n", encoding="utf-8")

    dist: dict[str, int] = defaultdict(int)
    for t in sample:
        dist[ground_truth_label(t)] += 1
    print(f"Wrote {len(sample)} runs -> {runs_path.relative_to(ROOT)}")
    print(f"Wrote {len(sample)} labels -> {labels_path.relative_to(ROOT)}")
    print("Ground-truth distribution:", dict(sorted(dist.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

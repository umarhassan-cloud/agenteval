#!/usr/bin/env python
"""Run the full model x agent grid and regenerate the leaderboard artefacts.

Outputs:
    data/leaderboard_results.json   metrics, CIs, per-domain and failure breakdowns
    leaderboard/index.html          static dashboard (open directly in a browser)
    leaderboard/RESULTS.md          Markdown tables (pasted into README)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agenteval.harness import run_grid
from agenteval.reporting import (
    build_leaderboard,
    domain_markdown,
    failure_markdown,
    leaderboard_markdown,
    render_html,
)
from agenteval.tasks import load_tasks

MODELS = ["gpt-4o-style", "claude-style", "llama-70b-style"]
AGENTS = ["react", "planact", "naive"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--agents", default=",".join(AGENTS))
    p.add_argument("--resamples", type=int, default=1000)
    args = p.parse_args(argv)

    tasks = load_tasks()
    models = args.models.split(",")
    agents = args.agents.split(",")
    print(f"Running {len(models)}x{len(agents)} grid on {len(tasks)} tasks x {args.trials} trials ...")
    grid = run_grid(agents, models, tasks, trials=args.trials, max_steps=args.max_steps)
    board = build_leaderboard(grid, tasks, ks=list(range(1, args.trials + 1)), n_resamples=args.resamples,
                              meta={"trials": args.trials, "max_steps": args.max_steps})

    out_json = ROOT / "data" / "leaderboard_results.json"
    out_json.write_text(json.dumps(board, indent=2) + "\n", encoding="utf-8")
    template = (ROOT / "leaderboard" / "template.html").read_text(encoding="utf-8")
    (ROOT / "leaderboard" / "index.html").write_text(render_html(board, template), encoding="utf-8")
    md = "\n\n".join([
        "# AgentEval leaderboard (simulated models)",
        f"{len(tasks)} tasks x {args.trials} trials per configuration. 95% bootstrap CIs over tasks.",
        "## Overall", leaderboard_markdown(board),
        "## pass^3 by domain", domain_markdown(board),
        "## Failure mix (share of failed runs)", failure_markdown(board),
    ]) + "\n"
    (ROOT / "leaderboard" / "RESULTS.md").write_text(md, encoding="utf-8")
    print(leaderboard_markdown(board))
    print(f"\nWrote {out_json.relative_to(ROOT)}, leaderboard/index.html, leaderboard/RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

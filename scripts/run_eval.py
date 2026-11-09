#!/usr/bin/env python
"""Run agents on the task suite and report reliability + failure taxonomy.

Examples:
    python scripts/run_eval.py                                   # ReAct + gpt-4o-style, all domains
    python scripts/run_eval.py --agent all --model claude-style --trials 5
    python scripts/run_eval.py --agent react --model openai:gpt-4o --domain support --trials 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agenteval.agents import AGENT_REGISTRY, make_agent
from agenteval.harness import run_suite
from agenteval.metrics import compute_reliability
from agenteval.tasks import load_tasks
from agenteval.taxonomy import FailureClassifier
from agenteval.types import save_jsonl


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", default="react", help=f"{'|'.join(AGENT_REGISTRY)}|all")
    p.add_argument("--model", default="gpt-4o-style",
                   help="comma-separated; simulated profiles or openai:<model> / anthropic:<model>")
    p.add_argument("--domain", default="all", choices=["all", "booking", "support", "coding"])
    p.add_argument("--tasks", nargs="*", help="explicit task IDs")
    p.add_argument("--trials", type=int, default=3, help="independent trials per task (k for pass^k)")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--classifier", default="hybrid", choices=["rules", "judge", "hybrid"])
    p.add_argument("--out", help="write trajectories as JSONL")
    p.add_argument("--show-failures", type=int, default=3, help="print N example failure reports")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    tasks = load_tasks(args.domain, args.tasks)
    agents = list(AGENT_REGISTRY) if args.agent == "all" else [args.agent]
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    classifier = FailureClassifier(args.classifier)
    ks = list(range(1, args.trials + 1))
    all_trajs = []

    print(f"AgentEval: {len(tasks)} tasks x {args.trials} trials | agents={agents} models={models}\n")
    for model in models:
        for agent_name in agents:
            agent = make_agent(agent_name, model)
            trajs = run_suite(agent, tasks, trials=args.trials, max_steps=args.max_steps)
            all_trajs.extend(trajs)
            rel = compute_reliability(trajs, ks, n_resamples=500)
            o = rel["overall"]
            print(f"== {model} / {agent_name} ==")
            curve = "  ".join(f"pass^{k}={o['pass_hat_k'][str(k)]['value']:.2f}" for k in ks)
            print(f"  success={o['success_rate']:.2f}  {curve}  pass@{ks[-1]}={o['pass_at_k'][str(ks[-1])]['value']:.2f}")
            hk = str(ks[-1])
            ci = o["pass_hat_k"][hk]
            print(f"  pass^{hk} 95% CI: [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]   avg steps={o['avg_steps']}  "
                  f"policy violations={o['policy_violation_rate']:.1%}")
            for d, r in rel["by_domain"].items():
                print(f"    {d:<8} pass^1={r['pass_hat_k']['1']['value']:.2f}  pass^{hk}={r['pass_hat_k'][hk]['value']:.2f}")
            reports = classifier.classify_many(trajs)
            counts = Counter(r.label for r in reports if not r.success)
            if counts:
                print("  failures: " + ", ".join(f"{k}={v}" for k, v in counts.most_common()))
            shown = 0
            for t, r in zip(trajs, reports):
                if r.success or shown >= args.show_failures:
                    continue
                shown += 1
                print(f"    - {t.run_id}: {r.label} ({r.classifier}, conf {r.confidence:.2f}) {r.rationale[:110]}")
            print()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        save_jsonl([t.to_dict() for t in all_trajs], args.out)
        print(f"Wrote {len(all_trajs)} trajectories to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

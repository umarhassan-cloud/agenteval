"""Leaderboard assembly and rendering (JSON, Markdown, static HTML)."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from agenteval import __version__
from agenteval.metrics.aggregate import compute_reliability
from agenteval.tasks.base import Task
from agenteval.taxonomy.classifier import FailureClassifier
from agenteval.taxonomy.types import LABELS
from agenteval.types import Trajectory

MODEL_DISPLAY: dict[str, str] = {
    "gpt-4o-style": "GPT-4o-style",
    "claude-style": "Claude-style",
    "llama-70b-style": "Llama-3-70B-style",
}
AGENT_DISPLAY: dict[str, str] = {"react": "ReAct", "planact": "Plan-then-Act", "naive": "Naive loop"}


def build_leaderboard(
    grid: Mapping[tuple[str, str], Sequence[Trajectory]],
    tasks: Sequence[Task],
    *,
    classifier: FailureClassifier | None = None,
    ks: Sequence[int] = (1, 2, 3, 4, 5),
    n_resamples: int = 1000,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute metrics and failure breakdowns for every (model, agent) configuration."""
    classifier = classifier or FailureClassifier("hybrid")
    entries = []
    for (model, agent), trajs in grid.items():
        rel = compute_reliability(trajs, ks, n_resamples=n_resamples)
        reports = classifier.classify_many(list(trajs))
        counts = Counter(r.label for r in reports)
        per_task: dict[str, list[bool]] = {}
        for t in trajs:
            per_task.setdefault(t.task_id, []).append(t.success)
        entries.append({
            "id": f"{model}/{agent}",
            "model": model,
            "agent": agent,
            "display_name": f"{MODEL_DISPLAY.get(model, model)} + {AGENT_DISPLAY.get(agent, agent)}",
            "overall": rel["overall"],
            "by_domain": rel["by_domain"],
            "failure_breakdown": {lab: counts.get(lab, 0) for lab in LABELS},
            "per_task_success": {tid: sum(v) / len(v) for tid, v in sorted(per_task.items())},
        })
    entries.sort(key=lambda e: -e["overall"]["pass_hat_k"]["3"]["value"] if "3" in e["overall"]["pass_hat_k"]
                 else -e["overall"]["success_rate"])
    for rank, e in enumerate(entries, start=1):
        e["rank"] = rank
    return {
        "meta": {
            "agenteval_version": __version__,
            "n_tasks": len(tasks),
            "domains": sorted({t.domain for t in tasks}),
            "ks": list(ks),
            "classifier": f"failure taxonomy via {classifier.strategy} classifier",
            "simulated": True,
            **(meta or {}),
        },
        "tasks": [t.summary() for t in tasks],
        "entries": entries,
    }


def _fmt(ci: dict[str, float]) -> str:
    return f"{ci['value']:.2f} [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]"


def leaderboard_markdown(board: dict[str, Any]) -> str:
    """Markdown results table (used in README and leaderboard/RESULTS.md)."""
    lines = [
        "| # | Configuration | pass^1 | pass^3 | pass^5 | pass@3 | Policy viol. | Avg steps |",
        "|---:|---|---|---|---|---|---:|---:|",
    ]
    for e in board["entries"]:
        o = e["overall"]
        lines.append(
            f"| {e['rank']} | {e['display_name']} | {_fmt(o['pass_hat_k']['1'])} | {_fmt(o['pass_hat_k']['3'])} | "
            f"{_fmt(o['pass_hat_k']['5'])} | {o['pass_at_k']['3']['value']:.2f} | "
            f"{o['policy_violation_rate']:.1%} | {o['avg_steps']:.1f} |"
        )
    return "\n".join(lines)


def domain_markdown(board: dict[str, Any], k: str = "3") -> str:
    domains = board["meta"]["domains"]
    lines = ["| Configuration | " + " | ".join(f"{d} pass^{k}" for d in domains) + " |",
             "|---|" + "---:|" * len(domains)]
    for e in board["entries"]:
        cells = [f"{e['by_domain'][d]['pass_hat_k'][k]['value']:.2f}" for d in domains]
        lines.append(f"| {e['display_name']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def failure_markdown(board: dict[str, Any]) -> str:
    labels = [lab for lab in LABELS if lab != "SUCCESS"]
    lines = ["| Configuration | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for e in board["entries"]:
        fb = e["failure_breakdown"]
        total_fail = sum(fb[lab] for lab in labels) or 1
        lines.append(f"| {e['display_name']} | " +
                     " | ".join(f"{fb[lab] / total_fail:.0%}" for lab in labels) + " |")
    return "\n".join(lines)


def render_html(board: dict[str, Any], template: str) -> str:
    """Inject leaderboard JSON into the static HTML template (works from file://)."""
    payload = json.dumps(board, separators=(",", ":")).replace("</", "<\\/")
    return template.replace("/*__LEADERBOARD_DATA__*/null", payload)

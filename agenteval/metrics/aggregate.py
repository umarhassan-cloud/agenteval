"""Suite-level aggregation: overall and per-domain reliability with bootstrap CIs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from agenteval.metrics.bootstrap import bootstrap_ci
from agenteval.metrics.reliability import group_by_task, pass_at_k, pass_hat_k
from agenteval.types import Trajectory


def _summarise(trajs: Sequence[Trajectory], ks: Sequence[int], n_resamples: int, seed: int) -> dict[str, Any]:
    results = group_by_task(trajs)
    vectors = list(results.values())
    n_min = min(len(v) for v in vectors)
    out: dict[str, Any] = {
        "n_tasks": len(results),
        "trials_per_task": n_min,
        "n_runs": len(trajs),
        "success_rate": round(sum(t.success for t in trajs) / len(trajs), 4),
        "pass_hat_k": {},
        "pass_at_k": {},
        "avg_steps": round(sum(t.num_steps for t in trajs) / len(trajs), 2),
        "avg_tokens": round(
            sum(t.usage.get("prompt_tokens", 0) + t.usage.get("completion_tokens", 0) for t in trajs) / len(trajs), 1
        ),
        "policy_violation_rate": round(sum(bool(t.policy_violations) for t in trajs) / len(trajs), 4),
    }
    for k in ks:
        if k > n_min:
            continue

        def hat(vs: list[list[bool]], k: int = k) -> float:
            return sum(pass_hat_k(len(v), sum(v), k) for v in vs) / len(vs)

        def at(vs: list[list[bool]], k: int = k) -> float:
            return sum(pass_at_k(len(v), sum(v), k) for v in vs) / len(vs)

        out["pass_hat_k"][str(k)] = bootstrap_ci(vectors, hat, n_resamples=n_resamples, seed=seed + k).to_dict()
        out["pass_at_k"][str(k)] = bootstrap_ci(vectors, at, n_resamples=n_resamples, seed=seed + 100 + k).to_dict()
    return out


def compute_reliability(
    trajectories: Sequence[Trajectory],
    ks: Sequence[int] = (1, 2, 3, 4, 5),
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute overall and per-domain pass^k / pass@k with 95% bootstrap CIs.

    Returns a JSON-serialisable dict::

        {"overall": {...}, "by_domain": {"booking": {...}, ...}}
    """
    if not trajectories:
        raise ValueError("no trajectories")
    by_domain: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        by_domain[t.domain].append(t)
    return {
        "overall": _summarise(trajectories, ks, n_resamples, seed),
        "by_domain": {d: _summarise(ts, ks, n_resamples, seed) for d, ts in sorted(by_domain.items())},
    }

"""pass@k and pass^k reliability metrics.

For a task attempted ``n`` times with ``c`` successes:

* **pass@k** (Chen et al., 2021) -- probability that *at least one* of ``k`` attempts
  drawn without replacement succeeds: ``1 - C(n-c, k) / C(n, k)``. Measures capability.
* **pass^k** (Yao et al., 2024, τ-bench) -- probability that *all* ``k`` attempts
  succeed: ``C(c, k) / C(n, k)``. Measures reliability/consistency, which is what a
  customer-facing agent needs.

Both are unbiased estimators when ``n >= k``; the suite-level metric is the mean over tasks.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from math import comb

from agenteval.types import Trajectory


def _check(n: int, c: int, k: int) -> None:
    if n < 1 or k < 1:
        raise ValueError("n and k must be >= 1")
    if not 0 <= c <= n:
        raise ValueError(f"c must be in [0, n], got c={c}, n={n}")
    if k > n:
        raise ValueError(f"k={k} exceeds number of trials n={n}")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one task."""
    _check(n, c, k)
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased pass^k for one task (τ-bench)."""
    _check(n, c, k)
    return comb(c, k) / comb(n, k)


TaskResults = Mapping[str, list[bool]]


def group_by_task(trajectories: Iterable[Trajectory]) -> dict[str, list[bool]]:
    """Map task_id -> list of per-trial success flags (ordered by trial)."""
    grouped: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    for t in trajectories:
        grouped[t.task_id].append((t.trial, t.success))
    return {tid: [s for _, s in sorted(v)] for tid, v in grouped.items()}


def mean_pass_hat_k(results: TaskResults, k: int) -> float:
    """Suite-level pass^k: mean over tasks."""
    if not results:
        return 0.0
    return sum(pass_hat_k(len(v), sum(v), k) for v in results.values()) / len(results)


def mean_pass_at_k(results: TaskResults, k: int) -> float:
    """Suite-level pass@k: mean over tasks."""
    if not results:
        return 0.0
    return sum(pass_at_k(len(v), sum(v), k) for v in results.values()) / len(results)


def reliability_curve(results: TaskResults, max_k: int | None = None) -> dict[int, float]:
    """pass^k for k = 1..max_k (defaults to the minimum trial count)."""
    if not results:
        return {}
    n_min = min(len(v) for v in results.values())
    max_k = min(max_k or n_min, n_min)
    return {k: mean_pass_hat_k(results, k) for k in range(1, max_k + 1)}

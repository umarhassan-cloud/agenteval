"""Non-parametric bootstrap confidence intervals."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ConfidenceInterval:
    point: float
    low: float
    high: float
    level: float = 0.95

    def to_dict(self) -> dict[str, float]:
        return {"value": round(self.point, 4), "ci_low": round(self.low, 4), "ci_high": round(self.high, 4)}

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile of pre-sorted values, q in [0, 1]."""
    if not sorted_vals:
        raise ValueError("empty sample")
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def bootstrap_ci(
    data: Sequence[T],
    statistic: Callable[[list[T]], float],
    *,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    """Percentile bootstrap CI for ``statistic`` over resamples of ``data``.

    The resampling unit is whatever ``data`` contains. For pass^k we resample *tasks*
    (each element is one task's trial vector), because tasks, not trials, are the
    independent units; resampling trials would understate variance.
    """
    if not data:
        raise ValueError("bootstrap_ci requires non-empty data")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    rng = random.Random(seed)
    items = list(data)
    n = len(items)
    stats = sorted(statistic([items[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples))
    alpha = (1 - level) / 2
    return ConfidenceInterval(
        point=statistic(items),
        low=_percentile(stats, alpha),
        high=_percentile(stats, 1 - alpha),
        level=level,
    )

"""Inter-rater agreement: Cohen's kappa, confusion matrices, per-class metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from agenteval.metrics.bootstrap import ConfidenceInterval, bootstrap_ci


def _labels(a: Sequence[str], b: Sequence[str], labels: Sequence[str] | None) -> list[str]:
    if len(a) != len(b):
        raise ValueError(f"rater sequences differ in length: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("cannot compute agreement on empty sequences")
    if labels is None:
        return sorted(set(a) | set(b))
    unknown = (set(a) | set(b)) - set(labels)
    if unknown:
        raise ValueError(f"labels not in label set: {sorted(unknown)}")
    return list(labels)


def confusion_matrix(a: Sequence[str], b: Sequence[str], labels: Sequence[str] | None = None) -> tuple[list[str], list[list[int]]]:
    """Return ``(labels, matrix)`` where ``matrix[i][j]`` counts a=labels[i], b=labels[j]."""
    labs = _labels(a, b, labels)
    idx = {lab: i for i, lab in enumerate(labs)}
    m = [[0] * len(labs) for _ in labs]
    for x, y in zip(a, b):
        m[idx[x]][idx[y]] += 1
    return labs, m


def percent_agreement(a: Sequence[str], b: Sequence[str]) -> float:
    _labels(a, b, None)
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohen_kappa(a: Sequence[str], b: Sequence[str], labels: Sequence[str] | None = None) -> float:
    """Cohen's kappa: chance-corrected agreement between two raters.

    ``kappa = (p_o - p_e) / (1 - p_e)`` where ``p_o`` is observed agreement and
    ``p_e = sum_l p_a(l) * p_b(l)`` is agreement expected from the raters' marginals.
    Returns 1.0 when both raters use a single identical label (p_e = 1, p_o = 1).
    """
    _labels(a, b, labels)
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    p_e = sum((ca[lab] / n) * (cb[lab] / n) for lab in set(ca) | set(cb))
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def kappa_ci(
    a: Sequence[str], b: Sequence[str], *, n_resamples: int = 2000, level: float = 0.95, seed: int = 0
) -> ConfidenceInterval:
    """Bootstrap CI for kappa, resampling items (paired)."""
    pairs = list(zip(a, b))
    return bootstrap_ci(
        pairs,
        lambda ps: cohen_kappa([p[0] for p in ps], [p[1] for p in ps]),
        n_resamples=n_resamples,
        level=level,
        seed=seed,
    )


def interpret_kappa(kappa: float) -> str:
    """Landis & Koch (1977) verbal scale."""
    if kappa < 0:
        return "poor"
    for threshold, name in ((0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")):
        if kappa <= threshold:
            return name
    return "almost perfect"


def per_class_metrics(reference: Sequence[str], predicted: Sequence[str],
                      labels: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
    """Precision/recall/F1 of ``predicted`` against ``reference`` for each label."""
    labs = _labels(reference, predicted, labels)
    out: dict[str, dict[str, Any]] = {}
    for lab in labs:
        tp = sum(r == lab and p == lab for r, p in zip(reference, predicted))
        fp = sum(r != lab and p == lab for r, p in zip(reference, predicted))
        fn = sum(r == lab and p != lab for r, p in zip(reference, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[lab] = {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
                    "support": tp + fn}
    return out

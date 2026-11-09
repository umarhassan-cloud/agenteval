"""Reliability metrics: pass@k, pass^k, bootstrap confidence intervals."""

from agenteval.metrics.aggregate import compute_reliability
from agenteval.metrics.bootstrap import ConfidenceInterval, bootstrap_ci
from agenteval.metrics.reliability import (
    group_by_task,
    mean_pass_at_k,
    mean_pass_hat_k,
    pass_at_k,
    pass_hat_k,
    reliability_curve,
)

__all__ = [
    "ConfidenceInterval",
    "bootstrap_ci",
    "compute_reliability",
    "group_by_task",
    "mean_pass_at_k",
    "mean_pass_hat_k",
    "pass_at_k",
    "pass_hat_k",
    "reliability_curve",
]

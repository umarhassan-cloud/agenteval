"""Tests for pass@k, pass^k and bootstrap CIs."""

from __future__ import annotations

from math import comb

import pytest

from agenteval.metrics import (
    bootstrap_ci,
    group_by_task,
    mean_pass_at_k,
    mean_pass_hat_k,
    pass_at_k,
    pass_hat_k,
    reliability_curve,
)
from agenteval.types import EvalResult, Trajectory


class TestPassHatK:
    def test_known_value(self):
        # C(3,2)/C(5,2) = 3/10
        assert pass_hat_k(5, 3, 2) == pytest.approx(0.3)

    def test_all_success_is_one(self):
        assert pass_hat_k(4, 4, 4) == 1.0

    def test_any_failure_with_k_equals_n_is_zero(self):
        assert pass_hat_k(4, 3, 4) == 0.0

    def test_zero_successes(self):
        assert pass_hat_k(5, 0, 1) == 0.0

    def test_k1_equals_success_rate(self):
        assert pass_hat_k(8, 5, 1) == pytest.approx(5 / 8)

    def test_monotonically_non_increasing_in_k(self):
        vals = [pass_hat_k(8, 6, k) for k in range(1, 9)]
        assert all(a >= b for a, b in zip(vals, vals[1:]))

    def test_matches_brute_force_expectation(self):
        # Probability all k of a random k-subset (without replacement) are successes.
        from itertools import combinations

        outcomes = [1, 1, 0, 1, 0, 1]
        for k in range(1, 7):
            subsets = list(combinations(outcomes, k))
            expected = sum(all(s) for s in subsets) / len(subsets)
            assert pass_hat_k(6, 4, k) == pytest.approx(expected)

    @pytest.mark.parametrize("n,c,k", [(3, 1, 4), (0, 0, 1), (3, 4, 1), (3, -1, 1), (3, 1, 0)])
    def test_invalid_inputs_raise(self, n, c, k):
        with pytest.raises(ValueError):
            pass_hat_k(n, c, k)


class TestPassAtK:
    def test_known_value(self):
        # 1 - C(2,2)/C(5,2) = 0.9
        assert pass_at_k(5, 3, 2) == pytest.approx(0.9)

    def test_guaranteed_hit(self):
        assert pass_at_k(5, 4, 2) == 1.0

    def test_zero_successes(self):
        assert pass_at_k(5, 0, 3) == 0.0

    def test_k1_equals_pass_hat_1(self):
        assert pass_at_k(7, 3, 1) == pytest.approx(pass_hat_k(7, 3, 1))

    def test_monotonically_non_decreasing_in_k(self):
        vals = [pass_at_k(10, 3, k) for k in range(1, 11)]
        assert all(a <= b for a, b in zip(vals, vals[1:]))

    def test_formula(self):
        assert pass_at_k(10, 4, 3) == pytest.approx(1 - comb(6, 3) / comb(10, 3))

    def test_pass_at_k_dominates_pass_hat_k(self):
        for c in range(6):
            for k in range(1, 6):
                assert pass_at_k(5, c, k) >= pass_hat_k(5, c, k) - 1e-12


class TestSuiteMetrics:
    RESULTS = {"a": [True, True, True], "b": [True, False, True], "c": [False, False, False]}

    def test_mean_pass_hat_k(self):
        # pass^2: a=1, b=C(2,2)/C(3,2)=1/3, c=0 -> mean 4/9
        assert mean_pass_hat_k(self.RESULTS, 2) == pytest.approx(4 / 9)

    def test_mean_pass_at_k(self):
        # pass@2: a=1, b=1, c=0
        assert mean_pass_at_k(self.RESULTS, 2) == pytest.approx(2 / 3)

    def test_reliability_curve(self):
        curve = reliability_curve(self.RESULTS)
        assert list(curve) == [1, 2, 3]
        assert curve[1] == pytest.approx(5 / 9)
        assert curve[3] == pytest.approx(1 / 3)

    def test_empty(self):
        assert mean_pass_hat_k({}, 1) == 0.0
        assert reliability_curve({}) == {}

    def test_group_by_task_orders_by_trial(self):
        def traj(task: str, trial: int, ok: bool) -> Trajectory:
            return Trajectory(run_id=f"{task}{trial}", task_id=task, domain="d", agent="a", model="m",
                              trial=trial, seed=0, eval=EvalResult(ok, ok, ok, True, 1.0))

        grouped = group_by_task([traj("x", 1, False), traj("x", 0, True), traj("y", 0, True)])
        assert grouped == {"x": [True, False], "y": [True]}


class TestBootstrap:
    def test_ci_contains_point_and_is_ordered(self):
        data = [0.0, 1.0] * 20
        ci = bootstrap_ci(data, lambda xs: sum(xs) / len(xs), n_resamples=500, seed=1)
        assert ci.low <= ci.point <= ci.high
        assert ci.point == pytest.approx(0.5)
        assert 0.3 < ci.low < 0.5 < ci.high < 0.7

    def test_deterministic_given_seed(self):
        data = list(range(30))
        stat = lambda xs: sum(xs) / len(xs)
        assert bootstrap_ci(data, stat, seed=3) == bootstrap_ci(data, stat, seed=3)

    def test_constant_data_has_zero_width(self):
        ci = bootstrap_ci([1.0] * 10, lambda xs: sum(xs) / len(xs), n_resamples=200)
        assert ci.low == ci.high == ci.point == 1.0

    def test_more_data_narrows_interval(self):
        stat = lambda xs: sum(xs) / len(xs)
        small = bootstrap_ci([0, 1] * 5, stat, n_resamples=1000, seed=0)
        large = bootstrap_ci([0, 1] * 200, stat, n_resamples=1000, seed=0)
        assert (large.high - large.low) < (small.high - small.low)

    def test_invalid(self):
        with pytest.raises(ValueError):
            bootstrap_ci([], lambda xs: 0.0)
        with pytest.raises(ValueError):
            bootstrap_ci([1], lambda xs: 0.0, level=1.5)

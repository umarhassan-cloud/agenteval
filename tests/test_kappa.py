"""Tests for Cohen's kappa and agreement utilities."""

from __future__ import annotations

import pytest

from agenteval.judge import (
    cohen_kappa,
    confusion_matrix,
    interpret_kappa,
    kappa_ci,
    per_class_metrics,
    percent_agreement,
)


def _wikipedia_example() -> tuple[list[str], list[str]]:
    # 50 items: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> p_o=0.7, p_e=0.5, kappa=0.4
    a = ["yes"] * 20 + ["yes"] * 5 + ["no"] * 10 + ["no"] * 15
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    return a, b


def test_kappa_wikipedia_example():
    a, b = _wikipedia_example()
    assert percent_agreement(a, b) == pytest.approx(0.7)
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_kappa_is_symmetric():
    a, b = _wikipedia_example()
    assert cohen_kappa(a, b) == pytest.approx(cohen_kappa(b, a))


def test_perfect_agreement():
    labels = ["SUCCESS", "LOOPING", "WRONG_TOOL", "SUCCESS"]
    assert cohen_kappa(labels, labels) == pytest.approx(1.0)


def test_chance_agreement_is_zero():
    a = ["x", "x", "y", "y"]
    b = ["x", "y", "x", "y"]
    assert cohen_kappa(a, b) == pytest.approx(0.0)


def test_systematic_disagreement_is_negative():
    a = ["x", "y"] * 10
    b = ["y", "x"] * 10
    assert cohen_kappa(a, b) == pytest.approx(-1.0)


def test_single_shared_label():
    assert cohen_kappa(["a", "a"], ["a", "a"]) == 1.0


def test_multiclass_known_value():
    # p_o = 6/9; marginals a: A3 B3 C3, b: A4 B2 C3 -> p_e = (12+6+9)/81 = 1/3; kappa = 0.5
    a = ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
    b = ["A", "A", "B", "B", "A", "C", "C", "C", "A"]
    assert percent_agreement(a, b) == pytest.approx(5 / 9)
    p_o, p_e = 5 / 9, (3 * 4 + 3 * 2 + 3 * 3) / 81
    assert cohen_kappa(a, b) == pytest.approx((p_o - p_e) / (1 - p_e))


def test_confusion_matrix():
    labs, m = confusion_matrix(["a", "a", "b"], ["a", "b", "b"], labels=["a", "b"])
    assert labs == ["a", "b"]
    assert m == [[1, 1], [0, 1]]


def test_confusion_matrix_rejects_unknown_labels():
    with pytest.raises(ValueError):
        confusion_matrix(["a"], ["z"], labels=["a", "b"])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        cohen_kappa(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        cohen_kappa([], [])


def test_per_class_metrics():
    ref = ["a", "a", "b", "b"]
    pred = ["a", "b", "b", "b"]
    m = per_class_metrics(ref, pred)
    assert m["a"] == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(0.667, abs=1e-3), "support": 2}
    assert m["b"]["precision"] == pytest.approx(0.667, abs=1e-3)
    assert m["b"]["recall"] == 1.0


@pytest.mark.parametrize("k,name", [(-0.1, "poor"), (0.1, "slight"), (0.3, "fair"), (0.5, "moderate"),
                                    (0.72, "substantial"), (0.9, "almost perfect")])
def test_interpretation(k, name):
    assert interpret_kappa(k) == name


def test_kappa_ci_brackets_point():
    a, b = _wikipedia_example()
    ci = kappa_ci(a, b, n_resamples=500, seed=0)
    assert ci.low < ci.point < ci.high
    assert ci.point == pytest.approx(0.4)

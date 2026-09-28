from __future__ import annotations

import pytest

from par.evaluation.metrics import (
    classification_accuracy,
    completion_agreement,
    confusion_matrix,
    latency_stats,
    rate,
    summarize,
)


def test_rate_handles_zero_denominator():
    assert rate(0, 0) == 0.0


def test_rate_basic():
    assert rate(3, 4) == pytest.approx(0.75)


def test_summarize_empty():
    stats = summarize([])
    assert stats.n == 0
    assert stats.mean == 0.0


def test_summarize_single_value_has_zero_stdev_and_ci():
    stats = summarize([5.0])
    assert stats.n == 1
    assert stats.mean == 5.0
    assert stats.stdev == 0.0
    assert stats.ci95_half_width == 0.0


def test_summarize_multiple_values():
    stats = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats.n == 5
    assert stats.mean == pytest.approx(3.0)
    assert stats.median == pytest.approx(3.0)
    assert stats.stdev > 0
    assert stats.ci95_half_width > 0
    low, high = stats.ci95
    assert low < stats.mean < high


def test_latency_stats_p95_uses_highest_value_for_small_n():
    stats = latency_stats([1.0, 2.0, 3.0, 4.0, 100.0])
    assert stats["n"] == 5
    assert stats["p95"] == 100.0


def test_confusion_matrix_counts_correctly():
    expected = ["allow", "allow", "deny"]
    actual = ["allow", "deny", "deny"]
    matrix = confusion_matrix(expected, actual, ["allow", "deny"])
    assert matrix["allow"]["allow"] == 1
    assert matrix["allow"]["deny"] == 1
    assert matrix["deny"]["deny"] == 1
    assert matrix["deny"]["allow"] == 0


def test_confusion_matrix_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        confusion_matrix(["allow"], ["allow", "deny"], ["allow", "deny"])


def test_classification_accuracy():
    assert classification_accuracy(["a", "b", "c"], ["a", "x", "c"]) == pytest.approx(2 / 3)


def test_classification_accuracy_empty():
    assert classification_accuracy([], []) == 0.0


def test_completion_agreement_all_true_positive():
    metrics = completion_agreement([(True, True), (True, True)])
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.disagreement_rate == 0.0


def test_completion_agreement_false_positive_is_the_dangerous_case():
    # Model claims success (True) but the goal was not actually verified (False).
    metrics = completion_agreement([(True, False), (True, True)])
    assert metrics.false_positive == 1
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.disagreement_rate == pytest.approx(0.5)


def test_completion_agreement_empty():
    metrics = completion_agreement([])
    assert metrics.n == 0
    assert metrics.precision == 0.0
    assert metrics.disagreement_rate == 0.0

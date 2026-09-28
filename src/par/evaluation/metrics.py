from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

_Z_95 = 1.96  # normal approximation; fine for the trial counts these experiments use


def rate(numerator: int, denominator: int) -> float:
    """Returns 0.0 for an empty denominator rather than raising - an
    experiment with zero applicable trials should report "no data", and a
    silent ZeroDivisionError crash is a worse failure mode than a 0.0 that
    trial-count reporting nearby will make obviously suspicious."""
    return numerator / denominator if denominator else 0.0


@dataclass
class SummaryStats:
    n: int
    mean: float
    median: float
    stdev: float
    ci95_half_width: float

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.mean - self.ci95_half_width, self.mean + self.ci95_half_width)


def summarize(values: Sequence[float]) -> SummaryStats:
    """Mean/median/stdev/95% CI (normal approximation) over a sample.

    Uses a z=1.96 normal approximation rather than a t-distribution critical
    value, since scipy isn't a dependency here. This understates the interval
    slightly for small n (n<30); fine for "is this in the right ballpark"
    reporting, not for a submission-grade confidence interval.
    """
    values = list(values)
    n = len(values)
    if n == 0:
        return SummaryStats(n=0, mean=0.0, median=0.0, stdev=0.0, ci95_half_width=0.0)
    mean = statistics.fmean(values)
    median = statistics.median(values)
    stdev = statistics.stdev(values) if n > 1 else 0.0
    half_width = _Z_95 * stdev / (n**0.5) if n > 1 else 0.0
    return SummaryStats(n=n, mean=mean, median=median, stdev=stdev, ci95_half_width=half_width)


def latency_stats(latencies: Sequence[float]) -> dict[str, float]:
    values = sorted(latencies)
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "stdev": 0.0, "p95": 0.0}
    p95_index = min(n - 1, int(round(0.95 * (n - 1))))
    return {
        "n": n,
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if n > 1 else 0.0,
        "p95": values[p95_index],
    }


def confusion_matrix(expected: Sequence[str], actual: Sequence[str], labels: Sequence[str]) -> dict[str, dict[str, int]]:
    if len(expected) != len(actual):
        raise ValueError("expected and actual must be the same length")
    matrix = {e: {a: 0 for a in labels} for e in labels}
    for e, a in zip(expected, actual):
        matrix[e][a] += 1
    return matrix


def classification_accuracy(expected: Sequence[str], actual: Sequence[str]) -> float:
    if not expected:
        return 0.0
    correct = sum(1 for e, a in zip(expected, actual) if e == a)
    return rate(correct, len(expected))


@dataclass
class AgreementMetrics:
    n: int
    precision: float  # of reported successes, fraction that were verified true
    recall: float  # of verified successes, fraction the planner also reported
    disagreement_rate: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


def completion_agreement(pairs: Sequence[tuple[bool, bool]]) -> AgreementMetrics:
    """Compares planner-reported success against environment-verified success.

    Treats verified success as ground truth (per Experiment 22): a
    false_positive is the model claiming success when the goal wasn't
    actually achieved - the failure mode that matters most for trusting a
    physical agent's own reporting.
    """
    tp = sum(1 for reported, verified in pairs if reported and verified)
    fp = sum(1 for reported, verified in pairs if reported and not verified)
    tn = sum(1 for reported, verified in pairs if not reported and not verified)
    fn = sum(1 for reported, verified in pairs if not reported and verified)
    n = len(pairs)
    disagreements = fp + fn
    return AgreementMetrics(
        n=n,
        precision=rate(tp, tp + fp),
        recall=rate(tp, tp + fn),
        disagreement_rate=rate(disagreements, n),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )


@dataclass
class OutcomeCounts:
    allow: int = 0
    modify: int = 0
    deny: int = 0
    escalate: int = 0

    @property
    def total(self) -> int:
        return self.allow + self.modify + self.deny + self.escalate

    def rate_of(self, outcome: str) -> float:
        return rate(getattr(self, outcome), self.total)

    @classmethod
    def from_outcomes(cls, outcomes: Sequence[str]) -> "OutcomeCounts":
        counts = cls()
        for outcome in outcomes:
            setattr(counts, outcome, getattr(counts, outcome) + 1)
        return counts

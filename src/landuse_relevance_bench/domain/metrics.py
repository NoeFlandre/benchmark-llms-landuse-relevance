"""Classification metrics for a benchmark run.

The positive class is ``yes`` (the sentence carries land-use signal). Outputs the
model failed to express as a verdict are counted as errors, never dropped.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Any

from landuse_relevance_bench.domain.labels import Label

Outcome = tuple[Label, Label | None]


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Outcome counts, with unparsable generations tracked as their own bucket."""

    true_positive: int
    false_negative: int
    true_negative: int
    false_positive: int
    unparsed: int

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.false_negative
            + self.true_negative
            + self.false_positive
            + self.unparsed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "true_positive": self.true_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "unparsed": self.unparsed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConfusionMatrix":
        return cls(
            true_positive=payload["true_positive"],
            false_negative=payload["false_negative"],
            true_negative=payload["true_negative"],
            false_positive=payload["false_positive"],
            unparsed=payload["unparsed"],
        )


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Headline scores for one model on the whole benchmark."""

    confusion: ConfusionMatrix
    n_items: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    matthews_corrcoef: float
    unparsed_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "confusion": self.confusion.to_dict(),
            "n_items": self.n_items,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "balanced_accuracy": self.balanced_accuracy,
            "matthews_corrcoef": self.matthews_corrcoef,
            "unparsed_rate": self.unparsed_rate,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClassificationMetrics":
        return cls(
            confusion=ConfusionMatrix.from_dict(payload["confusion"]),
            n_items=payload["n_items"],
            accuracy=payload["accuracy"],
            precision=payload["precision"],
            recall=payload["recall"],
            f1=payload["f1"],
            balanced_accuracy=payload["balanced_accuracy"],
            matthews_corrcoef=payload["matthews_corrcoef"],
            unparsed_rate=payload["unparsed_rate"],
        )


def confusion_of(outcomes: Sequence[Outcome]) -> ConfusionMatrix:
    """Tally ``(expected, predicted)`` pairs into a confusion matrix."""
    counts = dict.fromkeys(
        ("true_positive", "false_negative", "true_negative", "false_positive", "unparsed"), 0
    )
    for expected, predicted in outcomes:
        counts[_bucket(expected, predicted)] += 1
    return ConfusionMatrix(**counts)


def evaluate(outcomes: Sequence[Outcome]) -> ClassificationMetrics:
    """Score ``(expected, predicted)`` pairs; every item counts, parsed or not."""
    if not outcomes:
        raise ValueError("cannot evaluate an empty set of outcomes")
    matrix = confusion_of(outcomes)
    correct = matrix.true_positive + matrix.true_negative
    positives = matrix.true_positive + matrix.false_negative
    negatives = matrix.true_negative + matrix.false_positive
    precision = _ratio(matrix.true_positive, matrix.true_positive + matrix.false_positive)
    recall = _ratio(matrix.true_positive, matrix.true_positive + matrix.false_negative)
    return ClassificationMetrics(
        confusion=matrix,
        n_items=matrix.total,
        accuracy=_ratio(correct, matrix.total),
        precision=precision,
        recall=recall,
        f1=_ratio(2 * precision * recall, precision + recall),
        balanced_accuracy=(
            _ratio(matrix.true_positive, positives) + _ratio(matrix.true_negative, negatives)
        )
        / 2,
        matthews_corrcoef=_matthews(matrix),
        unparsed_rate=_ratio(matrix.unparsed, matrix.total),
    )


def _bucket(expected: Label, predicted: Label | None) -> str:
    if predicted is None:
        return "unparsed"
    if predicted is Label.YES:
        return "true_positive" if expected is Label.YES else "false_positive"
    return "false_negative" if expected is Label.YES else "true_negative"


def _matthews(matrix: ConfusionMatrix) -> float:
    tp, tn = matrix.true_positive, matrix.true_negative
    fp, fn = matrix.false_positive, matrix.false_negative
    denominator = sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return _ratio(float(tp * tn - fp * fn), denominator)


def _ratio(numerator: float, denominator: float) -> float:
    """Degenerate denominators score zero, which is the honest reading here."""
    return numerator / denominator if denominator else 0.0

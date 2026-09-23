"""Re-deciding a scoring model's verdicts at other thresholds.

A reranker's score ranks documents for retrieval; it is not calibrated to a 0.5
boundary. Published runs keep the native score for every item, so the effect of the
boundary can be measured after the fact, without re-running a model.
"""

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import ClassificationMetrics, evaluate
from landuse_relevance_bench.domain.records import Prediction

#: Log-spaced boundaries plus the usual ones. A reranker's positive scores sit orders
#: of magnitude below 0.5, so a linear grid would measure nothing but the same verdict
#: over and over.
DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.0,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
)


class ScoreFormatError(ValueError):
    """Raised when a stored prediction does not carry parseable native scores."""


@dataclass(frozen=True, slots=True)
class ThresholdPoint:
    """What the same predictions score when `yes` needs at least ``threshold``."""

    threshold: float
    metrics: ClassificationMetrics


def parse_scores(raw_output: str) -> dict[Label, float]:
    """Read back the ``label=score`` text a scoring run stores for one item."""
    scores: dict[Label, float] = {}
    for part in raw_output.split():
        name, separator, value = part.partition("=")
        if not separator:
            raise ScoreFormatError(f"not a label=score pair: {part!r}")
        try:
            parsed = float(value)
            if name != "native":
                scores[Label(name)] = parsed
        except (TypeError, ValueError) as exc:
            raise ScoreFormatError(f"unreadable score {part!r}") from exc
    if not scores:
        raise ScoreFormatError(f"no scores in {raw_output!r}")
    return scores


def yes_scores(predictions: Sequence[Prediction]) -> tuple[float, ...]:
    """The `yes` score of every prediction, in order."""
    return tuple(parse_scores(p.raw_output).get(Label.YES, 0.0) for p in predictions)


def decide_at(predictions: Sequence[Prediction], threshold: float) -> ClassificationMetrics:
    """Score the same items again, calling `yes` only at or above ``threshold``."""
    if not predictions:
        raise ValueError("cannot re-decide an empty set of predictions")
    scores = yes_scores(predictions)
    outcomes = [
        (p.expected, Label.YES if score >= threshold else Label.NO)
        for p, score in zip(predictions, scores, strict=True)
    ]
    return evaluate(outcomes)


def sweep(
    predictions: Sequence[Prediction],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> tuple[ThresholdPoint, ...]:
    """Re-decide the predictions at each threshold, in the order given."""
    return tuple(ThresholdPoint(t, decide_at(predictions, t)) for t in thresholds)


def roc_auc(predictions: Sequence[Prediction]) -> float:
    """How often a `yes` item outscores a `no` item, ties counting half.

    Threshold-free, so it measures how well the score separates the classes rather
    than how well a particular boundary happens to suit this benchmark.
    """
    scores = yes_scores(predictions)
    positive_scores, negative_scores = _scores_by_class(predictions, scores)
    if not positive_scores or not negative_scores:
        raise ValueError("ROC AUC needs at least one item of each class")
    ordered_negatives = sorted(negative_scores)
    favorable_pairs = sum(_wins_and_ties(ordered_negatives, score) for score in positive_scores)
    return favorable_pairs / (len(positive_scores) * len(negative_scores))


def _scores_by_class(
    predictions: Sequence[Prediction], scores: Sequence[float]
) -> tuple[list[float], list[float]]:
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    for prediction, score in zip(predictions, scores, strict=True):
        if prediction.expected is Label.YES:
            positive_scores.append(score)
        else:
            negative_scores.append(score)
    return positive_scores, negative_scores


def _wins_and_ties(ordered_negatives: Sequence[float], score: float) -> float:
    below = bisect_left(ordered_negatives, score)
    tied = bisect_right(ordered_negatives, score) - below
    return below + 0.5 * tied

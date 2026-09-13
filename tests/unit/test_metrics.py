import math

import pytest

from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import ConfusionMatrix, evaluate

Y, N = Label.YES, Label.NO


def test_confusion_counts_each_outcome() -> None:
    metrics = evaluate([(Y, Y), (Y, N), (N, N), (N, Y), (Y, None)])
    assert metrics.confusion == ConfusionMatrix(
        true_positive=1, false_negative=1, true_negative=1, false_positive=1, unparsed=1
    )


def test_perfect_predictions_score_one() -> None:
    metrics = evaluate([(Y, Y), (N, N), (Y, Y)])
    assert metrics.accuracy == 1.0
    assert metrics.f1 == 1.0
    assert metrics.matthews_corrcoef == 1.0
    assert metrics.balanced_accuracy == 1.0
    assert metrics.unparsed_rate == 0.0


def test_unparsed_predictions_count_as_errors_not_as_omissions() -> None:
    metrics = evaluate([(Y, Y), (N, None)])
    assert metrics.n_items == 2
    assert metrics.accuracy == 0.5
    assert metrics.unparsed_rate == 0.5


def test_metrics_of_a_typical_mixed_run() -> None:
    metrics = evaluate([(Y, Y), (Y, Y), (Y, N), (N, N), (N, Y)])
    assert metrics.accuracy == pytest.approx(3 / 5)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.balanced_accuracy == pytest.approx((2 / 3 + 1 / 2) / 2)


def test_degenerate_denominators_yield_zero_rather_than_nan() -> None:
    metrics = evaluate([(N, N), (N, N)])
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.matthews_corrcoef == 0.0
    assert not math.isnan(metrics.balanced_accuracy)


def test_rejects_an_empty_evaluation() -> None:
    with pytest.raises(ValueError):
        evaluate([])

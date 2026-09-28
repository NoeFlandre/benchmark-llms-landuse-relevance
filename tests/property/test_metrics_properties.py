"""Metric invariants over generated outcomes."""

from hypothesis import given
from hypothesis import strategies as st
from strategies import outcomes

from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import (
    ClassificationMetrics,
    ConfusionMatrix,
    evaluate,
)


@given(pairs=st.lists(outcomes, min_size=1, max_size=40))
def test_evaluation_is_invariant_to_a_canonical_permutation(pairs) -> None:
    reordered = sorted(
        pairs,
        key=lambda pair: (pair[0].value, "" if pair[1] is None else pair[1].value),
    )
    assert evaluate(pairs) == evaluate(reordered)


@given(pairs=st.lists(outcomes, min_size=1, max_size=40))
def test_swapping_yes_and_no_preserves_accuracy(pairs) -> None:
    swap = {Label.YES: Label.NO, Label.NO: Label.YES}
    mirrored = [
        (swap[expected], None if predicted is None else swap[predicted])
        for expected, predicted in pairs
    ]
    assert evaluate(mirrored).accuracy == evaluate(pairs).accuracy


@given(pairs=st.lists(outcomes, min_size=1, max_size=40))
def test_confusion_and_classification_metrics_round_trip_through_dicts(pairs) -> None:
    metrics = evaluate(pairs)
    assert ConfusionMatrix.from_dict(metrics.confusion.to_dict()) == metrics.confusion
    assert ClassificationMetrics.from_dict(metrics.to_dict()) == metrics

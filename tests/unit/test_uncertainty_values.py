"""Exact values of the interval, quantile and McNemar helpers."""

import pytest

from landuse_relevance_bench.domain import uncertainty
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.records import Prediction
from landuse_relevance_bench.domain.uncertainty import (
    exact_mcnemar_p_value,
    interval_text,
    paired_mcnemar_p_value,
    wilson_interval,
)

Y, N = Label.YES, Label.NO


def _predictions(predicted, expected=None):
    expected = expected or [Y] * len(predicted)
    return tuple(
        Prediction(item_id=f"{index:016d}", expected=exp, predicted=pred, raw_output="x")
        for index, (exp, pred) in enumerate(zip(expected, predicted, strict=True))
    )


def test_the_wilson_interval_for_seven_of_thirteen_is_exact_to_four_places() -> None:
    low, high = wilson_interval(7, 13)

    assert round(low, 4) == 0.2914
    assert round(high, 4) == 0.7679


def test_the_wilson_interval_is_undefined_for_an_empty_count() -> None:
    assert wilson_interval(0, 0) is None


def test_the_quantile_interpolates_between_the_two_nearest_order_statistics() -> None:
    values = [0.0, 10.0, 20.0, 30.0]

    assert uncertainty._quantile(values, 0.5) == 15.0
    assert uncertainty._quantile(values, 0.0) == 0.0
    assert uncertainty._quantile(values, 1.0) == 30.0


def test_an_exact_mcnemar_tail_of_three_discordant_pairs_is_one_quarter() -> None:
    assert exact_mcnemar_p_value(3, 0) == 0.25
    assert exact_mcnemar_p_value(0, 3) == 0.25


def test_an_exact_mcnemar_p_value_is_capped_at_one() -> None:
    assert exact_mcnemar_p_value(2, 2) == 1.0
    assert exact_mcnemar_p_value(0, 0) == 1.0


def test_paired_mcnemar_counts_only_the_items_where_exactly_one_run_is_right() -> None:
    # Ours is right on items 0-2 and theirs only on item 0: two discordant pairs favour ours.
    ours = _predictions([Y, Y, Y, N, N])
    theirs = _predictions([Y, N, N, N, N])

    assert paired_mcnemar_p_value(ours, theirs) == 0.5


def test_paired_mcnemar_is_symmetric_in_which_run_is_called_first() -> None:
    ours = _predictions([Y, Y, Y, N, N])
    theirs = _predictions([Y, N, N, N, N])

    assert paired_mcnemar_p_value(theirs, ours) == 0.5


def test_paired_mcnemar_is_undefined_when_the_item_sets_differ() -> None:
    ours = _predictions([Y, Y])
    theirs = _predictions([Y])

    assert paired_mcnemar_p_value(ours, theirs) is None


def test_paired_mcnemar_is_undefined_when_the_expected_labels_differ() -> None:
    ours = _predictions([Y, Y], expected=[Y, Y])
    theirs = _predictions([Y, Y], expected=[Y, N])

    assert paired_mcnemar_p_value(ours, theirs) is None


def test_interval_text_renders_the_lower_then_the_upper_bound() -> None:
    assert interval_text((0.1, 0.25)) == "[0.1000, 0.2500]"
    assert interval_text(None) is None


def test_bootstrap_intervals_are_reproducible_for_a_given_seed() -> None:
    outcomes = [(Y, Y), (Y, N), (N, N), (N, Y), (Y, Y), (N, N)]

    first = uncertainty.bootstrap_interval(outcomes, "f1", seed=0, resamples=50)
    second = uncertainty.bootstrap_interval(outcomes, "f1", seed=0, resamples=50)

    assert first == second
    with pytest.raises(ValueError, match="cannot bootstrap an empty set of outcomes"):
        uncertainty.bootstrap_interval([], "f1")


def test_the_wilson_upper_bound_is_capped_at_one_when_rounding_overshoots() -> None:
    # For 16 of 16 the uncapped bound is 1.0000000000000002 in floating point.
    assert wilson_interval(16, 16)[1] == 1.0

import pytest

from factories import make_result
from landuse_relevance_bench.adapters.results_store import leaderboard_rows
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import Prediction
from landuse_relevance_bench.domain.uncertainty import (
    _quantile,
    bootstrap_interval,
    exact_mcnemar_p_value,
    interval_text,
    paired_mcnemar_p_value,
    wilson_interval,
)


def _predictions(values: list[tuple[Label, Label | None]]) -> tuple[Prediction, ...]:
    return tuple(
        Prediction(
            item_id=f"{i:016x}",
            expected=expected,
            predicted=predicted,
            raw_output="",
        )
        for i, (expected, predicted) in enumerate(values)
    )


def test_wilson_interval_is_bounded_and_centered_for_half_successes() -> None:
    low, high = wilson_interval(5, 10)
    assert 0.0 < low < 0.5 < high < 1.0
    assert abs((low + high) / 2 - 0.5) < 0.01


def test_wilson_interval_caps_the_upper_bound_for_all_successes() -> None:
    assert wilson_interval(5, 5)[1] == 1.0


def test_wilson_interval_is_missing_for_a_zero_denominator() -> None:
    assert wilson_interval(0, 0) is None


@pytest.mark.parametrize("successes,total", [(-1, 1), (2, 1), (0, -1)])
def test_wilson_interval_rejects_invalid_counts(successes: int, total: int) -> None:
    with pytest.raises(
        ValueError,
        match=r"^successes and total must describe a valid binomial count$",
    ):
        wilson_interval(successes, total)


def test_bootstrap_interval_is_seeded_and_deterministic() -> None:
    outcomes = [(Label.YES, Label.YES), (Label.NO, Label.YES), (Label.NO, Label.NO)]
    first = bootstrap_interval(outcomes, "f1", seed=7, resamples=300)
    second = bootstrap_interval(outcomes, "f1", seed=7, resamples=300)
    assert first == second
    assert first is not None and first[0] <= first[1]


def test_bootstrap_rejects_empty_outcomes_and_zero_resamples() -> None:
    with pytest.raises(ValueError, match=r"^cannot bootstrap an empty set of outcomes$"):
        bootstrap_interval([], "f1")
    with pytest.raises(ValueError, match=r"^resamples must be at least 1$"):
        bootstrap_interval([(Label.YES, Label.YES)], "f1", resamples=0)


def test_bootstrap_accepts_a_single_resample() -> None:
    interval = bootstrap_interval([(Label.YES, Label.YES)], "f1", resamples=1)
    assert interval == (1.0, 1.0)


def test_quantile_interpolates_at_the_requested_endpoints() -> None:
    assert _quantile([1.0, 3.0], 1.0) == 3.0
    assert _quantile([1.0, 3.0], 0.975) == 2.95


def test_exact_mcnemar_uses_the_two_sided_binomial_tail() -> None:
    assert exact_mcnemar_p_value(2, 0) == 0.5
    assert exact_mcnemar_p_value(1, 1) == 1.0
    assert exact_mcnemar_p_value(0, 0) == 1.0


def test_exact_mcnemar_rejects_negative_discordant_counts() -> None:
    with pytest.raises(
        ValueError,
        match=r"^discordant pair counts cannot be negative$",
    ):
        exact_mcnemar_p_value(-1, 0)


def test_paired_mcnemar_requires_matching_nonempty_item_sets() -> None:
    first = _predictions([(Label.YES, Label.YES)])
    other_id = tuple(
        Prediction("different", item.expected, item.predicted, item.raw_output) for item in first
    )
    wrong_label = _predictions([(Label.NO, Label.NO)])
    assert paired_mcnemar_p_value((), ()) is None
    assert paired_mcnemar_p_value(first, other_id) is None
    assert paired_mcnemar_p_value(first, wrong_label) is None


def test_leaderboard_adds_intervals_and_paired_top_run_comparison() -> None:
    top_predictions = _predictions(
        [(Label.YES, Label.YES), (Label.NO, Label.NO), (Label.YES, Label.NO)]
    )
    weaker_predictions = _predictions(
        [(Label.YES, Label.NO), (Label.NO, Label.NO), (Label.YES, Label.NO)]
    )
    top = make_result("a/top", top_predictions)
    weaker = make_result("b/weaker", weaker_predictions)
    rows = {row["model_id"]: row for row in leaderboard_rows([weaker, top])}
    assert rows["a/top"]["mcnemar_p_vs_top"] == 1.0
    assert rows["b/weaker"]["mcnemar_p_vs_top"] == 1.0
    assert rows["b/weaker"]["f1_ci95"] is not None


def test_mcnemar_is_blank_when_the_benchmark_or_coverage_differs() -> None:
    first = make_result("a/one")
    other = make_result("b/two", benchmark_sha256="other")
    row = next(row for row in leaderboard_rows([first, other]) if row["model_id"] == "b/two")
    assert row["mcnemar_p_vs_top"] is None


def test_a_prediction_pair_scores_missing_answers_as_incorrect() -> None:
    predictions = _predictions([(Label.YES, None), (Label.NO, Label.NO)])
    metrics = evaluate([p.outcome for p in predictions])
    assert metrics.accuracy == 0.5


@pytest.mark.parametrize(
    "successes,total,z,expected",
    [
        (5, 10, 1.959963984540054, (0.236593090512564, 0.7634069094874361)),
        (3, 10, 1.959963984540054, (0.10779126740630099, 0.6032218525388546)),
        (0, 5, 1.959963984540054, (0.0, 0.43448246478317476)),
        (1, 2, 1.0, (0.21132486540518713, 0.7886751345948129)),
    ],
)
def test_wilson_interval_matches_reference_values(
    successes: int, total: int, z: float, expected: tuple[float, float]
) -> None:
    assert wilson_interval(successes, total, z=z) == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_wilson_interval_uses_the_95_percent_z_by_default() -> None:
    assert wilson_interval(3, 10) == wilson_interval(3, 10, z=1.959963984540054)


def test_exact_mcnemar_matches_asymmetric_reference_values() -> None:
    assert exact_mcnemar_p_value(3, 1) == 0.625
    assert exact_mcnemar_p_value(1, 3) == 0.625
    assert exact_mcnemar_p_value(5, 0) == 0.0625


def test_paired_mcnemar_counts_each_direction_of_discordance() -> None:
    yes, no = Label.YES, Label.NO
    ours = _predictions([(yes, yes), (no, no), (yes, yes), (no, yes), (yes, None)])
    theirs = _predictions([(yes, no), (no, yes), (yes, no), (no, no), (yes, yes)])
    # ours alone correct on items 0-2, theirs alone correct on items 3-4.
    assert paired_mcnemar_p_value(ours, theirs) == exact_mcnemar_p_value(3, 2)
    assert paired_mcnemar_p_value(theirs, ours) == exact_mcnemar_p_value(2, 3)
    assert paired_mcnemar_p_value(ours, ours) == 1.0


def test_paired_mcnemar_detects_a_one_sided_difference() -> None:
    yes, no = Label.YES, Label.NO
    ours = _predictions([(yes, yes)] * 5 + [(no, no)])
    theirs = _predictions([(yes, no)] * 5 + [(no, no)])
    assert paired_mcnemar_p_value(ours, theirs) == 0.0625


def test_interval_text_rounds_both_bounds_to_four_decimals() -> None:
    assert interval_text((0.12345, 0.9)) == "[0.1235, 0.9000]"
    assert interval_text(None) is None


def test_quantile_interpolates_inside_and_at_the_lower_endpoint() -> None:
    values = [0.0, 1.0, 4.0, 9.0]
    assert _quantile(values, 0.0) == 0.0
    assert _quantile(values, 0.5) == 2.5
    assert _quantile(values, 1 / 3) == pytest.approx(1.0)
    assert _quantile([2.0], 0.975) == 2.0


def test_wilson_interval_clamps_a_float_overshoot_above_one() -> None:
    # Unclamped, 16/16 evaluates to 1.0000000000000002.
    assert wilson_interval(16, 16)[1] == 1.0

"""Exact reported values: metric rounding, intervals, leaderboard ties and scoring summaries."""

import re
from dataclasses import replace

import pytest

from factories import make_metadata, make_result
from landuse_relevance_bench.adapters.results_store import (
    aggregate_rows,
    leaderboard_rows,
    score_columns,
    scoring_summary_rows,
    threshold_sweep_rows,
)
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import Prediction, RunResult

Y, N = Label.YES, Label.NO

# 13 items: TP=2, FP=1, FN=4, TN=5, one unparsed. Every reported metric has a
# non-terminating decimal expansion, so the rounding digits are observable.
THIRTEEN = (
    (Y, Y),
    (Y, Y),
    (N, Y),
    (Y, N),
    (Y, N),
    (Y, N),
    (Y, N),
    (N, N),
    (N, N),
    (N, N),
    (N, N),
    (N, N),
    (Y, None),
)


def _predictions(pairs):
    return tuple(
        Prediction(item_id=f"{index:016d}", expected=expected, predicted=predicted, raw_output="x")
        for index, (expected, predicted) in enumerate(pairs)
    )


def _scoring(model_id, language, items, *, throughput=1.2345, peak=100):
    """A scoring run whose raw output carries each item's yes score; decided at 0.5."""
    predictions = tuple(
        Prediction(
            item_id=f"{index:016d}",
            expected=expected,
            predicted=Y if score >= 0.5 else N,
            raw_output=f"no={1 - score:.6f} yes={score:.6f} native=1.0",
        )
        for index, (expected, score) in enumerate(items)
    )
    metadata = replace(
        make_metadata(model_id=model_id, language=language),
        inference="scoring",
        decision_rule="argmax over the native yes/no scores",
        max_new_tokens=0,
        decoding="none",
        throughput_items_per_second=throughput,
        peak_vram_bytes=peak,
    )
    return RunResult(
        metadata=metadata,
        predictions=predictions,
        metrics=evaluate([p.outcome for p in predictions]),
    )


def test_score_columns_rounds_every_metric_to_four_decimal_places() -> None:
    result = make_result(predictions=_predictions(THIRTEEN))

    columns = score_columns(result.metrics, result.predictions)

    assert columns["accuracy"] == 0.5385
    assert columns["balanced_accuracy"] == 0.5833
    assert columns["f1"] == 0.4444
    assert columns["precision"] == 0.6667
    assert columns["recall"] == 0.3333
    assert columns["matthews_corrcoef"] == 0.1925
    assert columns["unparsed_rate"] == 0.0769
    assert columns["truncated"] == 0


def test_score_columns_reports_wilson_and_bootstrap_intervals_as_text() -> None:
    result = make_result(predictions=_predictions(THIRTEEN))

    columns = score_columns(result.metrics, result.predictions)

    assert columns["accuracy_ci95"] == "[0.2914, 0.7679]"
    assert columns["precision_ci95"] == "[0.2077, 0.9385]"
    assert columns["recall_ci95"] == "[0.0968, 0.7000]"
    assert columns["f1_ci95"] == "[0.0000, 0.8000]"
    assert columns["matthews_corrcoef_ci95"] == "[-0.3860, 0.6831]"


def test_leaderboard_rounds_duration_and_throughput_to_two_decimal_places() -> None:
    result = make_result(
        predictions=_predictions(THIRTEEN),
        duration_seconds=1.2345,
        throughput_items_per_second=2.3456,
    )

    (row,) = leaderboard_rows([result])

    assert row["duration_seconds"] == 1.23
    assert row["throughput_items_per_second"] == 2.35


def test_leaderboard_names_the_highest_f1_run_as_the_top_of_its_language() -> None:
    high = make_result(model_id="b/high", predictions=_predictions(((Y, Y), (N, N))))
    low = make_result(model_id="a/low", predictions=_predictions(((Y, N), (N, Y))))

    rows = {row["model_id"]: row for row in leaderboard_rows([low, high])}

    assert rows["a/low"]["mcnemar_top_run"] == "b/high"
    assert rows["b/high"]["mcnemar_top_run"] == "b/high"


def test_leaderboard_keeps_the_first_of_two_tied_runs_as_the_top() -> None:
    # Both "m" runs have F1 2/3. Against "o", the first run gives McNemar p = 1.0 and
    # the second gives p = 0.5, so the choice of top run is observable.
    first = make_result(
        model_id="m",
        predictions=_predictions(((Y, Y), (Y, Y), (N, Y), (N, Y), (N, Y), (Y, Y))),
    )
    second = make_result(
        model_id="m",
        predictions=_predictions(((Y, Y), (Y, Y), (N, Y), (N, N), (N, N), (Y, N))),
    )
    other = make_result(
        model_id="o",
        predictions=_predictions(((Y, Y), (Y, Y), (N, Y), (N, Y), (N, Y), (Y, N))),
    )

    rows = {row["model_id"]: row for row in leaderboard_rows([first, second, other])}

    assert rows["o"]["mcnemar_top_run"] == "m"
    assert rows["o"]["mcnemar_p_vs_top"] == 1.0


def test_aggregate_rows_report_macro_and_spread_metrics_to_four_places() -> None:
    thirteen = make_result(model_id="m", language="en", predictions=_predictions(THIRTEEN))
    # F1 2/3 and accuracy 2/3: a second language with a different F1 gives a spread.
    second = make_result(
        model_id="m",
        language="fr",
        predictions=_predictions(((Y, Y), (Y, Y), (N, Y), (N, N), (N, N), (Y, N))),
    )

    (row,) = aggregate_rows([thirteen, second])

    assert row["language_count"] == 2
    assert row["n_items_total"] == 19
    assert row["f1_min"] == 0.4444
    assert row["f1_max"] == 0.6667
    assert row["f1_std"] == 0.1111
    assert row["f1_macro"] == 0.5556
    assert row["accuracy_macro"] == 0.6026
    assert row["unparsed_rate_macro"] == 0.0385


def test_aggregate_rows_reject_a_repeated_model_language_pair() -> None:
    one = make_result(model_id="m", language="en")

    with pytest.raises(
        ValueError, match=re.escape("duplicate result for model-language pair ('m', 'en')")
    ):
        aggregate_rows([one, one])


def test_threshold_sweep_rows_report_macro_metrics_and_roc_auc_at_each_boundary() -> None:
    english = _scoring("m", "en", [(Y, 0.9), (N, 0.2), (Y, 0.4)])
    french = _scoring("m", "fr", [(Y, 0.8), (Y, 0.1), (N, 0.5), (N, 0.3), (Y, 0.6)])

    (row,) = [row for row in threshold_sweep_rows([english, french]) if row["threshold"] == 0.5]

    assert row == {
        "model_id": "m",
        "threshold": 0.5,
        "language_count": 2,
        "n_items_total": 8,
        "roc_auc_macro": 0.8333,
        "accuracy_macro": 0.6333,
        "balanced_accuracy_macro": 0.6667,
        "f1_macro": 0.6667,
        "precision_macro": 0.8333,
        "recall_macro": 0.5833,
        "matthews_corrcoef_macro": 0.3333,
    }


def test_threshold_sweep_skips_a_single_class_language_when_averaging_roc_auc() -> None:
    one_class = _scoring("m", "en", [(Y, 0.9), (Y, 0.4)])
    two_class = _scoring("m", "fr", [(Y, 0.8), (N, 0.2)])

    (row,) = [
        row for row in threshold_sweep_rows([one_class, two_class]) if row["threshold"] == 0.5
    ]

    assert row["roc_auc_macro"] == 1.0


def test_scoring_summary_uses_a_sweep_the_caller_supplies() -> None:
    run = _scoring("m", "en", [(Y, 0.9), (N, 0.2)])
    supplied = [
        {
            "model_id": "m",
            "threshold": 0.25,
            "roc_auc_macro": 0.5,
            "accuracy_macro": 0.1,
            "balanced_accuracy_macro": 0.55,
            "f1_macro": 0.6,
            "precision_macro": 0.4,
            "recall_macro": 0.3,
            "matthews_corrcoef_macro": 0.7,
        },
        {
            "model_id": "m",
            "threshold": 0.75,
            "roc_auc_macro": 0.75,
            "accuracy_macro": 0.2,
            "balanced_accuracy_macro": 0.65,
            "f1_macro": 0.8,
            "precision_macro": 0.5,
            "recall_macro": 0.4,
            "matthews_corrcoef_macro": 0.9,
        },
    ]

    (row,) = scoring_summary_rows([run], supplied)

    assert row["best_mcc"] == 0.9
    assert row["best_mcc_threshold"] == 0.75
    assert row["best_f1_threshold"] == 0.75
    assert row["roc_auc_macro"] == 0.5
    assert row["n_items_total"] == 2
    assert row["language_count"] == 1


def test_scoring_summary_skips_a_model_without_sweep_rows_and_keeps_the_rest() -> None:
    absent = _scoring("a/absent", "en", [(Y, 0.9), (N, 0.2)])
    present = _scoring("b/present", "en", [(Y, 0.9), (N, 0.2)])

    rows = scoring_summary_rows([absent, present], threshold_sweep_rows([present]))

    assert [row["model_id"] for row in rows] == ["b/present"]


def test_scoring_summary_averages_throughput_and_takes_the_peak_vram() -> None:
    first = _scoring("m", "en", [(Y, 0.9), (N, 0.2)], throughput=1.2345, peak=100)
    second = _scoring("m", "fr", [(Y, 0.8), (N, 0.3)], throughput=2.0, peak=300)

    (row,) = scoring_summary_rows([first, second])

    assert row["throughput_items_per_second_macro"] == 1.62
    assert row["peak_vram_bytes_max"] == 300


def test_scoring_summary_reports_missing_throughput_and_vram_as_none() -> None:
    run = _scoring("m", "en", [(Y, 0.9), (N, 0.2)], throughput=None, peak=None)

    (row,) = scoring_summary_rows([run])

    assert row["throughput_items_per_second_macro"] is None
    assert row["peak_vram_bytes_max"] is None

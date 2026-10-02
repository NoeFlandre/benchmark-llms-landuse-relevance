"""Exact report values: every column, rounding digit, ordering and file byte is pinned."""

import csv
import io
import json
from pathlib import Path

import pytest

from factories import make_metadata
from landuse_relevance_bench.adapters.results_store import (
    AGGREGATE_COLUMNS,
    LEADERBOARD_COLUMNS,
    SCORING_SUMMARY_COLUMNS,
    THRESHOLD_SWEEP_COLUMNS,
    aggregate_rows,
    leaderboard_rows,
    read_run,
    read_runs,
    run_filename,
    scoring_summary_rows,
    threshold_sweep_rows,
    write_aggregates_csv,
    write_leaderboard_csv,
    write_reports,
    write_run,
)
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import Prediction, RunResult

Y, N = Label.YES, Label.NO
MIXED = ((Y, Y), (Y, Y), (Y, N), (N, Y), (N, N), (N, N), (Y, None))


def _run(
    model_id: str,
    outcomes: tuple[tuple[Label, Label | None], ...],
    language: str = "en",
    **metadata: object,
) -> RunResult:
    predictions = tuple(
        Prediction(
            item_id=f"{i:016x}",
            expected=expected,
            predicted=predicted,
            raw_output="é" if predicted else "?",
            truncated=predicted is None,
        )
        for i, (expected, predicted) in enumerate(outcomes)
    )
    return RunResult(
        metadata=make_metadata(model_id=model_id, language=language, **metadata),
        predictions=predictions,
        metrics=evaluate([p.outcome for p in predictions]),
    )


def _scoring(
    model_id: str, scores: tuple[float, ...], language: str = "en", **metadata: object
) -> RunResult:
    labels = (Y, N, Y, N, Y, N)
    predictions = tuple(
        Prediction(
            item_id=f"{i:016x}",
            expected=expected,
            predicted=Y if score >= 0.5 else N,
            raw_output=f"no={1 - score:.6f} yes={score:.6f} native=0.0",
        )
        for i, (expected, score) in enumerate(zip(labels, scores, strict=False))
    )
    return RunResult(
        metadata=make_metadata(
            model_id=model_id,
            language=language,
            inference="scoring",
            decision_rule="argmax",
            max_new_tokens=0,
            decoding="none",
            **metadata,
        ),
        predictions=predictions,
        metrics=evaluate([p.outcome for p in predictions]),
    )


SCORER_EN = _scoring(
    "s/m", (0.9, 0.6, 0.4, 0.2, 0.7, 0.8), throughput_items_per_second=10.0, peak_vram_bytes=7
)
SCORER_FR = _scoring(
    "s/m", (0.3, 0.1), language="fr", throughput_items_per_second=12.3456, peak_vram_bytes=9
)


def test_a_leaderboard_row_pins_every_column() -> None:
    run = _run(
        "a/m",
        MIXED,
        duration_seconds=1.23456,
        throughput_items_per_second=12.3456,
        peak_vram_bytes=5,
    )
    (row,) = leaderboard_rows([run])
    assert row == {
        "model_id": "a/m",
        "language": "en",
        "n_items": 7,
        "accuracy": 0.5714,
        "balanced_accuracy": 0.6667,
        "f1": 0.6667,
        "precision": 0.6667,
        "recall": 0.6667,
        "matthews_corrcoef": 0.3333,
        "unparsed_rate": 0.1429,
        "truncated": 1,
        "accuracy_ci95": "[0.2505, 0.8418]",
        "precision_ci95": "[0.2077, 0.9385]",
        "recall_ci95": "[0.2077, 0.9385]",
        "f1_ci95": "[0.0000, 1.0000]",
        "matthews_corrcoef_ci95": "[-0.5774, 1.0000]",
        "duration_seconds": 1.23,
        "sentences_per_second": 7 / 1.23456,
        "latency_mean_seconds": None,
        "latency_p50_seconds": None,
        "latency_p95_seconds": None,
        "generated_tokens": None,
        "output_tokens_per_second": None,
        "mean_accept_length": None,
        "draft_accept_rate": None,
        "runtime": "transformers",
        "generation_mode": "static-batched",
        "package_version": "",
        "throughput_items_per_second": 12.35,
        "peak_vram_bytes": 5,
        "model_revision": "abc123",
        "mcnemar_top_run": "a/m",
        "mcnemar_p_vs_top": 1.0,
    }
    assert set(row) == set(LEADERBOARD_COLUMNS)


def test_the_top_run_per_language_is_the_highest_f1() -> None:
    weak = _run("a/weak", ((Y, N), (N, N), (Y, Y)))
    strong = _run("z/strong", ((Y, Y), (N, N), (Y, Y)))
    other = _run("b/french", ((Y, N), (N, N), (Y, Y)), language="fr")
    rows = leaderboard_rows([weak, strong, other])
    assert [(r["model_id"], r["mcnemar_top_run"]) for r in rows] == [
        ("z/strong", "z/strong"),
        ("a/weak", "z/strong"),
        ("b/french", "b/french"),
    ]


def test_aggregate_rows_pin_every_macro_and_spread_value() -> None:
    english = _run("a/m", MIXED)
    french = _run("a/m", ((Y, Y), (N, Y), (N, Y), (N, Y), (Y, N)), language="fr")
    (row,) = aggregate_rows([english, french])
    assert row == {
        "model_id": "a/m",
        "language_count": 2,
        "n_items_total": 12,
        "accuracy_macro": 0.3857,
        "balanced_accuracy_macro": 0.4583,
        "f1_macro": 0.5,
        "precision_macro": 0.4583,
        "recall_macro": 0.5833,
        "matthews_corrcoef_macro": -0.1395,
        "unparsed_rate_macro": 0.0714,
        "f1_min": 0.3333,
        "f1_max": 0.6667,
        "f1_std": 0.1667,
    }
    assert list(row) == list(AGGREGATE_COLUMNS)


def test_aggregate_rows_rank_by_descending_macro_f1() -> None:
    weak = _run("a/weak", ((Y, N), (Y, Y)))
    strong = _run("z/strong", ((Y, Y), (Y, Y)))
    assert [row["model_id"] for row in aggregate_rows([weak, strong])] == ["z/strong", "a/weak"]


def test_aggregate_rows_refuse_a_repeated_model_language_pair() -> None:
    run = _run("a/m", MIXED)
    with pytest.raises(
        ValueError, match=r"^duplicate result for model-language pair \('a/m', 'en'\)$"
    ):
        aggregate_rows([run, _run("a/m", ((Y, Y),), language="fr"), run])


def test_a_threshold_sweep_row_pins_every_column() -> None:
    rows = threshold_sweep_rows([SCORER_EN, SCORER_FR])
    (half,) = (row for row in rows if row["threshold"] == 0.5)
    assert half == {
        "model_id": "s/m",
        "threshold": 0.5,
        "language_count": 2,
        "n_items_total": 8,
        "roc_auc_macro": 0.8333,
        "accuracy_macro": 0.5,
        "balanced_accuracy_macro": 0.5,
        "f1_macro": 0.2857,
        "precision_macro": 0.25,
        "recall_macro": 0.3333,
        "matthews_corrcoef_macro": 0.0,
    }
    assert all(set(row) == set(THRESHOLD_SWEEP_COLUMNS) for row in rows)


def test_the_sweep_skips_only_the_single_class_run_for_roc_auc() -> None:
    single_class = _scoring("s/m", (0.3,), language="de")
    rows = threshold_sweep_rows([single_class, SCORER_EN])
    assert {row["roc_auc_macro"] for row in rows} == {0.6667}


def test_a_scoring_summary_row_pins_every_column() -> None:
    (row,) = scoring_summary_rows([SCORER_EN, SCORER_FR])
    assert row == {
        "model_id": "s/m",
        "language_count": 2,
        "n_items_total": 8,
        "roc_auc_macro": 0.8333,
        "throughput_items_per_second_macro": 11.17,
        "peak_vram_bytes_max": 9,
        "best_mcc": 0.7236,
        "best_mcc_threshold": 0.3,
        "best_f1": 0.875,
        "best_f1_threshold": 0.3,
        "best_balanced_accuracy": 0.8333,
        "best_balanced_accuracy_threshold": 0.3,
        "best_precision": 0.8,
        "best_precision_threshold": 0.3,
        "best_recall": 1.0,
        "best_recall_threshold": 0.0,
    }
    assert set(row) == set(SCORING_SUMMARY_COLUMNS)


def test_the_scoring_summary_reports_missing_performance_as_none() -> None:
    (row,) = scoring_summary_rows([_scoring("s/m", (0.9, 0.1))])
    assert row["throughput_items_per_second_macro"] is None
    assert row["peak_vram_bytes_max"] is None


def test_the_scoring_summary_uses_the_given_sweep_and_skips_models_missing_from_it() -> None:
    first = _scoring("a/first", (0.9, 0.1))
    second = _scoring("b/second", (0.9, 0.1))
    sweep = threshold_sweep_rows([second])
    assert [row["model_id"] for row in scoring_summary_rows([first, second], sweep)] == ["b/second"]
    assert scoring_summary_rows([first, second], []) == []


def test_the_scoring_summary_ranks_by_descending_best_f1() -> None:
    weak = _scoring("a/weak", (0.1, 0.9))
    strong = _scoring("z/strong", (0.9, 0.1))
    ordered = [row["model_id"] for row in scoring_summary_rows([weak, strong])]
    assert ordered == ["z/strong", "a/weak"]


def test_write_run_writes_sorted_indented_unicode_json(tmp_path: Path) -> None:
    run = _run("a/m", ((Y, Y),))
    path = write_run(run, tmp_path / "nested" / "results")
    expected = json.dumps(run.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    assert path.read_bytes() == expected.encode("utf-8")
    assert '\n  "metadata": {\n    "' in expected
    assert '"raw_output": "é"' in expected


def test_run_filename_names_the_rejected_language() -> None:
    with pytest.raises(ValueError, match=r"^invalid result language 'english'$"):
        run_filename("a/m", "english")


def _write_payload(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_read_run_keeps_a_recorded_language_over_the_legacy_default(tmp_path: Path) -> None:
    path = _write_payload(tmp_path / "run.json", _run("a/m", ((Y, Y),), language="fr").to_dict())
    assert read_run(path, legacy_language="en").metadata.language == "fr"


def test_read_run_fills_a_missing_language_from_the_legacy_default(tmp_path: Path) -> None:
    payload = _run("a/m", ((Y, Y),)).to_dict()
    del payload["metadata"]["language"]
    path = _write_payload(tmp_path / "run.json", payload)
    assert read_run(path, legacy_language="de").metadata.language == "de"


def test_read_run_rejects_a_payload_without_metadata(tmp_path: Path) -> None:
    payload = _run("a/m", ((Y, Y),)).to_dict()
    del payload["metadata"]
    path = _write_payload(tmp_path / "run.json", payload)
    with pytest.raises(ValueError, match=r"is not a valid run result: 'metadata'$"):
        read_run(path, legacy_language="en")


def test_read_runs_does_not_invent_a_language_for_active_results(tmp_path: Path) -> None:
    payload = _run("a/m", ((Y, Y),)).to_dict()
    del payload["metadata"]["language"]
    _write_payload(tmp_path / "en" / "a__m.json", payload)
    with pytest.raises(ValueError, match="archive-only run metadata is missing required language"):
        read_runs(tmp_path)


def test_recursive_reads_label_legacy_archive_results_as_english(tmp_path: Path) -> None:
    payload = _run("a/m", ((Y, Y),)).to_dict()
    del payload["metadata"]["language"]
    _write_payload(tmp_path / "archive" / "old.json", payload)
    (run,) = read_runs(tmp_path, recursive=True)
    assert run.metadata.language == "en"


def test_read_runs_refuses_an_active_result_outside_a_language_directory(tmp_path: Path) -> None:
    path = _write_payload(tmp_path / "a__m.json", _run("a/m", ((Y, Y),)).to_dict())
    with pytest.raises(ValueError) as caught:
        read_runs(tmp_path)
    assert str(caught.value) == (
        f"{path} is a legacy/archive-only result; expected a language directory"
    )


def test_write_reports_writes_exact_csv_files_into_new_directories(tmp_path: Path) -> None:
    runs = [_run("a/m", MIXED), SCORER_EN, SCORER_FR]
    leaderboard = tmp_path / "deep" / "reports" / "leaderboard.csv"
    written = write_reports(runs, leaderboard)
    assert written == {
        "leaderboard": leaderboard,
        "aggregates": leaderboard.with_name("aggregates.csv"),
        "threshold_sweep": leaderboard.with_name("threshold_sweep.csv"),
        "scoring_summary": leaderboard.with_name("scoring_summary.csv"),
    }
    expected = {
        "leaderboard": (LEADERBOARD_COLUMNS, leaderboard_rows(runs), "\r\n"),
        "aggregates": (AGGREGATE_COLUMNS, aggregate_rows(runs), "\n"),
        "threshold_sweep": (THRESHOLD_SWEEP_COLUMNS, threshold_sweep_rows(runs), "\n"),
        "scoring_summary": (SCORING_SUMMARY_COLUMNS, scoring_summary_rows(runs), "\n"),
    }
    for name, (columns, rows, terminator) in expected.items():
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator=terminator)
        writer.writeheader()
        writer.writerows(rows)
        assert written[name].read_bytes() == buffer.getvalue().encode("utf-8"), name
    assert len(scoring_summary_rows(runs)) == 1


def test_write_leaderboard_csv_creates_missing_parent_directories(tmp_path: Path) -> None:
    path = write_leaderboard_csv([_run("a/m", MIXED)], tmp_path / "x" / "y" / "board.csv")
    assert path.read_text(encoding="utf-8").startswith("model_id,language,n_items,")


def test_read_runs_refuses_a_result_in_another_language_directory(tmp_path: Path) -> None:
    _write_payload(tmp_path / "en" / "a__m.json", _run("a/m", ((Y, Y),), language="fr").to_dict())
    with pytest.raises(ValueError, match=r"expected a language directory$"):
        read_runs(tmp_path)


def test_write_aggregates_csv_creates_missing_parent_directories(tmp_path: Path) -> None:
    path = write_aggregates_csv([_run("a/m", MIXED)], tmp_path / "x" / "y" / "aggregates.csv")
    assert path.read_text(encoding="utf-8").startswith("model_id,language_count,")

import json
from dataclasses import replace
from pathlib import Path

import pytest

from landuse_relevance_bench.adapters.results_store import (
    leaderboard_rows,
    read_run,
    read_runs,
    run_filename,
    scoring_summary_rows,
    threshold_sweep_rows,
    write_leaderboard_csv,
    write_run,
)
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import Prediction, RunMetadata, RunResult
from landuse_relevance_bench.domain.thresholds import DEFAULT_THRESHOLDS


def _result(
    model_id: str = "LiquidAI/LFM2.5-350M",
    language: str = "en",
    accuracy_pair: tuple = (Label.YES, Label.YES),
):
    metadata = RunMetadata(
        model_id=model_id,
        language=language,
        model_revision="abc123",
        prompt_sha256="p" * 64,
        benchmark_sha256="b" * 64,
        max_new_tokens=8,
        batch_size=16,
        seed=0,
        decoding="greedy",
        dtype="bfloat16",
        started_at="2026-09-13T10:00:00Z",
        duration_seconds=1.0,
    )
    prediction = Prediction(
        item_id="0" * 16, expected=accuracy_pair[0], predicted=accuracy_pair[1], raw_output="yes"
    )
    return RunResult(
        metadata=metadata, predictions=(prediction,), metrics=evaluate([accuracy_pair])
    )


def test_a_written_run_reads_back_identically(tmp_path: Path) -> None:
    path = write_run(_result(), tmp_path)
    assert read_run(path) == _result()


def test_the_filename_is_derived_from_the_model_id(tmp_path: Path) -> None:
    path = write_run(_result(), tmp_path)
    assert path.name == "LiquidAI__LFM2.5-350M.json"
    assert path.parent == tmp_path / "en"


def test_run_filename_flattens_the_namespace_separator() -> None:
    assert run_filename("a/b", "fr") == Path("fr/a__b.json")


def test_two_languages_have_distinct_result_paths(tmp_path: Path) -> None:
    english = write_run(_result(language="en"), tmp_path)
    french = write_run(_result(language="fr"), tmp_path)

    assert english != french
    assert english.read_text(encoding="utf-8") != french.read_text(encoding="utf-8")


def test_the_file_is_pretty_printed_json_with_a_trailing_newline(tmp_path: Path) -> None:
    text = write_run(_result(), tmp_path).read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text)["metadata"]["model_id"] == "LiquidAI/LFM2.5-350M"


def test_writing_the_same_run_twice_is_idempotent(tmp_path: Path) -> None:
    first = write_run(_result(), tmp_path).read_text(encoding="utf-8")
    second = write_run(_result(), tmp_path).read_text(encoding="utf-8")
    assert first == second


def test_the_leaderboard_ranks_models_by_descending_f1(tmp_path: Path) -> None:
    weak = _result("weak/model", accuracy_pair=(Label.YES, Label.NO))
    strong = _result("strong/model", accuracy_pair=(Label.YES, Label.YES))
    rows = leaderboard_rows([weak, strong])
    assert [r["model_id"] for r in rows] == ["strong/model", "weak/model"]


def test_the_leaderboard_csv_has_a_header_and_one_row_per_model(tmp_path: Path) -> None:
    path = write_leaderboard_csv([_result("a/b"), _result("c/d")], tmp_path / "leaderboard.csv")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("model_id,")
    assert len(lines) == 3


def test_detailed_leaderboard_rows_carry_language() -> None:
    (row,) = leaderboard_rows([_result(language="fr")])

    assert row["language"] == "fr"


def test_reading_a_corrupt_run_file_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        read_run(path)


def test_reading_legacy_metadata_explains_that_it_is_archive_only(tmp_path: Path) -> None:
    payload = _result().to_dict()
    del payload["metadata"]["language"]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"archive-only.*language"):
        read_run(path)


def test_recursive_reading_skips_archive_results(tmp_path: Path) -> None:
    write_run(_result(), tmp_path)
    archive = tmp_path / "archive" / "en"
    archive.mkdir(parents=True)
    (archive / "old.json").write_text(json.dumps(_result("old/model").to_dict()), encoding="utf-8")

    runs = read_runs(tmp_path)

    assert [run.metadata.model_id for run in runs] == ["LiquidAI/LFM2.5-350M"]


def test_the_leaderboard_separates_truncated_generations_from_other_failures() -> None:
    """Unparsed covers both; only the truncation count says the model was cut off."""
    metadata = _result().metadata
    predictions = (
        Prediction(
            item_id="a" * 16,
            expected=Label.YES,
            predicted=None,
            raw_output="1. Analyze",
            truncated=True,
        ),
        Prediction(
            item_id="b" * 16,
            expected=Label.YES,
            predicted=None,
            raw_output="I cannot say",
            truncated=False,
        ),
    )
    result = RunResult(
        metadata=metadata,
        predictions=predictions,
        metrics=evaluate([(Label.YES, None), (Label.YES, None)]),
    )
    (row,) = leaderboard_rows([result])
    assert row["unparsed_rate"] == 1.0
    assert row["truncated"] == 1


def _scoring_result(model_id: str, scores: tuple[float, float]) -> RunResult:
    base = _result(model_id=model_id)
    predictions = (
        Prediction(
            item_id="a" * 16,
            expected=Label.YES,
            predicted=Label.YES if scores[0] >= 0.5 else Label.NO,
            raw_output=f"no={1 - scores[0]:.6f} yes={scores[0]:.6f} native=1.0",
        ),
        Prediction(
            item_id="b" * 16,
            expected=Label.NO,
            predicted=Label.YES if scores[1] >= 0.5 else Label.NO,
            raw_output=f"no={1 - scores[1]:.6f} yes={scores[1]:.6f} native=-1.0",
        ),
    )
    return RunResult(
        metadata=replace(
            base.metadata,
            inference="scoring",
            decision_rule="argmax over the native yes/no scores",
            max_new_tokens=0,
            decoding="none",
            prompt_sha256="s" * 64,
            throughput_items_per_second=12.5,
            peak_vram_bytes=987,
        ),
        predictions=predictions,
        metrics=evaluate([(p.expected, p.predicted) for p in predictions]),
    )


def test_threshold_sweep_reads_normalised_relevance_scores_with_native_scores_present() -> None:
    rows = threshold_sweep_rows([_scoring_result("a/model", (0.8, 0.2))])

    assert len(rows) == len(DEFAULT_THRESHOLDS)
    assert rows[0]["roc_auc_macro"] == 1.0
    assert rows[0]["language_count"] == 1


def test_scoring_summary_reports_the_best_metrics_and_performance() -> None:
    (row,) = scoring_summary_rows([_scoring_result("a/model", (0.8, 0.2))])

    assert row["model_id"] == "a/model"
    assert row["best_mcc"] == 1.0
    assert row["best_f1"] == 1.0
    assert row["best_balanced_accuracy"] == 1.0
    assert row["best_precision"] == 1.0
    assert row["best_recall"] == 1.0
    assert row["roc_auc_macro"] == 1.0
    assert row["throughput_items_per_second_macro"] == 12.5
    assert row["peak_vram_bytes_max"] == 987

import json
from pathlib import Path

import pytest

from landuse_relevance_bench.adapters.results_store import (
    leaderboard_rows,
    read_run,
    run_filename,
    write_leaderboard_csv,
    write_run,
)
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import Prediction, RunMetadata, RunResult


def _result(model_id: str = "LiquidAI/LFM2.5-350M", accuracy_pair: tuple = (Label.YES, Label.YES)):
    metadata = RunMetadata(
        model_id=model_id,
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
    assert path.parent == tmp_path
    assert run_filename("a/b") == "a__b.json"


def test_writing_the_same_run_twice_is_idempotent(tmp_path: Path) -> None:
    first = write_run(_result(), tmp_path).read_text(encoding="utf-8")
    second = write_run(_result(), tmp_path).read_text(encoding="utf-8")
    assert first == second


def test_the_leaderboard_ranks_models_by_descending_f1(tmp_path: Path) -> None:
    weak = _result("weak/model", (Label.YES, Label.NO))
    strong = _result("strong/model", (Label.YES, Label.YES))
    rows = leaderboard_rows([weak, strong])
    assert [r["model_id"] for r in rows] == ["strong/model", "weak/model"]


def test_the_leaderboard_csv_has_a_header_and_one_row_per_model(tmp_path: Path) -> None:
    path = write_leaderboard_csv([_result("a/b"), _result("c/d")], tmp_path / "leaderboard.csv")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("model_id,")
    assert len(lines) == 3


def test_reading_a_corrupt_run_file_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        read_run(path)


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


def test_reading_a_run_without_metadata_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "no_metadata.json"
    payload = _result().to_dict()
    del payload["metadata"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"no_metadata\.json"):
        read_run(path)


def test_reading_a_run_with_an_unknown_metadata_key_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "extra_key.json"
    payload = _result().to_dict()
    payload["metadata"]["surprise"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"extra_key\.json"):
        read_run(path)


def test_reading_a_run_whose_metrics_disagree_with_its_predictions_fails(tmp_path: Path) -> None:
    path = tmp_path / "tampered.json"
    payload = _result().to_dict()
    payload["metrics"] = evaluate([(Label.YES, Label.YES), (Label.NO, Label.NO)]).to_dict()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        read_run(path)

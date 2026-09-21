from pathlib import Path

import pytest
from scripts.snapshot_status import snapshot_status

from landuse_relevance_bench.adapters.results_store import write_run
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import Prediction, RunMetadata, RunResult
from landuse_relevance_bench.domain.roster import model_ids


def _result(model_id: str, language: str) -> RunResult:
    metadata = RunMetadata(
        model_id=model_id,
        language=language,
        model_revision="abc123",
        prompt_sha256="p" * 64,
        benchmark_sha256="b" * 64,
        max_new_tokens=4096,
        batch_size=16,
        seed=0,
        decoding="greedy",
        dtype="bfloat16",
        started_at="2026-09-21T10:00:00Z",
        duration_seconds=1.0,
    )
    prediction = Prediction(
        item_id="0" * 16, expected=Label.YES, predicted=Label.YES, raw_output="yes"
    )
    return RunResult(
        metadata=metadata, predictions=(prediction,), metrics=evaluate([(Label.YES, Label.YES)])
    )


def _store(tmp_path: Path, pairs: list[tuple[str, str]]) -> Path:
    for model_id, language in pairs:
        write_run(_result(model_id, language), tmp_path)
    return tmp_path


def test_status_counts_every_model_language_pair_against_the_roster(tmp_path: Path) -> None:
    first, second = model_ids()[0], model_ids()[1]
    status = snapshot_status(
        _store(tmp_path, [(first, "en"), (first, "fr"), (second, "en")]),
        benchmark_name="v3-multilingual",
    )

    assert f"- Completed model-language runs: 3 of {len(model_ids()) * 2}" in status
    assert f"- `{first}`: complete, 2/2 languages" in status
    assert f"- `{second}`: in progress, 1/2 languages" in status


def test_status_names_every_rostered_model_even_when_absent(tmp_path: Path) -> None:
    status = snapshot_status(
        _store(tmp_path, [(model_ids()[0], "en")]), benchmark_name="v3-multilingual"
    )

    for model_id in model_ids():
        assert f"- `{model_id}`:" in status
    assert "0/1 languages" in status


def test_status_reports_completion_only_when_the_sweep_is_whole(tmp_path: Path) -> None:
    pairs = [(model_id, "en") for model_id in model_ids()]
    status = snapshot_status(_store(tmp_path, pairs), benchmark_name="v3-multilingual")

    assert "- Status: complete." in status


def test_status_refuses_an_empty_tree(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no run results"):
        snapshot_status(tmp_path, benchmark_name="v3-multilingual")

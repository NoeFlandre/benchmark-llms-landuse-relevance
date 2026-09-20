import json
from dataclasses import replace
from pathlib import Path

import pytest

from landuse_relevance_bench.adapters.hf_publish import (
    dataset_card,
    publish_results,
    read_published_runs,
)
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import Prediction, RunMetadata, RunResult


def _result(model_id: str = "LiquidAI/LFM2.5-350M", language: str = "en") -> RunResult:
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
        item_id="0" * 16, expected=Label.YES, predicted=Label.YES, raw_output="yes"
    )
    return RunResult(
        metadata=metadata, predictions=(prediction,), metrics=evaluate([(Label.YES, Label.YES)])
    )


class FakeApi:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.uploaded: list[dict] = []

    def create_repo(self, **kwargs) -> None:
        self.created.append(kwargs)

    def upload_folder(self, **kwargs) -> str:
        self.uploaded.append(kwargs)
        return f"https://huggingface.co/datasets/{kwargs['repo_id']}"


def test_the_card_is_a_markdown_leaderboard_naming_every_model() -> None:
    card = dataset_card([_result("a/one"), _result("b/two")], benchmark_name="benchmark.csv")
    assert card.startswith("---")
    assert "| a/one |" in card and "| b/two |" in card
    assert "n_items" not in card
    assert "benchmark.csv" in card
    assert "language" in card and "benchmark_set" in card


def test_the_card_declares_the_prompt_and_benchmark_digests() -> None:
    card = dataset_card([_result()], benchmark_name="benchmark.csv")
    assert "b" * 64 in card
    assert "p" * 64 in card


def test_published_runs_reads_nested_result_folders_in_stable_order(tmp_path: Path) -> None:
    english = tmp_path / "en"
    english.mkdir()
    (english / "root__one.json").write_text(
        json.dumps(_result("a/one").to_dict()), encoding="utf-8"
    )
    extra = tmp_path / "fr"
    extra.mkdir()
    (extra / "nested__two.json").write_text(
        json.dumps(_result("b/two", language="fr").to_dict()), encoding="utf-8"
    )
    archive = tmp_path / "archive" / "en"
    archive.mkdir(parents=True)
    (archive / "old.json").write_text(json.dumps(_result("old/model").to_dict()), encoding="utf-8")
    published = read_published_runs(tmp_path)

    assert [result.metadata.model_id for result in published] == ["a/one", "b/two"]


def test_card_rejects_metrics_that_are_not_derived_from_predictions() -> None:
    result = _result()
    tampered = replace(result, metrics=replace(result.metrics, accuracy=0.0))

    with pytest.raises(ValueError, match="metrics do not match predictions"):
        dataset_card([tampered], benchmark_name="benchmark.csv")


def test_publishing_creates_the_dataset_repository_then_uploads_the_folder(
    tmp_path: Path,
) -> None:
    (tmp_path / "run.json").write_text("{}", encoding="utf-8")
    api = FakeApi()
    url = publish_results("me/bench", tmp_path, [_result()], api=api)
    assert api.created[0]["repo_id"] == "me/bench"
    assert api.created[0]["repo_type"] == "dataset"
    assert api.uploaded[0]["folder_path"] == str(tmp_path)
    assert url.endswith("me/bench")


def test_publishing_writes_the_card_into_the_uploaded_folder(tmp_path: Path) -> None:
    publish_results("me/bench", tmp_path, [_result()], api=FakeApi())
    assert "LiquidAI/LFM2.5-350M" in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_publishing_nothing_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        publish_results("me/bench", tmp_path, [], api=FakeApi())


def test_publishing_refuses_to_upload_an_archive_path(tmp_path: Path) -> None:
    archive = tmp_path / "archive" / "en"
    archive.mkdir(parents=True)
    (archive / "old.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="archive"):
        publish_results("me/bench", tmp_path, [_result()], api=FakeApi())


def test_the_card_leaderboard_shows_the_truncation_count() -> None:
    card = dataset_card([_result()], benchmark_name="benchmark.csv")
    assert "truncated" in card.split("## Scores")[1].split("\n")[2]


def test_card_is_terse_and_has_no_companion_prose() -> None:
    card = dataset_card([_result()], benchmark_name="benchmark.csv")

    assert "companion" not in card.lower()
    assert "configuration" not in card
    assert "Predictions and scores" not in card

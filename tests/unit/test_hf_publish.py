import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from landuse_relevance_bench.adapters.hf_publish import (
    dataset_card,
    publish_results,
    read_published_runs,
)
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import Prediction, RunMetadata, RunResult


def _result(
    model_id: str = "LiquidAI/LFM2.5-350M",
    predictions: tuple[Prediction, ...] | None = None,
    **metadata_overrides: object,
) -> RunResult:
    metadata = replace(
        RunMetadata(
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
        ),
        **metadata_overrides,
    )
    if predictions is None:
        predictions = (
            Prediction(item_id="0" * 16, expected=Label.YES, predicted=Label.YES, raw_output="yes"),
        )
    return RunResult(
        metadata=metadata,
        predictions=predictions,
        metrics=evaluate([(p.expected, p.predicted) for p in predictions]),
    )


def _predictions(pairs: list[tuple[Label, Label | None]], truncated: int = 0) -> tuple:
    return tuple(
        Prediction(
            item_id=f"{i:016x}",
            expected=expected,
            predicted=predicted,
            raw_output="" if predicted is None else str(predicted),
            truncated=i < truncated,
        )
        for i, (expected, predicted) in enumerate(pairs)
    )


def _score_rows(card: str, section: str = "Scores") -> list[dict[str, str]]:
    body = card.split(f"## {section}\n", 1)[1].split("\n## ", 1)[0]
    table = [line for line in body.splitlines() if line.strip().startswith("|")]
    cells = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in table]
    header, rows = cells[0], cells[2:]
    return [dict(zip(header, row, strict=True)) for row in rows]


class FakeApi:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.uploaded: list[dict] = []

    def create_repo(self, **kwargs) -> None:
        self.created.append(kwargs)

    def upload_folder(self, **kwargs) -> str:
        self.uploaded.append(kwargs)
        return f"https://huggingface.co/datasets/{kwargs['repo_id']}"


def test_the_card_declares_the_prompt_and_benchmark_digests() -> None:
    card = dataset_card([_result()], benchmark_name="benchmark.csv")
    assert "b" * 64 in card
    assert "p" * 64 in card


def test_published_runs_reads_nested_result_folders_in_stable_order(tmp_path: Path) -> None:
    (tmp_path / "root__one.json").write_text(
        json.dumps(_result("a/one").to_dict()), encoding="utf-8"
    )
    extra = tmp_path / "recent-models-20260913"
    extra.mkdir()
    (extra / "nested__two.json").write_text(
        json.dumps(_result("b/two").to_dict()), encoding="utf-8"
    )
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


def test_the_card_ranks_one_row_per_model_by_descending_f1() -> None:
    strong = _result("a/strong", _predictions([(Label.YES, Label.YES), (Label.NO, Label.NO)]))
    weak = _result("z/weak", _predictions([(Label.YES, Label.NO), (Label.NO, Label.YES)]))
    rows = _score_rows(dataset_card([weak, strong], benchmark_name="benchmark.csv"))
    assert [row["model_id"] for row in rows] == ["a/strong", "z/weak"]
    assert [float(row["f1"]) for row in rows] == [1.0, 0.0]


def test_the_card_reports_the_truncation_count_by_value() -> None:
    run = _result(
        "a/one",
        _predictions([(Label.YES, None), (Label.NO, None), (Label.YES, Label.YES)], truncated=2),
    )
    (row,) = _score_rows(dataset_card([run], benchmark_name="benchmark.csv"))
    assert row["truncated"] == "2"


def test_the_card_front_matter_declares_what_the_hub_needs() -> None:
    card = dataset_card([_result()], benchmark_name="benchmark.csv")
    front_matter = yaml.safe_load(card.split("---")[1])
    assert front_matter["license"]
    assert front_matter["task_categories"]


def test_the_card_reports_speed_recorded_in_the_predictions() -> None:
    timed = tuple(
        replace(p, latency_seconds=0.5, generated_tokens=10, verify_steps=4)
        for p in _predictions([(Label.YES, Label.YES), (Label.NO, Label.NO)])
    )
    run = _result("a/fast", timed, runtime="sglang", duration_seconds=2.0)
    (row,) = _score_rows(dataset_card([run], benchmark_name="benchmark.csv"), "Speed")
    assert row["runtime"] == "sglang"
    assert row["generated_tokens"] == "20"
    assert row["output_tokens_per_second"] == "10.0"
    assert row["mean_accept_length"] == "2.5"


def test_the_card_leaves_speed_blank_for_a_run_that_did_not_record_it() -> None:
    (row,) = _score_rows(dataset_card([_result()], benchmark_name="benchmark.csv"), "Speed")
    assert row["wall_seconds"] == "1.0"
    assert row["latency_p50_seconds"] == ""


def test_the_card_flags_a_speculative_run_that_changed_the_target_s_output() -> None:
    plain = _result("t/vl", _predictions([(Label.YES, Label.YES), (Label.NO, Label.NO)]))
    drafted = _result(
        "t/vl",
        _predictions([(Label.YES, Label.YES), (Label.NO, Label.YES)]),
        run_id="t/vl+draft",
        draft_model_id="t/draft",
    )
    card = dataset_card([plain, drafted], benchmark_name="benchmark.csv")
    assert "t/vl+draft vs t/vl (same runtime): MISMATCH, 1/2 verdicts" in card


def test_the_card_omits_the_speculative_check_without_a_speculative_run() -> None:
    assert "Speculative" not in dataset_card([_result()], benchmark_name="benchmark.csv")

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from landuse_relevance_bench.adapters.hashing import sha256_of_text
from landuse_relevance_bench.adapters.hf_publish import (
    dataset_card,
    publish_results,
    read_published_runs,
    write_scoring_plots,
    write_viewer_dataset,
)
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import Prediction, RunMetadata, RunResult

PROMPT = "Classify the sentence.\n\nTARGET SENTENCE: {}\n"
PROMPT_SHA256 = sha256_of_text(PROMPT)


def _result(model_id: str = "LiquidAI/LFM2.5-350M", language: str = "en") -> RunResult:
    metadata = RunMetadata(
        model_id=model_id,
        language=language,
        model_revision="abc123",
        prompt_sha256=PROMPT_SHA256,
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


def test_the_card_has_one_aggregate_row_per_model_and_language_count() -> None:
    card = dataset_card(
        [_result("a/one"), _result("a/one", language="fr"), _result("b/two")],
        benchmark_name="benchmark.csv",
        prompt_text=PROMPT,
    )
    assert card.startswith("---")
    assert "| model_id | language_count | n_items_total |" in card
    assert "| a/one | 2 | 2 |" in card
    assert "| b/two | 1 | 1 |" in card
    assert "| language |" not in card
    assert card.count("| a/one |") == 1
    assert card.count("| b/two |") == 1
    assert "| n_items |" not in card
    assert "benchmark.csv" in card


def test_the_card_declares_the_prompt_and_benchmark_digests() -> None:
    card = dataset_card([_result()], benchmark_name="benchmark.csv", prompt_text=PROMPT)
    assert PROMPT_SHA256 in card


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


def test_viewer_export_is_one_neat_multilingual_csv(tmp_path: Path) -> None:
    root = tmp_path / "translations"
    csv_template = (
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url,"
        "source_item_id,language\n"
        '"Forest covers the hill.",yes,Place,h3,1,2,source,region,url,s1,{language}\n'
    )
    files = {}
    for language in ("en", "fr"):
        path = root / language / f"v3-final-{language}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(csv_template.format(language=language), encoding="utf-8")
        files[language] = {
            "path": f"{language}/v3-final-{language}.csv",
            "rows": 1,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    inventory = "\n".join(f"{language}:{files[language]['sha256']}" for language in files)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "test/dataset",
                "revision": "test-revision",
                "split": "train",
                "languages": ["en", "fr"],
                "row_count": 1,
                "source_item_ids": ["s1"],
                "files": files,
                "whole_set_sha256": hashlib.sha256(inventory.encode()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    output = write_viewer_dataset(root, tmp_path / "data" / "benchmark.csv")

    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("item_id,source_item_id,language,sentence,label")
    assert len(lines) == 3
    assert ",en," in lines[1]
    assert ",fr," in lines[2]


def test_card_rejects_metrics_that_are_not_derived_from_predictions() -> None:
    result = _result()
    tampered = replace(result, metrics=replace(result.metrics, accuracy=0.0))

    with pytest.raises(ValueError, match="metrics do not match predictions"):
        dataset_card([tampered], benchmark_name="benchmark.csv", prompt_text=PROMPT)


def test_publishing_creates_the_dataset_repository_then_uploads_the_folder(
    tmp_path: Path,
) -> None:
    (tmp_path / "run.json").write_text("{}", encoding="utf-8")
    api = FakeApi()
    url = publish_results("me/bench", tmp_path, [_result()], api=api, prompt_text=PROMPT)
    assert api.created[0]["repo_id"] == "me/bench"
    assert api.created[0]["repo_type"] == "dataset"
    assert api.uploaded[0]["folder_path"] == str(tmp_path)
    assert url.endswith("me/bench")


def test_publishing_writes_the_card_into_the_uploaded_folder(tmp_path: Path) -> None:
    publish_results("me/bench", tmp_path, [_result()], api=FakeApi(), prompt_text=PROMPT)
    assert "LiquidAI/LFM2.5-350M" in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_publishing_nothing_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        publish_results("me/bench", tmp_path, [], api=FakeApi(), prompt_text=PROMPT)


def test_publishing_refuses_to_upload_an_archive_path(tmp_path: Path) -> None:
    archive = tmp_path / "archive" / "en"
    archive.mkdir(parents=True)
    (archive / "old.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="archive"):
        publish_results("me/bench", tmp_path, [_result()], api=FakeApi(), prompt_text=PROMPT)


def test_the_card_contains_only_aggregate_metrics() -> None:
    card = dataset_card([_result()], benchmark_name="benchmark.csv", prompt_text=PROMPT)
    table_header = card.split("## Aggregate scores")[1].splitlines()[2]

    assert "truncated" not in table_header
    assert "language_count" in table_header


def test_card_is_terse_and_has_no_companion_prose() -> None:
    card = dataset_card([_result()], benchmark_name="benchmark.csv", prompt_text=PROMPT)

    assert "companion" not in card.lower()
    assert "configuration" not in card
    assert "Predictions and scores" not in card


def test_the_card_documents_the_prompt_and_run_settings() -> None:
    card = dataset_card([_result()], benchmark_name="benchmark.csv", prompt_text=PROMPT)
    section = card.split("## Benchmark")[1].split("## Aggregate scores")[0]

    assert PROMPT in section
    assert "greedy decoding" in section
    assert "`max_new_tokens=8`" in section
    assert "| dtype | `bfloat16` |" in section
    assert "batch size 16" in section
    assert "seed 0" in section


def test_the_card_refuses_a_prompt_that_the_runs_did_not_use() -> None:
    with pytest.raises(ValueError, match="prompt text does not match"):
        dataset_card([_result()], benchmark_name="benchmark.csv", prompt_text="something else {}")


def test_the_card_refuses_runs_that_disagree_on_their_settings() -> None:
    result = _result()
    other = replace(result, metadata=replace(result.metadata, dtype="float16"))

    with pytest.raises(ValueError, match="runs disagree on dtype"):
        dataset_card([result, other], benchmark_name="benchmark.csv", prompt_text=PROMPT)


def test_the_card_states_the_per_language_benchmark_digests_are_recorded() -> None:
    card = dataset_card([_result()], benchmark_name="benchmark.csv", prompt_text=PROMPT)

    assert "b" * 64 not in card
    assert "| benchmark hashes | recorded per language run |" in card


def test_the_card_states_the_batch_size_when_the_sweep_agrees_on_one() -> None:
    card = dataset_card([_result()], benchmark_name="benchmark.csv", prompt_text=PROMPT)

    assert "batch size 16" in card


def test_the_card_points_at_the_runs_when_batch_sizes_differ() -> None:
    batched = _result("fits/in-a-batch")
    hybrid = _result("hybrid/ssm")
    unbatched = replace(hybrid, metadata=replace(hybrid.metadata, batch_size=1))
    card = dataset_card([batched, unbatched], benchmark_name="benchmark.csv", prompt_text=PROMPT)

    assert "batch size varying by model, recorded per run" in card
    assert "batch size 16" not in card


def test_the_card_still_refuses_settings_that_shape_a_verdict() -> None:
    result = _result("a/one")
    other = replace(
        _result("b/two"), metadata=replace(_result("b/two").metadata, max_new_tokens=32)
    )

    with pytest.raises(ValueError, match="runs disagree on token budget"):
        dataset_card([result, other], benchmark_name="benchmark.csv", prompt_text=PROMPT)


SCORER_PROMPT = "<Instruct>: judge it\n<Query>: is it?\n<Document>: {}\n"
SCORER_PROMPT_SHA256 = sha256_of_text(SCORER_PROMPT)


def _scored(model_id: str, language: str = "en") -> RunResult:
    result = _result(model_id, language)
    predictions = tuple(
        replace(prediction, raw_output="no=0.100000 yes=0.900000")
        for prediction in result.predictions
    )
    return replace(
        result,
        predictions=predictions,
        metadata=replace(
            result.metadata,
            inference="scoring",
            decision_rule="argmax over the native yes/no scores",
            max_new_tokens=0,
            prompt_sha256=SCORER_PROMPT_SHA256,
            sequence_length=8192,
            throughput_items_per_second=10.0,
            peak_vram_bytes=1024**3,
        ),
    )


def test_scoring_models_are_reported_in_their_own_section() -> None:
    card = dataset_card(
        [_result("gen/one"), _scored("score/two")],
        benchmark_name="benchmark.csv",
        prompt_text=PROMPT,
        scorer_prompt_text=SCORER_PROMPT,
    )
    generative, scoring = card.split("## Scoring models")

    assert "| gen/one |" in generative and "| gen/one |" not in scoring
    assert "| score/two |" in scoring and "| score/two |" not in generative
    assert "argmax over the native yes/no scores" in scoring


def test_the_card_says_why_scoring_models_never_look_unparsed() -> None:
    card = dataset_card(
        [_result("gen/one"), _scored("score/two")],
        benchmark_name="benchmark.csv",
        prompt_text=PROMPT,
        scorer_prompt_text=SCORER_PROMPT,
    )

    assert "by construction" not in card


def test_generation_settings_ignore_the_scoring_runs() -> None:
    card = dataset_card(
        [_result("gen/one"), _scored("score/two")],
        benchmark_name="benchmark.csv",
        prompt_text=PROMPT,
        scorer_prompt_text=SCORER_PROMPT,
    )

    assert "`max_new_tokens=8`" in card


def test_a_card_of_only_scoring_runs_is_refused() -> None:
    with pytest.raises(ValueError, match="without any generative run"):
        dataset_card(
            [_scored("score/two")],
            benchmark_name="benchmark.csv",
            prompt_text=PROMPT,
            scorer_prompt_text=SCORER_PROMPT,
        )


def test_a_sweep_with_no_scoring_models_has_no_scoring_section() -> None:
    card = dataset_card([_result("gen/one")], benchmark_name="benchmark.csv", prompt_text=PROMPT)

    assert "## Scoring models" not in card


def test_the_card_shows_the_reranker_input_it_was_actually_given() -> None:
    card = dataset_card(
        [_result("gen/one"), _scored("score/two")],
        benchmark_name="benchmark.csv",
        prompt_text=PROMPT,
        scorer_prompt_text=SCORER_PROMPT,
    )
    scoring = card.split("## Scoring models")[1]

    assert SCORER_PROMPT in scoring
    assert SCORER_PROMPT_SHA256 in scoring
    assert "Judge whether the Document meets the requirements" not in scoring
    assert "Scoring prompt" in scoring
    assert PROMPT not in scoring


def test_the_card_refuses_a_scoring_prompt_the_runs_did_not_use() -> None:
    with pytest.raises(ValueError, match="scoring prompt text does not match"):
        dataset_card(
            [_result("gen/one"), _scored("score/two")],
            benchmark_name="benchmark.csv",
            prompt_text=PROMPT,
            scorer_prompt_text="a different reranker prompt {}",
        )


def test_the_card_warns_that_a_reranker_score_is_not_calibrated_to_a_boundary() -> None:
    card = dataset_card(
        [_result("gen/one"), _scored("score/two")],
        benchmark_name="benchmark.csv",
        prompt_text=PROMPT,
        scorer_prompt_text=SCORER_PROMPT,
    )
    scoring = card.split("## Scoring models")[1]

    assert "not calibrated" not in scoring
    assert "threshold_sweep.csv" in scoring


def test_the_card_is_minimal_but_keeps_benchmark_settings_and_sequence_length() -> None:
    card = dataset_card(
        [_result("gen/one"), _scored("score/two")],
        benchmark_name="benchmark.csv",
        prompt_text=PROMPT,
        scorer_prompt_text=SCORER_PROMPT,
    )

    assert "85" not in card
    assert "items per language" in card
    assert "sequence length" in card.lower()
    assert "max_new_tokens=8" in card
    assert "Predictions and scores" not in card


def test_the_card_keeps_a_compact_model_specific_scoring_setup() -> None:
    card = dataset_card(
        [
            _result("gen/one"),
            _scored("Alibaba-NLP/gte-multilingual-reranker-base"),
            _scored("mixedbread-ai/mxbai-rerank-base-v2"),
            _scored("convaiinnovations/laya-multilingual"),
        ],
        benchmark_name="benchmark.csv",
        prompt_text=PROMPT,
        scorer_prompt_text=SCORER_PROMPT,
    )

    assert "### Scoring setup" in card
    assert "prompt + sentence pair" in card
    assert "official query/document turn" in card
    assert "JSON state + one `noul` question" in card
    assert "Laya `noul` yes probability" in card
    assert "| sequence length |" in card


def test_scoring_plots_are_deterministic_and_include_performance(tmp_path: Path) -> None:
    results = [_result("gen/one"), _scored("score/two"), _scored("score/one", "fr")]

    first = write_scoring_plots(results, tmp_path / "first")
    second = write_scoring_plots(results, tmp_path / "second")

    assert [path.name for path in first] == ["quality_metrics.svg", "performance.svg"]
    assert [path.read_text(encoding="utf-8") for path in first] == [
        path.read_text(encoding="utf-8") for path in second
    ]
    assert "score/one" in first[0].read_text(encoding="utf-8")
    assert "items / s" in second[1].read_text(encoding="utf-8")


def test_the_card_links_to_deterministic_plots() -> None:
    card = dataset_card(
        [_result("gen/one"), _scored("score/two")],
        benchmark_name="benchmark.csv",
        prompt_text=PROMPT,
        scorer_prompt_text=SCORER_PROMPT,
        plot_files=("plots/quality_metrics.svg", "plots/performance.svg"),
    )

    assert "## Plots" in card
    assert "![Best scoring metrics](plots/quality_metrics.svg)" in card
    assert "![Inference performance](plots/performance.svg)" in card

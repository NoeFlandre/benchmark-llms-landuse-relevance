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


def _result_with_outcomes(model_id: str, outcomes: tuple[tuple[Label, Label], ...]) -> RunResult:
    predictions = tuple(
        Prediction(
            item_id=f"{index:016x}",
            expected=expected,
            predicted=predicted,
            raw_output=predicted.value,
        )
        for index, (expected, predicted) in enumerate(outcomes)
    )
    return replace(
        _result(model_id),
        predictions=predictions,
        metrics=evaluate([(expected, predicted) for expected, predicted in outcomes]),
    )


def _scoring_result(model_id: str, scores: tuple[float, ...], *, vram_gib: int) -> RunResult:
    expected = (Label.YES, Label.YES, Label.NO, Label.NO)
    predictions = tuple(
        Prediction(
            item_id=f"{index:016x}",
            expected=label,
            predicted=Label.YES if score >= 0.5 else Label.NO,
            raw_output=f"no={1 - score:.6f} yes={score:.6f}",
        )
        for index, (label, score) in enumerate(zip(expected, scores, strict=True))
    )
    result = _scored(model_id)
    return replace(
        result,
        predictions=predictions,
        metrics=evaluate(
            [(prediction.expected, prediction.predicted) for prediction in predictions]
        ),
        metadata=replace(
            result.metadata,
            throughput_items_per_second=float(4 - vram_gib),
            peak_vram_bytes=vram_gib * 1024**3,
        ),
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
    assert "| model_id | language_count | accuracy_macro |" in card
    assert "| a/one | 2 |" in card
    assert "| b/two | 1 |" in card
    assert "| language |" not in card
    assert card.count("| a/one |") == 1
    assert card.count("| b/two |") == 1
    assert "| n_items |" not in card
    assert "benchmark.csv" in card


def test_the_card_emphasizes_best_and_second_best_aggregate_metrics() -> None:
    card = dataset_card(
        [
            _result_with_outcomes("a/best", ((Label.YES, Label.YES), (Label.NO, Label.NO))),
            _result_with_outcomes("b/second", ((Label.YES, Label.YES), (Label.NO, Label.YES))),
            _result_with_outcomes("c/last", ((Label.YES, Label.NO), (Label.NO, Label.YES))),
        ],
        benchmark_name="benchmark.csv",
        prompt_text=PROMPT,
    )
    aggregate = card.split("## Aggregate scores")[1]
    header = next(line for line in aggregate.splitlines() if line.startswith("| model_id"))
    best = next(line for line in aggregate.splitlines() if line.startswith("| a/best |"))
    second = next(line for line in aggregate.splitlines() if line.startswith("| b/second |"))
    last = next(line for line in aggregate.splitlines() if line.startswith("| c/last |"))
    columns = [cell.strip() for cell in header.strip("|").split("|")]
    best_cells = dict(
        zip(columns, (cell.strip() for cell in best.strip("|").split("|")), strict=True)
    )
    second_cells = dict(
        zip(columns, (cell.strip() for cell in second.strip("|").split("|")), strict=True)
    )
    last_cells = dict(
        zip(columns, (cell.strip() for cell in last.strip("|").split("|")), strict=True)
    )

    assert best_cells["accuracy_macro"] == "**1.0**"
    assert second_cells["accuracy_macro"] == "<u>0.5</u>"
    assert best_cells["matthews_corrcoef_macro"] == "**1.0**"
    assert second_cells["recall_macro"] == "**1.0**"
    assert last_cells["recall_macro"] == "<u>0.0</u>"


def test_the_card_keeps_the_prompt_without_repeating_its_hash() -> None:
    card = dataset_card([_result()], benchmark_name="benchmark.csv", prompt_text=PROMPT)

    assert PROMPT in card
    assert PROMPT_SHA256 not in card


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


def test_publishing_removes_legacy_plots_locally_and_remotely(tmp_path: Path) -> None:
    plots = tmp_path / "plots"
    plots.mkdir()
    (plots / "quality_metrics.svg").write_text("old quality plot", encoding="utf-8")
    (plots / "performance.svg").write_text("old performance plot", encoding="utf-8")
    api = FakeApi()

    publish_results("me/bench", tmp_path, [_result()], api=api, prompt_text=PROMPT)

    assert not plots.exists()
    assert api.uploaded[0]["delete_patterns"] == [
        "plots/quality_metrics.svg",
        "plots/performance.svg",
    ]
    card = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "## Plots" not in card
    assert "![" not in card


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
    table_header = next(
        line
        for line in card.split("## Aggregate scores")[1].splitlines()
        if line.startswith("| model_id")
    )

    assert "truncated" not in table_header
    assert "language_count" in table_header


def test_card_is_terse_and_has_no_companion_prose() -> None:
    card = dataset_card([_result()], benchmark_name="benchmark.csv", prompt_text=PROMPT)

    assert "companion" not in card.lower()
    assert "configuration" not in card
    assert "Predictions and scores" not in card


def test_the_card_documents_the_prompt_and_run_settings() -> None:
    card = dataset_card([_result()], benchmark_name="benchmark.csv", prompt_text=PROMPT)
    section = card.split("## Task and prompt")[1].split("## Aggregate scores")[0]

    assert PROMPT in section
    assert "greedy" in section
    assert "`max_new_tokens=8`" in section
    assert "bfloat16" in section
    assert "batch 16" in section
    assert "seed 0" in section


def test_the_card_refuses_a_prompt_that_the_runs_did_not_use() -> None:
    with pytest.raises(ValueError, match="prompt text does not match"):
        dataset_card([_result()], benchmark_name="benchmark.csv", prompt_text="something else {}")


def test_the_card_refuses_runs_that_disagree_on_their_settings() -> None:
    result = _result()
    other = replace(result, metadata=replace(result.metadata, dtype="float16"))

    with pytest.raises(ValueError, match="runs disagree on dtype"):
        dataset_card([result, other], benchmark_name="benchmark.csv", prompt_text=PROMPT)


def test_the_card_omits_internal_hash_notes() -> None:
    card = dataset_card([_result()], benchmark_name="benchmark.csv", prompt_text=PROMPT)

    assert "b" * 64 not in card
    assert "benchmark hashes" not in card


def test_the_card_states_the_batch_size_when_the_sweep_agrees_on_one() -> None:
    card = dataset_card([_result()], benchmark_name="benchmark.csv", prompt_text=PROMPT)

    assert "batch 16" in card


def test_the_card_points_at_the_runs_when_batch_sizes_differ() -> None:
    batched = _result("fits/in-a-batch")
    hybrid = _result("hybrid/ssm")
    unbatched = replace(hybrid, metadata=replace(hybrid.metadata, batch_size=1))
    card = dataset_card([batched, unbatched], benchmark_name="benchmark.csv", prompt_text=PROMPT)

    assert "batch varies by model" in card
    assert "batch 16" not in card


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


def test_scoring_summary_emphasizes_best_values_and_lower_vram() -> None:
    card = dataset_card(
        [
            _result_with_outcomes(
                "gen/one",
                (
                    (Label.YES, Label.YES),
                    (Label.YES, Label.YES),
                    (Label.NO, Label.NO),
                    (Label.NO, Label.NO),
                ),
            ),
            _scoring_result("score/best", (0.9, 0.8, 0.2, 0.1), vram_gib=1),
            _scoring_result("score/second", (0.9, 0.4, 0.7, 0.2), vram_gib=2),
            _scoring_result("score/last", (0.2, 0.1, 0.9, 0.8), vram_gib=3),
        ],
        benchmark_name="benchmark.csv",
        prompt_text=PROMPT,
        scorer_prompt_text=SCORER_PROMPT,
    )
    summary = card.split("### Best thresholded scoring metrics")[1]
    header = next(line for line in summary.splitlines() if line.startswith("| model |"))
    best = next(line for line in summary.splitlines() if line.startswith("| score/best |"))
    second = next(line for line in summary.splitlines() if line.startswith("| score/second |"))
    columns = [cell.strip() for cell in header.strip("|").split("|")]
    best_cells = dict(
        zip(columns, (cell.strip() for cell in best.strip("|").split("|")), strict=True)
    )
    second_cells = dict(
        zip(columns, (cell.strip() for cell in second.strip("|").split("|")), strict=True)
    )

    assert best_cells["F1 @ threshold"].startswith("**")
    assert second_cells["F1 @ threshold"].startswith("<u>")
    assert best_cells["items/s"] == "**3.00**"
    assert second_cells["items/s"] == "<u>2.00</u>"
    assert best_cells["peak VRAM (GiB)"] == "**1.00**"
    assert second_cells["peak VRAM (GiB)"] == "<u>2.00</u>"


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
    assert SCORER_PROMPT_SHA256 not in scoring
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
    assert "item/language" in card
    assert "sequence length" in card.lower()
    assert "max_new_tokens=8" in card
    assert "Predictions and scores" not in card
    assert "| setting | value |" not in card
    assert "result layout" not in card
    assert "dataset viewer |" not in card


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
    assert "JSON state + 4 `noul` questions/call" in card
    assert "Laya `noul` yes probability" in card
    assert "sequence length (tokens)" in card
    assert "bfloat16; batch 16; seed 0" in card
    assert "abc123" in card
    assert "peak VRAM (GiB)" in card
    assert "1073741824" not in card
    assert "| Qwen/Qwen3-Reranker-0.6B |" not in card


def test_scoring_setup_does_not_repeat_the_score_formula_as_its_cutoff() -> None:
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
    setup_rows = {
        model_id: next(line for line in card.splitlines() if line.startswith(f"| {model_id} |"))
        for model_id in (
            "Alibaba-NLP/gte-multilingual-reranker-base",
            "mixedbread-ai/mxbai-rerank-base-v2",
            "convaiinnovations/laya-multilingual",
        )
    }

    assert (
        "sigmoid relevance logit; yes if score ≥ 0.5"
        in setup_rows["Alibaba-NLP/gte-multilingual-reranker-base"]
    )
    assert (
        "sigmoid(1-logit - 0-logit - 4.5); yes if score ≥ 0.5"
        in setup_rows["mixedbread-ai/mxbai-rerank-base-v2"]
    )
    assert (
        "Laya `noul` yes probability; yes if score ≥ 0.5"
        in setup_rows["convaiinnovations/laya-multilingual"]
    )


def test_scoring_setup_describes_settings_that_vary_between_runs() -> None:
    float16_run = _scored("convaiinnovations/laya-multilingual", language="en")
    float16_run = replace(float16_run, metadata=replace(float16_run.metadata, dtype="float16"))
    bfloat16_run = _scored("convaiinnovations/laya-multilingual", language="fr")
    card = dataset_card(
        [_result("gen/one"), float16_run, bfloat16_run],
        benchmark_name="benchmark.csv",
        prompt_text=PROMPT,
        scorer_prompt_text=SCORER_PROMPT,
    )
    laya_setup = next(
        line
        for line in card.splitlines()
        if line.startswith("| convaiinnovations/laya-multilingual |")
    )

    assert "varies across runs (dtype is recorded per run)" in laya_setup
    assert "varies by model" not in laya_setup


def test_scoring_summary_pairs_each_best_metric_with_its_threshold() -> None:
    card = dataset_card(
        [_result("gen/one"), _scored("score/two")],
        benchmark_name="benchmark.csv",
        prompt_text=PROMPT,
        scorer_prompt_text=SCORER_PROMPT,
    )
    summary = card.split("### Best thresholded scoring metrics")[1]
    header = summary.splitlines()[2]

    assert header == (
        "| model | languages | MCC @ threshold | F1 @ threshold | "
        "balanced accuracy @ threshold | precision @ threshold | recall @ threshold | "
        "ROC-AUC | items/s | peak VRAM (GiB) |"
    )
    assert "| score/two |" in summary
    score_row = next(line for line in summary.splitlines() if line.startswith("| score/two |"))
    assert score_row.count(" @ ") == 5
    assert "| **10.00** | **1.00** |" in score_row
    assert "best_mcc_threshold" not in header
    assert "threshold_sweep.csv" in card


def test_public_card_omits_redundant_scoring_table_and_aggregate_columns() -> None:
    card = dataset_card(
        [_result("gen/one"), _scored("score/two")],
        benchmark_name="benchmark.csv",
        prompt_text=PROMPT,
        scorer_prompt_text=SCORER_PROMPT,
    )
    aggregate = card.split("## Aggregate scores")[1].split("## Scoring models")[0]
    scoring = card.split("## Scoring models")[1]

    assert "| model_id | language_count | accuracy_macro | balanced_accuracy_macro |" in aggregate
    assert "n_items_total" not in aggregate
    assert "f1_min" not in aggregate
    assert "unparsed_rate_macro" not in aggregate
    assert "| model_id | language_count |" not in scoring
    assert "thresholds are selected on this benchmark" in scoring


def test_the_card_uses_model_defined_for_legacy_missing_sequence_lengths() -> None:
    base = _scored("Qwen/Qwen3-Reranker-0.6B")
    scored = replace(base, metadata=replace(base.metadata, sequence_length=None))
    card = dataset_card(
        [_result("gen/one"), scored],
        benchmark_name="benchmark.csv",
        prompt_text=PROMPT,
        scorer_prompt_text=SCORER_PROMPT,
    )

    assert "| Qwen/Qwen3-Reranker-0.6B |" in card
    assert (
        "| Qwen/Qwen3-Reranker-0.6B | causal-LM reranker: manual yes/no reranker turn | "
        "yes/no next-token probability; argmax over the native yes/no scores | "
        "model-defined |"
    ) in card


def test_the_card_explicitly_selects_the_viewer_split() -> None:
    card = dataset_card(
        [_result("gen/one")],
        benchmark_name="benchmark.csv",
        prompt_text=PROMPT,
        viewer_file="data/train.csv",
    )

    assert "configs:" in card
    assert "path: data/train.csv" in card
    assert "data_dir:" not in card


def test_the_card_uses_an_explicit_viewer_file_path() -> None:
    card = dataset_card(
        [_result("gen/one")],
        benchmark_name="benchmark.csv",
        prompt_text=PROMPT,
        viewer_file="data/train.csv",
    )
    metadata = card.split("---", 2)[1]

    assert "data_dir:" not in metadata
    assert "path: data/train.csv" in metadata
    assert "path: train.csv" not in metadata


def test_the_card_uses_one_compact_line_for_shared_benchmark_settings() -> None:
    card = dataset_card([_result()], benchmark_name="benchmark.csv", prompt_text=PROMPT)
    benchmark = card.split("## Task and prompt")[1].split("## Aggregate scores")[0]

    assert "| setting | value |" not in benchmark
    assert "1 language x 1 item/language" in card
    assert "`yes`/`no`" in card
    assert "English" in benchmark
    assert "greedy" in benchmark
    assert "max_new_tokens=8" in benchmark
    assert "bfloat16" in benchmark
    assert "batch 16" in benchmark


def test_the_card_contains_no_plot_content() -> None:
    card = dataset_card(
        [_result("gen/one"), _scored("score/two")],
        benchmark_name="benchmark.csv",
        prompt_text=PROMPT,
        scorer_prompt_text=SCORER_PROMPT,
    )

    assert "## Plots" not in card
    assert "![" not in card


def test_the_card_has_neat_spacing_between_sections() -> None:
    card = dataset_card(
        [_result("gen/one"), _scored("score/two")],
        benchmark_name="benchmark.csv",
        prompt_text=PROMPT,
        scorer_prompt_text=SCORER_PROMPT,
    )

    assert "\n\n\n" not in card

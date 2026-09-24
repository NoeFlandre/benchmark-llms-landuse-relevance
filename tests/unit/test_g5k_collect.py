from pathlib import Path

import pytest
from scripts.g5k_collect import (
    CollectionError,
    SiteSpec,
    allocate_pairs,
    collect_results,
    expected_pairs_for_models,
)

from landuse_relevance_bench.adapters.results_store import write_run
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import Prediction, RunMetadata, RunResult
from landuse_relevance_bench.domain.sharding import model_language_pairs


def _result(model_id: str, language: str, source_commit: str = "commit-a") -> RunResult:
    metadata = RunMetadata(
        model_id=model_id,
        language=language,
        model_revision="model-rev",
        prompt_sha256="p" * 64,
        benchmark_sha256="b" * 64,
        max_new_tokens=4096,
        batch_size=16,
        seed=0,
        decoding="greedy",
        dtype="bfloat16",
        started_at="2026-09-20T10:00:00Z",
        duration_seconds=1.0,
        source_commit=source_commit,
    )
    return RunResult(
        metadata=metadata,
        predictions=(
            Prediction(
                item_id="0" * 16,
                expected=Label.YES,
                predicted=Label.YES,
                raw_output="yes",
            ),
        ),
        metrics=evaluate([(Label.YES, Label.YES)]),
    )


def test_weighted_site_allocation_is_deterministic_and_complete() -> None:
    pairs = model_language_pairs(["a/model", "b/model"], ["en", "fr", "de"])
    sites = (SiteSpec("nancy", 2), SiteSpec("grenoble", 1))

    allocation = allocate_pairs(pairs, sites)

    assert [len(allocation[site.name]) for site in sites] == [4, 2]
    assert {pair for site in sites for pair in allocation[site.name]} == set(pairs)
    assert set(allocation["nancy"]).isdisjoint(allocation["grenoble"])


def test_collection_can_target_scoring_models_without_changing_default_roster() -> None:
    pairs = expected_pairs_for_models(
        (
            "convaiinnovations/laya-multilingual",
            "Alibaba-NLP/gte-multilingual-reranker-base",
        ),
        ("en", "fr"),
    )

    assert pairs == (
        ("Alibaba-NLP/gte-multilingual-reranker-base", "en"),
        ("Alibaba-NLP/gte-multilingual-reranker-base", "fr"),
        ("convaiinnovations/laya-multilingual", "en"),
        ("convaiinnovations/laya-multilingual", "fr"),
    )


def test_duplicate_pairs_fail_during_collection(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_run(_result("a/model", "en"), first)
    write_run(_result("a/model", "en"), second)

    with pytest.raises(CollectionError, match="duplicate"):
        collect_results(
            {"first": first, "second": second},
            expected_pairs=(("a/model", "en"),),
            output=tmp_path / "merged",
        )


def test_mixed_source_commits_fail_during_collection(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_run(_result("a/model", "en", "commit-a"), first)
    write_run(_result("a/model", "fr", "commit-b"), second)

    with pytest.raises(CollectionError, match="source commit"):
        collect_results(
            {"first": first, "second": second},
            expected_pairs=(("a/model", "en"), ("a/model", "fr")),
            output=tmp_path / "merged",
        )


def test_missing_pairs_are_reported(tmp_path: Path) -> None:
    site = tmp_path / "site"
    write_run(_result("a/model", "en"), site)

    with pytest.raises(CollectionError, match=r"missing.*fr"):
        collect_results(
            {"site": site},
            expected_pairs=(("a/model", "en"), ("a/model", "fr")),
            output=tmp_path / "merged",
        )


def test_complete_union_is_merged_after_validation(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_run(_result("a/model", "en"), first)
    write_run(_result("a/model", "fr"), second)

    report = collect_results(
        {"first": first, "second": second},
        expected_pairs=(("a/model", "en"), ("a/model", "fr")),
        output=tmp_path / "merged",
    )

    assert report.pairs == (("a/model", "en"), ("a/model", "fr"))
    assert (tmp_path / "merged" / "en" / "a__model.json").is_file()
    assert (tmp_path / "merged" / "fr" / "a__model.json").is_file()

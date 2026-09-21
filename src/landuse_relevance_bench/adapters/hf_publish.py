"""Publishing run results to a Hugging Face dataset repository."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from landuse_relevance_bench.adapters.results_store import (
    AGGREGATE_COLUMNS,
    aggregate_rows,
    read_runs,
)
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import RunResult, outcomes_of

CARD_COLUMNS = AGGREGATE_COLUMNS


class DatasetHub(Protocol):
    """The slice of :class:`huggingface_hub.HfApi` this project depends on."""

    def create_repo(self, **kwargs: Any) -> Any: ...
    def upload_folder(self, **kwargs: Any) -> Any: ...


def read_published_runs(results_dir: Path) -> tuple[RunResult, ...]:
    """Read every result below ``results_dir`` in stable model-id order."""
    runs = read_runs(results_dir)
    return tuple(
        sorted(runs, key=lambda result: (result.metadata.model_id, result.metadata.language))
    )


def dataset_card(
    results: Sequence[RunResult],
    *,
    benchmark_name: str,
) -> str:
    """Build a terse card whose scores are recomputed from every prediction."""
    if not results:
        raise ValueError("cannot build a card from an empty set of results")
    for result in results:
        derived = evaluate(outcomes_of(result.predictions))
        if derived != result.metrics:
            raise ValueError(f"{result.metadata.model_id} metrics do not match predictions")
    rows = aggregate_rows(results)
    reference = results[0].metadata
    header = "| " + " | ".join(CARD_COLUMNS) + " |"
    divider = "|" + "|".join(["---"] * len(CARD_COLUMNS)) + "|"
    body = "\n".join(
        "| " + " | ".join(str(row[column]) for column in CARD_COLUMNS) + " |" for row in rows
    )
    return f"""---
license: mit
task_categories:
- text-classification
tags:
- land-use
- land-cover
- remote-sensing
- llm-benchmark
---

# Land-use relevance benchmark

`{benchmark_name}`; model-level macro averages over the completed languages;
greedy decoding;
`max_new_tokens={reference.max_new_tokens}`; seed {reference.seed}.
Scores are recomputed from the published predictions. `language_count` reports how many
language checkpoints contributed to each model row.

- Benchmark sha256: `{reference.benchmark_sha256}`; prompt sha256: `{reference.prompt_sha256}`
- Code: https://github.com/NoeFlandre/benchmark-llms-landuse-relevance

## Aggregate scores

{header}
{divider}
{body}
"""


def publish_results(
    repo_id: str,
    results_dir: Path,
    results: Sequence[RunResult],
    *,
    api: DatasetHub | None = None,
    private: bool = False,
    commit_message: str = "Publish small-LLM land-use relevance benchmark results",
    benchmark_name: str = "v3-multilingual",
) -> str:
    """Write the card next to the results, then push the whole folder to the Hub."""
    if not results:
        raise ValueError("refusing to publish an empty set of results")
    _reject_archive_paths(results_dir)
    hub = api if api is not None else _default_api()
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "README.md").write_text(
        dataset_card(
            results,
            benchmark_name=benchmark_name,
        ),
        encoding="utf-8",
    )
    hub.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    hub.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(results_dir),
        commit_message=commit_message,
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def _default_api() -> DatasetHub:
    from huggingface_hub import HfApi

    # HfApi satisfies DatasetHub in practice; its **kwargs signatures are wider than
    # the protocol can express.
    return cast(DatasetHub, HfApi())


def _reject_archive_paths(results_dir: Path) -> None:
    paths = results_dir.rglob("*") if results_dir.exists() else ()
    for path in paths:
        if "archive" in path.relative_to(results_dir).parts:
            raise ValueError(f"refusing to upload archive path: {path}")

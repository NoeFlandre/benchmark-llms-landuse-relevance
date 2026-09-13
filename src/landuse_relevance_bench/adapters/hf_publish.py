"""Publishing run results to a Hugging Face dataset repository."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from landuse_relevance_bench.adapters.results_store import LEADERBOARD_COLUMNS, leaderboard_rows
from landuse_relevance_bench.domain.records import RunResult

CARD_COLUMNS = (
    "model_id",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "matthews_corrcoef",
    "unparsed_rate",
)


class DatasetHub(Protocol):
    """The slice of :class:`huggingface_hub.HfApi` this project depends on."""

    def create_repo(self, **kwargs: Any) -> Any: ...
    def upload_folder(self, **kwargs: Any) -> Any: ...


def dataset_card(results: Sequence[RunResult], *, benchmark_name: str) -> str:
    """A dataset card whose leaderboard is generated from the results themselves."""
    rows = leaderboard_rows(results)
    reference = results[0].metadata
    n_items = results[0].metrics.n_items
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

# Land-use relevance: small-LLM benchmark results

Predictions and scores for small open-weight LLMs asked to judge whether a sentence
about a place carries land-use, land-cover, or geographic-environment signal that
could be observed by remote sensing.

- Benchmark: `{benchmark_name}` ({n_items} labelled sentences)
- Benchmark sha256: `{reference.benchmark_sha256}`
- Prompt sha256: `{reference.prompt_sha256}`
- Decoding: {reference.decoding}, `max_new_tokens={reference.max_new_tokens}`, \
seed {reference.seed}
- Code: https://github.com/NoeFlandre/benchmark-llms-landuse-relevance

## Leaderboard

{header}
{divider}
{body}

## Files

- `<namespace>__<model>.json` — one file per model: run metadata, every raw generation,
  the parsed verdict, and the scores computed from exactly those predictions.
- `leaderboard.csv` — the table above, with columns {", ".join(LEADERBOARD_COLUMNS)}.

Generations the model did not express as `yes`/`no` are counted as errors, never dropped.
"""


def publish_results(
    repo_id: str,
    results_dir: Path,
    results: Sequence[RunResult],
    *,
    api: DatasetHub | None = None,
    private: bool = False,
    commit_message: str = "Publish small-LLM land-use relevance benchmark results",
    benchmark_name: str = "benchmark.csv",
) -> str:
    """Write the card next to the results, then push the whole folder to the Hub."""
    if not results:
        raise ValueError("refusing to publish an empty set of results")
    hub = api if api is not None else _default_api()
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "README.md").write_text(
        dataset_card(results, benchmark_name=benchmark_name), encoding="utf-8"
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

    return HfApi()

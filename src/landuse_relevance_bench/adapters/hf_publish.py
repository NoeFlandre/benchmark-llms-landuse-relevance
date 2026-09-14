"""Publishing run results to a Hugging Face dataset repository."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from landuse_relevance_bench.adapters.results_store import read_run
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import RunResult, outcomes_of

CARD_COLUMNS = (
    "configuration",
    "model_id",
    "n_items",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "matthews_corrcoef",
    "unparsed_rate",
    "truncated",
)


class DatasetHub(Protocol):
    """The slice of :class:`huggingface_hub.HfApi` this project depends on."""

    def create_repo(self, **kwargs: Any) -> Any: ...
    def upload_folder(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class PublishedRun:
    """A result together with the configuration used to produce it."""

    configuration: str
    result: RunResult


def read_published_runs(results_dir: Path) -> tuple[PublishedRun, ...]:
    """Read every result below ``results_dir`` with a stable configuration label."""
    published = []
    for path in sorted(results_dir.rglob("*.json")):
        relative = path.relative_to(results_dir)
        configuration = "strict-8-tokens" if "strict-8-tokens" in relative.parts else "standard"
        published.append(PublishedRun(configuration=configuration, result=read_run(path)))
    return tuple(
        sorted(published, key=lambda item: (item.configuration, item.result.metadata.model_id))
    )


def dataset_card(
    results: Sequence[RunResult],
    *,
    benchmark_name: str,
    configurations: Sequence[str] = (),
) -> str:
    """Build a terse card whose scores are recomputed from every prediction."""
    if not results:
        raise ValueError("cannot build a card from an empty set of results")
    labels = tuple(configurations) or ("standard",) * len(results)
    if len(labels) != len(results):
        raise ValueError("one configuration label is required for each result")
    rows = _card_rows(results, labels)
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

# Land-use relevance benchmark

`{benchmark_name}`; {n_items} labelled sentences; greedy decoding;
`max_new_tokens={reference.max_new_tokens}`; seed {reference.seed}.
Scores are recomputed from the published predictions.

- Benchmark sha256: `{reference.benchmark_sha256}`; prompt sha256: `{reference.prompt_sha256}`
- Code: https://github.com/NoeFlandre/benchmark-llms-landuse-relevance

## Scores

{header}
{divider}
{body}
"""


def _card_rows(results: Sequence[RunResult], configurations: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for result, configuration in zip(results, configurations, strict=True):
        derived = evaluate(outcomes_of(result.predictions))
        if derived != result.metrics:
            raise ValueError(
                f"{result.metadata.model_id} metrics do not match predictions"
            )
        rows.append(
            {
                "configuration": configuration,
                "model_id": result.metadata.model_id,
                "n_items": derived.n_items,
                "accuracy": round(derived.accuracy, 4),
                "balanced_accuracy": round(derived.balanced_accuracy, 4),
                "f1": round(derived.f1, 4),
                "precision": round(derived.precision, 4),
                "recall": round(derived.recall, 4),
                "matthews_corrcoef": round(derived.matthews_corrcoef, 4),
                "unparsed_rate": round(derived.unparsed_rate, 4),
                "truncated": sum(prediction.truncated for prediction in result.predictions),
            }
        )
    return sorted(rows, key=lambda row: (-row["f1"], row["configuration"], row["model_id"]))


def publish_results(
    repo_id: str,
    results_dir: Path,
    results: Sequence[RunResult],
    *,
    api: DatasetHub | None = None,
    private: bool = False,
    configurations: Sequence[str] = (),
    commit_message: str = "Publish small-LLM land-use relevance benchmark results",
    benchmark_name: str = "benchmark.csv",
) -> str:
    """Write the card next to the results, then push the whole folder to the Hub."""
    if not results:
        raise ValueError("refusing to publish an empty set of results")
    hub = api if api is not None else _default_api()
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "README.md").write_text(
        dataset_card(
            results,
            benchmark_name=benchmark_name,
            configurations=configurations,
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

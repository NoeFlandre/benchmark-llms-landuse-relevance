"""Publishing run results to a Hugging Face dataset repository."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from landuse_relevance_bench.adapters.results_store import (
    SPEED_KEYS,
    by_f1_then_name,
    read_runs,
    rounded,
    score_columns,
    speed_columns,
)
from landuse_relevance_bench.domain.agreement import speculative_agreements
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import RunResult, outcomes_of

CARD_COLUMNS = (
    "model_id",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "matthews_corrcoef",
    "unparsed_rate",
    "truncated",
)

SPEED_COLUMNS = (
    "model_id",
    "runtime",
    "batch_size",
    "wall_seconds",
    *SPEED_KEYS,
)


def _table(columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    body = [
        "| " + " | ".join("" if row[c] is None else str(row[c]) for c in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _speed_rows(results: Sequence[RunResult]) -> list[dict[str, Any]]:
    rows = [
        {
            "model_id": r.metadata.name,
            "runtime": r.metadata.runtime,
            "batch_size": r.metadata.batch_size,
            "wall_seconds": round(r.metadata.duration_seconds, 2),
            **speed_columns(r),
        }
        for r in results
    ]
    return sorted(rows, key=lambda row: row["model_id"])


DSPARK_COLUMNS = (
    "run",
    "output_tokens_per_second",
    "latency_mean_seconds",
    "speedup",
    "mean_accept_length",
    "identical_predictions",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "matthews_corrcoef",
    "unparsed_rate",
)


def _dspark_row(result: RunResult, baseline_tps: float | None, identical: str) -> dict[str, Any]:
    speed = result.speed
    tps = speed.output_tokens_per_second
    return {
        "run": result.metadata.name,
        "output_tokens_per_second": rounded(tps, 1),
        "latency_mean_seconds": rounded(speed.latency_mean_seconds, 4),
        "speedup": None if tps is None or not baseline_tps else f"{tps / baseline_tps:.2f}x",
        "mean_accept_length": rounded(speed.mean_accept_length, 3),
        "identical_predictions": identical,
        **{
            name: round(getattr(result.metrics, name), 4)
            for name in DSPARK_COLUMNS[DSPARK_COLUMNS.index("accuracy") :]
        },
    }


def _agreement_section(results: Sequence[RunResult]) -> str:
    """One table per target: its same-runtime baseline against its DSpark run."""
    by_name = {r.metadata.name: r for r in results}
    tables = []
    for agreement in speculative_agreements(results):
        if not agreement.same_runtime:
            continue
        baseline = by_name[agreement.baseline_run]
        drafted = by_name[agreement.speculative_run]
        base_tps = baseline.speed.output_tokens_per_second
        identical = "yes" if agreement.lossless else "no"
        rows = [_dspark_row(baseline, base_tps, "-"), _dspark_row(drafted, base_tps, identical)]
        tables.append(f"### {drafted.metadata.model_id}\n\n{_table(DSPARK_COLUMNS, rows)}\n")
    if not tables:
        return ""
    body = "\n".join(tables)
    return f"""
## DSpark speculative decoding

Same SGLang launch with and without the draft; greedy, so predictions must be identical.

{body}"""


class DatasetHub(Protocol):
    """The slice of :class:`huggingface_hub.HfApi` this project depends on."""

    def create_repo(self, **kwargs: Any) -> Any: ...
    def upload_folder(self, **kwargs: Any) -> Any: ...


def read_published_runs(results_dir: Path) -> tuple[RunResult, ...]:
    """Read every result below ``results_dir``, subfolders included, in stable model-id order."""
    runs = read_runs(results_dir, recursive=True)
    return tuple(sorted(runs, key=lambda result: result.metadata.name))


def dataset_card(
    results: Sequence[RunResult],
    *,
    benchmark_name: str,
) -> str:
    """Build a terse card whose scores are recomputed from every prediction."""
    if not results:
        raise ValueError("cannot build a card from an empty set of results")
    rows = _card_rows(results)
    reference = results[0].metadata
    n_items = results[0].metrics.n_items
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

{_table(CARD_COLUMNS, rows)}

## Speed

Wall time covers generation only (model loading excluded). Latency is the wall time
of the generator call that answered a sentence, so with batching it is the batch's
latency. Throughput is not comparable across GPUs; each file records its settings.

{_table(SPEED_COLUMNS, _speed_rows(results))}
{_agreement_section(results)}"""


def _card_rows(results: Sequence[RunResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        derived = evaluate(outcomes_of(result.predictions))
        if derived != result.metrics:
            raise ValueError(f"{result.metadata.name} metrics do not match predictions")
        rows.append(
            {"model_id": result.metadata.name, **score_columns(derived, result.predictions)}
        )
    return sorted(rows, key=by_f1_then_name)


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

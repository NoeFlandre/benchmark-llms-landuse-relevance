"""Persisting run results and the leaderboard derived from them."""

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from landuse_relevance_bench.domain.agreement import Agreement
from landuse_relevance_bench.domain.metrics import ClassificationMetrics
from landuse_relevance_bench.domain.records import Prediction, RunResult

LEADERBOARD_COLUMNS = (
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
    "duration_seconds",
    "sentences_per_second",
    "latency_p50_seconds",
    "latency_p95_seconds",
    "generated_tokens",
    "output_tokens_per_second",
    "mean_accept_length",
    "draft_accept_rate",
    "runtime",
    "model_revision",
)


def run_filename(model_id: str) -> str:
    """A filesystem-safe name that still shows which model produced the run."""
    return f"{model_id.replace('/', '__')}.json"


def write_run(result: RunResult, directory: Path) -> Path:
    """Write ``result`` under ``directory``; the same result always writes the same bytes."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / run_filename(result.metadata.name)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def read_run(path: Path) -> RunResult:
    """Read back a run written by :func:`write_run`."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    try:
        return RunResult.from_dict(payload)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path} is not a valid run result: {exc}") from exc


def read_runs(directory: Path) -> list[RunResult]:
    """Read every run in ``directory``, in a stable filename order."""
    return [read_run(p) for p in sorted(directory.glob("*.json"))]


SPEED_KEYS = (
    "sentences_per_second",
    "latency_p50_seconds",
    "latency_p95_seconds",
    "generated_tokens",
    "output_tokens_per_second",
    "mean_accept_length",
    "draft_accept_rate",
)


def rounded(value: float | None, digits: int) -> float | None:
    """``value`` rounded to ``digits``, or ``None`` when the run did not record it."""
    return None if value is None else round(value, digits)


def score_columns(
    metrics: ClassificationMetrics, predictions: Sequence[Prediction]
) -> dict[str, Any]:
    """The rounded scores and the truncation count shown for one run."""
    return {
        "accuracy": round(metrics.accuracy, 4),
        "balanced_accuracy": round(metrics.balanced_accuracy, 4),
        "f1": round(metrics.f1, 4),
        "precision": round(metrics.precision, 4),
        "recall": round(metrics.recall, 4),
        "matthews_corrcoef": round(metrics.matthews_corrcoef, 4),
        "unparsed_rate": round(metrics.unparsed_rate, 4),
        "truncated": sum(p.truncated for p in predictions),
    }


def by_f1_then_name(row: dict[str, Any]) -> tuple[float, str]:
    """Sort key: best F1 first, ties broken by model id for determinism."""
    return (-row["f1"], row["model_id"])


def speed_columns(result: RunResult) -> dict[str, Any]:
    """The speed figures shown next to the scores; blank where a run did not record them."""
    speed = result.speed
    return {
        "sentences_per_second": rounded(speed.sentences_per_second, 3),
        "latency_p50_seconds": rounded(speed.latency_p50_seconds, 4),
        "latency_p95_seconds": rounded(speed.latency_p95_seconds, 4),
        "generated_tokens": speed.generated_tokens,
        "output_tokens_per_second": rounded(speed.output_tokens_per_second, 1),
        "mean_accept_length": rounded(speed.mean_accept_length, 3),
        "draft_accept_rate": rounded(speed.draft_accept_rate, 4),
    }


def leaderboard_rows(results: Sequence[RunResult]) -> list[dict[str, Any]]:
    """One row per run, best F1 first, ties broken by run name for determinism."""
    rows = [
        {
            "model_id": r.metadata.name,
            "n_items": r.metrics.n_items,
            **score_columns(r.metrics, r.predictions),
            "duration_seconds": round(r.metadata.duration_seconds, 2),
            **speed_columns(r),
            "runtime": r.metadata.runtime,
            "model_revision": r.metadata.model_revision,
        }
        for r in results
    ]
    return sorted(rows, key=by_f1_then_name)


def write_leaderboard_csv(results: Sequence[RunResult], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LEADERBOARD_COLUMNS))
        writer.writeheader()
        writer.writerows(leaderboard_rows(results))
    return path


def describe_agreement(agreement: Agreement) -> str:
    """One line for the lossless check of a speculative run against its baseline."""
    verdict = "lossless" if agreement.lossless else "MISMATCH"
    runtime = "same runtime" if agreement.same_runtime else "different runtime"
    return (
        f"{agreement.speculative_run} vs {agreement.baseline_run} ({runtime}): {verdict}, "
        f"{agreement.verdicts_differ}/{agreement.n_compared} verdicts and "
        f"{agreement.texts_differ}/{agreement.n_compared} generations differ"
    )

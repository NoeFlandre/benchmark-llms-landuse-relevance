"""Persisting run results and the leaderboard derived from them."""

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from landuse_relevance_bench.domain.records import RunResult

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
    "model_revision",
)


def run_filename(model_id: str) -> str:
    """A filesystem-safe name that still shows which model produced the run."""
    return f"{model_id.replace('/', '__')}.json"


def write_run(result: RunResult, directory: Path) -> Path:
    """Write ``result`` under ``directory``; the same result always writes the same bytes."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / run_filename(result.metadata.model_id)
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


def leaderboard_rows(results: Sequence[RunResult]) -> list[dict[str, Any]]:
    """One row per run, best F1 first, ties broken by model id for determinism."""
    rows = [
        {
            "model_id": r.metadata.model_id,
            "n_items": r.metrics.n_items,
            "accuracy": round(r.metrics.accuracy, 4),
            "balanced_accuracy": round(r.metrics.balanced_accuracy, 4),
            "f1": round(r.metrics.f1, 4),
            "precision": round(r.metrics.precision, 4),
            "recall": round(r.metrics.recall, 4),
            "matthews_corrcoef": round(r.metrics.matthews_corrcoef, 4),
            "unparsed_rate": round(r.metrics.unparsed_rate, 4),
            "truncated": sum(p.truncated for p in r.predictions),
            "duration_seconds": round(r.metadata.duration_seconds, 2),
            "model_revision": r.metadata.model_revision,
        }
        for r in results
    ]
    return sorted(rows, key=lambda row: (-row["f1"], row["model_id"]))


def write_leaderboard_csv(results: Sequence[RunResult], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LEADERBOARD_COLUMNS))
        writer.writeheader()
        writer.writerows(leaderboard_rows(results))
    return path

"""Persisting run results and the leaderboard derived from them."""

import csv
import json
import re
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from landuse_relevance_bench.domain.metrics import ClassificationMetrics
from landuse_relevance_bench.domain.records import Prediction, RunResult
from landuse_relevance_bench.domain.thresholds import (
    DEFAULT_THRESHOLDS,
    decide_scores,
    expected_labels,
    roc_auc_scores,
    yes_scores,
)
from landuse_relevance_bench.domain.uncertainty import (
    DEFAULT_BOOTSTRAP_SEED,
    bootstrap_interval,
    interval_text,
    paired_mcnemar_p_value,
    wilson_interval,
)

LEADERBOARD_COLUMNS = (
    "model_id",
    "language",
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
    "latency_mean_seconds",
    "latency_p50_seconds",
    "latency_p95_seconds",
    "generated_tokens",
    "output_tokens_per_second",
    "mean_accept_length",
    "draft_accept_rate",
    "runtime",
    "generation_mode",
    "package_version",
    "throughput_items_per_second",
    "peak_vram_bytes",
    "model_revision",
    "accuracy_ci95",
    "precision_ci95",
    "recall_ci95",
    "f1_ci95",
    "matthews_corrcoef_ci95",
    "mcnemar_top_run",
    "mcnemar_p_vs_top",
)
AGGREGATE_COLUMNS = (
    "model_id",
    "language_count",
    "n_items_total",
    "accuracy_macro",
    "balanced_accuracy_macro",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "matthews_corrcoef_macro",
    "unparsed_rate_macro",
    "f1_min",
    "f1_max",
    "f1_std",
)
CLASSIFICATION_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "matthews_corrcoef",
    "unparsed_rate",
)
_LANGUAGE_PATTERN = re.compile(r"^[a-z]{2,3}$")
ARCHIVE_COMPONENT = "archive"
_MIN_NESTED_RESULT_PARTS = 2


def rounded(value: float | None, digits: int) -> float | None:
    """Round a reported metric, preserving unavailable measurements as ``None``."""
    return None if value is None else round(value, digits)


def run_filename(model_id: str, language: str) -> Path:
    """Return the active nested path for one model-language result."""
    normalized_language = language.strip().lower()
    if not _LANGUAGE_PATTERN.fullmatch(normalized_language):
        raise ValueError(f"invalid result language {language!r}")
    return Path(normalized_language) / f"{model_id.replace('/', '__')}.json"


def dumps_run_payload(payload: dict[str, Any]) -> str:
    """Canonical JSON text of a saved run; every writer uses it so committed diffs stay clean."""
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def write_run(result: RunResult, directory: Path) -> Path:
    """Write ``result`` under ``directory``; the same result always writes the same bytes."""
    path = directory / run_filename(result.metadata.name, result.metadata.language)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_run_payload(result.to_dict()), encoding="utf-8")
    return path


def read_run(path: Path, *, legacy_language: str | None = None) -> RunResult:
    """Read back a run written by :func:`write_run`."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    try:
        metadata = payload.get("metadata", {})
        if legacy_language is not None and not metadata.get("language"):
            metadata["language"] = legacy_language
        return RunResult.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} is not a valid run result: {exc}") from exc


def read_runs(directory: Path, *, recursive: bool = False) -> list[RunResult]:
    """Read active runs, or every JSON run when explicitly asked to include archives."""
    results = []
    paths = sorted(directory.rglob("*.json")) if recursive else _active_result_paths(directory)
    for path in paths:
        relative = path.relative_to(directory)
        is_archive = relative.parts[0] == ARCHIVE_COMPONENT
        result = read_run(path, legacy_language="en" if is_archive else None)
        if not recursive and (
            len(relative.parts) < _MIN_NESTED_RESULT_PARTS
            or relative.parts[0] != result.metadata.language
        ):
            raise ValueError(
                f"{path} is a legacy/archive-only result; expected a language directory"
            )
        results.append(result)
    return results


def leaderboard_rows(results: Sequence[RunResult]) -> list[dict[str, Any]]:
    """One row per run, best F1 first, ties broken by model id for determinism."""
    best_by_language: dict[str, RunResult] = {}
    for result in results:
        current = best_by_language.get(result.metadata.language)
        if current is None or (-result.metrics.f1, result.metadata.name) < (
            -current.metrics.f1,
            current.metadata.name,
        ):
            best_by_language[result.metadata.language] = result

    rows = []
    for r in results:
        top = best_by_language[r.metadata.language]
        speed = r.speed
        row = {
            "model_id": r.metadata.name,
            "language": r.metadata.language,
            "n_items": r.metrics.n_items,
            **score_columns(r.metrics, r.predictions),
            "duration_seconds": round(r.metadata.duration_seconds, 2),
            "sentences_per_second": speed.sentences_per_second,
            "latency_mean_seconds": speed.latency_mean_seconds,
            "latency_p50_seconds": speed.latency_p50_seconds,
            "latency_p95_seconds": speed.latency_p95_seconds,
            "generated_tokens": speed.generated_tokens,
            "output_tokens_per_second": speed.output_tokens_per_second,
            "mean_accept_length": speed.mean_accept_length,
            "draft_accept_rate": speed.draft_accept_rate,
            "runtime": r.metadata.runtime,
            "generation_mode": r.metadata.generation_mode,
            "package_version": r.metadata.package_version,
            "throughput_items_per_second": (
                None
                if r.metadata.throughput_items_per_second is None
                else round(r.metadata.throughput_items_per_second, 2)
            ),
            "peak_vram_bytes": r.metadata.peak_vram_bytes,
            "model_revision": r.metadata.model_revision,
            "mcnemar_top_run": top.metadata.name,
            "mcnemar_p_vs_top": (
                None
                if r.metadata.benchmark_sha256 != top.metadata.benchmark_sha256
                else paired_mcnemar_p_value(r.predictions, top.predictions)
            ),
        }
        rows.append(row)
    return sorted(rows, key=lambda row: (-row["f1"], row["model_id"], row["language"]))


def score_columns(
    metrics: ClassificationMetrics, predictions: Sequence[Prediction]
) -> dict[str, Any]:
    """Scores and seeded confidence intervals for one complete language run."""
    matrix = metrics.confusion
    outcomes = [prediction.outcome for prediction in predictions]
    return {
        "accuracy": round(metrics.accuracy, 4),
        "balanced_accuracy": round(metrics.balanced_accuracy, 4),
        "f1": round(metrics.f1, 4),
        "precision": round(metrics.precision, 4),
        "recall": round(metrics.recall, 4),
        "matthews_corrcoef": round(metrics.matthews_corrcoef, 4),
        "unparsed_rate": round(metrics.unparsed_rate, 4),
        "truncated": sum(prediction.truncated for prediction in predictions),
        "accuracy_ci95": interval_text(
            wilson_interval(matrix.true_positive + matrix.true_negative, metrics.n_items)
        ),
        "precision_ci95": interval_text(
            wilson_interval(matrix.true_positive, matrix.true_positive + matrix.false_positive)
        ),
        "recall_ci95": interval_text(
            wilson_interval(matrix.true_positive, matrix.true_positive + matrix.false_negative)
        ),
        "f1_ci95": interval_text(bootstrap_interval(outcomes, "f1", seed=DEFAULT_BOOTSTRAP_SEED)),
        "matthews_corrcoef_ci95": interval_text(
            bootstrap_interval(outcomes, "matthews_corrcoef", seed=DEFAULT_BOOTSTRAP_SEED)
        ),
    }


def write_leaderboard_csv(results: Sequence[RunResult], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LEADERBOARD_COLUMNS))
        writer.writeheader()
        writer.writerows(leaderboard_rows(results))
    return path


THRESHOLD_SWEEP_COLUMNS = (
    "model_id",
    "threshold",
    "language_count",
    "n_items_total",
    "accuracy_macro",
    "balanced_accuracy_macro",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "matthews_corrcoef_macro",
    "roc_auc_macro",
)


def threshold_sweep_rows(results: Sequence[RunResult]) -> list[dict[str, Any]]:
    """Re-decide every scoring run at each boundary and average over languages.

    Averaged the same way as the published leaderboard, so a row here can be read
    against a row there. ``roc_auc_macro`` does not depend on the boundary and is
    repeated on every row of a model for convenience.
    """
    rows = []
    for model_id, runs in sorted(group_by_model(scoring_runs(results)).items()):
        # Each run's scores are parsed once and reused for every boundary.
        parsed = [(expected_labels(r.predictions), yes_scores(r.predictions)) for r in runs]
        auc_values = []
        for expected, scores in parsed:
            try:
                auc_values.append(roc_auc_scores(expected, scores))
            except ValueError:
                # A partial/debug fixture can contain one class; the thresholded
                # metrics remain useful and ROC-AUC is honestly unavailable.
                continue
        auc = None if not auc_values else round(fmean(auc_values), 4)
        for threshold in DEFAULT_THRESHOLDS:
            scored = [decide_scores(expected, scores, threshold) for expected, scores in parsed]
            row: dict[str, Any] = {
                "model_id": model_id,
                "threshold": threshold,
                "language_count": len(runs),
                "n_items_total": sum(m.n_items for m in scored),
                "roc_auc_macro": auc,
            }
            for metric_name in CLASSIFICATION_METRICS:
                if metric_name == "unparsed_rate":
                    continue
                row[f"{metric_name}_macro"] = round(
                    fmean(getattr(m, metric_name) for m in scored), 4
                )
            rows.append(row)
    return rows


def write_threshold_sweep_csv(results: Sequence[RunResult], path: Path) -> Path:
    """Write the sweep beside the leaderboard; empty of rows when nothing scores."""
    return _write_rows(threshold_sweep_rows(results), THRESHOLD_SWEEP_COLUMNS, path)


SCORING_SUMMARY_COLUMNS = (
    "model_id",
    "language_count",
    "n_items_total",
    "best_mcc",
    "best_mcc_threshold",
    "best_f1",
    "best_f1_threshold",
    "best_balanced_accuracy",
    "best_balanced_accuracy_threshold",
    "best_precision",
    "best_precision_threshold",
    "best_recall",
    "best_recall_threshold",
    "roc_auc_macro",
    "throughput_items_per_second_macro",
    "peak_vram_bytes_max",
)


def scoring_summary_rows(
    results: Sequence[RunResult], sweep: Sequence[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Select each model's best thresholded classification metrics.

    Ties use the lowest threshold, making the summary deterministic and favoring the
    least strict boundary among equally good operating points. Pass ``sweep`` (the
    output of :func:`threshold_sweep_rows` over the same runs) to avoid recomputing it.
    """
    scoring = group_by_model(scoring_runs(results))
    sweep_by_model = group_rows_by_model(
        sweep if sweep is not None else threshold_sweep_rows(results)
    )

    rows = []
    for model_id, runs in sorted(scoring.items()):
        sweep_rows = sweep_by_model.get(model_id, [])
        if not sweep_rows:
            continue
        row: dict[str, Any] = {
            "model_id": model_id,
            "language_count": len(runs),
            "n_items_total": sum(run.metrics.n_items for run in runs),
            "roc_auc_macro": sweep_rows[0]["roc_auc_macro"],
            "throughput_items_per_second_macro": _throughput_macro(runs),
            "peak_vram_bytes_max": _peak_vram_max(runs),
        }
        for output_name, source_name in (
            ("mcc", "matthews_corrcoef_macro"),
            ("f1", "f1_macro"),
            ("balanced_accuracy", "balanced_accuracy_macro"),
            ("precision", "precision_macro"),
            ("recall", "recall_macro"),
        ):
            best = min(
                sweep_rows,
                key=lambda candidate: (-candidate[source_name], candidate["threshold"]),
            )
            row[f"best_{output_name}"] = best[source_name]
            row[f"best_{output_name}_threshold"] = best["threshold"]
        rows.append(row)
    return sorted(rows, key=lambda row: (-row["best_f1"], row["model_id"]))


def scoring_runs(results: Sequence[RunResult]) -> list[RunResult]:
    return [result for result in results if not result.metadata.is_generative]


def group_by_model(results: Sequence[RunResult]) -> dict[str, list[RunResult]]:
    """Runs keyed by unique run name, in input order within each run."""
    grouped: dict[str, list[RunResult]] = {}
    for result in results:
        grouped.setdefault(result.metadata.name, []).append(result)
    return grouped


def group_rows_by_model(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["model_id"], []).append(row)
    return grouped


def write_reports(results: Sequence[RunResult], leaderboard: Path) -> dict[str, Path]:
    """Write the four report CSVs beside ``leaderboard``, computing the sweep once."""
    sweep = threshold_sweep_rows(results)
    return {
        "leaderboard": write_leaderboard_csv(results, leaderboard),
        "aggregates": write_aggregates_csv(results, leaderboard.with_name("aggregates.csv")),
        "threshold_sweep": _write_rows(
            sweep, THRESHOLD_SWEEP_COLUMNS, leaderboard.with_name("threshold_sweep.csv")
        ),
        "scoring_summary": _write_rows(
            scoring_summary_rows(results, sweep),
            SCORING_SUMMARY_COLUMNS,
            leaderboard.with_name("scoring_summary.csv"),
        ),
    }


def _write_rows(rows: Sequence[dict[str, Any]], columns: Sequence[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_scoring_summary_csv(results: Sequence[RunResult], path: Path) -> Path:
    """Write one best-threshold row per scoring model."""
    return _write_rows(scoring_summary_rows(results), SCORING_SUMMARY_COLUMNS, path)


def _throughput_macro(results: Sequence[RunResult]) -> float | None:
    values = [
        result.metadata.throughput_items_per_second
        for result in results
        if result.metadata.throughput_items_per_second is not None
    ]
    return None if not values else round(fmean(values), 2)


def _peak_vram_max(results: Sequence[RunResult]) -> int | None:
    values = [
        result.metadata.peak_vram_bytes
        for result in results
        if result.metadata.peak_vram_bytes is not None
    ]
    return None if not values else max(values)


def aggregate_rows(results: Sequence[RunResult]) -> list[dict[str, Any]]:
    """Aggregate one detailed run per language into deterministic model rows."""
    seen_pairs: set[tuple[str, str]] = set()
    for result in results:
        pair = (result.metadata.name, result.metadata.language)
        if pair in seen_pairs:
            raise ValueError(f"duplicate result for model-language pair {pair!r}")
        seen_pairs.add(pair)
    grouped = group_by_model(results)

    rows = []
    for model_id, model_results in sorted(grouped.items()):
        metrics = [result.metrics for result in model_results]
        f1_values = [metric.f1 for metric in metrics]
        row: dict[str, Any] = {
            "model_id": model_id,
            "language_count": len(model_results),
            "n_items_total": sum(metric.n_items for metric in metrics),
        }
        for metric_name in CLASSIFICATION_METRICS:
            row[f"{metric_name}_macro"] = round(
                fmean(getattr(metric, metric_name) for metric in metrics), 4
            )
        row.update(
            f1_min=round(min(f1_values), 4),
            f1_max=round(max(f1_values), 4),
            f1_std=round(pstdev(f1_values), 4),
        )
        rows.append(row)
    return sorted(rows, key=lambda row: (-row["f1_macro"], row["model_id"]))


def write_aggregates_csv(results: Sequence[RunResult], path: Path) -> Path:
    """Write model-level macro and spread metrics beside a detailed leaderboard."""
    return _write_rows(aggregate_rows(results), AGGREGATE_COLUMNS, path)


def _active_result_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*.json")
        if ARCHIVE_COMPONENT not in path.relative_to(directory).parts
    )

"""Output and error helpers shared by the CLI command definitions."""

import json
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, NoReturn

import typer

from landuse_relevance_bench import __version__
from landuse_relevance_bench.adapters.benchmark_csv import BenchmarkFileError
from landuse_relevance_bench.adapters.hf_publish import dataset_card
from landuse_relevance_bench.adapters.pipeline import (
    GeneratorProvider,
    RunRequest,
    execute,
)
from landuse_relevance_bench.adapters.prompt_file import PromptFileError
from landuse_relevance_bench.adapters.results_store import (
    describe_agreement,
    leaderboard_rows,
)
from landuse_relevance_bench.domain.agreement import Agreement
from landuse_relevance_bench.domain.records import RunResult
from landuse_relevance_bench.domain.selection import (
    ComparabilityReport,
    select_run_names,
)
from landuse_relevance_bench.domain.speed import SpeedMetrics

LOGGER_NAME = "landuse_relevance_bench"


@dataclass(frozen=True, slots=True)
class PublishPreviewRequest:
    """Inputs needed to show a publish preview without writing or uploading."""

    repo_id: str
    results_dir: Path
    runs: Sequence[RunResult]
    private: bool
    benchmark: Path
    allow_mixed: bool


class _EchoHandler(logging.Handler):
    """Log records go to stderr through Typer, so test runners capture them too."""

    def emit(self, record: logging.LogRecord) -> None:
        typer.echo(self.format(record), err=True)


def print_version(value: bool) -> None:  # noqa: FBT001
    """Typer callback that prints the package version and exits early."""
    if value:
        typer.echo(__version__)
        raise typer.Exit


def generator_provider() -> GeneratorProvider:
    """Import the chosen runtime only when a benchmark command is executed."""
    return import_module("landuse_relevance_bench.adapters.generators").provide


def selected_run_names(
    names: Iterable[str], only: str | None, runtimes: Sequence[str] | None
) -> list[str]:
    """Adapt invalid domain run selectors to Typer's usage error."""
    try:
        return select_run_names(tuple(names), only=only, runtimes=runtimes)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def configure_logging(verbosity: int) -> None:
    """-q shows errors only, the default shows warnings, -v progress, -vv debug detail."""
    levels = {-1: logging.ERROR, 0: logging.WARNING, 1: logging.INFO}
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(levels.get(verbosity, logging.DEBUG))
    if not any(isinstance(handler, _EchoHandler) for handler in logger.handlers):
        handler = _EchoHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False


def echo_json(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def benchmark_one(
    request: RunRequest,
    provider_factory: Callable[[], GeneratorProvider],
    commit_resolver: Callable[[], str],
    *,
    as_json: bool = False,
) -> None:
    """Run one model, reporting input problems as usage errors rather than tracebacks."""
    try:
        result = execute(request, provider_factory(), source_commit=commit_resolver())
    except (OSError, BenchmarkFileError, PromptFileError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if as_json:
        echo_json(
            {
                "model_id": request.name,
                "metrics": result.metrics.to_dict(),
                "speed": result.speed.to_dict(),
                "duration_seconds": result.metadata.duration_seconds,
            }
        )
        return
    metrics = result.metrics
    typer.echo(
        f"{request.model_id}  accuracy={metrics.accuracy:.3f}  f1={metrics.f1:.3f}  "
        f"mcc={metrics.matthews_corrcoef:.3f}  unparsed={metrics.unparsed_rate:.3f}  "
        f"({result.metadata.duration_seconds:.1f}s, {speed_summary(result.speed)})"
    )


def speed_summary(speed: SpeedMetrics) -> str:
    parts = [f"{speed.sentences_per_second or 0:.2f} sent/s"]
    if speed.latency_p50_seconds is not None:
        parts.append(f"p50={speed.latency_p50_seconds:.3f}s p95={speed.latency_p95_seconds:.3f}s")
    if speed.output_tokens_per_second is not None:
        parts.append(f"{speed.output_tokens_per_second:.1f} tok/s")
    if speed.mean_accept_length is not None:
        parts.append(f"accept={speed.mean_accept_length:.2f}")
    if speed.draft_accept_rate is not None:
        parts.append(f"accept_rate={speed.draft_accept_rate:.3f}")
    return " ".join(parts)


def load_runs(results_dir: Path, reader: Callable[[Path], Iterable[RunResult]]) -> list[RunResult]:
    """Read stored runs, reporting a missing or empty directory as a usage error."""
    if not results_dir.is_dir():
        raise typer.BadParameter(f"no run results directory at {results_dir}")
    try:
        runs = list(reader(results_dir))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not runs:
        raise typer.BadParameter(f"no run results found in {results_dir}")
    return runs


def echo_report(
    runs: Sequence[RunResult],
    agreements: Sequence[Agreement],
    destination: Path,
    comparability: ComparabilityReport,
) -> None:
    for row in leaderboard_rows(runs):
        typer.echo(
            f"{row['model_id']:<34} f1={row['f1']:.3f} acc={row['accuracy']:.3f} "
            f"mcc={row['matthews_corrcoef']:.3f} unparsed={row['unparsed_rate']:.3f}  "
            f"{speed_cells(row)}"
        )
    for agreement in agreements:
        typer.echo(f"speculative check: {describe_agreement(agreement)}")
    if not comparability.comparable:
        typer.echo(f"warning: {comparability.summary}", err=True)
    typer.echo(f"\nleaderboard written to {destination}")


def speed_cells(row: dict[str, Any]) -> str:
    cells = [
        f"{label}={row[key]}"
        for label, key in (
            ("sent/s", "sentences_per_second"),
            ("p50", "latency_p50_seconds"),
            ("p95", "latency_p95_seconds"),
            ("tok/s", "output_tokens_per_second"),
            ("accept", "mean_accept_length"),
            ("accept_rate", "draft_accept_rate"),
        )
        if row[key] is not None
    ]
    return " ".join(cells)


def preview_publish(request: PublishPreviewRequest) -> None:
    """Show the publish target and generated card without touching disk or the Hub."""
    generated = {"README.md", "leaderboard.csv"}
    existing = {
        path.relative_to(request.results_dir).as_posix()
        for path in request.results_dir.rglob("*")
        if path.is_file()
    }
    visibility = "private" if request.private else "public"
    typer.echo(
        f"dry run: would publish {len(request.runs)} run(s) to dataset "
        f"{request.repo_id} ({visibility})"
    )
    typer.echo("files that would be uploaded:")
    for name in sorted(existing | generated):
        typer.echo(f"  {name}{'  (generated)' if name in generated else ''}")
    typer.echo("\n--- README.md ---")
    typer.echo(
        dataset_card(
            request.runs,
            benchmark_name=request.benchmark.name,
            allow_mixed=request.allow_mixed,
        )
    )


def first_line(exc: BaseException) -> str:
    return (str(exc).strip().splitlines() or [type(exc).__name__])[0]


def fail(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)

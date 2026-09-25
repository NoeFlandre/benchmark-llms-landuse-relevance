"""Scriptable entry points for running, scoring and publishing the benchmark."""

import json
import logging
import subprocess
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Annotated

import typer

from landuse_relevance_bench import __version__
from landuse_relevance_bench.adapters.benchmark_csv import BenchmarkFileError
from landuse_relevance_bench.adapters.pipeline import (
    DEFAULT_DTYPE,
    DEFAULT_MAX_NEW_TOKENS,
    GeneratorProvider,
    RunRequest,
    execute,
)
from landuse_relevance_bench.adapters.prompt_file import PromptFileError
from landuse_relevance_bench.adapters.results_store import (
    describe_agreement,
    leaderboard_rows,
    read_runs,
    write_leaderboard_csv,
)
from landuse_relevance_bench.domain.agreement import Agreement, speculative_agreements
from landuse_relevance_bench.domain.orchestration import DEFAULT_BATCH_SIZE
from landuse_relevance_bench.domain.records import RunResult
from landuse_relevance_bench.domain.roster import ROSTER, model_ids
from landuse_relevance_bench.domain.speed import SpeedMetrics

DEFAULT_BENCHMARK = Path("data/benchmark.csv")
DEFAULT_PROMPT = Path("data/prompt.txt")
DEFAULT_RESULTS = Path("results")

app = typer.Typer(help=__doc__)

LOGGER_NAME = "landuse_relevance_bench"

Benchmark = Annotated[Path, typer.Option("--benchmark", help="Labelled benchmark CSV.")]
Prompt = Annotated[Path, typer.Option("--prompt", help="Prompt template with a {} placeholder.")]
Results = Annotated[
    Path,
    typer.Option(
        "--out", "--results-dir", help="Directory to write run results into (--out is an alias)."
    ),
]
ResultsDir = Annotated[
    Path, typer.Option("--results-dir", help="Directory holding stored run results.")
]
BatchSize = Annotated[
    int | None,
    typer.Option(help=f"Prompts per forward pass [default: roster's, else {DEFAULT_BATCH_SIZE}]."),
]
MaxNewTokens = Annotated[int, typer.Option(help="Generation budget per prompt, in tokens.")]
Seed = Annotated[int, typer.Option(help="Random seed set before loading the model.")]
Dtype = Annotated[str, typer.Option(help="Torch dtype name.")]
Json = Annotated[bool, typer.Option("--json", help="Print machine-readable JSON instead of text.")]


class _EchoHandler(logging.Handler):
    """Log records go to stderr through typer, so test runners capture them too."""

    def emit(self, record: logging.LogRecord) -> None:
        typer.echo(self.format(record), err=True)


def configure_logging(verbosity: int) -> None:
    """-q shows errors only, the default shows warnings, -v progress, -vv debug detail."""
    levels = {-1: logging.ERROR, 0: logging.WARNING, 1: logging.INFO}
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(levels.get(verbosity, logging.DEBUG))
    if not any(isinstance(h, _EchoHandler) for h in logger.handlers):
        handler = _EchoHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False


def _print_version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_print_version, is_eager=True, help="Print the version and exit."
        ),
    ] = False,
    verbose: Annotated[
        int, typer.Option("--verbose", "-v", count=True, help="More output; repeat for debug.")
    ] = 0,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Only report errors.")] = False,
) -> None:
    """Scriptable entry points for running, scoring and publishing the benchmark."""
    del version
    configure_logging(-1 if quiet else verbose)


def _echo_json(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def generator_provider() -> GeneratorProvider:
    """Imported lazily so the CLI stays usable without a model runtime installed."""
    from landuse_relevance_bench.adapters.generators import provide

    return provide


def source_commit() -> str:
    """The commit the code was run from, recorded alongside every result."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def _benchmark_one(request: RunRequest, *, as_json: bool = False) -> None:
    """Run one model, reporting input problems as usage errors rather than tracebacks."""
    try:
        result = execute(request, generator_provider(), source_commit=source_commit())
    except (OSError, BenchmarkFileError, PromptFileError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if as_json:
        _echo_json(
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
        f"({result.metadata.duration_seconds:.1f}s, {_speed_summary(result.speed)})"
    )


def _speed_summary(speed: SpeedMetrics) -> str:
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


@app.command(epilog="Examples:  lrb models  |  lrb models --json")
def models(as_json: Json = False) -> None:
    """List the models in the benchmark roster (name, size, runtime, note; tab-separated)."""
    if as_json:
        _echo_json(
            [
                {
                    "name": spec.name,
                    "model_id": spec.model_id,
                    "total_parameters": spec.total_parameters,
                    "runtime": spec.runtime,
                    "note": spec.note,
                }
                for spec in ROSTER
            ]
        )
        return
    for spec in ROSTER:
        typer.echo(f"{spec.name}\t{spec.total_parameters / 1e9:.2f}B\t{spec.runtime}\t{spec.note}")


@app.command(
    epilog="Examples:  lrb run LiquidAI/LFM2.5-350M  |  "
    "lrb run some/model --revision c0ffee --results-dir results --json"
)
def run(
    model_id: Annotated[
        str, typer.Argument(help="A rostered run name, or any Hugging Face model id.")
    ],
    benchmark: Benchmark = DEFAULT_BENCHMARK,
    prompt: Prompt = DEFAULT_PROMPT,
    out: Results = DEFAULT_RESULTS,
    revision: Annotated[str | None, typer.Option(help="Pin the model to a commit.")] = None,
    batch_size: BatchSize = None,
    max_new_tokens: MaxNewTokens = DEFAULT_MAX_NEW_TOKENS,
    seed: Seed = 0,
    dtype: Dtype = DEFAULT_DTYPE,
    as_json: Json = False,
) -> None:
    """Benchmark one model and write its result under --out."""
    request = RunRequest.for_run(
        model_id,
        benchmark_path=benchmark,
        prompt_path=prompt,
        output_dir=out,
        revision=revision,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        seed=seed,
        dtype=dtype,
    )
    _benchmark_one(request, as_json=as_json)


@app.command(name="run-all", epilog="Examples:  lrb run-all --results-dir results")
def run_all(
    benchmark: Benchmark = DEFAULT_BENCHMARK,
    prompt: Prompt = DEFAULT_PROMPT,
    out: Results = DEFAULT_RESULTS,
    batch_size: BatchSize = None,
    max_new_tokens: MaxNewTokens = DEFAULT_MAX_NEW_TOKENS,
    seed: Seed = 0,
    dtype: Dtype = DEFAULT_DTYPE,
) -> None:
    """Benchmark every rostered model in turn."""
    for model in model_ids():
        _benchmark_one(
            RunRequest.for_run(
                model,
                benchmark_path=benchmark,
                prompt_path=prompt,
                output_dir=out,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                seed=seed,
                dtype=dtype,
            )
        )


@app.command(
    epilog="Examples:  lrb report --results-dir results  |  lrb report --json | python -m json.tool"
)
def report(
    results_dir: ResultsDir = DEFAULT_RESULTS,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Leaderboard CSV path [default: RESULTS_DIR/leaderboard.csv]."),
    ] = None,
    as_json: Json = False,
) -> None:
    """Aggregate stored runs into a leaderboard; exits 1 if a lossless check fails."""
    runs = _load_runs(results_dir, read_runs)
    destination = out or results_dir / "leaderboard.csv"
    write_leaderboard_csv(runs, destination)
    agreements = speculative_agreements(runs)
    mismatched = [a for a in agreements if a.same_runtime and not a.lossless]
    if as_json:
        _echo_json(
            {
                "leaderboard": leaderboard_rows(runs),
                "speculative_checks": [describe_agreement(a) for a in agreements],
                "leaderboard_csv": str(destination),
            }
        )
    else:
        _echo_report(runs, agreements, destination)
    if mismatched:
        typer.echo("speculative run disagrees with its same-runtime baseline", err=True)
        raise typer.Exit(code=1)


def _echo_report(
    runs: Sequence[RunResult], agreements: Sequence[Agreement], destination: Path
) -> None:
    for row in leaderboard_rows(runs):
        typer.echo(
            f"{row['model_id']:<34} f1={row['f1']:.3f} acc={row['accuracy']:.3f} "
            f"mcc={row['matthews_corrcoef']:.3f} unparsed={row['unparsed_rate']:.3f}  "
            f"{_speed_cells(row)}"
        )
    for agreement in agreements:
        typer.echo(f"speculative check: {describe_agreement(agreement)}")
    typer.echo(f"\nleaderboard written to {destination}")


def _load_runs(results_dir: Path, reader: Callable[[Path], Iterable[RunResult]]) -> list[RunResult]:
    """Read stored runs, reporting a missing or empty directory as a usage error."""
    if not results_dir.is_dir():
        raise typer.BadParameter(f"no run results directory at {results_dir}")
    runs = list(reader(results_dir))
    if not runs:
        raise typer.BadParameter(f"no run results found in {results_dir}")
    return runs


def _speed_cells(row: dict[str, object]) -> str:
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


@app.command(epilog="Examples:  lrb publish me/landuse-bench --results-dir results")
def publish(
    repo_id: Annotated[str, typer.Argument(help="Hugging Face dataset repository id.")],
    results_dir: ResultsDir = DEFAULT_RESULTS,
    benchmark: Benchmark = DEFAULT_BENCHMARK,
    private: Annotated[bool, typer.Option(help="Create the dataset repository private.")] = False,
) -> None:
    """Push the stored runs, leaderboard and a generated card to the Hub."""
    from landuse_relevance_bench.adapters.hf_publish import publish_results, read_published_runs

    runs = _load_runs(results_dir, read_published_runs)
    write_leaderboard_csv(runs, results_dir / "leaderboard.csv")
    url = publish_results(
        repo_id,
        results_dir,
        runs,
        private=private,
        benchmark_name=benchmark.name,
    )
    typer.echo(f"published {len(runs)} run(s) to {url}")

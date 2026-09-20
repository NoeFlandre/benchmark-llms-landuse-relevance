"""Scriptable entry points for running, scoring and publishing the benchmark."""

import subprocess
from pathlib import Path
from typing import Annotated

import typer

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
    leaderboard_rows,
    read_run,
    read_runs,
    run_filename,
    write_aggregates_csv,
    write_leaderboard_csv,
)
from landuse_relevance_bench.adapters.translations import (
    TranslationDataError,
    TranslationManifest,
    load_manifest,
)
from landuse_relevance_bench.domain.orchestration import DEFAULT_BATCH_SIZE
from landuse_relevance_bench.domain.roster import ROSTER, model_ids

DEFAULT_DATA_ROOT = Path("data/translations")
DEFAULT_PROMPT = Path("data/prompt.txt")
DEFAULT_RESULTS = Path("results")

app = typer.Typer(add_completion=False, help=__doc__)

DataRoot = Annotated[Path, typer.Option("--data-root", help="Vendored multilingual data root.")]
Prompt = Annotated[Path, typer.Option("--prompt", help="Prompt template with a {} placeholder.")]
Results = Annotated[Path, typer.Option("--out", help="Directory to write run results into.")]
Language = Annotated[
    list[str] | None,
    typer.Option("--language", help="Language code(s), repeatable or comma-separated."),
]


def generator_provider() -> GeneratorProvider:
    """Imported lazily so the CLI stays usable without a model runtime installed."""
    from landuse_relevance_bench.adapters.hf_generator import provide

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


def _benchmark_one(request: RunRequest) -> None:
    """Run one model, reporting input problems as usage errors rather than tracebacks."""
    result_path = request.output_dir / run_filename(request.model_id, request.language)
    if result_path.is_file():
        try:
            read_run(result_path)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(f"{request.model_id} [{request.language}] already complete; skipping")
        return
    try:
        result = execute(request, generator_provider(), source_commit=source_commit())
    except (OSError, BenchmarkFileError, PromptFileError, TranslationDataError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    metrics = result.metrics
    typer.echo(
        f"{request.model_id} [{request.language}]  accuracy={metrics.accuracy:.3f}  "
        f"f1={metrics.f1:.3f}  "
        f"mcc={metrics.matthews_corrcoef:.3f}  unparsed={metrics.unparsed_rate:.3f}  "
        f"({result.metadata.duration_seconds:.1f}s)"
    )


@app.command()
def models() -> None:
    """List the models in the benchmark roster."""
    for spec in ROSTER:
        typer.echo(f"{spec.model_id}\t{spec.total_parameters / 1e9:.2f}B\t{spec.note}")


@app.command()
def languages(data_root: DataRoot = DEFAULT_DATA_ROOT) -> None:
    """List every active language and its configured row count."""
    try:
        manifest = load_manifest(data_root)
    except (OSError, TranslationDataError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    for language in manifest.languages:
        typer.echo(f"{language}\t{manifest.files[language].rows}")


@app.command()
def run(
    model_id: Annotated[str, typer.Argument(help="Hugging Face model repository id.")],
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    prompt: Prompt = DEFAULT_PROMPT,
    out: Results = DEFAULT_RESULTS,
    language: Language = None,
    revision: Annotated[str | None, typer.Option(help="Pin the model to a commit.")] = None,
    batch_size: Annotated[int, typer.Option(help="Prompts per forward pass.")] = DEFAULT_BATCH_SIZE,
    max_new_tokens: Annotated[int, typer.Option()] = DEFAULT_MAX_NEW_TOKENS,
    seed: Annotated[int, typer.Option()] = 0,
    dtype: Annotated[str, typer.Option(help="Torch dtype name.")] = DEFAULT_DTYPE,
) -> None:
    """Benchmark one model on every selected language."""
    manifest, selected = _selected_languages(data_root, language)
    for selected_language in selected:
        _benchmark_one(
            RunRequest(
                model_id=model_id,
                language=selected_language,
                benchmark_path=data_root / manifest.files[selected_language].path,
                prompt_path=prompt,
                output_dir=out,
                revision=revision,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                seed=seed,
                dtype=dtype,
            )
        )


@app.command(name="run-all")
def run_all(
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    prompt: Prompt = DEFAULT_PROMPT,
    out: Results = DEFAULT_RESULTS,
    language: Language = None,
    batch_size: Annotated[int, typer.Option()] = DEFAULT_BATCH_SIZE,
    max_new_tokens: Annotated[int, typer.Option()] = DEFAULT_MAX_NEW_TOKENS,
    seed: Annotated[int, typer.Option()] = 0,
    dtype: Annotated[str, typer.Option()] = DEFAULT_DTYPE,
) -> None:
    """Benchmark every rostered model on every selected language."""
    manifest, selected = _selected_languages(data_root, language)
    for model in model_ids():
        for selected_language in selected:
            _benchmark_one(
                RunRequest(
                    model_id=model,
                    language=selected_language,
                    benchmark_path=data_root / manifest.files[selected_language].path,
                    prompt_path=prompt,
                    output_dir=out,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                    dtype=dtype,
                )
            )


@app.command()
def report(
    results_dir: Annotated[Path, typer.Option("--results-dir")] = DEFAULT_RESULTS,
    out: Annotated[Path | None, typer.Option("--out", help="Leaderboard CSV path.")] = None,
    language: Language = None,
) -> None:
    """Aggregate stored runs into detailed and model-level leaderboards."""
    if not results_dir.is_dir():
        raise typer.BadParameter(f"no run results directory at {results_dir}")
    runs = _filter_languages(read_runs(results_dir), language)
    if not runs:
        raise typer.BadParameter(f"no run results found in {results_dir}")
    destination = out or results_dir / "leaderboard.csv"
    write_leaderboard_csv(runs, destination)
    aggregate_destination = destination.with_name("aggregates.csv")
    write_aggregates_csv(runs, aggregate_destination)
    for row in leaderboard_rows(runs):
        typer.echo(
            f"{row['model_id']:<34} [{row['language']}] f1={row['f1']:.3f} "
            f"acc={row['accuracy']:.3f} "
            f"mcc={row['matthews_corrcoef']:.3f} unparsed={row['unparsed_rate']:.3f}"
        )
    typer.echo(f"\nleaderboard written to {destination}")
    typer.echo(f"aggregates written to {aggregate_destination}")


@app.command()
def publish(
    repo_id: Annotated[str, typer.Argument(help="Hugging Face dataset repository id.")],
    results_dir: Annotated[Path, typer.Option("--results-dir")] = DEFAULT_RESULTS,
    benchmark_name: Annotated[str, typer.Option("--benchmark-name")] = "v3-multilingual",
    private: Annotated[bool, typer.Option(help="Create the dataset repository private.")] = False,
) -> None:
    """Push the stored runs, leaderboard and a generated card to the Hub."""
    from landuse_relevance_bench.adapters.hf_publish import publish_results, read_published_runs

    if not results_dir.is_dir():
        raise typer.BadParameter(f"no run results directory at {results_dir}")
    published = read_published_runs(results_dir)
    runs = list(published)
    if not runs:
        raise typer.BadParameter(f"no run results found in {results_dir}")
    write_leaderboard_csv(runs, results_dir / "leaderboard.csv")
    write_aggregates_csv(runs, results_dir / "aggregates.csv")
    url = publish_results(
        repo_id,
        results_dir,
        runs,
        private=private,
        benchmark_name=benchmark_name,
    )
    typer.echo(f"published {len(runs)} run(s) to {url}")


def _selected_languages(
    data_root: Path,
    selectors: list[str] | None,
) -> tuple[TranslationManifest, tuple[str, ...]]:
    try:
        manifest = load_manifest(data_root)
    except (OSError, TranslationDataError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    selected = _normalize_language_selectors(selectors)
    unknown = sorted(set(selected) - set(manifest.languages))
    if unknown:
        raise typer.BadParameter(
            f"unknown language(s) {', '.join(unknown)}; available: {', '.join(manifest.languages)}"
        )
    return manifest, tuple(selected or manifest.languages)


def _normalize_language_selectors(selectors: list[str] | None) -> list[str]:
    values = [
        value.strip().lower() for selector in selectors or [] for value in selector.split(",")
    ]
    if any(not value for value in values):
        raise typer.BadParameter("language selectors cannot be empty")
    return sorted(set(values))


def _filter_languages(results, selectors: list[str] | None):
    selected = _normalize_language_selectors(selectors)
    if not selected:
        return list(results)
    return [result for result in results if result.metadata.language in selected]

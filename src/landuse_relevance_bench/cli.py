"""Scriptable entry points for running, scoring and publishing the benchmark."""

import json
from pathlib import Path
from typing import Annotated, TypedDict

import typer

from landuse_relevance_bench import __version__
from landuse_relevance_bench.adapters.pipeline import (
    DEFAULT_DTYPE,
    DEFAULT_MAX_NEW_TOKENS,
    RunRequest,
)
from landuse_relevance_bench.adapters.providers import (
    cached_generator_provider,
    cached_scorer_provider,
)
from landuse_relevance_bench.adapters.results_store import read_runs, write_reports
from landuse_relevance_bench.adapters.translations import TranslationManifest
from landuse_relevance_bench.application import (
    DEFAULT_BENCHMARK_NAME,
    DEFAULT_DATA_ROOT,
    DEFAULT_PROMPT,
    DEFAULT_RESULTS,
    DEFAULT_SCORER_PROMPT,
    benchmark_one,
    comparable_runs,
    completed_pairs,
    languages_for_scorer,
    manifest_for,
    mode_options,
    planned_pairs,
    print_run_plan,
    publish_to_hub,
    report_lines,
    score_one,
    selected_languages,
    selected_models,
)
from landuse_relevance_bench.domain.orchestration import DEFAULT_BATCH_SIZE
from landuse_relevance_bench.domain.roster import ROSTER, model_ids
from landuse_relevance_bench.domain.scorers import SCORER_ROSTER, scorer_for, scorer_ids
from landuse_relevance_bench.domain.sharding import pair_statuses

app = typer.Typer(
    add_completion=False,
    help=__doc__,
    invoke_without_command=True,
    no_args_is_help=True,
)


@app.callback()
def _root(version: Annotated[bool, typer.Option("--version", is_eager=True)] = False) -> None:
    """Expose the installed package version to scripts and container checks."""
    if version:
        typer.echo(__version__)
        raise typer.Exit()


DataRoot = Annotated[Path, typer.Option("--data-root", help="Vendored multilingual data root.")]
Prompt = Annotated[Path, typer.Option("--prompt", help="Prompt template with a {} placeholder.")]
Results = Annotated[Path, typer.Option("--out", help="Directory to write run results into.")]
Language = Annotated[
    list[str] | None,
    typer.Option("--language", help="Language code(s), repeatable or comma-separated."),
]
Revision = Annotated[str | None, typer.Option(help="Pin the model to a commit.")]
BatchSizeOption = typer.Option(help="Prompts per forward pass.")
BatchSize = Annotated[int | None, BatchSizeOption]
ResultsDir = Annotated[
    Path, typer.Option("--results-dir", help="Directory holding stored run results.")
]
MaxNewTokens = Annotated[int, typer.Option(help="Maximum tokens to generate per prompt.")]
Seed = Annotated[int, typer.Option(help="Random seed for decoding.")]
Dtype = Annotated[str, typer.Option(help="Torch dtype name.")]
ContinuousBatching = Annotated[
    bool, typer.Option("--continuous-batching", help="Use Transformers continuous batching.")
]
Throughput = Annotated[
    bool, typer.Option("--throughput", help="Use SGLang multi-request throughput mode.")
]
ShardIndex = Annotated[int, typer.Option("--shard-index", help="Zero-based index of this shard.")]
ShardCount = Annotated[
    int, typer.Option("--shard-count", help="Total number of shards the work is split into.")
]
SkipExisting = Annotated[
    bool,
    typer.Option(
        "--skip-existing/--no-skip-existing",
        help="Skip model-language pairs that already have stored results.",
    ),
]


def _plan(
    models: tuple[str, ...] | list[str],
    selected: tuple[str, ...],
    manifest: TranslationManifest,
    shard: tuple[int, int],
    out: Path,
) -> tuple[tuple[str, str], ...]:
    """Pair models with languages for this shard and print the plan."""
    pairs = planned_pairs(models, selected, *shard)
    print_run_plan(models, selected, manifest, pairs, out)
    return pairs


class _Where(TypedDict):
    language: str
    benchmark_path: Path
    prompt_path: Path
    output_dir: Path


def _request_fields(
    manifest: TranslationManifest, data_root: Path, language: str, prompt: Path, out: Path
) -> _Where:
    """The request fields every command derives the same way from its options."""
    return {
        "language": language,
        "benchmark_path": data_root / manifest.files[language].path,
        "prompt_path": prompt,
        "output_dir": out,
    }


@app.command()
def models(as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False) -> None:
    """List the models in the benchmark roster."""
    if as_json:
        typer.echo(
            json.dumps(
                [
                    {
                        "id": spec.name,
                        "parameters": spec.total_parameters,
                        "runtime": spec.runtime,
                        "note": spec.note,
                    }
                    for spec in ROSTER
                ],
                sort_keys=True,
            )
        )
        return
    for spec in ROSTER:
        typer.echo(f"{spec.name}\t{spec.total_parameters / 1e9:.2f}B\t{spec.runtime}\t{spec.note}")


@app.command()
def scorers() -> None:
    """List the non-generative models scored on the same benchmark."""
    for spec in SCORER_ROSTER:
        typer.echo(f"{spec.model_id}\t{spec.total_parameters / 1e9:.2f}B\t{spec.kind}\t{spec.note}")


@app.command()
def languages(data_root: DataRoot = DEFAULT_DATA_ROOT) -> None:
    """List every active language and its configured row count."""
    manifest = manifest_for(data_root)
    for language in manifest.languages:
        typer.echo(f"{language}\t{manifest.files[language].rows}")


@app.command()
def run(
    model_id: Annotated[str, typer.Argument(help="Hugging Face model repository id.")],
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    prompt: Prompt = DEFAULT_PROMPT,
    out: Results = DEFAULT_RESULTS,
    language: Language = None,
    revision: Revision = None,
    batch_size: BatchSize = None,
    max_new_tokens: MaxNewTokens = DEFAULT_MAX_NEW_TOKENS,
    seed: Seed = 0,
    dtype: Dtype = DEFAULT_DTYPE,
    continuous_batching: ContinuousBatching = False,
    throughput: Throughput = False,
    shard_index: ShardIndex = 0,
    shard_count: ShardCount = 1,
) -> None:
    """Benchmark one model on every selected language."""
    if model_id in scorer_ids():
        raise typer.BadParameter(
            f"{model_id!r} is a scoring model; `lrb score` benchmarks scoring models"
        )
    manifest, selected = selected_languages(data_root, language)
    pairs = _plan((model_id,), selected, manifest, (shard_index, shard_count), out)
    provider = cached_generator_provider()
    try:
        for _, selected_language in pairs:
            benchmark_one(
                RunRequest.for_run(
                    model_id,
                    **_request_fields(manifest, data_root, selected_language, prompt, out),
                    revision=revision,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                    dtype=dtype,
                    continuous_batching=continuous_batching,
                    throughput_mode=throughput,
                    close_generator=False,
                ),
                provider,
            )
    finally:
        provider.close_cached()


@app.command()
def score(
    model_id: Annotated[str, typer.Argument(help="Scoring roster id.")],
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    prompt: Annotated[
        Path | None,
        typer.Option("--prompt", help="Prompt template; defaults to the scorer's own."),
    ] = None,
    out: Results = DEFAULT_RESULTS,
    language: Language = None,
    revision: Revision = None,
    batch_size: Annotated[int, BatchSizeOption] = DEFAULT_BATCH_SIZE,
    seed: Seed = 0,
    dtype: Dtype = DEFAULT_DTYPE,
    shard_index: ShardIndex = 0,
    shard_count: ShardCount = 1,
) -> None:
    """Score one non-generative model on every selected language."""
    if model_id not in scorer_ids():
        raise typer.BadParameter(
            f"{model_id!r} is not in the scoring roster; `lrb run` benchmarks generative models"
        )
    spec = scorer_for(model_id)
    prompt = prompt or Path(spec.prompt)
    manifest, selected = selected_languages(data_root, language)
    selected = languages_for_scorer(spec, selected, language)
    pairs = _plan((model_id,), selected, manifest, (shard_index, shard_count), out)
    provider = cached_scorer_provider()
    try:
        for _, selected_language in pairs:
            score_one(
                RunRequest(
                    model_id=model_id,
                    **_request_fields(manifest, data_root, selected_language, prompt, out),
                    revision=revision,
                    batch_size=batch_size,
                    seed=seed,
                    dtype=dtype,
                ),
                provider,
            )
    finally:
        provider.close_cached()


@app.command(name="run-all")
def run_all(
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    prompt: Prompt = DEFAULT_PROMPT,
    out: Results = DEFAULT_RESULTS,
    language: Language = None,
    batch_size: BatchSize = None,
    continuous_batching: ContinuousBatching = False,
    throughput: Throughput = False,
    max_new_tokens: MaxNewTokens = DEFAULT_MAX_NEW_TOKENS,
    seed: Seed = 0,
    dtype: Dtype = DEFAULT_DTYPE,
    shard_index: ShardIndex = 0,
    shard_count: ShardCount = 1,
    only: Annotated[str | None, typer.Option("--only", help="Regex over roster run names.")] = None,
    runtime: Annotated[
        list[str] | None,
        typer.Option("--runtime", help="Restrict to transformers or sglang (repeatable)."),
    ] = None,
    skip_existing: SkipExisting = True,
    keep_going: Annotated[
        bool, typer.Option("--keep-going", help="Continue after a failed run; exit 1 at the end.")
    ] = False,
) -> None:
    """Benchmark every rostered model on every selected language."""
    manifest, selected = selected_languages(data_root, language)
    models = selected_models(only, runtime)
    if not models:
        raise typer.BadParameter("no rostered runs match the selected filters")
    pairs = _plan(models, selected, manifest, (shard_index, shard_count), out)
    pairs_by_model: dict[str, list[str]] = {}
    for model, selected_language in pairs:
        pairs_by_model.setdefault(model, []).append(selected_language)
    failures = False
    for model, languages_for_model in pairs_by_model.items():
        provider = cached_generator_provider()
        model_batch_size, use_continuous_batching, use_throughput = mode_options(
            model,
            batch_size,
            continuous_batching=continuous_batching,
            throughput=throughput,
        )
        try:
            for selected_language in languages_for_model:
                try:
                    benchmark_one(
                        RunRequest.for_run(
                            model,
                            **_request_fields(manifest, data_root, selected_language, prompt, out),
                            batch_size=model_batch_size,
                            continuous_batching=use_continuous_batching,
                            throughput_mode=use_throughput,
                            max_new_tokens=max_new_tokens,
                            seed=seed,
                            dtype=dtype,
                            close_generator=False,
                        ),
                        provider,
                        skip_existing=skip_existing,
                    )
                except Exception as exc:
                    if not keep_going:
                        raise
                    typer.echo(f"{model} [{selected_language}] failed: {exc}", err=True)
                    failures = True
        finally:
            provider.close_cached()
    if failures:
        raise typer.Exit(code=1)


@app.command()
def status(
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    results_dir: ResultsDir = DEFAULT_RESULTS,
    language: Language = None,
    shard_index: ShardIndex = 0,
    shard_count: ShardCount = 1,
    include_scorers: Annotated[
        bool, typer.Option("--include-scorers", help="Also list scoring models.")
    ] = False,
) -> None:
    """Show complete or pending state for the selected model-language shard."""
    _, selected = selected_languages(data_root, language)
    roster = model_ids() + (scorer_ids() if include_scorers else ())
    pairs = planned_pairs(roster, selected, shard_index, shard_count)
    for pair_status in pair_statuses(pairs, completed_pairs(pairs, results_dir)):
        typer.echo(f"{pair_status.model_id}\t{pair_status.language}\t{pair_status.status}")


@app.command()
def report(
    results_dir: ResultsDir = DEFAULT_RESULTS,
    out: Annotated[Path | None, typer.Option("--out", help="Leaderboard CSV path.")] = None,
    language: Language = None,
) -> None:
    """Aggregate stored runs into detailed and model-level leaderboards."""
    if not results_dir.is_dir():
        raise typer.BadParameter(f"no run results directory at {results_dir}")
    try:
        stored = read_runs(results_dir)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    runs = comparable_runs(stored, language, results_dir)
    written = write_reports(runs, out or results_dir / "leaderboard.csv")
    for line in report_lines(runs):
        typer.echo(line)
    typer.echo(f"\nleaderboard written to {written['leaderboard']}")
    typer.echo(f"aggregates written to {written['aggregates']}")
    typer.echo(f"threshold sweep written to {written['threshold_sweep']}")
    typer.echo(f"scoring summary written to {written['scoring_summary']}")


@app.command()
def publish(
    repo_id: Annotated[str, typer.Argument(help="Hugging Face dataset repository id.")],
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    results_dir: ResultsDir = DEFAULT_RESULTS,
    prompt: Prompt = DEFAULT_PROMPT,
    scorer_prompt: Annotated[
        Path, typer.Option("--scorer-prompt", help="Prompt template used by scoring models.")
    ] = DEFAULT_SCORER_PROMPT,
    benchmark_name: Annotated[
        str, typer.Option("--benchmark-name", help="Benchmark name shown on the dataset card.")
    ] = DEFAULT_BENCHMARK_NAME,
    timing_dir: Annotated[
        Path | None,
        typer.Option(
            "--timing-dir", help="Same-GPU generative reruns for the log-prob timing row."
        ),
    ] = None,
    language: Language = None,
    private: Annotated[bool, typer.Option(help="Create the dataset repository private.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Preview without Hub upload.")] = False,
) -> None:
    """Push the stored runs, leaderboard and a generated card to the Hub."""
    result = publish_to_hub(
        repo_id,
        data_root=data_root,
        results_dir=results_dir,
        prompt=prompt,
        scorer_prompt=scorer_prompt,
        benchmark_name=benchmark_name,
        timing_dir=timing_dir,
        language=language,
        private=private,
        dry_run=dry_run,
    )
    if dry_run:
        typer.echo(f"dry-run: would publish {result.run_count} result(s) to {repo_id}")
        typer.echo(result.output)
        return
    typer.echo(f"published {result.run_count} run(s) to {result.output}")

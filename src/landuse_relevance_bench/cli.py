"""Scriptable entry points for running, scoring and publishing the benchmark."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Generic, TypeVar

import typer

from landuse_relevance_bench.adapters.benchmark_csv import BenchmarkFileError
from landuse_relevance_bench.adapters.pipeline import (
    DEFAULT_DTYPE,
    DEFAULT_MAX_NEW_TOKENS,
    GeneratorProvider,
    RunRequest,
    ScorerProvider,
    execute,
    execute_scoring,
)
from landuse_relevance_bench.adapters.prompt_file import PromptFileError, load_prompt
from landuse_relevance_bench.adapters.provenance import source_commit
from landuse_relevance_bench.adapters.results_store import (
    leaderboard_rows,
    read_run,
    read_runs,
    run_filename,
    write_reports,
)
from landuse_relevance_bench.adapters.translations import (
    TranslationDataError,
    TranslationManifest,
    load_manifest,
)
from landuse_relevance_bench.domain.agreement import speculative_agreements
from landuse_relevance_bench.domain.engine import LabelScorer, TextGenerator
from landuse_relevance_bench.domain.orchestration import DEFAULT_BATCH_SIZE
from landuse_relevance_bench.domain.records import RunResult
from landuse_relevance_bench.domain.roster import ROSTER, SGLANG, TRANSFORMERS, model_ids, spec_for
from landuse_relevance_bench.domain.scorers import (
    SCORER_ROSTER,
    ScorerSpec,
    scorer_for,
    scorer_ids,
)
from landuse_relevance_bench.domain.selection import (
    collection_comparability_errors,
    select_run_names,
)
from landuse_relevance_bench.domain.sharding import (
    model_language_pairs,
    pair_statuses,
    shard_pairs,
)

DEFAULT_DATA_ROOT = Path("data/translations")
DEFAULT_PROMPT = Path("data/prompt.txt")
DEFAULT_SCORER_PROMPT = Path("data/prompt_reranker.txt")
DEFAULT_RESULTS = Path("results")

T = TypeVar("T")

app = typer.Typer(add_completion=False, help=__doc__)

DataRoot = Annotated[Path, typer.Option("--data-root", help="Vendored multilingual data root.")]
Prompt = Annotated[Path, typer.Option("--prompt", help="Prompt template with a {} placeholder.")]
Results = Annotated[Path, typer.Option("--out", help="Directory to write run results into.")]
Language = Annotated[
    list[str] | None,
    typer.Option("--language", help="Language code(s), repeatable or comma-separated."),
]


def generator_provider() -> GeneratorProvider:
    """Dispatch lazily so the CLI works without either optional runtime installed."""

    def provide(request: RunRequest) -> tuple[TextGenerator, str]:
        if request.runtime == SGLANG:
            from landuse_relevance_bench.adapters.sglang_generator import provide as provide_sglang

            return provide_sglang(request)
        from landuse_relevance_bench.adapters.hf_generator import provide as provide_transformers

        return provide_transformers(request)

    return provide


def scorer_provider() -> ScorerProvider:
    """Imported lazily so the CLI stays usable without a model runtime installed."""
    from landuse_relevance_bench.adapters.hf_scorer import provide_scorer

    return provide_scorer


class _CachedProvider(Generic[T]):
    """Load one model lazily and reuse it for that model's selected languages."""

    def __init__(self, factory: Callable[[], Callable[[RunRequest], T]]) -> None:
        self._factory = factory
        self._loaded: T | None = None
        self._loaded_name: str | None = None

    def __call__(self, request: RunRequest) -> T:
        if self._loaded_name != request.name:
            self.close_cached()
            self._loaded = self._factory()(request)
            self._loaded_name = request.name
        if self._loaded is None:
            raise RuntimeError("cached provider failed to load a model")
        return self._loaded

    def close_cached(self) -> None:
        cached = self._loaded
        self._loaded = None
        self._loaded_name = None
        if cached is None:
            return
        instance = cached[0] if isinstance(cached, tuple) else cached
        close = getattr(instance, "close", None)
        if callable(close):
            close()


def _cached(factory: Callable[[], Callable[[RunRequest], T]]) -> _CachedProvider[T]:
    return _CachedProvider(factory)


def _close_cached(provider: _CachedProvider[T]) -> None:
    provider.close_cached()


def _cached_scorer_provider() -> _CachedProvider[tuple[LabelScorer, str]]:
    return _cached(scorer_provider)


def _cached_generator_provider() -> _CachedProvider[tuple[TextGenerator, str]]:
    return _cached(generator_provider)


_INPUT_ERRORS = (OSError, BenchmarkFileError, PromptFileError, TranslationDataError, ValueError)


def _run_pair(
    request: RunRequest,
    run: Callable[[], RunResult],
    extra: Callable[[RunResult], str],
    *,
    skip_existing: bool = True,
) -> None:
    """Run one model-language pair unless it is checkpointed; input problems are usage errors."""
    result_path = request.output_dir / run_filename(request.name, request.language)
    if skip_existing and result_path.is_file():
        try:
            existing = read_run(result_path)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if existing.metadata.name != request.name or existing.metadata.language != request.language:
            raise typer.BadParameter(f"{result_path} contains a different model-language run")
        typer.echo(f"{request.name} [{request.language}] already complete; skipping")
        return
    try:
        result = run()
    except _INPUT_ERRORS as exc:
        raise typer.BadParameter(str(exc)) from exc
    metrics = result.metrics
    typer.echo(
        f"{request.name} [{request.language}]  accuracy={metrics.accuracy:.3f}  "
        f"f1={metrics.f1:.3f}  mcc={metrics.matthews_corrcoef:.3f}  {extra(result)}"
        f"({result.metadata.duration_seconds:.1f}s)"
    )


def _benchmark_one(
    request: RunRequest,
    provider: GeneratorProvider | None = None,
    *,
    skip_existing: bool = True,
) -> None:
    """Benchmark one generative model on one language."""
    _run_pair(
        request,
        lambda: execute(request, provider or generator_provider(), source_commit=source_commit()),
        lambda result: f"unparsed={result.metrics.unparsed_rate:.3f}  ",
        skip_existing=skip_existing,
    )


def _score_one(request: RunRequest, provider: ScorerProvider | None = None) -> None:
    """Score one non-generative model on one language."""
    _run_pair(
        request,
        lambda: execute_scoring(
            request, provider or scorer_provider(), source_commit=source_commit()
        ),
        lambda result: (
            f"throughput={result.metadata.throughput_items_per_second or 0.0:.1f}/s  "
            f"peak_vram={result.metadata.peak_vram_bytes or 0}B  "
        ),
    )


@app.command()
def models() -> None:
    """List the models in the benchmark roster."""
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
    try:
        manifest = load_manifest(data_root)
    except (OSError, TranslationDataError) as exc:
        label = data_root.name or str(data_root)
        raise typer.BadParameter(f"{label}: {exc}") from exc
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
    batch_size: Annotated[int | None, typer.Option(help="Prompts per forward pass.")] = None,
    max_new_tokens: Annotated[int, typer.Option()] = DEFAULT_MAX_NEW_TOKENS,
    seed: Annotated[int, typer.Option()] = 0,
    dtype: Annotated[str, typer.Option(help="Torch dtype name.")] = DEFAULT_DTYPE,
    continuous_batching: Annotated[bool, typer.Option("--continuous-batching")] = False,
    throughput: Annotated[
        bool, typer.Option(help="Use SGLang multi-request throughput mode.")
    ] = False,
    shard_index: Annotated[int, typer.Option("--shard-index")] = 0,
    shard_count: Annotated[int, typer.Option("--shard-count")] = 1,
) -> None:
    """Benchmark one model on every selected language."""
    if model_id in scorer_ids():
        raise typer.BadParameter(
            f"{model_id!r} is a scoring model; `lrb score` benchmarks scoring models"
        )
    manifest, selected = _selected_languages(data_root, language)
    pairs = _planned_pairs((model_id,), selected, shard_index, shard_count)
    _print_run_plan((model_id,), selected, manifest, pairs, out)
    provider = _cached_generator_provider()
    try:
        for _, selected_language in pairs:
            _benchmark_one(
                RunRequest.for_run(
                    model_id,
                    language=selected_language,
                    benchmark_path=data_root / manifest.files[selected_language].path,
                    prompt_path=prompt,
                    output_dir=out,
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
        _close_cached(provider)


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
    revision: Annotated[str | None, typer.Option(help="Pin the model to a commit.")] = None,
    batch_size: Annotated[int, typer.Option(help="Prompts per forward pass.")] = DEFAULT_BATCH_SIZE,
    seed: Annotated[int, typer.Option()] = 0,
    dtype: Annotated[str, typer.Option(help="Torch dtype name.")] = DEFAULT_DTYPE,
    shard_index: Annotated[int, typer.Option("--shard-index")] = 0,
    shard_count: Annotated[int, typer.Option("--shard-count")] = 1,
) -> None:
    """Score one non-generative model on every selected language."""
    if model_id not in scorer_ids():
        raise typer.BadParameter(
            f"{model_id!r} is not in the scoring roster; `lrb run` benchmarks generative models"
        )
    spec = scorer_for(model_id)
    prompt = prompt or Path(spec.prompt)
    manifest, selected = _selected_languages(data_root, language)
    selected = _languages_for_scorer(spec, selected, language)
    pairs = _planned_pairs((model_id,), selected, shard_index, shard_count)
    _print_run_plan((model_id,), selected, manifest, pairs, out)
    provider = _cached_scorer_provider()
    for _, selected_language in pairs:
        _score_one(
            RunRequest(
                model_id=model_id,
                language=selected_language,
                benchmark_path=data_root / manifest.files[selected_language].path,
                prompt_path=prompt,
                output_dir=out,
                revision=revision,
                batch_size=batch_size,
                seed=seed,
                dtype=dtype,
            ),
            provider,
        )


def _mode_options(
    model: str,
    batch_size: int | None,
    *,
    continuous_batching: bool,
    throughput: bool,
) -> tuple[int | None, bool, bool]:
    """Resolve mode flags per model, leaving unsupported vision runs unmodified."""
    try:
        spec = spec_for(model)
    except KeyError:
        spec = None
    runtime = TRANSFORMERS if spec is None else spec.runtime
    resolved_batch_size = batch_size
    if throughput and runtime == SGLANG and resolved_batch_size is None:
        resolved_batch_size = DEFAULT_BATCH_SIZE
    return (
        resolved_batch_size,
        continuous_batching and runtime == TRANSFORMERS and not (spec and spec.vision),
        throughput and runtime == SGLANG,
    )


def _selected_models(only: str | None, runtimes: Sequence[str] | None) -> tuple[str, ...]:
    try:
        return tuple(select_run_names(model_ids(), only=only, runtimes=runtimes))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command(name="run-all")
def run_all(
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    prompt: Prompt = DEFAULT_PROMPT,
    out: Results = DEFAULT_RESULTS,
    language: Language = None,
    batch_size: Annotated[int | None, typer.Option()] = None,
    continuous_batching: Annotated[
        bool, typer.Option("--continuous-batching", help="Use Transformers continuous batching.")
    ] = False,
    throughput: Annotated[
        bool, typer.Option("--throughput", help="Use SGLang multi-request throughput mode.")
    ] = False,
    max_new_tokens: Annotated[int, typer.Option()] = DEFAULT_MAX_NEW_TOKENS,
    seed: Annotated[int, typer.Option()] = 0,
    dtype: Annotated[str, typer.Option()] = DEFAULT_DTYPE,
    shard_index: Annotated[int, typer.Option("--shard-index")] = 0,
    shard_count: Annotated[int, typer.Option("--shard-count")] = 1,
    only: Annotated[str | None, typer.Option("--only", help="Regex over roster run names.")] = None,
    runtime: Annotated[
        list[str] | None,
        typer.Option("--runtime", help="Restrict to transformers or sglang (repeatable)."),
    ] = None,
    skip_existing: Annotated[bool, typer.Option("--skip-existing")] = True,
    keep_going: Annotated[bool, typer.Option("--keep-going")] = False,
) -> None:
    """Benchmark every rostered model on every selected language."""
    manifest, selected = _selected_languages(data_root, language)
    models = _selected_models(only, runtime)
    if not models:
        raise typer.BadParameter("no rostered runs match the selected filters")
    pairs = _planned_pairs(models, selected, shard_index, shard_count)
    _print_run_plan(models, selected, manifest, pairs, out)
    pairs_by_model: dict[str, list[str]] = {}
    for model, selected_language in pairs:
        pairs_by_model.setdefault(model, []).append(selected_language)
    failures = False
    for model, languages_for_model in pairs_by_model.items():
        provider = _cached_generator_provider()
        model_batch_size, use_continuous_batching, use_throughput = _mode_options(
            model,
            batch_size,
            continuous_batching=continuous_batching,
            throughput=throughput,
        )
        try:
            for selected_language in languages_for_model:
                try:
                    _benchmark_one(
                        RunRequest.for_run(
                            model,
                            language=selected_language,
                            benchmark_path=data_root / manifest.files[selected_language].path,
                            prompt_path=prompt,
                            output_dir=out,
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
            _close_cached(provider)
    if failures:
        raise typer.Exit(code=1)


@app.command()
def status(
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    results_dir: Annotated[Path, typer.Option("--results-dir")] = DEFAULT_RESULTS,
    language: Language = None,
    shard_index: Annotated[int, typer.Option("--shard-index")] = 0,
    shard_count: Annotated[int, typer.Option("--shard-count")] = 1,
    include_scorers: Annotated[
        bool, typer.Option("--include-scorers", help="Also list scoring models.")
    ] = False,
) -> None:
    """Show complete or pending state for the selected model-language shard."""
    _, selected = _selected_languages(data_root, language)
    roster = model_ids() + (scorer_ids() if include_scorers else ())
    pairs = _planned_pairs(roster, selected, shard_index, shard_count)
    completed = set()
    for model_id, selected_language in pairs:
        path = results_dir / run_filename(model_id, selected_language)
        if not path.is_file():
            continue
        try:
            result = read_run(path)
        except ValueError:
            continue
        if result.metadata.name == model_id and result.metadata.language == selected_language:
            completed.add((model_id, selected_language))
    for pair_status in pair_statuses(pairs, completed):
        typer.echo(f"{pair_status.model_id}\t{pair_status.language}\t{pair_status.status}")


def _planned_pairs(
    models: tuple[str, ...] | list[str],
    languages: tuple[str, ...],
    shard_index: int,
    shard_count: int,
) -> tuple[tuple[str, str], ...]:
    try:
        return shard_pairs(
            model_language_pairs(models, languages),
            shard_index,
            shard_count,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _print_run_plan(
    models: Sequence[str],
    selected: tuple[str, ...],
    manifest: TranslationManifest,
    pairs: Sequence[tuple[str, str]],
    results_dir: Path,
) -> None:
    """Show workload size and valid checkpoint state before loading a model."""
    completed = pending = invalid = 0
    for model_id, language in pairs:
        path = results_dir / run_filename(model_id, language)
        if not path.is_file():
            pending += 1
            continue
        try:
            result = read_run(path)
        except (OSError, ValueError):
            invalid += 1
            continue
        if result.metadata.name == model_id and result.metadata.language == language:
            completed += 1
        else:
            invalid += 1
    prompt_count = sum(manifest.files[language].rows for language in selected) * len(models)
    assigned_prompts = sum(manifest.files[language].rows for _, language in pairs)
    typer.echo(
        f"plan: models={len(models)} languages={len(selected)} [{','.join(selected)}] "
        f"prompts={prompt_count} assigned_pairs={len(pairs)} "
        f"assigned_prompts={assigned_prompts} "
        f"resume={completed} complete,{pending} pending,{invalid} invalid"
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
    try:
        runs = _filter_languages(read_runs(results_dir), language)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not runs:
        raise typer.BadParameter(f"no run results found in {results_dir}")
    incompatibilities = collection_comparability_errors(runs)
    if incompatibilities:
        raise typer.BadParameter("; ".join(incompatibilities))
    written = write_reports(runs, out or results_dir / "leaderboard.csv")
    for row in leaderboard_rows(runs):
        typer.echo(
            f"{row['model_id']:<34} [{row['language']}] f1={row['f1']:.3f} "
            f"acc={row['accuracy']:.3f} "
            f"mcc={row['matthews_corrcoef']:.3f} unparsed={row['unparsed_rate']:.3f}"
        )
    for agreement in speculative_agreements(runs):
        verdict = "lossless" if agreement.lossless else "MISMATCH"
        coverage = (
            f"{agreement.n_compared}/{agreement.n_speculative} speculative, "
            f"{agreement.n_compared}/{agreement.n_baseline} baseline"
        )
        typer.echo(
            f"agreement: {agreement.speculative_run} [{agreement.language}] vs "
            f"{agreement.baseline_run}: {verdict}; coverage={coverage}; "
            f"verdict_differences={agreement.verdicts_differ}; "
            f"text_differences={agreement.texts_differ}"
        )
    typer.echo(f"\nleaderboard written to {written['leaderboard']}")
    typer.echo(f"aggregates written to {written['aggregates']}")
    typer.echo(f"threshold sweep written to {written['threshold_sweep']}")
    typer.echo(f"scoring summary written to {written['scoring_summary']}")


@app.command()
def publish(
    repo_id: Annotated[str, typer.Argument(help="Hugging Face dataset repository id.")],
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    results_dir: Annotated[Path, typer.Option("--results-dir")] = DEFAULT_RESULTS,
    prompt: Prompt = DEFAULT_PROMPT,
    scorer_prompt: Annotated[
        Path, typer.Option("--scorer-prompt", help="Prompt template used by scoring models.")
    ] = DEFAULT_SCORER_PROMPT,
    benchmark_name: Annotated[str, typer.Option("--benchmark-name")] = "v3-multilingual",
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
    from landuse_relevance_bench.adapters.hf_publish import publish_results, read_published_runs

    if not results_dir.is_dir():
        raise typer.BadParameter(f"no run results directory at {results_dir}")
    try:
        published = read_published_runs(results_dir)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    selected_languages: list[str] | None = None
    if language is not None:
        _, selected = _selected_languages(data_root, language)
        selected_languages = list(selected)
    runs = _filter_languages(published, selected_languages)
    if not runs:
        raise typer.BadParameter(f"no run results found in {results_dir}")
    incompatibilities = collection_comparability_errors(runs)
    if incompatibilities:
        raise typer.BadParameter("; ".join(incompatibilities))
    # A reranker's score is not calibrated to a 0.5 boundary, so the sweep is published
    # alongside the headline rows.
    if not dry_run:
        write_reports(runs, results_dir / "leaderboard.csv")
    try:
        prompt_text = load_prompt(prompt)
        scorer_prompt_text = load_prompt(scorer_prompt) if scorer_prompt.is_file() else ""
    except (OSError, PromptFileError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if dry_run:
        from landuse_relevance_bench.adapters.hf_publish import dataset_card

        preview = dataset_card(
            runs,
            benchmark_name=benchmark_name,
            prompt_text=prompt_text,
            scorer_prompt_text=scorer_prompt_text,
            extra_scorer_prompt_texts=_extra_scorer_prompts(scorer_prompt),
            timing_results=read_runs(timing_dir) if timing_dir else (),
            viewer_file="data/train.csv",
        )
        typer.echo(f"dry-run: would publish {len(runs)} result(s) to {repo_id}")
        typer.echo(preview)
        return
    url = publish_results(
        repo_id,
        results_dir,
        runs,
        private=private,
        benchmark_name=benchmark_name,
        prompt_text=prompt_text,
        scorer_prompt_text=scorer_prompt_text,
        extra_scorer_prompt_texts=_extra_scorer_prompts(scorer_prompt),
        timing_results=read_runs(timing_dir) if timing_dir else (),
        data_root=data_root,
        allow_patterns=(
            _publish_allow_patterns(results_dir, runs) if selected_languages is not None else None
        ),
    )
    typer.echo(f"published {len(runs)} run(s) to {url}")


def _publish_allow_patterns(results_dir: Path, runs: Sequence[RunResult]) -> list[str]:
    """Limit a language-filtered upload to its runs and regenerated release files."""
    patterns = {
        "README.md",
        "leaderboard.csv",
        "aggregates.csv",
        "threshold_sweep.csv",
        "scoring_summary.csv",
        "data/train.csv",
        *(str(run_filename(run.metadata.model_id, run.metadata.language)) for run in runs),
    }
    if (results_dir / ".gitattributes").is_file():
        patterns.add(".gitattributes")
    return sorted(patterns)


def _extra_scorer_prompts(primary: Path) -> tuple[str, ...]:
    """Every other prompt file a rostered scorer declares, for the card to match digests."""
    paths = sorted({Path(spec.prompt) for spec in SCORER_ROSTER} - {primary, DEFAULT_PROMPT})
    try:
        return tuple(load_prompt(path) for path in paths if path.is_file())
    except (OSError, PromptFileError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _selected_languages(
    data_root: Path,
    selectors: list[str] | None,
) -> tuple[TranslationManifest, tuple[str, ...]]:
    try:
        manifest = load_manifest(data_root)
    except (OSError, TranslationDataError) as exc:
        label = data_root.name or str(data_root)
        raise typer.BadParameter(f"{label}: {exc}") from exc
    selected = _normalize_language_selectors(selectors)
    unknown = sorted(set(selected) - set(manifest.languages))
    if unknown:
        raise typer.BadParameter(
            f"unknown language(s) {', '.join(unknown)}; available: {', '.join(manifest.languages)}"
        )
    return manifest, tuple(selected or manifest.languages)


def _languages_for_scorer(
    spec: ScorerSpec, selected: tuple[str, ...], selectors: list[str] | None
) -> tuple[str, ...]:
    """Use only a scorer's declared language coverage, rejecting explicit mismatches."""
    if spec.supported_languages is None:
        return selected
    supported = set(spec.supported_languages)
    unsupported = sorted(set(selected) - supported)
    if selectors is not None and unsupported:
        raise typer.BadParameter(
            f"{spec.model_id} only supports {', '.join(spec.supported_languages)}; "
            f"unsupported requested languages: {', '.join(unsupported)}"
        )
    filtered = tuple(language for language in selected if language in supported)
    if not filtered:
        raise typer.BadParameter(f"no selected language is supported by {spec.model_id}")
    if unsupported:
        typer.echo(
            f"{spec.model_id} supports {', '.join(spec.supported_languages)}; "
            "running those languages only"
        )
    return filtered


def _normalize_language_selectors(selectors: list[str] | None) -> list[str]:
    values = [
        value.strip().lower() for selector in selectors or [] for value in selector.split(",")
    ]
    if any(not value for value in values):
        raise typer.BadParameter("language selectors cannot be empty")
    return sorted(set(values))


def _filter_languages(results: Sequence[RunResult], selectors: list[str] | None) -> list[RunResult]:
    selected = _normalize_language_selectors(selectors)
    if not selected:
        return list(results)
    return [result for result in results if result.metadata.language in selected]

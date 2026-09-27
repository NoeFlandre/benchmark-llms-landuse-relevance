"""Command use cases behind the CLI: selection, run planning, checkpointed runs, reporting.

Input problems surface as ``typer.BadParameter`` so every command reports them as usage
errors; the command definitions themselves stay in :mod:`landuse_relevance_bench.cli`.
"""

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import typer

from landuse_relevance_bench.adapters.benchmark_csv import BenchmarkFileError
from landuse_relevance_bench.adapters.pipeline import (
    GeneratorProvider,
    RunRequest,
    ScorerProvider,
    execute,
    execute_scoring,
)
from landuse_relevance_bench.adapters.prompt_file import PromptFileError, load_prompt
from landuse_relevance_bench.adapters.provenance import source_commit
from landuse_relevance_bench.adapters.providers import generator_provider, scorer_provider
from landuse_relevance_bench.adapters.results_store import leaderboard_rows, read_run, run_filename
from landuse_relevance_bench.adapters.translations import (
    TranslationDataError,
    TranslationManifest,
    load_manifest,
)
from landuse_relevance_bench.domain.agreement import speculative_agreements
from landuse_relevance_bench.domain.orchestration import DEFAULT_BATCH_SIZE
from landuse_relevance_bench.domain.records import RunResult
from landuse_relevance_bench.domain.roster import SGLANG, TRANSFORMERS, model_ids, spec_for
from landuse_relevance_bench.domain.scorers import SCORER_ROSTER, ScorerSpec
from landuse_relevance_bench.domain.selection import (
    collection_comparability_errors,
    select_run_names,
)
from landuse_relevance_bench.domain.sharding import model_language_pairs, shard_pairs

DEFAULT_DATA_ROOT = Path("data/translations")
DEFAULT_PROMPT = Path("data/prompt.txt")
DEFAULT_SCORER_PROMPT = Path("data/prompt_reranker.txt")
DEFAULT_RESULTS = Path("results")

_INPUT_ERRORS = (OSError, BenchmarkFileError, PromptFileError, TranslationDataError, ValueError)


def run_pair(
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


def benchmark_one(
    request: RunRequest,
    provider: GeneratorProvider | None = None,
    *,
    skip_existing: bool = True,
) -> None:
    """Benchmark one generative model on one language."""
    run_pair(
        request,
        lambda: execute(request, provider or generator_provider(), source_commit=source_commit()),
        lambda result: f"unparsed={result.metrics.unparsed_rate:.3f}  ",
        skip_existing=skip_existing,
    )


def score_one(request: RunRequest, provider: ScorerProvider | None = None) -> None:
    """Score one non-generative model on one language."""
    run_pair(
        request,
        lambda: execute_scoring(
            request, provider or scorer_provider(), source_commit=source_commit()
        ),
        lambda result: (
            f"throughput={result.metadata.throughput_items_per_second or 0.0:.1f}/s  "
            f"peak_vram={result.metadata.peak_vram_bytes or 0}B  "
        ),
    )


def mode_options(
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


def selected_models(only: str | None, runtimes: Sequence[str] | None) -> tuple[str, ...]:
    try:
        return tuple(select_run_names(model_ids(), only=only, runtimes=runtimes))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def planned_pairs(
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


def completed_pairs(pairs: Sequence[tuple[str, str]], results_dir: Path) -> set[tuple[str, str]]:
    """The pairs whose stored result is readable and belongs to that model-language pair."""
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
    return completed


def print_run_plan(
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


def comparable_runs(
    runs: Sequence[RunResult], selectors: list[str] | None, results_dir: Path
) -> list[RunResult]:
    """Filter runs to the selected languages and reject a collection that cannot be ranked."""
    selected = filter_languages(runs, selectors)
    if not selected:
        raise typer.BadParameter(f"no run results found in {results_dir}")
    incompatibilities = collection_comparability_errors(selected)
    if incompatibilities:
        raise typer.BadParameter("; ".join(incompatibilities))
    return selected


def report_lines(runs: Sequence[RunResult]) -> Iterator[str]:
    """The leaderboard rows, then each speculative run's agreement with its baseline."""
    for row in leaderboard_rows(runs):
        yield (
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
        yield (
            f"agreement: {agreement.speculative_run} [{agreement.language}] vs "
            f"{agreement.baseline_run}: {verdict}; coverage={coverage}; "
            f"verdict_differences={agreement.verdicts_differ}; "
            f"text_differences={agreement.texts_differ}"
        )


def publish_allow_patterns(results_dir: Path, runs: Sequence[RunResult]) -> list[str]:
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


def load_prompts(prompt: Path, scorer_prompt: Path) -> tuple[str, str]:
    """The generative and scorer prompt texts; the scorer's is empty when its file is absent."""
    try:
        prompt_text = load_prompt(prompt)
        scorer_prompt_text = load_prompt(scorer_prompt) if scorer_prompt.is_file() else ""
    except (OSError, PromptFileError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    return prompt_text, scorer_prompt_text


def extra_scorer_prompts(primary: Path) -> tuple[str, ...]:
    """Every other prompt file a rostered scorer declares, for the card to match digests."""
    paths = sorted({Path(spec.prompt) for spec in SCORER_ROSTER} - {primary, DEFAULT_PROMPT})
    try:
        return tuple(load_prompt(path) for path in paths if path.is_file())
    except (OSError, PromptFileError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def manifest_for(data_root: Path) -> TranslationManifest:
    try:
        return load_manifest(data_root)
    except (OSError, TranslationDataError) as exc:
        label = data_root.name or str(data_root)
        raise typer.BadParameter(f"{label}: {exc}") from exc


def selected_languages(
    data_root: Path,
    selectors: list[str] | None,
) -> tuple[TranslationManifest, tuple[str, ...]]:
    manifest = manifest_for(data_root)
    selected = normalize_language_selectors(selectors)
    unknown = sorted(set(selected) - set(manifest.languages))
    if unknown:
        raise typer.BadParameter(
            f"unknown language(s) {', '.join(unknown)}; available: {', '.join(manifest.languages)}"
        )
    return manifest, tuple(selected or manifest.languages)


def languages_for_scorer(
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


def normalize_language_selectors(selectors: list[str] | None) -> list[str]:
    values = [
        value.strip().lower() for selector in selectors or [] for value in selector.split(",")
    ]
    if any(not value for value in values):
        raise typer.BadParameter("language selectors cannot be empty")
    return sorted(set(values))


def filter_languages(results: Sequence[RunResult], selectors: list[str] | None) -> list[RunResult]:
    selected = normalize_language_selectors(selectors)
    if not selected:
        return list(results)
    return [result for result in results if result.metadata.language in selected]

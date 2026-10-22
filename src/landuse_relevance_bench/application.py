"""Command use cases behind the CLI: selection, run planning, checkpointed runs, reporting.

Input problems surface as ``typer.BadParameter`` so every command reports them as usage
errors; the command definitions themselves stay in :mod:`landuse_relevance_bench.cli`.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Literal, NamedTuple, TypedDict

import typer

from landuse_relevance_bench.adapters.benchmark_csv import BenchmarkFileError
from landuse_relevance_bench.adapters.hashing import sha256_of_file, sha256_of_text
from landuse_relevance_bench.adapters.paths import default_paths
from landuse_relevance_bench.adapters.pipeline import (
    EXPERIMENT_IDENTITY_SCHEMA_VERSION,
    GeneratorProvider,
    RunRequest,
    ScorerProvider,
    execute,
    execute_scoring,
    experiment_identity_for_request,
)
from landuse_relevance_bench.adapters.prompt_file import PromptFileError, load_prompt
from landuse_relevance_bench.adapters.provenance import source_commit
from landuse_relevance_bench.adapters.providers import generator_provider, scorer_provider
from landuse_relevance_bench.adapters.results_store import (
    leaderboard_rows,
    read_run,
    read_runs,
    run_filename,
    write_reports,
)
from landuse_relevance_bench.adapters.revision import is_immutable_revision, resolve_revision
from landuse_relevance_bench.adapters.translations import (
    TranslationDataError,
    TranslationManifest,
    load_manifest,
)
from landuse_relevance_bench.domain.agreement import speculative_agreements
from landuse_relevance_bench.domain.records import RunResult
from landuse_relevance_bench.domain.roster import model_ids
from landuse_relevance_bench.domain.run_modes import resolve_mode_options, spec_or_none
from landuse_relevance_bench.domain.scorers import SCORER_ROSTER, ScorerSpec
from landuse_relevance_bench.domain.selection import (
    collection_comparability_errors,
    select_run_names,
)
from landuse_relevance_bench.domain.sharding import model_language_pairs, shard_pairs
from landuse_relevance_bench.domain.variants import repository_of

_DEFAULTS = default_paths()
DEFAULT_DATA_ROOT = _DEFAULTS.data_root
DEFAULT_PROMPT = _DEFAULTS.prompt
DEFAULT_SCORER_PROMPT = _DEFAULTS.scorer_prompt
DEFAULT_RESULTS = _DEFAULTS.results

_INPUT_ERRORS = (OSError, BenchmarkFileError, PromptFileError, TranslationDataError, ValueError)


CheckpointState = Literal["complete", "pending", "invalid", "legacy"]


def _checkpoint_result(
    path: Path, model: str, language: str
) -> tuple[CheckpointState, RunResult | None]:
    """Read a usable checkpoint, distinguishing legacy results that lack identity."""
    if not path.is_file():
        return "pending", None
    try:
        result = read_run(path)
    except (OSError, ValueError):
        return "invalid", None
    if result.metadata.name != model or result.metadata.language != language:
        return "invalid", None
    identity = result.metadata.experiment_identity
    if (
        not isinstance(identity, Mapping)
        or type(identity.get("schema_version")) is not int
        or identity.get("schema_version") != EXPERIMENT_IDENTITY_SCHEMA_VERSION
    ):
        return "legacy", result
    return "complete", result


def _checkpoint_state(path: Path, model: str, language: str) -> CheckpointState:
    """Classify a saved model-language result and reject metadata without identity."""
    return _checkpoint_result(path, model, language)[0]


def _requested_experiment_identity(request: RunRequest) -> dict[str, object]:
    """Hash current inputs and resolve model refs before checking a saved result."""
    prompt = load_prompt(request.prompt_path)
    model_repository = repository_of(request.model_id)
    resolved_model_revision = resolve_revision(model_repository, request.revision)
    if not is_immutable_revision(resolved_model_revision):
        raise ValueError(
            f"could not resolve an immutable revision for model {model_repository}; "
            "refusing to reuse the stored result"
        )
    resolved_draft_revision = ""
    if request.draft_model_id:
        resolved_draft_revision = resolve_revision(request.draft_model_id, request.draft_revision)
        if not is_immutable_revision(resolved_draft_revision):
            raise ValueError(
                "could not resolve an immutable revision for draft model "
                f"{request.draft_model_id}; refusing to reuse the stored result"
            )
    return experiment_identity_for_request(
        request,
        prompt_sha256=sha256_of_text(prompt),
        benchmark_sha256=sha256_of_file(request.benchmark_path),
        resolved_model_revision=resolved_model_revision.lower(),
        resolved_draft_revision=resolved_draft_revision.lower(),
    )


def _identity_mismatches(stored: Mapping[str, object], expected: Mapping[str, object]) -> list[str]:
    """Name the request fields that differ from the saved experiment."""
    missing = object()
    return sorted(
        key
        for key in stored.keys() | expected.keys()
        if stored.get(key, missing) != expected.get(key, missing)
    )


def run_pair(
    request: RunRequest,
    run: Callable[[], RunResult],
    extra: Callable[[RunResult], str],
    *,
    skip_existing: bool = True,
) -> None:
    """Run one model-language pair unless it is checkpointed; input problems are usage errors."""
    result_path = request.output_dir / run_filename(request.name, request.language)
    if skip_existing:
        state, stored_result = _checkpoint_result(result_path, request.name, request.language)
        if state == "invalid":
            raise typer.BadParameter(
                f"{result_path} is unreadable or contains a different model-language run"
            )
        if state == "legacy":
            raise typer.BadParameter(
                f"{result_path} has legacy or unsupported metadata without a usable "
                "experiment identity; refusing to reuse or overwrite the stored result"
            )
        if state == "complete":
            try:
                expected = _requested_experiment_identity(request)
            except _INPUT_ERRORS as exc:
                raise typer.BadParameter(str(exc)) from exc
            if stored_result is None:
                raise typer.BadParameter(f"{result_path} could not be read as a run result")
            stored = stored_result.metadata.experiment_identity
            differences = _identity_mismatches(stored, expected)
            if differences:
                raise typer.BadParameter(
                    f"{result_path} experiment identity differs in "
                    f"{', '.join(differences)}; refusing to reuse or overwrite the stored result"
                )
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
    return resolve_mode_options(
        spec_or_none(model),
        batch_size,
        continuous_batching=continuous_batching,
        throughput=throughput,
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
    return {
        (model_id, selected_language)
        for model_id, selected_language in pairs
        if _checkpoint_state(
            results_dir / run_filename(model_id, selected_language), model_id, selected_language
        )
        == "complete"
    }


def print_run_plan(
    models: Sequence[str],
    selected: tuple[str, ...],
    manifest: TranslationManifest,
    pairs: Sequence[tuple[str, str]],
    results_dir: Path,
) -> None:
    """Show workload size and valid checkpoint state before loading a model."""
    states = [
        _checkpoint_state(results_dir / run_filename(model_id, language), model_id, language)
        for model_id, language in pairs
    ]
    completed = states.count("complete")
    pending = states.count("pending")
    invalid = len(states) - completed - pending
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


DEFAULT_BENCHMARK_NAME = "v3-multilingual"
VIEWER_FILE = "data/train.csv"
RELEASE_FILES = (
    "README.md",
    "docs/results-and-methodology.md",
    "leaderboard.csv",
    "aggregates.csv",
    "threshold_sweep.csv",
    "scoring_summary.csv",
    VIEWER_FILE,
)


def publish_allow_patterns(
    results_dir: Path,
    runs: Sequence[RunResult],
    *,
    include_snapshot_status: bool = False,
) -> list[str]:
    """Limit uploads to selected runs and the generated release files."""
    patterns = {
        *RELEASE_FILES,
        *(str(run_filename(run.metadata.name, run.metadata.language)) for run in runs),
    }
    if include_snapshot_status and (results_dir / "SNAPSHOT_STATUS.md").is_file():
        patterns.add("SNAPSHOT_STATUS.md")
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


class _CardInputs(TypedDict):
    """Card and timing inputs shared by the dry-run preview and the Hub upload."""

    benchmark_name: str
    prompt_text: str
    scorer_prompt_text: str
    extra_scorer_prompt_texts: tuple[str, ...]
    timing_results: Sequence[RunResult]


class PublishOutput(NamedTuple):
    run_count: int
    output: str  # the Hub URL, or the generated card on a dry run


def publish_to_hub(  # noqa: PLR0913 - one parameter per publish option, mirroring the CLI
    repo_id: str,
    *,
    data_root: Path,
    results_dir: Path,
    prompt: Path,
    scorer_prompt: Path,
    benchmark_name: str,
    timing_dir: Path | None,
    language: list[str] | None,
    private: bool,
    dry_run: bool,
) -> PublishOutput:
    """Push the stored runs, leaderboard and a generated card, or preview the card on a dry run."""
    # Only the publish path needs the Hub adapter, so the import stays inside the function.
    from landuse_relevance_bench.adapters.hf_publish import (  # noqa: PLC0415
        CardOptions,
        dataset_card,
        publish_results,
        read_published_runs,
    )

    if not results_dir.is_dir():
        raise typer.BadParameter(f"no run results directory at {results_dir}")
    try:
        published = read_published_runs(results_dir)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    language_filter: list[str] | None = None
    if language is not None:
        _, selected = selected_languages(data_root, language)
        language_filter = list(selected)
    runs = comparable_runs(published, language_filter, results_dir)
    if not dry_run:
        # A reranker's score is not calibrated to a 0.5 boundary, so the threshold sweep
        # is written beside the leaderboard and linked from the card.
        write_reports(runs, results_dir / "leaderboard.csv")
    prompt_text, scorer_prompt_text = load_prompts(prompt, scorer_prompt)
    card: _CardInputs = {
        "benchmark_name": benchmark_name,
        "prompt_text": prompt_text,
        "scorer_prompt_text": scorer_prompt_text,
        "extra_scorer_prompt_texts": extra_scorer_prompts(scorer_prompt),
        "timing_results": read_runs(timing_dir) if timing_dir else (),
    }
    if dry_run:
        preview = dataset_card(
            runs,
            benchmark_name=card["benchmark_name"],
            prompt_text=card["prompt_text"],
            options=CardOptions(
                scorer_prompt_text=card["scorer_prompt_text"],
                extra_scorer_prompt_texts=card["extra_scorer_prompt_texts"],
                timing_results=card["timing_results"],
                viewer_file=VIEWER_FILE,
            ),
        )
        return PublishOutput(len(runs), preview)
    url = publish_results(
        repo_id,
        results_dir,
        runs,
        private=private,
        data_root=data_root,
        allow_patterns=publish_allow_patterns(
            results_dir, runs, include_snapshot_status=language_filter is None
        ),
        **card,
    )
    return PublishOutput(len(runs), url)

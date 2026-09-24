"""Wiring one model to the benchmark and persisting what came out."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from landuse_relevance_bench.adapters.benchmark_csv import load_benchmark
from landuse_relevance_bench.adapters.hashing import sha256_of_file, sha256_of_text
from landuse_relevance_bench.adapters.prompt_file import load_prompt
from landuse_relevance_bench.adapters.results_store import write_run
from landuse_relevance_bench.domain.engine import LabelScorer, TextGenerator
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.orchestration import (
    DEFAULT_BATCH_SIZE,
    predict_all,
    score_all,
)
from landuse_relevance_bench.domain.records import (
    SCORING,
    Prediction,
    RunMetadata,
    RunResult,
    outcomes_of,
)
from landuse_relevance_bench.domain.roster import quantization_of
from landuse_relevance_bench.domain.scorers import scorer_for

DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_DTYPE = "bfloat16"
SCORING_SEQUENCE_LENGTH = 8192


@dataclass(frozen=True, slots=True)
class RunRequest:
    """Everything one benchmark run needs, resolved from the CLI."""

    model_id: str
    language: str
    benchmark_path: Path
    prompt_path: Path
    output_dir: Path
    revision: str | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    seed: int = 0
    dtype: str = DEFAULT_DTYPE


GeneratorProvider = Callable[[RunRequest], tuple[TextGenerator, str]]
ScorerProvider = Callable[[RunRequest], tuple[LabelScorer, str]]


def execute(
    request: RunRequest,
    provide_generator: GeneratorProvider,
    *,
    source_commit: str = "",
) -> RunResult:
    """Run the benchmark for one model and write the result next to the others."""
    items = load_benchmark(request.benchmark_path, expected_language=request.language)
    template = load_prompt(request.prompt_path)
    generator, revision = provide_generator(request)
    run = _timed(lambda: predict_all(items, template, generator, batch_size=request.batch_size))
    metadata = _metadata(
        request,
        generator,
        run,
        revision=revision,
        template=template,
        source_commit=source_commit,
        max_new_tokens=request.max_new_tokens,
        decoding="greedy",
        quantization=quantization_of(request.model_id),
    )
    return _store(request, metadata, run.predictions)


def execute_scoring(
    request: RunRequest,
    provide_scorer: ScorerProvider,
    *,
    source_commit: str = "",
) -> RunResult:
    """Score one non-generative model over the benchmark and store the result.

    Written into the same tree as a generative run so a language's results stay
    together, but marked as a scoring run and carrying the rule that produced its
    verdicts, so the two families can be reported apart.
    """
    spec = scorer_for(request.model_id)
    items = load_benchmark(request.benchmark_path, expected_language=request.language)
    template = load_prompt(request.prompt_path)
    scorer, revision = provide_scorer(request)

    def measured() -> tuple[Prediction, ...]:
        _begin_measurement(scorer)
        try:
            return score_all(items, template, scorer, batch_size=request.batch_size)
        finally:
            _end_measurement(scorer)

    run = _timed(measured)
    metadata = _metadata(
        request,
        scorer,
        run,
        revision=revision,
        template=template,
        source_commit=source_commit,
        # Nothing is decoded, so there is no token budget and no decoding strategy.
        max_new_tokens=0,
        decoding="none",
        inference=SCORING,
        decision_rule=spec.decision_rule,
        peak_vram_bytes=_peak_vram_bytes(scorer),
        sequence_length=_sequence_length(scorer),
    )
    return _store(request, metadata, run.predictions)


@dataclass(frozen=True, slots=True)
class _TimedRun:
    predictions: tuple[Prediction, ...]
    started_at: str
    duration: float


def _timed(run: Callable[[], tuple[Prediction, ...]]) -> _TimedRun:
    started_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    started = time.monotonic()
    predictions = run()
    return _TimedRun(predictions, started_at, time.monotonic() - started)


def _metadata(  # noqa: PLR0913 - the provenance fields are distinct inputs
    request: RunRequest,
    model: object,
    run: _TimedRun,
    *,
    revision: str,
    template: str,
    source_commit: str,
    **specific: Any,
) -> RunMetadata:
    """Fields every run records, plus the ones its method adds."""
    duration = run.duration
    return RunMetadata(
        model_id=request.model_id,
        language=request.language,
        model_revision=revision,
        prompt_sha256=sha256_of_text(template),
        benchmark_sha256=sha256_of_file(request.benchmark_path),
        batch_size=_effective_batch_size(model, request),
        seed=request.seed,
        dtype=str(getattr(model, "runtime_dtype", request.dtype)),
        started_at=run.started_at,
        duration_seconds=round(duration, 3),
        source_commit=source_commit,
        throughput_items_per_second=len(run.predictions) / duration if duration > 0.0 else None,
        device_name=_device_name(),
        **specific,
    )


def _store(
    request: RunRequest, metadata: RunMetadata, predictions: tuple[Prediction, ...]
) -> RunResult:
    result = RunResult(
        metadata=metadata,
        predictions=predictions,
        metrics=evaluate(outcomes_of(predictions)),
    )
    write_run(result, request.output_dir)
    return result


def _begin_measurement(scorer: LabelScorer) -> None:
    hook = getattr(scorer, "begin_measurement", None)
    if callable(hook):
        hook()


def _end_measurement(scorer: LabelScorer) -> None:
    hook = getattr(scorer, "end_measurement", None)
    if callable(hook):
        hook()


def _peak_vram_bytes(scorer: LabelScorer) -> int | None:
    value = getattr(scorer, "peak_vram_bytes", None)
    return int(value) if value is not None else None


def _device_name() -> str:
    """The GPU a run was timed on, so throughput claims name their hardware."""
    try:
        import torch
    except ImportError:
        return ""
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""


def _sequence_length(scorer: LabelScorer) -> int | None:
    """The input cap a scorer truncates at; ``None`` when its SDK decides internally."""
    value = getattr(scorer, "sequence_length", SCORING_SEQUENCE_LENGTH)
    return None if value is None else int(value)


def _effective_batch_size(model: object, request: RunRequest) -> int:
    """The batch actually run: an adapter that works item by item says so."""
    return int(getattr(model, "effective_batch_size", request.batch_size))

"""Wiring one model to the benchmark and persisting what came out."""

import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from landuse_relevance_bench import __version__
from landuse_relevance_bench.adapters.benchmark_csv import load_benchmark
from landuse_relevance_bench.adapters.gpu_memory import free_gpu_memory_bytes
from landuse_relevance_bench.adapters.hashing import sha256_of_file, sha256_of_text
from landuse_relevance_bench.adapters.prompt_file import load_prompt
from landuse_relevance_bench.adapters.results_store import write_run
from landuse_relevance_bench.domain.engine import (
    BatchSizeReporting,
    Closable,
    DraftRevisionReporting,
    Generation,
    LabelScorer,
    MeasurementEnd,
    MeasurementStart,
    PeakVRAMReporting,
    RuntimeDtypeReporting,
    SequenceLengthReporting,
    TextGenerator,
)
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
from landuse_relevance_bench.domain.roster import (
    TRANSFORMERS,
    ModelSpec,
    quantization_of,
    spec_for,
)
from landuse_relevance_bench.domain.run_modes import check_modes, resolve_batch_size, run_id_for
from landuse_relevance_bench.domain.scorers import scorer_for, scorer_ids

DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_DTYPE = "bfloat16"
SCORING_SEQUENCE_LENGTH = 8192
logger = logging.getLogger(__name__)


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
    run_id: str = ""
    runtime: str = TRANSFORMERS
    vision: bool = False
    draft_model_id: str = ""
    draft_revision: str | None = None
    speculative: Mapping[str, Any] = field(default_factory=dict)
    continuous_batching: bool = False
    throughput_mode: bool = False
    close_generator: bool = True

    @property
    def name(self) -> str:
        return self.run_id or self.model_id

    @classmethod
    def for_model(cls, name: str, *, batch_size: int | None = None, **options: Any) -> "RunRequest":
        """The one factory the CLI uses: a scoring id gets a plain request, a roster id its spec.

        Scoring ids are not in the generative roster, so they are built from their options
        alone; the default batch size applies when none is given.
        """
        if name in scorer_ids():
            return cls(
                name,
                batch_size=DEFAULT_BATCH_SIZE if batch_size is None else batch_size,
                **options,
            )
        return cls.for_run(name, batch_size=batch_size, **options)

    @classmethod
    def for_run(
        cls,
        name: str,
        *,
        revision: str | None = None,
        batch_size: int | None = None,
        continuous_batching: bool = False,
        throughput_mode: bool = False,
        **common: Any,
    ) -> "RunRequest":
        """Build a request from the roster, preserving plain runs for unlisted models."""
        if continuous_batching and throughput_mode:
            raise ValueError("choose either continuous batching or SGLang throughput mode")
        try:
            spec = spec_for(name)
        except KeyError:
            logger.warning(
                "%s is not in the benchmark roster; running it on %s with no pinned revision "
                "and the default batch size",
                name,
                TRANSFORMERS,
            )
            spec = ModelSpec(name, 0, "unlisted")
        resolved_batch_size = resolve_batch_size(spec, batch_size, throughput_mode=throughput_mode)
        check_modes(
            spec,
            resolved_batch_size,
            continuous_batching=continuous_batching,
            throughput_mode=throughput_mode,
        )
        run_id = run_id_for(
            spec,
            resolved_batch_size,
            continuous_batching=continuous_batching,
            throughput_mode=throughput_mode,
        )
        return cls(
            model_id=spec.model_id,
            run_id=run_id,
            revision=revision or spec.revision,
            batch_size=resolved_batch_size,
            runtime=spec.runtime,
            vision=spec.vision,
            draft_model_id=spec.draft_model_id,
            draft_revision=spec.draft_revision,
            speculative=spec.speculative,
            continuous_batching=continuous_batching,
            throughput_mode=throughput_mode,
            **common,
        )


def _close_generator(generator: TextGenerator) -> None:
    if isinstance(generator, Closable):
        generator.close()


def _close_run_generator(request: RunRequest, generator: TextGenerator, *, failed: bool) -> None:
    """Close a generator this run owns, then log GPU memory."""
    if not request.close_generator:
        return
    phase = "after prediction error" if failed else "after prediction"
    try:
        _close_generator(generator)
    except Exception:
        # Logged, not raised: after a failed run this must not replace the original
        # error, and after a successful run the predictions are already in hand.
        logger.exception("%s: generator cleanup failed %s", request.name, phase)
    _log_gpu_memory(
        request.name,
        "after failed run cleanup" if failed else "after run cleanup",
        free_gpu_memory_bytes(),
    )


def _log_gpu_memory(name: str, stage: str, free_bytes: int | None) -> None:
    if free_bytes is not None:
        logger.info("%s: free GPU memory %s: %.1f MiB", name, stage, free_bytes / 2**20)


GeneratorProvider = Callable[[RunRequest], tuple[TextGenerator, str]]
ScorerProvider = Callable[[RunRequest], tuple[LabelScorer, str]]


class ProgressReporting:
    """Log batch completion without changing generator outputs."""

    def __init__(
        self, inner: TextGenerator, name: str, total_prompts: int, batch_size: int
    ) -> None:
        self._inner = inner
        self._name = name
        self._total_prompts = total_prompts
        self._batches = max(1, math.ceil(total_prompts / max(1, batch_size)))
        self._done = 0
        self._batch = 0

    def generate(self, prompts: Sequence[str]) -> Sequence[Generation | str]:
        self._batch += 1
        self._done += len(prompts)
        logger.info(
            "%s: batch %d/%d (%d/%d prompts)",
            self._name,
            self._batch,
            self._batches,
            self._done,
            self._total_prompts,
        )
        return self._inner.generate(prompts)


def execute(
    request: RunRequest,
    provide_generator: GeneratorProvider,
    *,
    source_commit: str = "",
) -> RunResult:
    """Run the benchmark for one model and write the result next to the others."""
    items = load_benchmark(request.benchmark_path, expected_language=request.language)
    template = load_prompt(request.prompt_path)
    logger.info("%s: loading model (%d prompts)", request.name, len(items))
    free_before = free_gpu_memory_bytes()
    _log_gpu_memory(request.name, "before model load", free_before)
    generator, revision = provide_generator(request)
    progress_generator = ProgressReporting(generator, request.name, len(items), request.batch_size)
    completed = False
    try:
        run = _timed(
            lambda: predict_all(items, template, progress_generator, batch_size=request.batch_size)
        )
        completed = True
    finally:
        _close_run_generator(request, generator, failed=not completed)
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
    revision: str | None,
    template: str,
    source_commit: str,
    **specific: Any,
) -> RunMetadata:
    """Fields every run records, plus the ones its method adds."""
    duration = run.duration
    return RunMetadata(
        model_id=request.model_id,
        run_id=request.run_id,
        language=request.language,
        model_revision=revision,
        prompt_sha256=sha256_of_text(template),
        benchmark_sha256=sha256_of_file(request.benchmark_path),
        batch_size=_effective_batch_size(model, request),
        seed=request.seed,
        dtype=_runtime_dtype(model, request),
        started_at=run.started_at,
        duration_seconds=round(duration, 3),
        source_commit=source_commit,
        runtime=request.runtime,
        draft_model_id=request.draft_model_id,
        draft_model_revision=_draft_revision(model, request),
        speculative=dict(request.speculative),
        package_version=__version__,
        generation_mode=(
            "transformers-continuous-batching"
            if request.continuous_batching
            else "sglang-throughput"
            if request.throughput_mode
            else "static-batched"
        ),
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
    if isinstance(scorer, MeasurementStart):
        scorer.begin_measurement()


def _end_measurement(scorer: LabelScorer) -> None:
    if isinstance(scorer, MeasurementEnd):
        scorer.end_measurement()


def _peak_vram_bytes(scorer: LabelScorer) -> int | None:
    if not isinstance(scorer, PeakVRAMReporting):
        return None
    value = scorer.peak_vram_bytes
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
    if not isinstance(scorer, SequenceLengthReporting):
        return SCORING_SEQUENCE_LENGTH
    value = scorer.sequence_length
    return None if value is None else int(value)


def _effective_batch_size(model: object, request: RunRequest) -> int:
    """The batch actually run: an adapter that works item by item says so."""
    if isinstance(model, BatchSizeReporting):
        return int(model.effective_batch_size)
    return int(request.batch_size)


def _runtime_dtype(model: object, request: RunRequest) -> str:
    """The dtype the runtime selected, else the one requested."""
    if isinstance(model, RuntimeDtypeReporting):
        return str(model.runtime_dtype)
    return str(request.dtype)


def _draft_revision(model: object, request: RunRequest) -> str:
    """The draft revision the generator ran with, else the one requested."""
    if isinstance(model, DraftRevisionReporting) and model.draft_revision:
        return model.draft_revision
    return request.draft_revision or ""

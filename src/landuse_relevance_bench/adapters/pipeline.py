"""Wiring one model to the benchmark and persisting what came out."""

import logging
import math
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from landuse_relevance_bench import __version__
from landuse_relevance_bench.adapters.benchmark_csv import load_benchmark
from landuse_relevance_bench.adapters.hashing import sha256_of_file, sha256_of_text
from landuse_relevance_bench.adapters.prompt_file import load_prompt
from landuse_relevance_bench.adapters.results_store import write_run
from landuse_relevance_bench.domain.engine import Generation, TextGenerator
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.orchestration import DEFAULT_BATCH_SIZE, predict_all
from landuse_relevance_bench.domain.records import RunMetadata, RunResult, outcomes_of
from landuse_relevance_bench.domain.roster import SGLANG, TRANSFORMERS, ModelSpec, spec_for

DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_DTYPE = "bfloat16"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RunRequest:
    """Everything one benchmark run needs, resolved from the CLI."""

    model_id: str
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

    @property
    def name(self) -> str:
        return self.run_id or self.model_id

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
        """A request for a rostered run, or a plain Transformers run of any Hub model.

        Explicit ``revision`` and ``batch_size`` override the roster's pins.
        """
        if continuous_batching and throughput_mode:
            raise ValueError("choose either continuous batching or SGLang throughput mode")
        try:
            spec = spec_for(name)
        except KeyError:
            logger.warning(
                "%s is not in the benchmark roster; running it on %s with no pinned "
                "revision and the default batch size",
                name,
                TRANSFORMERS,
            )
            spec = ModelSpec(name, 0, "unlisted")
        resolved_batch_size = _first_set(batch_size, spec.batch_size, DEFAULT_BATCH_SIZE)
        if continuous_batching and (spec.runtime != TRANSFORMERS or spec.vision):
            raise ValueError("continuous batching requires the Transformers runtime")
        if throughput_mode and spec.runtime != SGLANG:
            raise ValueError("throughput mode requires the SGLang runtime")
        if throughput_mode and resolved_batch_size <= 1:
            raise ValueError("throughput mode requires batch_size > 1")
        run_id = spec.run_id
        if continuous_batching:
            run_id = f"{spec.name}@continuous-b{resolved_batch_size}"
        elif throughput_mode:
            run_id = f"{spec.name}-throughput-b{resolved_batch_size}"
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


def _first_set(*values: int | None) -> int:
    return next(v for v in values if v is not None)


def _free_gpu_memory_bytes() -> int | None:
    """Return the least free NVIDIA GPU memory without importing the inference stack."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        probe = subprocess.run(  # noqa: S603 - executable is resolved; arguments are fixed.
            [
                executable,
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    readings = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    if not readings:
        return None
    try:
        return min(int(reading) for reading in readings) * 2**20
    except ValueError:
        logger.debug("Could not parse free NVIDIA GPU memory: %r", probe.stdout)
        return None


def _close_generator(generator: TextGenerator) -> None:
    close = getattr(generator, "close", None)
    if callable(close):
        close()


def _log_gpu_memory(name: str, stage: str, free_bytes: int | None) -> None:
    if free_bytes is not None:
        logger.info("%s: free GPU memory %s: %.1f MiB", name, stage, free_bytes / 2**20)


GeneratorProvider = Callable[[RunRequest], tuple[TextGenerator, str]]


class ProgressReporting:
    """Wraps a generator to log each batch as it is sent, so long runs show progress."""

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
    items = load_benchmark(request.benchmark_path)
    template = load_prompt(request.prompt_path)
    logger.info("%s: loading model (%d prompts)", request.name, len(items))
    free_before = _free_gpu_memory_bytes()
    _log_gpu_memory(request.name, "before model load", free_before)
    model_generator, revision = provide_generator(request)
    progress_generator = ProgressReporting(
        model_generator, request.name, len(items), request.batch_size
    )

    started_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    started = time.monotonic()
    try:
        predictions = predict_all(
            items, template, progress_generator, batch_size=request.batch_size
        )
        duration = time.monotonic() - started
    except BaseException:
        try:
            _close_generator(model_generator)
        except Exception:
            logger.exception("%s: generator cleanup failed after prediction error", request.name)
        _log_gpu_memory(request.name, "after failed run cleanup", _free_gpu_memory_bytes())
        raise
    else:
        try:
            _close_generator(model_generator)
        except Exception:
            logger.exception("%s: generator cleanup failed after prediction", request.name)
        _log_gpu_memory(request.name, "after run cleanup", _free_gpu_memory_bytes())

    result = RunResult(
        metadata=RunMetadata(
            model_id=request.model_id,
            model_revision=revision,
            prompt_sha256=sha256_of_text(template),
            benchmark_sha256=sha256_of_file(request.benchmark_path),
            max_new_tokens=request.max_new_tokens,
            batch_size=request.batch_size,
            seed=request.seed,
            decoding="greedy",
            dtype=request.dtype,
            started_at=started_at,
            duration_seconds=round(duration, 3),
            source_commit=source_commit,
            run_id=request.run_id,
            runtime=request.runtime,
            draft_model_id=request.draft_model_id,
            draft_model_revision=(
                getattr(model_generator, "draft_revision", "") or request.draft_revision or ""
            ),
            speculative=dict(request.speculative),
            package_version=__version__,
            generation_mode=(
                "transformers-continuous-batching"
                if request.continuous_batching
                else "sglang-throughput"
                if request.throughput_mode
                else "static-batched"
            ),
        ),
        predictions=predictions,
        metrics=evaluate(outcomes_of(predictions)),
    )
    write_run(result, request.output_dir)
    return result

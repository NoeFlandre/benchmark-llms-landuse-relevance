"""Wiring one model to the benchmark and persisting what came out."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from landuse_relevance_bench.adapters.benchmark_csv import load_benchmark
from landuse_relevance_bench.adapters.hashing import sha256_of_file, sha256_of_text
from landuse_relevance_bench.adapters.prompt_file import load_prompt
from landuse_relevance_bench.adapters.results_store import write_run
from landuse_relevance_bench.domain.engine import TextGenerator
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.orchestration import DEFAULT_BATCH_SIZE, predict_all
from landuse_relevance_bench.domain.records import RunMetadata, RunResult, outcomes_of

DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_DTYPE = "bfloat16"


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

    started_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    started = time.monotonic()
    predictions = predict_all(items, template, generator, batch_size=request.batch_size)
    duration = time.monotonic() - started

    result = RunResult(
        metadata=RunMetadata(
            model_id=request.model_id,
            language=request.language,
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
        ),
        predictions=predictions,
        metrics=evaluate(outcomes_of(predictions)),
    )
    write_run(result, request.output_dir)
    return result

"""Wiring one model to the benchmark and persisting what came out."""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from landuse_relevance_bench.adapters.benchmark_csv import load_benchmark
from landuse_relevance_bench.adapters.hashing import sha256_of_file, sha256_of_text
from landuse_relevance_bench.adapters.prompt_file import load_prompt
from landuse_relevance_bench.adapters.results_store import write_run
from landuse_relevance_bench.domain.engine import TextGenerator
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.orchestration import DEFAULT_BATCH_SIZE, predict_all
from landuse_relevance_bench.domain.records import RunMetadata, RunResult, outcomes_of
from landuse_relevance_bench.domain.roster import TRANSFORMERS, ModelSpec, spec_for

DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_DTYPE = "bfloat16"


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
        **common: Any,
    ) -> "RunRequest":
        """A request for a rostered run, or a plain Transformers run of any Hub model.

        Explicit ``revision`` and ``batch_size`` override the roster's pins.
        """
        try:
            spec = spec_for(name)
        except KeyError:
            spec = ModelSpec(name, 0, "")
        return cls(
            model_id=spec.model_id,
            run_id=spec.run_id,
            revision=revision or spec.revision,
            batch_size=_first_set(batch_size, spec.batch_size, DEFAULT_BATCH_SIZE),
            runtime=spec.runtime,
            vision=spec.vision,
            draft_model_id=spec.draft_model_id,
            draft_revision=spec.draft_revision,
            speculative=spec.speculative,
            **common,
        )


def _first_set(*values: int | None) -> int:
    return next(v for v in values if v is not None)


GeneratorProvider = Callable[[RunRequest], tuple[TextGenerator, str]]


def execute(
    request: RunRequest,
    provide_generator: GeneratorProvider,
    *,
    source_commit: str = "",
) -> RunResult:
    """Run the benchmark for one model and write the result next to the others."""
    items = load_benchmark(request.benchmark_path)
    template = load_prompt(request.prompt_path)
    generator, revision = provide_generator(request)

    started_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    started = time.monotonic()
    predictions = predict_all(items, template, generator, batch_size=request.batch_size)
    duration = time.monotonic() - started

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
            draft_model_revision=request.draft_revision or "",
            speculative=dict(request.speculative),
        ),
        predictions=predictions,
        metrics=evaluate(outcomes_of(predictions)),
    )
    write_run(result, request.output_dir)
    return result

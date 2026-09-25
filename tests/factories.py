"""Shared builders for test doubles and records."""

from collections.abc import Sequence
from dataclasses import replace

from landuse_relevance_bench.domain.engine import Generation
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import Prediction, RunMetadata, RunResult


def make_metadata(**overrides: object) -> RunMetadata:
    """A complete run metadata record; override any field by keyword."""
    return replace(
        RunMetadata(
            model_id="LiquidAI/LFM2.5-350M",
            model_revision="abc123",
            prompt_sha256="p" * 64,
            benchmark_sha256="b" * 64,
            max_new_tokens=8,
            batch_size=16,
            seed=0,
            decoding="greedy",
            dtype="bfloat16",
            started_at="2026-09-13T10:00:00Z",
            duration_seconds=1.0,
        ),
        **overrides,
    )


def make_result(
    model_id: str = "LiquidAI/LFM2.5-350M",
    predictions: tuple[Prediction, ...] | None = None,
    **metadata_overrides: object,
) -> RunResult:
    """A run whose metrics are computed from ``predictions`` (one correct YES by default)."""
    if predictions is None:
        predictions = (
            Prediction(item_id="0" * 16, expected=Label.YES, predicted=Label.YES, raw_output="yes"),
        )
    return RunResult(
        metadata=make_metadata(model_id=model_id, **metadata_overrides),
        predictions=predictions,
        metrics=evaluate([(p.expected, p.predicted) for p in predictions]),
    )


class ScriptedGenerator:
    """Returns queued outputs and records the prompts and batch shapes it saw."""

    def __init__(self, outputs: Sequence[str | Generation]) -> None:
        self._outputs: list[str | Generation] = list(outputs)
        self.prompts: list[str] = []
        self.batch_sizes: list[int] = []

    def generate(self, prompts: Sequence[str]) -> Sequence[str | Generation]:
        self.prompts.extend(prompts)
        self.batch_sizes.append(len(prompts))
        taken, self._outputs = self._outputs[: len(prompts)], self._outputs[len(prompts) :]
        return taken

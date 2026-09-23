"""Serialisable records describing one model's run over the benchmark."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import ClassificationMetrics, Outcome

GENERATION = "generation"
SCORING = "scoring"


@dataclass(frozen=True, slots=True)
class Prediction:
    """What one model produced for one benchmark item."""

    item_id: str
    expected: Label
    predicted: Label | None
    raw_output: str
    truncated: bool = False

    @property
    def outcome(self) -> Outcome:
        return (self.expected, self.predicted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "expected": self.expected.value,
            "predicted": None if self.predicted is None else self.predicted.value,
            "raw_output": self.raw_output,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Prediction":
        predicted = payload["predicted"]
        return cls(
            item_id=payload["item_id"],
            expected=Label(payload["expected"]),
            predicted=None if predicted is None else Label(predicted),
            raw_output=payload["raw_output"],
            truncated=payload.get("truncated", False),
        )


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Everything needed to reproduce or audit a run."""

    model_id: str
    language: str
    model_revision: str
    prompt_sha256: str
    benchmark_sha256: str
    max_new_tokens: int
    batch_size: int
    seed: int
    decoding: str
    dtype: str
    started_at: str
    duration_seconds: float
    source_commit: str = ""
    # How the verdict was obtained. Generative models are prompted and their text is
    # parsed; scoring models emit a native score per label and never produce text, so
    # they can neither leave a verdict unparsed nor run out of budget. Defaulted, so
    # every result written before scoring models existed still reads back.
    inference: str = GENERATION
    # The rule that turned a scoring model's native scores into a verdict; empty for
    # generative runs, whose rule is the parser.
    decision_rule: str = ""
    # Optional performance telemetry. Legacy result files did not carry these keys,
    # so defaults keep them readable while GPU runs can report their measurements.
    throughput_items_per_second: float | None = None
    peak_vram_bytes: int | None = None
    sequence_length: int | None = None
    # GGUF quant label for llama.cpp runs; empty for full-precision checkpoints, whose
    # precision is ``dtype``. Kept separate so a quant is never mistaken for a dtype.
    quantization: str = ""
    # Accelerator the timing was measured on; omitted when unknown, like quantization.
    device_name: str = ""

    @property
    def is_generative(self) -> bool:
        return self.inference == GENERATION

    def __post_init__(self) -> None:
        if not isinstance(self.language, str) or not self.language.strip():
            raise ValueError(
                "run metadata requires a non-empty language; legacy records are archive-only"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Omitted when empty, so every full-precision run keeps its exact published bytes.
        for optional in ("quantization", "device_name"):
            if not payload[optional]:
                del payload[optional]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunMetadata":
        if "language" not in payload or not payload["language"]:
            raise ValueError("legacy/archive-only run metadata is missing required language")
        try:
            return cls(**dict(payload))
        except TypeError as exc:
            raise ValueError(f"invalid run metadata: {exc}") from exc


@dataclass(frozen=True, slots=True)
class RunResult:
    """Predictions plus the scores computed from exactly those predictions."""

    metadata: RunMetadata
    predictions: tuple[Prediction, ...]
    metrics: ClassificationMetrics

    def __post_init__(self) -> None:
        if self.metrics.n_items != len(self.predictions):
            raise ValueError(
                f"metrics cover {self.metrics.n_items} items but the run holds "
                f"{len(self.predictions)} predictions"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "metrics": self.metrics.to_dict(),
            "predictions": [p.to_dict() for p in self.predictions],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunResult":
        return cls(
            metadata=RunMetadata.from_dict(payload["metadata"]),
            predictions=tuple(Prediction.from_dict(p) for p in payload["predictions"]),
            metrics=ClassificationMetrics.from_dict(payload["metrics"]),
        )


def outcomes_of(predictions: Sequence[Prediction]) -> list[Outcome]:
    return [p.outcome for p in predictions]

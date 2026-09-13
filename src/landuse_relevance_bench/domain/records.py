"""Serialisable records describing one model's run over the benchmark."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import ClassificationMetrics, Outcome


@dataclass(frozen=True, slots=True)
class Prediction:
    """What one model produced for one benchmark item."""

    item_id: str
    expected: Label
    predicted: Label | None
    raw_output: str

    @property
    def outcome(self) -> Outcome:
        return (self.expected, self.predicted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "expected": self.expected.value,
            "predicted": None if self.predicted is None else self.predicted.value,
            "raw_output": self.raw_output,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Prediction":
        predicted = payload["predicted"]
        return cls(
            item_id=payload["item_id"],
            expected=Label(payload["expected"]),
            predicted=None if predicted is None else Label(predicted),
            raw_output=payload["raw_output"],
        )


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Everything needed to reproduce or audit a run."""

    model_id: str
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunMetadata":
        return cls(**payload)


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
    def from_dict(cls, payload: dict[str, Any]) -> "RunResult":
        return cls(
            metadata=RunMetadata.from_dict(payload["metadata"]),
            predictions=tuple(Prediction.from_dict(p) for p in payload["predictions"]),
            metrics=ClassificationMetrics.from_dict(payload["metrics"]),
        )


def outcomes_of(predictions: Sequence[Prediction]) -> list[Outcome]:
    return [p.outcome for p in predictions]

"""Serialisable records describing one model's run over the benchmark."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import ClassificationMetrics, Outcome, evaluate
from landuse_relevance_bench.domain.parsing import ParseMode
from landuse_relevance_bench.domain.roster import TRANSFORMERS
from landuse_relevance_bench.domain.speed import SpeedMetrics, summarise_speed

#: Optional per-generation measurements; absent from older result files.
_COUNTERS = (
    "latency_seconds",
    "generated_tokens",
    "verify_steps",
    "accepted_drafts",
    "proposed_drafts",
)


@dataclass(frozen=True, slots=True)
class Prediction:
    """What one model produced for one benchmark item."""

    item_id: str
    expected: Label
    predicted: Label | None
    raw_output: str
    truncated: bool = False
    parse_mode: ParseMode | None = None
    #: Wall time of the generator call that produced this answer (its whole batch).
    latency_seconds: float | None = None
    generated_tokens: int | None = None
    verify_steps: int | None = None
    accepted_drafts: int | None = None
    proposed_drafts: int | None = None

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
            "parse_mode": self.parse_mode,
            **{name: getattr(self, name) for name in _COUNTERS},
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
            parse_mode=payload.get("parse_mode"),
            **{name: payload.get(name) for name in _COUNTERS},
        )


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Everything needed to reproduce or audit a run."""

    model_id: str
    model_revision: str | None
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
    #: The roster name of the run; differs from ``model_id`` when one model is run
    #: several ways (e.g. with and without a speculative draft).
    run_id: str = ""
    runtime: str = TRANSFORMERS
    draft_model_id: str = ""
    draft_model_revision: str | None = ""
    speculative: Mapping[str, Any] = field(default_factory=dict)
    package_version: str = ""
    generation_mode: str = "static-batched"

    @property
    def name(self) -> str:
        return self.run_id or self.model_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["speculative"] = dict(self.speculative)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunMetadata":
        return cls(**dict(payload))


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
        identifiers = [prediction.item_id for prediction in self.predictions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("run contains a duplicate item id")
        if self.metrics != evaluate(outcomes_of(self.predictions)):
            raise ValueError("metrics do not match predictions")

    @property
    def speed(self) -> SpeedMetrics:
        """Derived from the stored predictions, so older result files still yield one."""
        return summarise_speed(self.predictions, self.metadata.duration_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "metrics": self.metrics.to_dict(),
            "speed": self.speed.to_dict(),
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

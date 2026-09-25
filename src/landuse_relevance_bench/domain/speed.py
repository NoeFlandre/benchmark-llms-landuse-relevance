"""How fast a run was, derived only from what the run recorded.

Every figure is recomputed from the per-prediction timings and token counts, so a
result file carries its own evidence and an older file without them still yields the
wall-time figures it can support. See ADR-0006.
"""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class Timed(Protocol):
    """The slice of a prediction that speed is computed from."""

    @property
    def latency_seconds(self) -> float | None: ...
    @property
    def generated_tokens(self) -> int | None: ...
    @property
    def verify_steps(self) -> int | None: ...
    @property
    def accepted_drafts(self) -> int | None: ...
    @property
    def proposed_drafts(self) -> int | None: ...


@dataclass(frozen=True, slots=True)
class SpeedMetrics:
    """Throughput and latency of one run; ``None`` where the run did not record it."""

    wall_seconds: float
    n_items: int
    sentences_per_second: float | None
    latency_mean_seconds: float | None
    latency_p50_seconds: float | None
    latency_p95_seconds: float | None
    generated_tokens: int | None
    output_tokens_per_second: float | None
    mean_accept_length: float | None
    draft_accept_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear interpolation between closest ranks (NumPy's default method)."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _per_second(amount: float, seconds: float) -> float | None:
    return amount / seconds if seconds > 0 else None


def _complete(values: Sequence[T | None]) -> list[T] | None:
    """All values, or ``None`` if any prediction lacks one (a partial sum misleads)."""
    present = [v for v in values if v is not None]
    return present if values and len(present) == len(values) else None


def _ratio(numerators: list[int] | None, denominators: list[int] | None) -> float | None:
    """Pooled ratio over the run: long generations weigh by their length."""
    if numerators is None or denominators is None or sum(denominators) == 0:
        return None
    return sum(numerators) / sum(denominators)


def _latency_stats(latencies: list[float] | None) -> tuple[float | None, ...]:
    """Mean, median and 95th percentile, or all ``None`` without complete timings."""
    if latencies is None:
        return (None, None, None)
    mean = sum(latencies) / len(latencies)
    return (mean, percentile(latencies, 0.5), percentile(latencies, 0.95))


def _total(values: list[int] | None) -> int | None:
    return None if values is None else sum(values)


def _rate(amount: int | None, seconds: float) -> float | None:
    return None if amount is None else _per_second(amount, seconds)


def summarise_speed(predictions: Sequence[Timed], wall_seconds: float) -> SpeedMetrics:
    tokens = _complete([p.generated_tokens for p in predictions])
    mean, p50, p95 = _latency_stats(_complete([p.latency_seconds for p in predictions]))
    total_tokens = _total(tokens)
    return SpeedMetrics(
        wall_seconds=wall_seconds,
        n_items=len(predictions),
        sentences_per_second=_per_second(len(predictions), wall_seconds),
        latency_mean_seconds=mean,
        latency_p50_seconds=p50,
        latency_p95_seconds=p95,
        generated_tokens=total_tokens,
        output_tokens_per_second=_rate(total_tokens, wall_seconds),
        mean_accept_length=_ratio(tokens, _complete([p.verify_steps for p in predictions])),
        draft_accept_rate=_ratio(
            _complete([p.accepted_drafts for p in predictions]),
            _complete([p.proposed_drafts for p in predictions]),
        ),
    )

"""Pure planning for deterministic model-language execution shards."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Literal

Pair = tuple[str, str]
Status = Literal["complete", "pending"]


@dataclass(frozen=True, slots=True)
class PairStatus:
    """Completion state for one planned model-language pair."""

    model_id: str
    language: str
    status: Status


def model_language_pairs(model_ids: Sequence[str], languages: Sequence[str]) -> tuple[Pair, ...]:
    """Return the canonical sorted Cartesian product without duplicate pairs."""
    models = _unique_values(model_ids, "model")
    language_codes = _unique_values(languages, "language")
    return tuple((model_id, language) for model_id in models for language in language_codes)


def shard_pairs(pairs: Sequence[Pair], shard_index: int, shard_count: int) -> tuple[Pair, ...]:
    """Select one stable modulo partition of a complete sorted pair list."""
    _validate_shard(shard_index, shard_count)
    canonical = _canonical_pairs(pairs)
    return tuple(pair for index, pair in enumerate(canonical) if index % shard_count == shard_index)


def pair_statuses(pairs: Sequence[Pair], completed: Collection[Pair]) -> tuple[PairStatus, ...]:
    """Classify planned pairs from a caller-provided set of completed pairs."""
    canonical = _canonical_pairs(pairs)
    return tuple(
        PairStatus(
            model_id, language, "complete" if (model_id, language) in completed else "pending"
        )
        for model_id, language in canonical
    )


def _canonical_pairs(pairs: Sequence[Pair]) -> tuple[Pair, ...]:
    canonical = tuple(sorted(pairs))
    if len(set(canonical)) != len(canonical):
        raise ValueError("shard pair list contains duplicate pairs")
    for model_id, language in canonical:
        if not model_id.strip() or not language.strip():
            raise ValueError("shard pairs require non-empty model and language values")
    return canonical


def _unique_values(values: Sequence[str], kind: str) -> tuple[str, ...]:
    normalized = tuple(sorted(set(values)))
    if any(not value.strip() for value in normalized):
        raise ValueError(f"{kind} values cannot be empty")
    return normalized


def _validate_shard(shard_index: int, shard_count: int) -> None:
    if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"invalid shard index/count: index={shard_index}, count={shard_count}")

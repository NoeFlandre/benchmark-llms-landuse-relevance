"""Integrity checks that a dataset card must pass before it describes any run."""

from collections.abc import Sequence
from typing import Any, NamedTuple

from landuse_relevance_bench.adapters.hashing import sha256_of_text
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import RunResult, outcomes_of
from landuse_relevance_bench.domain.selection import collection_comparability_errors


class ValidatedRuns(NamedTuple):
    """The runs a card may describe, split into generative and scoring runs."""

    generative: list[RunResult]
    scoring: list[RunResult]


def uniform(results: Sequence[RunResult], name: str, value_of: Any) -> Any:
    """Return the single value every run agrees on, or refuse to describe the sweep."""
    values = {value_of(result) for result in results}
    if len(values) != 1:
        raise ValueError(f"runs disagree on {name}: {sorted(values)}")
    return values.pop()


def validate_for_card(results: Sequence[RunResult], prompt_text: str) -> ValidatedRuns:
    """Refuse results a card must not describe; return the generative and scoring runs."""
    if not results:
        raise ValueError("cannot build a card from an empty set of results")
    incompatibilities = collection_comparability_errors(results)
    if incompatibilities:
        raise ValueError("; ".join(incompatibilities))
    for result in results:
        derived = evaluate(outcomes_of(result.predictions))
        if derived != result.metrics:
            raise ValueError(f"{result.metadata.model_id} metrics do not match predictions")
    generative = [r for r in results if r.metadata.is_generative]
    if not generative:
        raise ValueError("cannot build a card without any generative run")
    prompt_sha256 = uniform(generative, "prompt digest", lambda r: r.metadata.prompt_sha256)
    if sha256_of_text(prompt_text) != prompt_sha256:
        raise ValueError("prompt text does not match the digest recorded in the runs")
    return ValidatedRuns(generative, [r for r in results if not r.metadata.is_generative])

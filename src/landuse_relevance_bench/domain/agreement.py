"""The lossless check for speculative decoding.

Under greedy decoding a draft model only proposes tokens the target then verifies,
so a speculative run must reproduce its plain baseline exactly. A disagreement means
the two runs were not the same computation. See ADR-0007.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from landuse_relevance_bench.domain.records import Prediction, RunResult


@dataclass(frozen=True, slots=True)
class Agreement:
    """How a speculative run compares with a plain run of the same target."""

    speculative_run: str
    baseline_run: str
    same_runtime: bool
    n_compared: int
    n_speculative: int
    n_baseline: int
    verdicts_differ: int
    texts_differ: int
    complete_coverage: bool

    @property
    def lossless(self) -> bool:
        return self.complete_coverage and self.verdicts_differ == 0 and self.texts_differ == 0


def _compare(speculative: RunResult, baseline: RunResult) -> Agreement:
    base, pairs = _matching_predictions(speculative.predictions, baseline.predictions)
    return Agreement(
        speculative_run=speculative.metadata.name,
        baseline_run=baseline.metadata.name,
        same_runtime=speculative.metadata.runtime == baseline.metadata.runtime,
        n_compared=len(pairs),
        n_speculative=len(speculative.predictions),
        n_baseline=len(baseline.predictions),
        verdicts_differ=_verdict_differences(pairs),
        texts_differ=sum(s.raw_output != b.raw_output for s, b in pairs),
        complete_coverage=_complete_coverage(speculative.predictions, pairs, base),
    )


def _matching_predictions(
    speculative: Sequence[Prediction], baseline: Sequence[Prediction]
) -> tuple[dict[str, Prediction], list[tuple[Prediction, Prediction]]]:
    baseline_by_id = {prediction.item_id: prediction for prediction in baseline}
    pairs = [
        (prediction, baseline_by_id[prediction.item_id])
        for prediction in speculative
        if prediction.item_id in baseline_by_id
    ]
    return baseline_by_id, pairs


def _verdict_differences(pairs: Sequence[tuple[Prediction, Prediction]]) -> int:
    return sum(
        speculative.predicted != baseline.predicted or speculative.expected != baseline.expected
        for speculative, baseline in pairs
    )


def _complete_coverage(
    speculative: Sequence[Prediction],
    pairs: Sequence[tuple[Prediction, Prediction]],
    baseline_by_id: dict[str, Prediction],
) -> bool:
    return len(pairs) == len(speculative) == len(baseline_by_id) and {
        prediction.item_id for prediction in speculative
    } == set(baseline_by_id)


def _is_baseline_for(candidate: RunResult, speculative: RunResult) -> bool:
    ours, theirs = speculative.metadata, candidate.metadata
    return (
        not theirs.draft_model_id
        and theirs.model_id == ours.model_id
        and theirs.max_new_tokens == ours.max_new_tokens
        and theirs.prompt_sha256 == ours.prompt_sha256
        and theirs.benchmark_sha256 == ours.benchmark_sha256
        and theirs.dtype == ours.dtype
        and theirs.model_revision == ours.model_revision
        and theirs.decoding == ours.decoding
    )


def speculative_agreements(results: Sequence[RunResult]) -> list[Agreement]:
    """Every (speculative run, plain baseline) pair among ``results``."""
    return [
        _compare(speculative, candidate)
        for speculative in results
        if speculative.metadata.draft_model_id
        for candidate in results
        if _is_baseline_for(candidate, speculative)
    ]

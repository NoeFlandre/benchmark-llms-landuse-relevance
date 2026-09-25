"""The lossless check for speculative decoding.

Under greedy decoding a draft model only proposes tokens the target then verifies,
so a speculative run must reproduce its plain baseline exactly. A disagreement means
the two runs were not the same computation. See ADR-0007.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from landuse_relevance_bench.domain.records import RunResult


@dataclass(frozen=True, slots=True)
class Agreement:
    """How a speculative run compares with a plain run of the same target."""

    speculative_run: str
    baseline_run: str
    same_runtime: bool
    n_compared: int
    verdicts_differ: int
    texts_differ: int

    @property
    def lossless(self) -> bool:
        return self.verdicts_differ == 0 and self.texts_differ == 0


def _compare(speculative: RunResult, baseline: RunResult) -> Agreement:
    base = {p.item_id: p for p in baseline.predictions}
    pairs = [(p, base[p.item_id]) for p in speculative.predictions if p.item_id in base]
    return Agreement(
        speculative_run=speculative.metadata.name,
        baseline_run=baseline.metadata.name,
        same_runtime=speculative.metadata.runtime == baseline.metadata.runtime,
        n_compared=len(pairs),
        verdicts_differ=sum(s.predicted != b.predicted for s, b in pairs),
        texts_differ=sum(s.raw_output != b.raw_output for s, b in pairs),
    )


def _is_baseline_for(candidate: RunResult, speculative: RunResult) -> bool:
    ours, theirs = speculative.metadata, candidate.metadata
    return (
        not theirs.draft_model_id
        and theirs.model_id == ours.model_id
        and theirs.max_new_tokens == ours.max_new_tokens
        and theirs.prompt_sha256 == ours.prompt_sha256
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

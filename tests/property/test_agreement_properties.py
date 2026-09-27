"""Speculative agreement invariants over generated prediction sets."""

from dataclasses import replace

from hypothesis import given
from strategies import run_results

from landuse_relevance_bench.domain.agreement import speculative_agreements
from landuse_relevance_bench.domain.records import RunResult


@given(baseline=run_results())
def test_an_identical_speculative_copy_is_lossless_and_input_order_does_not_matter(
    baseline: RunResult,
) -> None:
    speculative = replace(
        baseline,
        metadata=replace(
            baseline.metadata,
            run_id=f"{baseline.metadata.name}+draft",
            draft_model_id="draft/model",
        ),
    )
    forward = speculative_agreements([baseline, speculative])
    reverse = speculative_agreements([speculative, baseline])
    assert len(forward) == len(reverse) == 1
    assert forward == reverse
    assert forward[0].lossless

from dataclasses import replace

import pytest

from factories import make_metadata
from landuse_relevance_bench.domain.agreement import speculative_agreements
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import Prediction, RunResult

META = make_metadata(
    model_id="t/model",
    model_revision="r",
    prompt_sha256="p",
    benchmark_sha256="b",
    max_new_tokens=64,
    batch_size=1,
    started_at="2026-09-25T00:00:00Z",
)


def _run(outputs: list[str], **metadata: object) -> RunResult:
    predictions = tuple(
        Prediction(
            item_id=str(i),
            expected=Label.YES,
            predicted=Label(text.split()[-1]),
            raw_output=text,
        )
        for i, text in enumerate(outputs)
    )
    return RunResult(
        metadata=replace(META, **metadata),
        predictions=predictions,
        metrics=evaluate([p.outcome for p in predictions]),
    )


PLAIN = _run(["yes", "no"], run_id="t/model@sglang", runtime="sglang")


def _drafted(outputs: list[str], **metadata: object) -> RunResult:
    return _run(
        outputs, run_id="t/model+DSpark", runtime="sglang", draft_model_id="t/draft", **metadata
    )


def test_a_speculative_run_that_reproduces_its_baseline_is_lossless() -> None:
    (agreement,) = speculative_agreements([PLAIN, _drafted(["yes", "no"])])
    assert agreement.lossless
    assert agreement.same_runtime
    assert (agreement.speculative_run, agreement.baseline_run) == (
        "t/model+DSpark",
        "t/model@sglang",
    )
    assert agreement.n_compared == 2


def test_a_changed_verdict_or_generation_is_counted() -> None:
    (agreement,) = speculative_agreements([PLAIN, _drafted(["so yes", "yes"])])
    assert not agreement.lossless
    assert (agreement.verdicts_differ, agreement.texts_differ) == (1, 2)


def test_every_plain_run_of_the_target_is_a_baseline_and_runtimes_are_told_apart() -> None:
    transformers_run = _run(["yes", "no"])
    agreements = speculative_agreements([transformers_run, PLAIN, _drafted(["yes", "no"])])
    assert sorted(a.same_runtime for a in agreements) == [False, True]


def test_runs_with_different_budgets_are_not_compared() -> None:
    assert speculative_agreements([PLAIN, _drafted(["yes", "no"], max_new_tokens=8)]) == []


def test_partial_or_disjoint_coverage_cannot_be_lossless() -> None:
    baseline = _run(["yes", "no"], run_id="t/model@sglang", runtime="sglang")
    drafted = _run(
        ["yes"],
        run_id="t/model+DSpark",
        runtime="sglang",
        draft_model_id="t/draft",
    )
    (agreement,) = speculative_agreements([baseline, drafted])
    assert not agreement.complete_coverage
    assert not agreement.lossless
    assert (agreement.n_speculative, agreement.n_baseline, agreement.n_compared) == (1, 2, 1)


@pytest.mark.parametrize(
    "metadata",
    [
        {"benchmark_sha256": "other"},
        {"dtype": "float16"},
        {"model_revision": "new-revision"},
    ],
)
def test_baseline_requires_matching_benchmark_dtype_and_target_revision(metadata) -> None:
    assert speculative_agreements([PLAIN, _drafted(["yes", "no"], **metadata)]) == []

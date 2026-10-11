"""The integrity checks a dataset card must pass, pinned before they move into the domain."""

from dataclasses import replace

import pytest

from factories import make_result
from landuse_relevance_bench.adapters.hashing import sha256_of_text
from landuse_relevance_bench.adapters.publishing import card_validation
from landuse_relevance_bench.adapters.publishing.card_validation import (
    ValidatedRuns,
    uniform,
    validate_for_card,
)
from landuse_relevance_bench.domain.records import SCORING, RunResult

PROMPT = "Classify the sentence.\n\nTARGET SENTENCE: {}\n"
PROMPT_SHA256 = sha256_of_text(PROMPT)


def _generative(model_id: str = "a/model", **overrides: object) -> RunResult:
    settings = {"prompt_sha256": PROMPT_SHA256, **overrides}
    return make_result(model_id=model_id, **settings)


def _scoring(model_id: str = "b/scorer") -> RunResult:
    return make_result(model_id=model_id, inference=SCORING)


def test_validate_splits_the_generative_runs_from_the_scoring_runs() -> None:
    generative = _generative()
    scoring = _scoring()

    assert validate_for_card([generative, scoring], PROMPT) == ValidatedRuns(
        [generative], [scoring]
    )


def test_validate_refuses_an_empty_set_of_results() -> None:
    with pytest.raises(ValueError, match=r"^cannot build a card from an empty set of results$"):
        validate_for_card([], PROMPT)


def test_validate_refuses_runs_whose_settings_are_not_comparable() -> None:
    with pytest.raises(ValueError, match=r"^runs are not comparable: .*max_new_tokens"):
        validate_for_card([_generative("a/one"), _generative("b/two", max_new_tokens=32)], PROMPT)


def test_validate_refuses_generative_runs_with_different_prompt_digests_as_not_comparable() -> None:
    other = _generative("b/two", prompt_sha256="0" * 64)

    with pytest.raises(ValueError, match=r"^runs are not comparable: .*prompt_sha256"):
        validate_for_card([_generative(), other], PROMPT)


def test_validate_refuses_metrics_that_are_not_derived_from_the_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _generative()
    monkeypatch.setattr(
        card_validation,
        "evaluate",
        lambda _outcomes: replace(result.metrics, accuracy=0.0),
    )

    with pytest.raises(ValueError, match=r"^a/model metrics do not match predictions$"):
        validate_for_card([result], PROMPT)


def test_validate_refuses_a_set_with_no_generative_run() -> None:
    with pytest.raises(ValueError, match=r"^cannot build a card without any generative run$"):
        validate_for_card([_scoring()], PROMPT)


def test_validate_refuses_prompt_text_the_runs_did_not_use() -> None:
    with pytest.raises(
        ValueError, match=r"^prompt text does not match the digest recorded in the runs$"
    ):
        validate_for_card([_generative()], "something else {}")


def test_uniform_returns_the_value_every_run_agrees_on() -> None:
    results = [_generative("a/one"), _generative("b/two")]

    assert uniform(results, "seed", lambda result: result.metadata.seed) == 0


def test_uniform_names_the_distinct_values_when_runs_disagree() -> None:
    results = [_generative("a/one"), _generative("b/two", seed=1)]

    with pytest.raises(ValueError, match=r"^runs disagree on seed: \[0, 1\]$"):
        uniform(results, "seed", lambda result: result.metadata.seed)


def test_uniform_refuses_an_empty_set_of_runs() -> None:
    with pytest.raises(ValueError, match=r"^runs disagree on seed: \[\]$"):
        uniform([], "seed", lambda result: result.metadata.seed)


def test_validate_joins_every_incompatibility_into_one_refusal() -> None:
    other = _generative("b/two", max_new_tokens=32, benchmark_sha256="c" * 64)

    with pytest.raises(ValueError) as refusal:
        validate_for_card([_generative("a/one"), other], PROMPT)

    assert str(refusal.value) == (
        "runs are not comparable: benchmark_sha256 differs for language en; "
        "runs are not comparable: b/two.max_new_tokens=32 (expected 8 from a/one)"
    )

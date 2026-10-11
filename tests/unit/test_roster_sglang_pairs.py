"""The SGLang baseline and DSpark pair that the roster derives from one generative model."""

import re

import pytest

from landuse_relevance_bench.domain.roster import SGLANG, TRANSFORMERS, ModelSpec, sglang_pair


def _target() -> ModelSpec:
    return ModelSpec(
        "org/target",
        1_000,
        "target note",
        revision="rev-1",
        batch_size=8,
        speculative={"speculative_k": 3, "k": 1},
    )


def test_the_baseline_is_the_target_on_sglang_without_a_draft() -> None:
    baseline, _ = sglang_pair(_target(), "org/draft", "draft-rev", 42, {"speculative_k": 3, "k": 1})

    assert baseline.runtime == SGLANG
    assert baseline.batch_size == 1
    assert baseline.run_id == "org/target@sglang"
    assert baseline.note == "SGLang, no draft: the like-for-like baseline for DSpark."
    assert baseline.draft_model_id == ""
    assert baseline.draft_revision is None
    assert baseline.speculative == {"k": 1}


def test_the_drafted_run_adds_the_draft_and_keeps_every_setting() -> None:
    settings = {"speculative_k": 3, "k": 1}

    _, drafted = sglang_pair(_target(), "org/draft", "draft-rev", 42, settings)

    assert drafted.runtime == SGLANG
    assert drafted.batch_size == 1
    assert drafted.run_id == "org/target+DSpark"
    assert (
        drafted.note == "SGLang with the DSpark speculative draft; lossless under greedy decoding."
    )
    assert drafted.draft_model_id == "org/draft"
    assert drafted.draft_revision == "draft-rev"
    assert drafted.draft_parameters == 42
    assert drafted.speculative == settings


def test_the_drafted_run_is_a_copy_of_the_settings_it_was_given() -> None:
    settings = {"speculative_k": 3}

    _, drafted = sglang_pair(_target(), "org/draft", "draft-rev", 42, settings)
    settings["speculative_k"] = 9

    assert drafted.speculative == {"speculative_k": 3}


def test_a_speculative_draft_needs_the_sglang_runtime() -> None:
    with pytest.raises(ValueError, match=r"^a speculative draft needs the sglang runtime$"):
        ModelSpec("org/m", 1, "n", draft_model_id="org/draft", runtime=TRANSFORMERS)


def test_an_unknown_runtime_names_the_runtimes_it_accepts() -> None:
    with pytest.raises(ValueError, match=re.escape("unknown runtime 'vllm'")):
        ModelSpec("org/m", 1, "n", runtime="vllm")

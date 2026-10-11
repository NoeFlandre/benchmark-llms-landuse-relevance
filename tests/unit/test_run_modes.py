"""The run-mode rules: batch size, mode checks and run ids, pinned branch by branch."""

import pytest

from landuse_relevance_bench.domain.orchestration import DEFAULT_BATCH_SIZE
from landuse_relevance_bench.domain.roster import ROSTER, SGLANG, TRANSFORMERS, ModelSpec
from landuse_relevance_bench.domain.run_modes import (
    check_modes,
    continuous_batching_applies,
    resolve_batch_size,
    resolve_mode_options,
    run_id_for,
    runtime_for,
    spec_or_none,
    throughput_applies,
    throughput_batch_size,
)


def _spec(**overrides: object) -> ModelSpec:
    return ModelSpec("org/model", 1, "note", **overrides)


def test_throughput_mode_without_a_batch_size_uses_the_default_not_the_roster() -> None:
    spec = _spec(runtime=SGLANG, batch_size=4)

    assert resolve_batch_size(spec, None, throughput_mode=True) == DEFAULT_BATCH_SIZE


def test_throughput_mode_keeps_an_explicit_batch_size() -> None:
    spec = _spec(runtime=SGLANG, batch_size=4)

    assert resolve_batch_size(spec, 8, throughput_mode=True) == 8


def test_without_throughput_mode_the_roster_batch_size_applies_when_none_is_given() -> None:
    assert resolve_batch_size(_spec(batch_size=4), None, throughput_mode=False) == 4


def test_without_a_roster_batch_size_the_default_applies() -> None:
    assert resolve_batch_size(_spec(), None, throughput_mode=False) == DEFAULT_BATCH_SIZE


def test_an_explicit_batch_size_beats_the_roster_outside_throughput_mode() -> None:
    assert resolve_batch_size(_spec(batch_size=4), 2, throughput_mode=False) == 2


def test_continuous_batching_needs_the_transformers_runtime() -> None:
    with pytest.raises(
        ValueError, match=r"^continuous batching requires the Transformers runtime$"
    ):
        check_modes(
            _spec(runtime=SGLANG),
            8,
            continuous_batching=True,
            throughput_mode=False,
        )


def test_continuous_batching_refuses_a_vision_model() -> None:
    with pytest.raises(
        ValueError, match=r"^continuous batching requires the Transformers runtime$"
    ):
        check_modes(
            _spec(runtime=TRANSFORMERS, vision=True),
            8,
            continuous_batching=True,
            throughput_mode=False,
        )


def test_continuous_batching_accepts_a_plain_transformers_model() -> None:
    check_modes(_spec(runtime=TRANSFORMERS), 8, continuous_batching=True, throughput_mode=False)


def test_throughput_mode_needs_the_sglang_runtime() -> None:
    with pytest.raises(ValueError, match=r"^throughput mode requires the SGLang runtime$"):
        check_modes(
            _spec(runtime=TRANSFORMERS),
            8,
            continuous_batching=False,
            throughput_mode=True,
        )


def test_throughput_mode_needs_a_batch_larger_than_one() -> None:
    with pytest.raises(ValueError, match=r"^throughput mode requires batch_size > 1$"):
        check_modes(_spec(runtime=SGLANG), 1, continuous_batching=False, throughput_mode=True)


def test_throughput_mode_accepts_a_batch_of_two_on_sglang() -> None:
    check_modes(_spec(runtime=SGLANG), 2, continuous_batching=False, throughput_mode=True)


def test_plain_runs_pass_every_check_whatever_the_runtime_and_batch() -> None:
    check_modes(_spec(runtime=SGLANG), 1, continuous_batching=False, throughput_mode=False)
    check_modes(
        _spec(runtime=TRANSFORMERS, vision=True),
        1,
        continuous_batching=False,
        throughput_mode=False,
    )


def test_a_plain_run_keeps_the_roster_run_id() -> None:
    spec = _spec(run_id="org/model-variant")

    assert run_id_for(spec, 8, continuous_batching=False, throughput_mode=False) == (
        "org/model-variant"
    )


def test_a_continuous_run_is_named_after_the_model_and_its_batch_size() -> None:
    assert run_id_for(_spec(), 8, continuous_batching=True, throughput_mode=False) == (
        "org/model@continuous-b8"
    )


def test_a_continuous_run_uses_the_roster_run_id_as_its_name() -> None:
    spec = _spec(run_id="org/model-variant")

    assert run_id_for(spec, 8, continuous_batching=True, throughput_mode=False) == (
        "org/model-variant@continuous-b8"
    )


def test_a_throughput_run_is_named_after_the_model_and_its_batch_size() -> None:
    assert run_id_for(_spec(), 16, continuous_batching=False, throughput_mode=True) == (
        "org/model-throughput-b16"
    )


def test_continuous_naming_wins_when_both_modes_are_given() -> None:
    assert run_id_for(_spec(), 8, continuous_batching=True, throughput_mode=True) == (
        "org/model@continuous-b8"
    )


def test_an_unlisted_model_runs_on_transformers() -> None:
    assert runtime_for(None) == TRANSFORMERS


def test_a_rostered_model_runs_on_its_own_runtime() -> None:
    assert runtime_for(_spec(runtime=SGLANG)) == SGLANG


def test_an_unrostered_model_has_no_spec() -> None:
    assert spec_or_none("org/not-in-the-roster") is None


def test_a_rostered_model_resolves_to_its_spec() -> None:
    assert spec_or_none(ROSTER[0].name) == ROSTER[0]


def test_throughput_on_sglang_fills_an_unset_batch_size_with_the_default() -> None:
    assert throughput_batch_size(SGLANG, None, throughput=True) == DEFAULT_BATCH_SIZE


def test_throughput_keeps_an_explicit_batch_size_and_other_runtimes_keep_none() -> None:
    assert throughput_batch_size(SGLANG, 4, throughput=True) == 4
    assert throughput_batch_size(TRANSFORMERS, None, throughput=True) is None
    assert throughput_batch_size(SGLANG, None, throughput=False) is None


def test_continuous_batching_applies_to_a_plain_or_unlisted_transformers_model() -> None:
    assert continuous_batching_applies(None, TRANSFORMERS, continuous_batching=True) is True
    assert (
        continuous_batching_applies(
            _spec(runtime=TRANSFORMERS), TRANSFORMERS, continuous_batching=True
        )
        is True
    )


def test_continuous_batching_does_not_apply_to_vision_sglang_or_when_not_requested() -> None:
    vision = _spec(runtime=TRANSFORMERS, vision=True)
    sglang = _spec(runtime=SGLANG)

    assert continuous_batching_applies(vision, TRANSFORMERS, continuous_batching=True) is False
    assert continuous_batching_applies(sglang, SGLANG, continuous_batching=True) is False
    assert continuous_batching_applies(None, TRANSFORMERS, continuous_batching=False) is False


def test_throughput_applies_only_on_sglang_when_requested() -> None:
    assert throughput_applies(SGLANG, throughput=True) is True
    assert throughput_applies(TRANSFORMERS, throughput=True) is False
    assert throughput_applies(SGLANG, throughput=False) is False


def test_resolve_mode_options_combines_the_batch_size_and_the_two_mode_rules() -> None:
    sglang = _spec(runtime=SGLANG)

    assert resolve_mode_options(sglang, None, continuous_batching=True, throughput=True) == (
        DEFAULT_BATCH_SIZE,
        False,
        True,
    )
    assert resolve_mode_options(None, None, continuous_batching=True, throughput=False) == (
        None,
        True,
        False,
    )

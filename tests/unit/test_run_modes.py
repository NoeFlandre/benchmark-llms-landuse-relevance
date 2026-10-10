"""The run-mode rules: batch size, mode checks and run ids, pinned branch by branch."""

import pytest

from landuse_relevance_bench.domain.orchestration import DEFAULT_BATCH_SIZE
from landuse_relevance_bench.domain.roster import SGLANG, TRANSFORMERS, ModelSpec
from landuse_relevance_bench.domain.run_modes import check_modes, resolve_batch_size, run_id_for


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

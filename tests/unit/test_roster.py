import pytest

from landuse_relevance_bench.domain.roster import ROSTER, ModelSpec, model_ids, spec_for


def test_every_model_id_is_a_namespaced_hub_repository() -> None:
    assert all(spec.model_id.count("/") == 1 for spec in ROSTER)


def test_model_ids_are_unique() -> None:
    assert len(set(model_ids())) == len(ROSTER)


def test_every_model_is_small_enough_to_be_a_little_llm() -> None:
    assert all(spec.total_parameters < 10_000_000_000 for spec in ROSTER)


def test_lookup_returns_the_matching_spec() -> None:
    assert spec_for("LiquidAI/LFM2.5-350M").total_parameters == 354_483_968


def test_lookup_of_an_unknown_model_fails() -> None:
    with pytest.raises(KeyError, match=r"'nobody/nothing' is not in the benchmark roster"):
        spec_for("nobody/nothing")


def test_every_dspark_run_has_a_same_runtime_baseline_with_the_card_s_non_speculative_flags() -> (
    None
):
    drafted = [spec for spec in ROSTER if spec.draft_model_id]
    assert {spec.model_id for spec in drafted} == {
        "LiquidAI/LFM2.5-1.2B-Instruct",
        "LiquidAI/LFM2.5-2.6B",
        "LiquidAI/LFM2.5-8B-A1B",
        "LiquidAI/LFM2.5-VL-3B",
    }
    for spec in drafted:
        baseline = spec_for(f"{spec.model_id}@sglang")
        assert spec.name == f"{spec.model_id}+DSpark"
        assert spec.draft_model_id == f"{spec.model_id}-DSpark"
        assert spec.speculative["speculative_algorithm"] == "DSPARK"
        assert (baseline.runtime, baseline.revision, baseline.vision) == (
            spec.runtime,
            spec.revision,
            spec.vision,
        )
        assert baseline.speculative == {
            k: v for k, v in spec.speculative.items() if not k.startswith("speculative_")
        }


def test_every_rostered_run_pins_its_weights() -> None:
    assert all(spec.revision for spec in ROSTER)
    assert all(spec.draft_revision for spec in ROSTER if spec.draft_model_id)


def test_a_draft_outside_sglang_is_rejected() -> None:
    with pytest.raises(ValueError, match="sglang"):
        ModelSpec("a/b", 1, "", draft_model_id="a/draft")


def test_an_unknown_runtime_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown runtime"):
        ModelSpec("a/b", 1, "", runtime="vllm")

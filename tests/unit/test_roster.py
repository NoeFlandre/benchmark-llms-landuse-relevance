import pytest

from landuse_relevance_bench.domain.roster import (
    ROSTER,
    ModelSpec,
    dspark_settings,
    model_ids,
    sglang_pair,
    spec_for,
)

EXPECTED_PREVIOUS_MODEL_IDS = {
    "LiquidAI/LFM2.5-350M",
    "LiquidAI/LFM2.5-1.2B-Instruct",
    "LiquidAI/LFM2.5-2.6B",
    "LiquidAI/LFM2.5-8B-A1B",
    "HuggingFaceTB/SmolLM3-3B",
    "allenai/OLMo-2-1124-7B-Instruct",
    "ibm-granite/granite-3.3-2b-instruct",
    "tiiuae/Falcon3-1B-Instruct",
    "tiiuae/Falcon3-3B-Instruct",
    "tiiuae/Falcon3-7B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "allenai/Olmo-3-7B-Instruct",
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
}


def test_the_roster_is_not_empty() -> None:
    assert ROSTER


def test_every_model_id_is_a_namespaced_hub_repository() -> None:
    assert all(spec.model_id.count("/") == 1 for spec in ROSTER)


def test_model_ids_are_unique() -> None:
    assert len(set(model_ids())) == len(ROSTER)


def test_roster_includes_every_previously_tested_model() -> None:
    assert set(model_ids()) >= EXPECTED_PREVIOUS_MODEL_IDS


def test_roster_has_no_duplicate_model_ids() -> None:
    assert len(set(model_ids())) == len(model_ids())


def test_every_full_precision_model_is_small_enough_to_be_a_little_llm() -> None:
    assert all(spec.total_parameters < 10_000_000_000 for spec in ROSTER if not spec.quantization)


def test_every_quantized_model_names_its_quant_and_weights_file() -> None:
    quantized = [spec for spec in ROSTER if spec.quantization]
    assert quantized
    for spec in quantized:
        assert spec.model_id.endswith(f"@{spec.quantization}")
        assert spec.weights_file.endswith(".gguf")
        assert spec.quantization in spec.weights_file
        assert "@" not in spec.repository


def test_lookup_returns_the_matching_spec() -> None:
    assert spec_for("LiquidAI/LFM2.5-350M").total_parameters == 354_483_968


def test_lookup_of_an_unknown_model_fails() -> None:
    with pytest.raises(KeyError, match=r"'nobody/nothing' is not in the benchmark roster"):
        spec_for("nobody/nothing")


def test_dspark_runs_have_pinned_same_runtime_baselines() -> None:
    drafted = [spec for spec in ROSTER if spec.draft_model_id]
    assert {spec.model_id for spec in drafted} == {
        "LiquidAI/LFM2.5-1.2B-Instruct",
        "LiquidAI/LFM2.5-2.6B",
        "LiquidAI/LFM2.5-8B-A1B",
        "LiquidAI/LFM2.5-VL-3B",
    }
    assert all(spec.revision and spec.draft_revision for spec in drafted)
    for spec in drafted:
        baseline = spec_for(f"{spec.model_id}@sglang")
        assert baseline.runtime == spec.runtime == "sglang"
        assert baseline.revision == spec.revision
        assert baseline.vision == spec.vision
        assert baseline.speculative == {
            key: value
            for key, value in spec.speculative.items()
            if not key.startswith("speculative_")
        }


def test_an_invalid_runtime_or_non_sglang_draft_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown runtime"):
        ModelSpec("owner/model", 1, "", runtime="vllm")
    with pytest.raises(ValueError, match="a speculative draft needs the sglang runtime"):
        ModelSpec("owner/model", 1, "", draft_model_id="owner/draft")


def test_dspark_settings_and_pair_preserve_the_target_recipe() -> None:
    settings = dspark_settings(0.5, speculative_dspark_block_size=8)
    assert settings == {
        "speculative_algorithm": "DSPARK",
        "speculative_draft_attention_backend": "flashinfer",
        "disable_radix_cache": True,
        "mem_fraction_static": 0.5,
        "speculative_dspark_block_size": 8,
    }
    target = ModelSpec("owner/model", 10, "target", revision="r1", vision=True, batch_size=16)
    baseline, drafted = sglang_pair(target, "owner/draft", "r2", 3, dspark_settings(0.5))
    assert (baseline.name, drafted.name) == ("owner/model@sglang", "owner/model+DSpark")
    assert (baseline.runtime, drafted.runtime, drafted.draft_revision) == ("sglang", "sglang", "r2")
    assert drafted.draft_parameters == 3
    assert baseline.speculative == {"disable_radix_cache": True, "mem_fraction_static": 0.5}


def test_sglang_pair_builds_exact_baseline_and_drafted_specs() -> None:
    settings = dspark_settings(0.5)
    target = ModelSpec("owner/model", 10, "target", revision="r1", vision=True, batch_size=16)
    baseline, drafted = sglang_pair(target, "owner/draft", "r2", 3, settings)
    assert baseline == ModelSpec(
        "owner/model",
        10,
        "SGLang, no draft: the like-for-like baseline for DSpark.",
        run_id="owner/model@sglang",
        revision="r1",
        runtime="sglang",
        vision=True,
        batch_size=1,
        speculative={"disable_radix_cache": True, "mem_fraction_static": 0.5},
    )
    assert drafted == ModelSpec(
        "owner/model",
        10,
        "SGLang with the DSpark speculative draft; lossless under greedy decoding.",
        run_id="owner/model+DSpark",
        revision="r1",
        runtime="sglang",
        vision=True,
        draft_model_id="owner/draft",
        draft_revision="r2",
        draft_parameters=3,
        batch_size=1,
        speculative=settings,
    )


def test_a_draft_on_the_sglang_runtime_is_accepted() -> None:
    spec = ModelSpec("owner/model", 1, "", runtime="sglang", draft_model_id="owner/draft")
    assert spec.draft_model_id == "owner/draft"

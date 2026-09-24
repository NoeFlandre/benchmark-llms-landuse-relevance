import pytest

from landuse_relevance_bench.domain.roster import ROSTER, model_ids, spec_for

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

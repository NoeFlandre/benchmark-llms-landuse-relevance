"""The model IDs included in the active multilingual benchmark.

The roster reuses every model ID tested in the earlier benchmark, but all new
results are generated against the active multilingual dataset. Historical result
files remain archive-only.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One benchmarked model, with the scale needed to read its scores fairly."""

    model_id: str
    total_parameters: int
    note: str


ROSTER: tuple[ModelSpec, ...] = (
    ModelSpec("LiquidAI/LFM2.5-350M", 354_483_968, "Smallest LFM2.5 dense model."),
    ModelSpec("LiquidAI/LFM2.5-1.2B-Instruct", 1_170_340_608, "Instruction-tuned 1.2B."),
    ModelSpec("LiquidAI/LFM2.5-2.6B", 2_697_198_592, "Mid-size LFM2.5 dense model."),
    ModelSpec("LiquidAI/LFM2.5-8B-A1B", 8_467_856_832, "Sparse MoE, ~1B active parameters."),
    ModelSpec("HuggingFaceTB/SmolLM3-3B", 3_000_000_000, "Previously tested SmolLM3 3B."),
    ModelSpec("allenai/OLMo-2-1124-7B-Instruct", 7_000_000_000, "Previously tested OLMo 2 7B."),
    ModelSpec(
        "ibm-granite/granite-3.3-2b-instruct",
        2_000_000_000,
        "Previously tested Granite 3.3 2B.",
    ),
    ModelSpec("tiiuae/Falcon3-1B-Instruct", 1_000_000_000, "Previously tested Falcon 3 1B."),
    ModelSpec("tiiuae/Falcon3-3B-Instruct", 3_000_000_000, "Previously tested Falcon 3 3B."),
    ModelSpec("tiiuae/Falcon3-7B-Instruct", 7_000_000_000, "Previously tested Falcon 3 7B."),
    ModelSpec("Qwen/Qwen3-0.6B", 600_000_000, "Previously tested Qwen3 0.6B."),
    ModelSpec("Qwen/Qwen3-1.7B", 1_700_000_000, "Previously tested Qwen3 1.7B."),
    ModelSpec("Qwen/Qwen3-4B", 4_000_000_000, "Previously tested Qwen3 4B."),
    ModelSpec("Qwen/Qwen3-8B", 8_000_000_000, "Previously tested Qwen3 8B."),
    ModelSpec(
        "Qwen/Qwen3-4B-Instruct-2507",
        4_000_000_000,
        "Previously tested Qwen3 4B Instruct 2507.",
    ),
    ModelSpec("allenai/Olmo-3-7B-Instruct", 7_000_000_000, "Previously tested OLMo 3 7B."),
    ModelSpec("google/gemma-4-E2B-it", 2_000_000_000, "Previously tested Gemma 4 E2B."),
    ModelSpec("google/gemma-4-E4B-it", 4_000_000_000, "Previously tested Gemma 4 E4B."),
    ModelSpec("Qwen/Qwen3.5-4B", 4_659_900_000, "Qwen3.5 4B; vision-language, prompted text-only."),
    ModelSpec("Qwen/Qwen3.5-9B", 9_653_100_000, "Qwen3.5 9B; vision-language, prompted text-only."),
    ModelSpec(
        "Qwen/Qwen3.5-0.8B", 873_400_000, "Qwen3.5 0.8B; vision-language, prompted text-only."
    ),
    ModelSpec("Qwen/Qwen3.5-2B", 2_274_100_000, "Qwen3.5 2B; vision-language, prompted text-only."),
    ModelSpec("tiiuae/Falcon-H1-3B-Instruct", 3_149_400_000, "Falcon-H1 3B; hybrid attention-SSM."),
    ModelSpec("ibm-granite/granite-4.1-3b", 3_402_800_000, "Granite 4.1 3B."),
    ModelSpec("microsoft/Phi-4-mini-instruct", 3_836_000_000, "Phi-4 mini instruct."),
    ModelSpec(
        "mistralai/Ministral-3-3B-Instruct-2512-BF16",
        4_251_700_000,
        "Ministral 3 3B; vision-language, prompted text-only.",
    ),
    ModelSpec("swiss-ai/Apertus-8B-Instruct-2509", 8_053_300_000, "Apertus 8B instruct."),
    ModelSpec(
        "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        8_918_000_000,
        "Ministral 3 8B; vision-language, prompted text-only.",
    ),
    ModelSpec("utter-project/EuroLLM-9B-Instruct-2512", 9_152_300_000, "EuroLLM 9B instruct."),
)


def model_ids() -> tuple[str, ...]:
    return tuple(spec.model_id for spec in ROSTER)


def spec_for(model_id: str) -> ModelSpec:
    for spec in ROSTER:
        if spec.model_id == model_id:
            return spec
    raise KeyError(f"{model_id!r} is not in the benchmark roster")

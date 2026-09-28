"""The model IDs included in the active multilingual benchmark.

The roster reuses every model ID tested in the earlier benchmark, but all new
results are generated against the active multilingual dataset. Historical result
files remain archive-only.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from landuse_relevance_bench.domain.variants import repository_of

TRANSFORMERS = "transformers"
SGLANG = "sglang"
RUNTIMES = (TRANSFORMERS, SGLANG)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One benchmarked model, with the scale needed to read its scores fairly."""

    model_id: str
    total_parameters: int
    note: str
    # A GGUF quant label (e.g. ``UD-IQ2_XXS``). Empty for full-precision Transformers
    # checkpoints; set, the model runs through llama.cpp and results record the label.
    quantization: str = ""
    weights_file: str = ""
    run_id: str = ""
    revision: str | None = None
    runtime: str = TRANSFORMERS
    vision: bool = False
    draft_model_id: str = ""
    draft_revision: str | None = None
    draft_parameters: int = 0
    batch_size: int | None = None
    speculative: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.runtime not in RUNTIMES:
            raise ValueError(f"unknown runtime {self.runtime!r}; expected one of {RUNTIMES}")
        if self.draft_model_id and self.runtime != SGLANG:
            raise ValueError("a speculative draft needs the sglang runtime")

    @property
    def name(self) -> str:
        """The unique name for this model/runtime variant."""
        return self.run_id or self.model_id

    @property
    def repository(self) -> str:
        """The Hub repository to load; the roster id minus any ``@variant``."""
        return repository_of(self.model_id)


def dspark_settings(mem_fraction_static: float, **extra: object) -> dict[str, Any]:
    """The shared SGLang flags from the Liquid AI DSpark recipes."""
    return {
        "speculative_algorithm": "DSPARK",
        "speculative_draft_attention_backend": "flashinfer",
        "disable_radix_cache": True,
        "mem_fraction_static": mem_fraction_static,
        **extra,
    }


def sglang_pair(
    target: ModelSpec,
    draft_model_id: str,
    draft_revision: str,
    draft_parameters: int,
    settings: Mapping[str, Any],
) -> tuple[ModelSpec, ModelSpec]:
    """Return the SGLang baseline and its otherwise identical DSpark run."""
    baseline_settings = {
        key: value for key, value in settings.items() if not key.startswith("speculative_")
    }
    sglang_target = replace(target, runtime=SGLANG, batch_size=1)
    baseline = replace(
        sglang_target,
        note="SGLang, no draft: the like-for-like baseline for DSpark.",
        run_id=f"{target.model_id}@sglang",
        speculative=baseline_settings,
    )
    drafted = replace(
        sglang_target,
        note="SGLang with the DSpark speculative draft; lossless under greedy decoding.",
        run_id=f"{target.model_id}+DSpark",
        draft_model_id=draft_model_id,
        draft_revision=draft_revision,
        draft_parameters=draft_parameters,
        speculative=dict(settings),
    )
    return baseline, drafted


LFM_350M = ModelSpec(
    "LiquidAI/LFM2.5-350M",
    354_483_968,
    "Smallest LFM2.5 dense model.",
    revision="9e6c6ccf47cd318696e137d381a7ded8fe4df09f",
)
LFM_1_2B = ModelSpec(
    "LiquidAI/LFM2.5-1.2B-Instruct",
    1_170_340_608,
    "Instruction-tuned 1.2B.",
    revision="0f604ada3f766f9f257460c4c9f0b5d6f69d431b",
)
LFM_2_6B = ModelSpec(
    "LiquidAI/LFM2.5-2.6B",
    2_697_198_592,
    "Mid-size LFM2.5 dense model.",
    revision="654f9463ce32b05d0429d76fe1f580b27d4c1ac0",
)
LFM_8B_A1B = ModelSpec(
    "LiquidAI/LFM2.5-8B-A1B",
    8_467_856_832,
    "Sparse MoE, ~1B active parameters.",
    revision="5dd22602c2e9f6a097b1de4c4efe0658b605015c",
)
LFM_VL_3B = ModelSpec(
    "LiquidAI/LFM2.5-VL-3B",
    3_123_483_888,
    "Vision-language 3B, prompted with text only.",
    revision="35a118d938ce6d123ac2d371649f24a8efb69058",
    vision=True,
)
_TEXT_DSPARK = dspark_settings(0.75)
_VL_DSPARK = dspark_settings(0.8, speculative_dspark_block_size=9)


ROSTER: tuple[ModelSpec, ...] = (
    LFM_350M,
    LFM_1_2B,
    LFM_2_6B,
    LFM_8B_A1B,
    LFM_VL_3B,
    *sglang_pair(
        LFM_1_2B,
        "LiquidAI/LFM2.5-1.2B-Instruct-DSpark",
        "4876d04848e15a6fd48d7c1481110e7cf5d62621",
        295_725_953,
        _TEXT_DSPARK,
    ),
    *sglang_pair(
        LFM_2_6B,
        "LiquidAI/LFM2.5-2.6B-DSpark",
        "458cedab07d0f7b2b05700c77e1aa463d43d6f04",
        327_707_521,
        _TEXT_DSPARK,
    ),
    *sglang_pair(
        LFM_8B_A1B,
        "LiquidAI/LFM2.5-8B-A1B-DSpark",
        "5b285c827912834665b1915f171897e49ff0f388",
        327_707_521,
        _TEXT_DSPARK,
    ),
    *sglang_pair(
        LFM_VL_3B,
        "LiquidAI/LFM2.5-VL-3B-DSpark",
        "af77e9306a26e8625fde74d2a3051ab6d21bd955",
        279_468_801,
        _VL_DSPARK,
    ),
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
    ModelSpec(
        "unsloth/Qwen3.8-27B-GGUF@UD-IQ2_XXS",
        27_320_697_856,
        "Qwen3.8 27B at a ~2-bit GGUF quant (7.3 GB) through llama.cpp.",
        quantization="UD-IQ2_XXS",
        weights_file="Qwen3.8-27B-UD-IQ2_XXS.gguf",
    ),
)


def model_ids() -> tuple[str, ...]:
    return tuple(spec.name for spec in ROSTER)


def spec_for(model_id: str) -> ModelSpec:
    for spec in ROSTER:
        if spec.name == model_id:
            return spec
    raise KeyError(f"{model_id!r} is not in the benchmark roster")


def quantization_of(model_id: str) -> str:
    """The quant label a rostered model runs at; empty for unrostered or full precision."""
    try:
        return spec_for(model_id).quantization
    except KeyError:
        return ""

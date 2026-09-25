"""The models under benchmark.

Liquid AI's LFM2.5 family is the subject; each entry is an ungated Hugging Face
repository that was current at the time of writing.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

TRANSFORMERS = "transformers"
SGLANG = "sglang"
RUNTIMES = (TRANSFORMERS, SGLANG)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One benchmarked run, with the scale needed to read its scores fairly.

    ``run_id`` names the run; it defaults to ``model_id`` and differs only when one
    model is run several ways, e.g. with and without a speculative draft. ``vision``
    marks a vision-language model, prompted here with text only.
    """

    model_id: str
    total_parameters: int
    note: str
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
        return self.run_id or self.model_id


def _dspark_settings(mem_fraction_static: float, **extra: Any) -> dict[str, Any]:
    return {
        "speculative_algorithm": "DSPARK",
        "speculative_draft_attention_backend": "flashinfer",
        "disable_radix_cache": True,
        "mem_fraction_static": mem_fraction_static,
        **extra,
    }


def _sglang_pair(
    target: ModelSpec,
    draft_model_id: str,
    draft_revision: str,
    draft_parameters: int,
    settings: Mapping[str, Any],
) -> tuple[ModelSpec, ModelSpec]:
    """The card's recipe: the target under SGLang, then the same with the draft attached.

    The baseline is "the same command without the ``--speculative-*`` flags", which is
    what makes the pair a like-for-like speed comparison and a lossless check.
    """
    baseline_settings = {k: v for k, v in settings.items() if not k.startswith("speculative_")}
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

#: Each DSpark card's SGLang recipe. The text drafters read their block size from
#: the draft config; the VL card sets it explicitly for an H100.
_TEXT_DSPARK = _dspark_settings(0.75)
_VL_DSPARK = _dspark_settings(0.8, speculative_dspark_block_size=9)

ROSTER: tuple[ModelSpec, ...] = (
    LFM_350M,
    LFM_1_2B,
    LFM_2_6B,
    LFM_8B_A1B,
    LFM_VL_3B,
    *_sglang_pair(
        LFM_1_2B,
        "LiquidAI/LFM2.5-1.2B-Instruct-DSpark",
        "4876d04848e15a6fd48d7c1481110e7cf5d62621",
        295_725_953,
        _TEXT_DSPARK,
    ),
    *_sglang_pair(
        LFM_2_6B,
        "LiquidAI/LFM2.5-2.6B-DSpark",
        "458cedab07d0f7b2b05700c77e1aa463d43d6f04",
        327_707_521,
        _TEXT_DSPARK,
    ),
    *_sglang_pair(
        LFM_8B_A1B,
        "LiquidAI/LFM2.5-8B-A1B-DSpark",
        "5b285c827912834665b1915f171897e49ff0f388",
        327_707_521,
        _TEXT_DSPARK,
    ),
    *_sglang_pair(
        LFM_VL_3B,
        "LiquidAI/LFM2.5-VL-3B-DSpark",
        "af77e9306a26e8625fde74d2a3051ab6d21bd955",
        279_468_801,
        _VL_DSPARK,
    ),
)


def model_ids() -> tuple[str, ...]:
    """The name of every rostered run, in roster order."""
    return tuple(spec.name for spec in ROSTER)


def spec_for(name: str) -> ModelSpec:
    for spec in ROSTER:
        if spec.name == name:
            return spec
    raise KeyError(f"{name!r} is not in the benchmark roster")

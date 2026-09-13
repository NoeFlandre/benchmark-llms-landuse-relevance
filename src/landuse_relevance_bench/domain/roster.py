"""The models under benchmark.

Liquid AI's LFM2.5 family is the subject; each entry is an ungated Hugging Face
repository that was current at the time of writing.
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
)


def model_ids() -> tuple[str, ...]:
    return tuple(spec.model_id for spec in ROSTER)


def spec_for(model_id: str) -> ModelSpec:
    for spec in ROSTER:
        if spec.model_id == model_id:
            return spec
    raise KeyError(f"{model_id!r} is not in the benchmark roster")

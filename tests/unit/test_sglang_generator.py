"""SGLangGenerator's translation to and from the engine, with a fake engine."""

from pathlib import Path
from typing import Any

from landuse_relevance_bench.adapters.pipeline import RunRequest
from landuse_relevance_bench.adapters.sglang_generator import (
    SGLangGenerator,
    as_generation,
    engine_arguments,
)


def _request(name: str) -> RunRequest:
    return RunRequest.for_run(
        name,
        benchmark_path=Path("unused"),
        prompt_path=Path("unused"),
        output_dir=Path("unused"),
        max_new_tokens=64,
    )


def test_the_dspark_run_launches_the_card_s_recipe_on_pinned_weights() -> None:
    arguments = engine_arguments(_request("LiquidAI/LFM2.5-VL-3B+DSpark"))
    assert arguments == {
        "model_path": "LiquidAI/LFM2.5-VL-3B",
        "revision": "35a118d938ce6d123ac2d371649f24a8efb69058",
        "dtype": "bfloat16",
        "random_seed": 0,
        "speculative_algorithm": "DSPARK",
        "speculative_draft_model_path": "LiquidAI/LFM2.5-VL-3B-DSpark",
        "speculative_draft_model_revision": "af77e9306a26e8625fde74d2a3051ab6d21bd955",
        "speculative_draft_attention_backend": "flashinfer",
        "speculative_dspark_block_size": 9,
        "disable_radix_cache": True,
        "mem_fraction_static": 0.8,
    }


def test_the_baseline_is_the_same_launch_without_any_speculative_argument() -> None:
    drafted = engine_arguments(_request("LiquidAI/LFM2.5-2.6B+DSpark"))
    baseline = engine_arguments(_request("LiquidAI/LFM2.5-2.6B@sglang"))
    assert baseline == {k: v for k, v in drafted.items() if not k.startswith("speculative_")}
    assert baseline["mem_fraction_static"] == 0.75


def test_a_speculative_completion_reports_its_draft_statistics() -> None:
    generation = as_generation(
        {
            "text": " yes \n",
            "meta_info": {
                "completion_tokens": 12,
                "finish_reason": {"type": "stop"},
                "spec_verify_ct": 3,
                "spec_num_correct_drafts": 9,
                "spec_num_proposed_drafts": 24,
            },
        }
    )
    assert generation.text == "yes"
    assert not generation.truncated
    assert (generation.generated_tokens, generation.verify_steps) == (12, 3)
    assert (generation.accepted_drafts, generation.proposed_drafts) == (9, 24)


def test_a_plain_completion_that_hit_the_budget_is_truncated_without_draft_statistics() -> None:
    generation = as_generation(
        {
            "text": "1. Analyze",
            "meta_info": {"finish_reason": {"type": "length"}, "spec_verify_ct": 0},
        }
    )
    assert generation.truncated
    assert generation.verify_steps is None
    assert generation.generated_tokens is None


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.stopped = False

    def shutdown(self) -> None:
        self.stopped = True

    def generate(self, *, input_ids: list[list[int]], sampling_params: dict[str, Any]) -> Any:
        self.calls.append({"input_ids": input_ids, "sampling_params": sampling_params})
        return [{"text": "no", "meta_info": {"completion_tokens": 1}} for _ in input_ids]


def test_generation_is_greedy_within_budget_on_the_encoded_chat_prompts() -> None:
    engine = FakeEngine()
    generator = SGLangGenerator(engine, lambda prompt: [len(prompt)], max_new_tokens=64)
    outputs = generator.generate(["ab", "abc"])
    assert [o.text for o in outputs] == ["no", "no"]
    assert engine.calls == [
        {"input_ids": [[2], [3]], "sampling_params": {"temperature": 0.0, "max_new_tokens": 64}}
    ]


def test_an_empty_batch_never_reaches_the_engine() -> None:
    engine = FakeEngine()
    assert SGLangGenerator(engine, lambda _: [], max_new_tokens=1).generate([]) == []
    assert engine.calls == []


def test_closing_the_generator_shuts_the_engine_down() -> None:
    engine = FakeEngine()
    SGLangGenerator(engine, lambda _: [], max_new_tokens=1).close()
    assert engine.stopped

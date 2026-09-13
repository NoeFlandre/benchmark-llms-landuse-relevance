"""Smoke test of the real Transformers path against a genuinely tiny model.

Deselected by default; run with ``pytest -m integration``.
"""

from pathlib import Path

import pytest

from landuse_relevance_bench.adapters.hf_generator import GeneratorSettings, TransformersGenerator
from landuse_relevance_bench.adapters.pipeline import RunRequest, execute

pytestmark = pytest.mark.integration

TINY_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
SETTINGS = GeneratorSettings(max_new_tokens=4, dtype="float32", seed=0, device_map="cpu")


@pytest.fixture(scope="module")
def generator() -> TransformersGenerator:
    return TransformersGenerator.load(TINY_MODEL, SETTINGS)


def test_returns_one_completion_per_prompt(generator: TransformersGenerator) -> None:
    outputs = generator.generate(["Say yes.", "Say no.", "Say yes."])
    assert len(outputs) == 3
    assert all(isinstance(o, str) for o in outputs)


def test_an_empty_batch_needs_no_forward_pass(generator: TransformersGenerator) -> None:
    assert generator.generate([]) == []


def test_greedy_decoding_repeats_itself_exactly(generator: TransformersGenerator) -> None:
    assert generator.generate(["Reply with one word."]) == generator.generate(
        ["Reply with one word."]
    )


def test_batching_does_not_change_a_prompt_s_completion(generator: TransformersGenerator) -> None:
    alone = generator.generate(["Answer yes or no: is grass green?"])[0]
    batched = generator.generate(
        ["Answer yes or no: is grass green?", "Something else entirely to pad the batch."]
    )[0]
    assert alone == batched


def test_a_whole_run_completes_end_to_end(
    tmp_path: Path, benchmark_path: Path, prompt_path: Path, generator: TransformersGenerator
) -> None:
    result = execute(
        RunRequest(
            model_id=TINY_MODEL,
            benchmark_path=benchmark_path,
            prompt_path=prompt_path,
            output_dir=tmp_path,
            max_new_tokens=4,
            dtype="float32",
        ),
        lambda _request: (generator, "smoke"),
    )
    assert result.metrics.n_items == 2
    assert (tmp_path / "HuggingFaceTB__SmolLM2-135M-Instruct.json").exists()

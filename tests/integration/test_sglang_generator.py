"""The real SGLang path, plain and with the DSpark draft, on the VL-3B target.

Deselected by default; needs a CUDA GPU and the `speculative` extra
(``uv sync --extra speculative``), then ``pytest -m integration``.
"""

import importlib.util
from pathlib import Path

import pytest

from landuse_relevance_bench.adapters.pipeline import RunRequest
from landuse_relevance_bench.adapters.sglang_generator import SGLangGenerator

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        importlib.util.find_spec("sglang") is None, reason="needs the `speculative` extra"
    ),
]

PROMPTS = [
    "Answer yes or no: does 'Dense mangrove forest lines the lagoon.' describe land cover?",
    "Answer yes or no: does 'The council was dissolved in 1974.' describe land cover?",
]


def _generations(name: str) -> list:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    request = RunRequest.for_run(
        name,
        benchmark_path=Path("unused"),
        prompt_path=Path("unused"),
        output_dir=Path("unused"),
        max_new_tokens=32,
    )
    generator = SGLangGenerator.load(request)
    try:
        return generator.generate(PROMPTS)
    finally:
        generator.close()


def test_the_dspark_draft_changes_speed_but_not_the_greedy_output() -> None:
    plain = _generations("LiquidAI/LFM2.5-VL-3B@sglang")
    drafted = _generations("LiquidAI/LFM2.5-VL-3B+DSpark")
    assert [g.text for g in drafted] == [g.text for g in plain]
    assert all(g.verify_steps for g in drafted)
    assert all(g.verify_steps is None for g in plain)

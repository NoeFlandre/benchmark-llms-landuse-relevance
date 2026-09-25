from dataclasses import replace
from pathlib import Path

import pytest

from landuse_relevance_bench.adapters import generators, hf_generator, sglang_generator
from landuse_relevance_bench.adapters.pipeline import RunRequest

REQUEST = RunRequest(
    model_id="a/b", benchmark_path=Path("x"), prompt_path=Path("x"), output_dir=Path("x")
)


@pytest.mark.parametrize(
    ("runtime", "module"), [("transformers", hf_generator), ("sglang", sglang_generator)]
)
def test_the_request_s_runtime_picks_the_generator(monkeypatch, runtime: str, module) -> None:
    monkeypatch.setattr(module, "provide", lambda _request: (runtime, "rev"))
    assert generators.provide(replace(REQUEST, runtime=runtime)) == (runtime, "rev")


def test_an_unknown_runtime_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown runtime"):
        generators.provide(replace(REQUEST, runtime="vllm"))

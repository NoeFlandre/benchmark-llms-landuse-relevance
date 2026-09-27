"""Transformers adapter logic tests that run without importing the inference extra."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

from landuse_relevance_bench.adapters import hf_generator
from landuse_relevance_bench.adapters.hf_generator import GeneratorSettings, TransformersGenerator

EOS_TEXT = "sentinel"


class FakeRow:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def tolist(self) -> list[int]:
        return self.values


class FakeTensor:
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)

    def to(self, _device: str) -> "FakeTensor":
        return self

    def __getitem__(self, key: tuple[slice, slice]) -> "FakeTensor":
        _, completion_slice = key
        return FakeTensor([row[completion_slice] for row in self.rows])

    def __iter__(self):
        return iter(FakeRow(row) for row in self.rows)


class FakeTokenizer:
    eos_token_id = 0
    eos_token = EOS_TEXT
    pad_token_id = 0
    padding_side = "right"
    pad_token = None

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return messages[0]["content"]

    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, FakeTensor]:
        assert kwargs == {"return_tensors": "pt", "padding": True, "add_special_tokens": False}
        assert texts == ["first", "second"]
        return {"input_ids": FakeTensor([[1, 2], [1, 3]])}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [len(text)]

    def decode(self, row: FakeRow | list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        values = row.values if isinstance(row, FakeRow) else row
        return "yes" if values[0] == 7 else "thinking no"


class FakeModel:
    device = "cpu"
    generation_config = SimpleNamespace(eos_token_id=0)

    def generate(self, *, input_ids: FakeTensor, **kwargs: Any) -> FakeTensor:
        assert input_ids.shape == (2, 2)
        assert kwargs["do_sample"] is False
        assert kwargs["max_new_tokens"] == 2
        return FakeTensor([[1, 2, 7, 0], [1, 3, 8, 9]])


class FakeContinuousModel(FakeModel):
    def generate_batch(self, *, inputs: list[list[int]], generation_config: Any, **kwargs: Any):
        self.inputs = inputs
        self.generation_config_seen = generation_config
        self.batch_kwargs = kwargs
        return {
            "req_0": SimpleNamespace(
                request_id="req_0",
                generated_tokens=[7, 0],
                error=None,
                created_time=1.0,
                lifespan=(1.0, 1.125),
            ),
            "req_1": SimpleNamespace(
                request_id="req_1",
                generated_tokens=[8, 9],
                error=None,
                created_time=1.0,
                lifespan=(1.0, 1.250),
            ),
        }


def test_adapter_generates_and_decodes_completions_without_torch(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(inference_mode=nullcontext),
    )
    generator = TransformersGenerator(
        FakeTokenizer(),
        FakeModel(),
        GeneratorSettings(max_new_tokens=2, dtype="float32", seed=0),
    )

    results = generator.generate(["first", "second"])

    assert [(item.text, item.truncated, item.generated_tokens) for item in results] == [
        ("yes", False, 2),
        ("thinking no", True, 2),
    ]


def test_continuous_batching_uses_unpadded_inputs_and_keeps_request_order(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(inference_mode=nullcontext),
    )
    model = FakeContinuousModel()
    generator = TransformersGenerator(
        FakeTokenizer(),
        model,
        GeneratorSettings(
            max_new_tokens=2,
            dtype="float32",
            seed=0,
            continuous_batching=True,
        ),
    )

    results = generator.generate(["first", "second"])

    assert model.inputs == [[5], [6]]
    assert model.generation_config_seen.do_sample is False
    assert model.generation_config_seen.max_new_tokens == 2
    assert model.batch_kwargs == {"progress_bar": False, "warmup": True}
    assert [(item.text, item.truncated) for item in results] == [
        ("yes", False),
        ("thinking no", True),
    ]
    assert [item.latency_seconds for item in results] == [0.125, 0.25]


def test_tokenizer_preparation_left_pads_and_uses_eos_when_needed() -> None:
    tokenizer = FakeTokenizer()
    tokenizer.pad_token_id = None
    hf_generator._prepare_tokenizer(tokenizer)
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == EOS_TEXT


def test_close_releases_model_and_empty_cuda_cache(monkeypatch) -> None:
    import gc
    import sys

    calls: list[str] = []
    monkeypatch.setattr(gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True, empty_cache=lambda: calls.append("empty")
            )
        ),
    )
    generator = TransformersGenerator(
        FakeTokenizer(), FakeModel(), GeneratorSettings(max_new_tokens=2, dtype="float32", seed=0)
    )

    generator.close()

    assert generator._model is None
    assert calls == ["gc", "empty"]

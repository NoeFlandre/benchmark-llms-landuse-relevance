"""Which loader a checkpoint gets, decided from its config rather than assumed."""

from typing import Any

import pytest

from landuse_relevance_bench.adapters import hf_generator


class _Config:
    def __init__(self, architectures: list[str] | None, vision_config: Any = None) -> None:
        self.architectures = architectures
        self.vision_config = vision_config


def _pretend_config(monkeypatch: pytest.MonkeyPatch, config: _Config) -> None:
    import transformers

    def from_pretrained(*_args: Any, **_kwargs: Any) -> _Config:
        return config

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", from_pretrained)


def test_a_plain_causal_checkpoint_loads_as_a_causal_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pretend_config(monkeypatch, _Config(["Qwen3ForCausalLM"]))

    assert hf_generator._auto_class("some/model", None).__name__ == "AutoModelForCausalLM"


def test_a_vision_language_checkpoint_loads_as_an_image_text_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pretend_config(
        monkeypatch, _Config(["Qwen3_5ForConditionalGeneration"], vision_config={"depth": 1})
    )

    assert hf_generator._auto_class("some/vlm", None).__name__ == "AutoModelForImageTextToText"


def test_a_conditional_generation_checkpoint_without_a_vision_tower_still_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pretend_config(monkeypatch, _Config(["SomethingForConditionalGeneration"]))

    assert hf_generator._auto_class("some/cond", None).__name__ == "AutoModelForImageTextToText"


def test_a_checkpoint_that_declares_no_architecture_falls_back_to_causal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pretend_config(monkeypatch, _Config(None))

    assert hf_generator._auto_class("some/bare", None).__name__ == "AutoModelForCausalLM"

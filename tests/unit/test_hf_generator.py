"""TransformersGenerator's decision logic, exercised through its public surface with stand-ins."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from landuse_relevance_bench.adapters import hf_generator
from landuse_relevance_bench.adapters.hf_generator import (
    GeneratorSettings,
    TransformersGenerator,
    stop_token_ids,
)
from landuse_relevance_bench.adapters.pipeline import RunRequest
from landuse_relevance_bench.domain.engine import Generation

SETTINGS = GeneratorSettings(max_new_tokens=3, dtype="bfloat16", seed=0)
EOS = 2


MISSING = object()


class FakeBatch(dict):
    """What a tokenizer returns: named tensors that can be moved to a device."""


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __init__(self) -> None:
        self.texts: list[str] = []

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return messages[0]["content"]

    def __call__(self, texts: list[str], **kwargs: Any) -> FakeBatch:
        assert kwargs == {"return_tensors": "pt", "padding": True, "add_special_tokens": False}
        self.texts.extend(texts)
        return FakeBatch(input_ids=torch.zeros((len(texts), 1), dtype=torch.long))

    def decode(self, completion: Any, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return "  " + " ".join(f"t{i}" for i in completion.tolist() if i != EOS) + "\n"


class FakeModel:
    """Echoes the prompt column, then emits the chosen completion rows."""

    device = "cpu"

    def __init__(self, rows: list[list[int]], generation_config: Any, config: Any) -> None:
        self._rows = rows
        self.generation_config = generation_config
        self.config = config

    def generate(self, *, input_ids: Any, **kwargs: Any) -> Any:
        assert kwargs["do_sample"] is False
        assert kwargs["max_new_tokens"] == SETTINGS.max_new_tokens
        return torch.cat([input_ids, torch.tensor(self._rows, dtype=torch.long)], dim=1)


def _generator(
    rows: list[list[int]] | None = None,
    *,
    eos_token_id: Any = EOS,
    has_config: bool = True,
    commit: Any = "abc",
    tokenizer: Any = None,
) -> TransformersGenerator:
    config = SimpleNamespace(eos_token_id=eos_token_id) if has_config else None
    model_config = SimpleNamespace() if commit is MISSING else SimpleNamespace(_commit_hash=commit)
    model = FakeModel(rows or [[5, 6, 7]], config, model_config)
    return TransformersGenerator(tokenizer or FakeTokenizer(), model, SETTINGS)


def _one(generator: TransformersGenerator) -> Generation:
    (generation,) = generator.generate(["prompt"])
    return generation


def test_a_single_eos_id_is_the_only_stop_token() -> None:
    assert stop_token_ids(SimpleNamespace(eos_token_id=7), FakeTokenizer()) == frozenset({7})


def test_a_list_of_eos_ids_drops_missing_entries() -> None:
    config = SimpleNamespace(eos_token_id=[7, None, 8])
    assert stop_token_ids(config, FakeTokenizer()) == frozenset({7, 8})


def test_without_a_generation_config_the_tokenizer_eos_is_used() -> None:
    assert stop_token_ids(None, FakeTokenizer()) == frozenset({99})


def test_an_unset_eos_id_yields_no_stop_tokens() -> None:
    assert stop_token_ids(SimpleNamespace(eos_token_id=None), FakeTokenizer()) == frozenset()


def test_generation_stops_on_the_configured_eos_id() -> None:
    assert _one(_generator([[5, 7, 0]], eos_token_id=7)).truncated is False
    assert _one(_generator([[5, 7, 0]], eos_token_id=[8, None, 7])).generated_tokens == 2


def test_generation_falls_back_to_the_tokenizer_eos_without_a_config() -> None:
    generation = _one(_generator([[5, 99, 0]], has_config=False))
    assert generation.truncated is False
    assert generation.generated_tokens == 2


def test_an_empty_prompt_list_generates_nothing() -> None:
    assert _generator().generate([]) == []


def test_a_completion_that_hit_the_budget_is_truncated_and_stripped() -> None:
    generation = _one(_generator([[5, 6, 7]]))
    assert generation.text == "t5 t6 t7"
    assert generation.truncated is True


def test_a_completion_that_stopped_is_not_truncated() -> None:
    generation = _one(_generator([[5, EOS, 0]]))
    assert generation.text == "t5 t0"
    assert generation.truncated is False


def test_prompts_are_chat_templated_before_encoding() -> None:
    tokenizer = FakeTokenizer()
    _generator([[5, 6, 7], [5, 6, 7]], tokenizer=tokenizer).generate(["a", "b"])
    assert tokenizer.texts == ["a", "b"]


def test_the_revision_is_the_resolved_commit_hash() -> None:
    assert _generator(commit="deadbeef").revision == "deadbeef"


def test_the_revision_is_empty_when_the_commit_hash_is_missing() -> None:
    assert _generator(commit=MISSING).revision == ""


def test_the_revision_is_empty_when_the_commit_hash_is_null() -> None:
    assert _generator(commit=None).revision == ""


def _request(revision: str | None) -> RunRequest:
    return RunRequest(
        model_id="some/model",
        benchmark_path=Path("unused"),
        prompt_path=Path("unused"),
        output_dir=Path("unused"),
        revision=revision,
    )


@pytest.mark.parametrize(("requested", "expected"), [("pinned", "pinned"), (None, "resolved")])
def test_provide_prefers_an_explicit_revision_over_the_resolved_one(
    monkeypatch, requested: str | None, expected: str
) -> None:
    loads: list[str | None] = []

    def fake_load(model_id: str, settings: GeneratorSettings, revision: str | None = None) -> Any:
        loads.append(revision)
        return _generator(commit="resolved")

    monkeypatch.setattr(TransformersGenerator, "load", staticmethod(fake_load))
    _, revision = hf_generator.provide(_request(requested))
    assert revision == expected
    assert loads == [requested]


def test_the_generated_length_stops_at_the_first_stop_token() -> None:
    assert hf_generator.generated_length([5, EOS, EOS, EOS], {EOS}) == 2
    assert hf_generator.generated_length([5, 6, 7], {EOS}) == 3


def test_a_completion_reports_how_many_tokens_the_model_produced() -> None:
    assert _one(_generator([[5, EOS, EOS]])).generated_tokens == 2


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.messages: list[Any] = []

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        self.messages.append(messages)
        return "rendered"


def test_a_vision_language_model_gets_a_text_only_typed_user_turn() -> None:
    processor = FakeProcessor()
    model = FakeModel([[5, EOS, 0]], SimpleNamespace(eos_token_id=EOS), SimpleNamespace())
    generator = hf_generator.VisionLanguageGenerator(processor, model, SETTINGS)
    (generation,) = generator.generate(["Is this land use?"])
    assert generation.text == "t5 t0"
    assert processor.tokenizer.texts == ["rendered"]
    assert processor.messages == [
        [{"role": "user", "content": [{"type": "text", "text": "Is this land use?"}]}]
    ]


def test_provide_loads_a_vision_language_model_through_its_own_loader(monkeypatch) -> None:
    loaded: list[str] = []

    def fake_load(model_id: str, settings: GeneratorSettings, revision: str | None = None) -> Any:
        loaded.append(model_id)
        return _generator(commit="resolved")

    monkeypatch.setattr(hf_generator.VisionLanguageGenerator, "load", staticmethod(fake_load))
    request = RunRequest.for_run(
        "LiquidAI/LFM2.5-VL-3B",
        benchmark_path=Path("unused"),
        prompt_path=Path("unused"),
        output_dir=Path("unused"),
    )
    _, revision = hf_generator.provide(request)
    assert loaded == ["LiquidAI/LFM2.5-VL-3B"]
    assert revision == "35a118d938ce6d123ac2d371649f24a8efb69058"

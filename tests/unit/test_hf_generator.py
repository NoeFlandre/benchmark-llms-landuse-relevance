"""TransformersGenerator's decision logic, exercised with stand-ins rather than torch."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from landuse_relevance_bench.adapters import hf_generator
from landuse_relevance_bench.adapters.hf_generator import GeneratorSettings, TransformersGenerator
from landuse_relevance_bench.adapters.pipeline import RunRequest

SETTINGS = GeneratorSettings(max_new_tokens=3, dtype="bfloat16", seed=0)
EOS = 2


class FakeCompletion:
    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def tolist(self) -> list[int]:
        return self._ids


class FakeTokenizer:
    eos_token_id = 99

    def decode(self, completion: FakeCompletion, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return "  " + " ".join(f"t{i}" for i in completion.tolist() if i != EOS) + "\n"


def _generator(
    eos_token_id: Any = EOS, *, has_config: bool = True, commit: Any = "abc"
) -> TransformersGenerator:
    config = SimpleNamespace(eos_token_id=eos_token_id) if has_config else None
    model_config = SimpleNamespace() if commit is None else SimpleNamespace(_commit_hash=commit)
    model = SimpleNamespace(generation_config=config, config=model_config)
    return TransformersGenerator(FakeTokenizer(), model, SETTINGS)


def test_a_single_eos_id_is_the_only_stop_token() -> None:
    assert _generator(7)._stop_token_ids == frozenset({7})


def test_a_list_of_eos_ids_drops_missing_entries() -> None:
    assert _generator([7, None, 8])._stop_token_ids == frozenset({7, 8})


def test_without_a_generation_config_the_tokenizer_eos_is_used() -> None:
    assert _generator(has_config=False)._stop_token_ids == frozenset({99})


def test_an_unset_eos_id_yields_no_stop_tokens() -> None:
    assert _generator(None)._stop_token_ids == frozenset()


def test_a_completion_that_hit_the_budget_is_truncated_and_stripped() -> None:
    generation = _generator()._as_generation(FakeCompletion([5, 6, 7]))
    assert generation.text == "t5 t6 t7"
    assert generation.truncated is True


def test_a_completion_that_stopped_is_not_truncated() -> None:
    generation = _generator()._as_generation(FakeCompletion([5, EOS, 0]))
    assert generation.text == "t5 t0"
    assert generation.truncated is False


def test_the_revision_is_the_resolved_commit_hash() -> None:
    assert _generator(commit="deadbeef").revision == "deadbeef"


def test_the_revision_is_empty_when_the_commit_hash_is_missing() -> None:
    assert _generator(commit=None).revision == ""


def test_the_revision_is_empty_when_the_commit_hash_is_null() -> None:
    generator = _generator()
    generator._model.config._commit_hash = None
    assert generator.revision == ""


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
    assert _generator()._as_generation(FakeCompletion([5, EOS, EOS])).generated_tokens == 2


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def __init__(self) -> None:
        self.messages: list[Any] = []

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        self.messages.append(messages)
        return "rendered"


def test_a_vision_language_model_gets_a_text_only_typed_user_turn() -> None:
    processor = FakeProcessor()
    generator = hf_generator.VisionLanguageGenerator(processor, SimpleNamespace(), SETTINGS)
    assert generator._as_chat("Is this land use?") == "rendered"
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

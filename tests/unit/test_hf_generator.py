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

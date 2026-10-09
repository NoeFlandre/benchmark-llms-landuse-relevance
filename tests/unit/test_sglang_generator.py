"""SGLangGenerator's translation to and from the engine, with a fake engine."""

import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from landuse_relevance_bench.adapters import sglang_generator
from landuse_relevance_bench.adapters.pipeline import RunRequest
from landuse_relevance_bench.adapters.sglang_compat import SITE_DIR
from landuse_relevance_bench.adapters.sglang_generator import (
    SGLangGenerator,
    as_generation,
    engine_arguments,
)


def _request(name: str) -> RunRequest:
    return RunRequest.for_run(
        name,
        language="en",
        benchmark_path=Path("unused"),
        prompt_path=Path("unused"),
        output_dir=Path("unused"),
        max_new_tokens=64,
    )


def test_the_dspark_run_launches_the_card_s_recipe_on_pinned_weights() -> None:
    request = _request("LiquidAI/LFM2.5-VL-3B+DSpark")
    arguments = engine_arguments(request)
    assert arguments == {
        "model_path": "LiquidAI/LFM2.5-VL-3B",
        "revision": request.revision,
        "dtype": "bfloat16",
        "random_seed": 0,
        "speculative_algorithm": "DSPARK",
        "speculative_draft_model_path": "LiquidAI/LFM2.5-VL-3B-DSpark",
        "speculative_draft_model_revision": request.draft_revision,
        "speculative_draft_attention_backend": "flashinfer",
        "speculative_dspark_block_size": 9,
        "disable_radix_cache": True,
        "mem_fraction_static": 0.8,
    }


def test_engine_arguments_omit_unresolved_model_and_draft_revisions() -> None:
    request = replace(
        _request("LiquidAI/LFM2.5-2.6B+DSpark"),
        revision=None,
        draft_revision=None,
    )

    arguments = engine_arguments(request)

    assert "revision" not in arguments
    assert "speculative_draft_model_revision" not in arguments


def test_load_constructs_the_sglang_engine_lazily(monkeypatch) -> None:
    engine = FakeEngine()
    loaded: list[dict[str, Any]] = []
    monkeypatch.setitem(
        sys.modules,
        "sglang",
        SimpleNamespace(Engine=lambda **kwargs: loaded.append(kwargs) or engine),
    )
    monkeypatch.setattr(sglang_generator, "_chat_encoder", lambda _request: lambda _: [42])

    generator = SGLangGenerator.load(_request("custom/model"))

    assert loaded == [engine_arguments(_request("custom/model"))]
    assert generator.generate(["prompt"])[0].text == "no"


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
                "e2e_latency": 0.75,
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
    assert generation.latency_seconds == 0.75


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


def test_unpinned_revisions_are_resolved_before_the_engine_starts_and_passed_to_it(
    monkeypatch,
) -> None:
    events: list[str] = []
    engine_kwargs: list[dict[str, Any]] = []

    def resolve(model_id: str, *candidates: str | None) -> str:
        events.append(f"resolve {model_id}")
        return candidates[0] or f"resolved-{model_id}"

    def start_engine(**kwargs: Any) -> FakeEngine:
        events.append("engine")
        engine_kwargs.append(kwargs)
        return FakeEngine()

    monkeypatch.setattr(sglang_generator, "resolve_revision", resolve)
    _install_engine(monkeypatch, start_engine)
    request = replace(_request("LiquidAI/LFM2.5-2.6B+DSpark"), revision=None, draft_revision=None)

    generator, target_revision = sglang_generator.provide(request)

    assert events == [
        "resolve LiquidAI/LFM2.5-2.6B-DSpark",
        "resolve LiquidAI/LFM2.5-2.6B",
        "engine",
    ]
    assert target_revision == "resolved-LiquidAI/LFM2.5-2.6B"
    assert engine_kwargs[0]["revision"] == target_revision
    assert (
        engine_kwargs[0]["speculative_draft_model_revision"]
        == "resolved-LiquidAI/LFM2.5-2.6B-DSpark"
    )
    assert generator.draft_revision == "resolved-LiquidAI/LFM2.5-2.6B-DSpark"


def test_load_hands_the_draft_revision_to_the_constructor(monkeypatch) -> None:
    _install_engine(monkeypatch, lambda **_: FakeEngine())
    request = replace(
        _request("custom/model"), draft_model_id="custom/draft", draft_revision="draft-sha"
    )

    assert SGLangGenerator.load(request).draft_revision == "draft-sha"


def test_load_without_a_draft_records_no_draft_revision(monkeypatch) -> None:
    _install_engine(monkeypatch, lambda **_: FakeEngine())

    assert SGLangGenerator.load(_request("custom/model")).draft_revision == ""


def test_the_engine_sees_the_shim_on_pythonpath_while_it_starts(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/existing/path")
    seen: list[str | None] = []
    _install_engine(monkeypatch, _recording_engine(seen))

    SGLangGenerator.load(_request("custom/model"))

    assert seen[0] is not None
    assert seen[0].split(os.pathsep) == [str(SITE_DIR), "/existing/path"]


def test_load_restores_the_previous_pythonpath(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/existing/path")
    _install_engine(monkeypatch, _recording_engine([]))

    SGLangGenerator.load(_request("custom/model"))

    assert os.environ["PYTHONPATH"] == "/existing/path"


def test_load_unsets_pythonpath_again_when_the_process_had_none(monkeypatch) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    _install_engine(monkeypatch, _recording_engine([]))

    SGLangGenerator.load(_request("custom/model"))

    assert "PYTHONPATH" not in os.environ


def test_load_restores_pythonpath_when_the_engine_fails_to_start(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/existing/path")
    _install_engine(monkeypatch, _recording_engine([], fail=True))

    with pytest.raises(RuntimeError, match="engine failed to start"):
        SGLangGenerator.load(_request("custom/model"))

    assert os.environ["PYTHONPATH"] == "/existing/path"


def _install_engine(monkeypatch, start: Any) -> None:
    """Stand in for ``sglang.Engine`` and the tokenizer so ``load`` runs without SGLang."""
    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=start))
    monkeypatch.setattr(sglang_generator, "_chat_encoder", lambda _request: lambda _: [1])


def _recording_engine(seen: list[str | None], *, fail: bool = False) -> Any:
    """An engine constructor that records ``PYTHONPATH`` as the constructor sees it."""

    def start(**_kwargs: Any) -> FakeEngine:
        seen.append(os.environ.get("PYTHONPATH"))
        if fail:
            raise RuntimeError("engine failed to start")
        return FakeEngine()

    return start


def test_plain_sglang_run_does_not_resolve_a_draft_revision(monkeypatch) -> None:
    calls: list[str] = []

    def resolve(model_id: str, *candidates: str | None) -> str:
        calls.append(model_id)
        return candidates[0] or "target-revision"

    class Loaded:
        draft_revision = ""

    monkeypatch.setattr(sglang_generator, "resolve_revision", resolve)
    monkeypatch.setattr(sglang_generator.SGLangGenerator, "load", lambda _request: Loaded())
    sglang_generator.provide(_request("LiquidAI/LFM2.5-2.6B@sglang"))

    assert calls == ["LiquidAI/LFM2.5-2.6B"]


def test_a_failed_target_revision_lookup_never_starts_the_engine(monkeypatch) -> None:
    started: list[FakeEngine] = []

    def fail_revision_lookup(_model_id: str, *_revisions: str | None) -> str:
        raise RuntimeError("Hub unavailable")

    def start_engine(**_kwargs: Any) -> FakeEngine:
        started.append(FakeEngine())
        return started[-1]

    monkeypatch.setattr(sglang_generator, "resolve_revision", fail_revision_lookup)
    _install_engine(monkeypatch, start_engine)

    with pytest.raises(RuntimeError, match="Hub unavailable"):
        sglang_generator.provide(_request("LiquidAI/LFM2.5-2.6B@sglang"))

    assert started == []


def test_chat_encoder_uses_the_auto_tokenizer_and_chat_template_without_loading_weights(
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class Tokenizer:
        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            calls.append({"messages": messages, **kwargs})
            return "rendered prompt"

        def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
            calls.append({"text": text, **kwargs})
            return {"input_ids": [11, 12]}

    class Loader:
        @staticmethod
        def from_pretrained(model_id: str, *, revision: str | None) -> Tokenizer:
            calls.append({"model_id": model_id, "revision": revision})
            return Tokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoProcessor=Loader, AutoTokenizer=Loader),
    )

    encode = sglang_generator._chat_encoder(_request("custom/model"))

    assert encode("Is this land use?") == [11, 12]
    assert calls[0] == {"model_id": "custom/model", "revision": None}
    assert calls[1] == {
        "messages": [{"role": "user", "content": "Is this land use?"}],
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    assert calls[2] == {"text": "rendered prompt", "add_special_tokens": False}


def test_vision_chat_encoder_uses_the_processor_s_tokenizer_and_typed_turn(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    class Tokenizer:
        def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
            calls.append({"text": text, **kwargs})
            return {"input_ids": [21]}

    class Processor:
        tokenizer = Tokenizer()

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            calls.append({"messages": messages, **kwargs})
            return "vision prompt"

    class Loader:
        @staticmethod
        def from_pretrained(model_id: str, *, revision: str | None) -> Processor:
            calls.append({"model_id": model_id, "revision": revision})
            return Processor()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoProcessor=Loader, AutoTokenizer=Loader),
    )

    encode = sglang_generator._chat_encoder(_request("LiquidAI/LFM2.5-VL-3B"))

    assert encode("Is this land use?") == [21]
    assert calls[1] == {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Is this land use?"}],
            }
        ],
        "tokenize": False,
        "add_generation_prompt": True,
    }


class _Child:
    def __init__(self, *, exits: bool) -> None:
        self.exits = exits
        self.timeouts: list[float] = []

    def join(self, timeout: float) -> None:
        self.timeouts.append(timeout)

    def is_alive(self) -> bool:
        return not self.exits


def test_close_waits_for_the_engine_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    from landuse_relevance_bench.adapters import sglang_generator

    child = _Child(exits=True)
    monkeypatch.setattr(sglang_generator.multiprocessing, "active_children", lambda: [child])
    engine = FakeEngine()
    SGLangGenerator(engine, lambda text: [1], 8).close()

    assert engine.stopped
    assert len(child.timeouts) == 1 and child.timeouts[0] > 0


def test_wait_for_children_shares_one_deadline_and_reports_survivors() -> None:
    from landuse_relevance_bench.adapters.sglang_generator import wait_for_children

    stuck, done = _Child(exits=False), _Child(exits=True)
    alive = wait_for_children([stuck, done], timeout_seconds=0.0)

    assert alive == [stuck]
    assert stuck.timeouts == [0.0] and done.timeouts == [0.0]

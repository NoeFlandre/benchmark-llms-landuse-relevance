import os
import runpy
import sys
import types
from pathlib import Path

import pytest

from landuse_relevance_bench.adapters import sglang_compat
from landuse_relevance_bench.adapters.sglang_compat import (
    SITE_DIR,
    forward_to_language_model,
    patch_lfm2_vl,
    with_site_dir,
)


class _Inner:
    lm_head = object()

    def set_dflash_layers_to_capture(self, layers: list[int]) -> tuple[str, list[int]]:
        return ("captured", layers)

    def get_embed_and_head(self) -> str:
        return "embed-and-head"

    def set_embed_and_head(self, embed: str, head: str) -> tuple[str, str]:
        return (embed, head)


def _wrapper_class() -> type:
    class Wrapper:
        def __init__(self) -> None:
            self.language_model = _Inner()

    return Wrapper


def test_wrapper_exposes_the_inner_head_and_hooks() -> None:
    wrapper = _wrapper_class()
    forward_to_language_model(wrapper)
    model = wrapper()
    assert model.lm_head is _Inner.lm_head
    assert model.set_dflash_layers_to_capture([1, 2]) == ("captured", [1, 2])
    assert model.get_embed_and_head() == "embed-and-head"
    assert model.set_embed_and_head("e", head="h") == ("e", "h")


def test_existing_attributes_are_left_alone() -> None:
    wrapper = _wrapper_class()
    own_head = object()
    wrapper.lm_head = own_head
    forward_to_language_model(wrapper)
    assert wrapper().lm_head is own_head


def test_the_dspark_hook_is_not_invented() -> None:
    # SGLang prefers set_dspark_* when present; the inner LFM2 model only has dflash.
    wrapper = _wrapper_class()
    forward_to_language_model(wrapper)
    assert not hasattr(wrapper, "set_dspark_layers_to_capture")


def test_patch_is_a_no_op_without_sglang(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sglang", None)
    patch_lfm2_vl()


def test_patch_targets_the_lfm2_vl_class(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = _wrapper_class()
    models = types.ModuleType("sglang.srt.models")
    models.lfm2_vl = types.SimpleNamespace(Lfm2VlForConditionalGeneration=wrapper)  # type: ignore[attr-defined]
    for name in ("sglang", "sglang.srt"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "sglang.srt.models", models)
    monkeypatch.setitem(sys.modules, "sglang.srt.models.lfm2_vl", models.lfm2_vl)  # type: ignore[attr-defined]
    patch_lfm2_vl()
    assert wrapper().lm_head is _Inner.lm_head


def test_site_dir_is_prepended_once() -> None:
    once = with_site_dir({"PYTHONPATH": f"/a{os.pathsep}/b", "OTHER": "x"})
    twice = with_site_dir(once)
    assert once == twice
    assert once["PYTHONPATH"].split(os.pathsep) == [str(SITE_DIR), "/a", "/b"]
    assert once["OTHER"] == "x"


def test_site_dir_without_existing_path() -> None:
    assert with_site_dir({})["PYTHONPATH"] == str(SITE_DIR)


def test_site_dir_holds_the_shim() -> None:
    assert (SITE_DIR / "sitecustomize.py").is_file()
    assert Path(sglang_compat.__file__).parent == SITE_DIR.parent


def test_sitecustomize_invokes_the_compatibility_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def record_patch() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(sglang_compat, "patch_lfm2_vl", record_patch)
    runpy.run_path(str(SITE_DIR / "sitecustomize.py"))
    assert calls == 1

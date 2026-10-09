import pytest

from landuse_relevance_bench.adapters import optional_import
from landuse_relevance_bench.adapters.optional_import import require


def _fail_import_with(monkeypatch: pytest.MonkeyPatch, error: ImportError) -> None:
    """Make every import raise ``error``, simulating a broken installed runtime."""

    def fake_import(name: str) -> object:
        raise error

    monkeypatch.setattr(optional_import.importlib, "import_module", fake_import)


def test_require_returns_the_imported_module() -> None:
    import json

    assert require("json", "inference") is json


def test_require_names_the_missing_extra() -> None:
    with pytest.raises(ImportError, match=r"not_a_real_module.*\[gguf\]") as caught:
        require("not_a_real_module", "gguf")

    assert isinstance(caught.value.__cause__, ImportError)


def test_require_names_the_extra_for_a_genuinely_absent_runtime() -> None:
    with pytest.raises(ImportError, match=r"not_a_real_module.*\[gguf\]") as caught:
        require("not_a_real_module", "gguf")

    assert isinstance(caught.value.__cause__, ModuleNotFoundError)
    assert caught.value.__cause__.name == "not_a_real_module"


def test_installed_runtime_raising_plain_import_error_propagates_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ImportError("cannot import name 'Llama' from 'llama_cpp'")
    _fail_import_with(monkeypatch, original)

    with pytest.raises(ImportError) as caught:
        require("llama_cpp", "gguf")

    assert caught.value is original
    assert "not installed" not in str(caught.value)


def test_missing_transitive_dependency_propagates_with_its_module_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ModuleNotFoundError("No module named 'sentencepiece'", name="sentencepiece")
    _fail_import_with(monkeypatch, original)

    with pytest.raises(ModuleNotFoundError) as caught:
        require("llama_cpp", "gguf")

    assert caught.value is original
    assert caught.value.name == "sentencepiece"
    assert "not installed" not in str(caught.value)


def test_missing_internal_submodule_is_not_mislabelled_as_absent_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ModuleNotFoundError(
        "No module named 'landuse_relevance_bench.gone'",
        name="landuse_relevance_bench.gone",
    )
    _fail_import_with(monkeypatch, original)

    with pytest.raises(ModuleNotFoundError) as caught:
        require("landuse_relevance_bench", "gguf")

    assert caught.value is original
    assert "not installed" not in str(caught.value)


def test_missing_dotted_submodule_request_is_not_mislabelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ModuleNotFoundError("No module named 'pkg.sub'", name="pkg.sub")
    _fail_import_with(monkeypatch, original)

    with pytest.raises(ModuleNotFoundError) as caught:
        require("pkg.sub", "gguf")

    assert caught.value is original


def test_module_not_found_without_a_name_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ModuleNotFoundError("broken")
    _fail_import_with(monkeypatch, original)

    with pytest.raises(ModuleNotFoundError) as caught:
        require("llama_cpp", "gguf")

    assert caught.value is original

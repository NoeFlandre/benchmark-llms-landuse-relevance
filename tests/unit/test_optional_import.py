import pytest

from landuse_relevance_bench.adapters.optional_import import require


def test_require_returns_the_imported_module() -> None:
    import json

    assert require("json", "inference") is json


def test_require_names_the_missing_extra() -> None:
    with pytest.raises(ImportError, match=r"not_a_real_module.*\[gguf\]") as caught:
        require("not_a_real_module", "gguf")

    assert isinstance(caught.value.__cause__, ImportError)

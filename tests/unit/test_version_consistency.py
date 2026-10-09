"""The package version is declared in three places; they must agree."""

import re
import tomllib
from pathlib import Path

from landuse_relevance_bench import __version__

ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_init_and_citation_versions_agree() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*['\"]?([^'\"\s]+)", citation, re.MULTILINE)

    assert match is not None
    assert pyproject["project"]["version"] == __version__ == match.group(1)


def test_package_ships_py_typed_marker() -> None:
    marker = ROOT / "src" / "landuse_relevance_bench" / "py.typed"

    assert marker.is_file()

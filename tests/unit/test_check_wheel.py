"""The built wheel, not the source tree, must ship the py.typed marker."""

import zipfile
from pathlib import Path

import pytest

from scripts import check_wheel


def _write_wheel(path: Path, members: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(member, "")
    return path


def test_wheel_containing_the_marker_passes(tmp_path: Path) -> None:
    wheel = _write_wheel(
        tmp_path / "landuse_relevance_bench-0.2.0-py3-none-any.whl",
        [
            "landuse_relevance_bench/__init__.py",
            "landuse_relevance_bench/py.typed",
            "landuse_relevance_bench-0.2.0.dist-info/METADATA",
        ],
    )

    check_wheel.require_marker(wheel)


def test_wheel_missing_the_marker_fails_even_if_the_source_has_it(tmp_path: Path) -> None:
    wheel = _write_wheel(
        tmp_path / "landuse_relevance_bench-0.2.0-py3-none-any.whl",
        [
            "landuse_relevance_bench/__init__.py",
            "landuse_relevance_bench-0.2.0.dist-info/METADATA",
        ],
    )

    with pytest.raises(ValueError, match=r"landuse_relevance_bench/py\.typed"):
        check_wheel.require_marker(wheel)


def test_marker_at_the_wrong_location_does_not_count(tmp_path: Path) -> None:
    wheel = _write_wheel(
        tmp_path / "landuse_relevance_bench-0.2.0-py3-none-any.whl",
        ["py.typed", "other_package/py.typed"],
    )

    with pytest.raises(ValueError, match=r"landuse_relevance_bench/py\.typed"):
        check_wheel.require_marker(wheel)


def test_only_one_wheel_is_accepted(tmp_path: Path) -> None:
    _write_wheel(tmp_path / "a-0.1-py3-none-any.whl", [])
    _write_wheel(tmp_path / "b-0.1-py3-none-any.whl", [])

    with pytest.raises(ValueError, match="exactly one wheel"):
        check_wheel.find_wheel(tmp_path)

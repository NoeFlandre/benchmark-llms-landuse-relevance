import json
from pathlib import Path

import pytest

from scripts import crap


def test_source_files_includes_every_python_module_in_a_target(tmp_path: Path) -> None:
    package = tmp_path / "package"
    (package / "nested").mkdir(parents=True)
    first = package / "first.py"
    second = package / "nested" / "second.py"
    first.write_text("def first():\n    return 1\n", encoding="utf-8")
    second.write_text("def second():\n    return 2\n", encoding="utf-8")

    assert crap.source_files(package) == {str(first), str(second)}


def test_target_coverage_reports_modules_missing_from_coverage(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    first = package / "first.py"
    second = package / "second.py"
    first.write_text("def first():\n    return 1\n", encoding="utf-8")
    second.write_text("def second():\n    return 2\n", encoding="utf-8")

    assert crap.missing_coverage_files(package, {str(first)}) == [str(second)]


def test_limit_pairs_parse_target_and_score() -> None:
    assert crap.parse_limit("src/package/domain=8") == ("src/package/domain", 8.0)


def test_allowlist_requires_review_reason_and_test_reference(tmp_path: Path) -> None:
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "src/package/module.py:render": {
                    "reason": "stable renderer",
                    "tests": "tests/unit/test_render.py",
                }
            }
        ),
        encoding="utf-8",
    )

    assert crap.load_allowlist(allowlist) == {
        "src/package/module.py:render": {
            "reason": "stable renderer",
            "tests": "tests/unit/test_render.py",
        }
    }


def test_allowlisted_crap_outlier_is_reported_and_consumed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    key = "src/package/module.py:render"
    allowlist = crap.CrapAllowlist({key: {"reason": "stable renderer", "tests": "test_render"}})

    assert crap._print_scores(
        "src/package",
        15,
        100,
        [(key, 20, 1.0, 20.0)],
        allowlist,
    )
    assert allowlist.used == {key}
    assert "approved exception" in capsys.readouterr().out


def test_stale_allowlist_entries_fail_the_gate(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(crap, "_gate_target", lambda *args, **kwargs: True)
    allowlist = crap.CrapAllowlist(
        {"src/package/module.py:removed": {"reason": "old", "tests": "old"}}
    )

    assert crap._run_gates([("src/package", 15)], [], Path("coverage.json"), allowlist) == 1
    assert "stale CRAP allowlist entries" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["src/package/domain", "=8", "src/package/domain=bad"])
def test_limit_pairs_reject_missing_targets_or_invalid_scores(value: str) -> None:
    with pytest.raises(ValueError, match="TARGET=MAX"):
        crap.parse_limit(value)


def test_coverage_percent_counts_lines_and_branches(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("def run():\n    return True\n", encoding="utf-8")
    report = tmp_path / "coverage.json"
    report.write_text(
        json.dumps(
            {
                "files": {
                    str(module): {
                        "summary": {
                            "covered_lines": 2,
                            "num_statements": 2,
                            "covered_branches": 1,
                            "num_branches": 2,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert crap.coverage_percent(module, report) == pytest.approx(100 * 3 / 4)


def test_coverage_percent_fails_when_a_source_module_is_absent(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("def run():\n    return True\n", encoding="utf-8")
    report = tmp_path / "coverage.json"
    report.write_text('{"files": {}}', encoding="utf-8")

    with pytest.raises(ValueError, match="absent from coverage"):
        crap.coverage_percent(module, report)

"""The CRAP gate, exercised through its command line over a synthetic package."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import crap

# Line numbers matter: the coverage reports below name these statements explicitly.
MODULE = """\
def flat(value):
    return value


def branchy(value):
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    if value == 0:
        return "zero"
    return "unreachable"
"""
FLAT_LINES = [1, 2]
BRANCHY_LINES = [5, 6, 7, 8, 9, 10, 11, 12]


@pytest.fixture
def package(tmp_path: Path) -> Path:
    directory = tmp_path / "package"
    directory.mkdir()
    (directory / "branchy.py").write_text(MODULE, encoding="utf-8")
    return directory


def _report(tmp_path: Path, module: Path, *, executed: list[int], missing: list[int]) -> Path:
    """A coverage.json with one measured module and no branch data."""
    path = tmp_path / "coverage.json"
    payload = {
        "files": {
            str(module): {
                "executed_lines": executed,
                "missing_lines": missing,
                "summary": {
                    "covered_lines": len(executed),
                    "num_statements": len(executed) + len(missing),
                    "covered_branches": 0,
                    "num_branches": 0,
                },
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fully_covered(tmp_path: Path, package: Path) -> Path:
    return _report(
        tmp_path,
        package / "branchy.py",
        executed=FLAT_LINES + BRANCHY_LINES,
        missing=[],
    )


def _run(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> int:
    monkeypatch.setattr(sys, "argv", ["crap.py", *arguments])
    return crap.main()


def test_gate_passes_when_every_function_is_within_its_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    package: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _fully_covered(tmp_path, package)

    status = _run(
        monkeypatch,
        "--limit",
        f"{package}=15",
        "--coverage-json",
        str(report),
    )

    assert status == 0
    assert "all 2 functions are at or below a CRAP score of 15" in capsys.readouterr().out


def test_gate_names_the_uncovered_function_that_exceeds_its_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    package: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _report(
        tmp_path,
        package / "branchy.py",
        executed=FLAT_LINES,
        missing=BRANCHY_LINES,
    )

    status = _run(
        monkeypatch,
        "--limit",
        f"{package}=15",
        "--coverage-json",
        str(report),
    )
    output = capsys.readouterr().out

    assert status == 1
    assert "CRAP above 15:" in output
    assert "branchy.py:branchy" in output
    assert "branchy.py:flat" not in output.split("CRAP above 15:")[1]


def test_full_coverage_target_fails_below_one_hundred_percent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    package: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _report(
        tmp_path,
        package / "branchy.py",
        executed=FLAT_LINES + BRANCHY_LINES[:-1],
        missing=BRANCHY_LINES[-1:],
    )

    status = _run(
        monkeypatch,
        "--limit",
        f"{package}=15",
        "--full-coverage",
        str(package),
        "--coverage-json",
        str(report),
    )

    assert status == 1
    assert "required 100.00% line and branch coverage" in capsys.readouterr().out


def test_full_coverage_target_passes_at_one_hundred_percent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    package: Path,
) -> None:
    report = _fully_covered(tmp_path, package)

    status = _run(
        monkeypatch,
        "--limit",
        f"{package}=15",
        "--full-coverage",
        str(package),
        "--coverage-json",
        str(report),
    )

    assert status == 0


def test_missing_target_is_reported_rather_than_measured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    package: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _fully_covered(tmp_path, package)

    status = _run(
        monkeypatch,
        "--limit",
        f"{tmp_path / 'absent'}=15",
        "--coverage-json",
        str(report),
    )

    assert status == 1
    assert "target does not exist" in capsys.readouterr().out


def test_module_absent_from_coverage_fails_the_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    package: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _report(tmp_path, tmp_path / "elsewhere.py", executed=[1], missing=[])

    status = _run(
        monkeypatch,
        "--limit",
        f"{package}=15",
        "--coverage-json",
        str(report),
    )

    assert status == 1
    assert "source files absent from coverage" in capsys.readouterr().out


def test_target_without_functions_is_an_error_not_a_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    constants = tmp_path / "constants"
    constants.mkdir()
    module = constants / "settings.py"
    module.write_text("LIMIT = 3\n", encoding="utf-8")
    report = _report(tmp_path, module, executed=[1], missing=[])

    status = _run(
        monkeypatch,
        "--limit",
        f"{constants}=15",
        "--coverage-json",
        str(report),
    )

    assert status == 1
    assert "no functions measured" in capsys.readouterr().out


def test_allowlisted_outlier_is_approved_with_its_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    package: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _report(
        tmp_path,
        package / "branchy.py",
        executed=FLAT_LINES,
        missing=BRANCHY_LINES,
    )
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                f"{package / 'branchy.py'}:branchy": {
                    "reason": "dispatch table is fixed",
                    "tests": "tests/scripts/test_crap_gate.py",
                }
            }
        ),
        encoding="utf-8",
    )

    status = _run(
        monkeypatch,
        "--limit",
        f"{package}=15",
        "--coverage-json",
        str(report),
        "--allowlist",
        str(allowlist),
    )
    output = capsys.readouterr().out

    assert status == 0
    assert "approved exception: " in output
    assert "dispatch table is fixed" in output


def test_allowlist_entry_for_a_function_within_limit_is_stale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    package: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _fully_covered(tmp_path, package)
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                f"{package / 'branchy.py'}:branchy": {
                    "reason": "no longer needed",
                    "tests": "tests/scripts/test_crap_gate.py",
                }
            }
        ),
        encoding="utf-8",
    )

    status = _run(
        monkeypatch,
        "--limit",
        f"{package}=15",
        "--coverage-json",
        str(report),
        "--allowlist",
        str(allowlist),
    )

    assert status == 1
    assert "stale CRAP allowlist entries" in capsys.readouterr().out


def test_malformed_allowlist_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    package: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(json.dumps({"no-colon": {"reason": "x", "tests": "y"}}), encoding="utf-8")

    with pytest.raises(SystemExit) as exit_info:
        _run(
            monkeypatch,
            "--limit",
            f"{package}=15",
            "--allowlist",
            str(allowlist),
        )

    assert exit_info.value.code == 2
    assert "invalid CRAP allowlist" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["package", "=15", "package=tall"])
def test_malformed_limit_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    value: str,
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _run(monkeypatch, "--limit", value)

    assert exit_info.value.code == 2
    assert "expected TARGET=MAX" in capsys.readouterr().err


def test_limit_and_legacy_target_cannot_be_combined(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _run(monkeypatch, "--limit", "package=15", "--target", "package")

    assert exit_info.value.code == 2
    assert "use either --limit or --target" in capsys.readouterr().err


def test_legacy_target_and_maximum_gate_a_single_layer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    package: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _fully_covered(tmp_path, package)

    status = _run(
        monkeypatch,
        "--target",
        str(package),
        "--max",
        "15",
        "--coverage-json",
        str(report),
    )

    assert status == 0
    assert "CRAP limit 15" in capsys.readouterr().out

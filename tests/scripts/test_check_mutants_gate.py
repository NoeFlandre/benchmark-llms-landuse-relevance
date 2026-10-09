"""The mutation gate's command line, with mutmut's output supplied by a fake process."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import check_mutants

SURVIVOR = "landuse_relevance_bench.domain.labels.x__mutmut_1"
KILLED = "landuse_relevance_bench.domain.labels.x__mutmut_2"
OTHER_KILLED = "landuse_relevance_bench.domain.metrics.y__mutmut_3"


def _mutmut(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    *,
    returncode: int = 0,
    stderr: str = "",
) -> None:
    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[1:] == ["-m", "mutmut", "results", "--all", "true"]
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    monkeypatch.setattr(check_mutants.subprocess, "run", fake_run)


def _allowlist(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "mutation-allowlist.txt"
    path.write_text(content, encoding="utf-8")
    return path


def _run(monkeypatch: pytest.MonkeyPatch, allowlist: Path) -> int:
    monkeypatch.setattr(sys, "argv", ["check_mutants.py", "--allowlist", str(allowlist)])
    return check_mutants.main()


def test_clean_run_passes_and_names_each_reviewed_equivalent_mutant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _mutmut(
        monkeypatch,
        f"{KILLED}: killed\n{OTHER_KILLED}: killed\n{SURVIVOR}: survived\n",
    )
    allowlist = _allowlist(tmp_path, f"{SURVIVOR}: the boundary is unreachable\n")

    status = _run(monkeypatch, allowlist)
    output = capsys.readouterr().out

    assert status == 0
    assert "mutmut status counts: killed=2, survived=1" in output
    assert f"reviewed equivalent: {SURVIVOR} — the boundary is unreachable" in output
    assert "mutation gate passed: 3 exact mutant id(s) checked" in output


def test_unreviewed_survivor_fails_the_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _mutmut(monkeypatch, f"{KILLED}: killed\n{SURVIVOR}: survived\n")

    status = _run(monkeypatch, _allowlist(tmp_path, ""))
    output = capsys.readouterr().out

    assert status == 1
    assert f"unreviewed survivor: {SURVIVOR}" in output
    assert "mutation gate failed with 1 issue(s)" in output


def test_allowlist_entry_without_a_surviving_mutant_is_stale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _mutmut(monkeypatch, f"{KILLED}: killed\n")

    status = _run(monkeypatch, _allowlist(tmp_path, f"{SURVIVOR}: once equivalent\n"))
    output = capsys.readouterr().out

    assert status == 1
    assert f"stale allowlist entry: {SURVIVOR}" in output


def test_unresolved_mutant_status_is_never_a_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _mutmut(monkeypatch, f"{KILLED}: killed\n{SURVIVOR}: timeout\n")

    status = _run(monkeypatch, _allowlist(tmp_path, ""))
    output = capsys.readouterr().out

    assert status == 1
    assert f"unresolved mutant status timeout: {SURVIVOR}" in output


def test_mutmut_failure_is_reported_as_a_gate_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _mutmut(monkeypatch, "", returncode=3, stderr="cache is corrupt\n")

    status = _run(monkeypatch, _allowlist(tmp_path, ""))

    assert status == 1
    assert (
        "mutation gate error: mutmut results failed with exit code 3: cache is corrupt"
        in capsys.readouterr().out
    )


def test_empty_mutmut_output_is_reported_as_a_gate_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _mutmut(monkeypatch, "\n")

    status = _run(monkeypatch, _allowlist(tmp_path, ""))

    assert status == 1
    assert "mutation gate error: mutmut results were empty" in capsys.readouterr().out


def test_missing_allowlist_is_reported_as_a_gate_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _mutmut(monkeypatch, f"{KILLED}: killed\n")

    status = _run(monkeypatch, tmp_path / "absent.txt")

    assert status == 1
    assert "mutation gate error:" in capsys.readouterr().out


def test_allowlist_line_without_a_reason_is_reported_as_a_gate_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _mutmut(monkeypatch, f"{KILLED}: killed\n")

    status = _run(monkeypatch, _allowlist(tmp_path, f"{SURVIVOR}:\n"))

    assert status == 1
    assert "allowlist line 1 needs a mutant id and reason" in capsys.readouterr().out

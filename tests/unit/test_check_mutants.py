"""The mutation gate refuses to pass a run that produced no results."""

import subprocess
from pathlib import Path

import pytest
from scripts.check_mutants import MutationRunError, survivors


def test_a_crashed_run_with_no_results_is_not_a_clean_pass(tmp_path: Path) -> None:
    with pytest.raises(MutationRunError, match="crash"):
        survivors(tmp_path)


def test_results_without_a_single_killed_mutant_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mutants").mkdir()
    (tmp_path / "mutants" / "x.meta").write_text("{}", encoding="utf-8")
    listing = subprocess.CompletedProcess(
        [], 0, stdout="    domain.x__mutmut_1: not checked\n", stderr=""
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: listing)

    with pytest.raises(MutationRunError, match="killed no mutants"):
        survivors(tmp_path)

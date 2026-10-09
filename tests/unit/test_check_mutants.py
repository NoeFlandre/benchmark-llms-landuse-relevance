import pytest

from scripts import check_mutants


def test_mutmut_results_parse_exact_ids_and_statuses() -> None:
    parsed = check_mutants.parse_results(
        "package.domain.metrics.evaluate__mutant_a: killed\n"
        "package.domain.parsing.parse_label__mutant_b: survived\n"
    )
    assert parsed == {
        "package.domain.metrics.evaluate__mutant_a": "killed",
        "package.domain.parsing.parse_label__mutant_b": "survived",
    }


@pytest.mark.parametrize("output", ["", "summary changed", "one: unknown status"])
def test_empty_or_unrecognized_mutmut_output_is_an_error(output: str) -> None:
    with pytest.raises(ValueError, match="unparsable"):
        check_mutants.parse_results(output)


def test_results_without_a_killed_mutant_are_not_a_clean_pass() -> None:
    with pytest.raises(ValueError, match="killed no mutants"):
        check_mutants.parse_results("domain.x__mutmut_1: not checked\n")


def test_allowlist_requires_one_reason_per_unique_mutant() -> None:
    assert check_mutants.parse_allowlist(
        "# Exact reviewed equivalents\nmodule.function__mutant: equivalent behavior\n"
    ) == {"module.function__mutant": "equivalent behavior"}
    with pytest.raises(ValueError, match="reason"):
        check_mutants.parse_allowlist("module.function__mutant\n")


def test_gate_reports_new_survivors_stale_entries_and_other_unresolved_states() -> None:
    actual = {
        "module.function__allowed": "survived",
        "module.function__new": "survived",
        "module.function__timeout": "timeout",
        "module.function__killed": "killed",
    }
    errors = check_mutants.validate_results(
        actual,
        {"module.function__allowed": "documented equivalence", "module.function__stale": "old"},
    )
    assert "module.function__new" in "\n".join(errors)
    assert "module.function__stale" in "\n".join(errors)
    assert "module.function__timeout" in "\n".join(errors)


def test_read_mutmut_results_requests_all_mutant_ids(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        assert command[-3:] == ["results", "--all", "true"]
        assert kwargs["check"] is False
        return type("Completed", (), {"returncode": 0, "stdout": "id: killed\n", "stderr": ""})()

    monkeypatch.setattr(check_mutants.subprocess, "run", fake_run)

    assert check_mutants.read_results() == {"id": "killed"}


def test_read_mutmut_results_fails_when_the_runner_exits_non_zero(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        return type("Completed", (), {"returncode": 1, "stdout": "", "stderr": "boom"})()

    monkeypatch.setattr(check_mutants.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="exit code 1: boom"):
        check_mutants.read_results()


def test_main_exits_non_zero_when_mutmut_results_cannot_be_read(
    monkeypatch, tmp_path, capsys
) -> None:
    """A failed mutmut runner must fail the gate, never pass it silently."""
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("module.function__allowed: equivalent behavior\n", encoding="utf-8")

    def failing_read_results(root=None):
        raise RuntimeError("mutmut results failed with exit code 1: stats failed")

    monkeypatch.setattr(check_mutants, "read_results", failing_read_results)
    monkeypatch.setattr("sys.argv", ["check_mutants.py", "--allowlist", str(allowlist)])

    assert check_mutants.main() == 1
    assert "mutation gate error: mutmut results failed" in capsys.readouterr().out


def test_gate_script_runs_main_and_fails_on_an_invalid_allowlist(tmp_path) -> None:
    """Regression: without a ``__main__`` guard the script did nothing and exited 0."""
    import subprocess
    import sys
    from pathlib import Path

    script = Path(check_mutants.__file__)
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("a-mutant-id-without-a-reason\n", encoding="utf-8")
    done = subprocess.run(  # noqa: S603 - runs this repository's own script
        [sys.executable, str(script), "--allowlist", str(allowlist)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert done.returncode != 0
    assert "mutation gate error: allowlist line 1" in done.stdout

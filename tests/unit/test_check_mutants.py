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
        assert command[-2:] == ["results", "--all"]
        assert kwargs["check"] is False
        return type("Completed", (), {"returncode": 0, "stdout": "id: killed\n", "stderr": ""})()

    monkeypatch.setattr(check_mutants.subprocess, "run", fake_run)

    assert check_mutants.read_results() == {"id": "killed"}

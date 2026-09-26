from factories import make_result
from landuse_relevance_bench.domain.selection import check_comparable, select_run_names


def test_run_selection_filters_by_regex_and_runtime() -> None:
    names = ("LiquidAI/LFM2.5-2.6B@sglang", "LiquidAI/LFM2.5-350M", "other/model")
    assert select_run_names(names, only="2.6B", runtimes=("sglang",)) == [names[0]]


def test_run_selection_reports_an_invalid_regex_as_value_error() -> None:
    try:
        select_run_names(("model",), only="(")
    except ValueError as exc:
        assert "invalid --only regex" in str(exc)
    else:
        raise AssertionError("invalid regex was accepted")


def test_run_selection_defaults_unknown_names_to_transformers() -> None:
    assert select_run_names(("other/model",), runtimes=("transformers",)) == ["other/model"]


def test_comparability_accepts_no_runs() -> None:
    report = check_comparable([])
    assert report.comparable
    assert report.reference_name is None


def test_comparable_runs_share_the_required_metadata() -> None:
    report = check_comparable([make_result("a/model"), make_result("b/model")])
    assert report.comparable
    assert report.differences == ()


def test_mixed_metadata_report_names_each_run_and_difference() -> None:
    baseline = make_result("a/model")
    changed = make_result("b/model", max_new_tokens=16, decoding="sample")
    report = check_comparable([baseline, changed])
    assert not report.comparable
    assert {(d.run_name, d.field, d.actual) for d in report.differences} == {
        ("b/model", "max_new_tokens", 16),
        ("b/model", "decoding", "sample"),
    }

"""End-to-end CLI acceptance scenarios with a deterministic local generator."""

import csv
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from pytest_bdd import given, parsers, scenarios, then, when
from typer.testing import CliRunner

from factories import make_result
from landuse_relevance_bench import application, cli
from landuse_relevance_bench.adapters import providers
from landuse_relevance_bench.adapters.benchmark_csv import load_benchmark
from landuse_relevance_bench.adapters.results_store import write_run

scenarios("features/cli_workflow.feature")


@dataclass(frozen=True)
class CliHarness:
    benchmark: Path
    prompt: Path
    results: Path
    runner: CliRunner
    calls: list[str]


class OracleGenerator:
    def __init__(self, labels: dict[str, str]) -> None:
        self.labels = labels

    def generate(self, prompts: Sequence[str]) -> list[str]:
        return [self.labels[self._sentence(prompt)] for prompt in prompts]

    @staticmethod
    def _sentence(prompt: str) -> str:
        return prompt.rsplit("TARGET SENTENCE:", 1)[1].strip()


@given("a ready CLI harness", target_fixture="harness")
def _cli_harness(monkeypatch, real_benchmark_path: Path, real_prompt_path: Path, tmp_path: Path):
    labels = {item.sentence: item.label.value for item in load_benchmark(real_benchmark_path)}
    calls: list[str] = []
    results = tmp_path / "cli-results"
    runner = CliRunner()

    def provide(request):
        calls.append(request.name)
        return OracleGenerator(labels), "stub-revision"

    monkeypatch.setattr(providers, "generator_provider", lambda: provide)
    monkeypatch.setattr(cli, "model_ids", lambda: ("stub/first", "stub/second"))
    monkeypatch.setattr(application, "model_ids", lambda: ("stub/first", "stub/second"))
    return CliHarness(
        benchmark=real_benchmark_path.parent.parent,
        prompt=real_prompt_path,
        results=results,
        runner=runner,
        calls=calls,
    )


def _invoke(harness: CliHarness, arguments: list[str]):
    runner = harness.runner
    result = runner.invoke(cli.app, arguments)
    assert result.exit_code == 0, result.output
    return result


@when(parsers.parse('I run "{model}" through lrb'), target_fixture="cli_run_result")
def _cli_run(harness: CliHarness, model: str):
    return _invoke(
        harness,
        [
            "run",
            model,
            "--data-root",
            str(harness.benchmark),
            "--language",
            "en",
            "--prompt",
            str(harness.prompt),
            "--out",
            str(harness.results),
        ],
    )


@when("I report through lrb", target_fixture="cli_report_result")
def _cli_report(harness: CliHarness):
    runner = harness.runner
    return runner.invoke(cli.app, ["report", "--results-dir", str(harness.results)])


@then("the leaderboard CSV contains the one-row perfect run")
def _cli_csv_is_perfect(harness: CliHarness, cli_run_result) -> None:
    del cli_run_result
    path = Path(harness.results) / "leaderboard.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["model_id"] == "stub/gold"
    assert float(rows[0]["accuracy"]) == 1.0


@when("I preview publication through lrb", target_fixture="cli_publish_preview")
def _cli_publish_preview(monkeypatch, harness: CliHarness):
    from landuse_relevance_bench.adapters import hf_publish

    monkeypatch.setattr(
        hf_publish,
        "publish_results",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not call the Hub publisher"),
    )
    runner = harness.runner
    result = runner.invoke(
        cli.app,
        [
            "publish",
            "me/benchmark",
            "--data-root",
            str(harness.benchmark),
            "--results-dir",
            str(harness.results),
            "--prompt",
            str(harness.prompt),
            "--language",
            "en",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    return result


@then("the preview lists the run and makes no Hub call")
def _preview_is_offline(harness: CliHarness, cli_publish_preview) -> None:
    assert "stub/gold" in cli_publish_preview.stdout
    assert not (Path(harness.results) / "README.md").exists()


@given(parsers.parse('a completed result for "{model}"'))
def _completed_result(harness: CliHarness, model: str) -> None:
    _invoke(
        harness,
        [
            "run",
            model,
            "--data-root",
            str(harness.benchmark),
            "--language",
            "en",
            "--prompt",
            str(harness.prompt),
            "--out",
            str(harness.results),
        ],
    )
    harness.calls.clear()


@when(parsers.parse('I resume run-all for "{first}" and "{second}"'))
def _resume_run_all(harness: CliHarness, first: str, second: str) -> None:
    _invoke(
        harness,
        [
            "run-all",
            "--data-root",
            str(harness.benchmark),
            "--language",
            "en",
            "--prompt",
            str(harness.prompt),
            "--out",
            str(harness.results),
            "--skip-existing",
            "--only",
            f"{first}|{second}",
        ],
    )


@then(parsers.parse('only "{model}" is generated'))
def _only_missing_is_generated(harness: CliHarness, model: str) -> None:
    assert harness.calls == [model]


@given("a truncated result file")
def _truncated_result(harness: CliHarness) -> None:
    results = Path(harness.results)
    results.mkdir(parents=True, exist_ok=True)
    (results / "broken.json").write_text("{not valid json", encoding="utf-8")


@then("reporting fails and names the truncated file")
def _report_identifies_bad_file(cli_report_result) -> None:
    assert cli_report_result.exit_code != 0
    assert "broken.json" in cli_report_result.output


@given("identical baseline and speculative results")
def _identical_speculative_runs(harness: CliHarness) -> None:
    baseline = make_result("stub/target", run_id="stub/baseline", runtime="sglang")
    speculative = replace(
        baseline,
        metadata=replace(baseline.metadata, run_id="stub/speculative", draft_model_id="stub/draft"),
    )
    write_run(baseline, Path(harness.results))
    write_run(speculative, Path(harness.results))


@then("the report states that the speculative run is lossless")
def _report_is_lossless(cli_report_result) -> None:
    assert cli_report_result.exit_code == 0, cli_report_result.output
    assert "lossless" in cli_report_result.stdout


@given(parsers.parse('stored runs with a mismatched "{setting}" digest'))
def _mixed_runs(harness: CliHarness, setting: str) -> None:
    first = make_result("stub/one")
    second = make_result("stub/two", **{f"{setting}_sha256": "different"})
    write_run(first, Path(harness.results))
    write_run(second, Path(harness.results))


@then("reporting refuses the mixed settings")
def _report_refuses_mixed(cli_report_result) -> None:
    assert cli_report_result.exit_code != 0
    assert "not comparable" in cli_report_result.output or "differs" in cli_report_result.output

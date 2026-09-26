"""Executable acceptance criteria for an end-to-end benchmark run."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from pytest_bdd import given, parsers, scenarios, then, when
from typer.testing import CliRunner

from factories import make_result
from landuse_relevance_bench import cli
from landuse_relevance_bench.adapters import results_store
from landuse_relevance_bench.adapters.benchmark_csv import load_benchmark
from landuse_relevance_bench.adapters.hashing import sha256_of_file, sha256_of_text
from landuse_relevance_bench.adapters.pipeline import RunRequest, execute
from landuse_relevance_bench.adapters.prompt_file import load_prompt
from landuse_relevance_bench.adapters.results_store import (
    leaderboard_rows,
    read_run,
    run_filename,
    write_run,
)
from landuse_relevance_bench.domain.dataset import item_id_for
from landuse_relevance_bench.domain.engine import Generation

scenarios("features/benchmark.feature")
scenarios("features/cli_workflow.feature")

GOLD_MODEL = "stub/gold"
SECOND_MODEL = "stub/second"


class OracleGenerator:
    """Answers with the gold label, recovered from the sentence in the prompt."""

    def __init__(self, gold: dict[str, str]) -> None:
        self._gold = gold

    def generate(self, prompts: Sequence[str]) -> list[str]:
        return [self._gold[self._sentence_of(p)] for p in prompts]

    @staticmethod
    def _sentence_of(prompt: str) -> str:
        return prompt.rsplit("TARGET SENTENCE: ", 1)[1].strip()


class ConstantGenerator:
    def __init__(self, answer: str) -> None:
        self._answer = answer

    def generate(self, prompts: Sequence[str]) -> list[str]:
        return [self._answer] * len(prompts)


class TruncatedGenerator:
    """Always runs out of budget mid-thought, mentioning both verdicts on the way."""

    def __init__(self, partial: str) -> None:
        self._partial = partial

    def generate(self, prompts: Sequence[str]) -> list[Generation]:
        return [Generation(self._partial, truncated=True)] * len(prompts)


@given("the project benchmark of labelled sentences", target_fixture="benchmark")
def _benchmark(real_benchmark_path: Path) -> Path:
    return real_benchmark_path


@given("the project prompt template", target_fixture="prompt")
def _prompt(real_prompt_path: Path) -> Path:
    return real_prompt_path


@given("a model that answers every sentence with its gold label", target_fixture="models")
def _oracle(benchmark: Path) -> dict[str, object]:
    gold = {i.sentence: i.label.value for i in load_benchmark(benchmark)}
    return {GOLD_MODEL: OracleGenerator(gold)}


@given(parsers.parse('a model that always answers "{answer}"'), target_fixture="models")
def _constant(answer: str) -> dict[str, object]:
    return {GOLD_MODEL: ConstantGenerator(answer)}


@given("a model whose answer is cut off by the token budget", target_fixture="models")
def _truncated() -> dict[str, object]:
    return {GOLD_MODEL: TruncatedGenerator('Criteria for "yes": vegetation, terrain')}


@given(parsers.parse('a second model that always answers "{answer}"'), target_fixture="models")
def _second(models: dict[str, object], answer: str) -> dict[str, object]:
    return {**models, SECOND_MODEL: ConstantGenerator(answer)}


@when("I benchmark that model", target_fixture="runs")
@when("I benchmark both models", target_fixture="runs")
def _benchmark_models(models, benchmark: Path, prompt: Path, tmp_path: Path):
    results_dir = tmp_path / "results"
    runs = {
        model_id: execute(
            RunRequest(
                model_id=model_id,
                benchmark_path=benchmark,
                prompt_path=prompt,
                output_dir=results_dir,
                batch_size=8,
            ),
            lambda _request, gen=generator: (gen, "stubrev"),
        )
        for model_id, generator in models.items()
    }
    return {"runs": runs, "dir": results_dir}


@then(parsers.parse("the run scores an accuracy of {value:f}"))
def _accuracy(runs, value: float) -> None:
    assert runs["runs"][GOLD_MODEL].metrics.accuracy == value


@then("no generation is left unparsed")
def _none_unparsed(runs) -> None:
    assert runs["runs"][GOLD_MODEL].metrics.confusion.unparsed == 0


@then("every generation is left unparsed")
def _all_unparsed(runs) -> None:
    assert runs["runs"][GOLD_MODEL].metrics.unparsed_rate == 1.0


@then("the stored result covers every sentence in the benchmark")
def _covers_all(runs, benchmark: Path) -> None:
    items = load_benchmark(benchmark)
    stored = read_run(runs["dir"] / run_filename(GOLD_MODEL))
    assert [p.item_id for p in stored.predictions] == [i.item_id for i in items]
    assert stored.predictions[0].item_id == item_id_for(items[0].sentence)


@then("the run recalls every relevant sentence")
def _full_recall(runs) -> None:
    assert runs["runs"][GOLD_MODEL].metrics.recall == 1.0


@then("the run misses no relevant sentence")
def _no_misses(runs) -> None:
    assert runs["runs"][GOLD_MODEL].metrics.confusion.false_negative == 0


@then("the raw generations are kept in the stored result")
def _raw_kept(runs) -> None:
    stored = read_run(runs["dir"] / run_filename(GOLD_MODEL))
    assert {p.raw_output for p in stored.predictions} == {"I cannot decide"}


@then("the stored result marks every generation as truncated")
def _truncation_recorded(runs) -> None:
    stored = read_run(runs["dir"] / run_filename(GOLD_MODEL))
    assert all(p.truncated for p in stored.predictions)
    assert all(p.predicted is None for p in stored.predictions)


@then("the leaderboard ranks the accurate model first")
def _ranking(runs) -> None:
    rows = leaderboard_rows(list(runs["runs"].values()))
    assert rows[0]["model_id"] == GOLD_MODEL


@then("the leaderboard has one row per model")
def _one_row_each(runs) -> None:
    assert len(leaderboard_rows(list(runs["runs"].values()))) == len(runs["runs"])


@then("the stored result pins the benchmark and prompt digests")
def _digests(runs, benchmark: Path, prompt: Path) -> None:
    metadata = read_run(runs["dir"] / run_filename(GOLD_MODEL)).metadata
    assert metadata.benchmark_sha256 == sha256_of_file(benchmark)
    assert metadata.prompt_sha256 == sha256_of_text(load_prompt(prompt))


@then("the stored result names the model revision that was used")
def _revision(runs) -> None:
    assert read_run(runs["dir"] / run_filename(GOLD_MODEL)).metadata.model_revision == "stubrev"


@given("a ready CLI harness", target_fixture="harness")
def _cli_harness(
    monkeypatch, real_benchmark_path: Path, real_prompt_path: Path, tmp_path: Path
) -> dict[str, object]:
    gold = {item.sentence: item.label.value for item in load_benchmark(real_benchmark_path)}
    calls: list[str] = []
    runner = CliRunner()
    results_dir = tmp_path / "cli-results"

    def provider(request: RunRequest):
        calls.append(request.name)
        return OracleGenerator(gold), "stubrev"

    monkeypatch.setattr(cli, "generator_provider", lambda: provider)
    monkeypatch.setattr(cli, "model_ids", lambda: ("stub/first", "stub/second"))
    monkeypatch.setattr(
        results_store,
        "bootstrap_interval",
        lambda *_args, **_kwargs: (0.0, 0.0),
    )
    return {
        "benchmark": real_benchmark_path,
        "prompt": real_prompt_path,
        "results": results_dir,
        "runner": runner,
        "calls": calls,
    }


def _invoke_cli(harness: dict[str, object], arguments: list[str]):
    runner = harness["runner"]
    assert isinstance(runner, CliRunner)
    result = runner.invoke(cli.app, arguments)
    assert result.exit_code == 0, result.output
    return result


@when(parsers.parse('I run "{model}" through lrb'), target_fixture="cli_run_result")
def _cli_run(harness: dict[str, object], model: str):
    return _invoke_cli(
        harness,
        [
            "run",
            model,
            "--benchmark",
            str(harness["benchmark"]),
            "--prompt",
            str(harness["prompt"]),
            "--out",
            str(harness["results"]),
        ],
    )


@when("I report through lrb", target_fixture="cli_report_result")
def _cli_report(harness: dict[str, object]):
    runner = harness["runner"]
    assert isinstance(runner, CliRunner)
    return runner.invoke(cli.app, ["report", "--results-dir", str(harness["results"])])


@then("the leaderboard CSV contains the one-row perfect run")
def _cli_csv_is_perfect(harness: dict[str, object], cli_run_result) -> None:
    del cli_run_result
    import csv

    with (harness["results"] / "leaderboard.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["model_id"] == "stub/gold"
    assert float(rows[0]["accuracy"]) == 1.0


@when("I preview publication through lrb", target_fixture="cli_publish_preview")
def _cli_publish_preview(monkeypatch, harness: dict[str, object]):
    from landuse_relevance_bench.adapters import hf_publish

    def forbidden_publish(*_args, **_kwargs):
        pytest.fail("dry-run must not call the Hub publisher")

    monkeypatch.setattr(hf_publish, "publish_results", forbidden_publish)
    runner = harness["runner"]
    assert isinstance(runner, CliRunner)
    result = runner.invoke(
        cli.app,
        [
            "publish",
            "me/benchmark",
            "--results-dir",
            str(harness["results"]),
            "--benchmark",
            str(harness["benchmark"]),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    return result


@then("the preview lists the run and makes no Hub call")
def _preview_is_offline(harness: dict[str, object], cli_publish_preview) -> None:
    assert "stub/gold" in cli_publish_preview.stdout
    assert not (harness["results"] / "README.md").exists()


@given(parsers.parse('a completed result for "{model}"'))
def _completed_result(harness: dict[str, object], model: str) -> None:
    _invoke_cli(
        harness,
        [
            "run",
            model,
            "--benchmark",
            str(harness["benchmark"]),
            "--prompt",
            str(harness["prompt"]),
            "--out",
            str(harness["results"]),
        ],
    )
    harness["calls"].clear()


@when(parsers.parse('I resume run-all for "{first}" and "{second}"'))
def _resume_run_all(harness: dict[str, object], first: str, second: str) -> None:
    _invoke_cli(
        harness,
        [
            "run-all",
            "--benchmark",
            str(harness["benchmark"]),
            "--prompt",
            str(harness["prompt"]),
            "--out",
            str(harness["results"]),
            "--skip-existing",
            "--only",
            f"{first}|{second}",
        ],
    )


@then(parsers.parse('only "{model}" is generated'))
def _only_missing_is_generated(harness: dict[str, object], model: str) -> None:
    assert harness["calls"] == [model]


@given("a truncated result file")
def _truncated_result(harness: dict[str, object]) -> None:
    directory = harness["results"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "broken.json").write_text("{not valid json", encoding="utf-8")


@then("reporting fails and names the truncated file")
def _report_identifies_bad_file(cli_report_result) -> None:
    assert cli_report_result.exit_code != 0
    assert "broken.json" in cli_report_result.output


@given("identical baseline and speculative results")
def _identical_speculative_runs(harness: dict[str, object]) -> None:
    baseline = make_result(
        "stub/target",
        run_id="stub/baseline",
        runtime="sglang",
    )
    speculative = replace(
        baseline,
        metadata=replace(
            baseline.metadata,
            run_id="stub/speculative",
            draft_model_id="stub/draft",
        ),
    )
    write_run(baseline, harness["results"])
    write_run(speculative, harness["results"])


@then("the report states that the speculative run is lossless")
def _report_is_lossless(cli_report_result) -> None:
    assert cli_report_result.exit_code == 0, cli_report_result.output
    assert "lossless" in cli_report_result.stdout


@given(parsers.parse('stored runs with a mismatched "{setting}" digest'))
def _mixed_runs(harness: dict[str, object], setting: str) -> None:
    first = make_result("stub/one")
    second = make_result("stub/two", **{f"{setting}_sha256": "different"})
    write_run(first, harness["results"])
    write_run(second, harness["results"])


@then("reporting refuses the mixed settings")
def _report_refuses_mixed(cli_report_result) -> None:
    assert cli_report_result.exit_code != 0
    assert "not comparable" in cli_report_result.output

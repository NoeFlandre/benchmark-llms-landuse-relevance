import json
from collections.abc import Sequence
from pathlib import Path

import typer
from typer.testing import CliRunner, Result

from landuse_relevance_bench import cli
from landuse_relevance_bench.adapters.pipeline import RunRequest

runner = CliRunner()


class AlwaysYes:
    def generate(self, prompts: Sequence[str]) -> Sequence[str]:
        return ["yes"] * len(prompts)


def _fake_provider(_: RunRequest) -> tuple[AlwaysYes, str]:
    return AlwaysYes(), "fakerev"


def _run(model: str, benchmark: Path, prompt: Path, out: Path, *extra: str) -> Result:
    return runner.invoke(
        cli.app,
        [
            "run",
            model,
            "--benchmark",
            str(benchmark),
            "--prompt",
            str(prompt),
            "--out",
            str(out),
            *extra,
        ],
    )


def test_models_lists_every_rostered_model() -> None:
    result = runner.invoke(cli.app, ["models"])
    assert result.exit_code == 0
    assert "LiquidAI/LFM2.5-350M" in result.stdout


def test_run_writes_a_result_file(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    result = _run("some/model", benchmark_path, prompt_path, tmp_path)
    assert result.exit_code == 0, result.stdout
    payload = json.loads((tmp_path / "some__model.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["model_id"] == "some/model"
    assert payload["metrics"]["n_items"] == 2


def test_run_reports_a_missing_benchmark_without_a_traceback(
    monkeypatch, tmp_path: Path, prompt_path: Path
) -> None:
    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    result = _run("some/model", tmp_path / "missing.csv", prompt_path, tmp_path)
    assert result.exit_code == 2
    assert "missing.csv" in result.stderr


def test_report_builds_a_leaderboard_from_stored_runs(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    results_dir = tmp_path / "results"
    for model in ("a/one", "b/two"):
        _run(model, benchmark_path, prompt_path, results_dir)
    report = runner.invoke(cli.app, ["report", "--results-dir", str(results_dir)])
    assert report.exit_code == 0, report.stdout
    lines = (results_dir / "leaderboard.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert "a/one" in report.stdout


def test_report_on_an_empty_directory_fails_clearly(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["report", "--results-dir", str(tmp_path)])
    assert result.exit_code == 2
    assert "no run" in result.stderr.lower()


def test_publish_pushes_the_stored_runs(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    from landuse_relevance_bench.adapters import hf_publish

    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    results_dir = tmp_path / "results"
    _run("some/model", benchmark_path, prompt_path, results_dir)
    _run("other/model", benchmark_path, prompt_path, results_dir / "recent-models-20260913")
    calls: list[dict] = []

    class FakeApi:
        def create_repo(self, **kwargs) -> None:
            calls.append({"create": kwargs})

        def upload_folder(self, **kwargs) -> None:
            calls.append({"upload": kwargs})

    monkeypatch.setattr(hf_publish, "_default_api", FakeApi)
    result = runner.invoke(
        cli.app,
        [
            "publish",
            "me/bench",
            "--results-dir",
            str(results_dir),
            "--benchmark",
            str(benchmark_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert calls[0]["create"]["repo_id"] == "me/bench"
    assert (results_dir / "README.md").exists()
    assert (results_dir / "leaderboard.csv").exists()
    card = (results_dir / "README.md").read_text(encoding="utf-8")
    assert "some/model" in card and "other/model" in card


def test_publish_refuses_an_empty_results_directory(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["publish", "me/bench", "--results-dir", str(tmp_path)])
    assert result.exit_code == 2
    assert "no run" in result.stderr.lower()


class RecordingProvider:
    def __init__(self) -> None:
        self.requests: list[RunRequest] = []

    def __call__(self, request: RunRequest) -> tuple[AlwaysYes, str]:
        self.requests.append(request)
        return AlwaysYes(), request.revision or "resolved"


def test_run_all_writes_one_result_per_rostered_model(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    monkeypatch.setattr(cli, "model_ids", lambda: ("a/one", "b/two"))
    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    result = runner.invoke(
        cli.app,
        [
            "run-all",
            "--benchmark",
            str(benchmark_path),
            "--prompt",
            str(prompt_path),
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["a__one.json", "b__two.json"]


def test_run_passes_an_explicit_revision_to_the_provider(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    provider = RecordingProvider()
    monkeypatch.setattr(cli, "generator_provider", lambda: provider)
    result = _run("some/model", benchmark_path, prompt_path, tmp_path, "--revision", "c0ffee")
    assert result.exit_code == 0, result.stdout
    assert [r.revision for r in provider.requests] == ["c0ffee"]
    payload = json.loads((tmp_path / "some__model.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["model_revision"] == "c0ffee"


def test_run_reports_a_prompt_without_a_placeholder_without_a_traceback(
    monkeypatch, tmp_path: Path, benchmark_path: Path
) -> None:
    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    bad_prompt = tmp_path / "bad_prompt.txt"
    bad_prompt.write_text("Classify this sentence.", encoding="utf-8")
    result = _run("some/model", benchmark_path, bad_prompt, tmp_path)
    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert not (tmp_path / "some__model.json").exists()


def test_run_reports_a_zero_batch_size_without_a_traceback(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    result = _run("some/model", benchmark_path, prompt_path, tmp_path, "--batch-size", "0")
    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert not (tmp_path / "some__model.json").exists()


def test_report_writes_the_leaderboard_to_a_custom_path(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    results_dir = tmp_path / "results"
    _run("a/one", benchmark_path, prompt_path, results_dir)
    destination = tmp_path / "elsewhere" / "board.csv"
    report = runner.invoke(
        cli.app, ["report", "--results-dir", str(results_dir), "--out", str(destination)]
    )
    assert report.exit_code == 0, report.stdout
    assert destination.read_text(encoding="utf-8").startswith("model_id,")
    assert not (results_dir / "leaderboard.csv").exists()


def test_report_on_a_missing_directory_fails_clearly(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["report", "--results-dir", str(tmp_path / "nowhere")])
    assert result.exit_code == 2
    assert "no run results directory" in result.stderr.lower()


def test_report_only_reads_runs_at_the_top_of_the_results_directory(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    """Pins current behaviour: unlike ``publish``, ``report`` does not descend into
    dated sub-folders such as ``results/qwen3-20260913/``."""
    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    results_dir = tmp_path / "results"
    _run("top/model", benchmark_path, prompt_path, results_dir)
    _run("nested/model", benchmark_path, prompt_path, results_dir / "qwen3-20260913")
    report = runner.invoke(cli.app, ["report", "--results-dir", str(results_dir)])
    assert report.exit_code == 0, report.stdout
    board = (results_dir / "leaderboard.csv").read_text(encoding="utf-8")
    assert "top/model" in board
    assert "nested/model" not in board


class AlwaysNo:
    def generate(self, prompts: Sequence[str]) -> Sequence[str]:
        return ["no"] * len(prompts)


def _run_pair(monkeypatch, tmp_path: Path, benchmark: Path, prompt: Path, drafted: type) -> Result:
    def provider(request: RunRequest) -> tuple[object, str]:
        return (drafted() if request.draft_model_id else AlwaysYes()), "rev"

    monkeypatch.setattr(cli, "generator_provider", lambda: provider)
    for name in ("LiquidAI/LFM2.5-2.6B@sglang", "LiquidAI/LFM2.5-2.6B+DSpark"):
        assert _run(name, benchmark, prompt, tmp_path).exit_code == 0
    return runner.invoke(cli.app, ["report", "--results-dir", str(tmp_path)])


def test_report_confirms_a_lossless_speculative_run(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    report = _run_pair(monkeypatch, tmp_path, benchmark_path, prompt_path, AlwaysYes)
    assert report.exit_code == 0, report.stdout
    assert "LiquidAI/LFM2.5-2.6B+DSpark vs LiquidAI/LFM2.5-2.6B@sglang" in report.stdout
    assert "lossless" in report.stdout
    assert "p50=" in report.stdout


def test_report_fails_when_a_speculative_run_changed_its_target_s_output(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    report = _run_pair(monkeypatch, tmp_path, benchmark_path, prompt_path, AlwaysNo)
    assert report.exit_code == 1
    assert "MISMATCH, 2/2 verdicts" in report.stdout


def test_version_prints_the_package_version() -> None:
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_models_json_lists_every_rostered_run() -> None:
    result = runner.invoke(cli.app, ["models", "--json"])
    assert result.exit_code == 0
    rows = json.loads(result.stdout)
    assert "LiquidAI/LFM2.5-350M" in [r["name"] for r in rows]
    assert {"name", "model_id", "total_parameters", "runtime", "note"} <= set(rows[0])


def test_run_json_prints_the_metrics(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    result = _run("some/model", benchmark_path, prompt_path, tmp_path, "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["model_id"] == "some/model"
    assert payload["metrics"]["n_items"] == 2


def test_run_accepts_results_dir_as_an_alias_of_out(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    result = runner.invoke(
        cli.app,
        [
            "run",
            "some/model",
            "--benchmark",
            str(benchmark_path),
            "--prompt",
            str(prompt_path),
            "--results-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "some__model.json").exists()


def test_report_json_is_valid_json(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    _run("a/one", benchmark_path, prompt_path, tmp_path)
    report = runner.invoke(cli.app, ["report", "--results-dir", str(tmp_path), "--json"])
    assert report.exit_code == 0, report.output
    payload = json.loads(report.stdout)
    assert [row["model_id"] for row in payload["leaderboard"]] == ["a/one"]


def test_verbose_run_reports_batch_progress(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    result = runner.invoke(
        cli.app,
        [
            "-v",
            "run",
            "some/model",
            "--benchmark",
            str(benchmark_path),
            "--prompt",
            str(prompt_path),
            "--out",
            str(tmp_path),
            "--batch-size",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "batch 2/2 (2/2 prompts)" in result.stderr


def test_quiet_hides_progress(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    result = runner.invoke(
        cli.app,
        [
            "-q",
            "run",
            "some/model",
            "--benchmark",
            str(benchmark_path),
            "--prompt",
            str(prompt_path),
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "batch" not in result.stderr


def test_every_option_has_help_text() -> None:
    command = typer.main.get_command(cli.app)
    subcommands = getattr(command, "commands", {})
    assert subcommands
    for sub in (command, *subcommands.values()):
        for param in sub.params:
            if param.param_type_name == "option":
                assert getattr(param, "help", None), f"{sub.name} {param.opts} has no help"

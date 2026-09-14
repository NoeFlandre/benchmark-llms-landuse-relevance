import json
from collections.abc import Sequence
from pathlib import Path

from typer.testing import CliRunner

from landuse_relevance_bench import cli
from landuse_relevance_bench.adapters.pipeline import RunRequest

runner = CliRunner()


class AlwaysYes:
    def generate(self, prompts: Sequence[str]) -> Sequence[str]:
        return ["yes"] * len(prompts)


def _fake_provider(_: RunRequest) -> tuple[AlwaysYes, str]:
    return AlwaysYes(), "fakerev"


def test_models_lists_every_rostered_model() -> None:
    result = runner.invoke(cli.app, ["models"])
    assert result.exit_code == 0
    assert "LiquidAI/LFM2.5-350M" in result.stdout


def test_run_writes_a_result_file(
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
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads((tmp_path / "some__model.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["model_id"] == "some/model"
    assert payload["metrics"]["n_items"] == 2


def test_run_reports_a_missing_benchmark_without_a_traceback(
    monkeypatch, tmp_path: Path, prompt_path: Path
) -> None:
    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    result = runner.invoke(
        cli.app,
        [
            "run",
            "some/model",
            "--benchmark",
            str(tmp_path / "missing.csv"),
            "--prompt",
            str(prompt_path),
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert "missing.csv" in result.stderr


def test_report_builds_a_leaderboard_from_stored_runs(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    results_dir = tmp_path / "results"
    for model in ("a/one", "b/two"):
        runner.invoke(
            cli.app,
            [
                "run",
                model,
                "--benchmark",
                str(benchmark_path),
                "--prompt",
                str(prompt_path),
                "--out",
                str(results_dir),
            ],
        )
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
    runner.invoke(
        cli.app,
        [
            "run",
            "some/model",
            "--benchmark",
            str(benchmark_path),
            "--prompt",
            str(prompt_path),
            "--out",
            str(results_dir),
        ],
    )
    runner.invoke(
        cli.app,
        [
            "run",
            "other/model",
            "--benchmark",
            str(benchmark_path),
            "--prompt",
            str(prompt_path),
            "--out",
            str(results_dir / "recent-models-20260913"),
        ],
    )
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

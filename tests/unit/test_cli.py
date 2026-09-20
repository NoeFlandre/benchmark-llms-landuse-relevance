import hashlib
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


def _translation_root(tmp_path: Path, benchmark_path: Path) -> Path:
    root = tmp_path / "translations"
    source_item_ids = ["source-1", "source-2"]
    files = {}
    for language in ("en", "fr"):
        path = root / language / f"v3-final-{language}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        text = benchmark_path.read_text(encoding="utf-8").replace(",en\n", f",{language}\n")
        path.write_text(text, encoding="utf-8")
        files[language] = {
            "path": f"{language}/v3-final-{language}.csv",
            "rows": 2,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    inventory = "\n".join(f"{language}:{files[language]['sha256']}" for language in files)
    manifest = {
        "dataset": "test/dataset",
        "revision": "test-revision",
        "split": "train",
        "languages": ["en", "fr"],
        "row_count": 2,
        "source_item_ids": source_item_ids,
        "files": files,
        "whole_set_sha256": hashlib.sha256(inventory.encode()).hexdigest(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_models_lists_every_rostered_model() -> None:
    result = runner.invoke(cli.app, ["models"])
    assert result.exit_code == 0
    assert "LiquidAI/LFM2.5-350M" in result.stdout


def test_languages_lists_the_active_language_inventory() -> None:
    result = runner.invoke(cli.app, ["languages", "--data-root", "data/translations"])

    assert result.exit_code == 0, result.stdout
    assert "en\t300" in result.stdout
    assert len(result.stdout.strip().splitlines()) == 85


def test_run_writes_a_result_file(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    data_root = _translation_root(tmp_path, benchmark_path)
    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    result = runner.invoke(
        cli.app,
        [
            "run",
            "some/model",
            "--data-root",
            str(data_root),
            "--language",
            "en",
            "--prompt",
            str(prompt_path),
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads((tmp_path / "en" / "some__model.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["model_id"] == "some/model"
    assert payload["metrics"]["n_items"] == 2


def test_run_reuses_one_generator_across_languages(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    data_root = _translation_root(tmp_path, benchmark_path)
    provider_languages: list[str] = []

    def recording_provider(request: RunRequest) -> tuple[AlwaysYes, str]:
        provider_languages.append(request.language)
        return AlwaysYes(), "fakerev"

    monkeypatch.setattr(cli, "generator_provider", lambda: recording_provider)
    result = runner.invoke(
        cli.app,
        [
            "run",
            "some/model",
            "--data-root",
            str(data_root),
            "--prompt",
            str(prompt_path),
            "--out",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert provider_languages == ["en"]
    assert (tmp_path / "en" / "some__model.json").is_file()
    assert (tmp_path / "fr" / "some__model.json").is_file()


def test_run_reports_a_missing_benchmark_without_a_traceback(
    monkeypatch, tmp_path: Path, prompt_path: Path
) -> None:
    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    result = runner.invoke(
        cli.app,
        [
            "run",
            "some/model",
            "--data-root",
            str(tmp_path / "missing-translations"),
            "--prompt",
            str(prompt_path),
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert "missing-translations" in "".join(result.output.split())


def test_run_defaults_to_every_language(monkeypatch, tmp_path: Path) -> None:
    requests: list[RunRequest] = []
    monkeypatch.setattr(
        cli, "_benchmark_one", lambda request, _provider=None: requests.append(request)
    )

    result = runner.invoke(
        cli.app,
        ["run", "some/model", "--data-root", "data/translations", "--prompt", "data/prompt.txt"],
    )

    assert result.exit_code == 0, result.stdout
    assert [request.language for request in requests] == sorted(
        request.language for request in requests
    )
    assert len(requests) == 85


def test_run_accepts_repeated_and_comma_separated_language_filters(
    monkeypatch,
) -> None:
    languages: list[str] = []
    monkeypatch.setattr(
        cli,
        "_benchmark_one",
        lambda request, _provider=None: languages.append(request.language),
    )

    result = runner.invoke(
        cli.app,
        [
            "run",
            "some/model",
            "--data-root",
            "data/translations",
            "--language",
            "fr",
            "--language",
            "en,de",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert languages == ["de", "en", "fr"]


def test_run_all_shard_selects_a_deterministic_subset(monkeypatch) -> None:
    requests: list[RunRequest] = []
    monkeypatch.setattr(
        cli, "_benchmark_one", lambda request, _provider=None: requests.append(request)
    )

    result = runner.invoke(
        cli.app,
        [
            "run-all",
            "--data-root",
            "data/translations",
            "--shard-index",
            "1",
            "--shard-count",
            "3",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert len(requests) == 510
    assert [(request.model_id, request.language) for request in requests] == sorted(
        (request.model_id, request.language) for request in requests
    )


def test_status_reports_pending_pairs_for_a_selected_language() -> None:
    result = runner.invoke(
        cli.app,
        [
            "status",
            "--data-root",
            "data/translations",
            "--language",
            "en",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert len(result.stdout.strip().splitlines()) == 18
    assert all(line.endswith("\tpending") for line in result.stdout.strip().splitlines())


def test_status_marks_a_valid_rostered_pair_complete(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    data_root = _translation_root(tmp_path, benchmark_path)
    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    run = runner.invoke(
        cli.app,
        [
            "run",
            "LiquidAI/LFM2.5-350M",
            "--data-root",
            str(data_root),
            "--language",
            "en",
            "--prompt",
            str(prompt_path),
            "--out",
            str(tmp_path / "results"),
        ],
    )
    assert run.exit_code == 0, run.stdout

    status_result = runner.invoke(
        cli.app,
        [
            "status",
            "--data-root",
            str(data_root),
            "--language",
            "en",
            "--results-dir",
            str(tmp_path / "results"),
        ],
    )

    assert status_result.exit_code == 0, status_result.stdout
    assert "LiquidAI/LFM2.5-350M\ten\tcomplete" in status_result.stdout


def test_run_skips_an_existing_valid_model_language_pair(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    data_root = _translation_root(tmp_path, benchmark_path)
    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    first = runner.invoke(
        cli.app,
        [
            "run",
            "some/model",
            "--data-root",
            str(data_root),
            "--language",
            "en",
            "--prompt",
            str(prompt_path),
            "--out",
            str(tmp_path / "results"),
        ],
    )
    assert first.exit_code == 0, first.stdout
    monkeypatch.setattr(
        cli, "execute", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError)
    )

    second = runner.invoke(
        cli.app,
        [
            "run",
            "some/model",
            "--data-root",
            str(data_root),
            "--language",
            "en",
            "--prompt",
            str(prompt_path),
            "--out",
            str(tmp_path / "results"),
        ],
    )

    assert second.exit_code == 0, second.stdout
    assert "skipping" in second.stdout


def test_report_builds_a_leaderboard_from_stored_runs(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    data_root = _translation_root(tmp_path, benchmark_path)
    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    results_dir = tmp_path / "results"
    for model in ("a/one", "b/two"):
        runner.invoke(
            cli.app,
            [
                "run",
                model,
                "--data-root",
                str(data_root),
                "--language",
                "en",
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
    assert (results_dir / "aggregates.csv").exists()


def test_report_filters_languages_and_writes_both_csvs(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    data_root = _translation_root(tmp_path, benchmark_path)
    monkeypatch.setattr(cli, "generator_provider", lambda: _fake_provider)
    results_dir = tmp_path / "results"
    for language in ("en", "fr"):
        run = runner.invoke(
            cli.app,
            [
                "run",
                "a/one",
                "--data-root",
                str(data_root),
                "--language",
                language,
                "--prompt",
                str(prompt_path),
                "--out",
                str(results_dir),
            ],
        )
        assert run.exit_code == 0, run.stdout

    report = runner.invoke(
        cli.app,
        ["report", "--results-dir", str(results_dir), "--language", "fr"],
    )

    assert report.exit_code == 0, report.stdout
    assert len((results_dir / "leaderboard.csv").read_text(encoding="utf-8").splitlines()) == 2
    assert len((results_dir / "aggregates.csv").read_text(encoding="utf-8").splitlines()) == 2


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
    data_root = _translation_root(tmp_path, benchmark_path)
    runner.invoke(
        cli.app,
        [
            "run",
            "some/model",
            "--data-root",
            str(data_root),
            "--language",
            "en",
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
            "--data-root",
            str(data_root),
            "--language",
            "fr",
            "--prompt",
            str(prompt_path),
            "--out",
            str(results_dir),
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
            "--benchmark-name",
            "v3-multilingual",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert calls[0]["create"]["repo_id"] == "me/bench"
    assert (results_dir / "README.md").exists()
    assert (results_dir / "leaderboard.csv").exists()
    assert (results_dir / "aggregates.csv").exists()
    card = (results_dir / "README.md").read_text(encoding="utf-8")
    assert "some/model" in card and "other/model" in card


def test_publish_refuses_an_empty_results_directory(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["publish", "me/bench", "--results-dir", str(tmp_path)])
    assert result.exit_code == 2
    assert "no run" in result.stderr.lower()

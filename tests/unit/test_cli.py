import csv
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from landuse_relevance_bench import application, cli
from landuse_relevance_bench.adapters import providers
from landuse_relevance_bench.adapters.pipeline import RunRequest
from landuse_relevance_bench.adapters.translations import load_manifest
from landuse_relevance_bench.domain.roster import model_ids
from landuse_relevance_bench.domain.scorers import scorer_for

runner = CliRunner()


def test_version_option_prints_the_package_version() -> None:
    import landuse_relevance_bench

    result = runner.invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == landuse_relevance_bench.__version__


def test_models_json_lists_the_rostered_ids() -> None:
    result = runner.invoke(cli.app, ["models", "--json"])

    assert result.exit_code == 0
    assert {row["id"] for row in json.loads(result.stdout)} == set(model_ids())


class _ClosableProvider:
    def __init__(self) -> None:
        self.closed = 0

    def close_cached(self) -> None:
        self.closed += 1


def _score(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: _ClosableProvider,
    failure: Exception | None,
) -> None:
    manifest = SimpleNamespace(files={"en": SimpleNamespace(path=Path("en.csv"))})
    monkeypatch.setattr(cli, "selected_languages", lambda *_: (manifest, ("en",)))
    monkeypatch.setattr(cli, "print_run_plan", lambda *args: None)
    monkeypatch.setattr(cli, "cached_scorer_provider", lambda: provider)

    def score_one(request: RunRequest, used: object) -> None:
        if failure is not None:
            raise failure

    monkeypatch.setattr(cli, "score_one", score_one)
    cli.score("LiquidAI/LFM2.5-Encoder-350M", data_root=tmp_path)


def test_score_command_releases_its_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = _ClosableProvider()
    _score(monkeypatch, tmp_path, provider, None)
    assert provider.closed == 1


def test_score_command_releases_its_provider_when_scoring_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = _ClosableProvider()
    with pytest.raises(RuntimeError, match="scoring failed"):
        _score(monkeypatch, tmp_path, provider, RuntimeError("scoring failed"))
    assert provider.closed == 1


def test_score_command_dispatches_each_selected_scoring_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_id = "LiquidAI/LFM2.5-Encoder-350M"
    manifest = SimpleNamespace(files={"en": SimpleNamespace(path=Path("en.csv"))})
    requests = []
    monkeypatch.setattr(cli, "selected_languages", lambda *_: (manifest, ("en",)))
    monkeypatch.setattr(cli, "print_run_plan", lambda *args: None)
    provider = _ClosableProvider()
    monkeypatch.setattr(cli, "cached_scorer_provider", lambda: provider)
    monkeypatch.setattr(
        cli, "score_one", lambda request, provider: requests.append((request, provider))
    )

    cli.score(model_id, data_root=tmp_path, out=tmp_path / "results")

    assert len(requests) == 1
    request, used = requests[0]
    assert used is provider
    assert request.model_id == model_id
    assert request.language == "en"
    assert request.benchmark_path == tmp_path / "en.csv"
    assert request.output_dir == tmp_path / "results"


def test_score_command_rejects_a_non_scoring_roster_id() -> None:
    with pytest.raises(typer.BadParameter, match="is not in the scoring roster"):
        cli.score("some/generative-model")


def test_scoring_language_selection_respects_encoder_coverage() -> None:
    spec = scorer_for("LiquidAI/LFM2.5-Encoder-350M")

    assert application.languages_for_scorer(spec, ("af", "de", "en", "fr"), None) == (
        "de",
        "en",
        "fr",
    )
    with pytest.raises(typer.BadParameter, match="only supports"):
        application.languages_for_scorer(spec, ("af", "en"), ["af", "en"])

    general_scorer = scorer_for("Alibaba-NLP/gte-multilingual-reranker-base")
    assert application.languages_for_scorer(general_scorer, ("de", "en"), None) == ("de", "en")
    with pytest.raises(typer.BadParameter, match="no selected language"):
        application.languages_for_scorer(spec, ("af",), None)


def test_cli_providers_remain_lazy_and_dispatch_to_the_selected_runtime(monkeypatch) -> None:
    from landuse_relevance_bench.adapters import hf_scorer

    calls: list[tuple[str, RunRequest]] = []
    transformers = ModuleType("landuse_relevance_bench.adapters.hf_generator")
    transformers.provide = lambda request: calls.append(("transformers", request)) or "hf"
    sglang = ModuleType("landuse_relevance_bench.adapters.sglang_generator")
    sglang.provide = lambda request: calls.append(("sglang", request)) or "sg"
    monkeypatch.setitem(sys.modules, transformers.__name__, transformers)
    monkeypatch.setitem(sys.modules, sglang.__name__, sglang)
    monkeypatch.setattr(hf_scorer, "provide_scorer", lambda request: ("scorer", request.name))

    def request(model_id: str) -> RunRequest:
        return RunRequest.for_run(
            model_id,
            language="en",
            benchmark_path=Path("benchmark.csv"),
            prompt_path=Path("prompt.txt"),
            output_dir=Path("results"),
        )

    hf_request = request("LiquidAI/LFM2.5-350M")
    sglang_request = request("LiquidAI/LFM2.5-1.2B-Instruct@sglang")
    assert providers.generator_provider()(hf_request) == "hf"
    assert providers.generator_provider()(sglang_request) == "sg"
    assert providers.scorer_provider()(request("LiquidAI/LFM2.5-Encoder-350M")) == (
        "scorer",
        "LiquidAI/LFM2.5-Encoder-350M",
    )
    assert [name for name, _ in calls] == ["transformers", "sglang"]


def test_cached_provider_reuses_closes_and_rejects_a_missing_model(monkeypatch) -> None:
    closed: list[str] = []
    created: list[str] = []

    class Generator:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    provider = providers.CachedProvider(
        lambda: lambda request: (created.append(request.name) or Generator(request.name), "rev")
    )
    request = RunRequest.for_run(
        "LiquidAI/LFM2.5-350M",
        language="en",
        benchmark_path=Path("benchmark.csv"),
        prompt_path=Path("prompt.txt"),
        output_dir=Path("results"),
    )
    first = provider(request)
    assert provider(request) is first
    second_request = RunRequest.for_run(
        "LiquidAI/LFM2.5-1.2B-Instruct@sglang",
        language="en",
        benchmark_path=Path("benchmark.csv"),
        prompt_path=Path("prompt.txt"),
        output_dir=Path("results"),
    )
    provider(second_request)
    provider.close_cached()
    provider.close_cached()

    assert created == [request.name, second_request.name]
    assert closed == [request.name, second_request.name]
    missing = providers.CachedProvider(lambda: lambda _request: None)
    with pytest.raises(RuntimeError, match="failed to load"):
        missing(request)


def test_cli_language_validation_rejects_empty_and_unknown_codes(
    tmp_path: Path, benchmark_path: Path
) -> None:
    data_root = _translation_root(tmp_path, benchmark_path)
    with pytest.raises(typer.BadParameter, match="cannot be empty"):
        application.normalize_language_selectors(["en,"])
    with pytest.raises(typer.BadParameter, match="unknown language"):
        application.selected_languages(data_root, ["xx"])


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
    monkeypatch.setattr(providers, "generator_provider", lambda: _fake_provider)
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


def test_run_prints_prompt_count_and_resume_state(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    data_root = _translation_root(tmp_path, benchmark_path)
    monkeypatch.setattr(providers, "generator_provider", lambda: _fake_provider)
    arguments = [
        "run",
        "some/model",
        "--data-root",
        str(data_root),
        "--prompt",
        str(prompt_path),
        "--out",
        str(tmp_path / "results"),
    ]

    first = runner.invoke(cli.app, arguments)
    second = runner.invoke(cli.app, arguments)

    assert first.exit_code == 0, first.stdout
    assert "languages=2 [en,fr]" in first.stdout
    assert "prompts=4" in first.stdout
    assert "resume=0 complete,2 pending,0 invalid" in first.stdout
    assert second.exit_code == 0, second.stdout
    assert "resume=2 complete,0 pending,0 invalid" in second.stdout


def test_run_reuses_one_generator_across_languages(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    data_root = _translation_root(tmp_path, benchmark_path)
    provider_languages: list[str] = []

    def recording_provider(request: RunRequest) -> tuple[AlwaysYes, str]:
        provider_languages.append(request.language)
        return AlwaysYes(), "fakerev"

    monkeypatch.setattr(providers, "generator_provider", lambda: recording_provider)
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
    monkeypatch.setattr(providers, "generator_provider", lambda: _fake_provider)
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
        cli, "benchmark_one", lambda request, _provider=None, **_kwargs: requests.append(request)
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
        "benchmark_one",
        lambda request, _provider=None, **_kwargs: languages.append(request.language),
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
        cli, "benchmark_one", lambda request, _provider=None, **_kwargs: requests.append(request)
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
    pairs = len(model_ids()) * len(load_manifest(Path("data/translations")).languages)
    assert len(requests) == len(range(1, pairs, 3))
    assert [(request.name, request.language) for request in requests] == sorted(
        (request.name, request.language) for request in requests
    )


def test_run_all_records_sglang_throughput_variant(monkeypatch) -> None:
    requests: list[RunRequest] = []
    monkeypatch.setattr(
        cli, "benchmark_one", lambda request, _provider=None, **_kwargs: requests.append(request)
    )

    result = runner.invoke(
        cli.app,
        [
            "run-all",
            "--data-root",
            "data/translations",
            "--language",
            "en",
            "--runtime",
            "sglang",
            "--throughput",
            "--only",
            "LFM2.5-1.2B-Instruct",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert len(requests) == 2
    assert all(request.throughput_mode and request.batch_size == 16 for request in requests)
    assert {request.name for request in requests} == {
        "LiquidAI/LFM2.5-1.2B-Instruct@sglang-throughput-b16",
        "LiquidAI/LFM2.5-1.2B-Instruct+DSpark-throughput-b16",
    }


def test_run_all_uses_continuous_batching_only_for_supported_transformers(
    monkeypatch,
) -> None:
    requests: list[RunRequest] = []
    monkeypatch.setattr(
        cli, "benchmark_one", lambda request, _provider=None, **_kwargs: requests.append(request)
    )

    result = runner.invoke(
        cli.app,
        [
            "run-all",
            "--data-root",
            "data/translations",
            "--language",
            "en",
            "--runtime",
            "transformers",
            "--continuous-batching",
            "--only",
            "350M|VL-3B",
        ],
    )

    assert result.exit_code == 0, result.stdout
    by_model = {request.model_id: request for request in requests}
    assert by_model["LiquidAI/LFM2.5-350M"].continuous_batching
    assert not by_model["LiquidAI/LFM2.5-VL-3B"].continuous_batching


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
    assert len(result.stdout.strip().splitlines()) == len(model_ids())
    assert all(line.endswith("\tpending") for line in result.stdout.strip().splitlines())


def test_status_marks_a_valid_rostered_pair_complete(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    data_root = _translation_root(tmp_path, benchmark_path)
    monkeypatch.setattr(providers, "generator_provider", lambda: _fake_provider)
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
    monkeypatch.setattr(providers, "generator_provider", lambda: _fake_provider)
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
        application, "execute", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError)
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
    monkeypatch.setattr(providers, "generator_provider", lambda: _fake_provider)
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
    assert (results_dir / "threshold_sweep.csv").exists()
    assert (results_dir / "scoring_summary.csv").exists()


def test_report_filters_languages_and_writes_both_csvs(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    data_root = _translation_root(tmp_path, benchmark_path)
    monkeypatch.setattr(providers, "generator_provider", lambda: _fake_provider)
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
    assert (results_dir / "threshold_sweep.csv").exists()
    assert (results_dir / "scoring_summary.csv").exists()


def test_report_on_an_empty_directory_fails_clearly(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["report", "--results-dir", str(tmp_path)])
    assert result.exit_code == 2
    assert "no run" in result.stderr.lower()


def test_publish_pushes_the_stored_runs(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    from landuse_relevance_bench.adapters import hf_publish

    monkeypatch.setattr(providers, "generator_provider", lambda: _fake_provider)
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
            "--data-root",
            str(data_root),
            "--results-dir",
            str(results_dir),
            "--prompt",
            str(prompt_path),
            "--benchmark-name",
            "v3-multilingual",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert calls[0]["create"]["repo_id"] == "me/bench"
    assert (results_dir / "README.md").exists()
    assert (results_dir / "leaderboard.csv").exists()
    assert (results_dir / "aggregates.csv").exists()
    assert (results_dir / "threshold_sweep.csv").exists()
    assert (results_dir / "scoring_summary.csv").exists()
    assert (results_dir / "data" / "train.csv").exists()
    card = (results_dir / "README.md").read_text(encoding="utf-8")
    assert "some/model" in card and "other/model" in card


def test_publish_filters_language_artifacts_and_viewer_rows(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    from landuse_relevance_bench.adapters import hf_publish

    monkeypatch.setattr(providers, "generator_provider", lambda: _fake_provider)
    results_dir = tmp_path / "results"
    data_root = _translation_root(tmp_path, benchmark_path)
    for model, language in (("some/model", "en"), ("other/model", "fr")):
        runner.invoke(
            cli.app,
            [
                "run",
                model,
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
    captured: dict = {}

    class FakeApi:
        def create_repo(self, **kwargs) -> None:
            captured["create"] = kwargs

        def upload_folder(self, **kwargs) -> None:
            captured["upload"] = kwargs
            upload_dir = Path(kwargs["folder_path"])
            captured["card"] = (upload_dir / "README.md").read_text(encoding="utf-8")
            captured["viewer"] = (upload_dir / "data" / "train.csv").read_text(encoding="utf-8")

    monkeypatch.setattr(hf_publish, "_default_api", FakeApi)
    result = runner.invoke(
        cli.app,
        [
            "publish",
            "me/bench",
            "--data-root",
            str(data_root),
            "--results-dir",
            str(results_dir),
            "--prompt",
            str(prompt_path),
            "--language",
            "fr",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "other/model" in captured["card"]
    assert "some/model" not in captured["card"]
    viewer_rows = list(csv.DictReader(captured["viewer"].splitlines()))
    assert viewer_rows and {row["language"] for row in viewer_rows} == {"fr"}
    assert "fr/other__model.json" in captured["upload"]["allow_patterns"]
    assert "en/some__model.json" not in captured["upload"]["allow_patterns"]
    assert (results_dir / "en" / "some__model.json").is_file()
    assert (results_dir / "fr" / "other__model.json").is_file()


def test_publish_allowlist_includes_snapshot_and_excludes_local_cache(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    from landuse_relevance_bench.adapters import hf_publish

    monkeypatch.setattr(providers, "generator_provider", lambda: _fake_provider)
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
    (results_dir / "SNAPSHOT_STATUS.md").write_text("complete\n", encoding="utf-8")
    cache_file = results_dir / ".cache" / "hub-download-metadata"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("local-only\n", encoding="utf-8")
    captured: dict = {}

    class FakeApi:
        def create_repo(self, **kwargs) -> None:
            captured["create"] = kwargs

        def upload_folder(self, **kwargs) -> None:
            captured["upload"] = kwargs

    monkeypatch.setattr(hf_publish, "_default_api", FakeApi)
    result = runner.invoke(
        cli.app,
        [
            "publish",
            "me/bench",
            "--data-root",
            str(data_root),
            "--results-dir",
            str(results_dir),
            "--prompt",
            str(prompt_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    patterns = captured["upload"].get("allow_patterns")
    assert patterns is not None
    assert "SNAPSHOT_STATUS.md" in patterns
    assert "en/some__model.json" in patterns
    assert not any(pattern.startswith(".cache/") for pattern in patterns)


@pytest.mark.parametrize(
    "run_id",
    [
        "LiquidAI/LFM2.5-2.6B@sglang-throughput-b16",
        "LiquidAI/LFM2.5-2.6B+DSpark-throughput-b16",
    ],
)
def test_publish_allowlist_uses_variant_run_id_not_base_model_id(run_id: str) -> None:
    from factories import make_result

    result = make_result(model_id="LiquidAI/LFM2.5-2.6B", run_id=run_id)

    patterns = application.publish_allow_patterns(Path("results"), [result])

    assert f"en/{run_id.replace('/', '__')}.json" in patterns


def test_publish_dry_run_previews_card_without_writing_or_uploading(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    monkeypatch.setattr(providers, "generator_provider", lambda: _fake_provider)
    results_dir = tmp_path / "results"
    data_root = _translation_root(tmp_path, benchmark_path)
    run = runner.invoke(
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
    assert run.exit_code == 0, run.stdout

    result = runner.invoke(
        cli.app,
        [
            "publish",
            "me/bench",
            "--data-root",
            str(data_root),
            "--results-dir",
            str(results_dir),
            "--prompt",
            str(prompt_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "dry-run: would publish 1 result(s)" in result.stdout
    assert "some/model" in result.stdout
    assert not (results_dir / "README.md").exists()


def test_publish_refuses_an_empty_results_directory(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["publish", "me/bench", "--results-dir", str(tmp_path)])
    assert result.exit_code == 2
    assert "no run" in result.stderr.lower()


def test_scorers_lists_the_non_generative_roster() -> None:
    result = runner.invoke(cli.app, ["scorers"])

    assert result.exit_code == 0
    assert "LiquidAI/LFM2.5-Encoder-350M" in result.stdout


def test_run_refuses_to_overwrite_a_corrupt_checkpoint(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    from landuse_relevance_bench.adapters.results_store import run_filename

    data_root = _translation_root(tmp_path, benchmark_path)
    results_dir = tmp_path / "results"
    corrupt = results_dir / run_filename("some/model", "en")
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(providers, "generator_provider", lambda: _fake_provider)

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
            str(results_dir),
        ],
    )

    assert result.exit_code == 2
    assert "already complete" not in result.stdout
    assert "invalid" in result.stderr.lower()


def test_run_all_keep_going_reports_pair_errors_and_exits_nonzero(
    monkeypatch, tmp_path: Path, benchmark_path: Path, prompt_path: Path
) -> None:
    data_root = _translation_root(tmp_path, benchmark_path)

    def fail(_request: RunRequest):
        raise RuntimeError("simulated model failure")

    monkeypatch.setattr(providers, "generator_provider", lambda: fail)
    result = runner.invoke(
        cli.app,
        [
            "run-all",
            "--data-root",
            str(data_root),
            "--prompt",
            str(prompt_path),
            "--out",
            str(tmp_path / "results"),
            "--language",
            "en",
            "--only",
            "LiquidAI/LFM2.5-350M",
            "--keep-going",
        ],
    )

    assert result.exit_code == 1
    assert "simulated model failure" in result.stderr
    assert "failed" in result.stderr

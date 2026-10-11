"""Usage-error messages, planning lines and checkpoint handling of the CLI use cases."""

import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from factories import make_result
from landuse_relevance_bench import application
from landuse_relevance_bench.adapters.pipeline import RunRequest
from landuse_relevance_bench.adapters.prompt_file import PromptFileError
from landuse_relevance_bench.adapters.results_store import run_filename
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.records import Prediction
from landuse_relevance_bench.domain.roster import ROSTER, SGLANG, TRANSFORMERS
from landuse_relevance_bench.domain.scorers import SCORER_ROSTER

Y, N = Label.YES, Label.NO


def _predictions(pairs):
    return tuple(
        Prediction(item_id=f"{index:016d}", expected=expected, predicted=predicted, raw_output="x")
        for index, (expected, predicted) in enumerate(pairs)
    )


THIRTEEN = _predictions(
    (
        (Y, Y),
        (Y, Y),
        (N, Y),
        (Y, N),
        (Y, N),
        (Y, N),
        (Y, N),
        (N, N),
        (N, N),
        (N, N),
        (N, N),
        (N, N),
        (Y, None),
    )
)


@pytest.fixture(autouse=True)
def _no_git_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """The source commit is looked up through git; keep every test independent of the checkout."""
    monkeypatch.setattr(application, "source_commit", lambda: "")


def _request(tmp_path: Path, model: str = "m", language: str = "en") -> RunRequest:
    return RunRequest(
        model_id=model,
        language=language,
        benchmark_path=tmp_path / "b.csv",
        prompt_path=tmp_path / "p.txt",
        output_dir=tmp_path,
    )


def _store_checkpoint(tmp_path: Path, model: str = "m", language: str = "en") -> Path:
    path = tmp_path / run_filename(model, language)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return path


# --- run_pair and the three use cases that wrap it ---------------------------------


def test_run_pair_skips_a_complete_checkpoint_unless_told_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _store_checkpoint(tmp_path)
    meta = SimpleNamespace(name="m", language="en")
    monkeypatch.setattr(application, "read_run", lambda _path: SimpleNamespace(metadata=meta))

    def never_runs():
        raise AssertionError("a complete pair must not run again")

    application.run_pair(_request(tmp_path), never_runs, lambda _result: "")

    assert capsys.readouterr().out == "m [en] already complete; skipping\n"


def test_run_pair_refuses_a_checkpoint_it_cannot_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored = _store_checkpoint(tmp_path)

    def unreadable(_path):
        raise ValueError("corrupt")

    monkeypatch.setattr(application, "read_run", unreadable)

    with pytest.raises(
        typer.BadParameter,
        match=re.escape(f"{stored} is unreadable or contains a different model-language run"),
    ):
        application.run_pair(_request(tmp_path), lambda: pytest.fail("must not run"), lambda _r: "")


def test_run_pair_reports_an_input_failure_as_a_usage_error(tmp_path: Path) -> None:
    def broken():
        raise ValueError("bad batch size")

    with pytest.raises(typer.BadParameter, match=r"^bad batch size$"):
        application.run_pair(_request(tmp_path), broken, lambda _r: "")


def test_run_pair_prints_the_metrics_and_duration_of_a_finished_pair(
    tmp_path: Path, capsys
) -> None:
    result = make_result(predictions=THIRTEEN, duration_seconds=1.5)

    application.run_pair(
        _request(tmp_path),
        lambda: result,
        lambda finished: f"unparsed={finished.metrics.unparsed_rate:.3f}  ",
    )

    assert capsys.readouterr().out == (
        "m [en]  accuracy=0.538  f1=0.444  mcc=0.192  unparsed=0.077  (1.5s)\n"
    )


def test_benchmark_one_runs_the_roster_provider_with_the_resolved_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    seen: dict[str, object] = {}

    def fake_execute(request, provider, *, source_commit):
        seen.update(provider=provider, source_commit=source_commit)
        return make_result(predictions=THIRTEEN, duration_seconds=1.5)

    monkeypatch.setattr(application, "execute", fake_execute)
    monkeypatch.setattr(application, "generator_provider", lambda: "roster-provider")
    monkeypatch.setattr(application, "source_commit", lambda: "abc123")

    application.benchmark_one(_request(tmp_path))

    assert seen == {"provider": "roster-provider", "source_commit": "abc123"}
    assert capsys.readouterr().out == (
        "m [en]  accuracy=0.538  f1=0.444  mcc=0.192  unparsed=0.077  (1.5s)\n"
    )


def test_benchmark_one_prefers_an_explicit_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_execute(request, provider, *, source_commit):
        seen["provider"] = provider
        return make_result(predictions=THIRTEEN)

    monkeypatch.setattr(application, "execute", fake_execute)
    monkeypatch.setattr(application, "generator_provider", lambda: "roster-provider")

    application.benchmark_one(_request(tmp_path), provider="explicit")

    assert seen["provider"] == "explicit"


def test_benchmark_one_runs_again_when_asked_to_ignore_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _store_checkpoint(tmp_path)
    meta = SimpleNamespace(name="m", language="en")
    monkeypatch.setattr(application, "read_run", lambda _path: SimpleNamespace(metadata=meta))
    ran: list[str] = []

    def fake_execute(request, provider, *, source_commit):
        ran.append(request.name)
        return make_result(predictions=THIRTEEN)

    monkeypatch.setattr(application, "execute", fake_execute)

    application.benchmark_one(_request(tmp_path), provider="p", skip_existing=False)

    assert ran == ["m"]


def test_score_one_scores_through_the_scorer_provider_and_prints_throughput(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    seen: dict[str, object] = {}

    def fake_execute_scoring(request, provider, *, source_commit):
        seen.update(request=request, provider=provider, source_commit=source_commit)
        return make_result(
            predictions=THIRTEEN,
            duration_seconds=1.5,
            throughput_items_per_second=12.5,
            peak_vram_bytes=987,
        )

    monkeypatch.setattr(application, "execute_scoring", fake_execute_scoring)
    monkeypatch.setattr(application, "scorer_provider", lambda: "roster-scorer")
    monkeypatch.setattr(application, "source_commit", lambda: "abc123")
    request = _request(tmp_path)

    application.score_one(request)

    assert seen["request"] is request
    assert seen["provider"] == "roster-scorer"
    assert seen["source_commit"] == "abc123"
    assert capsys.readouterr().out == (
        "m [en]  accuracy=0.538  f1=0.444  mcc=0.192  throughput=12.5/s  peak_vram=987B  (1.5s)\n"
    )


def test_score_one_prefers_an_explicit_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_execute_scoring(request, provider, *, source_commit):
        seen["provider"] = provider
        return make_result(predictions=THIRTEEN)

    monkeypatch.setattr(application, "execute_scoring", fake_execute_scoring)
    monkeypatch.setattr(application, "scorer_provider", lambda: "roster-scorer")

    application.score_one(_request(tmp_path), provider="explicit")

    assert seen["provider"] == "explicit"


# --- planning, selection and language messages -------------------------------------


def test_print_run_plan_counts_each_checkpoint_state_and_the_prompt_workload(
    tmp_path: Path, capsys
) -> None:
    manifest = SimpleNamespace(files={"en": SimpleNamespace(rows=3)})
    (tmp_path / "en").mkdir()
    (tmp_path / "en" / "m2.json").write_text("{}", encoding="utf-8")

    application.print_run_plan(
        ["m1", "m2"],
        ("en",),
        manifest,
        (("m1", "en"), ("m2", "en")),
        tmp_path,
    )

    assert capsys.readouterr().out == (
        "plan: models=2 languages=1 [en] prompts=6 assigned_pairs=2 assigned_prompts=6 "
        "resume=0 complete,1 pending,1 invalid\n"
    )


def test_comparable_runs_reports_every_incompatibility_in_one_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        application,
        "collection_comparability_errors",
        lambda _runs: ("first problem", "second problem"),
    )

    with pytest.raises(typer.BadParameter, match=re.escape("first problem; second problem")):
        application.comparable_runs([make_result(model_id="m")], None, tmp_path)


def test_comparable_runs_names_the_directory_when_nothing_matches(tmp_path: Path) -> None:
    with pytest.raises(typer.BadParameter, match=re.escape(f"no run results found in {tmp_path}")):
        application.comparable_runs([make_result(model_id="m", language="en")], ["fr"], tmp_path)


def test_selected_models_turns_a_selection_error_into_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_match(*_args, **_kwargs):
        raise ValueError("no model matches the selector")

    monkeypatch.setattr(application, "select_run_names", no_match)

    with pytest.raises(typer.BadParameter, match=r"^no model matches the selector$"):
        application.selected_models("nothing", None)


def test_planned_pairs_turns_a_sharding_error_into_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bad_shard(*_args, **_kwargs):
        raise ValueError("shard index out of range")

    monkeypatch.setattr(application, "shard_pairs", bad_shard)

    with pytest.raises(typer.BadParameter, match=r"^shard index out of range$"):
        application.planned_pairs(["m"], ("en",), 3, 2)


def test_manifest_for_names_the_data_root_in_its_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(_root):
        raise OSError("no manifest")

    monkeypatch.setattr(application, "load_manifest", missing)
    root = tmp_path / "bad-root"

    with pytest.raises(typer.BadParameter, match=f"^{re.escape('bad-root')}: no manifest$"):
        application.manifest_for(root)


def test_manifest_for_labels_a_root_without_a_name_by_its_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_root):
        raise OSError("no manifest")

    monkeypatch.setattr(application, "load_manifest", missing)

    with pytest.raises(typer.BadParameter, match=f"^{re.escape('/')}: no manifest$"):
        application.manifest_for(Path("/"))


def test_selected_languages_lists_every_unknown_and_every_available_language(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        application, "load_manifest", lambda _root: SimpleNamespace(languages=("en", "fr"))
    )

    with pytest.raises(
        typer.BadParameter,
        match=re.escape("unknown language(s) xx, yy; available: en, fr"),
    ):
        application.selected_languages(tmp_path, ["xx", "yy"])


def test_normalize_language_selectors_refuses_an_empty_selector() -> None:
    with pytest.raises(typer.BadParameter, match=r"^language selectors cannot be empty$"):
        application.normalize_language_selectors([" "])


def test_languages_for_scorer_names_the_supported_and_requested_languages() -> None:
    spec = replace(SCORER_ROSTER[0], supported_languages=("en", "fr"))

    with pytest.raises(
        typer.BadParameter,
        match=re.escape(
            f"{spec.model_id} only supports en, fr; unsupported requested languages: de, it"
        ),
    ):
        application.languages_for_scorer(spec, ("en", "de", "it", "fr"), ["de", "it"])


def test_languages_for_scorer_announces_the_languages_it_will_run(capsys) -> None:
    spec = replace(SCORER_ROSTER[0], supported_languages=("en", "fr"))

    chosen = application.languages_for_scorer(spec, ("en", "fr", "de"), None)

    assert chosen == ("en", "fr")
    assert capsys.readouterr().out == (
        f"{spec.model_id} supports en, fr; running those languages only\n"
    )


def test_languages_for_scorer_rejects_when_no_selected_language_is_supported() -> None:
    spec = replace(SCORER_ROSTER[0], supported_languages=("en",))

    with pytest.raises(typer.BadParameter, match=re.escape(spec.model_id)):
        application.languages_for_scorer(spec, ("fr",), None)


# --- prompt files --------------------------------------------------------------------


def test_load_prompts_reads_both_templates_when_the_scorer_prompt_exists(tmp_path: Path) -> None:
    primary = tmp_path / "prompt.txt"
    scorer = tmp_path / "scorer.txt"
    primary.write_text("Classify {}", encoding="utf-8")
    scorer.write_text("Score {}", encoding="utf-8")

    assert application.load_prompts(primary, scorer) == ("Classify {}", "Score {}")


def test_load_prompts_treats_an_absent_scorer_prompt_as_empty(tmp_path: Path) -> None:
    primary = tmp_path / "prompt.txt"
    primary.write_text("Classify {}", encoding="utf-8")

    assert application.load_prompts(primary, tmp_path / "absent.txt") == ("Classify {}", "")


def test_load_prompts_reports_an_unreadable_template_as_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreadable(_path):
        raise PromptFileError("prompt file has no placeholder")

    monkeypatch.setattr(application, "load_prompt", unreadable)

    with pytest.raises(typer.BadParameter, match=r"^prompt file has no placeholder$"):
        application.load_prompts(tmp_path / "p.txt", tmp_path / "s.txt")


def test_extra_scorer_prompts_reports_an_unreadable_prompt_as_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unreadable(_path):
        raise OSError("permission denied")

    monkeypatch.setattr(application, "load_prompt", unreadable)

    with pytest.raises(typer.BadParameter, match=r"^permission denied$"):
        application.extra_scorer_prompts(Path("elsewhere.txt"))


def test_score_one_prints_zero_for_throughput_and_memory_that_were_not_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(
        application,
        "execute_scoring",
        lambda request, provider, *, source_commit: make_result(
            predictions=THIRTEEN,
            duration_seconds=1.5,
            throughput_items_per_second=None,
            peak_vram_bytes=None,
        ),
    )
    monkeypatch.setattr(application, "scorer_provider", lambda: "roster-scorer")

    application.score_one(_request(tmp_path))

    assert capsys.readouterr().out == (
        "m [en]  accuracy=0.538  f1=0.444  mcc=0.192  throughput=0.0/s  peak_vram=0B  (1.5s)\n"
    )


def _named(runtime: str, *, vision: bool = False) -> str:
    return next(
        spec.name
        for spec in ROSTER
        if spec.runtime == runtime and spec.vision == vision and not spec.draft_model_id
    )


@pytest.mark.parametrize(
    ("model", "batch_size", "continuous", "throughput", "expected"),
    [
        ("sglang", None, False, True, (16, False, True)),
        ("transformers", None, False, True, (None, False, False)),
        ("sglang", 4, False, True, (4, False, True)),
        ("transformers", None, True, False, (None, True, False)),
        ("vision", None, True, False, (None, False, False)),
        ("sglang", None, True, False, (None, False, False)),
        ("unlisted", 8, False, False, (8, False, False)),
    ],
)
def test_mode_options_resolve_the_batch_size_and_the_modes_each_runtime_supports(
    model: str, batch_size, *, continuous: bool, throughput: bool, expected
) -> None:
    names = {
        "sglang": _named(SGLANG),
        "transformers": _named(TRANSFORMERS),
        "vision": _named(TRANSFORMERS, vision=True),
        "unlisted": "org/unlisted-model",
    }

    options = application.mode_options(
        names[model],
        batch_size,
        continuous_batching=continuous,
        throughput=throughput,
    )

    assert options == expected


# Pinned before the mode rules moved into the domain: every roster runtime and vision pair,
# plus an unlisted model, against each batch size and mode flag.
_MODE_MODELS = {
    "transformers": _named(TRANSFORMERS),
    "transformers-vision": _named(TRANSFORMERS, vision=True),
    "sglang": _named(SGLANG),
    "sglang-vision": _named(SGLANG, vision=True),
    "unlisted": "org/unlisted-model",
}


@pytest.mark.parametrize(
    ("model", "batch_size", "continuous", "throughput", "expected"),
    [
        ("transformers", None, False, False, (None, False, False)),
        ("transformers", None, False, True, (None, False, False)),
        ("transformers", None, True, False, (None, True, False)),
        ("transformers", None, True, True, (None, True, False)),
        ("transformers", 1, False, False, (1, False, False)),
        ("transformers", 1, False, True, (1, False, False)),
        ("transformers", 1, True, False, (1, True, False)),
        ("transformers", 1, True, True, (1, True, False)),
        ("transformers", 4, False, False, (4, False, False)),
        ("transformers", 4, False, True, (4, False, False)),
        ("transformers", 4, True, False, (4, True, False)),
        ("transformers", 4, True, True, (4, True, False)),
        ("transformers-vision", None, False, False, (None, False, False)),
        ("transformers-vision", None, False, True, (None, False, False)),
        ("transformers-vision", None, True, False, (None, False, False)),
        ("transformers-vision", None, True, True, (None, False, False)),
        ("transformers-vision", 1, False, False, (1, False, False)),
        ("transformers-vision", 1, False, True, (1, False, False)),
        ("transformers-vision", 1, True, False, (1, False, False)),
        ("transformers-vision", 1, True, True, (1, False, False)),
        ("transformers-vision", 4, False, False, (4, False, False)),
        ("transformers-vision", 4, False, True, (4, False, False)),
        ("transformers-vision", 4, True, False, (4, False, False)),
        ("transformers-vision", 4, True, True, (4, False, False)),
        ("sglang", None, False, False, (None, False, False)),
        ("sglang", None, False, True, (16, False, True)),
        ("sglang", None, True, False, (None, False, False)),
        ("sglang", None, True, True, (16, False, True)),
        ("sglang", 1, False, False, (1, False, False)),
        ("sglang", 1, False, True, (1, False, True)),
        ("sglang", 1, True, False, (1, False, False)),
        ("sglang", 1, True, True, (1, False, True)),
        ("sglang", 4, False, False, (4, False, False)),
        ("sglang", 4, False, True, (4, False, True)),
        ("sglang", 4, True, False, (4, False, False)),
        ("sglang", 4, True, True, (4, False, True)),
        ("sglang-vision", None, False, False, (None, False, False)),
        ("sglang-vision", None, False, True, (16, False, True)),
        ("sglang-vision", None, True, False, (None, False, False)),
        ("sglang-vision", None, True, True, (16, False, True)),
        ("sglang-vision", 1, False, False, (1, False, False)),
        ("sglang-vision", 1, False, True, (1, False, True)),
        ("sglang-vision", 1, True, False, (1, False, False)),
        ("sglang-vision", 1, True, True, (1, False, True)),
        ("sglang-vision", 4, False, False, (4, False, False)),
        ("sglang-vision", 4, False, True, (4, False, True)),
        ("sglang-vision", 4, True, False, (4, False, False)),
        ("sglang-vision", 4, True, True, (4, False, True)),
        ("unlisted", None, False, False, (None, False, False)),
        ("unlisted", None, False, True, (None, False, False)),
        ("unlisted", None, True, False, (None, True, False)),
        ("unlisted", None, True, True, (None, True, False)),
        ("unlisted", 1, False, False, (1, False, False)),
        ("unlisted", 1, False, True, (1, False, False)),
        ("unlisted", 1, True, False, (1, True, False)),
        ("unlisted", 1, True, True, (1, True, False)),
        ("unlisted", 4, False, False, (4, False, False)),
        ("unlisted", 4, False, True, (4, False, False)),
        ("unlisted", 4, True, False, (4, True, False)),
        ("unlisted", 4, True, True, (4, True, False)),
    ],
)
def test_mode_options_pins_every_runtime_vision_and_flag_combination(
    model: str,
    batch_size: int | None,
    *,
    continuous: bool,
    throughput: bool,
    expected: tuple[int | None, bool, bool],
) -> None:
    options = application.mode_options(
        _MODE_MODELS[model],
        batch_size,
        continuous_batching=continuous,
        throughput=throughput,
    )

    assert options == expected


def test_report_lines_prints_each_speculative_agreement_with_its_verdict_and_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(application, "leaderboard_rows", lambda _runs: [])
    lossless = SimpleNamespace(
        lossless=True,
        speculative_run="m+DSpark",
        baseline_run="m@sglang",
        language="en",
        n_compared=3,
        n_speculative=4,
        n_baseline=5,
        verdicts_differ=0,
        texts_differ=0,
    )
    mismatch = SimpleNamespace(
        lossless=False,
        speculative_run="n+DSpark",
        baseline_run="n@sglang",
        language="fr",
        n_compared=2,
        n_speculative=2,
        n_baseline=2,
        verdicts_differ=1,
        texts_differ=2,
    )
    monkeypatch.setattr(application, "speculative_agreements", lambda _runs: [lossless, mismatch])

    assert list(application.report_lines([])) == [
        "agreement: m+DSpark [en] vs m@sglang: lossless; coverage=3/4 speculative, "
        "3/5 baseline; verdict_differences=0; text_differences=0",
        "agreement: n+DSpark [fr] vs n@sglang: MISMATCH; coverage=2/2 speculative, "
        "2/2 baseline; verdict_differences=1; text_differences=2",
    ]

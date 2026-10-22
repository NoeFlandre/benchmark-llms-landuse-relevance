"""Usage-error messages, planning lines and checkpoint handling of the CLI use cases."""

import hashlib
import json
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
from landuse_relevance_bench.adapters.results_store import (
    dumps_run_payload,
    run_filename,
    write_run,
)
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

    def resolve(model_id: str, *candidates: str | None) -> str:
        requested = next((candidate for candidate in candidates if candidate), "main")
        if re.fullmatch(r"[0-9a-f]{40}", requested, re.IGNORECASE):
            return requested
        return hashlib.sha256(f"{model_id}:{requested}".encode()).hexdigest()[:40]

    monkeypatch.setattr(application, "resolve_revision", resolve)


def _request(tmp_path: Path, model: str = "m", language: str = "en") -> RunRequest:
    benchmark_path = tmp_path / "b.csv"
    prompt_path = tmp_path / "p.txt"
    benchmark_path.write_text("id,expected,sentence\n1,yes,example\n", encoding="utf-8")
    prompt_path.write_text("Judge this sentence: {}", encoding="utf-8")
    return RunRequest(
        model_id=model,
        language=language,
        benchmark_path=benchmark_path,
        prompt_path=prompt_path,
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
    request = _request(tmp_path)
    meta = SimpleNamespace(
        name="m",
        language="en",
        experiment_identity=application._requested_experiment_identity(request),
    )
    monkeypatch.setattr(application, "read_run", lambda _path: SimpleNamespace(metadata=meta))

    def never_runs():
        raise AssertionError("a complete pair must not run again")

    application.run_pair(request, never_runs, lambda _result: "")

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


@pytest.mark.parametrize(
    ("change", "identity_field"),
    [
        ("prompt", "prompt_sha256"),
        ("dataset", "benchmark_sha256"),
        ("revision", "requested_revision"),
        ("max_new_tokens", "max_new_tokens"),
        ("seed", "seed"),
        ("dtype", "dtype"),
        ("batch_size", "batch_size"),
        ("generation_mode", "generation_mode"),
    ],
)
def test_run_pair_rejects_changed_experiment_identity_without_overwriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    identity_field: str,
) -> None:
    request = _request(tmp_path)
    stored_path = _store_checkpoint(tmp_path)
    original_file = stored_path.read_bytes()
    saved_metadata = SimpleNamespace(
        name=request.name,
        language=request.language,
        experiment_identity=application._requested_experiment_identity(request),
    )
    monkeypatch.setattr(
        application, "read_run", lambda _path: SimpleNamespace(metadata=saved_metadata)
    )

    if change == "prompt":
        request.prompt_path.write_text("Changed prompt: {}", encoding="utf-8")
    elif change == "dataset":
        request.benchmark_path.write_text(
            "id,expected,sentence\n1,no,different example\n", encoding="utf-8"
        )
    else:
        changed_values: dict[str, object] = {
            "revision": "new-revision",
            "max_new_tokens": 123,
            "seed": 7,
            "dtype": "float16",
            "batch_size": 4,
            "generation_mode": True,
        }
        field = "continuous_batching" if change == "generation_mode" else change
        request = replace(request, **{field: changed_values[change]})

    with pytest.raises(typer.BadParameter, match=identity_field):
        application.run_pair(
            request, lambda: pytest.fail("mismatched checkpoint must not run"), lambda _r: ""
        )

    assert stored_path.read_bytes() == original_file


def test_run_pair_rejects_legacy_metadata_without_overwriting(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    stored_path = write_run(make_result(model_id=request.model_id), tmp_path)
    payload = json.loads(stored_path.read_text(encoding="utf-8"))
    payload["metadata"].pop("experiment_identity")
    stored_path.write_text(dumps_run_payload(payload), encoding="utf-8")
    original_file = stored_path.read_bytes()

    with pytest.raises(typer.BadParameter) as error:
        application.run_pair(
            request, lambda: pytest.fail("legacy checkpoint must not run"), lambda _r: ""
        )

    assert str(error.value) == (
        f"{stored_path} has legacy or unsupported metadata without a usable "
        "experiment identity; refusing to reuse or overwrite the stored result"
    )
    assert stored_path.read_bytes() == original_file


def test_run_pair_rejects_incomplete_versioned_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    stored_path = _store_checkpoint(tmp_path)
    saved_metadata = SimpleNamespace(
        name=request.name,
        language=request.language,
        experiment_identity={"schema_version": 2},
    )
    monkeypatch.setattr(
        application, "read_run", lambda _path: SimpleNamespace(metadata=saved_metadata)
    )

    with pytest.raises(typer.BadParameter, match="requested_revision"):
        application.run_pair(
            request, lambda: pytest.fail("incomplete checkpoint must not run"), lambda _r: ""
        )

    assert stored_path.read_text(encoding="utf-8") == "{}"


def test_run_pair_reports_each_changed_identity_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    stored_path = _store_checkpoint(tmp_path)
    original_file = stored_path.read_bytes()
    saved_metadata = SimpleNamespace(
        name=request.name,
        language=request.language,
        experiment_identity=application._requested_experiment_identity(request),
    )
    monkeypatch.setattr(
        application, "read_run", lambda _path: SimpleNamespace(metadata=saved_metadata)
    )
    request = replace(request, revision="new-revision", max_new_tokens=123)

    with pytest.raises(typer.BadParameter) as error:
        application.run_pair(
            request, lambda: pytest.fail("mismatched checkpoint must not run"), lambda _r: ""
        )

    message = str(error.value)
    assert "experiment identity differs in max_new_tokens, requested_revision" in message
    assert "resolved_model_revision" in message
    assert "refusing to reuse or overwrite the stored result" in message
    assert stored_path.read_bytes() == original_file


def test_run_pair_rejects_a_mutable_model_ref_that_resolved_to_a_new_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = replace(_request(tmp_path), revision="main")
    stored_path = _store_checkpoint(tmp_path)
    original_file = stored_path.read_bytes()
    saved_identity = application._requested_experiment_identity(request)
    metadata = SimpleNamespace(
        name=request.name,
        language=request.language,
        experiment_identity=saved_identity,
    )
    monkeypatch.setattr(application, "read_run", lambda _path: SimpleNamespace(metadata=metadata))
    monkeypatch.setattr(application, "resolve_revision", lambda *_args: "b" * 40)

    with pytest.raises(typer.BadParameter, match="resolved_model_revision"):
        application.run_pair(
            request, lambda: pytest.fail("moved model ref must not resume"), lambda _r: ""
        )

    assert stored_path.read_bytes() == original_file


def test_run_pair_rejects_a_mutable_draft_ref_that_resolved_to_a_new_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = replace(
        _request(tmp_path),
        revision="a" * 40,
        draft_model_id="draft/model",
        draft_revision="main",
    )
    stored_path = _store_checkpoint(tmp_path)
    original_file = stored_path.read_bytes()
    saved_identity = application._requested_experiment_identity(request)
    metadata = SimpleNamespace(
        name=request.name,
        language=request.language,
        experiment_identity=saved_identity,
    )
    monkeypatch.setattr(application, "read_run", lambda _path: SimpleNamespace(metadata=metadata))

    def resolve(model_id: str, *candidates: str | None) -> str:
        return "d" * 40 if model_id == "draft/model" else "a" * 40

    monkeypatch.setattr(application, "resolve_revision", resolve)

    with pytest.raises(typer.BadParameter, match="resolved_draft_revision"):
        application.run_pair(
            request, lambda: pytest.fail("moved draft ref must not resume"), lambda _r: ""
        )

    assert stored_path.read_bytes() == original_file


@pytest.mark.parametrize("unresolved_ref", ["target", "draft"])
def test_run_pair_refuses_reuse_when_a_model_ref_cannot_be_resolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unresolved_ref: str,
) -> None:
    request = _request(tmp_path)
    if unresolved_ref == "target":
        request = replace(request, revision="main")
        unresolved_repository = "m"
        message = "immutable revision for model m"
    else:
        request = replace(
            request,
            revision="a" * 40,
            draft_model_id="draft/model",
            draft_revision="main",
        )
        unresolved_repository = "draft/model"
        message = "immutable revision for draft model draft/model"
    stored_path = _store_checkpoint(tmp_path)
    original_file = stored_path.read_bytes()
    metadata = SimpleNamespace(
        name=request.name,
        language=request.language,
        experiment_identity=application._requested_experiment_identity(request),
    )
    monkeypatch.setattr(application, "read_run", lambda _path: SimpleNamespace(metadata=metadata))

    calls: list[tuple[str, tuple[str | None, ...]]] = []

    def resolve(repository: str, *candidates: str | None) -> str:
        calls.append((repository, candidates))
        return "" if repository == unresolved_repository else "a" * 40

    monkeypatch.setattr(application, "resolve_revision", resolve)

    with pytest.raises(typer.BadParameter, match=re.escape(message)) as error:
        application.run_pair(
            request, lambda: pytest.fail("unresolved mutable ref must not resume"), lambda _r: ""
        )

    expected_error = f"could not resolve an {message}; refusing to reuse the stored result"
    assert str(error.value) == expected_error
    if unresolved_ref == "draft":
        assert ("draft/model", ("main",)) in calls
    assert stored_path.read_bytes() == original_file


def test_requested_identity_normalizes_resolved_commit_ids(
    tmp_path: Path,
) -> None:
    request = replace(
        _request(tmp_path),
        revision="A" * 40,
        draft_model_id="draft/model",
        draft_revision="B" * 40,
    )

    identity = application._requested_experiment_identity(request)

    assert identity["resolved_model_revision"] == "a" * 40
    assert identity["resolved_draft_revision"] == "b" * 40


def test_run_pair_preserves_input_error_when_identity_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    request.prompt_path.unlink()
    stored_path = _store_checkpoint(tmp_path)
    metadata = SimpleNamespace(
        name=request.name,
        language=request.language,
        experiment_identity={"schema_version": 2},
    )
    monkeypatch.setattr(application, "read_run", lambda _path: SimpleNamespace(metadata=metadata))

    with pytest.raises(typer.BadParameter) as error:
        application.run_pair(
            request,
            lambda: pytest.fail("identity input failure must stop before run"),
            lambda _r: "",
        )

    assert str(request.prompt_path) in str(error.value)
    assert stored_path.read_text(encoding="utf-8") == "{}"


def test_run_pair_rejects_unusable_result_from_checkpoint_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    stored_path = _store_checkpoint(tmp_path)
    monkeypatch.setattr(application, "_checkpoint_result", lambda *_: ("complete", None))

    with pytest.raises(typer.BadParameter, match="could not be read as a run result"):
        application.run_pair(
            request, lambda: pytest.fail("unusable checkpoint must not run"), lambda _r: ""
        )

    assert stored_path.read_text(encoding="utf-8") == "{}"


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

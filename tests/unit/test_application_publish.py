"""Hub publishing as the CLI use case assembles it: card inputs, upload scope and outputs."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from factories import make_result
from landuse_relevance_bench import application
from landuse_relevance_bench.adapters import hf_publish
from landuse_relevance_bench.adapters.prompt_file import load_prompt
from landuse_relevance_bench.adapters.results_store import write_run
from landuse_relevance_bench.domain.scorers import SCORER_ROSTER

SCORER_PROMPT = next(
    Path(spec.prompt) for spec in SCORER_ROSTER if Path(spec.prompt) != application.DEFAULT_PROMPT
)


def _other_scorer_prompt_texts(primary: Path) -> tuple[str, ...]:
    """Texts of every rostered prompt except the primary and the default, read from disk."""
    paths = sorted({Path(spec.prompt) for spec in SCORER_ROSTER} - {primary, application.DEFAULT_PROMPT})
    return tuple(load_prompt(path) for path in paths if path.is_file())


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch):
    """Replace the Hub adapter's entry points with recorders; returns the recorded calls."""
    recorded: dict[str, object] = {
        "runs": [make_result(model_id="m", language="en")],
        "read_error": None,
    }

    def read_published_runs(results_dir):
        recorded["read_from"] = results_dir
        if recorded["read_error"] is not None:
            raise recorded["read_error"]
        return list(recorded["runs"])

    def dataset_card(runs, *, benchmark_name, prompt_text, options):
        recorded["card"] = {
            "runs": list(runs),
            "benchmark_name": benchmark_name,
            "prompt_text": prompt_text,
            "options": options,
        }
        return "CARD"

    def publish_results(  # noqa: PLR0913 - mirrors the adapter's keyword options
        repo_id, results_dir, runs, *, private, data_root, allow_patterns, **card
    ):
        recorded["publish"] = {
            "repo_id": repo_id,
            "results_dir": results_dir,
            "runs": list(runs),
            "private": private,
            "data_root": data_root,
            "allow_patterns": allow_patterns,
            "card": card,
        }
        return "https://hub.example/org/repo"

    monkeypatch.setattr(hf_publish, "read_published_runs", read_published_runs)
    monkeypatch.setattr(hf_publish, "dataset_card", dataset_card)
    monkeypatch.setattr(hf_publish, "publish_results", publish_results)
    return recorded


@pytest.fixture
def inputs(tmp_path: Path):
    """A results directory, a data root, and prompt templates; returns the keyword arguments."""
    results = tmp_path / "results"
    results.mkdir()
    primary = tmp_path / "prompt.txt"
    scorer = tmp_path / "scorer.txt"
    primary.write_text("Classify {}", encoding="utf-8")
    scorer.write_text("Score {}", encoding="utf-8")
    return {
        "repo_id": "org/repo",
        "data_root": tmp_path / "data",
        "results_dir": results,
        "prompt": primary,
        "scorer_prompt": scorer,
        "benchmark_name": "bench-x",
        "timing_dir": None,
        "language": None,
        "private": False,
        "dry_run": False,
    }


def test_a_dry_run_previews_the_card_built_from_the_same_inputs(hub, inputs) -> None:
    inputs["dry_run"] = True

    output = application.publish_to_hub(**inputs)

    assert output == application.PublishOutput(1, "CARD")
    assert "publish" not in hub
    card = hub["card"]
    assert card["benchmark_name"] == "bench-x"
    assert card["prompt_text"] == "Classify {}"
    options = card["options"]
    assert options.scorer_prompt_text == "Score {}"
    assert options.extra_scorer_prompt_texts == _other_scorer_prompt_texts(inputs["scorer_prompt"])
    assert options.timing_results == ()
    assert options.viewer_file == application.VIEWER_FILE


def test_a_dry_run_writes_no_leaderboard_files(hub, inputs) -> None:
    inputs["dry_run"] = True

    application.publish_to_hub(**inputs)

    assert not (inputs["results_dir"] / "leaderboard.csv").exists()


def test_publishing_uploads_the_runs_with_the_card_and_visibility_it_was_given(hub, inputs) -> None:
    inputs["private"] = True

    output = application.publish_to_hub(**inputs)

    assert output == application.PublishOutput(1, "https://hub.example/org/repo")
    published = hub["publish"]
    assert published["repo_id"] == "org/repo"
    assert published["results_dir"] == inputs["results_dir"]
    assert published["private"] is True
    assert published["data_root"] == inputs["data_root"]
    assert [run.metadata.name for run in published["runs"]] == ["m"]
    card = published["card"]
    assert card["benchmark_name"] == "bench-x"
    assert card["prompt_text"] == "Classify {}"
    assert card["scorer_prompt_text"] == "Score {}"
    assert card["timing_results"] == ()
    assert card["extra_scorer_prompt_texts"] == _other_scorer_prompt_texts(inputs["scorer_prompt"])


def test_publishing_writes_the_leaderboard_files_beside_the_runs(hub, inputs) -> None:
    application.publish_to_hub(**inputs)

    assert (inputs["results_dir"] / "leaderboard.csv").is_file()
    assert (inputs["results_dir"] / "threshold_sweep.csv").is_file()


def test_publishing_uploads_only_the_release_files_and_the_selected_runs(hub, inputs) -> None:
    application.publish_to_hub(**inputs)

    assert hub["publish"]["allow_patterns"] == sorted({*application.RELEASE_FILES, "en/m.json"})


def test_publishing_uploads_the_snapshot_status_when_every_language_is_published(
    hub, inputs
) -> None:
    (inputs["results_dir"] / "SNAPSHOT_STATUS.md").write_text("status", encoding="utf-8")

    application.publish_to_hub(**inputs)

    assert "SNAPSHOT_STATUS.md" in hub["publish"]["allow_patterns"]


def test_publishing_a_language_subset_leaves_the_snapshot_status_out(
    hub, inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    (inputs["results_dir"] / "SNAPSHOT_STATUS.md").write_text("status", encoding="utf-8")
    monkeypatch.setattr(
        application, "load_manifest", lambda _root: SimpleNamespace(languages=("en", "fr"))
    )
    inputs["language"] = ["en"]
    hub["runs"] = [make_result(model_id="m", language="en")]

    application.publish_to_hub(**inputs)

    assert "SNAPSHOT_STATUS.md" not in hub["publish"]["allow_patterns"]


def test_publishing_carries_the_timing_results_it_is_given(hub, inputs, tmp_path: Path) -> None:
    timing = tmp_path / "timing"
    write_run(make_result(model_id="t", language="en"), timing)
    inputs["timing_dir"] = timing

    application.publish_to_hub(**inputs)

    timing_results = hub["publish"]["card"]["timing_results"]
    assert [run.metadata.name for run in timing_results] == ["t"]


def test_publishing_carries_the_scorer_prompt_texts_of_the_other_scorers(hub, inputs) -> None:
    inputs["scorer_prompt"] = SCORER_PROMPT

    application.publish_to_hub(**inputs)

    expected = _other_scorer_prompt_texts(SCORER_PROMPT)
    assert hub["publish"]["card"]["extra_scorer_prompt_texts"] == expected


def test_a_missing_results_directory_is_a_usage_error(hub, inputs) -> None:
    missing = inputs["results_dir"].parent / "missing"
    inputs["results_dir"] = missing

    with pytest.raises(
        typer.BadParameter, match=re.escape(f"no run results directory at {missing}")
    ):
        application.publish_to_hub(**inputs)


def test_unreadable_published_runs_are_a_usage_error(hub, inputs) -> None:
    hub["read_error"] = ValueError("published runs are corrupt")

    with pytest.raises(typer.BadParameter, match=r"^published runs are corrupt$"):
        application.publish_to_hub(**inputs)


def test_publishing_nothing_names_the_results_directory(hub, inputs) -> None:
    hub["runs"] = []

    with pytest.raises(
        typer.BadParameter,
        match=re.escape(f"no run results found in {inputs['results_dir']}"),
    ):
        application.publish_to_hub(**inputs)


def test_publish_allow_patterns_limits_uploads_to_release_files_and_selected_runs(
    tmp_path: Path,
) -> None:
    run = make_result(model_id="m", language="en")

    patterns = application.publish_allow_patterns(tmp_path, [run])

    assert patterns == sorted({*application.RELEASE_FILES, "en/m.json"})


def test_publish_allow_patterns_leaves_the_snapshot_status_out_unless_asked(tmp_path: Path) -> None:
    (tmp_path / "SNAPSHOT_STATUS.md").write_text("status", encoding="utf-8")

    assert "SNAPSHOT_STATUS.md" not in application.publish_allow_patterns(tmp_path, [])
    assert "SNAPSHOT_STATUS.md" in application.publish_allow_patterns(
        tmp_path, [], include_snapshot_status=True
    )


def test_publish_allow_patterns_includes_the_gitattributes_file_when_present(
    tmp_path: Path,
) -> None:
    (tmp_path / ".gitattributes").write_text("*.json binary\n", encoding="utf-8")

    assert ".gitattributes" in application.publish_allow_patterns(tmp_path, [])


def test_a_dry_run_previews_the_timing_results_it_is_given(hub, inputs, tmp_path: Path) -> None:
    timing = tmp_path / "timing"
    write_run(make_result(model_id="t", language="en"), timing)
    inputs["timing_dir"] = timing
    inputs["dry_run"] = True

    application.publish_to_hub(**inputs)

    assert [run.metadata.name for run in hub["card"]["options"].timing_results] == ["t"]

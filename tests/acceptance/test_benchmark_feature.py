"""Executable acceptance criteria for an end-to-end benchmark run."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from pytest_bdd import given, parsers, scenarios, then, when

from landuse_relevance_bench.adapters.benchmark_csv import load_benchmark
from landuse_relevance_bench.adapters.hashing import sha256_of_file, sha256_of_text
from landuse_relevance_bench.adapters.pipeline import RunRequest, execute
from landuse_relevance_bench.adapters.prompt_file import load_prompt
from landuse_relevance_bench.adapters.results_store import (
    aggregate_rows,
    leaderboard_rows,
    read_run,
    read_runs,
    run_filename,
)
from landuse_relevance_bench.adapters.translations import (
    TranslationManifest,
    load_language_benchmark,
    load_manifest,
)
from landuse_relevance_bench.domain.dataset import item_id_for
from landuse_relevance_bench.domain.engine import Generation
from landuse_relevance_bench.domain.sharding import model_language_pairs, shard_pairs

scenarios("features/benchmark.feature")

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


@given("the active multilingual benchmark inventory", target_fixture="manifest")
def _manifest() -> TranslationManifest:
    return load_manifest(Path(__file__).resolve().parents[2] / "data" / "translations")


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
                language="en",
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
    stored = read_run(runs["dir"] / run_filename(GOLD_MODEL, "en"))
    assert [p.item_id for p in stored.predictions] == [i.item_id for i in items]
    assert stored.predictions[0].item_id == item_id_for(items[0].source_item_id, items[0].language)


@then("the run recalls every relevant sentence")
def _full_recall(runs) -> None:
    assert runs["runs"][GOLD_MODEL].metrics.recall == 1.0


@then("the run misses no relevant sentence")
def _no_misses(runs) -> None:
    assert runs["runs"][GOLD_MODEL].metrics.confusion.false_negative == 0


@then("the raw generations are kept in the stored result")
def _raw_kept(runs) -> None:
    stored = read_run(runs["dir"] / run_filename(GOLD_MODEL, "en"))
    assert {p.raw_output for p in stored.predictions} == {"I cannot decide"}


@then("the stored result marks every generation as truncated")
def _truncation_recorded(runs) -> None:
    stored = read_run(runs["dir"] / run_filename(GOLD_MODEL, "en"))
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
    metadata = read_run(runs["dir"] / run_filename(GOLD_MODEL, "en")).metadata
    assert metadata.benchmark_sha256 == sha256_of_file(benchmark)
    assert metadata.prompt_sha256 == sha256_of_text(load_prompt(prompt))


@then("the stored result names the model revision that was used")
def _revision(runs) -> None:
    assert (
        read_run(runs["dir"] / run_filename(GOLD_MODEL, "en")).metadata.model_revision == "stubrev"
    )


@then("85 languages are available with 300 rows each")
def _language_counts(manifest: TranslationManifest) -> None:
    assert len(manifest.languages) == 85
    assert all(manifest.files[language].rows == 300 for language in manifest.languages)


@then("every language shares the same source identity sequence")
def _aligned_source_ids(manifest: TranslationManifest) -> None:
    root = Path(__file__).resolve().parents[2] / "data" / "translations"
    expected = tuple(item.source_item_id for item in load_language_benchmark(root, "en"))
    for language in manifest.languages:
        items = load_language_benchmark(root, language)
        assert tuple(item.source_item_id for item in items) == expected
        assert len({item.item_id for item in items}) == 300


@given("a gold model run for each of two languages", target_fixture="multilingual_runs")
def _multilingual_runs(real_prompt_path: Path, tmp_path: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2] / "data" / "translations"
    results_dir = tmp_path / "results"
    runs = []
    for language in ("en", "fr"):
        benchmark = root / language / f"v3-final-{language}.csv"
        items = load_benchmark(benchmark)
        gold = {item.sentence: item.label.value for item in items}
        generator = OracleGenerator(gold)
        result = execute(
            RunRequest(
                model_id=GOLD_MODEL,
                language=language,
                benchmark_path=benchmark,
                prompt_path=real_prompt_path,
                output_dir=results_dir,
                batch_size=32,
            ),
            lambda _request, current_generator=generator: (current_generator, "stubrev"),
        )
        runs.append(result)
    return {"dir": results_dir, "runs": runs}


@when("I read the active results", target_fixture="active_results")
def _read_active_results(multilingual_runs: dict[str, Any]):
    directory = cast(Path, multilingual_runs["dir"])
    return {
        "dir": directory,
        "runs": read_runs(directory),
    }


@then("each language has its own checkpoint")
def _language_checkpoints(active_results) -> None:
    directory = active_results["dir"]
    assert (directory / run_filename(GOLD_MODEL, "en")).is_file()
    assert (directory / run_filename(GOLD_MODEL, "fr")).is_file()
    assert {result.metadata.language for result in active_results["runs"]} == {"en", "fr"}


@then("the model aggregate covers both languages")
def _aggregate_covers_languages(active_results) -> None:
    rows = aggregate_rows(active_results["runs"])
    assert rows == [
        {
            "model_id": GOLD_MODEL,
            "language_count": 2,
            "n_items_total": 600,
            "accuracy_macro": 1.0,
            "balanced_accuracy_macro": 1.0,
            "f1_macro": 1.0,
            "precision_macro": 1.0,
            "recall_macro": 1.0,
            "matthews_corrcoef_macro": 1.0,
            "unparsed_rate_macro": 0.0,
            "f1_min": 1.0,
            "f1_max": 1.0,
            "f1_std": 0.0,
        }
    ]


@given("an active result directory with an archived invalid file", target_fixture="archived_dir")
def _archived_dir(tmp_path: Path) -> Path:
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "old.json").write_text("not active JSON", encoding="utf-8")
    return tmp_path


@when("I read the active archive directory", target_fixture="active_archive_results")
def _read_archive_results(archived_dir: Path):
    return read_runs(archived_dir)


@then("the archived file is ignored")
def _archive_ignored(active_archive_results) -> None:
    assert active_archive_results == []


@given("a multilingual roster and four shard slots", target_fixture="shard_plan")
def _shard_plan() -> tuple[tuple[str, str], ...]:
    pairs = model_language_pairs(("model/a", "model/b", "model/c"), ("de", "en", "fr"))
    shards = [shard_pairs(pairs, index, 4) for index in range(4)]
    flattened = tuple(pair for shard in shards for pair in shard)
    assert len(flattened) == len(set(flattened))
    return flattened


@then("the shard union is complete and disjoint")
def _shards_complete(shard_plan: tuple[tuple[str, str], ...]) -> None:
    expected = model_language_pairs(("model/a", "model/b", "model/c"), ("de", "en", "fr"))
    assert set(shard_plan) == set(expected)
    assert len(shard_plan) == len(expected)

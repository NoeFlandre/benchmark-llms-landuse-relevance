"""Executable acceptance criteria for an end-to-end benchmark run."""

from collections.abc import Sequence
from pathlib import Path

from pytest_bdd import given, parsers, scenarios, then, when

from landuse_relevance_bench.adapters.benchmark_csv import load_benchmark
from landuse_relevance_bench.adapters.hashing import sha256_of_file, sha256_of_text
from landuse_relevance_bench.adapters.pipeline import RunRequest, execute
from landuse_relevance_bench.adapters.prompt_file import load_prompt
from landuse_relevance_bench.adapters.results_store import leaderboard_rows, read_run, run_filename
from landuse_relevance_bench.domain.dataset import item_id_for
from landuse_relevance_bench.domain.engine import Generation

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

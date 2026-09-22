from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from landuse_relevance_bench.adapters.hashing import sha256_of_file
from landuse_relevance_bench.adapters.pipeline import DEFAULT_MAX_NEW_TOKENS, RunRequest, execute
from landuse_relevance_bench.adapters.results_store import read_run, run_filename
from landuse_relevance_bench.domain.engine import LabelScores
from landuse_relevance_bench.domain.labels import Label


class StubGenerator:
    def __init__(self, outputs: Sequence[str]) -> None:
        self._outputs = list(outputs)
        self.prompts: list[str] = []

    def generate(self, prompts: Sequence[str]) -> Sequence[str]:
        self.prompts.extend(prompts)
        taken, self._outputs = self._outputs[: len(prompts)], self._outputs[len(prompts) :]
        return taken


class MeasuredScorer:
    peak_vram_bytes = 123456

    def begin_measurement(self) -> None:
        self.started = True

    def end_measurement(self) -> None:
        self.finished = True

    def score(self, inputs):
        return [LabelScores({Label.YES: 0.9, Label.NO: 0.1}) for _ in inputs]


@pytest.fixture
def request_for(tmp_path: Path, benchmark_path: Path, prompt_path: Path):
    def build(**overrides: Any) -> RunRequest:
        return replace(
            RunRequest(
                model_id="LiquidAI/LFM2.5-350M",
                language="en",
                benchmark_path=benchmark_path,
                prompt_path=prompt_path,
                output_dir=tmp_path / "results",
            ),
            **overrides,
        )

    return build


class StubProvider:
    """A generator provider that keeps the generator reachable for assertions."""

    def __init__(self, outputs: Sequence[str], revision: str = "rev0") -> None:
        self.generator = StubGenerator(outputs)
        self._revision = revision

    def __call__(self, _: RunRequest) -> tuple[StubGenerator, str]:
        return self.generator, self._revision


def _provider(outputs: Sequence[str], revision: str = "rev0") -> StubProvider:
    return StubProvider(outputs, revision)


def test_scores_the_whole_benchmark_and_returns_the_result(request_for) -> None:
    result = execute(request_for(), _provider(["yes", "no"]))
    assert result.metrics.n_items == 2
    assert result.metrics.accuracy == 1.0
    assert [p.predicted for p in result.predictions] == [Label.YES, Label.NO]


def test_writes_one_result_file_named_after_the_model(request_for, tmp_path: Path) -> None:
    result = execute(request_for(), _provider(["yes", "no"]))
    path = tmp_path / "results" / run_filename("LiquidAI/LFM2.5-350M", "en")
    assert read_run(path) == result


def test_records_the_inputs_it_actually_used(request_for, benchmark_path: Path) -> None:
    result = execute(request_for(batch_size=1, max_new_tokens=4, seed=7), _provider(["yes", "no"]))
    metadata = result.metadata
    assert metadata.benchmark_sha256 == sha256_of_file(benchmark_path)
    assert metadata.model_revision == "rev0"
    assert metadata.language == "en"
    assert metadata.batch_size == 1
    assert metadata.max_new_tokens == 4
    assert metadata.seed == 7
    assert metadata.decoding == "greedy"
    assert metadata.duration_seconds >= 0.0


def test_active_default_generation_budget_is_4096(request_for) -> None:
    result = execute(request_for(), _provider(["yes", "no"]))

    assert DEFAULT_MAX_NEW_TOKENS == 4096
    assert result.metadata.max_new_tokens == 4096


def test_feeds_the_file_prompt_to_the_generator(request_for) -> None:
    provider = _provider(["yes", "no"])
    execute(request_for(), provider)
    assert provider.generator.prompts[0].startswith("Classify.")
    assert provider.generator.prompts[0].endswith("Dense mangrove forest lines the lagoon.")


def test_unparsable_generations_survive_into_the_stored_result(request_for) -> None:
    result = execute(request_for(), _provider(["I cannot tell", "no"]))
    assert result.predictions[0].predicted is None
    assert result.predictions[0].raw_output == "I cannot tell"
    assert result.metrics.unparsed_rate == 0.5


def test_scoring_records_throughput_and_peak_vram(request_for) -> None:
    scorer = MeasuredScorer()
    request = request_for(model_id="Alibaba-NLP/gte-multilingual-reranker-base")

    from landuse_relevance_bench.adapters.pipeline import execute_scoring

    result = execute_scoring(request, lambda _: (scorer, "gte-rev"))

    assert scorer.started and scorer.finished
    assert result.metadata.model_revision == "gte-rev"
    assert result.metadata.sequence_length == 8192
    assert result.metadata.throughput_items_per_second > 0.0
    assert result.metadata.peak_vram_bytes == 123456


def test_scoring_records_the_scorer_specific_sequence_length(request_for) -> None:
    scorer = MeasuredScorer()
    scorer.sequence_length = 1024
    request = request_for(model_id="convaiinnovations/laya-multilingual")

    from landuse_relevance_bench.adapters.pipeline import execute_scoring

    result = execute_scoring(request, lambda _: (scorer, "laya-rev"))

    assert result.metadata.sequence_length == 1024

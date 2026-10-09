import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from landuse_relevance_bench.adapters import pipeline
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
    peak_vram_bytes: int = 123456
    sequence_length: int = 8192

    def begin_measurement(self) -> None:
        self.started = True

    def end_measurement(self) -> None:
        self.finished = True

    def score(self, inputs):
        return [LabelScores({Label.YES: 0.9, Label.NO: 0.1}) for _ in inputs]


class RuntimeDtypeScorer(MeasuredScorer):
    runtime_dtype = "float16"


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
    assert result.metadata.throughput_items_per_second is not None
    assert result.metadata.throughput_items_per_second > 0.0
    assert result.metadata.peak_vram_bytes == 123456


def test_scoring_records_the_scorer_specific_sequence_length(request_for) -> None:
    scorer = MeasuredScorer()
    scorer.sequence_length = 1024
    request = request_for(model_id="convaiinnovations/laya-multilingual")

    from landuse_relevance_bench.adapters.pipeline import execute_scoring

    result = execute_scoring(request, lambda _: (scorer, "laya-rev"))

    assert result.metadata.sequence_length == 1024


def test_scoring_records_a_scorer_runtime_dtype_when_provided(request_for) -> None:
    scorer = RuntimeDtypeScorer()
    request = request_for(model_id="convaiinnovations/laya-multilingual")

    from landuse_relevance_bench.adapters.pipeline import execute_scoring

    result = execute_scoring(request, lambda _: (scorer, "laya-rev"))

    assert result.metadata.dtype == "float16"


def test_execute_closes_a_model_once_after_success(request_for) -> None:
    class CloseableGenerator(StubGenerator):
        def __init__(self) -> None:
            super().__init__(["yes", "no"])
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    generator = CloseableGenerator()
    result = execute(request_for(), lambda _: (generator, "rev0"))

    assert result.metrics.n_items == 2
    assert generator.close_calls == 1


def test_execute_closes_a_model_after_prediction_failure_without_masking_it(request_for) -> None:
    class FailingGenerator(StubGenerator):
        def __init__(self) -> None:
            super().__init__([])
            self.close_calls = 0

        def generate(self, prompts: Sequence[str]) -> Sequence[str]:
            raise RuntimeError("generation failed")

        def close(self) -> None:
            self.close_calls += 1
            raise OSError("cleanup failed")

    generator = FailingGenerator()

    with pytest.raises(RuntimeError, match="generation failed"):
        execute(request_for(), lambda _: (generator, "rev0"))

    assert generator.close_calls == 1


def test_execute_respects_a_caller_owned_generator(request_for) -> None:
    class CloseableGenerator(StubGenerator):
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    generator = CloseableGenerator(["yes", "no"])
    result = execute(
        request_for(close_generator=False),
        lambda _: (generator, "rev0"),
    )

    assert result.metrics.n_items == 2
    assert generator.close_calls == 0


def test_device_name_handles_missing_torch_cpu_and_cuda(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    assert pipeline._device_name() == ""

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert pipeline._device_name() == ""

    torch.cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "A100")
    assert pipeline._device_name() == "A100"


class BareScorer:
    """Declares only the required method, so every optional capability takes its default."""

    def score(self, inputs):
        return [LabelScores({Label.YES: 0.9, Label.NO: 0.1}) for _ in inputs]


class PropertyScorer(BareScorer):
    """Declares its optional capabilities as read-only properties, not plain attributes."""

    @property
    def peak_vram_bytes(self) -> int:
        return 777

    @property
    def runtime_dtype(self) -> str:
        return "bfloat16"

    @property
    def sequence_length(self) -> None:
        return None


class StartOnlyScorer(BareScorer):
    started = False

    def begin_measurement(self) -> None:
        self.started = True


def test_scoring_defaults_apply_when_the_scorer_declares_no_optional_capability(
    request_for,
) -> None:
    from landuse_relevance_bench.adapters.pipeline import execute_scoring

    request = request_for(model_id="convaiinnovations/laya-multilingual")
    result = execute_scoring(request, lambda _: (BareScorer(), "bare-rev"))

    assert result.metadata.sequence_length == pipeline.SCORING_SEQUENCE_LENGTH
    assert result.metadata.peak_vram_bytes is None
    assert result.metadata.dtype == request.dtype


def test_scoring_reads_optional_capabilities_declared_as_properties(request_for) -> None:
    from landuse_relevance_bench.adapters.pipeline import execute_scoring

    request = request_for(model_id="convaiinnovations/laya-multilingual")
    result = execute_scoring(request, lambda _: (PropertyScorer(), "prop-rev"))

    assert result.metadata.peak_vram_bytes == 777
    assert result.metadata.dtype == "bfloat16"
    assert result.metadata.sequence_length is None


def test_scoring_calls_a_begin_hook_without_requiring_an_end_hook(request_for) -> None:
    from landuse_relevance_bench.adapters.pipeline import execute_scoring

    scorer = StartOnlyScorer()
    execute_scoring(
        request_for(model_id="convaiinnovations/laya-multilingual"),
        lambda _: (scorer, "start-rev"),
    )

    assert scorer.started


def test_generation_records_the_generator_declared_batch_size_dtype_and_draft(request_for) -> None:
    class ReportingGenerator(StubGenerator):
        effective_batch_size = 2
        runtime_dtype = "float16"
        draft_revision = "draft-rev"

    result = execute(
        request_for(batch_size=1, draft_revision="requested-draft"),
        lambda _: (ReportingGenerator(["yes", "no"]), "rev0"),
    )

    assert result.metadata.batch_size == 2
    assert result.metadata.dtype == "float16"
    assert result.metadata.draft_model_revision == "draft-rev"


def test_generation_falls_back_to_the_request_when_the_generator_declares_nothing(
    request_for,
) -> None:
    request = request_for(batch_size=1, draft_revision="requested-draft")
    result = execute(request, _provider(["yes", "no"]))

    assert result.metadata.batch_size == 1
    assert result.metadata.dtype == request.dtype
    assert result.metadata.draft_model_revision == "requested-draft"


def test_an_empty_generator_draft_revision_defers_to_the_request(request_for) -> None:
    class EmptyDraftGenerator(StubGenerator):
        draft_revision = ""

    result = execute(
        request_for(draft_revision="requested-draft"),
        lambda _: (EmptyDraftGenerator(["yes", "no"]), "rev0"),
    )

    assert result.metadata.draft_model_revision == "requested-draft"

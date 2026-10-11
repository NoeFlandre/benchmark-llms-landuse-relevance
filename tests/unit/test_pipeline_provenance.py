"""What a run records about itself: request construction, generation mode and metadata fields."""

import logging
import re
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import landuse_relevance_bench
from factories import ScriptedGenerator
from landuse_relevance_bench.adapters import pipeline
from landuse_relevance_bench.adapters.pipeline import RunRequest, execute, execute_scoring
from landuse_relevance_bench.domain.engine import LabelScores
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.orchestration import DEFAULT_BATCH_SIZE
from landuse_relevance_bench.domain.roster import ROSTER, SGLANG, TRANSFORMERS, quantization_of
from landuse_relevance_bench.domain.scorers import SCORER_ROSTER, scorer_for

LOGGER = "landuse_relevance_bench.adapters.pipeline"
MODEL = "LiquidAI/LFM2.5-350M"
QUANTIZED = "unsloth/Qwen3.8-27B-GGUF@UD-IQ2_XXS"
PATHS = {
    "language": "en",
    "benchmark_path": Path("en.csv"),
    "prompt_path": Path("prompt.txt"),
    "output_dir": Path("out"),
}


def _for_run(name: str, **options: object) -> RunRequest:
    """``RunRequest.for_run`` with the path fields every caller supplies."""
    return RunRequest.for_run(name, **PATHS, **options)


def _roster(runtime: str, *, vision: bool = False, draft: bool = False):
    return next(
        spec.name
        for spec in ROSTER
        if spec.runtime == runtime and spec.vision == vision and bool(spec.draft_model_id) == draft
    )


@pytest.fixture
def request_for(tmp_path: Path, benchmark_path: Path, prompt_path: Path):
    def build(**overrides: object) -> RunRequest:
        return replace(
            RunRequest(
                model_id=MODEL,
                language="en",
                benchmark_path=benchmark_path,
                prompt_path=prompt_path,
                output_dir=tmp_path / "results",
            ),
            **overrides,
        )

    return build


def _provider(generator, revision: str = "rev0"):
    return lambda _request: (generator, revision)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch):
    """Control the two monotonic readings a timed run takes; returns a setter."""

    def set_duration(start: float, end: float) -> None:
        ticks = iter((start, end))
        monkeypatch.setattr(pipeline, "time", SimpleNamespace(monotonic=lambda: next(ticks)))

    return set_duration


def test_metadata_records_the_identity_and_provenance_of_the_request(
    request_for, monkeypatch
) -> None:
    monkeypatch.setattr(pipeline, "free_gpu_memory_bytes", lambda: None)
    request = request_for(
        run_id="m@variant",
        runtime=SGLANG,
        draft_model_id="draft/model",
        draft_revision="draft-rev",
        speculative={"k": 2},
        seed=5,
        max_new_tokens=7,
    )

    metadata = execute(
        request, _provider(ScriptedGenerator(["yes", "no"]), "rev-9"), source_commit="abc123"
    ).metadata

    assert metadata.model_id == MODEL
    assert metadata.run_id == "m@variant"
    assert metadata.name == "m@variant"
    assert metadata.model_revision == "rev-9"
    assert metadata.runtime == SGLANG
    assert metadata.draft_model_id == "draft/model"
    assert metadata.draft_model_revision == "draft-rev"
    assert metadata.speculative == {"k": 2}
    assert metadata.source_commit == "abc123"
    assert metadata.package_version == landuse_relevance_bench.__version__
    assert metadata.seed == 5
    assert metadata.max_new_tokens == 7
    assert metadata.quantization == ""


def test_metadata_takes_the_device_name_from_the_timing_host(request_for, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "free_gpu_memory_bytes", lambda: None)
    monkeypatch.setattr(pipeline, "_device_name", lambda: "GPU-Z")

    metadata = execute(request_for(), _provider(ScriptedGenerator(["yes", "no"]))).metadata

    assert metadata.device_name == "GPU-Z"


def test_metadata_defaults_the_source_commit_to_empty(request_for, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "free_gpu_memory_bytes", lambda: None)

    metadata = execute(request_for(), _provider(ScriptedGenerator(["yes", "no"]))).metadata

    assert metadata.source_commit == ""


def test_metadata_records_the_quantization_of_a_quantized_roster_model(
    request_for, monkeypatch
) -> None:
    monkeypatch.setattr(pipeline, "free_gpu_memory_bytes", lambda: None)
    assert quantization_of(QUANTIZED) == "UD-IQ2_XXS"

    metadata = execute(
        request_for(model_id=QUANTIZED), _provider(ScriptedGenerator(["yes", "no"]))
    ).metadata

    assert metadata.quantization == "UD-IQ2_XXS"


def test_metadata_leaves_the_draft_revision_empty_when_none_is_named(
    request_for, monkeypatch
) -> None:
    monkeypatch.setattr(pipeline, "free_gpu_memory_bytes", lambda: None)

    metadata = execute(request_for(), _provider(ScriptedGenerator(["yes", "no"]))).metadata

    assert metadata.draft_model_revision == ""


@pytest.mark.parametrize(
    ("overrides", "mode"),
    [
        ({"continuous_batching": True}, "transformers-continuous-batching"),
        ({"throughput_mode": True}, "sglang-throughput"),
        ({}, "static-batched"),
    ],
)
def test_metadata_names_the_generation_mode_of_the_run(
    request_for, monkeypatch, overrides: dict, mode: str
) -> None:
    monkeypatch.setattr(pipeline, "free_gpu_memory_bytes", lambda: None)

    metadata = execute(
        request_for(**overrides), _provider(ScriptedGenerator(["yes", "no"]))
    ).metadata

    assert metadata.generation_mode == mode


def test_metadata_reports_throughput_and_duration_from_the_measured_clock(
    request_for, monkeypatch, clock
) -> None:
    monkeypatch.setattr(pipeline, "free_gpu_memory_bytes", lambda: None)
    clock(10.0, 10.5)

    metadata = execute(request_for(), _provider(ScriptedGenerator(["yes", "no"]))).metadata

    assert metadata.duration_seconds == 0.5
    assert metadata.throughput_items_per_second == 4.0


def test_metadata_reports_no_throughput_for_a_run_with_no_measurable_duration(
    request_for, monkeypatch, clock
) -> None:
    monkeypatch.setattr(pipeline, "free_gpu_memory_bytes", lambda: None)
    clock(5.0, 5.0)

    metadata = execute(request_for(), _provider(ScriptedGenerator(["yes", "no"]))).metadata

    assert metadata.duration_seconds == 0.0
    assert metadata.throughput_items_per_second is None


def test_metadata_rounds_the_duration_to_milliseconds(request_for, monkeypatch, clock) -> None:
    monkeypatch.setattr(pipeline, "free_gpu_memory_bytes", lambda: None)
    clock(10.0, 11.23456)

    metadata = execute(request_for(), _provider(ScriptedGenerator(["yes", "no"]))).metadata

    assert metadata.duration_seconds == 1.235


def test_metadata_stamps_the_start_time_in_utc_to_the_second(request_for, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "free_gpu_memory_bytes", lambda: None)
    real = datetime(2026, 1, 2, 3, 4, 5, 789000, tzinfo=UTC)

    class FixedClock:
        @staticmethod
        def now(tz):
            assert tz is UTC, "the start time must be read in UTC"
            return real

    monkeypatch.setattr(pipeline, "datetime", FixedClock)

    metadata = execute(request_for(), _provider(ScriptedGenerator(["yes", "no"]))).metadata

    assert metadata.started_at == "2026-01-02T03:04:05Z"


def test_device_name_asks_the_first_cuda_device_by_index(monkeypatch) -> None:
    asked: list[int] = []
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda index: asked.append(index) or "A100",
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert pipeline._device_name() == "A100"
    assert asked == [0]


def test_a_scorer_that_reports_no_peak_memory_records_none(request_for, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "free_gpu_memory_bytes", lambda: None)
    spec = scorer_for(SCORER_ROSTER[0].model_id)

    class NoPeak:
        peak_vram_bytes = None

        def score(self, inputs):
            return [LabelScores({Label.YES: 0.9, Label.NO: 0.1}) for _ in inputs]

    result = execute_scoring(
        request_for(model_id=spec.model_id, language="en"), lambda _: (NoPeak(), "rev")
    )

    assert result.metadata.peak_vram_bytes is None


def test_scoring_metadata_records_the_scorer_rule_and_no_decoding(request_for) -> None:
    spec = SCORER_ROSTER[0]

    class Scorer:
        def score(self, inputs):
            return [LabelScores({Label.YES: 0.9, Label.NO: 0.1}) for _ in inputs]

    result = execute_scoring(
        request_for(model_id=spec.model_id), lambda _request: (Scorer(), "rev")
    )

    metadata = result.metadata
    assert metadata.inference == "scoring"
    assert metadata.decision_rule == spec.decision_rule
    assert metadata.max_new_tokens == 0
    assert metadata.decoding == "none"
    assert metadata.source_commit == ""
    assert metadata.generation_mode == "static-batched"


def test_scoring_hands_the_provider_the_request_and_batches_its_inputs(request_for) -> None:
    spec = SCORER_ROSTER[0]
    seen: list[object] = []
    sizes: list[int] = []

    class Scorer:
        def score(self, inputs):
            sizes.append(len(inputs))
            return [LabelScores({Label.YES: 0.9, Label.NO: 0.1}) for _ in inputs]

    request = request_for(model_id=spec.model_id, batch_size=1)

    def provide(received):
        seen.append(received)
        return Scorer(), "rev"

    execute_scoring(request, provide)

    assert seen == [request]
    assert sizes == [1, 1]


def test_scoring_uses_the_default_batch_size_when_none_is_given(request_for) -> None:
    spec = SCORER_ROSTER[0]
    sizes: list[int] = []

    class Scorer:
        def score(self, inputs):
            sizes.append(len(inputs))
            return [LabelScores({Label.YES: 0.9, Label.NO: 0.1}) for _ in inputs]

    execute_scoring(request_for(model_id=spec.model_id), lambda _request: (Scorer(), "rev"))

    assert sizes == [2]


def test_for_run_warns_about_an_unlisted_model_and_runs_it_plainly(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        request = _for_run("org/unlisted-model")

    assert caplog.messages == [
        "org/unlisted-model is not in the benchmark roster; running it on "
        f"{TRANSFORMERS} with no pinned revision and the default batch size"
    ]
    assert request.model_id == "org/unlisted-model"
    assert request.runtime == TRANSFORMERS
    assert request.revision is None
    assert request.batch_size == DEFAULT_BATCH_SIZE
    assert request.run_id == ""


def test_for_run_refuses_continuous_batching_together_with_throughput_mode() -> None:
    with pytest.raises(
        ValueError,
        match=r"^choose either continuous batching or SGLang throughput mode$",
    ):
        _for_run(_roster(TRANSFORMERS), continuous_batching=True, throughput_mode=True)


def test_for_run_names_a_throughput_variant_and_keeps_its_mode() -> None:
    name = _roster(SGLANG)

    request = _for_run(name, throughput_mode=True)

    spec = next(spec for spec in ROSTER if spec.name == name)
    assert request.throughput_mode is True
    assert request.continuous_batching is False
    assert request.batch_size == DEFAULT_BATCH_SIZE
    assert request.run_id == f"{spec.name}-throughput-b{DEFAULT_BATCH_SIZE}"


def test_for_run_names_a_continuous_batching_variant_and_keeps_its_mode() -> None:
    name = _roster(TRANSFORMERS)

    request = _for_run(name, continuous_batching=True)

    spec = next(spec for spec in ROSTER if spec.name == name)
    assert request.continuous_batching is True
    assert request.throughput_mode is False
    assert request.run_id == f"{spec.name}@continuous-b{DEFAULT_BATCH_SIZE}"


def test_for_run_refuses_a_mode_the_runtime_cannot_run() -> None:
    with pytest.raises(ValueError, match=re.escape("throughput mode requires the SGLang runtime")):
        _for_run(_roster(TRANSFORMERS), throughput_mode=True)
    with pytest.raises(
        ValueError, match=re.escape("continuous batching requires the Transformers runtime")
    ):
        _for_run(_roster(SGLANG), continuous_batching=True)


def test_for_run_keeps_a_plain_roster_run_unchanged() -> None:
    name = _roster(TRANSFORMERS)

    request = _for_run(name)

    assert request.run_id == ""
    assert request.continuous_batching is False
    assert request.throughput_mode is False
    assert request.model_id == name

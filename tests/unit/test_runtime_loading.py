"""Loaders and dispatch for the optional runtimes, with those runtimes faked."""

import sys
import types
from pathlib import Path

import pytest

from landuse_relevance_bench import cli
from landuse_relevance_bench.adapters import hf_generator, hf_scorer, llama_generator
from landuse_relevance_bench.adapters.pipeline import RunRequest
from landuse_relevance_bench.domain.scorers import GENERATIVE_PROMPT, ZEROSHOT_PROMPT

GGUF_ID = "unsloth/Qwen3.8-27B-GGUF@UD-IQ2_XXS"


def _request(model_id: str) -> RunRequest:
    return RunRequest(
        model_id=model_id,
        language="en",
        benchmark_path=Path("b.csv"),
        prompt_path=Path("p.txt"),
        output_dir=Path("out"),
        max_new_tokens=8,
    )


class _FakeLlama:
    metadata = {"tokenizer.chat_template": "{{ messages[0].content }}"}

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def detokenize(self, ids, special=False):
        return b""


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    downloads: list[tuple] = []

    class HfApi:
        def model_info(self, repository):
            return types.SimpleNamespace(sha="resolved-sha")

    def hf_hub_download(repository, filename, revision=None):
        downloads.append((repository, filename, revision))
        return f"/cache/{filename}"

    hub = types.SimpleNamespace(
        HfApi=HfApi,
        hf_hub_download=hf_hub_download,
        snapshot_download=lambda repository, revision=None: "/cache/snapshots/abc",
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=_FakeLlama))
    return downloads


def test_a_gguf_roster_id_downloads_its_quant_and_offloads_every_layer(fake_hub) -> None:
    generator, revision = llama_generator.provide(_request(GGUF_ID))

    assert revision == "resolved-sha"
    assert fake_hub == [("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-IQ2_XXS.gguf", "resolved-sha")]
    assert generator._llama.kwargs["n_gpu_layers"] == -1
    assert generator._llama.kwargs["model_path"] == "/cache/Qwen3.8-27B-UD-IQ2_XXS.gguf"


def test_the_generator_provider_dispatches_quants_to_llama_cpp(fake_hub) -> None:
    generator, _ = hf_generator.provide(_request(GGUF_ID))

    assert isinstance(generator, llama_generator.LlamaCppGenerator)


def test_publish_collects_every_other_scorer_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[Path] = []
    monkeypatch.setattr(cli, "load_prompt", lambda path: loaded.append(path) or str(path))
    monkeypatch.setattr(Path, "is_file", lambda self: True)

    prompts = cli._extra_scorer_prompts(cli.DEFAULT_SCORER_PROMPT)

    assert Path(ZEROSHOT_PROMPT) in loaded
    assert cli.DEFAULT_SCORER_PROMPT not in loaded
    assert Path(GENERATIVE_PROMPT) not in loaded
    assert prompts == tuple(str(path) for path in loaded)


def test_the_nli_loader_builds_the_zero_shot_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict = {}

    class Loader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return types.SimpleNamespace(
                model_max_length=10**9, config=types.SimpleNamespace(_commit_hash="nli-sha")
            )

    def pipeline(task, **kwargs):
        calls["task"] = task
        return types.SimpleNamespace(model=kwargs["model"])

    fake = types.SimpleNamespace(
        AutoTokenizer=Loader, AutoModelForSequenceClassification=Loader, pipeline=pipeline
    )
    monkeypatch.setattr(hf_scorer, "_load_transformers", lambda: fake)
    monkeypatch.setattr(hf_scorer, "_torch_device", lambda: "cpu")

    scorer = hf_scorer.NliZeroShotScorer.load("m/nli", hf_scorer.ScorerSettings(dtype="float32"))

    assert calls["task"] == "zero-shot-classification"
    assert scorer.revision == "nli-sha"


def test_the_gliner2_loader_uses_the_auto_extractor(
    fake_hub, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Extractor:
        def eval(self):
            return self

    extractor = types.SimpleNamespace(from_pretrained=lambda path, map_location: Extractor())
    monkeypatch.setitem(sys.modules, "gliner2", types.SimpleNamespace(AutoExtractor=extractor))
    monkeypatch.setattr(hf_scorer, "_torch_device", lambda: "cpu")

    scorer = hf_scorer.Gliner2Scorer.load("fastino/x", hf_scorer.ScorerSettings(dtype="float32"))

    assert scorer.revision == "abc"

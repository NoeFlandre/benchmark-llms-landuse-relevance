"""The zero-shot, log-probability and GGUF adapters keep their intended interfaces."""

import pytest

from landuse_relevance_bench.adapters.hf_scorer import (
    CausalLogprobScorer,
    GliClassScorer,
    Gliner2Scorer,
    NliZeroShotScorer,
    ScorerSettings,
    scorer_class_for,
    zeroshot_hypothesis,
)
from landuse_relevance_bench.adapters.llama_generator import (
    LlamaCppGenerator,
    LlamaSettings,
    render_chat,
)
from landuse_relevance_bench.domain.engine import ScoringInput
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.records import RunMetadata
from landuse_relevance_bench.domain.scorers import repository_of, scorer_for

HYPOTHESIS = "It is about land use."
PROMPT = f"TARGET SENTENCE: forest\n\nHYPOTHESIS: {HYPOTHESIS}\n"


def _inputs(*sentences: str) -> list[ScoringInput]:
    return [ScoringInput(PROMPT.replace("forest", s), s) for s in sentences]


def test_new_scorers_select_their_intended_adapters() -> None:
    assert scorer_class_for("LiquidAI/LFM2.5-2.6B@logprob") is CausalLogprobScorer
    assert scorer_class_for("knowledgator/gliclass-multilang-mini") is GliClassScorer
    assert scorer_class_for("fastino/gliner2.5-multi-v1") is Gliner2Scorer
    for model_id in (
        "MoritzLaurer/bge-m3-zeroshot-v2.0",
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        "BalaRajesh1/mmbert-small-nli",
    ):
        assert scorer_class_for(model_id) is NliZeroShotScorer
        assert scorer_for(model_id).prompt == "data/prompt_zeroshot.txt"


def test_a_variant_id_loads_its_plain_repository() -> None:
    assert repository_of("LiquidAI/LFM2.5-2.6B@logprob") == "LiquidAI/LFM2.5-2.6B"
    assert scorer_for("LiquidAI/LFM2.5-2.6B@logprob").prompt == "data/prompt.txt"


def test_the_hypothesis_is_read_from_the_prompt_file() -> None:
    assert zeroshot_hypothesis(PROMPT) == HYPOTHESIS
    with pytest.raises(ValueError, match="HYPOTHESIS"):
        zeroshot_hypothesis("TARGET SENTENCE: forest")


class FakeNliPipeline:
    model = None

    def __init__(self) -> None:
        self.kwargs: dict = {}

    def __call__(self, sequences, **kwargs):
        self.kwargs = kwargs
        return [
            {"labels": [HYPOTHESIS], "scores": [0.9 if s == "forest" else 0.2]} for s in sequences
        ]


def test_nli_scores_the_sentence_as_premise_against_one_hypothesis() -> None:
    pipeline = FakeNliPipeline()
    scores = NliZeroShotScorer(pipeline, "rev").score(_inputs("forest", "mayor"))

    assert [s.verdict for s in scores] == [Label.YES, Label.NO]
    assert scores[0].scores[Label.YES] == pytest.approx(0.9)
    assert pipeline.kwargs["candidate_labels"] == [HYPOTHESIS]
    assert pipeline.kwargs["hypothesis_template"] == "{}"
    assert pipeline.kwargs["multi_label"] is True


def test_gliclass_reads_the_hypothesis_label_score() -> None:
    def pipeline(texts, labels, **_):
        return [[{"label": labels[0], "score": 0.7 if t == "forest" else 0.1}] for t in texts]

    scores = GliClassScorer(pipeline, "rev").score(_inputs("forest", "mayor"))
    assert [round(s.scores[Label.YES], 2) for s in scores] == [0.7, 0.1]


def test_gliner2_reads_the_label_confidence() -> None:
    class FakeExtractor:
        def classify_text(self, text, tasks, include_confidence):
            label = tasks["landuse"]["labels"][0]
            return {"landuse": [{"label": label, "confidence": 0.8 if text == "forest" else 0.3}]}

    scores = Gliner2Scorer(FakeExtractor(), "rev").score(_inputs("forest", "mayor"))
    assert [s.verdict for s in scores] == [Label.YES, Label.NO]


class FakeTokenizer:
    vocabulary = {"yes": 1, "no": 2, "Yes": 3, "No": 4}

    def encode(self, text, add_special_tokens=False):
        return [self.vocabulary[text]] if text in self.vocabulary else [9, 9]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return (
            f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n<think>"
        )


def test_logprob_turn_closes_the_think_block_so_the_verdict_comes_next() -> None:
    scorer = CausalLogprobScorer(FakeTokenizer(), None, ScorerSettings(dtype="bfloat16"))
    assert scorer.as_chat("Q").endswith("assistant\n<think></think>")
    assert scorer._yes_ids == [1, 3]
    assert scorer._no_ids == [2, 4]


def test_gguf_turn_is_rendered_from_the_embedded_template_without_thinking() -> None:
    template = (
        "{{ bos_token }}{% for m in messages %}<u>{{ m.content }}</u>{% endfor %}"
        "{% if add_generation_prompt %}<a>{% if not enable_thinking %}<think></think>"
        "{% endif %}{% endif %}"
    )
    assert render_chat(template, "Q", bos_token="<s>") == "<s><u>Q</u><a><think></think>"


class FakeLlama:
    metadata = {"tokenizer.chat_template": "{{ messages[0].content }}"}

    def __init__(self, finish: str) -> None:
        self.finish = finish
        self.calls: list[dict] = []

    def reset(self) -> None: ...

    def detokenize(self, ids, special=False):
        return b""

    def create_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"text": " yes ", "finish_reason": self.finish}]}


def test_gguf_generation_is_greedy_and_reports_truncation() -> None:
    done = LlamaCppGenerator(FakeLlama("stop"), LlamaSettings(max_new_tokens=16, seed=0), "rev")
    cut = LlamaCppGenerator(FakeLlama("length"), LlamaSettings(max_new_tokens=16, seed=0), "rev")

    (finished,) = done.generate(["Q"])
    (truncated,) = cut.generate(["Q"])

    assert (finished.text, finished.truncated) == ("yes", False)
    assert truncated.truncated is True
    call = done._llama.calls[0]
    assert (call["temperature"], call["top_k"], call["max_tokens"]) == (0.0, 1, 16)


def _metadata(**changes) -> RunMetadata:
    from dataclasses import replace

    base = RunMetadata(
        model_id="m/x",
        language="en",
        model_revision="r",
        prompt_sha256="p",
        benchmark_sha256="b",
        max_new_tokens=1,
        batch_size=1,
        seed=0,
        decoding="greedy",
        dtype="bfloat16",
        started_at="t",
        duration_seconds=1.0,
    )
    return replace(base, **changes)


def test_quantization_is_omitted_for_full_precision_runs_to_keep_their_bytes() -> None:
    assert "quantization" not in _metadata().to_dict()
    quant = _metadata(quantization="UD-IQ2_XXS", dtype="gguf")
    assert RunMetadata.from_dict(quant.to_dict()) == quant


def test_gliner2_records_its_window_as_model_defined() -> None:
    assert Gliner2Scorer.sequence_length is None

"""The non-generative models scored on the same benchmark.

These are kept apart from :mod:`domain.roster` on purpose. A generative model is
prompted and its text is parsed into a verdict; a scoring model returns a native
score per label and never writes text. They answer the same question, but not by
the same method, so they are rostered, run and reported separately.
"""

from dataclasses import dataclass

#: How a scoring model's native scores become a verdict. Recorded on every scoring
#: run, because without it a scoring row cannot be compared to a generative one.
ARGMAX = "argmax over the native yes/no scores"
RELEVANCE_THRESHOLD = "sigmoid of the native relevance logit at 0.5"
MXBAI_RELEVANCE_THRESHOLD = "sigmoid(native yes-minus-no logit minus 4.5) at 0.5"
LAYA_RELEVANCE_THRESHOLD = "Laya noul yes probability at 0.5"
LOGPROB_ARGMAX = "argmax over the first-token yes/no log-probabilities"
ENTAILMENT_THRESHOLD = "zero-shot pipeline entailment probability at 0.5"
LABEL_THRESHOLD = "zero-shot label probability at 0.5"

#: Prompt files, relative to the project root. Each scorer names the one it is asked
#: with, so a run's recorded prompt digest always matches its model's intended input.
RERANKER_PROMPT = "data/prompt_reranker.txt"
GENERATIVE_PROMPT = "data/prompt.txt"
ZEROSHOT_PROMPT = "data/prompt_zeroshot.txt"

#: Separates a roster id from the variant it names (``repo@variant``), so two methods
#: on one checkpoint write distinct result files.
VARIANT_SEPARATOR = "@"


@dataclass(frozen=True, slots=True)
class ScorerSpec:
    """One non-generative model, and how it is asked for a verdict."""

    model_id: str
    total_parameters: int
    kind: str
    decision_rule: str
    note: str
    prompt: str = RERANKER_PROMPT

    @property
    def repository(self) -> str:
        """The Hub repository to load; the roster id minus any ``@variant``."""
        return repository_of(self.model_id)


SCORER_ROSTER: tuple[ScorerSpec, ...] = (
    ScorerSpec(
        "Alibaba-NLP/gte-multilingual-reranker-base",
        306_000_000,
        "sequence-classifier",
        RELEVANCE_THRESHOLD,
        "Multilingual sequence classifier; sigmoid of its relevance logit is the yes score.",
    ),
    ScorerSpec(
        "mixedbread-ai/mxbai-rerank-base-v2",
        500_000_000,
        "reranker",
        MXBAI_RELEVANCE_THRESHOLD,
        "Binary relevance reranker; scores the official 1/0 continuation and normalises it.",
    ),
    ScorerSpec(
        "convaiinnovations/laya-multilingual",
        322_000_000,
        "typed-decision-model",
        LAYA_RELEVANCE_THRESHOLD,
        "Non-autoregressive multilingual decision model; scores a typed noul question.",
    ),
    ScorerSpec(
        "Qwen/Qwen3-Reranker-0.6B",
        595_800_000,
        "reranker",
        ARGMAX,
        "Relevance reranker; scores the yes/no continuation, never generates.",
    ),
    ScorerSpec(
        "Qwen/Qwen3-Reranker-4B",
        4_021_800_000,
        "reranker",
        ARGMAX,
        "Relevance reranker; scores the yes/no continuation, never generates.",
    ),
    ScorerSpec(
        "LiquidAI/LFM2.5-2.6B@logprob",
        2_697_198_592,
        "causal-lm-logprob",
        LOGPROB_ARGMAX,
        "The generative LFM2.5-2.6B, read from its first-token yes/no log-probabilities.",
        GENERATIVE_PROMPT,
    ),
    ScorerSpec(
        "knowledgator/gliclass-multilang-mini",
        283_736_833,
        "zero-shot-classifier",
        LABEL_THRESHOLD,
        "GLiClass multilingual zero-shot classifier; one land-use label.",
        ZEROSHOT_PROMPT,
    ),
    ScorerSpec(
        "MoritzLaurer/bge-m3-zeroshot-v2.0",
        567_756_802,
        "nli-cross-encoder",
        ENTAILMENT_THRESHOLD,
        "BGE-M3 entailment classifier through the zero-shot-classification pipeline.",
        ZEROSHOT_PROMPT,
    ),
    ScorerSpec(
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        278_812_163,
        "nli-cross-encoder",
        ENTAILMENT_THRESHOLD,
        "mDeBERTa-v3 multilingual NLI through the zero-shot-classification pipeline.",
        ZEROSHOT_PROMPT,
    ),
    ScorerSpec(
        "BalaRajesh1/mmbert-small-nli",
        140_642_691,
        "nli-cross-encoder",
        ENTAILMENT_THRESHOLD,
        "mmBERT-small multilingual NLI through the zero-shot-classification pipeline.",
        ZEROSHOT_PROMPT,
    ),
    ScorerSpec(
        "fastino/gliner2.5-multi-v1",
        287_355_159,
        "schema-classifier",
        LABEL_THRESHOLD,
        "GLiNER2.5 multilingual schema classifier; one land-use label.",
        ZEROSHOT_PROMPT,
    ),
)


def repository_of(model_id: str) -> str:
    """Strip a ``@variant`` suffix, leaving the loadable Hub repository id."""
    return model_id.split(VARIANT_SEPARATOR, 1)[0]


def scorer_ids() -> tuple[str, ...]:
    return tuple(spec.model_id for spec in SCORER_ROSTER)


def scorer_for(model_id: str) -> ScorerSpec:
    for spec in SCORER_ROSTER:
        if spec.model_id == model_id:
            return spec
    raise KeyError(f"{model_id!r} is not in the scoring roster")

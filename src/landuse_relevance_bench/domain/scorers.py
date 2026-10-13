"""The non-generative models scored on the same benchmark.

These are kept apart from :mod:`domain.roster` on purpose. A generative model is
prompted and its text is parsed into a verdict; a scoring model returns a native
score per label and never writes text. They answer the same question, but not by
the same method, so they are rostered, run and reported separately.
"""

from dataclasses import dataclass

from landuse_relevance_bench.domain.variants import repository_of

#: How a scoring model's native scores become a verdict. Recorded on every scoring
#: run, because without it a scoring row cannot be compared to a generative one.
ARGMAX = "argmax over the native yes/no scores"
D1_ARGMAX = f"{ARGMAX}; exact ties resolve to no"
RELEVANCE_THRESHOLD = "sigmoid of the native relevance logit at 0.5"
#: mxbai-rerank-v2's published estimate of its maximum logit difference; the official
#: normaliser subtracts it before the sigmoid.
MXBAI_LOGIT_OFFSET = 4.5
MXBAI_RELEVANCE_THRESHOLD = f"sigmoid(native yes-minus-no logit minus {MXBAI_LOGIT_OFFSET}) at 0.5"
LAYA_RELEVANCE_THRESHOLD = "Laya noul yes probability at 0.5"
LOGPROB_ARGMAX = "argmax over the first-token yes/no log-probabilities"
ENTAILMENT_THRESHOLD = "zero-shot pipeline entailment probability at 0.5"
LABEL_THRESHOLD = "zero-shot label probability at 0.5"
GLICLASS_ULTRA_ID = "knowledgator/gliclass-multilang-ultra"
GLICLASS_ULTRA_REVISION = "33c3ffec64056b0d37173d2dd90b885ac29b72c3"
GLICLASS_ULTRA_NATIVE_LANGUAGES = (
    "ar",
    "cs",
    "de",
    "es",
    "et",
    "fi",
    "fr",
    "he",
    "hi",
    "it",
    "lt",
    "lv",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "sv",
    "uk",
    "zh",
)
GLICLASS_ULTRA_THRESHOLD = 0.5
GLICLASS_ULTRA_NATIVE_THRESHOLD = (
    f"GLiClass multi-label sigmoid label score >= {GLICLASS_ULTRA_THRESHOLD}; "
    "exact ties resolve to yes"
)
D1_MODEL_CARD_LANGUAGE_COUNT = 16

#: Prompt files, relative to the project root. Each scorer names the one it is asked
#: with, so a run's recorded prompt digest always matches its model's intended input.
RERANKER_PROMPT = "data/prompt_reranker.txt"
GENERATIVE_PROMPT = "data/prompt.txt"
ZEROSHOT_PROMPT = "data/prompt_zeroshot.txt"
MASKED_LM_PROMPT = "data/prompt_masked_lm.txt"


#: Card description for a scoring run whose model is no longer rostered.
DEFAULT_CARD = ("scoring adapter", "recorded prompt + sentence", "normalized yes score")
LOGPROB_KIND = "causal-lm-logprob"
LFM25_ENCODER_LANGUAGES = (
    "ar",
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "it",
    "ja",
    "nl",
    "pl",
    "pt",
    "ru",
    "tr",
    "vi",
    "zh",
)


@dataclass(frozen=True, slots=True)
class ScorerSpec:
    """One non-generative model, and how it is asked for a verdict."""

    model_id: str
    total_parameters: int
    kind: str
    decision_rule: str
    note: str
    prompt: str = RERANKER_PROMPT
    #: How the published card describes the adapter: (family, input handling, score).
    card: tuple[str, str, str] = DEFAULT_CARD
    #: A model's declared language coverage, or None for the full benchmark inventory.
    supported_languages: tuple[str, ...] | None = None
    #: Whether the checkpoint's own Python code may run. Off unless the model needs it.
    #: The loader requires a full commit SHA, but that pins the model repository only: a
    #: model whose ``auto_map`` names another Hub repo is not safe to opt in until that repo
    #: is pinned too.
    trust_remote_code: bool = False
    #: Optional immutable model snapshot used when a run is created.
    revision: str | None = None

    @property
    def repository(self) -> str:
        """The Hub repository to load; the roster id minus any ``@variant``."""
        return repository_of(self.model_id)


SCORER_ROSTER: tuple[ScorerSpec, ...] = (
    # trust_remote_code stays off. The config.json auto_map names Alibaba-NLP/new-impl for
    # its modeling code, another repo that the model-revision pin does not cover, so GTE
    # is refused until that repo is pinned too (issue #120).
    ScorerSpec(
        "Alibaba-NLP/gte-multilingual-reranker-base",
        306_000_000,
        "sequence-classifier",
        RELEVANCE_THRESHOLD,
        "Multilingual sequence classifier; sigmoid of its relevance logit is the yes score.",
        card=("sequence classifier", "prompt + sentence pair", "sigmoid relevance logit"),
    ),
    ScorerSpec(
        "mixedbread-ai/mxbai-rerank-base-v2",
        500_000_000,
        "reranker",
        MXBAI_RELEVANCE_THRESHOLD,
        "Binary relevance reranker; scores the official 1/0 continuation and normalises it.",
        card=(
            "causal-LM reranker",
            "official query/document turn",
            "sigmoid(1-logit - 0-logit - 4.5)",
        ),
    ),
    ScorerSpec(
        "convaiinnovations/laya-multilingual",
        322_000_000,
        "typed-decision-model",
        LAYA_RELEVANCE_THRESHOLD,
        "Non-autoregressive multilingual decision model; scores a typed noul question.",
        card=(
            "typed decision model",
            "JSON state + 4 `noul` questions/call",
            "Laya `noul` yes probability",
        ),
    ),
    # The pinned config maps AutoModel to modeling_d1.D1Model in this same repository.
    # Its reviewed System One code loads the tokenizer from name_or_path, so the adapter
    # first materializes this exact snapshot and passes that local path to Transformers.
    ScorerSpec(
        "LiquidAI/d1-3B",
        3_123_483_888,
        "typed-decision-model",
        D1_ARGMAX,
        "Single-pass typed yes/no decision model; official documentation lists 16 languages.",
        GENERATIVE_PROMPT,
        card=(
            "typed decision model",
            "sentence state + one `noul` question/request",
            "d1 `noul` P(yes)",
        ),
        trust_remote_code=True,
        revision="051bcc464b01b9f92942b364d9586b0ef5912432",
    ),
    ScorerSpec(
        "Qwen/Qwen3-Reranker-0.6B",
        595_800_000,
        "reranker",
        ARGMAX,
        "Relevance reranker; scores the yes/no continuation, never generates.",
        card=("causal-LM reranker", "manual yes/no reranker turn", "yes/no next-token probability"),
    ),
    ScorerSpec(
        "Qwen/Qwen3-Reranker-4B",
        4_021_800_000,
        "reranker",
        ARGMAX,
        "Relevance reranker; scores the yes/no continuation, never generates.",
        card=("causal-LM reranker", "manual yes/no reranker turn", "yes/no next-token probability"),
    ),
    ScorerSpec(
        "LiquidAI/LFM2.5-2.6B@logprob",
        2_697_198_592,
        LOGPROB_KIND,
        LOGPROB_ARGMAX,
        "The generative LFM2.5-2.6B, read from its first-token yes/no log-probabilities.",
        GENERATIVE_PROMPT,
        card=(
            "causal LM, no decoding",
            "LLM prompt + chat turn, empty think block",
            "first-token P(yes) vs P(no)",
        ),
    ),
    ScorerSpec(
        "knowledgator/gliclass-multilang-mini",
        283_736_833,
        "zero-shot-classifier",
        LABEL_THRESHOLD,
        "GLiClass multilingual zero-shot classifier; one land-use label.",
        ZEROSHOT_PROMPT,
        card=("GLiClass zero-shot", "sentence + hypothesis label", "label probability"),
    ),
    ScorerSpec(
        GLICLASS_ULTRA_ID,
        1_720_000_000,
        "zero-shot-classifier",
        GLICLASS_ULTRA_NATIVE_THRESHOLD,
        "GLiClass Multilang Ultra; native training coverage is 20 languages, evaluated on all 85.",
        ZEROSHOT_PROMPT,
        card=("GLiClass multi-label", "sentence + one English hypothesis", "sigmoid label score"),
        revision=GLICLASS_ULTRA_REVISION,
    ),
    ScorerSpec(
        "MoritzLaurer/bge-m3-zeroshot-v2.0",
        567_756_802,
        "nli-cross-encoder",
        ENTAILMENT_THRESHOLD,
        "BGE-M3 entailment classifier through the zero-shot-classification pipeline.",
        ZEROSHOT_PROMPT,
        card=("NLI zero-shot pipeline", "sentence premise + hypothesis", "entailment probability"),
    ),
    ScorerSpec(
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        278_812_163,
        "nli-cross-encoder",
        ENTAILMENT_THRESHOLD,
        "mDeBERTa-v3 multilingual NLI through the zero-shot-classification pipeline.",
        ZEROSHOT_PROMPT,
        card=("NLI zero-shot pipeline", "sentence premise + hypothesis", "entailment probability"),
    ),
    ScorerSpec(
        "BalaRajesh1/mmbert-small-nli",
        140_642_691,
        "nli-cross-encoder",
        ENTAILMENT_THRESHOLD,
        "mmBERT-small multilingual NLI through the zero-shot-classification pipeline.",
        ZEROSHOT_PROMPT,
        card=("NLI zero-shot pipeline", "sentence premise + hypothesis", "entailment probability"),
    ),
    ScorerSpec(
        "fastino/gliner2.5-multi-v1",
        287_355_159,
        "schema-classifier",
        LABEL_THRESHOLD,
        "GLiNER2.5 multilingual schema classifier; one land-use label.",
        ZEROSHOT_PROMPT,
        card=("GLiNER2 classify_text", "sentence + hypothesis label", "label confidence"),
    ),
    ScorerSpec(
        "LiquidAI/LFM2.5-Encoder-350M",
        354_500_000,
        "masked-lm",
        "argmax over the masked-token yes/no logits",
        "Bidirectional masked-LM encoder; reads the yes/no verbalizer at one mask.",
        MASKED_LM_PROMPT,
        card=(
            "bidirectional masked-LM encoder",
            "task prompt + one mask; 15 supported languages",
            "yes/no masked-token logits",
        ),
        supported_languages=LFM25_ENCODER_LANGUAGES,
    ),
)


def logprob_pairs() -> dict[str, str]:
    """Each log-probability scorer mapped to the generative roster id it reads."""
    return {spec.model_id: spec.repository for spec in SCORER_ROSTER if spec.kind == LOGPROB_KIND}


def scorer_ids() -> tuple[str, ...]:
    return tuple(spec.model_id for spec in SCORER_ROSTER)


def scorer_for(model_id: str) -> ScorerSpec:
    for spec in SCORER_ROSTER:
        if spec.model_id == model_id:
            return spec
    raise KeyError(f"{model_id!r} is not in the scoring roster")

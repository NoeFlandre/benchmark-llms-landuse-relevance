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


@dataclass(frozen=True, slots=True)
class ScorerSpec:
    """One non-generative model, and how it is asked for a verdict."""

    model_id: str
    total_parameters: int
    kind: str
    decision_rule: str
    note: str


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
)
# knowledgator/gliclass-multilang-mini is deliberately not rostered yet. The adapter for
# it exists, but rostering a model nothing runs would make the published coverage claim
# count languages that were never attempted.


def scorer_ids() -> tuple[str, ...]:
    return tuple(spec.model_id for spec in SCORER_ROSTER)


def scorer_for(model_id: str) -> ScorerSpec:
    for spec in SCORER_ROSTER:
        if spec.model_id == model_id:
            return spec
    raise KeyError(f"{model_id!r} is not in the scoring roster")

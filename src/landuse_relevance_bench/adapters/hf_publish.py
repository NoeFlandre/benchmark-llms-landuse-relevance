"""Publishing run results to a Hugging Face dataset repository."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from landuse_relevance_bench.adapters.hashing import sha256_of_text
from landuse_relevance_bench.adapters.hf_scorer_prompt import RERANKER_SYSTEM
from landuse_relevance_bench.adapters.results_store import (
    AGGREGATE_COLUMNS,
    aggregate_rows,
    read_runs,
)
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import RunResult, outcomes_of

CARD_COLUMNS = AGGREGATE_COLUMNS


class DatasetHub(Protocol):
    """The slice of :class:`huggingface_hub.HfApi` this project depends on."""

    def create_repo(self, **kwargs: Any) -> Any: ...
    def upload_folder(self, **kwargs: Any) -> Any: ...


def read_published_runs(results_dir: Path) -> tuple[RunResult, ...]:
    """Read every result below ``results_dir`` in stable model-id order."""
    runs = read_runs(results_dir)
    return tuple(
        sorted(runs, key=lambda result: (result.metadata.model_id, result.metadata.language))
    )


def _uniform(results: Sequence[RunResult], name: str, value_of: Any) -> Any:
    """Return the single value every run agrees on, or refuse to describe the sweep."""
    values = {value_of(result) for result in results}
    if len(values) != 1:
        raise ValueError(f"runs disagree on {name}: {sorted(values)}")
    return values.pop()


def dataset_card(
    results: Sequence[RunResult],
    *,
    benchmark_name: str,
    prompt_text: str,
    scorer_prompt_text: str = "",
) -> str:
    """Build a terse card whose scores are recomputed from every prediction."""
    if not results:
        raise ValueError("cannot build a card from an empty set of results")
    for result in results:
        derived = evaluate(outcomes_of(result.predictions))
        if derived != result.metrics:
            raise ValueError(f"{result.metadata.model_id} metrics do not match predictions")
    # Generative and scoring models are given different inputs on purpose, so each
    # family's prompt is checked against the runs that actually used it.
    generative_pre = [r for r in results if r.metadata.is_generative]
    if not generative_pre:
        raise ValueError("cannot build a card without any generative run")
    prompt_sha256 = _uniform(generative_pre, "prompt digest", lambda r: r.metadata.prompt_sha256)
    if sha256_of_text(prompt_text) != prompt_sha256:
        raise ValueError("prompt text does not match the digest recorded in the runs")
    # Generation settings describe how a generative model was prompted. A scoring
    # model is never prompted for text, so asserting them over every run would make
    # the two families look like one method described badly.
    generative = generative_pre
    scoring = [r for r in results if not r.metadata.is_generative]
    scorer_prompt_sha256 = ""
    if scoring:
        scorer_prompt_sha256 = _uniform(
            scoring, "scoring prompt digest", lambda r: r.metadata.prompt_sha256
        )
        if sha256_of_text(scorer_prompt_text) != scorer_prompt_sha256:
            raise ValueError(
                "scoring prompt text does not match the digest recorded in the scoring runs"
            )
    decoding = _uniform(generative, "decoding", lambda r: r.metadata.decoding)
    dtype = _uniform(generative, "dtype", lambda r: r.metadata.dtype)
    seed = _uniform(generative, "seed", lambda r: r.metadata.seed)
    max_new_tokens = _uniform(generative, "token budget", lambda r: r.metadata.max_new_tokens)
    # Batch size is a throughput knob, not a decoding setting, and one rostered model
    # cannot be batched at all: Falcon-H1 is a hybrid attention-SSM model and its state
    # handling breaks under the left padding batching needs. So the card reports the
    # batch size when the sweep agrees on one and points at the runs when it does not,
    # rather than refusing to describe a sweep over the settings that shape a verdict.
    batch_sizes = sorted({result.metadata.batch_size for result in generative})
    batch_size_line = (
        f"batch size {batch_sizes[0]}"
        if len(batch_sizes) == 1
        else "batch size varying by model, recorded per run"
    )
    n_items = _uniform(results, "items per language", lambda r: r.metrics.n_items)
    languages = sorted({result.metadata.language for result in results})
    header = "| " + " | ".join(CARD_COLUMNS) + " |"
    divider = "|" + "|".join(["---"] * len(CARD_COLUMNS)) + "|"

    def table(subset: Sequence[RunResult]) -> str:
        return "\n".join(
            "| " + " | ".join(str(row[column]) for column in CARD_COLUMNS) + " |"
            for row in aggregate_rows(subset)
        )

    body = table(generative)
    scoring_section = (
        ""
        if not scoring
        else _scoring_section(
            scoring,
            header=header,
            divider=divider,
            table=table,
            prompt_text=scorer_prompt_text,
            prompt_sha256=scorer_prompt_sha256,
        )
    )
    return f"""---
license: mit
task_categories:
- text-classification
tags:
- land-use
- land-cover
- remote-sensing
- llm-benchmark
---

# Land-use relevance benchmark

`{benchmark_name}`; model-level macro averages over the completed languages.
Scores are recomputed from the published predictions. `language_count` reports how many
language checkpoints contributed to each model row.

- Code: https://github.com/NoeFlandre/benchmark-llms-landuse-relevance

## Benchmark

Binary sentence classification: does the sentence describe its target place in a way that
helps characterise land use, land cover or the geographic environment from remote sensing?
The label set is the two lowercase tokens `yes` and `no`.

- {len(languages)} languages, {n_items} labelled sentences each, one result file per
  model and language.
- {decoding} decoding, seed {seed}, `max_new_tokens={max_new_tokens}`, dtype `{dtype}`,
  {batch_size_line}.
- The verdict is the last standalone `yes` or `no` in the generation. A generation that
  reaches the token budget without stopping is recorded as truncated and yields no
  verdict; `unparsed_rate_macro` reports how often that happened.
- Prompt sha256 `{prompt_sha256}`. Each language has its own benchmark file, whose
  `benchmark_sha256` is recorded in every run it produced.

Prompt template, used verbatim for every language, where `{{}}` is replaced by the target
sentence:

```text
{prompt_text}```

## Aggregate scores

{header}
{divider}
{body}
{scoring_section}"""


def _scoring_section(
    scoring: Sequence[RunResult],
    *,
    header: str,
    divider: str,
    table: Any,
    prompt_text: str,
    prompt_sha256: str,
) -> str:
    """Report the non-generative models apart, with the rule that produced them."""
    rules = sorted({result.metadata.decision_rule or "unrecorded" for result in scoring})
    joined = "; ".join(rules)
    batch_sizes = sorted({result.metadata.batch_size for result in scoring})
    batch_line = (
        f"batch size {batch_sizes[0]}"
        if len(batch_sizes) == 1
        else "batch size varying by model, recorded per run"
    )
    return f"""
## Scoring models

These are rerankers, not chat models: nothing is generated and no text is parsed. Each
item is fed as one relevance judgement, and the verdict is read from the model's own
next-token scores for `yes` and `no`, softmaxed over just those two tokens. The verdict
is then taken by: {joined}. Every prediction stores the scores it came from, as
`no=<score> yes=<score>`, so a different rule — a threshold rather than an argmax, say —
can be recomputed from the published results without re-running anything.

Because a scoring model cannot produce unparseable or unfinished text,
`unparsed_rate_macro` is zero for every row below by construction. That is a property
of the method, not a comparison won against the generative models above.

Read these rows with their decision rule in mind. A reranker's score is trained to
rank documents for retrieval, where almost nothing is relevant, so it is not calibrated
to a 0.5 boundary on a roughly balanced task: taking the argmax makes it answer `no`
almost always, which lowers F1 far more than it reflects how well the score separates
the two classes. The published per-item scores are what to use for that question, and
they support recomputing any other rule without re-running the models.

`threshold_sweep.csv` reports exactly that: every metric for each model at a range of
boundaries, plus `roc_auc_macro`, which does not depend on a boundary at all. Read the
best row there as an upper bound rather than a score, because the boundary that
produces it was chosen on this same benchmark.

- dtype `{_uniform(scoring, "scoring dtype", lambda r: r.metadata.dtype)}`, {batch_line},
  seed {_uniform(scoring, "scoring seed", lambda r: r.metadata.seed)}.
- Scoring prompt sha256 `{prompt_sha256}`.

The reranker is given this system turn, which is the one it was trained to judge under:

```text
{RERANKER_SYSTEM}
```

and this user turn, where `{{}}` is replaced by the target sentence:

```text
{prompt_text}```

{header}
{divider}
{table(scoring)}
"""


def publish_results(
    repo_id: str,
    results_dir: Path,
    results: Sequence[RunResult],
    *,
    api: DatasetHub | None = None,
    private: bool = False,
    commit_message: str = "Publish small-LLM land-use relevance benchmark results",
    benchmark_name: str = "v3-multilingual",
    prompt_text: str,
    scorer_prompt_text: str = "",
) -> str:
    """Write the card next to the results, then push the whole folder to the Hub."""
    if not results:
        raise ValueError("refusing to publish an empty set of results")
    _reject_archive_paths(results_dir)
    hub = api if api is not None else _default_api()
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "README.md").write_text(
        dataset_card(
            results,
            benchmark_name=benchmark_name,
            prompt_text=prompt_text,
            scorer_prompt_text=scorer_prompt_text,
        ),
        encoding="utf-8",
    )
    hub.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    hub.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(results_dir),
        commit_message=commit_message,
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def _default_api() -> DatasetHub:
    from huggingface_hub import HfApi

    # HfApi satisfies DatasetHub in practice; its **kwargs signatures are wider than
    # the protocol can express.
    return cast(DatasetHub, HfApi())


def _reject_archive_paths(results_dir: Path) -> None:
    paths = results_dir.rglob("*") if results_dir.exists() else ()
    for path in paths:
        if "archive" in path.relative_to(results_dir).parts:
            raise ValueError(f"refusing to upload archive path: {path}")

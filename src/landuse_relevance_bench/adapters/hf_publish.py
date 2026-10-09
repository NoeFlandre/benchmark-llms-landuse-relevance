"""Publishing run results to a Hugging Face dataset repository."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from landuse_relevance_bench.adapters.hashing import sha256_of_text
from landuse_relevance_bench.adapters.results_store import (
    ARCHIVE_COMPONENT,
    aggregate_rows,
    group_by_model,
    read_runs,
    scoring_summary_rows,
)
from landuse_relevance_bench.domain.agreement import speculative_agreements
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import RunResult, outcomes_of
from landuse_relevance_bench.domain.scorers import DEFAULT_CARD, logprob_pairs, scorer_for
from landuse_relevance_bench.domain.selection import collection_comparability_errors
from landuse_relevance_bench.domain.speed import summarise_speed

CARD_COLUMNS = (
    "model_id",
    "language_count",
    "accuracy_macro",
    "balanced_accuracy_macro",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "matthews_corrcoef_macro",
)

SPEED_COLUMNS = (
    "model_id",
    "runtime",
    "device",
    "generation_mode",
    "batch_size",
    "language_count",
    "cumulative_wall_seconds",
    "sentences_per_second",
    "latency_mean_seconds",
    "latency_p50_seconds",
    "latency_p95_seconds",
    "generated_tokens",
    "output_tokens_per_second",
    "mean_accept_length",
    "draft_accept_rate",
)

DSPARK_COLUMNS = (
    "target_model",
    "sglang_baseline",
    "dspark_run",
    "baseline_output_tokens_per_second",
    "dspark_output_tokens_per_second",
    "speedup",
    "identical_predictions",
    "languages_compared",
    "items_compared",
    "verdict_differences",
    "text_differences",
)

EXPECTED_FULL_SWEEP_LANGUAGE_COUNT = 85


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


def _package_version_section(results: Sequence[RunResult]) -> str:
    versions = sorted(
        {result.metadata.package_version for result in results if result.metadata.package_version}
    )
    if not versions:
        return ""
    label = "Package version" if len(versions) == 1 else "Package versions"
    values = ", ".join(f"`{version}`" for version in versions)
    missing_note = (
        " (some runs lack version metadata)"
        if any(not result.metadata.package_version for result in results)
        else ""
    )
    return f"\n\n{label} recorded in run metadata: {values}{missing_note}."


def _validated_runs(
    results: Sequence[RunResult], prompt_text: str
) -> tuple[list[RunResult], list[RunResult]]:
    """Refuse results a card must not describe; return the generative and scoring runs."""
    if not results:
        raise ValueError("cannot build a card from an empty set of results")
    incompatibilities = collection_comparability_errors(results)
    if incompatibilities:
        raise ValueError("; ".join(incompatibilities))
    for result in results:
        derived = evaluate(outcomes_of(result.predictions))
        if derived != result.metrics:
            raise ValueError(f"{result.metadata.model_id} metrics do not match predictions")
    generative = [r for r in results if r.metadata.is_generative]
    if not generative:
        raise ValueError("cannot build a card without any generative run")
    prompt_sha256 = _uniform(generative, "prompt digest", lambda r: r.metadata.prompt_sha256)
    if sha256_of_text(prompt_text) != prompt_sha256:
        raise ValueError("prompt text does not match the digest recorded in the runs")
    return generative, [r for r in results if not r.metadata.is_generative]


def _generation_settings(generative: Sequence[RunResult]) -> str:
    """The prompt-and-settings line: decoding, seed, budget, precision, batch, quants."""
    decoding = _uniform(generative, "decoding", lambda r: r.metadata.decoding)
    # A GGUF quant has no torch dtype; its precision is its recorded quant label.
    full_precision = [r for r in generative if not r.metadata.quantization]
    dtype = _uniform(full_precision or generative, "dtype", lambda r: r.metadata.dtype)
    quantized = sorted(
        {(r.metadata.model_id, r.metadata.quantization) for r in generative}
        - {(r.metadata.model_id, "") for r in generative}
    )
    quant_line = "".join(
        f"\n`{model_id}` runs the `{quant}` GGUF quant through llama.cpp (same prompt, "
        "template, greedy decoding and budget)."
        for model_id, quant in quantized
    )
    seed = _uniform(generative, "seed", lambda r: r.metadata.seed)
    max_new_tokens = _uniform(generative, "token budget", lambda r: r.metadata.max_new_tokens)
    batch_sizes = sorted({result.metadata.batch_size for result in generative})
    batch_size_line = (
        f"batch {batch_sizes[0]}" if len(batch_sizes) == 1 else "batch varies by model"
    )
    return (
        f"English prompt · {decoding} decoding · seed {seed} · "
        f"`max_new_tokens={max_new_tokens}` ·\n"
        f"`{dtype}` · {batch_size_line}.{quant_line}"
    )


def _scale_line(results: Sequence[RunResult], benchmark_name: str) -> str:
    n_items = _uniform(results, "items per language", lambda r: r.metrics.n_items)
    languages = sorted({result.metadata.language for result in results})
    language_label = "language" if len(languages) == 1 else "languages"
    item_label = "item" if n_items == 1 else "items"
    return (
        f"`{benchmark_name}` · {len(languages)} {language_label} x {n_items} "
        f"{item_label}/language ·\n"
        f"{len(languages) * n_items:,} items · binary `yes`/`no` labels."
    )


def _aggregate_table(generative: Sequence[RunResult]) -> str:
    """The per-model macro table, with best and second-best cells emphasised."""
    header = "| " + " | ".join(CARD_COLUMNS) + " |"
    divider = "|" + "|".join(["---"] * len(CARD_COLUMNS)) + "|"
    rows = aggregate_rows(generative)
    metric_columns = tuple(
        column for column in CARD_COLUMNS if column not in {"model_id", "language_count"}
    )
    rankings = _metric_rankings(rows, metric_columns)
    body = "\n".join(
        "| "
        + " | ".join(
            _ranked_cell(row, column, str(row[column]), rankings) for column in CARD_COLUMNS
        )
        + " |"
        for row in rows
    )
    return f"{header}\n{divider}\n{body}"


def _optional_sections(
    results: Sequence[RunResult],
    scoring: Sequence[RunResult],
    scoring_prompts: Sequence[tuple[str, str]],
    timing_results: Sequence[RunResult],
) -> str:
    """Scoring, log-prob comparison, speed and agreement sections, each only if non-empty."""
    sections = [
        section
        for section in (
            _scoring_section(scoring, prompts=scoring_prompts) if scoring else "",
            _logprob_comparison(results, timing_results),
            _speed_section(results),
            _agreement_section(results),
        )
        if section
    ]
    return "\n\n" + "\n\n".join(sections) if sections else ""


def dataset_card(
    results: Sequence[RunResult],
    *,
    benchmark_name: str,
    prompt_text: str,
    scorer_prompt_text: str = "",
    extra_scorer_prompt_texts: Sequence[str] = (),
    timing_results: Sequence[RunResult] = (),
    viewer_file: str = "",
) -> str:
    """Build a terse card whose scores are recomputed from every prediction."""
    generative, scoring = _validated_runs(results, prompt_text)
    scoring_prompts = _scoring_prompts(
        scoring, prompt_text, (scorer_prompt_text, *extra_scorer_prompt_texts)
    )
    version_section = _package_version_section(results)
    settings = _generation_settings(generative)
    scale = _scale_line(results, benchmark_name)
    viewer_path = Path(viewer_file or "data/train.csv")
    table = _aggregate_table(generative)
    sections_block = _optional_sections(results, scoring, scoring_prompts, timing_results)
    return f"""---
license: mit
configs:
- config_name: default
  data_files:
  - split: train
    path: {viewer_path.as_posix()}
task_categories:
- text-classification
tags:
- land-use
- land-cover
- remote-sensing
- llm-benchmark
---

# Land-use relevance benchmark

{scale}

[Code](https://github.com/NoeFlandre/benchmark-llms-landuse-relevance){version_section}

## Task and prompt

Does a sentence describe a place's land or environment in ways visible to satellites?

{settings}

### Prompt text

Replace `{{}}` with the target sentence.

```text
{prompt_text}```

## Aggregate scores

Per-model macro averages across languages. Per-language 95% intervals and paired tests:
[`leaderboard.csv`](leaderboard.csv); full macro metrics: [`aggregates.csv`](aggregates.csv).
Bold = best; underline = second best in each metric column.

{table}{sections_block}"""


def _markdown_table(columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    body = [
        "| "
        + " | ".join("" if row.get(column) is None else str(row[column]) for column in columns)
        + " |"
        for row in rows
    ]
    return "\n".join((header, divider, *body))


def _single_setting(results: Sequence[RunResult], field: str) -> Any:
    values = {getattr(result.metadata, field) for result in results}
    return values.pop() if len(values) == 1 else "varies"


def _speed_rows(results: Sequence[RunResult]) -> list[dict[str, Any]]:
    rows = []
    for model_id, runs in sorted(group_by_model(results).items()):
        predictions = [prediction for run in runs for prediction in run.predictions]
        speed = summarise_speed(predictions, sum(run.metadata.duration_seconds for run in runs))
        rows.append(
            {
                "model_id": model_id,
                "runtime": _single_setting(runs, "runtime"),
                "device": _single_setting(runs, "device_name"),
                "generation_mode": _single_setting(runs, "generation_mode"),
                "batch_size": _single_setting(runs, "batch_size"),
                "language_count": len({run.metadata.language for run in runs}),
                "cumulative_wall_seconds": round(speed.wall_seconds, 2),
                "sentences_per_second": (
                    None
                    if speed.sentences_per_second is None
                    else round(speed.sentences_per_second, 3)
                ),
                "latency_mean_seconds": (
                    None
                    if speed.latency_mean_seconds is None
                    else round(speed.latency_mean_seconds, 4)
                ),
                "latency_p50_seconds": (
                    None
                    if speed.latency_p50_seconds is None
                    else round(speed.latency_p50_seconds, 4)
                ),
                "latency_p95_seconds": (
                    None
                    if speed.latency_p95_seconds is None
                    else round(speed.latency_p95_seconds, 4)
                ),
                "generated_tokens": speed.generated_tokens,
                "output_tokens_per_second": (
                    None
                    if speed.output_tokens_per_second is None
                    else round(speed.output_tokens_per_second, 2)
                ),
                "mean_accept_length": (
                    None if speed.mean_accept_length is None else round(speed.mean_accept_length, 3)
                ),
                "draft_accept_rate": (
                    None if speed.draft_accept_rate is None else round(speed.draft_accept_rate, 4)
                ),
            }
        )
    return rows


def _speed_section(results: Sequence[RunResult]) -> str:
    sections = [
        "## Runtime performance",
        """Timings are generation wall seconds summed across language runs; latency and throughput
are recomputed from prediction telemetry. Different devices and runtimes are not directly
comparable. Full per-language measurements are in `leaderboard.csv`.""",
    ]
    reproducibility = _reproducibility_note(results)
    if reproducibility:
        sections.append(reproducibility)
    sections.append(_markdown_table(SPEED_COLUMNS, _speed_rows(results)))
    return "\n\n".join(sections)


def _reproducibility_note(results: Sequence[RunResult]) -> str:
    run_ids = {
        "LiquidAI/LFM2.5-VL-3B@sglang-throughput-b16",
        "LiquidAI/LFM2.5-VL-3B+DSpark-throughput-b16",
    }
    languages_by_run = {run_id: set() for run_id in run_ids}
    for result in results:
        run_id = result.metadata.run_id
        if run_id in languages_by_run:
            languages_by_run[run_id].add(result.metadata.language)
    if any(
        len(languages) != EXPECTED_FULL_SWEEP_LANGUAGE_COUNT
        for languages in languages_by_run.values()
    ):
        return ""
    return (
        "Reproducibility probes show GPU sensitivity: on 15 overlapping LFM2.5-2.6B "
        "SGLang-throughput languages, A40 versus RTX 6000 Ada changed 474/4,500 "
        "verdicts; LFM2.5-8B-A1B changed 31/300 verdicts across GPU types. VL-3B "
        "SGLang throughput changed 158/25,500 verdicts between RTX A6000 and RTX "
        "6000 Ada, while its same-GPU DSpark comparison changed 0/25,500. Throughput "
        "mode versus batch size 1 changed 2/300 VL-3B English verdicts. Transformers "
        "continuous batching fails on LFM2 with `Invalid group type: conv`. Compare runs "
        "only with the same runtime, mode and GPU model."
    )


def _agreement_section(results: Sequence[RunResult]) -> str:
    by_name = group_by_model(results)
    agreements: dict[tuple[str, str], list[Any]] = {}
    for agreement in speculative_agreements(results):
        if agreement.same_runtime:
            agreements.setdefault((agreement.speculative_run, agreement.baseline_run), []).append(
                agreement
            )

    speed_rows = {row["model_id"]: row for row in _speed_rows(results)}
    rows = []
    for (draft_name, baseline_name), values in sorted(agreements.items()):
        draft_runs = by_name[draft_name]
        baseline_runs = by_name[baseline_name]
        draft_languages = {run.metadata.language for run in draft_runs}
        baseline_languages = {run.metadata.language for run in baseline_runs}
        by_language = {agreement.language: agreement for agreement in values}
        complete = (
            draft_languages == baseline_languages
            and set(by_language) == draft_languages
            and all(agreement.lossless for agreement in values)
        )
        baseline_tps = speed_rows[baseline_name]["output_tokens_per_second"]
        draft_tps = speed_rows[draft_name]["output_tokens_per_second"]
        rows.append(
            {
                "target_model": draft_runs[0].metadata.model_id,
                "sglang_baseline": baseline_name,
                "dspark_run": draft_name,
                "baseline_output_tokens_per_second": baseline_tps,
                "dspark_output_tokens_per_second": draft_tps,
                "speedup": (
                    None
                    if not baseline_tps or draft_tps is None
                    else f"{draft_tps / baseline_tps:.2f}x"
                ),
                "identical_predictions": "yes" if complete else "no",
                "languages_compared": len(by_language),
                "items_compared": sum(value.n_compared for value in values),
                "verdict_differences": sum(value.verdicts_differ for value in values),
                "text_differences": sum(value.texts_differ for value in values),
            }
        )
    if not rows:
        return ""
    return f"""## DSpark speculative decoding

Greedy DSpark runs should match the same-target SGLang baseline across every language;
the check requires complete item coverage and identical generated text.

{_markdown_table(DSPARK_COLUMNS, rows)}"""


def _metric_rankings(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
    *,
    lower_is_better: frozenset[str] = frozenset(),
) -> dict[str, tuple[float, float | None]]:
    """Return the best and next distinct value for each metric column."""
    rankings = {}
    for column in columns:
        values = sorted(
            {float(row[column]) for row in rows if row.get(column) is not None},
            reverse=column not in lower_is_better,
        )
        if values:
            rankings[column] = (values[0], values[1] if len(values) > 1 else None)
    return rankings


def _ranked_cell(
    row: dict[str, Any],
    column: str,
    display: str,
    rankings: dict[str, tuple[float, float | None]],
) -> str:
    """Emphasize tied best values and the runner-up without changing the data."""
    if column not in rankings or row.get(column) is None:
        return display
    best, second = rankings[column]
    value = float(row[column])
    if value == best:
        return f"**{display}**"
    if second is not None and value == second:
        return f"<u>{display}</u>"
    return display


def _scoring_section(
    scoring: Sequence[RunResult],
    *,
    prompts: Sequence[tuple[str, str]],
) -> str:
    """Render compact public-facing scoring details and tables."""
    summary = scoring_summary_rows(scoring)
    metric_columns = tuple(
        value_column
        for value_column, _, _ in SCORING_SUMMARY_CARD_COLUMNS
        if value_column not in {"model_id", "language_count"}
    )
    rankings = _metric_rankings(
        summary, metric_columns, lower_is_better=frozenset({"peak_vram_bytes_max"})
    )
    summary_header = "| " + " | ".join(label for _, label, _ in SCORING_SUMMARY_CARD_COLUMNS) + " |"
    summary_divider = "|" + "|".join(["---"] * len(SCORING_SUMMARY_CARD_COLUMNS)) + "|"
    summary_body = "\n".join(
        "| "
        + " | ".join(
            _ranked_cell(
                row,
                value_column,
                _card_summary_value(row, value_column, threshold_column),
                rankings,
            )
            for value_column, _, threshold_column in SCORING_SUMMARY_CARD_COLUMNS
        )
        + " |"
        for row in summary
    )
    setup_header = (
        "| model | handling | relevance score / decision rule | sequence length (tokens) | "
        "dtype / batch / seed | revision |"
    )
    setup_divider = "|" + "|".join(["---"] * 6) + "|"
    by_model = group_by_model(scoring)
    setup_body = "\n".join(_scoring_setup_row(by_model[model_id]) for model_id in sorted(by_model))
    prompt_blocks = "\n\n".join(
        f"{label}:\n\n```text\n{text}```" if text else f"{label}." for label, text in prompts
    )
    behavior_note = _encoder_output_behavior_note(scoring)
    behavior_block = f"\n\n{behavior_note}" if behavior_note else ""
    return f"""## Scoring models

Scores are normalized to [0, 1]. Best thresholds are selected on this benchmark (an upper
bound); ROC-AUC needs no threshold. Full sweep: [`threshold_sweep.csv`](threshold_sweep.csv).

### Scoring setup

{setup_header}
{setup_divider}
{setup_body}

### Scoring prompts

{prompt_blocks}

### Best thresholded scoring metrics

{summary_header}
{summary_divider}
{summary_body}{behavior_block}"""


def _encoder_output_behavior_note(scoring: Sequence[RunResult]) -> str:
    """Summarize the encoder's observed yes rate without assuming its behavior."""
    encoder_id = "LiquidAI/LFM2.5-Encoder-350M"
    runs = [result for result in scoring if result.metadata.model_id == encoder_id]
    if not runs:
        return ""

    yes_counts = [
        sum(prediction.predicted is Label.YES for prediction in result.predictions)
        for result in runs
    ]
    item_counts = [len(result.predictions) for result in runs]
    yes_range = (
        str(yes_counts[0])
        if min(yes_counts) == max(yes_counts)
        else f"{min(yes_counts)} to {max(yes_counts)}"
    )
    languages = {result.metadata.language for result in runs}
    if len(languages) == len(runs) and len(set(item_counts)) == 1:
        language_label = "language" if len(languages) == 1 else "languages"
        return (
            "**Observed output behavior:** In these runs, LFM2.5-Encoder-350M predicted "
            f"`yes` for {yes_range} of {item_counts[0]} items per language across "
            f"{len(languages)} {language_label}."
        )

    item_range = (
        str(item_counts[0])
        if min(item_counts) == max(item_counts)
        else f"{min(item_counts)} to {max(item_counts)}"
    )
    return (
        "**Observed output behavior:** In these runs, LFM2.5-Encoder-350M predicted "
        f"`yes` for {yes_range} of {item_range} items per run across {len(runs)} runs "
        f"in {len(languages)} languages."
    )


def _scoring_prompts(
    scoring: Sequence[RunResult], llm_prompt: str, candidates: Sequence[str]
) -> list[tuple[str, str]]:
    """Match every scoring run's prompt digest to a supplied text, labelled by its users.

    A scorer that reads the LLM prompt (log-probability scoring) is pointed back at the
    task prompt above instead of repeating it.
    """
    llm_digest = sha256_of_text(llm_prompt)
    texts = {sha256_of_text(text): text for text in candidates if text}
    users: dict[str, list[str]] = {}
    for result in scoring:
        users.setdefault(result.metadata.prompt_sha256, [])
        if result.metadata.model_id not in users[result.metadata.prompt_sha256]:
            users[result.metadata.prompt_sha256].append(result.metadata.model_id)
    blocks: list[tuple[str, str]] = []
    for digest, models in users.items():
        if digest == llm_digest:
            continue
        if digest not in texts:
            raise ValueError(
                "scoring prompt text does not match the digest recorded in the scoring runs "
                f"of {', '.join(models)}"
            )
        blocks.append((", ".join(f"`{model}`" for model in sorted(models)), texts[digest]))
    if llm_digest in users:
        names = sorted(users[llm_digest])
        verb = "uses" if len(names) == 1 else "use"
        models = ", ".join(f"`{model}`" for model in names)
        blocks.append((f"{models} {verb} the task prompt above", ""))
    return blocks


def _logprob_comparison(
    results: Sequence[RunResult], timing_results: Sequence[RunResult] = ()
) -> str:
    """Compare log-probability scoring against parsing the same model's generations.

    ``timing_results`` are generative reruns on the log-prob runs' GPU, kept out of the
    leaderboards; they add a same-hardware time row over their languages.
    """
    by_model = group_by_model(results)
    timing = group_by_model(timing_results)
    sections = [
        _logprob_section(scored_id, generated_id, by_model, timing)
        for scored_id, generated_id in logprob_pairs().items()
    ]
    return "\n\n".join(section for section in sections if section)


def _logprob_section(
    scored_id: str,
    generated_id: str,
    by_model: dict[str, list[RunResult]],
    timing: dict[str, list[RunResult]],
) -> str:
    scored, generated = by_model.get(scored_id), by_model.get(generated_id)
    if not scored or not generated:
        return ""
    shared = {r.metadata.language for r in scored} & {r.metadata.language for r in generated}
    rows = [
        _comparison_row(method, [r for r in runs if r.metadata.language in shared], scored_id)
        for method, runs in (("generation + parsing", generated), ("yes/no log-probs", scored))
    ]
    same_gpu = _same_gpu_row(scored, timing.get(generated_id, []))
    if same_gpu:
        rows.append(same_gpu)
    name = generated_id.rsplit("/", 1)[-1]
    return (
        f"### {name}: log-probabilities vs generation\n\n"
        "Same model, prompt and languages; log-probs call yes when P(yes) > P(no).\n\n"
        "| method | F1 | MCC | unparsed | ROC-AUC | GPU hours | ms/item |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(rows)
    )


def _comparison_row(method: str, subset: Sequence[RunResult], scored_id: str) -> str:
    (aggregate,) = aggregate_rows(subset)
    seconds = sum(r.metadata.duration_seconds for r in subset)
    items = sum(r.metrics.n_items for r in subset)
    auc = "n/a"
    if not subset[0].metadata.is_generative:
        (summary,) = (row for row in scoring_summary_rows(subset) if row["model_id"] == scored_id)
        value = summary["roc_auc_macro"]
        auc = "n/a" if value is None else _format_card_float(float(value))
    return (
        f"| {method} | {aggregate['f1_macro']} | {aggregate['matthews_corrcoef_macro']} "
        f"| {aggregate['unparsed_rate_macro']} | {auc} | {seconds / 3600:.2f} "
        f"| {1000 * seconds / items:.1f} |"
    )


def _same_gpu_row(scored: Sequence[RunResult], reruns: Sequence[RunResult]) -> str:
    """Time generation and log-probs over the same languages on the same GPU model."""
    if not reruns:
        return ""
    languages = {r.metadata.language for r in reruns}
    devices = {r.metadata.device_name for r in reruns}
    matched = [
        r for r in scored if r.metadata.language in languages and r.metadata.device_name in devices
    ]
    if len(devices) != 1 or len(matched) != len(reruns):
        return ""
    label = f"same GPU ({devices.pop()}; {', '.join(sorted(languages))})"
    generated_s = sum(r.metadata.duration_seconds for r in reruns)
    scored_s = sum(r.metadata.duration_seconds for r in matched)
    return (
        f"| {label}: generation vs log-probs | | | | | {generated_s:.0f} s vs "
        f"{scored_s:.1f} s ({generated_s / scored_s:.0f}x) | |"
    )


def _scoring_setup_row(results: Sequence[RunResult]) -> str:
    """Render the reproducibility-relevant adapter settings for one model."""
    model_id = results[0].metadata.model_id
    try:
        spec = scorer_for(model_id)
    except KeyError:
        spec = None
    family, input_handling, score = spec.card if spec else DEFAULT_CARD
    rule = spec.decision_rule if spec else results[0].metadata.decision_rule or "recorded per run"
    if rule.endswith(" at 0.5"):
        rule = "yes if score ≥ 0.5"
    sequence = _setting_or_varying(results, "sequence length", lambda r: r.metadata.sequence_length)
    if sequence == "None":
        sequence = "model-defined"
    dtype = _setting_or_varying(results, "dtype", lambda r: r.metadata.dtype)
    batch = _setting_or_varying(results, "batch size", lambda r: r.metadata.batch_size)
    seed = _setting_or_varying(results, "seed", lambda r: r.metadata.seed)
    revision = _setting_or_varying(results, "revision", lambda r: r.metadata.model_revision)
    cells = (
        model_id,
        f"{family}: {input_handling}",
        f"{score}; {rule}",
        sequence,
        f"{dtype}; batch {batch}; seed {seed}",
        revision,
    )
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _card_summary_value(
    row: dict[str, Any], value_column: str, threshold_column: str | None
) -> str:
    """Format compact card values without changing the downloadable CSV schema."""
    value = row.get(value_column)
    if value is None:
        return "n/a"
    if threshold_column is not None:
        threshold = row[threshold_column]
        return f"{_format_card_float(float(value))} @ {_format_card_threshold(float(threshold))}"
    if value_column == "peak_vram_bytes_max":
        return f"{float(value) / 1024**3:.2f}"
    if value_column == "throughput_items_per_second_macro":
        return f"{float(value):.2f}"
    return str(value) if isinstance(value, str | int) else _format_card_float(float(value))


def _format_card_float(value: float) -> str:
    """Format a score to four useful decimals, omitting redundant zeroes."""
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _format_card_threshold(value: float) -> str:
    """Keep small score boundaries readable without losing significant digits."""
    return f"{value:.6g}"


def _setting_or_varying(results: Sequence[RunResult], name: str, value_of: Any) -> str:
    """Describe a setting without rejecting a mixed, explicitly recorded roster."""
    values = sorted({value_of(result) for result in results}, key=str)
    if len(values) == 1:
        return str(values[0])
    return f"varies across runs ({name} is recorded per run)"


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
    extra_scorer_prompt_texts: Sequence[str] = (),
    timing_results: Sequence[RunResult] = (),
    data_root: Path | None = None,
    allow_patterns: Sequence[str] | None = None,
) -> str:
    """Write the card next to the results, then push the whole folder to the Hub."""
    if not results:
        raise ValueError("refusing to publish an empty set of results")
    _reject_archive_paths(results_dir)
    hub = api if api is not None else _default_api()
    results_dir.mkdir(parents=True, exist_ok=True)
    viewer_file = ""
    if data_root is not None:
        write_viewer_dataset(
            data_root,
            results_dir / "data" / "train.csv",
            languages=sorted({result.metadata.language for result in results}),
        )
        viewer_file = "data/train.csv"
    (results_dir / "README.md").write_text(
        dataset_card(
            results,
            benchmark_name=benchmark_name,
            prompt_text=prompt_text,
            scorer_prompt_text=scorer_prompt_text,
            extra_scorer_prompt_texts=extra_scorer_prompt_texts,
            timing_results=timing_results,
            viewer_file=viewer_file,
        ),
        encoding="utf-8",
    )
    hub.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    upload_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "folder_path": str(results_dir),
        "commit_message": commit_message,
    }
    if allow_patterns is not None:
        upload_kwargs["allow_patterns"] = list(allow_patterns)
    hub.upload_folder(**upload_kwargs)
    return f"https://huggingface.co/datasets/{repo_id}"


def _default_api() -> DatasetHub:
    from huggingface_hub import HfApi

    # HfApi satisfies DatasetHub in practice; its **kwargs signatures are wider than
    # the protocol can express.
    return cast(DatasetHub, HfApi())


def _reject_archive_paths(results_dir: Path) -> None:
    paths = results_dir.rglob("*") if results_dir.exists() else ()
    for path in paths:
        if ARCHIVE_COMPONENT in path.relative_to(results_dir).parts:
            raise ValueError(f"refusing to upload archive path: {path}")


SCORING_SUMMARY_CARD_COLUMNS = (
    ("model_id", "model", None),
    ("language_count", "languages", None),
    ("best_mcc", "MCC @ threshold", "best_mcc_threshold"),
    ("best_f1", "F1 @ threshold", "best_f1_threshold"),
    (
        "best_balanced_accuracy",
        "balanced accuracy @ threshold",
        "best_balanced_accuracy_threshold",
    ),
    ("best_precision", "precision @ threshold", "best_precision_threshold"),
    ("best_recall", "recall @ threshold", "best_recall_threshold"),
    ("roc_auc_macro", "ROC-AUC", None),
    ("throughput_items_per_second_macro", "items/s", None),
    ("peak_vram_bytes_max", "peak VRAM (GiB)", None),
)


def write_viewer_dataset(
    data_root: Path,
    output: Path,
    *,
    languages: Sequence[str] | None = None,
) -> Path:
    """Export the validated multilingual benchmark as one Hub-viewable CSV."""
    import csv

    from landuse_relevance_bench.adapters.translations import (
        load_language_benchmark,
        load_manifest,
    )

    viewer_columns = (
        "item_id",
        "source_item_id",
        "language",
        "sentence",
        "label",
        "polygon_name",
        "h3_cell",
        "latitude",
        "longitude",
        "source",
        "region",
        "source_url",
    )
    manifest = load_manifest(data_root)
    selected = tuple(sorted(set(languages or manifest.languages)))
    unknown = sorted(set(selected) - set(manifest.languages))
    if unknown:
        raise ValueError(f"unknown viewer language(s): {', '.join(unknown)}")
    if not selected:
        raise ValueError("viewer dataset requires at least one language")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(viewer_columns), lineterminator="\n")
        writer.writeheader()
        for language in selected:
            items = load_language_benchmark(data_root, language)
            source_path = data_root / manifest.files[language].path
            with source_path.open(newline="", encoding="utf-8") as source_handle:
                rows = list(csv.DictReader(source_handle))
            if len(rows) != len(items):
                raise ValueError(
                    f"viewer export row mismatch for {language}: {len(rows)} != {len(items)}"
                )
            for item, source_row in zip(items, rows, strict=True):
                row = {column: source_row.get(column, "") or "" for column in viewer_columns}
                row.update(
                    item_id=item.item_id,
                    source_item_id=item.source_item_id,
                    language=item.language,
                    sentence=item.sentence,
                    label=item.label.value,
                )
                writer.writerow(row)
    return output

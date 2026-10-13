"""Compose the Hugging Face dataset card from validated runs."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from landuse_relevance_bench.adapters.publishing.card_agreement import agreement_section
from landuse_relevance_bench.adapters.publishing.card_format import (
    format_card_float,
    format_card_threshold,
    setting_or_varying,
)
from landuse_relevance_bench.adapters.publishing.card_logprob import logprob_comparison
from landuse_relevance_bench.adapters.publishing.card_overview import (
    aggregate_table,
    generation_settings,
    package_version_section,
    scale_line,
)
from landuse_relevance_bench.adapters.publishing.card_scoring import (
    scoring_prompt_blocks,
    scoring_section,
)
from landuse_relevance_bench.adapters.publishing.card_speed import (
    EXPECTED_FULL_SWEEP_LANGUAGE_COUNT,
    speed_section,
)
from landuse_relevance_bench.adapters.results_store import (
    aggregate_rows,
    group_by_model,
    scoring_summary_rows,
)
from landuse_relevance_bench.domain.card_invariants import validate_for_card
from landuse_relevance_bench.domain.records import RunResult
from landuse_relevance_bench.domain.scorers import (
    D1_MODEL_CARD_LANGUAGE_COUNT,
    GLICLASS_ULTRA_ID,
    GLICLASS_ULTRA_NATIVE_LANGUAGES,
    scorer_for,
)


@dataclass(frozen=True, slots=True)
class CardOptions:
    """The optional card inputs. An empty value leaves its section or link out."""

    scorer_prompt_text: str = ""
    extra_scorer_prompt_texts: Sequence[str] = ()
    timing_results: Sequence[RunResult] = ()
    viewer_file: str = ""


def methodology_card(
    results: Sequence[RunResult],
    *,
    benchmark_name: str,
    prompt_text: str,
    options: CardOptions | None = None,
) -> str:
    """Build a terse card whose scores are recomputed from every prediction."""
    section_options = CardOptions() if options is None else options
    validated = validate_for_card(results, prompt_text)
    generative, scoring = validated.generative, validated.scoring
    prompt_blocks = scoring_prompt_blocks(
        scoring,
        prompt_text,
        (section_options.scorer_prompt_text, *section_options.extra_scorer_prompt_texts),
    )
    version_section = package_version_section(results)
    settings = generation_settings(generative)
    scale = scale_line(results, benchmark_name)
    viewer_path = Path(section_options.viewer_file or "data/train.csv")
    table = aggregate_table(generative)
    sections_block = _optional_sections(
        results, scoring, prompt_blocks, section_options.timing_results
    )
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


def dataset_card(
    results: Sequence[RunResult],
    *,
    benchmark_name: str,
    prompt_text: str,
    options: CardOptions | None = None,
) -> str:
    """Build the concise dataset page from the same rows written to aggregates.csv."""
    section_options = CardOptions() if options is None else options
    validate_for_card(results, prompt_text)
    rows = aggregate_rows(results)
    rows_by_score = sorted(
        rows,
        key=lambda row: (-row["balanced_accuracy_macro"], row["model_id"]),
    )
    model_count = len(rows_by_score)
    generative_count = len(
        {result.metadata.name for result in results if result.metadata.is_generative}
    )
    scoring_count = model_count - generative_count
    full_language_models = sum(
        int(row["language_count"]) == EXPECTED_FULL_SWEEP_LANGUAGE_COUNT for row in rows
    )
    table_lines = [
        (
            "| Model | Languages | Native macro balanced accuracy | Native macro F1 | "
            "Native macro MCC | Unparsed macro rate |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    table_lines.extend(
        f"| `{row['model_id']}` | {row['language_count']}/"
        f"{EXPECTED_FULL_SWEEP_LANGUAGE_COUNT} | "
        f"{format_card_float(float(row['balanced_accuracy_macro']))} | "
        f"{format_card_float(float(row['f1_macro']))} | "
        f"{format_card_float(float(row['matthews_corrcoef_macro']))} | "
        f"{format_card_float(float(row['unparsed_rate_macro']))} |"
        for row in rows_by_score
    )
    coverage_notes = _coverage_notes(rows_by_score)
    coverage = "".join(f"\n\n{note}" for note in coverage_notes)
    viewer_path = Path(section_options.viewer_file or "data/train.csv").as_posix()
    package_versions = package_version_section(results)
    return f"""---
license: mit
configs:
- config_name: default
  data_files:
  - split: train
    path: {viewer_path}
task_categories:
- text-classification
tags:
- land-use
- land-cover
- remote-sensing
- llm-benchmark
---

# Land-use relevance benchmark

`{benchmark_name}` contains 25,500 human-labeled sentences in 85 languages, with 300
aligned sentences per language. Each label is `yes` or `no` for whether the sentence
describes land use, land cover, or geographic environment with a remote-sensing signal.

[Evaluation code](https://github.com/NoeFlandre/benchmark-llms-landuse-relevance) ·
[Full prompt, methods, pins, and runtime notes](docs/results-and-methodology.md){package_versions}

## Results

The published snapshot contains {model_count} models: {generative_count} generative and
{scoring_count} scoring. It has {len(results):,} completed model-language runs.
{full_language_models} models have results for all
{EXPECTED_FULL_SWEEP_LANGUAGE_COUNT} benchmark languages.{coverage}

The table reports native macro metrics from `aggregates.csv`. Each metric is averaged
over languages, not over all sentences pooled together. The 25,500 language entries
translate the same 300 source sentences, so they are not 25,500 independent source
examples. Unparsed rate is the macro-average fraction of outputs that could not be
parsed. Rows are sorted by native macro balanced accuracy. No threshold was tuned on
gold labels for these native results. `scoring_summary.csv` and `threshold_sweep.csv`
report thresholds selected on this benchmark as upper-bound comparisons; they do not
change the fixed native verdicts.

<details>
<summary>Native results for all {model_count} models</summary>

{chr(10).join(table_lines)}

</details>

## Files

- [`aggregates.csv`](aggregates.csv) contains native model-level macro metrics,
  including macro MCC.
- [`leaderboard.csv`](leaderboard.csv) contains per-language metrics, confidence
  intervals, paired tests, runtime metadata, and model revisions.
- [`scoring_summary.csv`](scoring_summary.csv) and
  [`threshold_sweep.csv`](threshold_sweep.csv) contain separately reported threshold
  sweeps. Their thresholds are selected on this benchmark, so treat them as an upper
  bound rather than held-out performance.
- [`SNAPSHOT_STATUS.md`](SNAPSHOT_STATUS.md) records model and language completion counts.
- [Supplementary methods and provenance](docs/results-and-methodology.md) preserves
  historical appendix material, model protocols, fixed decision rules, and precision
  caveats.

## License and citation

The dataset card metadata declares the MIT license. Cite this dataset and the [evaluation code](https://github.com/NoeFlandre/benchmark-llms-landuse-relevance).
"""


def methodology_supplement(
    results: Sequence[RunResult],
    *,
    benchmark_name: str,
    prompt_text: str,
    options: CardOptions | None = None,
) -> str:
    """Preserve the published appendix and add current models absent from its snapshot."""
    section_options = CardOptions() if options is None else options
    validate_for_card(results, prompt_text)
    template_path = (
        Path(__file__).resolve().parents[4]
        / "data"
        / "publication"
        / ("results-and-methodology.md")
    )
    template = template_path.read_text(encoding="utf-8")
    replacements = {
        "<!-- LRB:PACKAGE_VERSION_NOTE -->": _package_version_note(results),
        "<!-- LRB:SNAPSHOT_SUMMARY -->": _snapshot_summary(results, benchmark_name),
        "<!-- LRB:CURRENT_ADDITIONS -->": _current_scorer_additions(
            results, prompt_text, section_options
        ),
    }
    for marker, replacement in replacements.items():
        if marker not in template:
            raise ValueError(f"publication supplement template is missing {marker}")
        template = template.replace(marker, replacement)
    return template


def _package_version_note(results: Sequence[RunResult]) -> str:
    versions = sorted(
        {result.metadata.package_version for result in results if result.metadata.package_version}
    )
    if not versions:
        return "Current runs do not record a package version."
    label = "Package version" if len(versions) == 1 else "Package versions"
    values = ", ".join(f"`{version}`" for version in versions)
    if any(not result.metadata.package_version for result in results):
        return (
            f"{label} `{values}` appear in some run metadata; other runs do not record "
            "a package version."
        )
    return f"{label} recorded in current run metadata: {values}."


def _snapshot_summary(results: Sequence[RunResult], benchmark_name: str) -> str:
    rows = aggregate_rows(results)
    generative = {result.metadata.name for result in results if result.metadata.is_generative}
    scoring = {result.metadata.name for result in results if not result.metadata.is_generative}
    full = sum(int(row["language_count"]) == EXPECTED_FULL_SWEEP_LANGUAGE_COUNT for row in rows)
    lines = [
        f"The current `{benchmark_name}` publication includes {len(rows)} model IDs: "
        f"{len(generative)} generative and {len(scoring)} scoring, with {len(results):,} "
        f"completed model-language result files. {full} model IDs have records in all "
        f"{EXPECTED_FULL_SWEEP_LANGUAGE_COUNT} benchmark languages. The summary table in "
        "`../README.md` and current CSV files reflect this publication; the historical "
        "appendix tables above retain their original snapshot values.",
    ]
    lines.extend(_coverage_notes(rows))
    lines.append("The full coverage list is in [`../SNAPSHOT_STATUS.md`](../SNAPSHOT_STATUS.md).")
    return "\n\n".join(lines)


def _coverage_notes(rows: Sequence[dict[str, Any]]) -> list[str]:
    notes = []
    for row in rows:
        model_id = str(row["model_id"])
        language_count = int(row["language_count"])
        if model_id == GLICLASS_ULTRA_ID:
            native_count = len(GLICLASS_ULTRA_NATIVE_LANGUAGES)
            outside = max(0, EXPECTED_FULL_SWEEP_LANGUAGE_COUNT - native_count)
            native_languages = ", ".join(GLICLASS_ULTRA_NATIVE_LANGUAGES)
            notes.append(
                f"`{model_id}` was evaluated on {language_count} languages; its model card "
                f"lists {native_count} native training languages ({native_languages}), so "
                f"{outside} benchmark languages are outside its native training list. The "
                "card lists Apache-2.0 and describes cross-lingual use."
            )
        elif model_id == "LiquidAI/d1-3B":
            outside = max(0, EXPECTED_FULL_SWEEP_LANGUAGE_COUNT - D1_MODEL_CARD_LANGUAGE_COUNT)
            notes.append(
                f"`{model_id}` was evaluated on {language_count} languages; its model card "
                f"lists {D1_MODEL_CARD_LANGUAGE_COUNT} supported languages, so {outside} "
                "benchmark languages are outside its stated coverage."
            )
        else:
            try:
                spec = scorer_for(model_id)
            except KeyError:
                continue
            if spec.supported_languages is not None:
                notes.append(
                    f"`{model_id}` was evaluated on {language_count} languages; its model "
                    f"card lists {len(spec.supported_languages)} supported languages."
                )
    return notes


def _current_scorer_additions(
    results: Sequence[RunResult], prompt_text: str, options: CardOptions
) -> str:
    historical_ids = _historical_scoring_model_ids()
    current = [
        result
        for result in results
        if not result.metadata.is_generative and result.metadata.model_id not in historical_ids
    ]
    if not current:
        return ""
    aggregate_by_name = {row["model_id"]: row for row in aggregate_rows(current)}
    sweep_by_name = {row["model_id"]: row for row in scoring_summary_rows(current)}
    groups = group_by_model(current)
    additions = [
        "## Current additions to the historical appendix",
        "",
        "These rows describe scoring models added after the snapshot summarized above. "
        "The native values use each model's fixed decision rule. The separately labeled "
        "threshold results are selected on this benchmark and remain exploratory upper bounds.",
        "",
    ]
    additions.extend(_scoring_setup_rows(groups))
    additions.extend(_native_result_rows(groups, aggregate_by_name))
    additions.extend(_threshold_comparison_rows(sweep_by_name))
    additions.extend(_scoring_prompt_rows(current, prompt_text, options))
    return "\n".join(additions)


def _historical_scoring_model_ids() -> set[str]:
    return {
        line.strip("|").split("|", 1)[0].strip()
        for line in _template_setup_table().splitlines()
        if line.strip().startswith("|")
    }


def _scoring_setup_rows(groups: dict[str, list[RunResult]]) -> list[str]:
    rows = [
        "### Scoring setup",
        "",
        (
            "| Model | Handling | Score / native decision rule | Sequence length | "
            "Dtype / batch / seed | Revision |"
        ),
        "|---|---|---|---:|---|---|",
    ]
    for model_runs in groups.values():
        result = model_runs[0]
        try:
            spec = scorer_for(result.metadata.model_id)
        except KeyError:
            spec = None
        family, handling, score = (
            spec.card if spec else ("scoring adapter", "recorded input", "native score")
        )
        decision_rule = (
            spec.decision_rule if spec else result.metadata.decision_rule or "recorded per run"
        )
        sequence = setting_or_varying(
            model_runs, "sequence length", lambda run: run.metadata.sequence_length
        )
        if sequence == "None":
            sequence = "model-defined"
        dtype = setting_or_varying(model_runs, "dtype", lambda run: run.metadata.dtype)
        batch = setting_or_varying(model_runs, "batch size", lambda run: run.metadata.batch_size)
        seed = setting_or_varying(model_runs, "seed", lambda run: run.metadata.seed)
        revision = setting_or_varying(
            model_runs, "revision", lambda run: run.metadata.model_revision
        )
        rows.append(
            f"| `{result.metadata.model_id}` | {family}: {handling} | "
            f"{score}; {decision_rule} | {sequence} | {dtype}; batch {batch}; seed {seed} | "
            f"`{revision}` |"
        )
    return rows


def _native_result_rows(
    groups: dict[str, list[RunResult]], aggregate_by_name: dict[str, dict[str, Any]]
) -> list[str]:
    rows = [
        "",
        "### Native results",
        "",
        (
            "| Model | Languages | Native balanced accuracy | Native F1 | Native MCC | "
            "Unparsed rate | Fixed decision rule | Revision |"
        ),
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for name, model_runs in sorted(groups.items()):
        result = model_runs[0]
        aggregate = aggregate_by_name[name]
        rows.append(
            f"| `{result.metadata.model_id}` | {aggregate['language_count']}/"
            f"{EXPECTED_FULL_SWEEP_LANGUAGE_COUNT} | "
            f"{format_card_float(float(aggregate['balanced_accuracy_macro']))} | "
            f"{format_card_float(float(aggregate['f1_macro']))} | "
            f"{format_card_float(float(aggregate['matthews_corrcoef_macro']))} | "
            f"{format_card_float(float(aggregate['unparsed_rate_macro']))} | "
            f"{result.metadata.decision_rule} | `{result.metadata.model_revision}` |"
        )
    return rows


def _threshold_comparison_rows(sweep_by_name: dict[str, dict[str, Any]]) -> list[str]:
    rows = [
        "",
        "The following threshold sweep is separate from those native rows. It is "
        "selected on the same gold set and does not change the fixed native rule.",
        "",
        "### Threshold-selected comparison",
        "",
        (
            "| Model | Best MCC @ threshold | Best F1 @ threshold | "
            "Best balanced accuracy @ threshold | Best precision @ threshold | "
            "Best recall @ threshold | ROC-AUC |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, sweep in sorted(sweep_by_name.items()):
        auc = sweep.get("roc_auc_macro")
        rows.append(
            f"| `{name}` | {_metric_at_threshold(sweep, 'best_mcc')} | "
            f"{_metric_at_threshold(sweep, 'best_f1')} | "
            f"{_metric_at_threshold(sweep, 'best_balanced_accuracy')} | "
            f"{_metric_at_threshold(sweep, 'best_precision')} | "
            f"{_metric_at_threshold(sweep, 'best_recall')} | "
            f"{format_card_float(float(auc)) if auc is not None else 'n/a'} |"
        )
    return rows


def _scoring_prompt_rows(
    current: Sequence[RunResult], prompt_text: str, options: CardOptions
) -> list[str]:
    rows = []
    for block_label, block_text in scoring_prompt_blocks(
        current,
        prompt_text,
        (options.scorer_prompt_text, *options.extra_scorer_prompt_texts),
    ):
        if not block_text:
            continue
        rows.extend(
            (
                "",
                f"### Prompt used by {block_label}",
                "",
                "```text",
                block_text.rstrip(),
                "```",
            )
        )
    return rows


def _template_setup_table() -> str:
    template_path = (
        Path(__file__).resolve().parents[4]
        / "data"
        / "publication"
        / ("results-and-methodology.md")
    )
    template = template_path.read_text(encoding="utf-8")
    return template.split("### Scoring setup", 1)[1].split("### Scoring prompts", 1)[0]


def _metric_at_threshold(row: dict[str, Any], metric: str) -> str:
    value = row.get(f"best_{metric}")
    threshold = row.get(f"best_{metric}_threshold")
    if value is None or threshold is None:
        return "n/a"
    return f"{format_card_float(float(value))} @ {format_card_threshold(float(threshold))}"


def _optional_sections(
    results: Sequence[RunResult],
    scoring: Sequence[RunResult],
    prompt_blocks: Sequence[tuple[str, str]],
    timing_results: Sequence[RunResult],
) -> str:
    """Scoring, log-prob comparison, speed and agreement sections, each only if non-empty."""
    sections = [
        section
        for section in (
            scoring_section(scoring, prompts=prompt_blocks) if scoring else "",
            logprob_comparison(results, timing_results),
            speed_section(results),
            agreement_section(results),
        )
        if section
    ]
    return "\n\n" + "\n\n".join(sections) if sections else ""

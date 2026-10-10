"""Compose the Hugging Face dataset card from validated runs."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from landuse_relevance_bench.adapters.publishing.card_agreement import agreement_section
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
from landuse_relevance_bench.adapters.publishing.card_speed import speed_section
from landuse_relevance_bench.adapters.publishing.card_validation import validate_for_card
from landuse_relevance_bench.domain.records import RunResult


@dataclass(frozen=True, slots=True)
class CardOptions:
    """The optional card inputs. An empty value leaves its section or link out."""

    scorer_prompt_text: str = ""
    extra_scorer_prompt_texts: Sequence[str] = ()
    timing_results: Sequence[RunResult] = ()
    viewer_file: str = ""


def dataset_card(
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

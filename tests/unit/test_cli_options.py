"""Pin the option surface of the run, score and run-all commands."""

import typer.main

from landuse_relevance_bench import cli


def _surface(name: str) -> list[tuple]:
    command = typer.main.get_command(cli.app).commands[name]  # type: ignore[attr-defined]
    return [
        (p.name, tuple(p.opts), getattr(p, "help", None), str(p.default), p.required, p.type.name)
        for p in command.params
    ]


RUN = [
    ("model_id", ("model_id",), "Hugging Face model repository id.", "None", True, "str"),
    (
        "data_root",
        ("--data-root",),
        "Vendored multilingual data root.",
        "data/translations",
        False,
        "path",
    ),
    (
        "prompt",
        ("--prompt",),
        "Prompt template with a {} placeholder.",
        "data/prompt.txt",
        False,
        "path",
    ),
    ("out", ("--out",), "Directory to write run results into.", "results", False, "path"),
    (
        "language",
        ("--language",),
        "Language code(s), repeatable or comma-separated.",
        "None",
        False,
        "str",
    ),
    ("revision", ("--revision",), "Pin the model to a commit.", "None", False, "str"),
    ("batch_size", ("--batch-size",), "Prompts per forward pass.", "None", False, "int"),
    (
        "max_new_tokens",
        ("--max-new-tokens",),
        "Maximum tokens to generate per prompt.",
        "4096",
        False,
        "int",
    ),
    ("seed", ("--seed",), "Random seed for decoding.", "0", False, "int"),
    ("dtype", ("--dtype",), "Torch dtype name.", "bfloat16", False, "str"),
    (
        "continuous_batching",
        ("--continuous-batching",),
        "Use Transformers continuous batching.",
        "False",
        False,
        "boolean",
    ),
    (
        "throughput",
        ("--throughput",),
        "Use SGLang multi-request throughput mode.",
        "False",
        False,
        "boolean",
    ),
    ("shard_index", ("--shard-index",), "Zero-based index of this shard.", "0", False, "int"),
    (
        "shard_count",
        ("--shard-count",),
        "Total number of shards the work is split into.",
        "1",
        False,
        "int",
    ),
]

SCORE = [
    ("model_id", ("model_id",), "Scoring roster id.", "None", True, "str"),
    (
        "data_root",
        ("--data-root",),
        "Vendored multilingual data root.",
        "data/translations",
        False,
        "path",
    ),
    (
        "prompt",
        ("--prompt",),
        "Prompt template; defaults to the scorer's own.",
        "None",
        False,
        "path",
    ),
    ("out", ("--out",), "Directory to write run results into.", "results", False, "path"),
    (
        "language",
        ("--language",),
        "Language code(s), repeatable or comma-separated.",
        "None",
        False,
        "str",
    ),
    ("revision", ("--revision",), "Pin the model to a commit.", "None", False, "str"),
    ("batch_size", ("--batch-size",), "Prompts per forward pass.", "16", False, "int"),
    ("seed", ("--seed",), "Random seed for decoding.", "0", False, "int"),
    ("dtype", ("--dtype",), "Torch dtype name.", "bfloat16", False, "str"),
    ("shard_index", ("--shard-index",), "Zero-based index of this shard.", "0", False, "int"),
    (
        "shard_count",
        ("--shard-count",),
        "Total number of shards the work is split into.",
        "1",
        False,
        "int",
    ),
]

RUN_ALL = [
    (
        "data_root",
        ("--data-root",),
        "Vendored multilingual data root.",
        "data/translations",
        False,
        "path",
    ),
    (
        "prompt",
        ("--prompt",),
        "Prompt template with a {} placeholder.",
        "data/prompt.txt",
        False,
        "path",
    ),
    ("out", ("--out",), "Directory to write run results into.", "results", False, "path"),
    (
        "language",
        ("--language",),
        "Language code(s), repeatable or comma-separated.",
        "None",
        False,
        "str",
    ),
    ("batch_size", ("--batch-size",), "Prompts per forward pass.", "None", False, "int"),
    (
        "continuous_batching",
        ("--continuous-batching",),
        "Use Transformers continuous batching.",
        "False",
        False,
        "boolean",
    ),
    (
        "throughput",
        ("--throughput",),
        "Use SGLang multi-request throughput mode.",
        "False",
        False,
        "boolean",
    ),
    (
        "max_new_tokens",
        ("--max-new-tokens",),
        "Maximum tokens to generate per prompt.",
        "4096",
        False,
        "int",
    ),
    ("seed", ("--seed",), "Random seed for decoding.", "0", False, "int"),
    ("dtype", ("--dtype",), "Torch dtype name.", "bfloat16", False, "str"),
    ("shard_index", ("--shard-index",), "Zero-based index of this shard.", "0", False, "int"),
    (
        "shard_count",
        ("--shard-count",),
        "Total number of shards the work is split into.",
        "1",
        False,
        "int",
    ),
    ("only", ("--only",), "Regex over roster run names.", "None", False, "str"),
    (
        "runtime",
        ("--runtime",),
        "Restrict to transformers or sglang (repeatable).",
        "None",
        False,
        "str",
    ),
    (
        "skip_existing",
        ("--skip-existing",),
        "Skip model-language pairs that already have stored results.",
        "True",
        False,
        "boolean",
    ),
    (
        "keep_going",
        ("--keep-going",),
        "Continue after a failed run; exit 1 at the end.",
        "False",
        False,
        "boolean",
    ),
]


def test_run_options_are_pinned() -> None:
    assert _surface("run") == RUN


def test_score_options_are_pinned() -> None:
    assert _surface("score") == SCORE


def test_run_all_options_are_pinned() -> None:
    assert _surface("run-all") == RUN_ALL

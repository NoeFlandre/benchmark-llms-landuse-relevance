"""Scriptable entry points for running, scoring and publishing the benchmark."""

import enum
from pathlib import Path
from typing import Annotated

import typer

from landuse_relevance_bench.adapters.cli_support import (
    LOGGER_NAME as _LOGGER_NAME,
)
from landuse_relevance_bench.adapters.cli_support import (
    PublishPreviewRequest,
    benchmark_one,
    configure_logging,
    echo_json,
    echo_report,
    fail,
    first_line,
    generator_provider,
    load_runs,
    preview_publish,
    print_version,
    selected_run_names,
)
from landuse_relevance_bench.adapters.hf_publish import DEFAULT_COMMIT_MESSAGE
from landuse_relevance_bench.adapters.paths import default_paths
from landuse_relevance_bench.adapters.pipeline import (
    DEFAULT_DTYPE,
    DEFAULT_MAX_NEW_TOKENS,
    RunRequest,
)
from landuse_relevance_bench.adapters.provenance import source_commit
from landuse_relevance_bench.adapters.results_store import (
    describe_agreement,
    has_result,
    leaderboard_rows,
    read_runs,
    write_leaderboard_csv,
)
from landuse_relevance_bench.domain.agreement import speculative_agreements
from landuse_relevance_bench.domain.orchestration import DEFAULT_BATCH_SIZE
from landuse_relevance_bench.domain.roster import (
    ROSTER,
    RUNTIMES,
    model_ids,
)
from landuse_relevance_bench.domain.selection import (
    check_comparable,
)

LOGGER_NAME = _LOGGER_NAME

DEFAULT_BENCHMARK, DEFAULT_PROMPT, DEFAULT_RESULTS = default_paths()

app = typer.Typer(help=__doc__)

Runtime = enum.StrEnum("Runtime", {name: name for name in RUNTIMES})

Benchmark = Annotated[Path, typer.Option("--benchmark", help="Labelled benchmark CSV.")]
Prompt = Annotated[Path, typer.Option("--prompt", help="Prompt template with a {} placeholder.")]
Results = Annotated[
    Path,
    typer.Option(
        "--out", "--results-dir", help="Directory to write run results into (--out is an alias)."
    ),
]
ResultsDir = Annotated[
    Path, typer.Option("--results-dir", help="Directory holding stored run results.")
]
BatchSize = Annotated[
    int | None,
    typer.Option(help=f"Prompts per forward pass [default: roster's, else {DEFAULT_BATCH_SIZE}]."),
]
MaxNewTokens = Annotated[int, typer.Option(help="Generation budget per prompt, in tokens.")]
Seed = Annotated[int, typer.Option(help="Random seed set before loading the model.")]
Dtype = Annotated[str, typer.Option(help="Torch dtype name.")]
ContinuousBatching = Annotated[
    bool,
    typer.Option(
        "--continuous-batching",
        help="Use Transformers continuous batching (set --batch-size to the worklist size).",
    ),
]
Throughput = Annotated[
    bool,
    typer.Option("--throughput", help="Name and record an SGLang multi-request throughput run."),
]
Only = Annotated[str | None, typer.Option("--only", help="Regex; keep only run names it matches.")]
RuntimeFilter = Annotated[
    list[Runtime] | None,
    typer.Option("--runtime", help="Keep only runs on this runtime; repeatable."),
]
Json = Annotated[bool, typer.Option("--json", help="Print machine-readable JSON instead of text.")]
AllowMixed = Annotated[
    bool,
    typer.Option("--allow-mixed", help="Continue when runs use different benchmark settings."),
]


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=print_version, is_eager=True, help="Print the version and exit."
        ),
    ] = False,
    verbose: Annotated[
        int, typer.Option("--verbose", "-v", count=True, help="More output; repeat for debug.")
    ] = 0,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Only report errors.")] = False,
) -> None:
    """Scriptable entry points for running, scoring and publishing the benchmark."""
    del version
    configure_logging(-1 if quiet else verbose)


def _benchmark_one(request: RunRequest, *, as_json: bool = False) -> None:
    benchmark_one(request, generator_provider, source_commit, as_json=as_json)


@app.command(epilog="Examples:  lrb models  |  lrb models --runtime sglang --json")
def models(runtime: RuntimeFilter = None, as_json: Json = False) -> None:
    """List the models in the benchmark roster (name, size, runtime, note; tab-separated)."""
    specs = [spec for spec in ROSTER if not runtime or spec.runtime in runtime]
    if as_json:
        echo_json(
            [
                {
                    "name": spec.name,
                    "model_id": spec.model_id,
                    "total_parameters": spec.total_parameters,
                    "runtime": spec.runtime,
                    "note": spec.note,
                }
                for spec in specs
            ]
        )
        return
    for spec in specs:
        typer.echo(f"{spec.name}\t{spec.total_parameters / 1e9:.2f}B\t{spec.runtime}\t{spec.note}")


@app.command(
    epilog="Examples:  lrb run LiquidAI/LFM2.5-350M  |  "
    "lrb run some/model --revision c0ffee --results-dir results --json"
)
def run(
    model_id: Annotated[
        str, typer.Argument(help="A rostered run name, or any Hugging Face model id.")
    ],
    benchmark: Benchmark = DEFAULT_BENCHMARK,
    prompt: Prompt = DEFAULT_PROMPT,
    out: Results = DEFAULT_RESULTS,
    revision: Annotated[str | None, typer.Option(help="Pin the model to a commit.")] = None,
    batch_size: BatchSize = None,
    continuous_batching: ContinuousBatching = False,
    throughput: Throughput = False,
    max_new_tokens: MaxNewTokens = DEFAULT_MAX_NEW_TOKENS,
    seed: Seed = 0,
    dtype: Dtype = DEFAULT_DTYPE,
    as_json: Json = False,
) -> None:
    """Benchmark one model and write its result under --out."""
    request = RunRequest.for_run(
        model_id,
        benchmark_path=benchmark,
        prompt_path=prompt,
        output_dir=out,
        revision=revision,
        batch_size=batch_size,
        continuous_batching=continuous_batching,
        throughput_mode=throughput,
        max_new_tokens=max_new_tokens,
        seed=seed,
        dtype=dtype,
    )
    _benchmark_one(request, as_json=as_json)


@app.command(
    name="run-all",
    epilog="Examples:  lrb run-all --results-dir results  |  "
    "lrb run-all --runtime transformers --only 'LFM2.5-VL' --skip-existing --keep-going",
)
def run_all(
    benchmark: Benchmark = DEFAULT_BENCHMARK,
    prompt: Prompt = DEFAULT_PROMPT,
    out: Results = DEFAULT_RESULTS,
    batch_size: BatchSize = None,
    max_new_tokens: MaxNewTokens = DEFAULT_MAX_NEW_TOKENS,
    seed: Seed = 0,
    dtype: Dtype = DEFAULT_DTYPE,
    only: Only = None,
    runtime: RuntimeFilter = None,
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing", help="Skip runs whose result file already exists and is non-empty."
        ),
    ] = False,
    keep_going: Annotated[
        bool,
        typer.Option(
            "--keep-going", help="Carry on after a failed run; exit 1 at the end if any failed."
        ),
    ] = False,
) -> None:
    """Benchmark every rostered model in turn, optionally filtered and resumable."""
    failed: list[str] = []
    for model in selected_run_names(model_ids(), only, runtime):
        if skip_existing and has_result(out, model):
            typer.echo(f"skip {model} (result already in {out})")
            continue
        request = RunRequest.for_run(
            model,
            benchmark_path=benchmark,
            prompt_path=prompt,
            output_dir=out,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            seed=seed,
            dtype=dtype,
        )
        if not keep_going:
            _benchmark_one(request)
            continue
        try:
            _benchmark_one(request)
        except RUN_FAILURES as exc:
            typer.echo(f"FAILED {model}: {exc}", err=True)
            failed.append(model)
    if failed:
        typer.echo(f"{len(failed)} run(s) failed: {', '.join(failed)}", err=True)
        raise typer.Exit(code=1)


#: What ``--keep-going`` survives: bad input, runtime and CUDA errors, a missing runtime.
RUN_FAILURES = (typer.BadParameter, RuntimeError, OSError, ValueError, ImportError)


@app.command(
    epilog="Examples:  lrb report --results-dir results  |  lrb report --json | python -m json.tool"
)
def report(
    results_dir: ResultsDir = DEFAULT_RESULTS,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Leaderboard CSV path [default: RESULTS_DIR/leaderboard.csv]."),
    ] = None,
    as_json: Json = False,
    allow_mixed: AllowMixed = False,
) -> None:
    """Aggregate stored runs into a leaderboard; exits 1 if a lossless check fails."""
    runs = load_runs(results_dir, read_runs)
    comparability = check_comparable(runs)
    if not comparability.comparable and not allow_mixed:
        raise typer.BadParameter(f"{comparability.summary}; pass --allow-mixed to continue")
    destination = out or results_dir / "leaderboard.csv"
    write_leaderboard_csv(runs, destination)
    agreements = speculative_agreements(runs)
    mismatched = [a for a in agreements if a.same_runtime and not a.lossless]
    if as_json:
        echo_json(
            {
                "leaderboard": leaderboard_rows(runs),
                "speculative_checks": [describe_agreement(a) for a in agreements],
                "comparability": {
                    "comparable": comparability.comparable,
                    "reference_run": comparability.reference_name,
                    "summary": comparability.summary,
                    "differences": [
                        {
                            "run_name": difference.run_name,
                            "field": difference.field,
                            "expected": difference.expected,
                            "actual": difference.actual,
                        }
                        for difference in comparability.differences
                    ],
                },
                "leaderboard_csv": str(destination),
            }
        )
    else:
        echo_report(runs, agreements, destination, comparability)
    if mismatched:
        typer.echo("speculative run disagrees with its same-runtime baseline", err=True)
        raise typer.Exit(code=1)


@app.command(
    epilog="Examples:  lrb publish me/landuse-bench --results-dir results --dry-run  |  "
    "lrb publish me/landuse-bench --commit-message 'Add LFM2.5-VL runs'"
)
def publish(
    repo_id: Annotated[str, typer.Argument(help="Hugging Face dataset repository id.")],
    results_dir: ResultsDir = DEFAULT_RESULTS,
    benchmark: Benchmark = DEFAULT_BENCHMARK,
    private: Annotated[bool, typer.Option(help="Create the dataset repository private.")] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show the target, files and card that would be pushed; touch nothing.",
        ),
    ] = False,
    commit_message: Annotated[
        str, typer.Option("--commit-message", help="Commit message for the Hub upload.")
    ] = DEFAULT_COMMIT_MESSAGE,
    allow_mixed: AllowMixed = False,
) -> None:
    """Push the stored runs, leaderboard and a generated card to the Hub."""
    from landuse_relevance_bench.adapters import hf_publish

    runs = load_runs(results_dir, hf_publish.read_published_runs)
    comparability = check_comparable(runs)
    if not comparability.comparable and not allow_mixed:
        raise typer.BadParameter(f"{comparability.summary}; pass --allow-mixed to continue")
    if dry_run:
        preview_publish(
            PublishPreviewRequest(
                repo_id=repo_id,
                results_dir=results_dir,
                runs=runs,
                private=private,
                benchmark=benchmark,
                allow_mixed=allow_mixed,
            )
        )
        return
    write_leaderboard_csv(runs, results_dir / "leaderboard.csv")
    try:
        url = hf_publish.publish_results(
            repo_id,
            results_dir,
            runs,
            private=private,
            commit_message=commit_message,
            benchmark_name=benchmark.name,
            allow_mixed=allow_mixed,
        )
    except ImportError as exc:
        fail(f"publishing needs huggingface-hub; install with --extra publish ({exc})")
    except OSError as exc:  # huggingface_hub HTTP, auth and network errors are OSErrors
        fail(f"publishing to {repo_id} failed: {first_line(exc)}")
    typer.echo(f"published {len(runs)} run(s) to {url}")

# Contributing

Thanks for helping improve the benchmark. Keep changes reproducible and easy to
review, especially when they affect model execution, scoring, or published results.

## Before opening a pull request

1. Check the existing [issues](https://github.com/NoeFlandre/benchmark-llms-landuse-relevance/issues)
   and open the relevant form if the work needs discussion.
2. Create a focused branch and keep unrelated changes out of the pull request.
3. For code or configuration changes, install the development tools and enable the
   local hooks:

   ```bash
   uv sync --group dev
   uv tool install pre-commit
   pre-commit install
   ```

   The hooks run Ruff and ty through `uv` using the versions pinned in `uv.lock`.
   They do not install separate copies of those tools.

## Checks

Run the checks relevant to your change. The full deterministic quality suite is:

```bash
make check
```

For a quick focused check, use the matching Make target, such as `make lint`,
`make types`, `make test`, `make acceptance`, or `make docs-build`. `make check`
also runs mutation testing and dependency audits, so it can take longer than an
individual target.

Real-model integration tests download model weights and need a suitable runtime
and hardware. Run them explicitly with `make integration`; report the model,
runtime, hardware, and command in the pull request. Do not present a test skipped
for lack of hardware as a passing integration test.

## Benchmark and result changes

For changes that add a model, runtime, scoring rule, benchmark item, or result:

- Include the exact model and draft revisions, runtime, decoding settings, prompt
  and benchmark hashes, source commit, and command needed to reproduce the run.
- Record the hardware and distinguish measured results from estimates or
  unverified runs.
- Explain how the change affects comparability with existing runs. Keep raw model
  outputs and result metadata with the published artifacts when the benchmark
  workflow requires them.
- Preserve source URLs and attribution for benchmark text, and check the source's
  reuse terms before adding or redistributing material.
- Do not rewrite prior results to make a new method appear comparable. Describe
  corrections and re-scoring explicitly.

The benchmark uses small, curated data and pinned model revisions; a result from a
different prompt, data revision, decoding setup, or runtime should be identified as
a separate configuration.

## Pull requests

Use the pull request template. Link the issue, summarize user-visible or
methodological effects, and list the checks you ran with their outcomes. If a
section does not apply, say so briefly. Keep the title descriptive and use a
Conventional Commit prefix when practical (for example, `fix:`, `docs:`, or
`feat:`).

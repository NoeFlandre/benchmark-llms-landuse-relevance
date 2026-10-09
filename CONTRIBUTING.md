# Contributing

Thank you for your help with the benchmark. Make each change reproducible and easy to
review. This is very important for changes to model execution, scoring, or published
results.

Record each user-visible change in [CHANGELOG.md](CHANGELOG.md). Add an entry under
"Scoring changes" when a change affects a metric.

## Before you open a pull request

1. Read the existing [issues](https://github.com/NoeFlandre/benchmark-llms-landuse-relevance/issues).
   If the work needs discussion, open the applicable form.
2. Create a focused branch. Do not put unrelated changes in the pull request.
3. For a change to code or configuration, install the development tools and enable the
   local hooks:

   ```bash
   uv sync --group dev
   uv tool install pre-commit
   pre-commit install
   ```

   The hooks run Ruff and ty through `uv`. They use the versions that `uv.lock` pins.
   They do not install separate copies of these tools.

## Checks

Run the checks that apply to your change. This command runs the complete deterministic
quality suite:

```bash
make check
```

For a quick focused check, use the applicable Make target. Examples are `make lint`,
`make types`, `make test`, `make acceptance`, and `make docs-build`. The command
`make check` also runs mutation testing and dependency audits. It takes more time than
one target.

Real-model integration tests download model weights. They need a suitable runtime and
suitable hardware. Run them with `make integration`. In the pull request, report the
model, the runtime, the hardware, and the command.

WARNING: Do not report a test that was skipped for lack of hardware as a passed
integration test.

## Benchmark and result changes

These rules apply to a change that adds a model, a runtime, a scoring rule, a benchmark
item, or a result:

- Include the exact model and draft revisions, the runtime, the decoding settings, the
  prompt and benchmark hashes, the source commit, and the command that reproduces the
  run.
- Record the hardware. Show which results are measured and which are estimated or
  not verified.
- Explain how the change affects comparability with the existing runs. Keep the raw
  model outputs and the result metadata with the published artifacts when the
  benchmark workflow needs them.
- Keep the source URLs and the attribution for the benchmark text. Check the reuse
  terms of the source before you add or redistribute material.
- Do not rewrite prior results to make a new method look comparable. Describe each
  correction and each re-scoring in the text.

The benchmark uses a small set of curated data and pinned model revisions. Identify a
result from a different prompt, data revision, decoding setup, or runtime as a separate
configuration.

## Pull requests

Use the pull request template. Link the issue. Summarize the effects on users and on
method. List the checks that you ran and their results. If a section does not apply,
say so in a short sentence. Write a descriptive title. When possible, use a
Conventional Commit prefix, for example `fix:`, `docs:`, or `feat:`.

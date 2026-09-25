# ADR-0007 — DSpark speculative decoding, and checking it is lossless

**Status:** accepted · 2026-09-25

## Context

Liquid AI publishes DSpark draft models for LFM2.5-1.2B-Instruct, LFM2.5-2.6B,
LFM2.5-8B-A1B and LFM2.5-VL-3B. A draft proposes a block of tokens that the target
verifies in one pass; under greedy decoding the output is, by construction, what the
target alone would have produced. The drafts are served by SGLang (>= 0.5.19 for the
VL target; LFM2/LFM2-MoE text targets need DSpark support from SGLang PR #31041;
0.5.20 ships an `lfm2_dspark` draft model, so the extra requires `sglang>=0.5.20`,
unverified on a GPU at the time of writing).

## Decision

For each target with a draft the roster holds two SGLang runs on pinned weights:

- `<target>@sglang` — the card's launch "without the `--speculative-*` flags"
  (`disable_radix_cache`, `mem_fraction_static`), batch size 1;
- `<target>+DSpark` — the same launch with the draft attached exactly as the card
  prescribes: `speculative_algorithm=DSPARK`, `speculative_draft_model_path`,
  `speculative_draft_attention_backend=flashinfer`, and for VL-3B
  `speculative_dspark_block_size=9` (`mem_fraction_static` 0.8 for VL-3B, 0.75 for the
  text targets, whose block size is read from the draft config).

The plain Transformers run of each target stays in the roster, so the SGLang pair adds
a like-for-like speed comparison without replacing the reference run. VL-3B is
prompted with a text-only user turn through its processor's chat template.

Prompts reach SGLang as token ids produced by the same chat template as the
Transformers runs, so the three runs of a target see the same input tokens.

SGLang pins its own torch and transformers, so it is a separate `speculative` extra,
declared conflicting with `inference` in `[tool.uv]`; the Grid'5000 node script
re-syncs the environment between the two runtimes.

## The lossless check

`lrb report` and the dataset card compare every speculative run against every plain
run of the same target with the same budget and prompt, item by item, counting
differing verdicts and differing raw generations.

- Against the **same-runtime** baseline (`@sglang`) any difference is a defect —
  wrong settings, a non-greedy kernel, or a draft for another target — and `lrb
  report` exits non-zero.
- Against the **Transformers** run it is reported for information only: two runtimes'
  bf16 kernels can break a near-tie differently, so a small number of differing
  generations there does not mean the draft changed the output.

## Consequences

- Scores of `+DSpark` should equal those of `@sglang`; the speed columns (ADR-0006)
  carry the result, including SGLang's mean accept length and draft accept rate.
- The roster now has 13 runs, eight of them on SGLang at batch size 1; budget the
  Grid'5000 walltime accordingly.
- If a card's recipe changes, the roster's settings change with it and the result
  files record which settings were used (`speculative` in the metadata).

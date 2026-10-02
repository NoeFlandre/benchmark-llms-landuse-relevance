# ADR-0010 - DSpark speculative decoding and the lossless check

**Status:** accepted · 2026-09-25

## Context

Liquid AI publishes DSpark draft models for LFM2.5-1.2B-Instruct, LFM2.5-2.6B,
LFM2.5-8B-A1B, and LFM2.5-VL-3B. A draft proposes a block of tokens. The target verifies
the block in one pass. Under greedy decoding, the output is, by construction, the output
that the target alone produces.

SGLang serves the drafts:

- The VL target needs SGLang >= 0.5.19.
- The LFM2 and LFM2-MoE text targets need DSpark support from SGLang PR #31041.
- SGLang 0.5.20 includes an `lfm2_dspark` draft model. Therefore, the extra requires
  `sglang>=0.5.20`. At the time of writing, nobody verified this on a GPU.

## Decision

For each target that has a draft, the roster holds two SGLang runs on pinned weights:

- `<target>@sglang` - the launch of the card "without the `--speculative-*` flags"
  (`disable_radix_cache`, `mem_fraction_static`), batch size 1.
- `<target>+DSpark` - the same launch with the draft attached exactly as the card
  specifies:
    - `speculative_algorithm=DSPARK`
    - `speculative_draft_model_path`
    - `speculative_draft_attention_backend=flashinfer`
    - for VL-3B: `speculative_dspark_block_size=9`
    - `mem_fraction_static` is 0.8 for VL-3B and 0.75 for the text targets. For the text
      targets, the code reads the block size from the draft config.

The plain Transformers run of each target stays in the roster. The SGLang pair adds a
like-for-like speed comparison. It does not replace the reference run. VL-3B receives a
text-only user turn through the chat template of its processor.

Prompts reach SGLang as token ids. The same chat template as the Transformers runs
produces them. Therefore, the three runs of a target receive the same input tokens.

SGLang pins its own torch and transformers. For this reason, it is a separate
`speculative` extra. `[tool.uv]` declares that it conflicts with `inference`. The node
script of Grid'5000 syncs the environment again between the two runtimes.

## The lossless check

`lrb report` and the dataset card compare every speculative run with every plain run of
the same target. The runs must have the same budget and prompt. The comparison goes item
by item. It counts the verdicts that differ and the raw generations that differ.

- Compare with the baseline of the **same runtime** (`@sglang`). Any difference is a
  defect. The cause can be wrong settings, a kernel that is not greedy, or a draft for
  another target. In this case, `lrb report` exits with a non-zero code.
- Compare with the **Transformers** run. The report gives this information only. The
  bf16 kernels of two runtimes can break a near-tie in different ways. A small number of
  generations that differ does not show that the draft changed the output.

## Consequences

- The scores of `+DSpark` must equal the scores of `@sglang`. The speed columns (ADR-0009)
  carry the result. They include the mean accept length and the draft accept rate of
  SGLang.
- The roster now has 13 runs. Eight of them run on SGLang at batch size 1. Plan the
  Grid'5000 walltime for this.
- If the recipe of a card changes, the settings of the roster change with it. The result
  files record the settings that the run used (`speculative` in the metadata).
- The DSpark worker of SGLang reads `lm_head` and the hidden-state capture hook from the
  top-level target model. But `Lfm2VlForConditionalGeneration` keeps both on its inner
  `language_model`. For this reason, `LFM2.5-VL-3B+DSpark` failed at engine start.
- `adapters/sglang_compat.py` forwards them to the inner model (same weights). The code
  loads it into the spawned processes of SGLang through `PYTHONPATH`. On English, the
  output was byte-identical to `LFM2.5-VL-3B@sglang` on 300/300 items, with the draft
  engaged (accept rate 0.29). Remove the shim when SGLang resolves these on the wrapper.

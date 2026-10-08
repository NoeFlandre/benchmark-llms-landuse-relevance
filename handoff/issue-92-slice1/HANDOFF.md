# HANDOFF: issue #92, slice 1 (speed card renderers, rounding, CRAP)

Repository: NoeFlandre/benchmark-llms-landuse-relevance
Issue: #92 (open) "[refactor] hf_publish.py: split the 866-line module and the 130-line dataset_card"
Status: slice 1 complete locally and pushed to handoff branches. Issue #92 stays open. No PR was opened. Nothing was merged, closed or force-pushed.

## 1. Goal and exact completed scope

Goal of slice 1:
1. Replace repeated optional-value rounding in `_speed_rows` with `results_store.rounded`, keeping each field's digits and None handling.
2. Move the pure runtime-performance card renderers out of `adapters/hf_publish.py` into `adapters/publishing/card_sections.py`, with byte-identical output and preserved public names.
3. Reduce `_speed_rows` CRAP (it was 13, then 6, now 3).

Completed (commits on top of main 5e52649, in order):
- c5aef69183617337f89c4f02e9a7f9715492c597 test: pin runtime-performance card output before refactor (#92). Adds `tests/unit/test_card_speed.py` (7 tests) and `tests/golden/speed_section_synthetic.md`.
- 8dd9aee5ce7cb6d9d925dc746bb91c1e07bf819e refactor: round optional speed figures with results_store.rounded (#92). Changes `hf_publish.py` only.
- e004a97a0bca459055b893d775d65df430dd3f2c refactor: move speed card renderers into adapters/publishing (#92). Adds `adapters/publishing/__init__.py` and `adapters/publishing/card_sections.py`. Changes `hf_publish.py`. Changes one import line in `tests/unit/test_card_speed.py`.
- 0431a4ab37a5370093d439479710a9fd70fbd986 docs(publishing): describe the shared table helper in card_sections (#92). One docstring line.
- 6df2e46f96f1e1c91f9376e2ba4af49026f8ada6 refactor: pool a model's speed in its own helper (#92). Adds `_pooled_speed`. `_speed_rows` CRAP 6 to 3. HEAD of the slice.

Not in scope and not done: `card_validation`, `hub`, scoring, logprob and agreement renderers, `domain/agreement.py` move (see section 5).

## 2. Remote branches and SHAs

| Branch | Remote head | Status |
|---|---|---|
| handoff/issue-92-slice1-code-6df2e46 | 6df2e46f96f1e1c91f9376e2ba4af49026f8ada6 | Pushed. Code and tests only. Verified against GitHub (see section 6). |
| handoff/issue-92-slice1-notes-6df2e46 | recorded in the archive's inventory/SHA256SUMS.txt and in the final report (a commit cannot contain its own SHA) | Pushed. This HANDOFF.md, sanitised evidence, checkout inventory and restoration results. Parent is 6df2e46. |
| integration/92-with-81 | 00ab5a093f0cae061cf9dbb7295d81e4d9f2a032 | NOT pushed. Experimental scratch merge in the recovery archive only. Not a merge candidate. |
| main | 5e52649b1a583c6e96b77cb83e090d99feea6c6f | Unchanged. Not written to. |
| fix/mutation-gate-entrypoint (PR #81 head) | 8f4fbd4a0030a445faae236d82c2f50ab6569d90 | Unchanged. Read only (fetched). |

Prerequisites: slice commits need 5e52649 (main). Integration scratch needs 6df2e46 and 8f4fbd4.

Full SHAs of the slice chain, in order: 5e52649b1a583c6e96b77cb83e090d99feea6c6f, c5aef69183617337f89c4f02e9a7f9715492c597, 8dd9aee5ce7cb6d9d925dc746bb91c1e07bf819e, e004a97a0bca459055b893d775d65df430dd3f2c, 0431a4ab37a5370093d439479710a9fd70fbd986, 6df2e46f96f1e1c91f9376e2ba4af49026f8ada6.

Tree SHA of the slice head 6df2e46: aa4e0dc9486452d0ea892e5e159fd3054baf5490. Tree SHA of the integration scratch 00ab5a0: c26dcaa14fda7c6a84345fa6e4892474538393d8.

## 3. Exact-commit tests and reviews

Labels: PASSED, FAILED, SKIPPED, UNRUN, STALE. Results are for exact SHAs unless marked otherwise.

### Exact commit 6df2e46 (slice HEAD), Python 3.12, Linux x86_64 container

| Check | Result | Detail |
|---|---|---|
| make lint (ruff format and check) | PASSED | |
| make types (ty check src) | PASSED | |
| make test (unit and property) | PASSED | 580 passed. Coverage 97.45%, floor 95. |
| make acceptance | PASSED | 16 passed |
| make architecture (import-linter) | PASSED | 1 passed |
| crap (scripts/crap.py, same arguments as Makefile target) | PASSED | exit 0; `_speed_rows` 3.00, `_pooled_speed` 4.00. Run on coverage from make test. |
| make smoke | PASSED | |
| make wheel | PASSED | |
| make docs-build (mkdocs build --strict) | PASSED | exit 0, built in 0.49 s |
| make scripts (tests/scripts) | FAILED (environment) | 2 failed, 22 passed. Both fail with "ssh executable is unavailable" (no ssh binary in the container). The same 2 tests fail on base 5e52649 in the same environment, so this predates the slice. Not a pass. |
| make mutation | UNRUN | CI-required. Heavy. Not run. |
| make security (pip-audit) | UNRUN | Not run. |
| make lockfile (full target) | UNRUN for the speculative dry-run. The `uv lock --check` step FAILED on this branch, see the lock row. | |
| uv lock --check, pinned uv 0.11.16 (Dockerfile UV_VERSION) | FAILED (pre-existing) | exit 1 on main's lock. Cause: pyproject sets prerelease "if-necessary", lock records the default mode. Slice did not touch pyproject or uv.lock. Not rewritten. |
| uv lock --check, local uv 0.11.32 | FAILED (pre-existing) | exit 1, same cause |
| make integration (downloads a model) | UNRUN | Out of scope for this environment |
| Python 3.11 leg of CI | UNRUN | Only 3.12 was used |
| make docker / CI image build | UNRUN | |
| GitHub Actions on the handoff branch | UNRUN | No PR was opened, so no CI run was triggered by this handoff |

Reviews of 6df2e46:
- Haiku reconciled-SHA review (behaviour and CRAP claim, read only): PASS. Nits: `_pooled_speed` docstring says "every language run of a model", but the grouping key is the run name (run_id, or model_id when run_id is empty). Wording only. Commit trailer wording (see section 7).

### Earlier commits (stale for 6df2e46, kept for history)

- e004a97, behaviour lens: PASS. Structure lens: PASS. STALE for 6df2e46. The later diff 0431a4a..6df2e46 is covered only by the reconciled review above.
- 0431a4a, Haiku re-review of the docstring commit: PASS. STALE for 6df2e46 (same reason).
- 0431a4a gates (lint, types, test 580, acceptance, architecture, crap, smoke, wheel, docs build): PASSED at that SHA. STALE for 6df2e46 because `_speed_rows` changed.
- 5e52649 baseline gates (573 tests, coverage 97.45%, and the rest): PASSED at that SHA. Used only as the base reference.

### Integration scratch 00ab5a0 (experimental, not pushed, not a merge candidate)

| Check | Result | Detail |
|---|---|---|
| Merge of PR #81 head 8f4fbd4 into 6df2e46, with the proposed resolution | Conflict-free | Resolution = keep `results_store.rounded` and `test_rounding_is_idempotent` (see section 4) |
| Without the resolution (#81 as written) | FAILED | Importing `landuse_relevance_bench.adapters.hf_publish` raises ImportError: cannot import name 'rounded' from results_store. This was observed in the run and is transcribed here; the output was not saved to a file. |
| lint, types, make test (621 passed, 97.73%), acceptance (16), architecture, crap, smoke, wheel | PASSED | |
| uv lock --check, pinned uv 0.11.16, with #81 uv.lock | PASSED | exit 0 (the #81 `[options] prerelease-mode` hunk fixes the lock check) |
| make mutation | UNRUN | Heavy. PR #81's mutation allowlist is stale against main (its write_run entries describe the old inline json.dumps; main moved it into dumps_run_payload). Expected to fail; not verified. |
| make security, make scripts, make docs-build | UNRUN | |
| Haiku review of the scratch merge | FAIL | Major: mutation gate not run, and the #81 allowlist staleness. Minor: lockfile gate not reported (now reported above). This FAIL is unresolved; the mutation gate is the reason. |

### Other read only audits
- Review verdicts saved as JSON in evidence/reviews/: wf_727a89da-951 (two lenses on e004a97, both pass) and wf_3e35aabf-d7d (6df2e46 pass; 00ab5a0 fail, see above).

## 4. Remaining work, defects, dependencies, overlaps and decisions

Dependency on `results_store.rounded`:
- main 5e52649: `src/landuse_relevance_bench/adapters/results_store.py` defines `rounded` (returns None for None, else `round(value, digits)`).
- Genuine production caller on the slice: `src/landuse_relevance_bench/adapters/publishing/card_sections.py`, imported at module load. Existing test caller: `tests/property/test_leaderboard_properties.py` (`test_rounding_is_idempotent`).
- If `rounded` is removed, `hf_publish` fails to import. Fallback, not implemented: a private `_round_optional` in card_sections.py (7 call sites). Preferred: keep `rounded`.

Overlap with PR #81 (draft, owner NoeFlandre, base 72b05297, stale against main):
- DELETES `results_store.rounded` (def plus 4 lines, hunk @@ -89,11 +89,6 @@). Exact hunks are in evidence/pr81-relevant-hunks.diff.
- DELETES the `rounded` import and `test_rounding_is_idempotent` (and the `hypothesis.strategies` import it used) in `tests/property/test_leaderboard_properties.py`.
- Adds `[options] prerelease-mode = "if-necessary"` to `uv.lock`. Keep this: it makes the pinned-uv lock check pass.
- Its `mutation-allowlist.txt` write_run entries are stale against main. Regenerate with `make mutation` on a rebased #81 (heavy).
- Also touches `results_store.py` (threshold sweep rewrite, removes write_threshold_sweep_csv and write_scoring_summary_csv), `scripts/check_mutants.py`, `domain/uncertainty.py`, several tests.
- Proposed resolution (preferred, validated on the scratch): keep `rounded` and its property test. Revert those two #81 hunks. Keep all other #81 changes. Status: decided by Master 1. Not yet delivered.

Other open PRs and issues (ownership overlaps, not touched by this work):
- PR #134 (open): `domain/scorers.py` (adds `ScorerSpec.trust_remote_code`), `adapters/hf_scorer.py`, `tests/unit/test_hf_scorer_runtime.py`, CHANGELOG. The later scoring renderer in #92 reads `scorer_for(...).card`, so coordinate.
- PR #135 (open): `.github/workflows/ci.yml`, `docker-smoke.yml`.
- Issues #120 (trust_remote_code) relates to #134; #124 (duplicate docker builds) relates to #135.

Remaining #92 scope (not started):
1. Move `card_validation` (integrity checks returning ValidatedRuns) and `hub` (publish_results, _default_api, _reject_archive_paths, write_viewer_dataset) out of `hf_publish.py`, re-exporting public names.
2. Move scoring renderers (_scoring_section, _scoring_setup_row, _scoring_prompts, _encoder_output_behavior_note, _card_summary_value, SCORING_SUMMARY_CARD_COLUMNS), logprob renderers (_logprob_comparison, _logprob_section, _comparison_row, _same_gpu_row) and agreement rendering into `adapters/publishing/`.
3. Move `_agreement_section` coverage and equality logic into `domain/agreement.py`; remove `_agreement_section` from `scripts/crap-allowlist.json`.
4. Target: no file over about 300 lines. `hf_publish.py` is 775 lines now.
5. `docs/debt.md` still says hf_publish is about 580 lines (stale). Outside the slice boundary.

Known nits (not blocking):
- `_pooled_speed` docstring wording (see section 3).
- `docs/debt.md` line (above).

Pending decisions:
- Master 1: reconcile PR #81 (preferred: keep rounded and its property test; keep the uv.lock [options] hunk; regenerate the mutation allowlist on a rebased #81). Do this on a NEW branch. Do not force-push #81.
- Whether to run `make mutation` (heavy, CI budget 90 minutes).
- Whether to apply the two nits above.

Public API: `hf_publish` re-exports SPEED_COLUMNS and EXPECTED_FULL_SWEEP_LANGUAGE_COUNT with `# noqa: F401`. Four private names are no longer importable from hf_publish: `_markdown_table`, `_single_setting`, `_reproducibility_note`, and an incidental `summarise_speed` import. No in-repo consumer used them. The test file now imports `_reproducibility_note` from card_sections.

## 5. Processes and jobs

- No agents, workers, background jobs or test runs were active at wrap-up (checked: no matching processes, no running agents).
- No production jobs were touched.
- Local checkouts are listed in inventory/checkouts.txt. None has uncommitted or untracked changes. Ignored build outputs (.venv, site/, coverage.json, caches) are excluded from the archive.

## 6. Verification of this handoff (run at delivery time)

GitHub verification (read only, GitHub commit lookup):
- handoff/issue-92-slice1-code-6df2e46 head = 6df2e46f96f1e1c91f9376e2ba4af49026f8ada6. Parent chain: c5aef69 -> 8dd9aee -> e004a97 -> 0431a4a -> 6df2e46 -> main 5e52649. Verified.

Restoration tests (clean clones, no installs, no heavy jobs). Expected tree SHAs: 6df2e46^{tree} = aa4e0dc9486452d0ea892e5e159fd3054baf5490; 00ab5a0^{tree} = c26dcaa14fda7c6a84345fa6e4892474538393d8.
- A. Fresh clone of the GitHub code branch: HEAD 6df2e46, tree matches, the five changed files match by SHA-256 against the local checkout. PASSED.
- B1. Fresh clone of main (5e52649) plus the incremental slice bundle (requires 5e52649): verifies; fetched ref is 6df2e46 with the expected tree. PASSED.
- B2. Fresh clone of main plus git am of the five patches: tree aa4e0dc9486452d0ea892e5e159fd3054baf5490 matches. The commit SHA differs (0f3f0179d1574c12aa7bd22a3df59b686cbb699a), because committer timestamps differ. Content matches. PASSED (content only).
- C. Experimental integration bundle (requires 6df2e46 and 8f4fbd4 from GitHub): verifies; restored ref 00ab5a0 with the expected tree. PASSED. Not a merge candidate.

Incremental bundles do not embed their prerequisite commits. Doing so would embed the repository's data/ history into the attachment. Prerequisites are the public commits 5e52649 (main) and 6df2e46 and 8f4fbd4, which are on GitHub; their SHAs are listed above.

## 7. Safest first steps for Master 1

1. Read section 3 first. Do not treat FAILED or UNRUN rows as passes.
2. Fetch `handoff/issue-92-slice1-code-6df2e46` and confirm the head is 6df2e46f96f1e1c91f9376e2ba4af49026f8ada6. Do not push to main.
3. Confirm the lock problem with the pinned uv before anything else: `uvx --from uv==0.11.16 uv lock --check` (read only). Expect exit 1 on main's lock.
4. For #81: create a NEW branch from current main. Revert only the two `rounded` hunks (see evidence/pr81-relevant-hunks.diff). Keep the uv.lock [options] hunk. Do not force-push or edit `fix/mutation-gate-entrypoint`.
5. Run gates one at a time (heavy): `UV_FROZEN=1 uv sync --frozen --python 3.12 --extra inference --extra scoring --extra publish`, then `make lint types test acceptance architecture crap smoke wheel`. Expect scripts failures unless ssh is installed. Run `make mutation` only when you accept the cost.
6. Commit trailers: the slice commits use `Co-Authored-By: Claude <noreply@anthropic.com>` and the Claude-Session trailer. Keep them unchanged unless policy says otherwise.

## 8. Attribution and privacy

- Commit author and committer: Claude <noreply@anthropic.com>. No personal email was added. The archive and this file contain no credentials, tokens, machine paths or personal data (checked by pattern scan, see inventory).
- Paths in this file are repo-relative. Machine-specific paths were replaced by placeholders in the evidence.

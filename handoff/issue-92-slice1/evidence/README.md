# Evidence index (sanitised; machine paths replaced by placeholders)

Status key: PASSED, FAILED (environmental unless noted), UNRUN, STALE (superseded by a later SHA).

- gates-6df2e46-summary.txt: gates at the slice head 6df2e46. lint, types, tests (580), acceptance, architecture, crap, smoke, wheel, docs build: PASSED. make scripts: FAILED, environmental (ssh missing; reproduced on base 5e52649).
- gates-00ab5a0-integration-summary.txt: gates on the scratch integration 00ab5a0 (tests 621, coverage 97.73%): PASSED. Not a merge candidate.
- gates-0431a4a-summary-superseded.txt: STALE (before the pooled-speed helper).
- gates-5e52649-baseline-summary-superseded.txt: base reference, STALE for the slice.
- scripts-ssh-limitation.txt: the 2 ssh-related failures, reproduced on base 5e52649 and on 6df2e46.
- lock-checks.txt: uv lock --check, pinned uv 0.11.16 (rc 1 on slice lock; rc 0 with PR #81 lock) and local 0.11.32 (rc 1). Pre-existing lock mismatch; not rewritten.
- crap-changed-functions-6df2e46.txt: CRAP of _speed_rows (3.00) and _pooled_speed (4.00).
- pr81-relevant-hunks.diff: the PR #81 hunks that delete results_store.rounded and its property test, plus the uv.lock [options] hunk.
- reviews/wf_727a89da-951-verdicts.json: e004a97 review, two lenses, both PASS. STALE for 6df2e46.
- reviews/wf_3e35aabf-d7d-verdicts.json: 6df2e46 reconciled review PASS (nits); 00ab5a0 integration review FAIL (mutation gate not run; #81 allowlist stale).
- reviews/0431a4a-rereview-transcribed.txt: docstring commit review PASS, transcribed from the report. STALE for 6df2e46.

NOT RUN (no evidence exists): make mutation, make security, the speculative lock dry-run, integration tests (model download), Python 3.11 leg, Docker build, GitHub Actions.

# Archival integration bundle (experimental; NOT a code branch; NOT a merge candidate)

Bundle: integration-00ab5a0.experimental.incremental.bundle (this folder)
Size: 2168 bytes
SHA-256: 38b550b39e488f537a568bbb60bfe37a29cb6d7dbe344cc4321a7b47a8d8f029

What it contains (git bundle verify output, verified before upload):
- Ref refs/heads/integration/92-with-81 = 00ab5a093f0cae061cf9dbb7295d81e4d9f2a032
- That one commit is a merge of PR #81's head into the #92 slice head, with the proposed resolution (keep results_store.rounded and its property test). It is a scratch check, not a merge candidate.

Prerequisites (both public on GitHub; the bundle does not embed them):
- 6df2e46f96f1e1c91f9376e2ba4af49026f8ada6 = branch handoff/issue-92-slice1-code-6df2e46 (slice code head)
- 8f4fbd4a0030a445faae236d82c2f50ab6569d90 = branch fix/mutation-gate-entrypoint (PR #81 head, read only)

Expected after restoration:
- commit 00ab5a093f0cae061cf9dbb7295d81e4d9f2a032
- tree c26dcaa14fda7c6a84345fa6e4892474538393d8
- parents 6df2e46f96f1e1c91f9376e2ba4af49026f8ada6 and 8f4fbd4a0030a445faae236d82c2f50ab6569d90

Restoration, from a fresh clone of this branch:

    git clone --branch handoff/issue-92-slice1-notes-6df2e46 https://github.com/NoeFlandre/benchmark-llms-landuse-relevance restore-integration
    cd restore-integration
    sha256sum handoff/issue-92-slice1/archive/integration-00ab5a0.experimental.incremental.bundle   # expect 38b550b3...f029
    git fetch origin handoff/issue-92-slice1-code-6df2e46 fix/mutation-gate-entrypoint              # public prerequisites
    git bundle verify handoff/issue-92-slice1/archive/integration-00ab5a0.experimental.incremental.bundle
    git fetch handoff/issue-92-slice1/archive/integration-00ab5a0.experimental.incremental.bundle integration/92-with-81:refs/heads/restored/integration-92-with-81
    git rev-parse restored/integration-92-with-81          # expect 00ab5a093f0cae061cf9dbb7295d81e4d9f2a032
    git rev-parse restored/integration-92-with-81^{tree}   # expect c26dcaa14fda7c6a84345fa6e4892474538393d8

Do not push restored/*. It is a local check branch.

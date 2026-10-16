---
name: ai-infra-wiki-pages-publication
description: Build and publish source-grounded interactive learning pages in ai-infra-wiki while preserving a dirty checkout. Use for SGLang Pages guides, visualizations, and verification.
allowed-tools: Read, Grep, Bash
---

# Build and publish ai-infra-wiki interactive Pages content

## When to use

Use for a new or revised `ai-infra-wiki` learning page that needs interactive modules, source anchors, or GitHub Pages publication. Do not use it for unapproved production changes, or to reset a dirty checkout without the user's explicit remote-authoritative choice.

## Inputs / context to gather

1. Read `AGENTS.md`, relevant navigation/workflow files, and the closest existing learning page.
2. Identify the intended audience, required interactions, exact source baseline, and whether the checkout is dirty.
3. Locate existing test/build scripts before adding new ones; use the configured Node/Python runtimes when system paths are incomplete.

## Procedure

1. Pin the upstream SGLang commit and state boundaries between source facts, teaching models, and unvalidated performance claims.
2. Keep detailed explanation adjacent to the interactive module. Model independently meaningful state (for example, queue membership, execution references, resource ownership, and readiness gates) rather than reducing it to animation bars.
3. Add source anchors, navigation, responsive/keyboard behavior, and regression coverage for teaching invariants and source-anchor drift.
4. Run the repository documentation/link checks, topic-specific unit tests, browser tests at mobile and desktop widths, and the Pages build/lifecycle tests.
5. For a dirty checkout, publish from an isolated temporary/detached worktree containing only the authorized files. Verify its staged set and clean state before push.
6. Verify the GitHub Pages workflow and compare deployed HTML/content-versioned assets with the validated build. Report a live-browser timeout as a verification boundary, not a page failure, when byte comparison and CI succeeded.

## Efficiency plan

- Start from the nearest existing page and its tests; cache the pinned commit and test command list.
- Locate an available Node.js and Python runtime; pass their executable paths locally rather than committing machine-specific paths.
- Stop and ask before destructive synchronization; if the user chooses remote authority, back up and compare first.

## Pitfalls and fixes

- Generic source checker rejects an independently pinned new page -> make the validator baseline-aware while retaining checks for old pages.
- Browser navigation false-fails after a cross-page click -> wait for `load` before asserting deferred navigation state.
- `SyntaxError: Non-UTF-8 code` under system Python -> rerun with the configured bundled Python, then investigate only if it persists.
- A visual model implies performance -> label scenario values as illustrative unless it models arrivals, contention, failures, and runtime execution.

## Verification checklist

- [ ] Every major subsection has the requested interactive behavior and adjacent explanation.
- [ ] Source revision and teaching/performance boundaries are explicit.
- [ ] Unit, link, browser, navigation, responsive, and workflow checks pass.
- [ ] Published commit/ref and deployed assets match the validated build.
- [ ] Original dirty checkout was preserved.

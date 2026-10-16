---
name: publish-exact-files-from-dirty-worktree
description: Publish only named files when the current Git checkout has unrelated dirty, untracked, or submodule changes.
allowed-tools: Read, Grep, Bash
---

# Publish exact files from a dirty worktree

## When to use

Use for a small explicit file set in a dirty/diverged checkout. Do not use for a whole-worktree release or without user authorization for remote and branch.

## Inputs / context to gather

1. Confirm target-file repository, remote/branch, and exact allowed files.
2. Inspect `git status --short`, remotes, fetched remote branch, and divergence.
3. Record existing submodule/untracked state.

## Procedure

1. Make unique `/private/tmp` detached worktree from fresh `<remote>/<branch>`.
2. Copy/cherry-pick only authorized changes; never reset/clean/switch original checkout.
3. Explicitly `git add -- <file>...`; inspect `git diff --cached --name-only` and `git diff --cached --check`.
4. Commit and push `HEAD:<branch>` only after destination authorization.
5. Verify `git ls-remote` equals commit, remove temporary worktree, confirm original state remains.

## Efficiency plan

- Cache allowlist and remote/branch before mutations; stop if staged files differ or remote has unexpected unique commits.
- For `HTTP2 framing layer`, use per-command HTTP/1.1, then one-off SSH URL without editing remotes.

## Pitfalls and fixes

- Stale main: replay from fresh remote in temporary worktree.
- Markdown trailing spaces: determine whether intentional hard breaks before edits.
- Missing directory: `install -d` before copy. If narrow cleanup rejects `rm -f`, use `unlink` on explicit temporary file.

## Verification checklist

- [ ] Allowlist exactly matches staged files.
- [ ] Cached diff is clean or intentional formatting is understood.
- [ ] Remote ref equals pushed commit.
- [ ] Original dirty/untracked/submodule state was not altered.

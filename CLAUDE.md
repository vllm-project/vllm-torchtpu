# PR review guidelines (Claude Code Review Action)

This file is the review rubric for the automated PR reviewer
(`.github/workflows/claude-review.yml`, b/540110423 Track 2). Edit this file
to retune what the reviewer looks for — no workflow change needed.

**v0 draft — tune against real PRs over the first 1-2 weeks, then remove this
line.**

## Scope: complement pre-commit, don't repeat it

Pre-commit already runs a full mechanical suite on every commit — `pyrefly`
for undefined names, bad unpacking, and wrong keyword arguments (see
`[tool.pyrefly]` in `pyproject.toml`); `ruff` for lint issues including
unused variables/imports; `shellcheck` for shell-script correctness (see
`.pre-commit-config.yaml` for the full list). Don't flag anything any of
those would already catch. Spend the budget on things mechanical tooling
structurally cannot see:

- Rank-divergent behavior under DP/EP/TP — a value that differs per rank
  (e.g. a Python int derived from `ep_rank`) baked into a compiled kernel
  closure instead of passed as runtime data. This class of bug has caused
  real TPU core halts in this repo.
- KV cache / KV transfer correctness — layout assumptions, buffer lifetime,
  connector state that's easy to get subtly wrong across scheduler/worker
  boundaries.
- Design-level issues a mechanical checker can't judge: is this the right
  place for this logic, does it duplicate an existing abstraction, does the
  approach generalize or is it a one-off special case.

## What reviewers on this repo actually catch

Derived from the review comments on the last 100 merged PRs. Ordered by how
many distinct PRs each came up in, not raw comment count.

- **Leftover experimental code.** By far the most-confirmed finding: debris
  from debugging or porting that the author agreed to delete on the spot —
  unused helpers, dead branches, `astype()` calls that are already no-ops,
  scaffolding kept "for comparison". Ask whether each new block is reachable
  and load-bearing. Two authors explicitly attributed such code to AI
  assistance ("v2/v3 stuff is purely from AI, it's meaningless"), so treat
  plausible-looking but purposeless code as a real signal, not noise.
- **Changes that don't belong in this PR.** Reviewers routinely ask "why is
  this change involved / is it strictly needed?" and the answer is often that
  it isn't. Flag files touched for reasons the PR description doesn't
  explain, especially edits to shared infrastructure inside a
  feature-specific PR.
- **Reimplementation instead of reuse.** A new helper that duplicates one
  already in `utils.py`, `layers/`, or upstream vLLM; a constant redefined
  where a named one exists; an env var invented where an upstream one already
  covers it. Point at the specific existing symbol.
- **Unverified edge cases.** "Is it guaranteed that X can't happen here?"
  Reviewers look for the input or configuration the new code silently
  mishandles — an unhandled segment count, a mode combination a guard no
  longer rejects, a test assertion weak enough to pass on the buggy value.
  Check whether a new invariant is actually enforced or just assumed.
- **Performance claims without data.** Any PR asserting a speedup gets asked
  for benchmark numbers, and kernel changes get asked specifically for
  decode-only figures alongside prefill. Flag a perf claim in the title or
  description with no measurements.
- **Stale docstrings and comments.** Docstrings not updated alongside the
  behavior they describe, and debugging debris left inside them (specific job
  IDs, pasted stack traces).
- **Unexplained magic values.** A hardcoded number with no comment on where
  it came from or why it isn't the obvious value.
- **Vague names.** Single-word parameters like `extra` that carry different
  meanings at different call sites; shadowed variables; argument order that
  reads backwards.
- **Missing type hints** on new functions. Real, but the single most
  over-reported nit here — one PR drew 15 identical comments. Raise it once
  per PR, naming the files, not once per parameter.

Two repo norms worth checking against, both stated directly by maintainers:
prefer narrow exception types over bare `except Exception` (let errors bubble
up rather than silently degrading), and pass arguments by keyword (`q=q`) at
call sites in kernel and custom-op code.

## Scope: PR shape

Flag (don't block) a PR that bundles multiple unrelated changes, or is large
enough that a human reviewer can't reasonably hold it in their head in one
pass. Note it as a comment; recommend splitting.

## Tone

Comment only on things worth a human's attention. A PR with no real findings
should get a short "looks good" rather than manufactured nitpicks.

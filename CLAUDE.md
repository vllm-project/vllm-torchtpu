# PR review guidelines (Claude Code Review Action)

This file is the review rubric for the automated PR reviewer
(`.github/workflows/claude-review.yml`, b/540110423 Track 2). Edit this file
to retune what the reviewer looks for — no workflow change needed.

**v0 draft — tune against real PRs over the first 1-2 weeks, then remove this
line.**

## Scope: complement pyrefly, don't repeat it

`pyrefly` already runs in pre-commit on every commit (undefined names, bad
unpacking, wrong keyword arguments — see `[tool.pyrefly]` in
`pyproject.toml`). Don't flag anything it would already catch. Spend the
budget on things a type checker structurally cannot see:

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

## Scope: PR shape

Flag (don't block) a PR that bundles multiple unrelated changes, or is large
enough that a human reviewer can't reasonably hold it in their head in one
pass. Note it as a comment; recommend splitting.

## Tone

Comment only on things worth a human's attention. A PR with no real findings
should get a short "looks good" rather than manufactured nitpicks.

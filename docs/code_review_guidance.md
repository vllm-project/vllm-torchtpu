# vllm-torchtpu Code Review Guidance

How code review works in `vllm-project/vllm-torchtpu`, for both audiences:

- **Human reviewers** — what to look for, in what order, and how to phrase
  and route findings.
- **Review bots and coding agents** — treat the priority order below as the
  review protocol: report findings in that order, most severe first, and
  cite the rule each finding enforces.

This is the third document of the engineering-standard set. It says *how to
review*; the rules being enforced live in the other two:

| Document | Carries |
|---|---|
| [coding_guidance.md](coding_guidance.md) | `R-1.x`–`R-8.x` — style, architecture, reuse, derive-don't-declare, fail-fast, surface area, diff discipline, tests |
| [optimization_best_practices.md](optimization_best_practices.md) | `PR-x` (PR discipline), `P-x` (performance), `Q-x` (quantization) |

Cite those IDs in findings. A finding no rule covers is still valid — say so
and argue it on its merits.

Process mechanics — the `ready` label, the CODEOWNERS approval check, the
merge checklist, the revert policy — live in
[PRE_PUBLIC_DEV_GUIDE.md](PRE_PUBLIC_DEV_GUIDE.md) and are referenced from
§6 rather than restated here.

---

## 1. Two facts that set the priority order

**Bugs here usually do not crash.** A wrong sharding annotation, a bad
quantization scale, a dtype cast that papers over a mismatch, a silent
fallback to a path that returns different numbers: the model runs and quietly
produces wrong output. There is no stack trace — only an eval score a few
points lower, found weeks later. So silent-numerics risk outranks everything
else in review, and "it ran and the output looked reasonable" is not
evidence.

**Upstream moves fast.** vLLM and the torch stack refactor constantly. Code
that depends
on a *name* — a string-matched model id, a `getattr` with a default, a
hard-coded module path, a copied helper — breaks silently the day upstream
renames something. Code that depends on *structure* — shapes, dtypes, typed
configs, registries, hardware APIs — survives. So duplication and
name-based dispatch are review findings, not style preferences.

## 2. Priority order for findings

Check in this order, and report findings most severe first. The default
severity assumes no other information; adjust it with a stated reason.

1. **Silent numeric corruption / kernel breakage.** Unasserted quantization
   assumptions (R-5.2); dtype casts hiding a mismatch (R-5.5); scale-layout
   errors — a weight transpose does not imply a scale transpose (Q-5); calls
   outside a kernel's supported envelope, meaning mesh rank, EP-only, dtype
   packing, or the versions it was written against (P-17); a silent fallback where the result would be
   wrong instead of an actionable error (R-5.3); a swallowed exception that
   hides a shape or scale error (R-5.1); a kernel-backend switch that changes
   a compiled function's traced signature (R-5.6).
   *Default: blocker.*
2. **Hot-path cost.** Work that belongs at weight-loading time (P-14);
   unpipelined DMA, `start(); wait()` back-to-back (P-11); silent upcasts to
   16-bit compute (P-9); a construct in a kernel's main loop that blocks the
   compiler from optimizing across it (P-13); a "no-op on the hot path"
   claim with no large-batch trace (P-1).
   *Default: major; blocker once a regression is demonstrated.*
3. **Duplication / wrong home.** vLLM upstream already has it and should be
   subclassed (R-3.1); a per-model fork of a shared layer (R-3.2); shared
   logic not in its designated home (R-3.3); a file in the wrong directory
   against the R-2.1 table; an interface file carrying hardware-specific
   implementation (R-2.2); a kernel that reads env vars, takes vLLM config
   objects, or imports torch or vLLM outside a dedicated adapter file
   (R-2.3, R-2.1); dead code and unrelated hunks (R-7.3, R-7.5).
   *Default: major.*
4. **Fragility and scalability.** String or `hasattr` dispatch (R-4.1);
   `hasattr`/`try-import` compatibility shims for APIs newer than the pin
   (R-2.5); a hard-coded backend module path where the registry should
   resolve it (R-3.5); magic constants a shape, the mesh, or `config` could
   supply (R-4.x); env reads inside jitted code (R-4.3); a new knob where a
   derived default would do (R-6.1, R-6.2); a fix gated on the single config
   where the bug was observed instead of the variable that drives the
   behavior (R-6.5); "works for the official checkpoint" (R-6.6).
   *Default: major; minor for a single site with no similar occurrence
   elsewhere in the diff.*
5. **Tests and evidence.** Does a new test fail without the fix (R-8.2); is
   it actually wired into CI and not skipped by a pipeline ignore list
   (R-8.4); does it exercise the forward pass rather than import the module
   (R-8.1); does a model-affecting change ship with eval or benchmark
   results (R-8.10); are perf claims backed by a trace with a mechanism
   explanation (P-1); does a quantization change carry before/after numeric
   validation (Q-3) and a weight-loading change carry before/after HBM logs
   (Q-6).
   *Default: blocker for a missing required test; major for an unbacked
   claim.*
6. **Diagnosability.** New log lines are self-contained enough that a reader
   with only the log can find the root cause and the file to open (R-1.7).
   Comments state what the code does now — a constraint, an invariant, a
   non-obvious why — rather than why the change is being made today
   ("previously X", "this fixes…"), which belongs in the PR description
   (extends R-1.8).
   *Default: minor; major when a failure would be undiagnosable.*
7. **Public-repo hygiene.** Every link and reference that lands must be one
   an outside reader can open: a public issue, another PR, a Buildkite build,
   a vLLM source path, a model card, a published paper. The same applies to
   bug identifiers, storage paths, and dashboards — if it needs an internal
   account to open, it does not belong in code, comments, yml, scripts, the
   commit message or the PR text (R-7.12). When the reason behind a design
   needs stating, inline the reasoning instead of linking to where it was
   written down. The reviewer is the last check before this is public.
   *Default: blocker — it cannot be undone after merge.*
8. **Nits last, labeled as nits**, and deferred explicitly.

## 3. Severity rubric

| Severity | Meaning |
|---|---|
| `blocker` | Correctness bug, silent-numerics risk, demonstrated perf regression, missing required test, internal reference in public code, or a change that will break production or CI |
| `major` | Significant duplication, wrong layer or wrong home, perf-regression risk, unproven perf claim, name-based dispatch or a new knob, substantially missing coverage, a failure that would be undiagnosable |
| `minor` | Readability, naming, dead code, one isolated fragility site — worth doing, does not block merge |
| `nit` | Stylistic preference; say it is a nit and defer it |

## 4. Standard probes

Questions that repeatedly find real problems:

- *What happens if this line does not exist?* — the universal probe (R-7.4).
- *Can this be derived from the shapes or the config instead of passed in?*
  (R-4.x)
- *Is there an existing vLLM function to subclass instead?* (R-3.1)
- *Can you share a trace?* — for any perf claim (P-1).
- *What is the mechanism behind the gain?* A gain the mechanism does not
  predict is usually a measurement artifact — for a parallelism A/B, most
  often a batch-size one (P-1, P-4).
- *What happens with an odd block size, or when the checkpoint is not the
  official one?* (Q-5, R-6.6)
- *Does this test fail if you revert the fix?* (R-8.2)

## 5. Review conduct

- **Ask more than you direct.** A question that makes the author explain the
  mechanism finds more bugs than an instruction, and is often how the
  reviewer discovers they were wrong.
- **State your confidence.** "I may be wrong here, happy to hear the
  counterargument" is a normal thing to write. Reverse yourself publicly
  when the author's answer is right.
- **Escalate hard only for §2 items 1 and 3** — numeric safety and
  modularity. "Please roll this back" is reserved for those; everything else
  is a conversation.
- **Ask for the context you need.** A summary of the core issue and how the
  change fixes it is a reasonable thing to request before reviewing, and
  produces better feedback than guessing.
- **Verify the fix landed.** Re-read the diff after the author says done; a
  thread is not resolved because someone replied to it.
- **Redirect work that conflicts with the roadmap.** A contribution to a
  component scheduled for deletion should be pointed at the successor
  instead.
- **Name deferred scope in the thread** rather than letting it disappear on
  merge.

## 6. Routing and merge gating

- **Route approval by ownership.** `.github/CODEOWNERS` is the routing
  table, and the advisory `CODEOWNERS Approval` check reports it; a change
  under a kernel directory needs that owner in addition to a general LGTM.
  Tag the original author of the code being touched. For a large PR,
  delegate the first pass and reserve the second.
- **Oversized diffs may be refused.** "Oversized" is reviewer judgment, not
  a line count (R-7.1). A very large design starts as an RFC, not a PR.
- **Nothing merges until CI is green on the current head**, per the
  Pre-Merge Verification Checklist in
  [PRE_PUBLIC_DEV_GUIDE.md](PRE_PUBLIC_DEV_GUIDE.md): approval, all checks
  green, threads resolved. A green run on an older head is not a green PR.
  Presubmit runs when the `ready` label is applied, so for an external
  contributor the reviewer applies it and merges only on verified green.
- **One escape valve, for CI infrastructure only.** A trivial fix that
  unblocks CI or `main` — a pipeline yml, a broken test config — may merge
  before its own test or cleanup, with an explicit follow-up commitment
  recorded in the thread naming what was deferred. It does **not** cover a
  production change under `src/vllm_torchtpu/`, which needs its test
  (R-8.1), and it does not apply to a landed change that turns out to be
  numerically wrong.
- **A revert is a first-class mitigation.** When a landed change is
  numerically wrong or breaks `main`, revert first and root-cause offline
  (PR-2, and the Emergency Reversion Policy in the dev guide).

## 7. Reviewer quick checklist

Run top to bottom on any PR:

1. Does the description state the mechanism and the evidence, and does the
   title name the capability rather than the model that prompted it?
   (R-7.11)
2. Anything under §2 item 1 — asserts, dtype casts, scale layout, kernel
   envelope, silent fallbacks, swallowed exceptions?
3. Anything under §2 item 2 — work that belongs at load time, an
   unpipelined DMA, a silent upcast, an unproven no-op claim?
4. Does vLLM upstream already have this, and is each file in the directory
   the R-2.1 table assigns it?
5. Any string or `hasattr` dispatch, magic constant, or new knob?
6. Is there a test, does it fail without the fix, and does CI actually run
   it?
7. Do the new log lines identify a root cause on their own, and do the new
   comments say what the code does rather than why it changed?
8. Can an outside reader open every link and reference in the diff?
   (R-7.12)
9. Is CI green on the current head, and is every deferred item named in a
   thread?

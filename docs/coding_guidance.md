# vllm-torchtpu Coding Guidance

Engineering standard for code in `vllm-project/vllm-torchtpu`. It is normative
for two audiences:

- **Human authors** — apply it while writing code; run the author checklist
  (at the end of this doc) before sending a PR.
- **Coding agents** — treat every MUST/NEVER as a hard constraint and every
  SHOULD as the default that requires a stated reason to deviate from.

Reviewers and review bots cite these rule IDs (and the P-x/Q-x IDs from the
optimization doc) in their findings.

Rules are numbered `R-<section>.<n>` so findings and instructions can point at
them precisely. IDs are global across the document set: R-1.x–R-8.x live
here; PR-x (PR discipline), P-x (performance), and Q-x (quantization) live
in the companion
[optimization_best_practices.md](optimization_best_practices.md).

Two meta-principles organize this document set:

1. **Reuse upstream.** vLLM upstream is the first place to look for any
   functionality. This repo exists to add TPU behavior to vLLM, not to
   re-implement vLLM.
2. **Derive, don't declare.** Any value recoverable from tensor shapes, dtypes,
   hardware APIs, sharding specs, or cost models is derived at the point of
   use — never duplicated into a config knob, env var, magic number, or
   redundant member variable. (Rationale: see the
   [Appendix](#appendix--rationale-for-derive-dont-declare).)

---

## 1. Python style

- **R-1.1** Follow the
  [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
  for everything the repo tooling does not decide: naming, docstrings
  (Google-style `Args:`/`Returns:`/`Raises:`), comments, exception style,
  default-argument rules, comprehension complexity limits.
- **R-1.2** Formatting is decided by the repo's `pre-commit` config — `yapf`,
  `isort`, `ruff` — and the CI pre-commit check gates every PR. If
  `pre-commit install` was run in the clone, the hooks fire automatically on
  each commit — but only on that commit's staged files, and fresh
  clones/worktrees usually lack the install. So before every push, run
  `pre-commit run --all-files` (or at minimum `yapf -i` / `isort` / `ruff
  check --fix` on changed files). Never use `--no-verify` to push code the
  formatters haven't actually been run on; if hooks can't run locally, run
  yapf/isort/ruff manually first.
- **R-1.3** 80-column limit. When a multi-line call or literal keeps getting
  re-wrapped, add a trailing comma after the last element so yapf pins
  one-element-per-line.
- **R-1.4** Explicit imports only — no `from x import *`. Import modules or
  names directly; no aliasing to obscure single letters.
- **R-1.5** Use modern typing: PEP 604 unions (`int | None`, not
  `Optional[int]`), builtin generics (`list[int]`), `StrEnum` for string
  enums. Type-annotate public function signatures. The repo runs `pyrefly`
  at the `basic` preset in CI; code must pass it.
- **R-1.6** Every new file carries the Apache 2.0 license header (copy from a
  sibling file).
- **R-1.7** No `print` in library code — use `vllm_torchtpu.logger.init_logger`.
  Log lines are self-contained and AI-debuggable: tag the subsystem, include
  the relevant shapes/values, and name the failure mode. A reader with only
  the log (no source) should be able to identify root cause and the file to
  inspect. Use `warning_once`/`info_once` for per-process one-shot messages
  (e.g. unexpected fallback paths; tuned-table misses follow P-16 in the
  optimization doc); log only the branch the user didn't expect.
- **R-1.8** **Minimize comments — prefer self-documenting code.** Eliminate
  comments that restate what the code already says; rename or restructure
  instead. Write a comment only for what the code cannot show: a constraint,
  an invariant, a non-obvious why. Keep docstrings and comments brief and
  direct, and assume the reader is familiar with vLLM and this repo.

## 2. Architecture ground rules

vllm-torchtpu is a **vLLM platform plugin, not a fork**. vLLM installs as a
plain wheel; this repo injects TPU behavior through vLLM's
[plugin entry points](https://docs.vllm.ai/en/latest/design/plugin_system/)
(`vllm.platform_plugins`, `vllm.general_plugins`). Model execution is
**torch** (`TPUModelRunner` subclasses vLLM's `GPUModelRunner`, running under
`torch.compile(backend="tpu", fullgraph=True)`); kernels are **JAX/Pallas**,
bridged into torch as custom ops via `torch_tpu._internal.pallas.jax_op`
(plus the sharded variant, `distributed/sharded_jax_op.py`). This is **one
example** of the bridge pattern — there are many call sites
(`layers/vllm/custom_ops/*_op.py`, `fused_moe.py`, `linear_common.py`, …),
not a single entry point:

```python
# layers/vllm/linear_common.py — one instance of the kernel-bridge pattern
op = pallas.jax_op("pallas::quantized_matmul_kernel",
                   functools.partial(quantized_matmul_kernel, ...))
```

- **R-2.1** Directory responsibilities (all under `src/vllm_torchtpu/`):

  | Directory | Role | Allowed dependencies |
  |---|---|---|
  | `kernels/` | JAX/Pallas kernels, tuned tables, kernel-eligibility predicates | jax/pallas + small pure helpers (`utils.py`, `logger.py`). **No vLLM imports, no torch** (torch only inside dedicated adapter files, e.g. `vllm_adapter.py`, that bridge a kernel to the torch side), no env reads |
  | `layers/common/` | Shared numeric/layout logic usable from any front-end | jax first; keep torch and vLLM imports at the boundary and minimal |
  | `layers/vllm/` | The vLLM/torch-facing layer: quant methods, fused-MoE wiring, attention backends, `custom_ops/` jax_op bridges | torch, vLLM, layers/common, kernels |
  | `platforms/`, `runner/`, `worker/`, `executors/` | vLLM integration surface | thin translation only |

- **R-2.2** **Interface files stay thin.** `platforms/`, `runner/`, and the
  `layers/vllm/custom_ops/*_op.py` bridges only translate between vLLM/torch
  and the JAX side. Hardware-specific implementation belongs in
  `layers/common/` or inside the kernel. No model-specific logic in
  model-agnostic files.
- **R-2.3** **Kernels are self-contained modules.** Everything a kernel needs
  arrives via arguments: no `envs`/`os.environ` reads inside a kernel (the
  jitted-code rule R-4.3 applies doubly here), no vLLM config objects in
  kernel signatures. Tile-tuning logic and
  supported-envelope predicates (`is_supported_by_*`) live inside the kernel
  directory, next to the kernel, not at layer or runner level.
- **R-2.4** Name kernels and modules by **functionality, not by model**
  (`ragged_paged_attention/`, `quantized_matmul/`, `kernel_hd64.py`). A
  functional name states what it does, when it is useful, and what can be
  deleted without collateral damage. A model-named kernel directory is only
  acceptable for a genuinely model-specific architecture component, and it
  must not become the home for reusable pieces — factor those out.
- **R-2.5** **Write against the pinned versions.** The JAX / libtpu /
  torch-tpu / vLLM pins in `pyproject.toml` are load-bearing (startup patches
  in `env_override.py` and `__init__.py` depend on them). No `hasattr`/
  `try-import` compat shims for APIs newer than the pin — call the pinned API
  directly. Existing back-compat branches are removed only when the pin
  advances, in a dedicated PR, with before/after perf evidence.
- **R-2.6** Environment/startup mutation belongs in one place:
  `env_override.py` (which runs before libtpu loads). Never scatter
  `os.environ[...] =` or `LIBTPU_INIT_ARGS` edits through layer or runner
  code.
- **R-2.7** **Read the governing section before touching a gated area.** Do
  not modify code in these areas without first reading and following the
  listed rules; if they conflict with the requested change, stop and explain
  rather than proceeding:

  | Area | Read first |
  |---|---|
  | `kernels/` | R-2.3, R-2.4, and §1 (Performance) of [optimization_best_practices.md](optimization_best_practices.md) (perf evidence, tuned tables, envelopes) |
  | `layers/vllm/quantization/`, `layers/common/quantization.py` | §2 (Quantization) of [optimization_best_practices.md](optimization_best_practices.md) (numeric verification, memory validation) |
  | `env_override.py`, `__init__.py` startup patches, version pins in `pyproject.toml` | R-2.5, R-2.6 (pins are load-bearing) |
  | `platforms/`, `runner/`, `custom_ops/*_op.py` bridges | R-2.2 (interface files stay thin) |

## 3. Reuse upstream first

- **R-3.1** **Search order for any new functionality: vLLM upstream →
  vllm-torchtpu → JAX/Pallas utilities → write it.** Subclass the deepest
  applicable vLLM base class and override only what differs on TPU.
  Leveraging existing vLLM functions instead of copying them is what keeps
  this plugin maintainable against a fast-moving upstream. Example — a quant
  MoE method should inherit the upstream method and override the TPU-specific
  steps, not duplicate `create_weights`:

  ```python
  from vllm.model_executor.layers.quantization.compressed_tensors import (
      CompressedTensorsW8A8Fp8MoEMethod)

  class TpuW8A8Fp8MoEMethod(CompressedTensorsW8A8Fp8MoEMethod):
      # create_weights inherited from upstream — do not copy it.
      def process_weights_after_loading(self, layer):
          ...  # TPU-specific reshaping only

      def apply(self, layer, x, ...):
          ...  # dispatch to the Pallas kernel
  ```

- **R-3.2** **No per-model forks of shared layers.** New models extend or
  subclass the common attention/MoE/RoPE implementation — never carry private
  copies. If the common path can't express what a model needs, extend the
  common path.
- **R-3.3** Shared logic has a designated home (`layers/common/quantization.py`,
  `layers/common/utils.py`, `utils.py`). When flagging duplication, name the
  exact target file, not just "dedupe".
- **R-3.4** **When the defect is upstream, fix it upstream** (vLLM, torch-tpu,
  JAX). An emergency local workaround is acceptable only with an explicit
  upstream fix filed and a TODO referencing it. Sibling plugins (vllm-ascend,
  vllm-spyre) are legitimate precedent for where integration code should live.
- **R-3.5** Use framework extension points, not private APIs: vLLM's plugin
  hooks, `register_oot` registration, `AttentionBackendEnum` registration.
  A private upstream API needs existing precedent from a major framework
  before use. Resolve backend class paths via the registry, never hard-coded
  module strings. Backend names describe functionality (mla/mha), not
  implementation tech.

## 4. Derive, don't declare

- **R-4.1** **No string-based or attribute-probing dispatch.** No model-name
  matching, no `hasattr`/`getattr`-with-default on config objects, no
  dtype-by-string. Use `isinstance`, enums, and direct attribute access that
  fails loudly. Key behavior off capabilities, not names:

  ```python
  # Bad: if upstream renames the field, the default is returned forever —
  # no error, just silently wrong parallelism. And name matching breaks on
  # every finetune/renamed checkpoint.
  pp_size = getattr(parallel_config, "pipeline_parallel_size", 1)
  use_mrope = "qwen3" in model_config.model.lower()

  # Good: fails loudly on an upstream rename; keyed on the capability.
  assert isinstance(parallel_config, ParallelConfig)
  pp_size = parallel_config.pipeline_parallel_size
  use_mrope = model_config.uses_mrope
  ```

- **R-4.2** **Derive from argument shapes, not object state.** Logic that
  determines a value purely from the shapes/dtypes of weights and scales is
  robust to upstream changes. Question every argument that can be derived
  from another argument:

  ```python
  # Bad: caller must keep block_size consistent with the scale tensor.
  def dequant(w, scale, block_size): ...

  # Good: derived at the point of use; inconsistency is impossible.
  def dequant(w, scale):
      block_size = w.shape[-1] // scale.shape[-1]
  ```

- **R-4.3** **No env reads inside jitted/compiled functions.** Closed-over
  constants are a JAX footgun. Read envs in a thin wrapper and pass values as
  arguments, so one glance at the wrapper shows every env that affects
  behavior:

  ```python
  # Bad: the env is read at trace time and baked into the compiled graph.
  # Changing it later has no effect, and nothing reveals it was captured.
  @jax.jit
  def attention(q, k, v):
      if os.environ.get("USE_FANCY_PATH", "0") == "1":
          ...

  # Good: the un-jitted wrapper reads the env once and passes it in;
  # the jitted function is pure.
  def attention(q, k, v):
      return _attention_jit(q, k, v, use_fancy_path=envs.USE_FANCY_PATH)

  @functools.partial(jax.jit, static_argnames=("use_fancy_path", ))
  def _attention_jit(q, k, v, *, use_fancy_path):
      ...
  ```

- **R-4.4** **No hard-coded hardware constants.** VMEM/SMEM capacity, head-size
  alignment, cores-per-chip, device kind: use runtime queries
(`pltpu.get_tpu_info()`, `get_device_name()`, `get_dtype_packing()`), never
  `120 * 1024 * 1024` or `if "v6e" in name`. Prefer self-explanatory
  constants: `jnp.iinfo(jnp.int16).max`, not `32767`.
- **R-4.5** **No magic mesh-axis strings.** Define and use a shared
  axis-name constant; a hard-coded string breaks silently everywhere when
  the mesh layout changes. The good example in-repo is
  `distributed/ep_mesh.py`:

  ```python
  EP_AXIS_NAME = "d"
  def build_ep_mesh(axis_name: str = EP_AXIS_NAME): ...
  ```

  The anti-pattern is the literal `P("model")` repeated across
  `layers/common/` (e.g. `gdn_attention.py`, `attention_interface.py`) —
  every copy is a place a mesh-layout change can miss. New code uses a
  constant; touched code migrates.
- **R-4.6** **Config flows through vLLM's typed config**, never
  `additional_config` dict-digging wrapped in try/except. Use
  `ParallelConfig.data_parallel_size`, `load_config.load_format`; infer DP
  from the mesh axis size rather than re-reading a sharding dict. If a
  TPU-only field is unavoidable, extend the type rather than stuffing loose
  attributes onto `VllmConfig`.
- **R-4.7** **Typed containers over dicts/bools/sentinels.** Dataclass or
  NamedTuple over dict returns; one enum selector over N boolean flags;
  `int | None` over magic sentinel values (unless the sentinel genuinely
  unifies cases — then prefer the unification); mutually-exclusive bools
  become an enum; no `*args/**kwargs` pass-through in internal APIs; use
  separate classes for shared vs per-layer metadata so wrong access errors
  out instead of silently reading the wrong field.

## 5. Fail fast — no silent failure

- **R-5.1** **No defensive try/except, no bare excepts, no `except: pass`.**
  Catch a specific exception only where the pass is provably safe.
  `gc.collect()` and `atexit` hacks are band-aids indicating something else
  is broken — find it.
- **R-5.2** **Assert untested assumptions, especially in quantization and
  kernels**, where the failure mode is silently wrong numbers rather than a
  crash. An assert that fires is infinitely cheaper than a model that quietly
  produces garbage.
- **R-5.3** **Raise `NotImplementedError` + TODO for incomplete features —
  do not silently do the wrong thing.** Use `NotImplementedError` for
  unsupported options, not generic `RuntimeError`. The discriminator when
  widening a default (e.g. enabling a kernel for more models): an automatic
  fallback is acceptable only when it preserves correctness at a known perf
  cost and logs `warning_once` (R-1.7); when the result would be wrong or
  the config is unsupported, raise an actionable error instead — never fall
  back.
- **R-5.4** **Write error messages that tell the user the fix**: "unaligned
  dequantization requires an explicit block_size argument", not "invalid
  shape".
- **R-5.5** **Do not hide a dtype mismatch with a cast.** Adding
  `v = v.astype(k.dtype)` to make an error go away hides the real bug —
  find and fix the source of the dtype difference first.
- **R-5.6** **Guard recompilation with official APIs**
  (`jax.no_tracing`-style), wrapped so the raised error is actionable —
  never override private JAX internals. **Don't let a kernel-backend switch
  change a compiled function's traced signature** — the switch must error
  out instead.

## 6. Minimize surface area

- **R-6.1** **Question every new argument, flag, and env var.** If there is no
  case where a bool argument is the other value, remove the argument. Prefer
  implicit enablement from the use case over a user-facing knob. Before
  adding a tunable, ask: can a good default or heuristic be derived that
  works in most cases?
- **R-6.2** **Add a new env var/flag only as** one of: (a) a backend selector
  during a migration window — one enum value, not N bools, with its deletion
  condition stated; (b) a debug escape hatch that doesn't change defaults;
  (c) a documented quality/perf tradeoff dial whose exact mode stays
  available. **Never make tuning values (block sizes, thresholds, buffer
  sizes) knobs** — derive them from a cost model or add a tuned-table entry
  (P-16, optimization doc).
- **R-6.3** **Collapse special cases into existing arguments** when
  semantics allow: fold `zero_point` into `bias`; treat MLA as attention
  with `num_kv_heads=1`; decode and spec-decode must not behave differently.
- **R-6.4** **Unify parallelism paths.** One sharding formulation for DP and
  non-DP rather than `dp_size > 1` branches everywhere; share EP/TP override
  logic and vary only the dim size; don't mix `with_sharding_constraint` and
  `shard_map` idioms in one code path.
- **R-6.5** **Generalize from the mechanism, not the observed failure.** Gate
  a fix on the variable that actually drives the behavior
  (`actual_num_kv_heads == packing`), not on the one model/config where the
  bug was seen.
- **R-6.6** **Design for arbitrary checkpoints.** Branch on declared metadata
  (`quant_method` in `config.json`), never "works for the official checkpoint
  only". Checkpoint formats are standardized; modular loaders scale.

## 7. Diff discipline

- **R-7.1** **Small, single-purpose PRs.** Kernel changes get their own PR.
  Reviewers may refuse oversized diffs outright ("oversized" is reviewer
  judgment, not a fixed line count); very large designs need an RFC/design
  doc first. Stacked PRs cite their base explicitly.
- **R-7.2** **Refactor first, feature second.** Land the enabling refactor or
  cleanup as its own PR before the feature/optimization PR that needs it.
- **R-7.3** **Do not include changes unrelated to the PR's purpose.** If a
  hunk is not needed for what the PR is about — reformatting of lines the
  change didn't touch, a rename done in passing, an unrelated typo fix —
  remove it or move it to its own PR. Good-but-off-topic improvements get
  re-landed separately.
- **R-7.4** **Every line must be justified.** The standard probe — for the
  author to preempt and the reviewer to ask — is: *"what happens if this line
  does not exist?"*
- **R-7.5** **No dead code.** Unused dependencies, skipped tests,
  commented-out debug code: delete. Superseded kernel versions are deprecated
  as soon as the successor reaches feature parity, and deleted once no
  caller remains. Every deliberately-kept
  alternate path documents why it exists and when to use it.
- **R-7.6** Update the PR branch by **rebase, never merge** — a merge commit
  can skip the pre-merge CI. WIP goes up as a Draft.
- **R-7.7** **A human reviews every changed line before the PR goes up.**
  However the code was produced, the human submitter must have read the full
  diff, be able to explain and defend each change end-to-end, and have run
  the relevant tests. Never send a diff containing lines nobody can account
  for.
- **R-7.8** **Check for duplicate work before opening a PR.** Search open
  issues and PRs for the same fix or feature first:

  ```bash
  gh issue view <issue_number> --repo vllm-project/vllm-torchtpu --comments
  gh pr list --repo vllm-project/vllm-torchtpu --state open \
      --search "<short area keywords>"
  ```

  If an open PR already addresses it, don't open another; if your approach is
  materially different, explain the difference on the issue first.
- **R-7.9** **No low-value busywork PRs.** Don't open one-off PRs for a
  single typo, an isolated style fix, or one mutable default. A mechanical
  cleanup lands only when it belongs to the same purpose as substantive work
  in the same PR (R-7.3); a cleanup that is neither same-purpose nor
  substantive enough to stand alone is simply not done — drop it (R-7.10).
  This does not ban standalone refactor PRs: an enabling refactor per R-7.2,
  or a cleanup PR that meaningfully reduces duplication or dead code, is
  substantive work, not busywork.
- **R-7.10** **Fail closed.** If the work turns out to be a duplicate,
  trivial busywork, or blocked by a gated-area conflict (R-2.7), stop and
  report what is missing instead of opening the PR anyway. This applies
  doubly to coding agents: returning a short explanation is the correct
  outcome, not a failure.
- **R-7.11** **PR descriptions name the mechanism, not the trigger.** The
  title states the general capability unlocked ("support fp8
  compressed-tensors MoE"), not the model that prompted it; the body is a
  terse bullet list of mechanisms; fix PRs state the original bug and the
  fix mechanism; perf PRs carry per-mode speedups and every PR carries an
  honest Limitations section when limitations exist (PR-1 and PR-4 in
  the optimization doc).
- **R-7.12** **No internal-only references anywhere that lands in the public
  repo** — code, comments, yml, scripts, commit messages, PR text alike: no
  `go/` links, no `b/` bug numbers in comments (public issue trackers are
  fine), no personal GCS buckets in scripts or CI, no links to private
  notes. When code needs to cite the reasoning behind a design, inline the
  reasoning in the comment or cite a public source (vLLM source path, HF
  model card, published paper).

## 8. Tests

- **R-8.1** **Every production change under `src/vllm_torchtpu/` needs a unit
  test that exercises the real behavior** — the forward pass, not just weight
  processing; error paths too; real initialization over `mock.patch` where
  feasible. A test that merely imports the module doesn't count.
- **R-8.2** **A regression test must fail without the fix.** If reverting the
  fix doesn't turn the test red, the test isn't testing the fix.
- **R-8.3** **Parametrize, don't clone.** Factor setup/inference/assertion
  helpers out of giant tests; verify parametrization actually flows into the
  code under test (a decorator param that is silently ignored still passes).
- **R-8.4** **Verify tests are wired into CI.** Green CI ≠ code exercised —
  check the pipeline yml ignore lists and markers (`nightly`, `multichip`).
  Mark tests correctly: a mock-only test must not require a multichip agent.
- **R-8.5** **Tests must run on the smallest hardware that exercises the
  behavior.** Coverage that only exists on large/pod configurations doesn't
  count as coverage.
- **R-8.6** Accuracy tests are generic harnesses: model, config, and threshold
  are arguments, not hard-coded per model.
- **R-8.7** For MagicMock-heavy tests: explicitly set every attribute the code
  under test reads. MagicMock auto-creates missing attributes, and
  arithmetic/iteration over a Mock fails silently (empty iterators, more
  Mocks) instead of raising.
- **R-8.8** **Design the test before writing it.** Answer four questions
  first: what is the module for, what is its I/O contract, what failure am I
  guarding against, and what is the cheapest level that catches it (unit
  over integration over e2e)? Then: extend existing test files, fixtures,
  and helpers before creating a new file; assert observable behavior through
  public APIs and state the intent in the test name or docstring; one
  behavior per test with the smallest setup that triggers it. If the test
  diff dwarfs the code change, cut scope. Flaky tests are worse than no
  tests.
- **R-8.9** **No one-off kernel benchmarks in `tests/`.** Correctness belongs
  in pytest; perf measurement belongs in the benchmark harness. A
  timing-based assert in a unit test is a flake generator.
- **R-8.10** **Run evals proactively for model-affecting changes.** Any
  change that can affect output, accuracy, or serving behavior ships with
  eval/benchmark results in the PR description — don't wait for a reviewer
  to ask. (Quantization changes follow the stricter Q-3 standard in the
  optimization doc.)

---

## Reference / further reading

Background material for the rules above.

- [vLLM plugin system](https://docs.vllm.ai/en/latest/design/plugin_system/):
  the entry points and registration hooks §2 and R-3.5 build on.
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html):
  the style baseline for R-1.1.
- [optimization_best_practices.md](optimization_best_practices.md): PR
  discipline (PR-x), performance (P-x), and quantization (Q-x) rules, and its
  own reference list for TPU hardware and kernel material.

## Author checklist (run before sending a PR)

1. **Scope** — duplicate-work check done, not busywork (R-7.8, R-7.9);
   single purpose; kernel changes separated; refactor pre-landed; stacked
   base cited; no drive-by reformatting. (R-7.x)
2. **Reuse & placement** — nothing reimplements a vLLM / vllm-torchtpu /
   Pallas utility; code sits in the right directory per R-2.1; interface
   files thin; kernels dependency-free; no model-specific logic in shared
   files; dead and superseded code deleted. (R-2.x, R-3.x)
3. **Description** — title names the general capability, not the triggering
   model; mechanism bullet list; fix PRs state the bug and the fix
   mechanism; per-mode speedups for perf PRs; numeric + performance
   validation for quant PRs; honest Limitations section. (R-7.11; Q-3 in
   the optimization doc)
4. **Evidence** — xprof trace for hot-path claims; before/after eval on the
   same rig for numerics; eval results attached for any model-affecting
   change (R-8.10); HLO diff for "no compiler impact" claims; before/after
   HBM logs for weight-loading changes. (Optimization doc: P-1–P-4,
   Q-3, Q-6)
5. **No new knobs** unless one of R-6.2's three cases applies; tuning values
   are tabled or derived, never knobs.
6. **Derived values** — nothing hard-coded that `pltpu.get_tpu_info()`,
   shapes, axis-name constants, or config metadata can provide. (R-4.x)
7. **Fail-fast** — asserts on untested configs; `NotImplementedError` + TODO
   for gaps; actionable errors; no bare excepts. (R-5.x)
8. **Tests** — exercise the forward pass; fail without the fix;
   parameterized; wired into CI with correct markers; runnable on small
   hardware. (R-8.x)
9. **Thresholds** — CI perf thresholds updated in the same PR if performance
   moved. (PR-1, optimization doc)
10. **Hygiene** — `pre-commit run --all-files` clean; 80 columns; license
    headers; explicit imports; no internal links. (R-1.x, R-7.12)
11. **Lifecycle** — WIP as Draft; branch updated by rebase, never merge;
    nothing merges until CI is green on the current head; every changed line
    read and defensible by the human submitter (R-7.7).

---

## Appendix — rationale for "Derive, don't declare"

Every **declared** copy of a derivable value — a config knob, env var, magic
number, or cached member variable — creates an invariant that must be
maintained by hand: the declared value has to stay consistent with the
tensor shape, checkpoint layout, hardware, or mesh it mirrors. Nothing
enforces that consistency; humans and agents do, on every future change.
Derived values don't have this failure mode:

- **Upstream churn.** This repo sits on a fast-moving vLLM. When upstream
  changes a shape, a field, or a layout, code that derives the value at the
  point of use keeps working; a declared copy goes stale with no error.
- **Staleness is silent.** In an inference stack, a stale declared value
  usually doesn't crash — it produces silently wrong numerics or a perf
  cliff, the most expensive class of bug to detect (the same failure mode
  R-5.2 guards against).
- **Surface area.** Every knob must be documented, tested across its
  settings, and correctly set by every user; a derived value costs none of
  that (R-6.1).
- **Reviewability.** A derivation shows *why* the value is what it is; a
  literal `128` shows nothing, and the reviewer must reconstruct the
  reasoning to check it.

The practical test, for authors and reviewers alike: for any constant,
argument, or config field, ask *"can this be computed from something already
in scope — a shape, a dtype, `pltpu.get_tpu_info()`, the mesh, the vLLM
config?"* If yes, compute it there (R-4.2–R-4.7). If it genuinely cannot be
derived, prefer a tuned-table entry with a stated derivation over a
user-facing knob (P-15 and P-16 in the optimization doc).

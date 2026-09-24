# vllm-torchtpu Optimization Work Best Practices

Companion to [coding_guidance.md](coding_guidance.md), which carries the general
coding rules (R-1.x–R-8.x) and the author checklist. This document carries
the PR-discipline (§0), performance (§1), and quantization (§2) standards,
plus a reference reading list (§3).
Rules here are numbered **PR-\<n\>** (PR discipline), **P-\<n\>**
(performance), and **Q-\<n\>** (quantization); those IDs cited anywhere in
the document set refer to the rules below, and R-x.y cited below refers to
the coding guidance.

---

## 0. PR discipline

Applies to all optimization work — performance (§1) and quantization
(§2) alike.

- **PR-1** Optimization PRs are tiny and mechanism-described; perf PRs carry
  per-mode speedups (decode/prefill/mixed) in the description. Update CI
  perf thresholds in the same PR that moves performance.
- **PR-2** **Revert fast on numeric issues** — a revert PR is a first-class
  mitigation; root-cause offline.
- **PR-3** **When copying or porting code from another repo, state where it
  came from and what you changed** (source repo/file, in the code comment
  and PR description) — so the reviewer can diff against the original
  instead of reviewing from scratch, and maintainers know where upstream
  fixes land. Version kernels (v2, v3), then deprecate and delete the
  superseded one on R-7.5's timeline. A kernel that exists in two homes
  (internal + OSS copies) is updated in both on every change.
- **PR-4** PR descriptions carry an honest **Limitations** section:
  unsupported shapes/configs, known perf gaps, follow-up TODOs.

## 1. Performance work

### 1.1 Evidence standard: traces, not guesses

- **P-1** **xprof traces are the unit of evidence.** E2e numbers alone prove
  nothing; a perf claim needs the trace plus a mechanism explanation of where
  the gain comes from. Anything claimed to be a no-op on the hot path must be
  proven with a trace under realistic (large-batch) conditions — padding and
  repeat overhead is invisible at small batch.
- **P-2** Prove compiler impact from the compiled artifact (HLO dumps,
  graph viewer) — no guessing about layouts and fusions.
- **P-3** Quantify tradeoffs before merging: scheduler changes report
  throughput *and* latency; extra rounding ops are measured for the
  quality/perf tradeoff.
- **P-4** Parallelism A/Bs compare at matched **global** token budget (DP
  with per-rank budget × N ranks vs TP at the same total). Interrogate any
  win the mechanism doesn't predict — it's usually a batch-size artifact.

### 1.2 Let the compiler work

- **P-5** Shard via jit signatures (`in_shardings`/`out_shardings`) instead
  of manual reshard functions; use the global mesh context instead of
  threading `mesh` through every signature.
- **P-6** Cast to bf16 *before* collectives when numerics tolerate it — and
  verify like any dtype change.

### 1.3 Reason at the hardware-resource level

Argue kernel decisions in terms of VREG / VMEM / SMEM / HBM bandwidth / MXU
dtype throughput / DMA latency. The rules below are the points that come up
most often in review; for the underlying hardware model, see §3.

- **P-7** VREG lifetime: quantize per-loop-iteration to avoid spills; don't
  materialize zero-init accumulators up front; write results directly to
  `out_ref` on the last step instead of scratch-then-copy.
- **P-8** HBM traffic: avoid unnecessary zero-initialization; fuse ops to
  avoid HBM round trips.
- **P-9** MXU throughput: w8a16 beats w16a16 — never silently upcast
  weights or activations to 16-bit compute; ride 4-bit weights on the fp8
  path (u4→fp8, not u4→bf16); unpack packed formats with shifts/masks, not
  dtype up/down-cast chains.
- **P-10** SMEM: don't allocate the maximum — the compiler needs headroom.
- **P-11** DMA: small DMAs are latency-bound — fewer hops beats equal total
  bytes; compute the split crossover from the cost model. Never
  `start(); wait()` back-to-back — pipeline with buffering.
- **P-12** Layout is a first-class cost: fix interleaved splits by
  preprocessing weights at load time; sublane `jnp.repeat` inside a kernel is
  not cheap.
- **P-13** Optimization barriers hide in innocuous constructs:
  `jax.named_scope`, `lax.cond` in the hot loop (a `lax.cond` in the main
  kernel loop regresses *all* workloads). Don't allocate never-read memory
  just to keep a signature fixed.

### 1.4 Additional considerations

- **P-14** **Weight-loading time is where heavy tensor work belongs**:
  dequantization, scale broadcasting, KV-head replication, weight transposes
  normalized once by the kernel contract so no per-backend branching survives
  into the forward pass. Call-time `jnp.pad` on the inference path needs
  trace proof or a redesign — the runner pads shapes upstream; kernels don't
  carry unaligned-shape complexity. Memory validation for any weight-loading
  change — quantized or not — follows Q-6.
- **P-15** **Every threshold should be computable from a cost model** (DMA
  latency vs bandwidth, VMEM budget per TPU generation). Heuristics need
  theoretical backing with a code pointer to the derivation.
- **P-16** **Block sizes live in tuned tables inside the kernel directory**
  (`kernels/*/tuned_block_sizes.py`), keyed by workload, imported as normal
  Python data — never JSON resolved via `__file__`-relative paths. Misses log
  `warning_once` with the full lookup key; new workloads add table entries,
  not env overrides.
- **P-17** Know each kernel's hard envelope (mesh rank, EP-only, dtype
  packing, JAX version range) and gate callers on it explicitly.

## 2. Quantization

- **Q-1** **Build generic machinery, not per-format forks.** A format is a
  point in `(block_size, weight_dtype, scale_dtype)` space — e.g. mxfp4 is
  subchannel quantization with subchannel size 32, weight dtype fp4, scale
  dtype e8m0. Write one generic subchannel implementation with those as
  arguments. Layer the design: shared linear/MoE weight processing, with
  per-checkpoint-format pre-processing before the shared call.
- **Q-2** Recipe for a new quant method: config class → linear method
  (`NotImplementedError` for MoE initially) → `process_weights_after_loading`
  reshaping to TPU-friendly shapes → `apply` dispatching to the kernel →
  unit test that loads and runs a real model. Home:
  `layers/vllm/quantization/`, subclassing the upstream vLLM method (R-3.1).
- **Q-3** **Numeric verification is non-negotiable**: before/after eval
  (e.g. MMLU/gsm8k) plus serving benchmark on the identical rig, with exact
  repro commands in the PR, in separate "Numeric validation" and
  "Performance validation" sections. Layer-level standard: build the real
  vLLM layers, run random input through them, confirm the custom path
  triggers and numerics are within quantization error.
- **Q-4** Requantization is allowed when proven quality-neutral —
  checkpoint numerics aren't sacred; requantize for speed *after* verifying
  no quality drop. Deliberate approximations are quantified and exposed as an
  option with the exact mode still available (this is R-6.2's case (c)).
- **Q-5** **Track scale tensors independently of the weights they scale** —
  a weight transpose does not imply a scale transpose. Assert on untested
  granularities (odd block sizes, asymmetric 2D blocks, non-divisible dims).
- **Q-6** **Memory validation**: any weight-loading or model-level change
  compares server HBM logs (`total_hbm_used_gb`) before vs after — resident
  HBM growth directly shrinks the KV cache. Validate on the largest
  checkpoint that fits the rig (a small proxy under-stresses HBM), across
  TP/EP/DP, and on a shape whose sharded last dim is NOT 128-divisible.
  Report before/after load times.
- **Q-7** Dtype/scale choices are hardware-driven: state per-generation
  guidance explicitly; use the platform's `fp8_dtype` interface to declare
  hardware preference; emulate missing hardware support at load time; use
  TPU-native stochastic rounding when requantizing at load; never assume
  fp16→bf16 conversion is numerically free — verify.

## 3. Reference / further reading

Background material for the rules above. Read §1 and §2 as the standard this
repo enforces; read these to understand the hardware and tooling they rest on.

- [How to Scale Your Model — TPUs](https://jax-ml.github.io/scaling-book/tpus/)
  (JAX scaling book): the TPU hardware model — MXU, VREG/VMEM/HBM hierarchy,
  bandwidths, roofline arithmetic — behind §1.3, plus the chapters on
  sharding and collectives behind §1.2 and P-4.
- [Pallas TPU documentation](https://docs.jax.dev/en/latest/pallas/tpu/index.html):
  the kernel programming model — grid/BlockSpec, memory spaces, DMA — behind
  P-7 through P-13.
- [XProf / TensorBoard profiler guide](https://openxla.org/xprof):
  how to capture and read the traces P-1 requires.

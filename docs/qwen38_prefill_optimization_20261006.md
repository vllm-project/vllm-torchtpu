# Qwen3.8 Flash prefill optimization checkpoint

## 2026-10-08 review and implementation update

The Qwen branch has been rebased onto main `b94bc6d4f` (including PR #1641).
The measurements below belong to the October 6 checkpoint, before the following
cleanup. They are not performance or quality validation of the current code.

- QSA prefill keeps block-sparse A as its single implementation. Compact B,
  its frozen kernel copies, dense/gather/staged diagnostic implementations,
  and the three backend/threshold configuration keys have been removed.
  Retired keys produce an explicit configuration error.
- Qwen4Exp enables the unified pool by default through its architecture
  registration. The existing global pool override is still honored. Small
  token steps continue to use the existing batched RPA decode path; this is
  selected from static batch geometry, with no prefill backend switch.
- PR #1641's blocked scorer now reads both native BF16 index keys and BF16
  bytes stored in an 8-bit pool. It forwards the QSA causal offset to the
  top-k row limits, keeping incomplete compression groups out of selection.
  BF16 query head reduction continues to accumulate in FP32.
- Native-cache selection chunks query rows before expanding request page
  tables. The chunk size follows the chip's SMEM capacity. Ragged pooled
  prefill retains live requests after interior empty request rows.
- A reads the actual local KV head count and preserves the scale of unpadded
  head dimensions. Default query tiles shrink as local head counts grow.
  The measured Q24/KV2/D256 paired-page tile remains the production choice
  for its matching geometry.

### Scope relative to main

The reviewed source tree has 31 Qwen-specific files, including the early model
communication query, and 20 shared files with 464 added / 126 removed lines.
No other model implementation is edited; model registration is shared.

| Shared area | Why retained | Scope boundary |
| --- | --- | --- |
| Worker/platform/runner DP predicates | Resident PLE sharded over DP needs a shared communication world and matching busy/idle forwards even with EP disabled | Activated by EP or Qwen `embedding_across_dp`; ordinary DP remains independent |
| Runner/cache adapter hooks | Pass model-owned inputs and cache lifecycle operations | Selection returns no adapter for other architectures; Qwen logic lives under its model directory |
| GDN conv-state dtype plumbing | Qwen requires FP32 conv history while retaining BF16 recurrent state | Shared API defaults stay BF16; Qwen passes FP32 explicitly; unused PCP extensions removed |
| StreamIndex and batched RPA | BF16 storage, completed-group offsets, bounded chunk metadata, and a decode-only schedule are reusable kernel capabilities | Default StreamIndex offset and RPA scheduling remain unchanged |
| FP8 MoE loading and dense expert tails | TP shards smaller than a checkpoint quantization block need lossless scale refinement; 128-aligned intermediate tails need correct buffers | General shape/loader constraints; no model-name tests in these kernels |
| MoEv2 metadata/gather and GDN input conversion | Large token tables exceed scalar memory; layout conversions materialize avoidable HBM buffers | General memory/layout issues, with shared-kernel device regression still required |
| Compilation configuration/hash | Preserve explicit graph boundaries, disable a CUDA-only fusion, and distinguish cache geometry/APC modes | Platform correctness and cache invalidation |

Qwen-only gated residual code moved into `kernels/qwen4_exp`; its Torch bridge,
along with QSA/PLE/HC bridges, lives in that directory's `vllm_adapter.py`.
The retired prompt-length routing field and its runner-side copy were removed.

### Validation of the cleanup

Local checks use Python 3.12 and the repository's JAX/JAXLIB 0.11.2 pins.
Isolated CPU tests load the actual JAX kernel modules while bypassing package
startup that requires TorchTPU/vLLM. They check causal limits through 256K
context, bounded page-table expansion, ragged ownership, TP head shapes and
512/8192-row StreamIndex tracing with JAX's abstract TPU mesh. Abstract tracing
does not compile or execute TPU machine code.

The isolated run reports 40 passed, 46 TPU-gated cases skipped and five
Torch-dependent page-addressing cases deselected.

Pinned Ruff format/check, lightweight Pyrefly, environment-read and kernel
closure baseline checks pass. The full dependency typecheck could not access
the local container API. Runner/model tests need the missing TorchTPU/vLLM
environment; five native page-addressing tests also transitively need Torch.

Persistent TPU tests now cover blocked BF16/carrier score and selection parity,
and Qwen's 16 KQ / 48 V-head GDN with BF16 SSM and FP32 conv state across a
4K/8K prefill followed by three decode steps. These tests have not run on a
device. Current-code whole-model mixed batches, APC cold/hot ownership,
32K/128K/256K continuation, output quality and throughput remain unvalidated.

## Historical checkpoint

This checkpoint preserves exact QSA selection and cache semantics. It improves
short-context prefill and gives a modest long-context gain. It does not reach
130,000 input tokens/s or eliminate the long-context throughput drop. Further
optimization is paused at the user's request.

## Changes and rationale

| Area | Change | Purpose |
| --- | --- | --- |
| HC | Stream full mixer tiles and fuse gated residual materialization | Reduce intermediate HBM traffic while retaining FP32 gates and BF16 boundaries |
| GDN | Cast BF16 QKV directly into the compact FP32 Conv1D input layout | Avoid a separate FP32 HBM buffer and its layout conversion |
| MoEv2 | Stream routing metadata through bounded SMEM windows | Support large prefill token counts without oversized scalar-prefetch tables |
| QSA A | Use tuned Q256/KV256 sparse tiles and bounded metadata batches | Reuse selected KV across query rows and bound metadata scratch |
| QSA B | Form exact Q16 unions and membership masks; DMA original FP8 microblocks directly to TC | Reduce inactive tile work and avoid SC-to-HBM-to-TC staging |
| QSA B metadata | Use fixed row offsets and per-row page lookup | Remove owner/searchsorted and group indirection without changing consumed addresses |
| Indexer | Limit metadata construction to the scoring chunk | Reduce metadata work while preserving full-history scoring and exact top-k |
| Unified pool | Build Mamba copy plans from materialized layer mappings | Preserve state ownership and aliases when prefill cache pages are copied |

GDN split/compact-BF16 variants and MoE M256 experiments are excluded. No
MoE load-balancing or routing changes are included. FP8 weights/KV, BF16 SSM
and shared DRAM PLE remain the benchmark recipe.

## Historical QSA execution policy

Auto prefill uses sparse attention. Dense and per-query gather do not enter
automatic selection; explicit diagnostic dense/RPA modes remain available.
Dense was slower in the measured short-context cases. Per-query gather did
not establish a reliable context-only crossover on actual selector geometry.

The two sparse execution paths are:

- A: Q256 block-sparse attention, including the full-grid fallback.
- B: Q16 exact union/membership attention with direct FP8 microblock DMA.

B is restricted to the tested TPU v7 BF16 Q24/KV2/D256, FP8 cache, ratio-4,
physical-page-1664 envelope. Dynamic eligibility requires more than 512 active
KV256 tiles, no Q16 union above 4,096 microgroups, and at least 5% reduction
in the hybrid rounded transport-tile count. Otherwise A executes. Overflow
payloads are never consumed.

Context at or below 128K uses A. Above 128K is only eligibility for B:
selection geometry still determines whether B executes. This is a measured
experimental routing policy, not a universal context-length cutoff.

## Whole-model results

All long-context runs use DP8/TP1/EP8, FP8 weights and FP8 E4M3 KV, BF16 SSM,
FusedMoEv2 M128, shared DRAM PLE, chunk size 8,192 per rank, eight sequence
slots per rank, global concurrency 64, maximum model length 262,144, seed 42,
async scheduling and no prefix caching. Each request produces one token.
Each arm has two formal repetitions of 128 requests per context, plus separate
warmup and fixed-input greedy checks. Pooled TPS is total input tokens divided
by total elapsed time, excluding warmup.

The earlier QSA combination trial kept MoE/GDN/HC source-identical between arms:

| Input tokens | Control TPS | QSA combination TPS | Change |
| ---: | ---: | ---: | ---: |
| 32,768 | 89,245 | 88,931 | -0.35% |
| 65,536 | 74,891 | 74,794 | -0.13% |
| 131,072 | 59,076 | 59,091 | +0.03% |
| 262,143 | 43,853 | 45,101 | +2.85% |

The final trial isolates the fixed-stride metadata change against the already
optimized combination:

| Input tokens | Optimized control TPS | Fixed-stride TPS | Change |
| ---: | ---: | ---: | ---: |
| 32,768 | 89,097 | 88,717 | -0.43% |
| 65,536 | 74,891 | 74,696 | -0.26% |
| 131,072 | 59,051 | 59,012 | -0.07% |
| 262,143 | 45,086 | 45,553 | +1.04% |

These are separate control-first experiments, not independent randomized
samples; do not add their percentage gains or claim statistical significance.
The 32K-to-256K throughput drop remains about 49% in the final candidate.

An earlier short-context package trial also includes HC changes and uses
maximum model length 16,385. Its gains are package gains, not QSA attribution:

| Input tokens | Control TPS | Candidate TPS | Change |
| ---: | ---: | ---: | ---: |
| 1,024 | 48,052 | 52,701 | +9.68% |
| 2,048 | 80,390 | 87,423 | +8.75% |
| 4,096 | 92,801 | 102,184 | +10.11% |
| 8,192 | 114,855 | 116,600 | +1.52% |
| 16,384 | 104,322 | 108,691 | +4.19% |

The 8K result remains below the earlier historical 117,066 TPS measurement.
Different maximum lengths and run dates prevent treating these short/long
numbers as one matched context sweep. The L20A figures use a different rig
and recipe and are comparison targets, not matched A/B controls.

## Validation and limits

Both long-model trials independently audit all 2,688 successful requests,
raw SSE token/usage records, input lengths, first-token timestamps, TTFT
statistics, rank distribution, source/config hashes and process retirement.
The 64 fixed-input greedy comparisons cover two input families, four context
lengths and all eight ranks. They validate one output token, not task accuracy.

Both runtime arms have eight workers sharing one PLE mapping inode per arm,
with 51,200,245,760 mapped bytes per worker. FP8 model configuration and KV,
BF16 SSM and actual parent MoEv2 settings are checked. Worker `/proc/environ`
is clobbered by setproctitle, so it does not directly prove Python worker
environment contents.

Fixed-stride native gates cover empty/mixed unions, the 4,096-group boundary,
overflow, page remapping, exact consumed addresses, output bytes and unchanged
cache. Captured-selector fixtures repeat 256 rows to 512 and use synthetic
Q/KV; their 7.8%-9.8% layer speedups are not full-8K geometry or model TPS.

Late-256K native profiles identify QSA core and wrapper/metadata work as large
remaining costs. Interval unions are not additive critical-path percentages.
Full-history indexer scoring still grows with context; the isolated metadata
change gives less than 0.35% model benefit. FP8 cache does not by itself prove
FP8 MXU operands, measured HBM utilization or optimal lowering.

Regression tests in this checkpoint cover large MoEv2 combine tables, GDN
input values, gated-residual gate precision, QSA page remapping/membership,
and unified-pool state-copy ownership. All 75 focused TPU regression tests pass at submission time. Whole-project pyrefly has existing errors; the checkpoint
introduces none and does not expand the baseline.

The benchmark processes are retired and the prior installed package is
restored. Publishing this source checkpoint does not install the candidate
or restart a model server.

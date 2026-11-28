# VMEM-resident MoE expert weights (experiment)

Question: if one layer's expert weights stay in VMEM for every decode step,
instead of being read from HBM each step, how much faster is a step? The
target is TPU 8i (SunFish, 192 MiB VMEM per TensorCore) with Gemma 4 26B-A4B
expert shapes. v7x (GhostFish, 64 MiB) is the proxy that validates the kernel
and the method.

This is an experiment, not a serving path. Nothing outside this directory
imports it.

## What "resident" can mean on TPU today

VMEM is per-call scratch. jax has no mechanism that keeps VMEM contents across
`pallas_call` invocations (jax `1a1e875`, `jax/_src/pallas/mosaic/`; no test
covers it either way). Scratch persists across grid steps inside one call
(`jax/experimental/pallas/ops/tpu/matmul.py:28-40`,
`paged_attention/paged_attention_kernel.py:234-265`). So the only way to keep
weights resident for N decode steps is one call that runs all N steps, which
is what this kernel does.

Not tried here: XLA-placed VMEM arrays (`pltpu.with_memory_space_constraint`,
VMEM-coloured `out_shape`, `tests/pallas/tpu_pallas_memory_space_test.py`),
which might keep a weight in VMEM across separate calls in one jitted program.
No upstream test chains two calls through such a buffer.

## Design

`kernel.py` has two arms with the same per-item body over a flattened
`(step, expert)` work list (only experts some token routed to that step):

| arm | expert weights | per item |
| --- | --- | --- |
| `resident` | whole-array VMEM inputs, copied in once | index `w[e]` in VMEM |
| `streamed` | HBM (`pl.ANY`) | DMA `w[e]` into a 2-slot VMEM ring, prefetch item `t+1` |

* Expert: SwiGLU, `fp8_e4m3fn` weights, per-token dynamic `fp8_e4m3fn`
  activations, fp32 accumulate, per-channel fp32 scales.
* Step `s+1` consumes the RMS-normed output of step `s`, so steps are serial.
* Routing is synthetic: each real token picks one local expert per step
  (EP-sharded layer, one TensorCore's shard). Token rows are padded to
  `--rows`.
* Per-step time is the slope between two step counts (`--steps lo,hi`),
  which removes launch cost and the resident arm's one-time weight load.

Correctness: both arms are checked against a plain-JAX oracle with the same
quantization (`reference`). A corrupted expert must make the oracle disagree
(`FAULT`), so a green check means something.
`tests/kernels/experimental/test_vmem_resident_moe.py` runs both on the CPU
Pallas interpreter.

## Modelled ceiling (not measured)

One expert is `3 * 2816 * 704` fp8 bytes = 5.95 MB. Even resident, the MXU
must load (`vmatpush`) the weights each matmul. Its rate is 2 MXUs × one
256×256 tile per push. From the GhostFish/SunFish/ZebraFish spec tables, that
is 64 cycles at 2.2 GHz on v7x and 32 cycles at 2.4 GHz on TPU 8i.

| per expert, one TensorCore | v7x | TPU 8i |
| --- | --- | --- |
| HBM read (3.69 / 4.30 TB/s) | 1.61 µs | 1.38 µs |
| MXU weight push (4.5 / 9.8 TB/s) | 1.32 µs | 0.60 µs |
| VMEM read | 0.44 µs | 0.40 µs |
| **streamed / resident** | **1.22×** | **2.29×** |

Decode stays push-bound up to ~220-256 tokens per expert. On v7x, the push
rate is barely above HBM, so residency can buy at most ~1.2×. TPU 8i pushes
twice as fast, which is where residency pays.

## Results

### v7x, one TensorCore (job `j-b92e1ec1-959a-4139-b7ee`)

jax 0.11.2, libtpu 0.0.47, Gemma 4 26B-A4B expert shapes (2816 × 704, fp8),
64 padded rows, slope over 4 → 16 steps, median of 30 reps. µs per step is
for the whole step. µs per item is per (step, expert) pair, i.e. per expert
actually touched.

| local experts | B | resident µs/step | streamed µs/step | resident µs/item | streamed µs/item | streamed / resident |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 1 | 2.45 | 2.48 | 2.45 | 2.48 | 1.01× |
| 8 | 4 | 7.51 | 6.06 | 2.31 | 1.87 | 0.81× |
| 8 | 16 | 12.83 | 16.24 | 1.77 | 2.24 | **1.27×** |
| 8 | 64 | 14.43 | 17.91 | 1.82 | 2.26 | **1.24×** |
| 4 | 1 | 3.29 | 2.91 | 3.29 | 2.91 | 0.88× |
| 4 | 16 | 7.55 | 10.51 | 1.89 | 2.63 | **1.39×** |
| 4 | 64 | 8.31 | 9.68 | 2.08 | 2.42 | **1.16×** |

* **B ≥ 16: resident is 1.16–1.39× faster (1.24–1.27× at 8 experts).** That
  matches the modelled 1.22× ceiling. Neither arm is at its floor. Resident
  runs at ~1.8 µs per expert against the 1.32 µs push floor, and streamed at
  ~2.25 µs against the 1.61 µs HBM floor. Both sit at ~72-73% of their floors.
* **B ≤ 4: inconclusive.** A step there is 2.5-7.5 µs on top of a ~230 µs
  fixed cost (dispatch plus the one-time load). The slope of a 12-step
  difference resolves about ±0.5 µs per step, which is ±10-20% here. The
  B = 4 inversion (0.81×) is inside that band and has not been re-run.
* Measured with host wall clock, not xprof.

Correctness on TPU: both arms give identical output, max abs error 0.234
against the oracle (|ref| ≤ 4.18). An injected corrupted expert gives 5.91, so
the check discriminates. On the CPU interpreter the error is 0.0. The
residual on TPU is between the kernel and the XLA-compiled oracle. Most
likely a few fp8 roundings flip and RMS norm carries them through 4 serial
steps, but this has not been isolated. Do not read 0.234 as a tolerance.

### Arm 3: one `pallas_call` per step (`percall.py`, job `j-93ed3f76-ee90-4dba-bb49`)

v7x, one TensorCore, 4 local experts, µs per step. Each row is one decode
loop jitted as a whole.

| arm | Mosaic scratch limit | B = 16 | B = 64 | weights `S(1)` (VMEM) in the while body |
| --- | --- | --- | --- | --- |
| in-kernel resident (one call) | 62 MiB | 8.18 | 8.64 | by construction |
| in-kernel streamed | 62 MiB | 9.78 | 9.68 | - |
| `percall_vmem` | 62 MiB | 19.86 | 19.34 | not visible (annotation truncated) |
| `percall_vmem` | 20 MiB | 8.53 | 10.52 | yes |
| `percall_constrained` | 20 MiB | 9.52 | 9.46 | yes |
| `percall_streamed` | 20 MiB | 11.79 | 12.65 | yes (XLA placed the `pl.ANY` weights in VMEM) |

* With VMEM headroom, XLA's memory-space assignment keeps loop-invariant
  weights in VMEM across separate calls without being asked. One call per
  step then runs at 8.5 µs, against 8.2 µs for the single-call resident
  kernel.
* With the Mosaic scratch limit at 62 MiB, the per-call arm is 2.4× slower,
  which suggests a reload per call. That is inferred from time: the HLO scan
  found no copy op in any loop. `with_memory_space_constraint` then fails at
  compile time ("Too many buffers are colored in the alternate memory",
  `j-e116da82`).
* Kernel scratch and XLA-placed weights share one VMEM. On v7x, 8 experts
  (~48 MiB) leave no room for both, so that case was not run.
* This only holds within one jitted program. Nothing here shows VMEM
  surviving across program executions, which is how vLLM steps today.

### Stage L1: EP-sharded MoE block on 8 TensorCores (`layer.py`, job `j-3ba6ace4-bded-46fe-8f9d`)

v7x 2x2 (8 TensorCores), 64 experts (8 per TensorCore), top-8, tokens
replicated, one in-kernel recursive-doubling all-reduce per step, 64 padded
f32 rows on the wire. µs per step:

| B | resident + AR | streamed + AR | resident, no AR | streamed, no AR | AR cost |
| --- | --- | --- | --- | --- | --- |
| 1 | 38.6 | 43.7 | 3.3 | 2.6 | 35-41 |
| 16 | 43.5 | 54.2 | 10.1 | 15.8 | 33-38 |
| 64 | 50.9 | 55.0 | 12.1 | 16.7 | 38-39 |

* All-reduce arms match the single-device oracle on all 8 devices (7.2e-7).
* Without the all-reduce, residency is 1.39-1.55× at B ≥ 16 with real top-8
  routing. The all-reduce cost ~35 µs, against the 6 µs assumed in the
  whole-layer model, and absorbed most of the gain. That run sent 721 KB per
  round even at B = 1. 
L1 v2 (job `j-4e6685a1-6d7b-4858-8c44`). The wire is sized to the batch (rows =
B rounded up to 16). The bf16 wire rounds both operands, so replicas stay
identical. µs per step:

| wire | B | resident + AR | streamed + AR | streamed / resident | AR cost |
| --- | --- | --- | --- | --- | --- |
| f32 | 1 | 18.7 | 15.9 | 0.85× | 15-16 |
| f32 | 16 | 23.4 | 28.3 | 1.21× | 14-15 |
| f32 | 64 | 50.3 | 54.2 | 1.08× | 37 |
| bf16 | 1 | 10.3 | 13.9 | 1.36× | 9-12 |
| bf16 | 16 | 17.7 | 24.7 | 1.40× | 8-9 |
| bf16 | 64 | 35.2 | 38.4 | 1.09× | 16-25 |

* Sizing the wire to the batch cut the all-reduce from ~35 to ~15 µs at
  B ≤ 16. A bf16 wire brings it to ~9 µs. The residency gain on the full MoE
  block is then 1.36-1.40× at B ≤ 16.
* The all-reduce is still the bottleneck. Its cost fits ~7.7 µs fixed plus
  ~41 ns per KB on the wire. That is ~25 GB/s across 3 serial full-buffer
  rounds, far below ICI link rates. A reduce-scatter + all-gather (half the
  bytes) and overlapping rounds are the next levers.
* bf16 wire numerics: max abs error 0.585 against the f32 oracle over 4 steps
  (|ref| ≈ 4). It is a precision trade, not a free win. The experiment owner
  accepted it (2026-11-19), so `layer_bench.py` now defaults to
  `--ar-dtype bfloat16`. `float32` stays available as the exact arm.
* The bf16 check error is not a model-quality number. It compounds with the
  number of serial steps in this synthetic recurrence: random weights,
  RMS-normed output fed straight back through fp8 quantization, no residual.
  It is 0.58 after 4 steps and ~2.9 after 8 (bisect job
  `j-0ec45276-ea87-433f-af5c`, every rows / wire / quantize-once variant).
  The f32 wire stays at 1e-6 at any step count. `layer_bench.py` therefore
  checks at a fixed 4 steps, with the f32 wire as the regression check. The
  owner accepted the bf16 wire for this experiment without a served-model
  accuracy eval (2026-11-20).
* Noise: the no-AR arms do not depend on the wire dtype, yet B = 64
  streamed/no-AR read 17.4 µs in one run and 22.3 µs in the other, and B = 1
  rows swing by ~2 µs. A single run resolves about ±25%. Rows below 1.2× are
  not distinguishable from 1.0× without repeats.

### TPU 8i (XManager `xid/1000000001543729`, `sunfish_sampling=4x1x1`)

Run by the experiment owner through XManager (`tpu-npi-pilot`, `wl-kerneltest`),
jax 0.12.1, `--xla_tpu_scoped_vmem_limit_kib=190464`, 8 TensorCores on 2 hosts.
Raw logs are in the owner's x20 folder `Nov19`. Both hosts report the same
numbers to within ~1%.

**Stage L1, 128 experts EP=8 (16 per TensorCore, ~95 MiB resident), top-8.**
µs per step, slope over 4 → 16 steps:

| wire | B | resident + AR | streamed + AR | streamed / resident | resident, no AR | streamed, no AR | streamed / resident, no AR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bf16 | 1 | 8.36 | 10.69 | 1.28× | 0.75 | 2.31 | 3.09× |
| bf16 | 4 | 10.29 | 16.02 | 1.56× | 2.96 | 7.43 | 2.51× |
| bf16 | 16 | 16.58 | 28.65 | 1.73× | 9.02 | 19.89 | 2.20× |
| bf16 | 64 | 28.02 | 43.22 | 1.54× | 13.93 | 29.51 | 2.12× |
| f32 | 16 | 18.08 | 30.69 | 1.70× | 9.33 | 19.58 | 2.10× |
| f32 | 64 | 35.96 | 51.32 | 1.43× | 13.79 | 29.72 | 2.15× |

* The whole MoE block is 1.54-1.73× faster resident at B ≥ 4. The bf16
  all-reduce costs ~7.5 µs at B = 16 and ~14 µs at B = 64, about half of
  v7x's 16-TensorCore cost (3 rounds instead of 4).
* Against v7x on the same 128-expert block (`vmempin-pt-v7x8-*`, 8 chips):
  at B = 64 resident + AR, TPU 8i is 28.0 µs on 4 chips against 41.4 µs on
  8 chips. That is 571 k against 193 k tokens/s per chip, and ~2.5× per chip
  at B = 16.

**Single TensorCore, top-1, 64 rows, median of 6.**

| local experts | B=1 | B=4 | B=16 | B=64 | resident µs/item | streamed µs/item |
| --- | --- | --- | --- | --- | --- | --- |
| 16 (~96 MiB) | 1.44× | 1.39× | 1.45× | 1.41× | 1.31-1.63 | 1.90-2.34 |
| 8 | 1.84× | 1.63× | 1.40× | 1.47× | 1.34-1.64 | 1.90-3.01 |
| 4 | 1.64× | 1.45× | 1.50× | 1.16× | 1.28-1.62 | 1.85-2.17 |

* Below the modelled 2.29×. Streamed runs at ~1.9 µs per expert, ~72% of its
  1.38 µs HBM floor. Resident runs at ~1.3 µs per expert, only ~45% of its
  0.605 µs push floor, and about the same per-expert time as on v7x. So the
  resident arm is bound by chip-independent per-item work, not by the MXU
  weight push. In `expert_ffn`, the step input is re-quantized (abs-max,
  divide, fp8 cast over `[rows, 2816]`) for every expert item although it is
  the same within a step. The single-TensorCore benchmark also always
  processes 64 rows. Hoisting the input quantization out of the expert loop
  and sizing rows to B is the next change.

## Running

```bash
# CPU interpreter
JAX_PLATFORMS=cpu pytest tests/kernels/experimental/test_vmem_resident_moe.py

# v7x, one TensorCore, 8 local experts (~48 MiB resident)
LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=63488 \
python -m vllm_torchtpu.kernels.experimental.vmem_resident_moe.benchmark \
    --experts 8 --hidden 2816 --inter 704 --rows 64 --steps 4,16 \
    --batches 1,4,16,64 --vmem-limit-mib 61
```

### On exos (how `j-b92e1ec1` ran)

`cloud-devkit` at `48cfbe34` or later (it has the exos recipes). The directory
mapped in holds this package's `kernel.py` and `benchmark.py` and a `run.sh`
that runs the command above twice, at `--experts 8` and at `--experts 4`.

```bash
CDK_ROOT_DIR=<cloud-devkit checkout> cdk job create --cluster cdk-tpu-exos \
  exos-simple-run-command --active-deadline-seconds 1800 \
  --map-dir "<dir>:*:/root/vmemjob" -t agent-jobs,vmem-resident-moe \
  PLATFORM=ghostfish SLICE=2x2 \
  DOCKER_IMAGE=us-central1-docker.pkg.dev/cloud-tpu-inference-test/inferencex/vllm:harut-glm52-pd-agentx-c934e275d-tpusync1012-flat@sha256:bfb9752fe98af30dee4d6584f9f65e686197eea6e82556abca14c8f174ff1047 \
  "RUN_CMD=bash /root/vmemjob/run.sh"
```

The slice is 4 chips; the benchmark uses device 0 only.

`vmem_limit_bytes` above the default needs `--xla_tpu_scoped_vmem_limit_kib`
too (`jax/_src/pallas/mosaic/core.py:98-100`). On TPU 8i, 16 local experts
(one TensorCore's share of 128 experts at EP=8, ~96 MiB) fit in 192 MiB.

## Next

1. TPU 8i, `--experts 16`, on exos `sunfish_sampling`. The smallest slice is
   `4x1x1` (2 hosts). The run needs the arm64 TPU 8i image and a 2-host
   launcher.
2. Whole layer on 8 TensorCores: attention + dense MLP (TP) + expert
   all-to-all, all resident, N steps in one call.
3. Arm 3: XLA-placed VMEM weights across per-step calls.

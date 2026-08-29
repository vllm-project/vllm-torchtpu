# Kimi K3 MXFP4 batch-1 TEP32 optimization

Date: 2026-08-12

## Goal and workload

Optimize and verify the device latency of Kimi K3's two routed MXFP4 grouped
GEMMs for a real-model TEP32 batch-1 decode workload. The authoritative model
configuration has 896 routed experts, top-16 routing, hidden size 3584, and
expert intermediate size 3072. TEP32 therefore owns 28 experts per rank. A
rank commonly receives zero or one route, but in the worst case it can receive
all 16 routes; routing occupancy must remain dynamic. The exact static
one-token shapes are:

```text
local experts E=28; routed-row capacity M=16; active local rows=0..16;
hidden H=3584; expert intermediate I=3072; MXFP4 block size=32
```

This is distinct from the earlier TP8/EP8 pad-skipping investigation, where a
rank commonly received two routes.

## Milestone log

### M1: Baseline reconstruction and initial kernel finding

- Confirmed the real K3 dimensions and TEP32 local expert count from the
  published model configuration.
- Confirmed that the current `gmm_v2` auto-tiler selects one full-K/full-N tile
  for both exact K3 projections at `M=16`:
  - GMM1: `tm=16, tk=3584, tn=3072` per gate/up half.
  - GMM2: `tm=16, tk=3072, tn=3584`.
- This produces only one pipeline iteration for an active expert. Splitting N
  or K is the first tuning hypothesis because it can introduce DMA/MXU overlap
  without changing routing or numerical semantics.
- Added `benchmarks/kernels/benchmark_kimi_k3_mxfp4_gmm.py`, an exact-shape
  benchmark that passes FP4 weights as runtime arguments, queues compiled
  executions, and synchronizes once after timing.

### Issue I1: local environment initially empty

The newly created workspace `.venv` contained no JAX, torch, or pytest. A plain
`uv sync --extra test` failed because the private TPU Artifact Registry returned
HTTP 401. The service-account impersonation documented by `../download.sh` was
then supplied to uv through index credential environment variables. This avoids
the helper script's `echo $ACCESS_TOKEN`, which would expose a credential in
logs. All subsequent commands in this investigation use this workspace's
`.venv` as requested. The benchmark was first brought up with JAX 0.10.2,
jaxlib 0.10.2, libtpu 0.0.44.1, NumPy, and pytest. The exact private torch-tpu
wheel and vLLM 0.27.0 were installed afterward for the repository tests.

### Issue I2: invalid dependent timing chain

The first benchmark revision fed each GMM output back as the next invocation's
input. That is invalid for K3 because GMM1 changes width from 3584 to 3072 and
GMM2 changes it from 3072 to 3584; the AOT executable correctly rejected the
second call's shape. The benchmark now queues repeated calls with identical
runtime arguments and retains the single synchronization outside the timed
loop. This is the same independent-queue methodology used in the earlier
exact-shape investigation. A shape-preserving dependent chain is valid only
for the complete GMM1+GMM2 pair, whose output width returns to 3584.

### M2: device-loop timing removed the launch floor

Queued AOT calls varied between roughly 38 and 62 us for the same executable.
This variation is host/executable dispatch and TPU state, not a stable measure
of the two device kernels. The accepted benchmark wraps 500 dependent pairs in
one compiled `jax.lax.fori_loop`, synchronizes the one executable, and divides
by 500. Five repeats agree within 0.7%:

```text
auto pair: 20.457, 20.359, 20.355, 20.362, 20.326 us
median:    20.359 us
```

The benchmark still prints separately compiled and queued-pair diagnostics,
but they are not used to select tiles or claim device improvement.

### Issue I3: TPU lock during sequential sweep

The shell sweep was still compiling its fourth candidate after its first output
collection yielded. Starting another JAX process then failed on the libtpu
multi-process lock. No lockfile was removed: process inspection identified the
live owner, and waiting on the original benchmark completed cleanly. TPU tuning
runs remain serialized.

### Issue I4: local profiler ABI mismatch

Direct `jax.profiler.trace` capture failed before tracing: libtpu reported a
104-byte plugin profiler API while JAX expected 80 bytes, then segfaulted in
`PluginTracer::Start`. The in-device compiled loop is the fallback measurement
used here. Existing full-model XPlane capture from the torch-tpu profiler does
not have this failure, but a true TEP32 full-model capture needs 32 chips; this
host exposes one TPU7x-8 slice. That full-system confirmation remains pending.

### Issue I6: SSH push transport unavailable

The initial PR branch push failed because the checkout's `origin` used SSH but
the host had no usable GitHub SSH key. GitHub CLI authentication was healthy,
so the checkout was switched to its authenticated HTTPS transport before the
push was retried. This did not affect source or benchmark results.

## Results

### Baseline

The exact one-active-route TEP32 local baseline is **20.359 us** for GMM1 plus
GMM2 on TPU7x. This is 2.23x below the earlier compile-size-1 EP8 full-model
pair (45.399 us), but that cross-run difference includes the requested routing
occupancy change from roughly two local experts to one. The TEP32 baseline is
therefore the 20.359 us local result, not the older 45.399 us EP8 result.

### N/K tile sweep

| GMM1 tile `(tm,tk,tn)` | GMM2 tile `(tm,tk,tn)` | Pair |
|---|---|---:|
| auto = `(16,3584,3072)` | auto = `(16,3072,3584)` | 20.359 us |
| `(16,3584,1536)` | `(16,3072,1792)` | 18.546 us |
| `(16,3584,768)` | `(16,3072,896)` | 18.932 us |
| `(16,3584,512)` | `(16,3072,512)` | 20.944 us |
| `(16,1792,1536)` | `(16,1536,1792)` | 18.837 us |
| **`(16,3584,1024)`** | **`(16,3072,1792)`** | **18.183 us** |
| `(16,3584,1024)` | `(16,3072,2048)` | 18.659 us |
| `(16,3584,1024)` | `(16,3072,1536)` | 19.138 us |
| `(16,3584,1024)` | `(16,3072,1280)` | 18.564 us |
| `(16,3584,1280)` | `(16,3072,1792)` | 20.114 us |

The accepted split uses three N tiles for GMM1 and two for GMM2. Full-K remains
best; splitting K adds accumulation overhead. The result supports the narrower
DMA/MXU-overlap hypothesis: a small amount of N pipeline depth helps, while
fine-grained tiling loses the gain to pipeline overhead.

### M3: production implementation

`gmm_wrapper` now selects a two-way N split for GMM1 only, and only for
native-FP4 decode with the exact K3 shape, 28 local experts, and at most 16
routed rows. GMM2 retains its auto full-N tile. Other models, larger
batches/prefill, and non-FP4 kernels keep `calculate_tiling`. The one-route
production-wrapper benchmark gives:

```text
tuned pair: 18.928, 18.845, 18.846, 18.833, 18.819 us
median:     18.845 us
```

This is **7.4% lower latency / 1.080x faster** than the 20.359 us baseline.
The occupancy sweep that selected this compromise is:

| Active local routes | Auto pair | Production pair | Change |
|---:|---:|---:|---:|
| 1 | 20.607 us | 18.845 us | -8.6% |
| 2 | 29.975 us | 29.011 us | -3.2% |
| 4 | 49.687 us | 48.492 us | -2.4% |
| 8 | 86.504 us | 86.849 us | +0.4% |
| 16 | 159.990 us | 163.289 us | +2.1% |

The initially selected three-way GMM1 plus two-way GMM2 split was faster at
one route (18.117 us), but regressed the 16-route worst case by 11.0%. It was
replaced by this occupancy-robust GMM1-only split after clarifying that TEP32
does not statically cap a rank at one route.

### M4: one-route lower bound and the 12 us question

The initial dynamic-slice result was invalid because it used the
launch-dominated host harness. Repeating it inside the compiled 500-iteration
device loop shows that selecting one expert is effectively folded into the
weight access. Compacting the routed row from 16 to 1 then gives:

| Scale block | One-expert compact pair (median) | Numerical status |
|---|---:|---|
| 32 (checkpoint-native) | **15.889 us** | Lossless |
| 256 (load-time requantized) | **11.810 us** | Additional lossy requantization |
| no scales (diagnostic only) | about 11.1 us | Not a valid MXFP4 model |

So, **12 us is a realistic kernel target for one active expert, but it is not
the lossless block-32 result**. The decisive difference is scale granularity,
not expert selection alone. At block 32, `gmm_v2` dequantizes the weight tile
before the MXU operation. At block 256 (one MXU width), it can apply scaling
after the matmul, eliminating roughly 4.1 us from the compact pair.

The repository already supports opt-in load-time requantization through
`MOE_REQUANTIZE_BLOCK_SIZE`; setting it to 256 reaches the timing above. It is
not enabled by this change because requantizing block-32 MXFP4 weights to
block-256 introduces a model-quality tradeoff. Real-checkpoint perplexity and
task-quality validation is required before making it a default.

With fully dynamic 0–16 routing and static 16-row capacity, block-256 measures
**16.494 us** for the complete one-route fused operator (routing, both GMMs,
and combine), versus 22.037 us for lossless block-32. The 11.810 us figure is
an isolated compact-row lower bound, not a valid general-routing latency claim.

### Issue I5: a correctness guard erased the fast-path gain

A prototype used a runtime `jax.lax.cond`: zero or one local route
took the compact path, while two or more routes preserved generic semantics.
Although its isolated compact GMM pair was fast, the complete fused function
measured about **112 us** because conditionals around the two Pallas kernels
introduced device synchronization/dispatch boundaries. The prototype was
removed. An unconditional compact path would be fast, but is only correct if
the deployment can provide a static guarantee that no rank receives two
routes. Top-16 routing across 32 shards does not establish that guarantee by
itself, so silently assuming it in the general operator is unsafe.

For context, the current production operator—including local routing,
permutation, both GMMs, and combine—measures **22.037 us** median versus
**23.455 us** with auto tiles in the same device-loop methodology, a **6.0%**
end-to-end kernel improvement. Its one-route GMM-only pair is 18.845 us; about
3.2 us is local routing/combine overhead.

### Rejected experiments

- A runtime conditional around compact and generic paths measured about 112 us
  and was removed; see Issue I5.
- Replacing general metadata construction with direct single-group metadata
  saved only about 0.05 us in the valid device-loop harness and was removed.
- Keeping scales in byte-encoded E8M0 form and decoding in the kernel regressed
  the pair to about 24.5 us and was removed.
- BF16 scale storage is numerically exact for E8M0 powers of two, but Mosaic
  cannot perform the existing zero-stride replicated scale load on non-32-bit
  data. A normal BF16 load plus broadcast compiled but regressed badly, so the
  production FP32 scale path is unchanged.
- A dispatch-only executable benchmark hit the same approximately 60 us launch
  floor for both generic and single-route bookkeeping. Dispatch must be judged
  inside a full compiled model, not as a standalone executable.

## Validation

- Production wrapper microbenchmark: median 18.845 us over five in-device-loop
  samples, versus baseline median 20.359 us.
- Compact single-expert/block-32 pair: median 15.889 us.
- Compact single-expert/block-256 pair: median 11.810 us; performance validated,
  model quality not yet validated.
- Full production fused path on the local TPU7x rank: median 22.037 us, versus
  an auto-tiling full-fused median of 23.455 us.
- Tile-selector unit tests: `2 passed, 25 deselected`.
- Complete fused-MoE GMM suite: `27 passed, 14 warnings in 167.24 s`.
- Compressed-tensors MXFP4 MoE suite: `8 passed, 15 warnings in 52.81 s`.
- `py_compile` passed for both benchmarks, the implementation, and the focused
  test file.
- YAPF passed on the changed Python files. Ruff reports only existing findings
  in the touched full files; the new benchmark-specific findings were fixed.
- `git diff --check` passed.

The remaining validation gap is an end-to-end real-K3 TEP32 XPlane capture on
a 32-chip topology, including routing/dispatch categories and exact per-layer
GMM spans. The current TPU7x-8 host can validate the per-rank local kernel shape
but cannot instantiate that topology.

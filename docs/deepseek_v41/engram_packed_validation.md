# Packed Engram validation

The fused route consumes the existing packed int32 upload, computes the FP8
projection and gate in one Pallas call, and returns the BF16 residual. It is
selected for TPU v7, 288 tokens, 24 hash heads of 256 values, four hidden
streams of dimension 5120, and canonical E4M3 WKV weights `[6144, 25600]`
with per-channel FP32 scales. Host-offloaded embedding TP1/DP1 and no sequence
parallelism are required. Shared host tables can meet the embedding DP1
requirement inside a larger model DP configuration. Other configurations use
the original forward path.

## Numerics

The packed buffer has all raw FP8 words followed by all E8M0 scale words.
Dequantization, per-token activation quantization, WKV projection, RMS gate,
post-sigmoid token mask and residual follow the existing algorithm. The layer
passes its actual epsilon and clamp value; the model fixture uses `1e-20` and
`1e-6`, respectively. Masked finite rows retain the input hidden state.

The fused gate retains FP32 K projection statistics instead of materializing
and rounding K to BF16. V retains its BF16 boundary. Reduction and activation
quantization operation order can also affect rounding. This is consequently
a numerical change requiring measured tolerance and model evaluation, rather
than a claim of bitwise equivalence.

The real-layer replay constructs a vLLM `ColumnParallelLinear`, processes its
weights with the TPU FP8 method, and compares the original packed decoding
and linear runtime against the actual fused `Engram.forward`. It checks the
compiled FX graphs select distinct paths, reports maximum/mean absolute
error, checks a mixed token mask and requires `atol=0.02, rtol=0.01`.

```bash
python scripts/dsv41_bench/bench_engram_packed_forward.py \
  --output /output/engram-forward.json \
  --profile /output/engram-forward-traces
```

This replay uses synthetic resident embeddings and weights; it does not
allocate the embedding table or assess whole-model quality. Accelerator
numerical results and full-model evaluation must be recorded separately.

## Performance

The replay warms both compiled paths, alternates their order randomly for
40 iterations, synchronizes every result and reports wall time including
host dispatch. Profiling runs after formal timing. Its measurement excludes
hash lookup, host gathering, H2D, model startup and model serving.

Full-model performance requires separate baseline and candidate server
processes on the same rig, checkpoint and configuration. The baseline can
bind `Engram.forward` to `_forward_unfused` in the benchmark snapshot; the
candidate uses the default supported-shape dispatch. Production code has no
A/B environment switch. The candidate must show actual custom-call coverage
to establish that its measured requests use the new kernel.

## Current evidence

- Ruff 0.14.0 format/check, the closure/env ratchets and minimal Pyrefly
  1.3.0 pass without baseline changes.
- CPU abstract tracing with JAX 0.10.2 accepts the packed bool-mask interface
  and returns one BF16 `[288, 4, 5120]` tensor. This checks tracing, not TPU
  lowering or execution against the branch's newer runtime pins.
- Full-environment type checking requires the CI container; the local
  machine has no Docker.
- TPU numerical, kernel timing and complete model A/B results are pending.

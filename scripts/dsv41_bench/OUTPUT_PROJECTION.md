# Buffered decode output projection

The production specialization targets TPU v7 with BF16 activations, FP8 E4M3
weights and FP32 per-channel scales, for token buckets 240 and 288:

- O-A: already inverse-rotated `[T,64,512]` input, `[4096,8192]` weights.
  Three DMA generations reuse the existing projection arithmetic with the full
  token tile. Inverse-RoPE, explicit tiling and other shapes retain the original
  pipeline.
- O-B: `[T,8192]` input, `[8192,5120]` weights. Full-row FP32 activation
  quantization stays in VMEM and feeds a double-buffered matmul. Other dtypes,
  shapes and block scales retain their existing paths.

Selection occurs during JAX lowering, after Torch's dynamic-token export.
Weights keep the existing K-major representation; no extra resident copy is
introduced. Both kernels drain output DMA before reusing or releasing scratch.

## Reproduce

In the pinned TPU development environment:

```sh
export LIBTPU_INIT_ARGS="--xla_tpu_dvfs_p_state=7 --xla_enable_custom_call_region_trace=false"
python scripts/dsv41_bench/bench_output_projection.py --tokens 288 --output /tmp/output-projection-288
python scripts/dsv41_bench/bench_output_projection.py --tokens 240 --output /tmp/output-projection-240
pytest -q tests/models/vllm/test_dsv41_output_projection.py
```

The benchmark imports the production kernels and bypasses O-A auto-dispatch
with explicit original sub-tiling for its reference. It checks bitwise equality,
interleaves complete-call XPlane measurements, and saves HLO and raw profiles.
Supply native LLO dump flags through `LIBTPU_INIT_ARGS` before launching Python.
Use region trace only in separate diagnostic runs: instrumentation can change
register allocation and scheduling.

## Prior standalone qualification

Before production integration, at epic baseline `657a7a57a`, v7 P7,
200 samples per variant, complete XLA module median in microseconds:

| Tokens | Path | Reference | Buffered candidate |
| --- | --- | --- | --- |
| 288 | O-A | 25.391 | 21.573 |
| 288 | O-B | 25.288 | 20.420 |
| 288 | O-A then O-B | 51.179 | 41.788 |
| 240 | O-A | 22.630 | 20.399 |
| 240 | O-B | 23.419 | 20.005 |
| 240 | O-A then O-B | 45.635 | 40.105 |

Falcon experiment `exp-cp3logeuql`, artifact `art-fbz7qr7cou`, contains native
LLO, HLO, XPlane, 2400 formal timing samples and 40 numerical cases.
All numerical cases were bitwise equal to their references. Sixteen cases
were controls for the unchanged inverse-RoPE path, not new optimized coverage.

These are standalone qualification results, not a measurement of the final
Torch bridge or whole-model performance. The CI test covers actual CSA2 and
linear bridges, fallback buckets and dynamic export; the existing DSv4.1 CI
lane runs model smoke/GSM8K and decode/prefill benchmarks. No serving E2E or
prefill speedup is claimed from the table.

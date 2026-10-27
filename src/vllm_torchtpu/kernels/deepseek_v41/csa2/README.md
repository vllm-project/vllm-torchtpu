# CSA2 kernels

| Module | Entry point | Contract |
| --- | --- | --- |
| `decode.py` | `csa2_decode` | Decode and decoder-layer prefill; six Q by default, fixed 8-Q tiles for long sequences |
| `prefill.py` | `csa2_prefill` | Encoder-layer prefill and general ragged attention |
| `cache.py` | Cache operations | Quantization, publication, gather and address preparation |
| `pipeline.py` | `csa2_attention`, `csa2_pipeline` | Attention dispatch and overlapping KV gathers |
| `reference.py` | `prefill_reference` | Independent JAX math for numerical validation |

Decode and prefill retain separate compute implementations and public APIs. The QK stages are named `swa_qk_compute`,
`swa_qk_pop`, `main_qk_compute`, and `main_qk_pop`; the remaining stages are
`main_unpack`, `softmax`, `fused_pv`, and `inverse_rope`. Accumulator refs are
allocated directly with `jax.empty_ref` and indexed as `accumulators[mxu][slot]`.
Logits and sink values enter the base-2 softmax through
`log2_e = math.log2(math.e)`.

Encoder/decoder refer to the model's layer stages: the first and last 20 layers
in the 40-layer DSV41 configuration. The framework derives the boundary from its
model configuration and passes `_ced_decoder` as `decoder`; it never infers the
stage from causal or noncausal visibility.

Decoder-layer prefill reuses the decode pipeline for consecutive, equal-length
requests whose query count is divisible by eight. Both use FP32 KV scales.
Six-Q dispatch is unchanged. Ragged requests, padding, split SWA banks, and
explicit noncausal bounds retain the general prefill path. Both gather and KV
reuse propagate the same model-stage flag and authoritative main counts.

Six-query H64 decode with 64-token SWA pages and at least eight requests uses
four input buffers, carries PV across requests, and preloads the next SWA RHS.
Other decode shapes retain the buffered grid. Both schedules use the same decode
math and FP32 scales, including the native SWA physical-owner bounds checks.

Prefill uses four Q/KV/SWA buffers, two output buffers, and cross-tile PV.
Each 64-row packed-KV read is converted in four 16-row slices shared by QK/PV.
Scale and inverse-RoPE tables stay resident only when they fit within one quarter
of core VMEM; larger batches stream them through fixed-size buffers.
`dummy_metadata=True` is exclusively an all-keys-valid mathematical benchmark;
its timing does not represent real paged prefill.

Prefill keeps tile planning, validation, and launch setup in the same module.
Causal and noncausal visibility specialize its implementation at tracing; these
mask modes do not name encoder or decoder stages.

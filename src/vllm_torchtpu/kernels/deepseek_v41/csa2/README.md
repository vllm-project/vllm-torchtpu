# CSA2 kernels

| Module | Entry point | Contract |
| --- | --- | --- |
| `decode.py` | `csa2_decode` | Decode and decoder-layer prefill; six Q by default, fixed 8-Q tiles for long sequences |
| `prefill.py` | `csa2_prefill` | Encoder-layer prefill and general ragged attention |
| `cache.py` | Cache operations | Quantization, publication, gather and address preparation |
| `pipeline.py` | `csa2_attention`, `csa2_pipeline` | Attention dispatch and overlapping KV gathers |
| `reference.py` | `prefill_reference` | Independent JAX math for numerical validation |

Decode and prefill each own one Pallas kernel. The QK stages are named `swa_qk_compute`,
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
requests whose query count is divisible by eight. `decoder_prefill=True` retains
the prefill path's FP32 KV scales; ordinary decode keeps BF16-effective scales.
Six-Q dispatch is unchanged. Ragged requests, padding, split SWA banks, and
explicit noncausal bounds retain the general prefill path. Both gather and KV
reuse propagate the same model-stage flag and authoritative main counts.

Prefill keeps tile planning, validation, and launch setup in the same module.
Causal and noncausal visibility specialize its implementation at tracing; these
mask modes do not name encoder or decoder stages.

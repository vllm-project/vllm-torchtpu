# GLM-5.3-Flash

| | |
|---|---|
| Hugging Face repo | `zai-org/GLM-5.3-Flash` |
| Architecture classes | `Glm5NextForConditionalGeneration`, `Glm5NextForCausalLM`, `Glm5NextMTPModel` (`model_type: glm5_next`) |
| Code | `src/vllm_torchtpu/models/vllm/glm5_next/`, `src/vllm_torchtpu/layers/adapter/custom_ops/glm53_*`, `src/vllm_torchtpu/kernels/glm53_*` |

## Model summary

- **45 decoder layers**: 34 Kimi Delta Attention (KDA, linear attention with a
  fixed-size `[H, 128, 128]` recurrent state and a 4-tap causal short
  convolution) and 11 NoPE Multi-head Latent Attention (MLA) layers (`q_lora_rank=1536`,
  `kv_lora_rank=512`, `qk_nope_head_dim=256`, `qk_rope_head_dim=0`).
- **4-stream Manifold-Constrained Hyper-Connections (mHC)** around every
  attention and FFN sublayer (`hc_mult=4`, 20-step Sinkhorn-Knopp doubly
  stochastic projection).
- **Hybrid FFN schedule**: 3 dense clamped-SwiGLU MLP layers followed by 42
  sparse MoE layers (288 routed FP8 experts, top-8 `noaux_tc` sigmoid routing +
  1 shared FP8 expert, `swiglu_limit=10.0`).
- **Multi-Token Prediction (MTP)**: layer 45 (`Glm5NextMTPModel`) provides
  single-stream `enorm` + `hnorm` + `eh_proj` fusion followed by a NoPE MLA +
  288-expert MoE block and shared LM head.

## Architecture-specific TPU support

- **Unified KV & KDA state pool**: `Glm53TpuKimiDeltaAttention` runs the Kimi
  pooled KDA op (`build_kimi_pooled_kda_op` with `pool_tokens_per_row=1` on the
  RPA pool, `float32` conv state and `float32` recurrent state) so KDA state
  and MLA KV cache share a single unified block pool (`--mamba-cache-mode align`).
- **Pallas mHC kernels** (`VLLM_TPU_GLM53_MHC_IMPL=pallas`, default): fuses the
  post-sublayer stream update of site `k` with the Sinkhorn-Knopp pre-collapse
  and RMSNorm of site `k + 1`.
- **One-dispatch FP8 MoE** (`VLLM_TPU_GLM53_MOE_IMPL=fused`, default): fuses
  router GEMM, Pallas `noaux_tc` top-k routing, bitpacked activation sort,
  Megablox FP8 GMMs, and the shared expert into a single JAX op per layer.
- **Sequence-parallel prefill** (`VLLM_TPU_GLM53_SP_PREFILL=1`, default):
  shards the token axis across TP ranks when `num_tokens >= VLLM_TPU_GLM53_SP_MIN_TOKENS`
  so mHC sites, norms, and replicated MLA projections run on `T / tp` tokens.

## Serving on TPU v7x (`TP=8`)

```bash
vllm serve zai-org/GLM-5.3-Flash \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --mamba-cache-mode align \
  --max-model-len 131072 \
  --max-num-seqs 64
```

To enable Multi-Token Prediction (`k=3` speculative draft tokens):

```bash
vllm serve zai-org/GLM-5.3-Flash \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --mamba-cache-mode align \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```

## Limitations

- **Dense causal MLA**: Checkpoint `.self_attn.indexer.*` sparse-indexer tensors
  are skipped on TPU and MLA layers run dense causal attention. Outputs match
  the sparse top-2048 indexer up to `2,111` tokens (`index_topk=2048` plus tail
  pool); longer sequences attend over all prior tokens.
- **Text-only VLM wrapper**: `Glm5NextForConditionalGeneration` skips multimodal
  vision-tower checkpoint weights and serves text inputs only.
- **Unsupported features**: Dynamic EPLB (`--enable-eplb`) and disaggregated KV
  transfer connectors over the unified KV+KDA pool are rejected at startup.

## Decode megakernel

When `VLLM_TPU_MEGAKERNEL=glm53` (or `additional_config["megakernel"]`) is set
with `--no-async-scheduling` on a `TP=8` TPU v7x slice (`--kv-cache-dtype fp8`,
`--mamba-cache-mode align`), `TPUModelRunner` attaches the full-model Pallas
decode megakernel (`vllm_torchtpu.models.vllm.glm5_next.megakernel`) as its
`decode_owner`:

```bash
VLLM_TPU_MEGAKERNEL=glm53 vllm serve zai-org/GLM-5.3-Flash \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --mamba-cache-mode align \
  --no-async-scheduling \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```

- Runs all 45 decoder layers (and optional `k=3` MTP drafting + verification)
  inside one `pl.pallas_call` dispatch per decode step with on-chip barrier
  all-reduce collectives and async weight/KV double-buffering (`numerics="approx"`).
- Shares routed MoE expert weights in-place with the vLLM prefill model via a
  stacked `[42, E_local, ...]` expert store (`rhs_group_base`), avoiding HBM
  duplication.
- Synchronizes KDA recurrent/conv state and compressed FP8 MLA KV cache between
  vLLM prefill chunks and megakernel decode slots through the unified block
  pool. Greedy requests (`temperature=0`, `logprobs <= 64`) are served by the
  megakernel; non-greedy requests fall back to the stock runner.

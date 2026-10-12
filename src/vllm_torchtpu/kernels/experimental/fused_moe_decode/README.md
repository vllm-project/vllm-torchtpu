# Fused MoE Decode Kernel

One Pallas call per rank that runs a MoE block's routed experts, its shared
expert and the cross-rank all-reduce, for small token counts. Opt-in with
`FUSED_MOE_DECODE_KERNEL=1`.

## API

```python
from vllm_torchtpu.kernels.experimental.fused_moe_decode import (
    fused_moe_decode,
    vmem_bytes,
)

out = fused_moe_decode(
    rank,                 # int32[1], this rank's mesh index
    hidden,               # [T, H] bf16, the same on every rank
    w1, w2,               # [E, H, 2i] / [E, i, H], this rank's experts
    w1_scale, w2_scale,   # [E, 1, 1, 2i] / [E, 1, 1, H] f32 per channel
    shared_w1, shared_w1_scale,  # [H, 2i_s] / [1, 2i_s]
    shared_w2, shared_w2_scale,  # [i_s, H] / [1, H]
    shared_gate_weight,   # [H, 1], output scaled by sigmoid(hidden @ it)
    topk_weights, topk_ids,      # [T, k] from the caller's router
    n_ranks=8,
    expert_start=None,    # global id of the first local expert (EP); None for TP
)  # -> [T, H] bf16, all-reduced, identical on every rank
```

- Call it inside `shard_map` over the mesh, one call per rank.
- **TP experts:** every rank holds all experts with the intermediate dimension
  sliced (`i = I / n_ranks`).
- **EP experts:** each rank holds whole experts `[expert_start, expert_start + E)`.
  `topk_ids` stay global; routes owned by other ranks are skipped and summed in
  by the all-reduce.
- `vmem_bytes(tokens, h, i, i_s, experts, n_ranks, weight_dtype)` returns the
  VMEM one call declares. The adapter uses it to refuse a token cap that cannot
  compile.

### Serving

`layers/adapter/fused_moe_decode.py` arms eligible layers once after load. That
includes the MTP drafter. An armed layer runs this kernel for steps of up to
`FUSED_MOE_DECODE_KERNEL_MAX_TOKENS` tokens (default 256). Larger steps run the
regular GMM MoE, the shared expert and one all-reduce, inside the same exported
program.

## When to use it

Use it when all of these hold:

- Decode or other small batches: up to a few hundred tokens per step. Above
  that, prefill-style kernels win.
- Attention is TP. Every rank sees every token, and the MoE output is
  all-reduced. (DP attention: see below.)
- The MoE is TP-sharded, or EP with `ep_size == mesh size`.
- Weights are FP8 with per-channel scales, the activation is SiLU, and there
  are no expert biases.
- The shared expert has a sigmoid gate, `routed_scaling_factor` is 1, and there
  are no routed input/output transforms.
- `i`, `i_s` and `H / n_ranks` are multiples of 128, and `i_s <= i`.
- At least 3 slots of one expert's `w1 + w2` fit in half of VMEM. This holds
  for Qwen3.5-397B TP8 and EP8.

Do not use it for:

- Experts too large to hold whole in a VMEM slot. GLM-5.2 EP is one example.
- Prefill-sized steps.

The adapter declines a layer that fails these checks, and that layer keeps the
GMM path.

### DP attention

The kernel can serve DP attention, where each rank owns different tokens. The
caller all-gathers the tokens and their routing before the call, so the kernel
sees the global batch, and keeps only its own rows of the output. The token cap
then counts global tokens, not tokens per rank.

The serving adapter does not support DP attention yet; that is TBD. Today it
arms only with TP attention (DP 1).

## Design

- **No dispatch.** Each active expert runs on all `T` tokens in their original
  order. The route weight zeroes the rows the expert does not own. At decode
  sizes this is cheaper than sorting, permuting and combining. Nothing
  intermediate goes to HBM.
- **Active-expert compaction.** XLA builds a dense `[T, E]` route-weight table
  and an active mask. The kernel compacts the active expert ids into SMEM and
  visits only those.
- **Weight ring.** Each active expert's `w1/w2` slices stream from HBM through
  a ring of VMEM slots. Ring depth is between 3 and 6, picked by VMEM size. The
  ring is primed while the shared expert computes. Expert `j+1`'s first matmul
  is issued during expert `j`'s elementwise chain, so the MXU stays busy while
  the VPU quantizes.
- **Numerics.** Hidden states are quantized once per row, and the activated
  intermediate per row per expert. Both matmuls run in the weight dtype (FP8)
  with FP32 accumulation into one VMEM accumulator.
- **Shared expert in-kernel.** Its slice is loaded once at entry and computed
  first, gated by `sigmoid(x · w_g)`.
- **In-kernel all-reduce.** It is a remote-DMA reduce-scatter over column
  chunks, then an all-gather, with bf16 on the wire. The cross-rank barrier is
  signalled at entry and waited only before the reduce-scatter, so the expert
  loop hides it.
- **Limits.** All tokens, the accumulator and the reduce-scatter buffers stay
  resident in VMEM. That caps the token count, and `vmem_bytes` computes the
  cap.

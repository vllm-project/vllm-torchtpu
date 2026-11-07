# Fused MoE Prefill Kernel

The prefill-sized counterpart of the fused MoE decode kernel: a MoE block's
routed experts, its shared expert, the combine and the cross-rank all-reduce,
per rank, for steps of hundreds to hundreds of thousands of tokens. Opt-in with
`FUSED_MOE_PREFILL_KERNEL=1` on top of `FUSED_MOE_DECODE_KERNEL=1`.

## API

```python
from vllm_torchtpu.kernels.experimental.fused_moe_prefill import (
    fused_moe_prefill,
    plan_blocks,
)

out = fused_moe_prefill(
    hidden,               # [T, H] bf16, the same on every rank
    w1, w2,               # [E, H, 2i] / [E, i, H], this rank's experts
    w1_scale, w2_scale,   # [E, 1, 1, 2i] / [E, 1, 1, H] f32 per channel
    shared_w1, shared_w1_scale,  # [H, 2i_s] / 2i_s scales
    shared_w2, shared_w2_scale,  # [i_s, H] / H scales
    shared_gate_weight,   # [H, 1], output scaled by sigmoid(hidden @ it);
                          # None for an ungated shared expert
    topk_weights, topk_ids,      # [T, k] from the caller's router
    axis_name="d",        # mesh axis of the all-reduce
    expert_start=None,    # global id of the first local expert (EP); None for TP
    routed_scale=1.0,     # scales the routed experts only (GLM-5.2: 2.5)
    i_block=None,         # force blocked mode with this width (tests)
)  # -> [T, H] bf16, all-reduced, identical on every rank
```

- Call it inside `shard_map` over the mesh, one call per rank. Arguments,
  TP/EP experts and `routed_scale` mean what they mean for
  `fused_moe_decode`; routes of weight 0 (padded rows) are not computed.
- `plan_blocks(tokens, h, i, i_s, weight_dtype)` returns blocked mode's
  (block width, ring depth, chunk rows), or None for whole experts.

### Serving

With both env vars set, a TP-expert layer armed for the decode kernel runs this
kernel for steps above `FUSED_MOE_DECODE_KERNEL_MAX_TOKENS` instead of the GMM
MoE, inside the same exported program and with the same admission rules and
shared-expert conventions (`layers/adapter/fused_moe_decode.py`). With
`FUSED_MOE_PREFILL_KERNEL` unset those steps keep the GMM path. EP layers keep
the GMM path in either case for now: at GLM-5.2 shapes it was faster than
blocked mode at 4,096 tokens.

## When to use it

For the TP-expert layers the decode kernel serves, at steps too large for it.
Called directly, the kernel takes experts of any size: those whose 3
whole-expert slots do not fit VMEM (GLM-5.2 EP16) run blocked mode.

## Design

- **Routing plan in XLA.** Two sorts order the routes expert-major and give
  each route its slab row; no scatter or gather over the routes.
- **Expert loop.** The kernel visits the experts that received rows. Their
  weights stream through a 3-slot VMEM ring, and each tile's FP8 activation
  rows are gathered from HBM by one DMA per row, two tiles ahead. Each tile
  writes its unweighted BF16 rows into an expert-major slab in HBM. The shared
  expert is one more visit over every token, with contiguous rows.
- **Combine.** fused_moe/v2's destination combine: per token, the weighted
  sum of its routed rows and its shared row. It runs in chunks of at most 2048
  tokens, each followed by its own all-reduce, chained so each all-reduce runs
  under the next chunk's combine.
- **Blocked mode.** When 3 whole-expert slots do not fit VMEM (the decode
  kernel's rule, so it depends on expert size only), each expert's rows run in
  chunks of up to 512 with an FP32 running sum in VMEM, and the expert streams
  through a ring of intermediate blocks: 512 wide, 3 slots below 8192 tokens;
  1024 wide, 2 slots from 8192. The combine then reads only each token's live
  slots, since under EP most routes belong to other ranks. The kernel name
  ends in `-ib_<ib>-nb_<nb>-r_<rows>`.
- **Numerics.** Hidden states and the activated intermediate are quantized to
  FP8 with one scale per row per 512 columns, the block `gmm_v2` uses with the
  kernel off (in blocked mode the intermediate per row per block, in
  512-column pieces of wider blocks). Each block's partial product is scaled
  and summed in FP32; slab rows and the all-reduce are BF16. The hidden size
  must be a multiple of 512.

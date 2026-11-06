# NVFP4 Fused MoE Prefill Kernel (`fused_moe_prefill_nvfp4`)

NVFP4 counterpart of `fused_moe_prefill_kernel_v2` (Stage A) for prefill and
mixed-step MoE blocks whose routed weights are K-major `float4_e2m1fn` with
block `float32` scales and whose shared expert weights are per-channel
`float8_e4m3fn`.

Designed for **`EP2 x TP4` (`expert_start` primary)** and **`TP8`
(`expert_start=None` secondary)** on TPU v7x (`64 MiB` VMEM), sharing the exact
weight buffers of `moe_tp_decode_nvfp4` with zero weight duplication.

## API

```python
from vllm_torchtpu.kernels.experimental.fused_moe_prefill_nvfp4 import (
    can_use_fused_moe_prefill_nvfp4,
    fused_moe_prefill_nvfp4,
    fused_moe_prefill_nvfp4_stage_a,
    tall_rung_rows_for,
)

out = fused_moe_prefill_nvfp4(
    x,                        # [T, H] bf16, replicated across ranks
    w1, w2,                   # [E_local, H, 2*I] / [E_local, I, H] float4_e2m1fn
    w1_scale, w2_scale,       # [E_local, B13, 1, 2*I] / [E_local, B2, 1, H] f32
    shared_w1, shared_w1_scale,  # [H, 2*I_s] float8_e4m3fn / [2*I_s] f32
    shared_w2, shared_w2_scale,  # [I_s, H] float8_e4m3fn / [H] f32
    shared_gate_weight,       # [H, 1] bf16/f32, or None for ungated shared expert
    topk_weights, topk_ids,   # [T, K] bf16/f32, int32 (global expert IDs)
    axis_name="d",            # 8-rank mesh axis for combine + RS/AG
    expert_start=e0,          # int32 scalar (ep_rank * E_local) or None for TP
    routed_scale=2.5,         # scales routed experts only (never shared)
    act_mode="w4a16",         # "w4a16" (default, BF16) or "w4a8" (per-row FP8)
    tall_rung_rows="auto",    # default 4 rungs via tall_rung_rows_for
    include_shared=True,      # False omits shared-expert DMA/visit (routed-only)
    combine_chunk_tokens=2048,  # max tokens per combine + RS/AG chunk
)  # -> [T, H] bf16, all-reduced, identical on every rank
```

## Activation Modes

- **`act_mode="w4a16"` (Default / Primary)**: Gathers BF16 token rows into VMEM,
  widens each K-block of `float4_e2m1fn` weights to `bfloat16` in VMEM, runs
  `BF16 x BF16 -> F32` MXU dots with `F32` block-scale accumulation, and runs
  the shared-expert visit in online per-row `FP8` via `quantize_block` (matching
  `_tp_chain`'s `xla_quantized_matmul` numerics). Quality-neutral relative to
  the `gmm_v2` W4A16 path.
- **`act_mode="w4a8"` (Secondary)**: Quantizes token rows once in XLA via
  per-row `FP8` (`_quantize_rows`), gathers `FP8` rows (`0.5x` gather bytes),
  widens `float4_e2m1fn -> float8_e4m3fn` in VMEM, and runs `FP8 x FP8 -> F32`
  MXU dots (`2x` MXU rate) with per-row `FP8` SwiGLU requantization.

## Design Highlights

- **Whole-expert Stage-A ring (`nb = 4`) on `EP2 x TP4` and `TP8`**: Routed
  block scales stay 4D (`[E_local, B, 1, cols]`) across the `pallas_call` HBM
  operands and VMEM ring slots so the size-1 sublane-guard dimension (`T(1, 128)`)
  is preserved without an XLA boundary reshape+copy. At `GLM-5.2` `EP2 x TP4`
  (`E_local = 128, I = 512`), one NVFP4 expert + 4D scales budgets `5.06 MiB` in
  VMEM (`20.25 MiB` for `nb = 4`; `2.62 MiB/slot`, `10.50 MiB` on `TP8`), fitting
  comfortably within `64 MiB` VMEM (`36.70 / 48.45 MiB` in `W4A16` and
  `35.21 / 45.46 MiB` in `W4A8` at `tile_rows = 128 / 256` on `EP2 x TP4`;
  `26.95 / 38.70 MiB` in `W4A16` and `25.46 / 35.71 MiB` in `W4A8` on `TP8`, vs
  `62.72 MiB` `vmem_limit_bytes`). `EP8` (`I = 2048`, `20.25 MiB/slot`) is
  cleanly refused by `can_use_fused_moe_prefill_nvfp4`.
- **`w2_n_chunk = 2048` N-chunked down projection**: Replaces TP16-only
  `down="xmxu"` while preventing a full `[256, H] float32` accumulator from
  materializing in VMEM and writing `256`-column BF16 pairs directly into
  `out_vmem` via `pltpu.pack_elementwise`.
- **Rung selection (`tall_rung_rows_for`)**: Defaults to `4` rungs
  (`rung_rows = None`: `[32, 64, 96, 128]` at `tile_rows = 128`, and
  `[64, 128, 192, 256]` at `tile_rows = 256` for `T >= 4096`) across both
  `EP2 x TP4` and `TP8` and both `act_mode` values: `8` rungs
  (`rung_rows = 32`, `[32..256]`) at `256`-row tiles were measured much slower
  for `W4A16` (`EP2 x TP4` and `TP8`; `TP8`, `256`-row tiles: `509-546` vs
  `83-113 ms` per `75` layers) and no faster for `W4A8`, so `4` rungs are used
  at `256`-row tiles; `tall_rung_rows` still overrides (e.g.
  `tall_rung_rows=32`).
- **Single cross-rank reduction (`EP2 x TP4` & `TP8`)**: Under `EP2 x TP4`,
  `expert_start` filters non-local routes (`~50%` fewer live routed rows per
  rank), every rank computes `1/8` of the TP8-sharded `shared_experts`, and the
  single 8-rank chunked `reduce-scatter + all-gather` (`_psum_rows`) reduces
  EP2 disjoint expert groups, TP4 `I` partials, and TP8 shared-expert partials
  together (counting the shared expert once, with `routed_scale` applied only to
  routed experts).
- **Adapter integration (`FUSED_MOE_PREFILL_NVFP4_KERNEL`, default off)**: When
  enabled alongside `USE_MOE_TP_DECODE_NVFP4_KERNEL=1`,
  `layers/adapter/moe_tp_decode_nvfp4.py` dispatches `n > 64` buckets with
  `n >= FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS` (default `2048`; smaller `n > 64`
  buckets keep `_tp_chain` GMM) to `fused_moe_prefill_nvfp4_stage_a` inside the
  same `jax_op`. `FUSED_MOE_PREFILL_NVFP4_ACT_MODE` selects the activation path:
  `"w4a16"` (default, BF16 activations) or `"w4a8"` (per-row FP8 activations and
  `FP8 x FP8` MXU dots, e.g. when lowering
  `FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS=512`).

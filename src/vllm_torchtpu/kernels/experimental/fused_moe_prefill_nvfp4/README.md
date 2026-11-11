# FP4-block512 Fused MoE Prefill Kernel (W4A16 / W4A8; for NVFP4 Checkpoints)

The routed experts arrive as NVFP4 (FP4 e2m1, block-16 scales) and are
requantized once at load to FP4 with one `f32` scale per 512 elements
(`MOE_REQUANTIZE_BLOCK_SIZE=512`, clamped per matrix). The kernel
(`fused_moe_prefill_nvfp4`) computes in that FP4-block512 format, the same as it
would for an offline-converted or directly quantized block-512 checkpoint;
block 16 is not used (block-16 scales do not fit 4-chip HBM).

FP4-block512 counterpart of `fused_moe_prefill_kernel_v2` (Stage A) for prefill and
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
    fused_moe_prefill_nvfp4_unsupported_reason,
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
    tall_rung_rows=None,      # default 4 rungs; e.g. 32 overrides at 256-row tiles
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
- **Rung selection**: Defaults to `4` rungs (`tall_rung_rows=None`:
  `[32, 64, 96, 128]` at `tile_rows = 128`, and `[64, 128, 192, 256]` at
  `tile_rows = 256` for `T >= 4096`) across both `EP2 x TP4` and `TP8` and both
  `act_mode` values: `8` rungs (`tall_rung_rows=32`, `[32..256]`) at `256`-row
  tiles were measured much slower for `W4A16` (`EP2 x TP4` and `TP8`; `TP8`,
  `256`-row tiles: `509-546` vs `83-113 ms` per `75` layers) and no faster for
  `W4A8`, so `4` rungs are used at `256`-row tiles; `tall_rung_rows` still
  overrides (e.g. `tall_rung_rows=32`).
- **Single cross-rank reduction (`EP2 x TP4` & `TP8`)**: Under `EP2 x TP4`,
  `expert_start` filters non-local routes (`~50%` fewer live routed rows per
  rank), every rank computes `1/8` of the TP8-sharded `shared_experts`, and the
  single 8-rank chunked `reduce-scatter + all-gather` (`_psum_rows`) reduces
  EP2 disjoint expert groups, TP4 `I` partials, and TP8 shared-expert partials
  together (counting the shared expert once, with `routed_scale` applied only to
  routed experts).
- **Adapter integration (`FUSED_MOE_PREFILL_NVFP4_KERNEL`, default off)**: When
  enabled alongside `USE_MOE_TP_DECODE_NVFP4_KERNEL=1`,
  `layers/adapter/moe_tp_decode_nvfp4.py` dispatches `n > 64` buckets above the
  mode cutoff (`n >= 2048` for `"w4a16"`, `n >= 512` for `"w4a8"`; smaller
  `n > 64` buckets keep `_tp_chain` GMM) to `fused_moe_prefill_nvfp4_stage_a`
  inside the same `jax_op`, and logs once per layer shape if prefill admission
  is refused. `FUSED_MOE_PREFILL_NVFP4_ACT_MODE` selects the activation path:
  `"w4a16"` (default, BF16 activations) or `"w4a8"` (per-row FP8 activations and
  `FP8 x FP8` MXU dots).

## Per-layer latency

8-rank TPU7x per-layer benchmark at the branch tip, GLM-5.2 shapes (`H = 6144`,
`E = 256`, `topk = 8`), full MoE layer (routing + Stage-A + combine), ms per
step across 75 layers:

EP2xTP4 (uniform / skewed routing):

| Tokens | Kernel W4A16 | Kernel W4A8 | gmm_v2 W4A16 | gmm_v2 W4A8 |
| --: | :-- | :-- | :-- | :-- |
| 512 | 50.9 / 50.8 | 32.2 / 32.5 | 46.7 / 47.5 | 48.8 / 49.5 |
| 1024 | 59.3 / 59.6 | 41.5 / 41.7 | 58.9 / 60.9 | 60.5 / 61.6 |
| 2048 | 77.3 / 77.4 | 60.1 / 61.9 | 85.1 / 86.6 | 80.3 / 81.3 |
| 4096 | 103.8 / 109.3 | 93.5 / 97.8 | 135.4 / 142.5 | 130.2 / 135.1 |

TP8 (uniform routing; `#1559` / `#1935` FP8 Stage-A timing reference):

| Tokens | Kernel W4A16 | Kernel W4A8 | FP8 Stage-A reference |
| --: | :-- | :-- | :-- |
| 512 | 55.0 | 38.8 | 42.4 |
| 1024 | 64.2 | 48.3 | 52.2 |
| 2048 | 83.8 | 69.7 | 71.5 |
| 4096 | 113.0 | 111.1 | 112.5 |

Under `EP2 x TP4`, `w4a16` is slower than `gmm_v2` at `512` tokens and ties at
`1024`, so the adapter cutoff is `2048` tokens for `w4a16`; `w4a8` is faster
than `gmm_v2` from `512` tokens up, so its cutoff is `512` tokens. `8` rungs at
`256`-row tiles were measured much slower for `W4A16` (`EP2 x TP4` and `TP8`)
and no faster for `W4A8`, so `4` rungs are used at `256`-row tiles
(`tall_rung_rows` still overrides).

TPU-only cases of `tests/kernels/test_fused_moe_prefill_nvfp4.py` passed twice
across two 8-rank TPU7x pods at the branch tip (`12` passed + `4` by-design
parametrization skips, `0` failed), and all CPU interpret-mode and
adapter-dispatch tests pass on CPU.

## End-to-end model quality

Measured on 4-chip TPU7x (`EP2 x TP4`) for GLM-5.2-NVFP4: on GSM8K 5-shot,
`n = 1,319`, `c8`, real MTP (`k = 3`), exact match is `0.9712 ± 0.0046` (`3.577`
MTP acceptance) with `w4a16` (`>= 2048`) and `0.9689 ± 0.0048` (`3.578` MTP
acceptance) with `w4a8` (`>= 512`, plus online-FP8 drafter experts), vs
`0.9689 ± 0.0048` (`3.585` MTP acceptance) with the prefill kernel off
(`_tp_chain` `gmm_v2` W4A16). On LongBench-v2, `503` items, paired against the
prefill kernel off (`46.5%`, `234 / 503`; `c8`, thinking off, `temperature = 0`,
`256` new tokens, `32K` middle truncation), `w4a16` (`>= 2048`) measures
`-0.2 pp` (`46.3%`, 95% CI `-2.4..+2.0 pp`, McNemar `p = 1.0`, `90.7%`
agreement; `-0.6 pp` on the pre-registered `n = 180` short subset); for `w4a8`
(`>= 512`), the pre-registered `n = 180` short-subset reading was `-2.8 pp`
(`96` vs `101` of `180`, `INVESTIGATE` band) and its prescribed `503`-item
follow-up gave `+0.6 pp` (`47.1%`, 95% CI `-1.8..+3.0 pp`, McNemar `p = 0.74`,
`88.1%` agreement; per-length correct counts `w4a8` vs off: short `96` vs `101`
[`n = 180`], medium `84` vs `85` [`n = 215`], long `57` vs `48` [`n = 108`]).

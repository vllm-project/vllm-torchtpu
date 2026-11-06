# NVFP4 W4A8 Fused TP Decode MoE Kernel

One Pallas call per rank that runs an NVFP4 MoE layer's router top-k, its
online-FP8 shared expert, its W4A8 block-scaled routed experts, and the
in-kernel cross-rank all-reduce for decode batches (`n <= 64`). Opt-in with
`USE_MOE_TP_DECODE_NVFP4_KERNEL=1` (default `0`) and
`TPU_MOE_SKIP_PADDED_TOKENS=1`.

## API

```python
from vllm_torchtpu.kernels.experimental.moe_tp_decode_nvfp4 import (
    ROW_BUCKETS,
    can_use_moe_tp_decode_nvfp4,
    exact_in_bf16,
    group_size,
    kernel_name,
    moe_tp_decode_nvfp4,
)

# Signature (call inside jax.shard_map over an n_ranks-device 1D mesh):
def moe_tp_decode_nvfp4(
    x,                        # [m, H] bf16, m in ROW_BUCKETS = (16, 32, 64), replicated
    scores,                   # [m, E] f32 post-sigmoid router scores, replicated
    bias,                     # [E] f32 router e_score_correction_bias, replicated
    n_valid,                  # int32[], valid token count in [0, m]
    rank,                     # int32[], this rank's mesh index in [0, n_ranks)
    w13,                      # [E_local, H, 2 * I_rank] float4_e2m1fn (K-major)
    w2,                       # [E_local, I_rank, H] float4_e2m1fn (K-major)
    s13,                      # [E_local, H // b13, 1, 2 * I_rank] f32 block scales
    s2,                       # [E_local, I_rank // b2, 1, H] f32 block scales
    shared_w13,               # [H, 2 * I_shared] float8_e4m3fn
    shared_s13,               # [2 * I_shared] f32 per-channel scales
    shared_w2,                # [I_shared, H] float8_e4m3fn
    shared_s2,                # [H] f32 per-channel scales
    experts_start=0,          # int32[], first global expert id on this rank (EP x TP)
    *,
    routed_scale: float,      # required; bf16-exact (e.g. 2.5 for GLM-5.2)
    topk: int,                # required; experts selected per token
    n_ranks: int,             # required; mesh size (TP or EP * TP)
    fast: bool = False,
    hbm_operands: bool = False,
    split_meta: bool = True,
    w2_n_chunk: int = 2048,
):  # -> [m, H] bf16, all-reduced across n_ranks, rows >= n_valid are +0.0
    ...
```

- Call `moe_tp_decode_nvfp4` inside `jax.shard_map` over an `n_ranks`-device
  1D mesh, one call per rank.
- `can_use_moe_tp_decode_nvfp4(..., *, topk, n_ranks, w2_n_chunk=2048)`
  validates TPU generation (TPU7x FP8 MXU), dtypes, divisibility, `b13 == 512`,
  `b2 in (128, 256, 512)` with `I_rank % b2 == 0`, `w2_n_chunk` lane alignment,
  and VMEM budget (`need_bytes * 1.25 <= 0.875 * vmem_capacity_bytes`);
  `exact_in_bf16(routed_scale)` checks `routed_scale` before calling the kernel.
- `group_size(I_rank, E_local)` selects the number of active experts batched
  into each gate/up K-block loop
  (`min(4, E_local, max(1, 1024 // (2 * I_rank)))`: `g = 2` when
  `I_rank == 256`, `g = 1` when `I_rank >= 512`), keeping
  `g * 2 * I_rank <= 1024` and the 4-slot weight ring within
  `20.25–21.00 MiB` of VMEM (`35.03–44.73 MiB` total VMEM limit across
  `m in (16, 32, 64)` vs the `56.00 MiB` budget cap).

## When it runs and flags

`layers/adapter/moe_tp_decode_nvfp4.py` arms eligible `VllmNvfp4MoEMethod`
layers after model load when `USE_MOE_TP_DECODE_NVFP4_KERNEL=1` and
`TPU_MOE_SKIP_PADDED_TOKENS=1`:

- **Decode (`n <= 64`):** `n in (16, 32, 64)` calls `moe_tp_decode_nvfp4`
  directly; `n < 16` pads `x` and `scores` to `m = 16`, runs the `m = 16`
  bucket, and slices `[:n]`.
- **Mixed / prefill (`n > 64`):** calls `moe_tp_decode._tp_chain`
  (`fused_moe_func` -> `gmm_v2` W4A16 + online-FP8 shared expert + `lax.psum`)
  inside the same custom op on the same K-major FP4 weights and 4D block scales,
  with zero weight duplication.
- **Drafter / non-NVFP4 layers:** layers that do not use `VllmNvfp4MoEMethod`
  (such as GLM-5.2's BF16 MTP drafter layer 78) are refused by `_admit` and
  keep their existing execution path. When `USE_MOE_TP_DECODE_NVFP4_KERNEL=0`
  (the default), weight loading, custom-op dispatch, and HLO renders are
  unchanged. Although this kernel quantizes routed-expert decode activations to
  FP8 (W4A8) where `_tp_chain` (`gmm_v2`) keeps BF16 activations (W4A16), 5-shot
  GSM8K accuracy and real MTP acceptance match the `gmm_v2` W4A16 fallback (see
  [End-to-end model quality](#end-to-end-model-quality) below); the flag stays
  off (`0`) by default.

| Environment variable | Default | Effect |
| :--- | :--- | :--- |
| `USE_MOE_TP_DECODE_NVFP4_KERNEL` | `0` | Enables per-matrix K-major FP4 block requantization at load time and arms the fused NVFP4 TP decode op. |
| `TPU_MOE_SKIP_PADDED_TOKENS` | `0` | Required (`=1`) so `token_padding` provides the valid-token mask consumed as `n_valid`. |
| `MOE_REQUANTIZE_BLOCK_SIZE` | `512` (when kernel flag is `1`) | Target K-block size; clamped per matrix to `min(block, K_matrix)` when the fused EP W4A8 kernel is not used. |
| `TPU_MOE_TP_DECODE_FAST` | `1` | Uses packed-key top-k routing, vector active-expert compaction, and peeled 4-slot ring scheduling (bitwise identical to `0`). |
| `TPU_MOE_TP_DECODE_HBM_OPERANDS` | `drafter` | Pins large weight operands to HBM on armed layers (`''`, `'0'`, `'drafter'`, or `'all'`). |
| `ENABLE_EXPERT_PARALLEL` / `TPU_MOE_EP_TP_SIZE` | `false` / unset | Configures `EP x TP` sharding (`ep_size * tp_size == n_ranks`, e.g. `EP2xTP4` with `TPU_MOE_EP_TP_SIZE=4`). |

## Flag precedence

When `USE_MOE_TP_DECODE_NVFP4_KERNEL=1` and `TPU_MOE_SKIP_PADDED_TOKENS=1` are
set alongside other MoE kernel flags, admission is partitioned per layer by
quantization method (`VllmNvfp4MoEMethod` for NVFP4 layers,
`VllmFp8MoEMethodTPU` for FP8 layers, `VllmUnquantizedFusedMoEMethod` for BF16
layers):

| Co-enabled flag(s) | NVFP4 layers (`n <= 64`) | NVFP4 layers (`n > 64`) | FP8 layers (`n <= 64`) | FP8 layers (`n > 64`) | BF16 layers (all `n`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| *(none)* | `moe_tp_decode_nvfp4` | `_tp_chain` (`gmm_v2` W4A16 + `psum`) | Unarmed (`fused_moe_gmm`) | Unarmed (`fused_moe_gmm`) | Unarmed (`fused_moe_gmm`) |
| `USE_MOE_TP_DECODE_KERNEL=1` | `moe_tp_decode_nvfp4` | `_tp_chain` (`gmm_v2` W4A16 + `psum`) | `moe_tp_decode` | `_tp_chain` (`n <= 256` or EP); `fused_moe_prefill` (`n > 256`, TP, prefill on) | Unarmed (`fused_moe_gmm`) |
| `FUSED_MOE_DECODE_KERNEL=1` *(with or without `USE_MOE_TP_DECODE_KERNEL=1`)* | `moe_tp_decode_nvfp4` | `_tp_chain` (`gmm_v2` W4A16 + `psum`) | `fused_moe_decode` | `fused_moe_decode` (`n <= 256`); `fused_moe_prefill` (`n > 256`, TP, prefill on) or FP8 GMM | Unarmed (`fused_moe_gmm`) |
| `FUSED_MOE_PREFILL_KERNEL=1` *(default `1`, no FP8 decode flag)* | `moe_tp_decode_nvfp4` | `_tp_chain` (`gmm_v2` W4A16 + `psum`) | Unarmed (`fused_moe_gmm`) | Unarmed (`fused_moe_gmm`) | Unarmed (`fused_moe_gmm`) |
| `USE_MOE_FUSED_EP_KERNEL=1` *(pure EP admitted; `+ MOE_FUSED_EP_ENABLE_W4A8=1` or `adaptive` on NVFP4)* | `fused_moe_ep` / `adaptive_fused_moe` | `fused_moe_ep` / `adaptive_fused_moe` | `fused_moe_ep` / `adaptive_fused_moe` | `fused_moe_ep` / `adaptive_fused_moe` | Unarmed (`fused_moe_gmm`) |
| `USE_MOE_FUSED_EP_KERNEL=1` *(`TP` or `EP x TP` with `tp > 1`)* | `moe_tp_decode_nvfp4` | `_tp_chain` (`gmm_v2` W4A16 + `psum`) | `moe_tp_decode` if `USE_MOE_TP_DECODE_KERNEL=1`, else unarmed | `_tp_chain` / `fused_moe_prefill` if `USE_MOE_TP_DECODE_KERNEL=1`, else unarmed | Unarmed (`fused_moe_gmm`) |

- **No cross-dtype stand-down:** FP8-only flags (`USE_MOE_TP_DECODE_KERNEL`,
  `FUSED_MOE_DECODE_KERNEL`, `FUSED_MOE_PREFILL_KERNEL`,
  `TPU_MOE_FP8_ALL_GATHER`) neither arm nor disable NVFP4 layers; `256` above
  is `FUSED_MOE_DECODE_KERNEL_MAX_TOKENS`.
- **NVFP4 stand-downs:** `moe_tp_decode_nvfp4` yields only when another kernel
  or collective wrapper serves the NVFP4 layer—an admitted fused-EP op
  (`USE_MOE_FUSED_EP_KERNEL=1` with `MOE_FUSED_EP_ENABLE_W4A8=1`, or
  `MOE_FUSED_EP_KERNEL_IMPL=adaptive`) or collective chunking
  (`TPU_MOE_COLLECTION_CHUNK_SIZE > 0`).
- **Sub-bucket decode counts (`n <= 64`):** on armed NVFP4 layers, `n in (16,
  32, 64)` invokes `moe_tp_decode_nvfp4` directly, `n < 16` pads to `16` and
  slices `[:n]`, and any non-bucket `16 < n < 64` falls back to `_tp_chain`.

## Supported layouts (`TP8` and `EP2xTP4`)

When `USE_MOE_TP_DECODE_NVFP4_KERNEL=1`, `nvfp4._prepare_moe_weights` clamps
the requantization block size per matrix (`b13 = min(block, H)` and
`b2 = min(block, I_rank)`) and uses `requant_load_kmajor_fp4` without padding
`I_rank` up to `512`. For GLM-5.2 (`H = 6144, E = 256, I_tp8 = 256,
I_shared_tp8 = 256`, 8 TPU7x ranks):

| Layout | Sharding | Per-rank experts (`E_local`) | `I_rank` | Requant blocks (`b13 / b2`) | Group `g` | Ring slot (4-slot ring) | VMEM need (`m = 16 / 32 / 64`) | VMEM limit (`m = 16 / 32 / 64`) |
| :--- | :--- | ---: | ---: | :--- | ---: | ---: | :--- | :--- |
| `TP8` | `ep=1, tp=8` | `256` (`e0 = 0`) | `256` | `512 / 256` (`12` / `1` blocks) | `2` | `5.25 MiB` (`21.00 MiB`) | `28.77 / 30.98 / 35.78 MiB` | `35.97 / 38.73 / 44.73 MiB` |
| `EP2xTP4` | `ep=2, tp=4` | `128` (`e0 in {0, 128}`) | `512` | `512 / 512` (`12` / `1` blocks) | `1` | `5.06 MiB` (`20.25 MiB`) | `28.02 / 30.23 / 35.03 MiB` | `35.03 / 37.79 / 43.79 MiB` |

`s13` and `s2` stay 4D (`[E_local, B, 1, cols]`) across the `pallas_call` HBM
operands and VMEM scratch slots so the size-1 sublane-guard dimension is
preserved without an XLA `T(1, 128) -> T(8, 128)` reshape+copy at the
custom-call boundary. At `TP8`, clamping `b2` to `256` avoids padding `I_rank`
from `256` to `512`, which would otherwise double routed-expert HBM on 4-chip
hosts.

## Numerics

- **Routing:** matches `moe_routing` (`sigmoid` scores + `bias` for top-k
  selection with lowest-column tie-breaking, unbiased picked scores normalized
  by `max(sum, 1e-20)` and cast to `bf16`; rows `>= n_valid` masked to `0`).
- **Shared expert:** input `x` and intermediate `silu(gate) * up` are quantized
  per row to `float8_e4m3fn` via `quantize_block` and multiplied by `sw13` /
  `sw2` with `preferred_element_type=f32`, matching `moe_tp_decode._shared_expert`.
- **Routed experts (W4A8):**
  1. `x` is quantized per row per 512-column K-block in `f32`
     (`xs = max(|x_block|) / 448.0`, `xq = (x_block * (1.0 / xs)).astype(fp8)`).
  2. Each active expert group's K-major `float4_e2m1fn` weights are widened in
     VMEM to exact `float8_e4m3fn` immediately before the FP8 MXU matmul.
  3. Gate/up K-block partials are scaled and accumulated in `f32`:
     `acc1 += (matmul(xq_b, w13_b) * xs_b) * s13_b`.
  4. In `finish_group`, `act = silu(gate) * up` is computed in `f32`, multiplied
     by the token's `f32` routing weight `coeff`, quantized per row to
     `float8_e4m3fn` (`ms = max(|act|) / 448.0`,
     `mq = (act * (1.0 / ms)).astype(fp8)`), and multiplied by `w2` across
     `b2`-sized K-blocks and `w2_n_chunk` N-tiles:
     `owner += (matmul(mq_b, w2_b) * ms) * s2_b`.
  5. Combine and all-reduce: `bf16(bf16(shared) + bf16(bf16(owner) * routed_scale))`
     is summed across all `n_ranks` in `f32` and rounded to `bf16` inside
     `moe_tp_decode.all_reduce`, with rows `>= n_valid` zeroed to `+0.0`.

### Note on CPU interpret mode vs TPU7x hardware lowering

In 8-rank Pallas CPU interpret mode (`pltpu.force_tpu_interpret_mode` with
`--xla_allow_excess_precision=false`), every intermediate stage and the final
output of `moe_tp_decode_nvfp4` match the pure-JAX W4A8 oracle (`exp_body`) to
**0 ULP** (`expected_rel_l2 = 0.0`), while the measured spread between two
valid W4A8 oracle rounding/accumulation orderings on CPU is `0.0295–0.0436`.

On 8-rank TPU7x hardware (measured), across the per-layer benchmark's `25`
correctness cases (`n_valid = 0..64`, uniform/concentrated/tie routing), `16`
timed cells (`39` non-empty cells), and `4` `all_reduce` cases, measured
`rel_l2(kernel, exp_body)` is `0.0119–0.0161` (within the `0.0388–0.0434`
hardware spread between two valid W4A8 oracle orderings) while both the kernel
and `exp_body` have virtually identical distance to the dequantized `f32`
reference (`rel_l2 = 0.0438–0.0468`,
`|rel_l2(kernel, f32) - rel_l2(exp_body, f32)| <= 0.0005`,
`cos = 0.9989–0.9990`), and all hardware bitwise (`fast` vs `slow`,
`hbm_operands`, `split_meta`, and `w2_n_chunk`), rank-identity, `+0.0` padding,
and `NaN=0` checks matched to 0 ULP. Comparing the Mosaic MLIR against XLA:TPU
StableHLO shows two lowering differences immediately before
`.astype(float8_e4m3fn)`:

1. In `quantize_block`, Mosaic lowers `x / broadcast(scale)` to
   `vector.broadcast` + `arith.divf`, whereas XLA's `AlgebraicSimplifier`
   rewrites `divide(x, broadcast(scale))` to `multiply(x, broadcast(1.0 / scale))`.
2. In SwiGLU, Mosaic lowers `jax.nn.silu(x)` to `x * tpu.logistic(x)`, whereas
   XLA lowers `@silu` to `x * (1.0 / (1.0 + exp(-x)))`.

Because `float8_e4m3fn` has only 3 mantissa bits (`1 ULP = 12.5%` relative
step), a 1-ULP `f32`/`bf16` difference prior to `.astype(float8_e4m3fn)` flips
boundary elements into adjacent FP8 quantization buckets before the subsequent
MXU matmuls.

## Measured per-layer latency

8-rank TPU7x per-layer benchmark at the branch tip (measured), GLM-5.2 shapes
(`H = 6144`, `E = 256`, `topk = 8`, `I_shared_rank = 256`, seed `2026`; full MoE
layer in `us/layer [x75 ms]`, in-graph `lax.fori_loop` with calibrated loop
overhead subtracted; `n = 4, 8` run via `pad_to_m = 16`):

### Uniform routing (`us/layer [x75 ms]`)

| Layout | `n` | `moe_tp_decode_nvfp4` (`g=1/g=2, nb=4`) | FP8 TP decode (`#1763`) | `gmm_v2` W4A16 |
| :--- | ---: | ---: | ---: | ---: |
| `EP2xTP4` (`b512/b512`) | `4` | `42.96 [3.22]` | `65.17 [4.89]` | `65.22 [4.89]` |
| `EP2xTP4` (`b512/b512`) | `8` | `59.71 [4.48]` | `101.16 [7.59]` | `102.23 [7.67]` |
| `EP2xTP4` (`b512/b512`) | `16` | `90.83 [6.81]` | `163.64 [12.27]` | `156.61 [11.75]` |
| `EP2xTP4` (`b512/b512`) | `32` | `139.46 [10.46]` | `252.19 [18.91]` | `231.21 [17.34]` |
| `EP2xTP4` (`b512/b512`) | `64` | `203.77 [15.28]` | `319.91 [23.99]` | `294.99 [22.12]` |
| `TP8` (`b512/b256`) | `4` | `40.77 [3.06]` | `61.00 [4.57]` | `93.49 [7.01]` |
| `TP8` (`b512/b256`) | `8` | `59.04 [4.43]` | `98.74 [7.41]` | `151.27 [11.35]` |
| `TP8` (`b512/b256`) | `16` | `85.45 [6.41]` | `152.43 [11.43]` | `214.54 [16.09]` |
| `TP8` (`b512/b256`) | `32` | `136.83 [10.26]` | `238.76 [17.91]` | `332.86 [24.96]` |
| `TP8` (`b512/b256`) | `64` | `214.14 [16.06]` | `309.17 [23.19]` | `436.90 [32.77]` |

### Concentrated routing (`us/layer`)

| Layout | `n` | `moe_tp_decode_nvfp4` (`g=1/g=2, nb=4`) | FP8 TP decode (`#1763`) | `gmm_v2` W4A16 |
| :--- | ---: | ---: | ---: | ---: |
| `EP2xTP4` (`b512/b512`) | `4` | `32.50` | `42.54` | `42.55` |
| `EP2xTP4` (`b512/b512`) | `8` | `32.17` | `42.70` | `52.69` |
| `EP2xTP4` (`b512/b512`) | `16` | `32.04` | `41.96` | `53.66` |
| `TP8` (`b512/b256`) | `4` | `26.46` | `31.22` | `56.14` |
| `TP8` (`b512/b256`) | `8` | `26.80` | `31.41` | `64.43` |
| `TP8` (`b512/b256`) | `16` | `25.97` | `30.56` | `66.98` |

- **FP8 reference caveat:** The `#1763` FP8 TP decode kernel arm runs on the
  same synthetic weights requantized to per-channel FP8, so it is a timing
  reference, not an accuracy reference.
- **Schedule notes:** The 4-slot weight ring with `g = 1` (`EP2xTP4`) / `g = 2`
  (`TP8`) fits `m = 64` at `43.79–44.73 MiB` against the `56.00 MiB` VMEM budget
  cap; `w2_n_chunk=2048` and `w2_n_chunk=6144` stay within `0.02–1.02 us` across
  all uniform token counts.

## End-to-end model quality

On GSM8K 5-shot, `n = 1,319`, `c8`, real MTP (`k = 3`), measured on 4-chip
TPU7x (`EP2xTP4`) for GLM-5.2-NVFP4 with `moe_tp_decode_nvfp4` (W4A8) on decode
and `_tp_chain` (`gmm_v2` W4A16) on prefill, exact match is `0.9689 ± 0.0048`
with `3.585` mean MTP acceptance, matching the 4-chip `EP8` `gmm_v2` W4A16
fallback (`0.9682 ± 0.0048`, `3.594` MTP acceptance) and 8-chip FP8
(`0.9682 ± 0.0048`, `3.541` MTP acceptance).

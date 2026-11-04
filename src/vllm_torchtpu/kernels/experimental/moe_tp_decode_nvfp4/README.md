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
    group_size,
    kernel_name,
    moe_tp_decode_nvfp4,
    vmem_bytes,
)

out = moe_tp_decode_nvfp4(
    x,                    # [m, H] bf16, m in ROW_BUCKETS = (16, 32, 64), replicated
    scores,               # [m, E] f32 post-sigmoid router scores, replicated
    bias,                 # [E] f32 router e_score_correction_bias, replicated
    n_valid,              # int32[], valid token count in [0, m]
    rank,                 # int32[], this rank's mesh index in [0, n_ranks)
    w13,                  # [E_local, H, 2 * I_rank] float4_e2m1fn (K-major)
    w2,                   # [E_local, I_rank, H] float4_e2m1fn (K-major)
    s13,                  # [E_local, H // b13, 1, 2 * I_rank] f32 block scales
    s2,                   # [E_local, I_rank // b2, 1, H] f32 block scales
    sw13,                 # [H, 2 * I_shared] float8_e4m3fn
    ss13,                 # [2 * I_shared] f32 per-channel scales
    sw2,                  # [I_shared, H] float8_e4m3fn
    ss2,                  # [H] f32 per-channel scales
    experts_start=None,   # int32[], first global expert id on this rank (EP x TP)
    *,
    routed_scale=1.0,     # bf16-exact factor multiplying routed sum (GLM-5.2: 2.5)
    topk=8,
    n_ranks=8,
    router_bias=True,
    fast=True,
    hbm_operands=False,
    split_meta=True,
    w2_n_chunk=2048,
)  # -> [m, H] bf16, all-reduced across n_ranks, rows >= n_valid are +0.0
```

- Call `moe_tp_decode_nvfp4` inside `jax.shard_map` over an `n_ranks`-device
  1D mesh, one call per rank.
- `can_use_moe_tp_decode_nvfp4(...)` validates TPU generation (TPU7x FP8 MXU),
  dtypes, divisibility, `b13 == 512`, `b2 in (128, 256, 512)` with
  `I_rank % b2 == 0`, `routed_scale` exact in `bf16`, and VMEM budget
  (`vmem_bytes(...) * 1.25 <= 0.75 * vmem_capacity_bytes`).
- `group_size(I_rank, E_local)` selects the number of active experts batched
  into each gate/up K-block loop (`g = 4` when `I_rank <= 256`, `g = 2` when
  `I_rank > 256`), keeping `g * 2 * I_rank == 2048` and the 3-slot weight ring
  within ~13.5 MiB of VMEM.

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
  unchanged. Because this kernel quantizes routed-expert activations to FP8
  (W4A8) where today's NVFP4 path (`gmm_v2`) keeps BF16 activations (W4A16),
  end-to-end model quality (GSM8K and MTP acceptance) is not yet validated, and
  the flag stays off by default until it is.

| Environment variable | Default | Effect |
| :--- | :--- | :--- |
| `USE_MOE_TP_DECODE_NVFP4_KERNEL` | `0` | Enables per-matrix K-major FP4 block requantization at load time and arms the fused NVFP4 TP decode op. |
| `TPU_MOE_SKIP_PADDED_TOKENS` | `0` | Required (`=1`) so `token_padding` provides the valid-token mask consumed as `n_valid`. |
| `MOE_REQUANTIZE_BLOCK_SIZE` | `512` (when kernel flag is `1`) | Target K-block size; clamped per matrix to `min(block, K_matrix)` when the fused EP W4A8 kernel is not used. |
| `TPU_MOE_TP_DECODE_FAST` | `1` | Uses packed-key top-k routing, vector active-expert compaction, and peeled 3-slot ring scheduling (bitwise identical to `0`). |
| `TPU_MOE_TP_DECODE_HBM_OPERANDS` | `drafter` | Pins large weight operands to HBM on armed layers (`0`, `drafter`, or `1`). |
| `ENABLE_EXPERT_PARALLEL` / `TPU_MOE_EP_TP_SIZE` | `false` / unset | Configures `EP x TP` sharding (`ep_size * tp_size == n_ranks`, e.g. `EP2xTP4` with `TPU_MOE_EP_TP_SIZE=4`). |

## Supported layouts (`TP8` and `EP2xTP4`)

When `USE_MOE_TP_DECODE_NVFP4_KERNEL=1`, `nvfp4._prepare_moe_weights` clamps
the requantization block size per matrix (`b13 = min(block, H)` and
`b2 = min(block, I_rank)`) and uses `requant_load_kmajor_fp4` without padding
`I_rank` up to `512`. For GLM-5.2 (`H = 6144, E = 256, I_tp8 = 256,
I_shared_tp8 = 256`, 8 TPU7x ranks):

| Layout | Sharding | Per-rank experts (`E_local`) | `I_rank` | Requant blocks (`b13 / b2`) | Group `g` | Ring slot (3-slot ring) | VMEM (`m = 16 / 32 / 64`) |
| :--- | :--- | ---: | ---: | :--- | ---: | ---: | :--- |
| `TP8` | `ep=1, tp=8` | `256` (`e0 = 0`) | `256` | `512 / 256` (`12` / `1` blocks) | `4` | `2.25 MiB` (`6.75 MiB`) | `37.40 / 39.61 / 44.41 MiB` |
| `EP2xTP4` | `ep=2, tp=4` | `128` (`e0 in {0, 128}`) | `512` | `512 / 512` (`12` / `1` blocks) | `2` | `4.50 MiB` (`13.50 MiB`) | `36.27 / 38.48 / 43.28 MiB` |

At `TP8`, clamping `b2` to `256` avoids padding `I_rank` from `256` to `512`,
which would otherwise double routed-expert HBM on 4-chip hosts.

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

On TPU7x hardware across the 68 single-host screening correctness cases,
measured `rel_l2(kernel, exp_body)` is `0.0087–0.0190` while both the kernel
and `exp_body` have virtually identical distance to the dequantized `f32`
reference (`rel_l2 = 0.0437–0.0515`,
`|rel_l2(kernel, f32) - rel_l2(exp_body, f32)| <= 0.0007`,
`cos = 0.9987–0.9990`), and all bitwise (`fast` vs `slow`, `hbm_operands`,
`split_meta`, `w2_n_chunk=2048` vs `6144`), rank-identity, `+0.0` padding, and
`NaN=0` checks matched on hardware. Comparing the Mosaic MLIR against XLA:TPU
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

## Measured per-layer latency (single-host kernel screening)

Measured on an 8-rank TPU7x single-host kernel screening benchmark at GLM-5.2
per-rank shapes (`H = 6144`, `E = 256`, `topk = 8`, `I_shared_rank = 256`,
in-graph `lax.fori_loop` with calibrated loop overhead subtracted; `n = 4` runs
via `pad_to_m = 16`):

| Decode tokens `n` | Active experts (`TP8` / `EP2`) | Arm | Full layer (`us`) | 75 layers (`ms`) | Routed loop (`us`) | Routed 75x (`ms`) | Speedup vs `#1763` FP8 (full / loop) |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| `4` | `30 / 17` | `fp8_1763_tp8` (`#1763` FP8) | `59.83` | `4.49` | `42.15` | `3.16` | `1.00x / 1.00x` |
| `4` | `30 / 17` | `nvfp4_tp8_fast` (`b512/b256, g=4`) | `45.97` | `3.45` | `28.13` | `2.11` | `1.30x / 1.50x` |
| `4` | `30 / 17` | `nvfp4_ep2tp4_fast` (`b512/b512, g=2`) | `48.03` | `3.60` | `30.45` | `2.28` | `1.25x / 1.38x` |
| `4` | `30 / 17` | `r1_fp4_g4` (standalone loop) | `n/a` | `n/a` | `32.54` | `2.44` | `n/a / 1.30x` |
| `16` | `94 / 51` | `fp8_1763_tp8` (`#1763` FP8) | `147.51` | `11.06` | `130.49` | `9.79` | `1.00x / 1.00x` |
| `16` | `94 / 51` | `nvfp4_tp8_fast` (`b512/b256, g=4`) | `95.43` | `7.16` | `79.45` | `5.96` | `1.55x / 1.64x` |
| `16` | `94 / 51` | `nvfp4_ep2tp4_fast` (`b512/b512, g=2`) | `101.21` | `7.59` | `85.28` | `6.40` | `1.46x / 1.53x` |
| `16` | `94 / 51` | `r1_fp4_g4` (standalone loop) | `n/a` | `n/a` | `80.38` | `6.03` | `n/a / 1.62x` |
| `32` | `153 / 81` | `fp8_1763_tp8` (`#1763` FP8) | `231.97` | `17.40` | `210.57` | `15.79` | `1.00x / 1.00x` |
| `32` | `153 / 81` | `nvfp4_tp8_fast` (`b512/b256, g=4`) | `149.28` | `11.20` | `129.04` | `9.68` | `1.55x / 1.63x` |
| `32` | `153 / 81` | `nvfp4_ep2tp4_fast` (`b512/b512, g=2`) | `154.75` | `11.61` | `134.64` | `10.10` | `1.50x / 1.56x` |
| `32` | `153 / 81` | `r1_fp4_g4` (standalone loop) | `n/a` | `n/a` | `127.70` | `9.58` | `n/a / 1.65x` |
| `64` | `197 / 104` | `fp8_1763_tp8` (`#1763` FP8) | `301.85` | `22.64` | `272.25` | `20.42` | `1.00x / 1.00x` |
| `64` | `197 / 104` | `nvfp4_tp8_fast` (`b512/b256, g=4`) | `209.43` | `15.71` | `178.09` | `13.36` | `1.44x / 1.53x` |
| `64` | `197 / 104` | `nvfp4_ep2tp4_fast` (`b512/b512, g=2`) | `207.91` | `15.59` | `176.63` | `13.25` | `1.45x / 1.54x` |
| `64` | `197 / 104` | `r1_fp4_g4` (standalone loop) | `n/a` | `n/a` | `182.78` | `13.71` | `n/a / 1.49x` |

`w2_n_chunk=2048` and `w2_n_chunk=6144` produce bitwise-identical outputs and
are within `0.5 us` across all token counts.

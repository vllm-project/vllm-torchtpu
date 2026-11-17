# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 fused MoE prefill kernel (W4A16 primary and W4A8 secondary arm).

Ports the Stage-A fused MoE prefill pipeline of
``fused_moe_prefill_kernel_v2`` to K-major ``float4_e2m1fn`` routed weights with
block ``float32`` scales and TP-sharded per-channel ``float8_e4m3fn`` shared
expert weights, matching the exact weight buffers of ``moe_tp_decode_nvfp4``:

  1. Routed ``w1 [E_local, H, 2*I]`` and ``w2 [E_local, I, H]``
     (``float4_e2m1fn``) are bitcast in HBM to ``uint32 [E_local, H // 8, 2*I]``
     and ``uint32 [E_local, I // 8, H]`` (8 K-rows per 32-bit word) and streamed
     whole through an ``nb``-slot VMEM ring (``nb = 4`` in ``STAGE_A_NVFP4``).
  2. ``act_mode = "w4a16"`` (default): BF16 activations are gathered per row into
     VMEM, each ``qb1`` / ``bk2`` K-block of ``float4_e2m1fn`` weights is widened
     in VMEM to ``bfloat16`` for ``BF16 x BF16 -> F32`` MXU dots with ``F32``
     block-scale accumulation, and the shared-expert visit quantizes tile rows
     online to ``FP8`` via ``quantize_block`` (matching ``_tp_chain``'s
     ``xla_quantized_matmul`` numerics).
  3. ``act_mode = "w4a8"``: token rows are quantized once per call via
     ``_quantize_rows(x, FP8)`` (one scale per row), gathered as ``FP8``, and
     contracted against ``FP4 -> FP8`` widened K-blocks on the ``FP8`` MXU.
  4. Both arms step ``w2`` in ``w2_n_chunk = 2048`` N-chunks so a full
     ``[tile_rows, H] float32`` accumulator never materializes in VMEM, packing
     256-column BF16 pairs directly into ``out_vmem`` via ``pack_elementwise``.
  5. Supports both ``EP2 x TP4`` (``expert_start = ep_rank * E_local``, filtering
     non-local routes while counting the TP8-sharded ``shared_experts`` once) and
     ``TP8`` (``expert_start = None``) over one 8-rank chunked
     ``reduce-scatter + all-gather`` combine.
"""

import functools
import math
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.fused_moe_prefill import (
    fused_moe_prefill_kernel_v2 as prefill_v2,
)
from vllm_torchtpu.kernels.experimental.fused_moe_prefill.fused_moe_prefill_kernel import (  # noqa: E501
    _GATHER_UNROLL,
    _LANES,
    _LOOKAHEAD,
    _TILE_RUNGS,
    _VISIT_SLACK,
    _WEIGHT_DMA_PRIORITY,
    _byte_transpose4,
    _cdiv,
)
from vllm_torchtpu.kernels.experimental.fused_moe_prefill.fused_moe_prefill_kernel_v2 import (  # noqa: E501
    _quantize_rows,
    _routing_plan,
    tile_rows_for,
)
from vllm_torchtpu.kernels.experimental.moe_tp_decode import all_reduce
from vllm_torchtpu.kernels.fused_moe.v2 import host as v2_host
from vllm_torchtpu.kernels.fused_moe.v2.kernel import (
    DN,
    _intermediate_rows,
    _tile_row_scales,
)
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_block

ActMode = Literal["w4a16", "w4a8"]

FP4 = jnp.float4_e2m1fn
FP8 = jnp.float8_e4m3fn
BF16 = jnp.bfloat16
F32 = jnp.float32

_PACK4 = 8
_H_ALIGN = 512
_PAIR_COLS = 2 * _LANES
_DEFAULT_W2_N_CHUNK = 2048
_STATIC_MAX_TILE_ROWS = 256
_VMEM_ADMISSION_FRACTION = 0.875
_REF_BITCAST_JAX_VERSION = "0.9.1"

# Default Stage-A settings: 4-slot expert ring, 256-row tiles from T >= 4096,
# and 4 rungs (`tall_rung_rows=None`: [32, 64, 96, 128] at 128-row tiles and
# [64, 128, 192, 256] at 256-row tiles, where 8 rungs were measured much slower
# for W4A16 and no faster for W4A8).
STAGE_A_NVFP4: dict[str, Any] = dict(
    nb=4,
    tile_min=4096,
    flat=True,
    tall_rung_rows=None,
    w2_n_chunk=_DEFAULT_W2_N_CHUNK,
    act_mode="w4a16",
)


def _routing_plan_ids(topk_ids, route_valid, experts: int):
    """Expert-major slab order for ``act_mode="w4a16"`` (no per-row scale array).

    Matches ``fused_moe_prefill_kernel_v2._routing_plan``'s two-sort permutation
    and visit-table layout while omitting the activation-scale payload.
    """
    tokens, topk = topk_ids.shape
    routes = tokens * topk
    route_expert = jnp.where(
        route_valid.reshape(-1), topk_ids.reshape(-1).astype(jnp.int32), experts
    )
    iota = jnp.arange(routes, dtype=jnp.int32)
    if routes < 1 << 20 and experts < 1 << 11:
        key = (route_expert.astype(jnp.uint32) << 20) | iota.astype(jnp.uint32)
        order = (lax.sort(key, is_stable=False) & ((1 << 20) - 1)).astype(jnp.int32)
    else:
        _, order = lax.sort((route_expert, iota), num_keys=1, is_stable=True)
    if routes <= 1 << 16:
        key = (order.astype(jnp.uint32) << 16) | iota.astype(jnp.uint32)
        slab_row = (lax.sort(key, is_stable=False) & 0xFFFF).astype(jnp.int32)
    else:
        _, slab_row = lax.sort((order, iota), num_keys=1, is_stable=False)
    counts = jnp.sum(
        route_expert[None, :] == jnp.arange(experts, dtype=jnp.int32)[:, None],
        axis=1,
        dtype=jnp.int32,
    )
    first = jnp.cumsum(counts) - counts
    active = counts > 0
    (visit_expert,) = jnp.nonzero(
        active, size=experts + _VISIT_SLACK, fill_value=experts
    )
    visit_expert = visit_expert.astype(jnp.int32)
    counts = jnp.concatenate([counts, jnp.zeros((1,), jnp.int32)])
    first = jnp.concatenate([first, jnp.zeros((1,), jnp.int32)])
    return (
        order // topk,
        slab_row.reshape(tokens, topk),
        visit_expert,
        first[visit_expert],
        counts[visit_expert],
        jnp.sum(active, dtype=jnp.int32),
    )


def _scale_blocks(scale, experts: int, n_cols: int) -> int | None:
    """Extract K-block count ``B`` from ``[E, B, 1, N]`` or ``[E, B, N]`` scales."""
    if scale.ndim == 4:
        e, b, one, cols = scale.shape
        return b if (e == experts and one == 1 and cols == n_cols) else None
    if scale.ndim == 3:
        e, b, cols = scale.shape
        return b if (e == experts and cols == n_cols) else None
    return None


def _valid_shared_scale(scale, n_cols: int) -> bool:
    return (
        scale.ndim in (1, 2, 3)
        and scale.shape[-1] == n_cols
        and math.prod(scale.shape) == n_cols
    )


def _scratch_vmem_bytes(
    *,
    tile_rows: int,
    h: int,
    i: int,
    i_s: int,
    b13: int,
    b2: int,
    act_mode: ActMode,
    nb: int,
    info,
) -> int:
    """Exact static Pallas VMEM scratch bytes declared in ``grid_spec``."""
    t = functools.partial(all_reduce.tiled_bytes, info=info)
    n_act = _LOOKAHEAD + 1
    lane_blocks = h // _LANES
    scale_window = tile_rows // _LANES + 1
    has_act_scale = act_mode == "w4a8"
    act_dtype = FP8 if has_act_scale else BF16

    return (
        t((n_act, tile_rows, lane_blocks, _LANES), act_dtype)
        + (t((n_act, scale_window, _LANES), F32) if has_act_scale else 0)
        + t((2, tile_rows, lane_blocks, _LANES), BF16)
        + t((nb, h // _PACK4, 2 * i), jnp.uint32)
        + t((nb, i // _PACK4, h), jnp.uint32)
        + t((nb, b13, 1, 2 * i), F32)
        + t((nb, b2, 1, h), F32)
        + t((h, 2 * i_s), FP8)
        + t((i_s, h), FP8)
        + t((1, 2 * i_s), F32)
        + t((1, h), F32)
    )


def _vmem_plan(
    *,
    tile_rows: int,
    h: int,
    i: int,
    i_s: int,
    b13: int,
    b2: int,
    act_mode: ActMode,
    nb: int,
    w2_n_chunk: int,
    info,
) -> int | None:
    """Estimated VMEM bytes (scratch + peak body live values), or ``None``."""
    t = functools.partial(all_reduce.tiled_bytes, info=info)
    has_act_scale = act_mode == "w4a8"
    act_dtype = FP8 if has_act_scale else BF16
    scratch_bytes = _scratch_vmem_bytes(
        tile_rows=tile_rows,
        h=h,
        i=i,
        i_s=i_s,
        b13=b13,
        b2=b2,
        act_mode=act_mode,
        nb=nb,
        info=info,
    )

    qb1 = h // b13
    bk2 = i // b2
    n_chunk = min(h, w2_n_chunk)
    gate_up_live = (
        t((tile_rows, h), act_dtype)
        + t((qb1, 2 * i), act_dtype)
        + 2 * t((tile_rows, 2 * i), F32)
    )
    down_live = (
        t((tile_rows, i), act_dtype)
        + t((bk2, n_chunk), act_dtype)
        + 2 * t((tile_rows, n_chunk), F32)
    )
    shared_live = (
        t((tile_rows, h), BF16)
        + t((tile_rows, h), FP8)
        + 2 * t((tile_rows, max(2 * i_s, n_chunk)), F32)
    )
    need = scratch_bytes + max(gate_up_live, down_live, shared_live)
    if need > int(info.vmem_capacity_bytes * _VMEM_ADMISSION_FRACTION):
        return None
    return need


def fused_moe_prefill_nvfp4_unsupported_reason(
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w2,
    shared_w2_scale,
    topk_weights,
    topk_ids,
    *,
    shared_gate_weight=None,
    act_mode: ActMode = "w4a16",
    nb: int = 4,
    w2_n_chunk: int = _DEFAULT_W2_N_CHUNK,
) -> str | None:
    """Return a human-readable refusal reason, or ``None`` if admitted."""
    info = pltpu.get_tpu_info()
    if info.num_cores != 1 or not info.is_matmul_supported(FP8, FP8):
        return "needs a single-core TPU with native FP8 x FP8 matmul (TPU v7x+)"
    if act_mode not in ("w4a16", "w4a8"):
        return f"unsupported act_mode={act_mode!r}"
    if nb < 2:
        return f"nb={nb} < 2"
    if w2_n_chunk <= 0 or w2_n_chunk % _PAIR_COLS != 0:
        return f"w2_n_chunk={w2_n_chunk} is not a positive multiple of {_PAIR_COLS}"
    if x.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        return f"invalid operand ranks (x={x.ndim}D, w1={w1.ndim}D, w2={w2.ndim}D)"
    if (
        shared_w1 is None
        or shared_w2 is None
        or shared_w1_scale is None
        or shared_w2_scale is None
        or shared_w1.ndim != 2
        or shared_w2.ndim != 2
    ):
        return "missing or non-2D shared expert weights / scales"
    if topk_weights.ndim != 2 or topk_ids.ndim != 2:
        return "topk_weights and topk_ids must be 2D"

    tokens, h = x.shape
    experts, k1, two_i = w1.shape
    e2, i, out_h = w2.shape
    i_s = shared_w2.shape[0]
    topk = topk_ids.shape[1]
    if tokens <= 0 or topk <= 0 or experts <= 0:
        return (
            f"non-positive dimension (tokens={tokens}, topk={topk}, experts={experts})"
        )
    if e2 != experts or k1 != h or out_h != h or two_i != 2 * i:
        return (
            f"mismatched routed weight shapes w1={tuple(w1.shape)}, "
            f"w2={tuple(w2.shape)} for hidden={h}"
        )
    if topk_ids.shape != (tokens, topk) or topk_weights.shape != (tokens, topk):
        return "mismatched topk_ids / topk_weights shapes"

    packed_row_tile = info.num_sublanes * _PACK4
    if h <= 0 or h % _H_ALIGN != 0 or h % _PAIR_COLS != 0 or h % packed_row_tile != 0:
        return (
            f"hidden size {h} is not aligned to "
            f"{_H_ALIGN} / {_PAIR_COLS} / {packed_row_tile}"
        )
    if i <= 0 or i % _LANES != 0 or i % packed_row_tile != 0:
        return f"routed intermediate {i} is not aligned to {_LANES} / {packed_row_tile}"
    if i_s <= 0 or i_s % _LANES != 0:
        return f"shared intermediate {i_s} is not a positive multiple of {_LANES}"
    if shared_w1.shape != (h, 2 * i_s) or shared_w2.shape != (i_s, h):
        return (
            f"incompatible shared expert shapes sw1={tuple(shared_w1.shape)}, "
            f"sw2={tuple(shared_w2.shape)}"
        )
    if not _valid_shared_scale(shared_w1_scale, 2 * i_s):
        return f"incompatible shared_w1_scale shape {tuple(shared_w1_scale.shape)}"
    if not _valid_shared_scale(shared_w2_scale, h):
        return f"incompatible shared_w2_scale shape {tuple(shared_w2_scale.shape)}"
    if shared_gate_weight is not None and (
        shared_gate_weight.ndim not in (1, 2)
        or math.prod(shared_gate_weight.shape) != h
        or shared_gate_weight.dtype not in (BF16, F32)
    ):
        return "incompatible shared_gate_weight shape or dtype"

    b13 = _scale_blocks(w1_scale, experts, 2 * i)
    b2 = _scale_blocks(w2_scale, experts, h)
    min_block = max(_LANES, packed_row_tile)
    if b13 is None or b13 <= 0 or h % b13 != 0:
        return (
            f"incompatible w1_scale shape {tuple(w1_scale.shape)} "
            f"for h={h}, 2*i={2 * i}"
        )
    qb1 = h // b13
    if qb1 < min_block or qb1 % _LANES != 0 or qb1 % packed_row_tile != 0:
        return f"w1 quantization block {qb1} < {min_block} or unaligned"
    if b2 is None or b2 <= 0 or i % b2 != 0:
        return f"incompatible w2_scale shape {tuple(w2_scale.shape)} for i={i}, h={h}"
    bk2 = i // b2
    if bk2 < min_block or bk2 % _LANES != 0 or bk2 % packed_row_tile != 0:
        return f"w2 quantization block {bk2} < {min_block} or unaligned"

    if x.dtype != BF16:
        return f"activation dtype {x.dtype} != bfloat16"
    if w1.dtype != FP4 or w2.dtype != FP4:
        return f"routed weight dtypes {(w1.dtype, w2.dtype)} != float4_e2m1fn"
    if w1_scale.dtype != F32 or w2_scale.dtype != F32:
        return f"routed scale dtypes {(w1_scale.dtype, w2_scale.dtype)} != float32"
    if shared_w1.dtype != FP8 or shared_w2.dtype != FP8:
        return (
            f"shared weight dtypes {(shared_w1.dtype, shared_w2.dtype)} "
            "!= float8_e4m3fn"
        )
    if shared_w1_scale.dtype != F32 or shared_w2_scale.dtype != F32:
        return (
            f"shared scale dtypes {(shared_w1_scale.dtype, shared_w2_scale.dtype)} "
            "!= float32"
        )
    if topk_ids.dtype != jnp.int32 or topk_weights.dtype not in (BF16, F32):
        return (
            f"routing dtypes topk_ids={topk_ids.dtype}, "
            f"topk_weights={topk_weights.dtype} unsupported"
        )

    plan = _vmem_plan(
        tile_rows=_STATIC_MAX_TILE_ROWS,
        h=h,
        i=i,
        i_s=i_s,
        b13=b13,
        b2=b2,
        act_mode=act_mode,
        nb=nb,
        w2_n_chunk=w2_n_chunk,
        info=info,
    )
    if plan is None:
        budget_mib = info.vmem_capacity_bytes * _VMEM_ADMISSION_FRACTION / (1 << 20)
        return (
            f"exceeds VMEM admission budget ({budget_mib:.1f} MiB) at "
            f"tile_rows={_STATIC_MAX_TILE_ROWS}, i={i}, nb={nb}"
        )
    return None


def can_use_fused_moe_prefill_nvfp4(
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w2,
    shared_w2_scale,
    topk_weights,
    topk_ids,
    *,
    shared_gate_weight=None,
    act_mode: ActMode = "w4a16",
    nb: int = 4,
    w2_n_chunk: int = _DEFAULT_W2_N_CHUNK,
) -> bool:
    """Static trace-time admission check (independent of token count ``T``)."""
    return (
        fused_moe_prefill_nvfp4_unsupported_reason(
            x,
            w1,
            w2,
            w1_scale,
            w2_scale,
            shared_w1,
            shared_w1_scale,
            shared_w2,
            shared_w2_scale,
            topk_weights,
            topk_ids,
            shared_gate_weight=shared_gate_weight,
            act_mode=act_mode,
            nb=nb,
            w2_n_chunk=w2_n_chunk,
        )
        is None
    )


@functools.cache
def build_expert_kernel_nvfp4(
    *,
    experts: int,
    h: int,
    i: int,
    i_s: int,
    b13: int,
    b2: int,
    tokens: int,
    routes: int,
    tile_rows: int,
    act_mode: ActMode = "w4a16",
    nb: int = 4,
    rung_rows: int | None = None,
    w2_n_chunk: int = _DEFAULT_W2_N_CHUNK,
):
    """Build the per-rank NVFP4 routed + online-FP8 shared expert Pallas call."""
    if act_mode not in ("w4a16", "w4a8"):
        raise ValueError(f"Unsupported act_mode={act_mode!r}; expected w4a16 or w4a8")
    if w2_n_chunk <= 0 or w2_n_chunk % _PAIR_COLS != 0:
        raise ValueError(f"w2_n_chunk={w2_n_chunk} must be a positive multiple of 256")

    has_act_scale = act_mode == "w4a8"
    act_dtype = FP8 if has_act_scale else BF16
    n_act = _LOOKAHEAD + 1
    window = tile_rows + _LANES
    scale_window = tile_rows // _LANES + 1
    n_rungs = _TILE_RUNGS if rung_rows is None else tile_rows // rung_rows
    rungs = [tile_rows * (r + 1) // n_rungs for r in range(n_rungs)]
    lane_blocks = h // _LANES
    lhs_from_words = (
        has_act_scale
        and lane_blocks % 4 == 0
        and tile_rows % (4 * n_rungs * 8) == 0
        and math.gcd(lane_blocks, 32) <= 16
    )
    out_by_pairs = lane_blocks % 2 == 0 and tile_rows % (8 * n_rungs) == 0

    qb1 = h // b13
    bk2 = i // b2
    rows1_u32 = qb1 // _PACK4
    rows2_u32 = bk2 // _PACK4
    n_chunk = min(h, w2_n_chunk)

    shared = experts
    slab_rows = routes + tokens + tile_rows
    scale_rows = (_cdiv(slab_rows, _LANES) + 1) if has_act_scale else 0
    token_rows = _cdiv(routes, _LANES) * _LANES + window

    def kernel(*refs):
        refs = list(refs)
        visit_expert, visit_start, visit_rows, num_visits = refs[:4]
        k0 = 4
        if has_act_scale:
            x_hbm, row_scale_hbm, row_token_hbm = refs[k0 : k0 + 3]
            k0 += 3
        else:
            x_hbm, row_token_hbm = refs[k0 : k0 + 2]
            row_scale_hbm = None
            k0 += 2
        w1_hbm, w2_hbm, w1_scale_hbm, w2_scale_hbm = refs[k0 : k0 + 4]
        k0 += 4
        (
            shared_w1_hbm,
            shared_w2_hbm,
            shared_w1_scale_hbm,
            shared_w2_scale_hbm,
        ) = refs[k0 : k0 + 4]
        k0 += 4
        slab_hbm = refs[k0]
        k0 += 1
        act_vmem = refs[k0]
        k0 += 1
        act_scale_vmem = None
        if has_act_scale:
            act_scale_vmem = refs[k0]
            k0 += 1
        (
            out_vmem,
            w1_ring,
            w2_ring,
            w1_scale_ring,
            w2_scale_ring,
            token_window,
            act_sems,
            window_sems,
            out_sems,
            ring_sems,
            shared_w1_vmem,
            shared_w2_vmem,
            shared_w1_scale_vmem,
            shared_w2_scale_vmem,
            shared_sem,
        ) = refs[k0 : k0 + 15]
        k0 += 15
        assert k0 == len(refs), (k0, len(refs))

        if not hasattr(w1_hbm, "bitcast"):
            raise NotImplementedError(
                f"fused_moe_prefill_nvfp4 views 4-bit weight refs as packed uint32 "
                f"words via {type(w1_hbm).__name__}.bitcast, missing in jax "
                f"{jax.__version__} (built against jax {_REF_BITCAST_JAX_VERSION})."
            )
        w1_u32 = w1_hbm.bitcast(jnp.uint32)
        w2_u32 = w2_hbm.bitcast(jnp.uint32)

        n_visit = num_visits[0]

        def is_shared(v):
            return visit_expert[v] == shared

        def ring_copies(expert, slot):
            srcs = (
                w1_u32.at[expert],
                w2_u32.at[expert],
                w1_scale_hbm.at[expert],
                w2_scale_hbm.at[expert],
            )
            dsts = (
                w1_ring.at[slot],
                w2_ring.at[slot],
                w1_scale_ring.at[slot],
                w2_scale_ring.at[slot],
            )
            return [
                pltpu.make_async_copy(s, d, ring_sems.at[slot])
                for s, d in zip(srcs, dsts)
            ]

        shared_copies = [
            pltpu.make_async_copy(s, d, shared_sem)
            for s, d in zip(
                (
                    shared_w1_hbm.at[0],
                    shared_w2_hbm.at[0],
                    shared_w1_scale_hbm.at[0],
                    shared_w2_scale_hbm.at[0],
                ),
                (
                    shared_w1_vmem,
                    shared_w2_vmem,
                    shared_w1_scale_vmem,
                    shared_w2_scale_vmem,
                ),
            )
        ]

        def start_weights(v):
            @pl.when(jnp.logical_not(is_shared(v)))
            def _():
                for c in ring_copies(visit_expert[v], lax.rem(v, nb)):
                    c.start(priority=_WEIGHT_DMA_PRIORITY)

        def wait_weights(v):
            @pl.when(jnp.logical_not(is_shared(v)))
            def _():
                for c in ring_copies(0, lax.rem(v, nb)):
                    c.wait()

            @pl.when(is_shared(v))
            def _():
                for c in shared_copies:
                    c.wait()

        def next_tile(v, t):
            more = (t + 1) * tile_rows < visit_rows[v]
            return jnp.where(more, v, v + 1), jnp.where(more, t + 1, 0)

        def tile_start(v, t):
            return visit_start[v] + t * tile_rows

        def tile_live(v, t):
            return jnp.minimum(visit_rows[v] - t * tile_rows, tile_rows)

        def window_copy(start, slot):
            aligned = pl.multiple_of(start - lax.rem(start, _LANES), _LANES)
            lo = slot * window
            if not isinstance(lo, int):
                lo = pl.multiple_of(lo, _LANES)
            return pltpu.make_async_copy(
                row_token_hbm.at[pl.ds(aligned, window)],
                token_window.at[pl.ds(lo, window)],
                window_sems.at[slot],
            )

        def start_window(v, t, slot):
            @pl.when(jnp.logical_and(v < n_visit, jnp.logical_not(is_shared(v))))
            def _():
                window_copy(tile_start(v, t), slot).start()

        def wait_window(v, t, slot):
            @pl.when(jnp.logical_not(is_shared(v)))
            def _():
                window_copy(tile_start(v, t), slot).wait()

        def scale_copy(start, slot):
            assert row_scale_hbm is not None and act_scale_vmem is not None
            return pltpu.make_async_copy(
                row_scale_hbm.at[pl.ds(start // _LANES, scale_window)],
                act_scale_vmem.at[slot],
                act_sems.at[slot],
            )

        def row_copy(token, slot, r):
            return pltpu.make_async_copy(
                x_hbm.at[token], act_vmem.at[slot, r], act_sems.at[slot]
            )

        def issue_tile(v, t, slot, wslot):
            start = tile_start(v, t)

            def gather_routed():
                def block(b, ptr):
                    r0 = b * _GATHER_UNROLL
                    for u in range(_GATHER_UNROLL):
                        row_copy(token_window[ptr + u], slot, r0 + u).start()
                    return ptr + _GATHER_UNROLL

                ptr0 = wslot * window + lax.rem(start, _LANES)
                lax.fori_loop(0, _cdiv(tile_live(v, t), _GATHER_UNROLL), block, ptr0)

            @pl.when(is_shared(v))
            def _():
                n = tile_live(v, t)
                pltpu.make_async_copy(
                    x_hbm.at[pl.ds(t * tile_rows, n)],
                    act_vmem.at[slot, pl.ds(0, n)],
                    act_sems.at[slot],
                ).start()

            @pl.when(jnp.logical_not(is_shared(v)))
            def _():
                gather_routed()

            if has_act_scale:
                scale_copy(start, slot).start()

        def wait_tile(v, t, slot):
            routed_n = _cdiv(tile_live(v, t), _GATHER_UNROLL) * _GATHER_UNROLL
            n = jnp.where(is_shared(v), tile_live(v, t), routed_n)
            pltpu.make_async_copy(
                act_vmem.at[slot, pl.ds(0, n)],
                act_vmem.at[slot, pl.ds(0, n)],
                act_sems.at[slot],
            ).wait()
            if has_act_scale:
                scale_copy(tile_start(v, t), slot).wait()

        def out_copy(start, parity, n):
            return pltpu.make_async_copy(
                out_vmem.at[parity, pl.ds(0, n)],
                slab_hbm.at[pl.ds(start, n)],
                out_sems.at[parity],
            )

        def out_wait(parity, n):
            pltpu.make_async_copy(
                out_vmem.at[parity, pl.ds(0, n)],
                out_vmem.at[parity, pl.ds(0, n)],
                out_sems.at[parity],
            ).wait()

        # ---- prologue ----
        for c in shared_copies:
            c.start(priority=_WEIGHT_DMA_PRIORITY)
        for k in range(nb - 1):

            @pl.when(k < n_visit)
            def _(k=k):
                start_weights(jnp.int32(k))

        z = jnp.int32(0)
        start_window(z, z, 0)
        vi, ti = z, z
        for k in range(_LOOKAHEAD):

            @pl.when(vi < n_visit)
            def _(k=k, vi=vi, ti=ti):
                wait_window(vi, ti, k % 2)
                issue_tile(vi, ti, k % n_act, k % 2)
                vw, tw = next_tile(vi, ti)
                start_window(vw, tw, (k + 1) % 2)

            vi, ti = next_tile(vi, ti)

        def tile_body(carry):
            v, t, g, n_out0, n_out1, vi, ti = carry
            p = lax.rem(g, 2)
            aslot = lax.rem(g, n_act)

            @pl.when(t == 0)
            def _():
                wait_weights(v)

                @pl.when(v + nb - 1 < n_visit)
                def _():
                    start_weights(v + nb - 1)

            wait_tile(v, t, aslot)

            vn, tn = next_tile(v, t)
            gi = g + _LOOKAHEAD
            islot = lax.rem(gi, n_act)

            @pl.when(vi < n_visit)
            def _():
                wait_window(vi, ti, lax.rem(gi, 2))
                issue_tile(vi, ti, islot, lax.rem(gi, 2))
                vw, tw = next_tile(vi, ti)
                start_window(vw, tw, lax.rem(gi + 1, 2))

            vi_n, ti_n = next_tile(vi, ti)
            vi_n = jnp.where(vi < n_visit, vi_n, vi)
            ti_n = jnp.where(vi < n_visit, ti_n, ti)

            slot = lax.rem(v, nb)
            start = tile_start(v, t)
            live = tile_live(v, t)
            n_prev = jnp.where(p == 0, n_out0, n_out1)

            @pl.when(n_prev > 0)
            def _():
                out_wait(p, n_prev)

            def build_lhs(rows):
                if lhs_from_words:
                    wpr = lane_blocks // 4
                    a32 = (
                        act_vmem.at[aslot]
                        .bitcast(jnp.uint32)
                        .reshape(tile_rows * wpr, _LANES)
                    )
                    blocks = []
                    for k in range(wpr):
                        quads = [
                            a32[pl.ds(t_ * wpr + k, rows // 4, stride=4 * wpr)]
                            for t_ in range(4)
                        ]
                        blocks += [
                            pltpu.bitcast(b, act_dtype) for b in _byte_transpose4(quads)
                        ]
                    return jnp.concatenate(blocks, axis=1)
                act = act_vmem[aslot, pl.ds(0, rows)].astype(BF16)
                return act.reshape(rows, h).astype(act_dtype)

            def store_out_chunk(out_chunk, par, out32, half, rows, n0, width):
                if out_by_pairs:
                    assert out32 is not None
                    q_base = n0 // _PAIR_COLS
                    for q_local in range(width // _PAIR_COLS):
                        pair = [
                            out_chunk[
                                :,
                                (2 * q_local + j) * _LANES : (2 * q_local + j + 1)
                                * _LANES,
                            ]
                            for j in range(2)
                        ]
                        out32[pl.ds(q_base + q_local, rows, stride=half)] = (
                            pltpu.pack_elementwise(pair, packed_dtype=BF16)
                        )
                else:
                    lb0 = n0 // _LANES
                    lb_w = width // _LANES
                    out_vmem[par, pl.ds(0, rows), pl.ds(lb0, lb_w), :] = (
                        out_chunk.reshape(rows, lb_w, _LANES).astype(BF16)
                    )

            def compute_routed(rows):
                act = build_lhs(rows)
                if has_act_scale:
                    assert act_scale_vmem is not None
                    act_scale = _tile_row_scales(act_scale_vmem[aslot], start, rows)
                else:
                    act_scale = None
                acc1 = None
                for b in range(b13):
                    wb1 = pltpu.bitcast(
                        w1_ring[slot, pl.ds(b * rows1_u32, rows1_u32), :], FP4
                    ).astype(act_dtype)
                    part = lax.dot_general(
                        # pyrefly: ignore[bad-argument-type]
                        act[:, b * qb1 : (b + 1) * qb1],
                        wb1,
                        DN,
                        preferred_element_type=F32,
                    )
                    scaled = part * w1_scale_ring[slot, b, :, :]
                    acc1 = scaled if acc1 is None else acc1 + scaled
                assert acc1 is not None
                mid, mid_scale = _intermediate_rows(
                    acc1,
                    act_scale,
                    i,
                    w1s=None,
                    act_fn="silu",
                    w1b=None,
                    requantize=has_act_scale,
                )
                half = lane_blocks // 2
                for par in range(2):

                    @pl.when(p == par)
                    def _(par=par):
                        out32 = (
                            out_vmem.at[par]
                            .bitcast(jnp.uint32)
                            .reshape(tile_rows * half, _LANES)
                            if out_by_pairs
                            else None
                        )
                        for n0 in range(0, h, n_chunk):
                            width = min(n_chunk, h - n0)
                            acc2 = None
                            for c in range(b2):
                                wb2 = pltpu.bitcast(
                                    w2_ring[
                                        slot,
                                        pl.ds(c * rows2_u32, rows2_u32),
                                        pl.ds(n0, width),
                                    ],
                                    FP4,
                                ).astype(act_dtype)
                                part2 = lax.dot_general(
                                    mid[:, c * bk2 : (c + 1) * bk2],
                                    wb2,
                                    DN,
                                    preferred_element_type=F32,
                                )
                                s_part = (
                                    part2 * w2_scale_ring[slot, c, :, n0 : n0 + width]
                                )
                                acc2 = s_part if acc2 is None else acc2 + s_part
                            assert acc2 is not None
                            out_chunk = (
                                acc2 * mid_scale if mid_scale is not None else acc2
                            )
                            store_out_chunk(
                                out_chunk, par, out32, half, rows, n0, width
                            )

            def compute_shared(rows):
                # Shared-expert quantization and rounding contract:
                # Match the TP-chain FP8 shared-expert path operation-for-operation.
                # In w4a16, quantize each BF16 activation row dynamically to FP8
                # via quantize_block(..., axis=1), accumulate the FP8xFP8 matmul in
                # F32, multiply by the row and channel scales, round to BF16 before
                # SiLU*up, and dynamically requantize the BF16 intermediate per row
                # for the down projection. In w4a8, reuse the pre-quantized FP8
                # input activations and scales directly.
                if has_act_scale:
                    assert act_scale_vmem is not None
                    act_fp8 = build_lhs(rows)
                    act_scale = _tile_row_scales(act_scale_vmem[aslot], start, rows)
                    gu = lax.dot_general(
                        # pyrefly: ignore[bad-argument-type]
                        act_fp8,
                        shared_w1_vmem[...],
                        DN,
                        preferred_element_type=F32,
                    )
                    mid_q, mid_scale = _intermediate_rows(
                        gu,
                        act_scale,
                        i_s,
                        w1s=shared_w1_scale_vmem[...],
                        act_fn="silu",
                        w1b=None,
                        requantize=True,
                    )
                    assert mid_scale is not None
                else:
                    act_bf16 = build_lhs(rows)
                    sxq, sxs = quantize_block(act_bf16, axis=1, target_dtype=FP8)
                    sxs = sxs.astype(F32)
                    gu = lax.dot_general(
                        sxq,
                        shared_w1_vmem[...],
                        DN,
                        preferred_element_type=F32,
                    )
                    gu_bf16 = (gu * sxs * shared_w1_scale_vmem[...]).astype(BF16)
                    mid_bf16 = (
                        jax.nn.silu(gu_bf16[:, :i_s].astype(F32)).astype(BF16)
                        * gu_bf16[:, i_s:]
                    )
                    mid_q, mid_scale = quantize_block(
                        mid_bf16, axis=1, target_dtype=FP8
                    )
                    mid_scale = mid_scale.astype(F32)

                half = lane_blocks // 2
                for par in range(2):

                    @pl.when(p == par)
                    def _(par=par):
                        out32 = (
                            out_vmem.at[par]
                            .bitcast(jnp.uint32)
                            .reshape(tile_rows * half, _LANES)
                            if out_by_pairs
                            else None
                        )
                        for n0 in range(0, h, n_chunk):
                            width = min(n_chunk, h - n0)
                            acc2 = lax.dot_general(
                                mid_q,
                                shared_w2_vmem[:, pl.ds(n0, width)],
                                DN,
                                preferred_element_type=F32,
                            )
                            out_chunk = (
                                acc2
                                * mid_scale
                                * shared_w2_scale_vmem[:, n0 : n0 + width]
                            )
                            store_out_chunk(
                                out_chunk, par, out32, half, rows, n0, width
                            )

            def compute_visit(rows):
                @pl.when(is_shared(v))
                def _():
                    compute_shared(rows)

                @pl.when(jnp.logical_not(is_shared(v)))
                def _():
                    compute_routed(rows)

            for r, rows in enumerate(rungs):
                below = rungs[r - 1] if r else 0

                @pl.when(jnp.logical_and(live > below, live <= rows))
                def _(rows=rows):
                    compute_visit(rows)

            out_copy(start, p, live).start()
            n_out0 = jnp.where(p == 0, live, n_out0)
            n_out1 = jnp.where(p == 1, live, n_out1)
            return vn, tn, g + 1, n_out0, n_out1, vi_n, ti_n

        _, _, _, n_out0, n_out1, _, _ = lax.while_loop(
            lambda carry: carry[0] < n_visit, tile_body, (z, z, z, z, z, vi, ti)
        )

        for par, n in ((0, n_out0), (1, n_out1)):

            @pl.when(n > 0)
            def _(par=par, n=n):
                out_wait(par, n)

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    n_in = 11 if has_act_scale else 10
    scratch: list[Any] = [
        pltpu.VMEM((n_act, tile_rows, lane_blocks, _LANES), act_dtype)
    ]
    if has_act_scale:
        scratch.append(pltpu.VMEM((n_act, scale_window, _LANES), F32))
    scratch += [
        pltpu.VMEM((2, tile_rows, lane_blocks, _LANES), BF16),
        pltpu.VMEM((nb, h // _PACK4, 2 * i), jnp.uint32),
        pltpu.VMEM((nb, i // _PACK4, h), jnp.uint32),
        pltpu.VMEM((nb, b13, 1, 2 * i), F32),
        pltpu.VMEM((nb, b2, 1, h), F32),
        pltpu.SMEM((2 * window,), jnp.int32),
        pltpu.SemaphoreType.DMA((n_act,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((nb,)),
        pltpu.VMEM((h, 2 * i_s), FP8),
        pltpu.VMEM((i_s, h), FP8),
        pltpu.VMEM((1, 2 * i_s), F32),
        pltpu.VMEM((1, h), F32),
        pltpu.SemaphoreType.DMA(()),
    ]
    tag = f"{act_mode}-nb{nb}-nc{n_chunk}" + (
        "" if rung_rows is None else f"-r{rung_rows}"
    )
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=[hbm] * n_in,
            out_specs=hbm,
            scratch_shapes=tuple(scratch),
            grid=(),
        ),
        out_shape=jax.ShapeDtypeStruct((slab_rows, lane_blocks, _LANES), BF16),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=v2_host.vmem_limit(), disable_bounds_checks=True
        ),
        name=(
            f"fused_moe_prefill_nvfp4-t_{tokens}-e_{experts}-i_{i}-is_{i_s}"
            f"-m_{tile_rows}-{tag}"
        ),
    )
    return call, scale_rows, token_rows


def fused_moe_prefill_nvfp4(
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w2,
    shared_w2_scale,
    shared_gate_weight,
    topk_weights,
    topk_ids,
    *,
    axis_name,
    expert_start=None,
    routed_scale: float = 1.0,
    act_mode: ActMode = "w4a16",
    nb: int = 4,
    tile_min: int = 4096,
    flat: bool = True,
    tall_rung_rows: int | None = None,
    w2_n_chunk: int = _DEFAULT_W2_N_CHUNK,
):
    """Per-rank NVFP4 fused MoE prefill op (routed + shared + combine + AR).

    Call inside ``jax.shard_map(..., mesh=mesh)`` over ``axis_name``. Under
    ``EP2 x TP4``, pass ``expert_start = ep_rank * E_local``: non-local routes
    are filtered out on each rank, every rank computes its ``1 / N_ranks`` TP
    slice of ``shared_experts`` once, ``routed_scale`` scales only routed
    experts, and one ``reduce-scatter + all-gather`` sums everything across
    ``axis_name``.
    """
    if act_mode not in ("w4a16", "w4a8"):
        raise ValueError(f"Unsupported act_mode={act_mode!r}; expected w4a16 or w4a8")
    if x.dtype != BF16:
        raise ValueError(f"Expected bfloat16 activations, got {x.dtype}")
    if w1.dtype != FP4 or w2.dtype != FP4:
        raise ValueError(
            f"Expected float4_e2m1fn routed weights, got {w1.dtype}, {w2.dtype}"
        )
    if shared_w1 is None or shared_w2 is None:
        raise ValueError("shared_w1 and shared_w2 are required")
    if shared_w1.dtype != FP8 or shared_w2.dtype != FP8:
        raise ValueError(
            "Expected float8_e4m3fn shared weights, got "
            f"{shared_w1.dtype}, {shared_w2.dtype}"
        )

    tokens, h = x.shape
    experts, _, two_i = w1.shape
    i = two_i // 2
    i_s = shared_w2.shape[0]
    routes = tokens * topk_ids.shape[1]
    b13 = _scale_blocks(w1_scale, experts, two_i)
    b2 = _scale_blocks(w2_scale, experts, h)
    if b13 is None or b2 is None:
        raise ValueError(
            f"Incompatible routed scale shapes w1_scale={w1_scale.shape}, "
            f"w2_scale={w2_scale.shape} for experts={experts}, h={h}, i={i}"
        )

    tile_rows = tile_rows_for(tokens, tile_min)
    rung_rows = tall_rung_rows if tile_rows > 128 else None
    call, scale_rows, token_rows = build_expert_kernel_nvfp4(
        experts=experts,
        h=h,
        i=i,
        i_s=i_s,
        b13=b13,
        b2=b2,
        tokens=tokens,
        routes=routes,
        tile_rows=tile_rows,
        act_mode=act_mode,
        nb=nb,
        rung_rows=rung_rows,
        w2_n_chunk=w2_n_chunk,
    )

    x_clean = None
    if act_mode == "w4a16" or shared_gate_weight is not None:
        row_finite = jnp.all(jnp.isfinite(x), axis=-1, keepdims=True)
        x_clean = jnp.where(row_finite, x, jnp.zeros_like(x))

    route_valid = topk_weights != 0
    if expert_start is not None:
        topk_ids = topk_ids - expert_start
        local = (topk_ids >= 0) & (topk_ids < experts)
        route_valid = route_valid & local
        topk_ids = jnp.where(local, topk_ids, 0)

    if act_mode == "w4a8":
        x_q, x_scale = _quantize_rows(x, FP8)
        (
            row_token,
            slab_row,
            sorted_scale,
            visit_expert,
            visit_start,
            visit_rows,
            routed_visits,
        ) = _routing_plan(topk_ids, route_valid, experts, x_scale)
        # pyrefly: ignore[bad-argument-type]
        row_scale = jnp.concatenate([sorted_scale, x_scale])
        row_scale = jnp.pad(row_scale, (0, scale_rows * _LANES - row_scale.shape[0]))
        x_in = x_q
    else:
        assert x_clean is not None
        (
            row_token,
            slab_row,
            visit_expert,
            visit_start,
            visit_rows,
            routed_visits,
        ) = _routing_plan_ids(topk_ids, route_valid, experts)
        row_scale = None
        x_in = x_clean.reshape(tokens, h // _LANES, _LANES)

    visit_expert = visit_expert.at[routed_visits].set(experts)
    visit_start = visit_start.at[routed_visits].set(routes)
    visit_rows = visit_rows.at[routed_visits].set(tokens)
    n_vis = (routed_visits + 1).reshape(1)
    row_token = jnp.pad(row_token, (0, token_rows - routes))

    args: list[Any] = [
        visit_expert,
        visit_start,
        visit_rows,
        n_vis,
        x_in,
    ]
    if row_scale is not None:
        args.append(row_scale.reshape(-1, _LANES))
    args.append(row_token)

    w1_s4d = w1_scale[:, :, None, :] if w1_scale.ndim == 3 else w1_scale
    w2_s4d = w2_scale[:, :, None, :] if w2_scale.ndim == 3 else w2_scale
    if w1_s4d.dtype != F32:
        w1_s4d = w1_s4d.astype(F32)
    if w2_s4d.dtype != F32:
        w2_s4d = w2_s4d.astype(F32)
    weight_args: list[Any] = [
        w1,
        w2,
        w1_s4d,
        w2_s4d,
        shared_w1.reshape(1, h, 2 * i_s),
        shared_w2.reshape(1, i_s, h),
        shared_w1_scale.reshape(1, 1, 2 * i_s).astype(F32),
        shared_w2_scale.reshape(1, 1, h).astype(F32),
    ]
    slab = call(*args, *weight_args)

    coeff = jnp.where(route_valid, topk_weights.astype(F32), 0.0)
    if routed_scale != 1.0:
        coeff = coeff * routed_scale

    if shared_gate_weight is None:
        gate = jnp.ones((tokens, 1), F32)
    else:
        assert x_clean is not None
        gate = jax.nn.sigmoid(
            jnp.dot(
                x_clean,
                shared_gate_weight.reshape(h, 1).astype(x_clean.dtype),
                preferred_element_type=F32,
            )
        )
    shared_row = routes + jnp.arange(tokens, dtype=jnp.int32)[:, None]
    slab_row = jnp.where(route_valid, slab_row, shared_row)
    return prefill_v2.combine(
        slab,
        jnp.concatenate([slab_row, shared_row], 1),
        jnp.concatenate([coeff, gate], 1),
        axis_name,
        flat=flat,
    )


def fused_moe_prefill_nvfp4_stage_a(*args, **kwargs):
    """``fused_moe_prefill_nvfp4`` with ``STAGE_A_NVFP4`` defaults."""
    merged = {**STAGE_A_NVFP4, **kwargs}
    return fused_moe_prefill_nvfp4(*args, **merged)

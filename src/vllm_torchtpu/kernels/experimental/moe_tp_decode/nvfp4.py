# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""moe_tp_decode with Nvfp4Format: FP4-block512 W4A8 routed experts.

The kernel is kernel._kernel (routing, compaction, shared expert, combine and
all-reduce shared with the FP8 format; the FP4 ring is kernel._nvfp4_ring, see
the kernel module docstring). This module holds what only the NVFP4 format
needs: its admission (can_use_moe_tp_decode_nvfp4), group size, VMEM plan,
kernel name and the jitted entry point moe_tp_decode_nvfp4. NVFP4.md has the
layouts, flags and measurements.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.moe_tp_decode import all_reduce
from vllm_torchtpu.kernels.experimental.moe_tp_decode.kernel import (
    _FP4_PACK,
    _NVFP4_NUM_BUFFERS,
    _Q_BLOCK,
    FP4,
    ROW_BUCKETS,
    VMEM_HEADROOM,
    Nvfp4Format,
    _admit_shapes,
    _call,
    _fixed_vmem_bytes,
    exact_in_bf16,
)

__all__ = [
    "ROW_BUCKETS",
    "can_use_moe_tp_decode_nvfp4",
    "exact_in_bf16",
    "group_size",
    "kernel_name",
    "moe_tp_decode_nvfp4",
]

_W2_N_CHUNK = 2048
_GROUP_COLS = 1024
_MAX_GROUP = 4
# NVFP4 ring depth (kernel._NUM_BUFFERS is the FP8 ring's).
_NUM_BUFFERS = _NVFP4_NUM_BUFFERS
VMEM_FRACTION = 0.875


def group_size(i: int, local_experts: int) -> int:
    """Experts grouped per gate/up dot slot (2 at I=256, 1 at I>=512)."""
    return min(_MAX_GROUP, local_experts, max(1, _GROUP_COLS // (2 * i)))


def _vmem_plan(
    m,
    h,
    i,
    b13,
    b2,
    group,
    num_experts,
    shared_inter,
    info,
    spec,
):
    """(need_bytes, vmem_limit_bytes), or None if limit exceeds budget."""
    t = functools.partial(all_reduce.tiled_bytes, info=info)
    f32, u32 = jnp.float32, jnp.uint32
    fixed = _fixed_vmem_bytes(m, h, num_experts, shared_inter, info, xs_dtype=f32)
    slot = (
        t((h // _FP4_PACK, group * 2 * i), u32)
        + t((group, i // _FP4_PACK, h), u32)
        + t((b13, 1, group * 2 * i), f32)
        + t((group, b2, 1, h), f32)
    )
    need = fixed + _NUM_BUFFERS * slot + all_reduce.vmem_bytes(spec, info)
    limit = int(need * VMEM_HEADROOM)
    if limit > int(info.vmem_capacity_bytes * VMEM_FRACTION):
        return None
    return need, min(limit, info.vmem_capacity_bytes)


def kernel_name(
    experts: int,
    m: int,
    h: int,
    i: int,
    shared_inter: int,
    n_ranks: int,
    num_experts: int | None = None,
    fast: bool = False,
    split_meta: bool = True,
    w2_n_chunk: int = _W2_N_CHUNK,
) -> str:
    g = group_size(i, experts)
    name = (
        f"moe_tp_decode_nvfp4-e_{experts}-m_{m}-h_{h}-i_{i}-g_{g}"
        f"-fs_{shared_inter}-nb_{_NUM_BUFFERS}-n_{n_ranks}"
    )
    if w2_n_chunk != _W2_N_CHUNK:
        name += f"-nc_{w2_n_chunk}"
    if split_meta:
        name += "-sm"
    if num_experts is not None and num_experts != experts:
        name += f"-ne_{num_experts}"
    if fast:
        name += "-fast"
    return name


def can_use_moe_tp_decode_nvfp4(
    x,
    scores,
    bias,
    w13,
    w2,
    s13,
    s2,
    shared_w13,
    shared_s13,
    shared_w2,
    shared_s2,
    *,
    topk: int,
    n_ranks: int,
    w2_n_chunk: int = _W2_N_CHUNK,
) -> bool:
    """Trace-time admission for the NVFP4 W4A8 TP/EPxTP decode kernel."""
    info = pltpu.get_tpu_info()
    if w2_n_chunk <= 0 or w2_n_chunk % info.num_lanes != 0:
        return False
    if s13.ndim != 4 or s2.ndim != 4:
        return False
    dims = _admit_shapes(
        x,
        scores,
        bias,
        w13,
        w2,
        s13,
        s2,
        shared_w13,
        shared_s13,
        shared_w2,
        shared_s2,
        topk=topk,
        n_ranks=n_ranks,
        info=info,
        weight_dtype=FP4,
    )
    if dims is None:
        return False
    m, h, experts, i, num_experts, shared_inter = dims
    packed_row_tile = info.num_sublanes * _FP4_PACK
    if h % packed_row_tile or i % packed_row_tile:
        return False
    b13 = h // _Q_BLOCK
    if s13.shape != (experts, b13, 1, 2 * i):
        return False
    b2 = s2.shape[1]
    min_bk2 = max(info.num_lanes, packed_row_tile)
    if b2 <= 0 or i % b2 != 0:
        return False
    bk2 = i // b2
    if bk2 < min_bk2 or bk2 % info.num_lanes != 0 or bk2 % packed_row_tile != 0:
        return False
    if s2.shape != (experts, b2, 1, h):
        return False
    spec = all_reduce.make_spec(n_ranks, m, h)
    g = group_size(i, experts)
    plan = _vmem_plan(
        m,
        h,
        i,
        b13,
        b2,
        g,
        num_experts,
        shared_inter,
        info,
        spec,
    )
    return plan is not None


@functools.partial(
    jax.jit,
    static_argnames=(
        "routed_scale",
        "topk",
        "n_ranks",
        "fast",
        "hbm_operands",
        "split_meta",
        "w2_n_chunk",
    ),
)
def moe_tp_decode_nvfp4(
    x,
    scores,
    bias,
    n_valid,
    rank,
    w13,
    w2,
    s13,
    s2,
    shared_w13,
    shared_s13,
    shared_w2,
    shared_s2,
    experts_start=0,
    *,
    routed_scale: float,
    topk: int,
    n_ranks: int,
    fast: bool = False,
    hbm_operands: bool = False,
    split_meta: bool = True,
    w2_n_chunk: int = _W2_N_CHUNK,
):
    """The layer's bf16 [N, D] output, summed over the n_ranks TP/EPxTP mesh.

    Args:
      x: [N, D] bf16 hidden states (identical across ranks).
      scores: [N, E] f32 router scores (sigmoid of logits).
      bias: [E] f32 e_score_correction_bias.
      n_valid: int32 scalar, rows [0, n_valid) are real tokens (identical across
        ranks).
      rank: int32 scalar, this device's index in the 1-D mesh of n_ranks.
      w13, w2: [El, D, 2I] / [El, I, D] float4_e2m1fn K-major routed weights for
        global experts [experts_start, experts_start + El).
      s13, s2: [El, D // 512, 1, 2I] / [El, B2, 1, D] f32 output-side block
        scales.
      shared_w13, shared_w2: [D, 2Fs] / [Fs, D] fp8 shared-expert weights;
        shared_s13, shared_s2: [2Fs] / [D] f32 per-channel scales.
      experts_start: int32 scalar, first global expert id on this rank (0 under
        plain TP).
      routed_scale: routed_scaling_factor, exact in bf16.
      topk: experts selected per token.
      n_ranks: mesh size (TP ranks or EP * TP ranks).
      fast: hardware-scheduled fast path (packed-key routing, vector compaction,
        unpredicated ring priming/refill).
      hbm_operands: constrain the 8 weight/scale operands to HBM.
      split_meta: pass n_valid, rank, experts_start as three int32[1] scalar
        prefetch buffers rather than one stacked int32[3].
      w2_n_chunk: output-column tile width for the w2 down projection.

    Returns:
      bf16(sum over ranks of shared + routed_scale * routed), rows >= n_valid
      identically +0.0.
    """
    if not exact_in_bf16(routed_scale):
        raise ValueError(f"routed_scale {routed_scale!r} is not exact in bf16")
    m, h = x.shape
    experts, i, _ = w2.shape
    b13 = s13.shape[1]
    b2 = s2.shape[1]
    bk2 = i // b2
    num_experts = scores.shape[1]
    shared_inter = shared_w2.shape[0]
    info = pltpu.get_tpu_info()
    if w2_n_chunk <= 0 or w2_n_chunk % info.num_lanes != 0:
        raise ValueError(
            f"w2_n_chunk {w2_n_chunk} must be positive and a multiple of "
            f"{info.num_lanes}"
        )
    ar_spec = all_reduce.make_spec(n_ranks, m, h)
    g = group_size(i, experts)
    plan = _vmem_plan(
        m,
        h,
        i,
        b13,
        b2,
        g,
        num_experts,
        shared_inter,
        info,
        ar_spec,
    )
    if plan is None:
        raise ValueError(
            "moe_tp_decode_nvfp4 does not fit VMEM; check can_use_moe_tp_decode_nvfp4"
        )
    _, limit = plan
    nb = _NUM_BUFFERS

    s13_4d = s13[:, :, None, :] if s13.ndim == 3 else s13
    s2_4d = s2[:, :, None, :] if s2.ndim == 3 else s2
    if s13_4d.shape != (experts, b13, 1, 2 * i) or s2_4d.shape != (experts, b2, 1, h):
        raise ValueError(
            f"Expected 4D scale shapes {(experts, b13, 1, 2 * i)} and "
            f"{(experts, b2, 1, h)}, got {s13.shape} and {s2.shape}"
        )
    return _call(
        x,
        scores,
        bias,
        n_valid,
        rank,
        experts_start,
        (w13, w2, s13_4d, s2_4d),
        shared_w13,
        shared_s13,
        shared_w2,
        shared_s2,
        info=info,
        ar_spec=ar_spec,
        limit=limit,
        name=kernel_name(
            experts,
            m,
            h,
            i,
            shared_inter,
            n_ranks,
            num_experts,
            fast,
            split_meta,
            w2_n_chunk,
        ),
        split=split_meta,
        fast=fast,
        hbm_operands=hbm_operands,
        ring_scratch=(
            pltpu.VMEM((nb, h // _FP4_PACK, g * 2 * i), jnp.uint32),
            pltpu.VMEM((nb, g, i // _FP4_PACK, h), jnp.uint32),
            pltpu.VMEM((nb, b13, 1, g * 2 * i), jnp.float32),
            pltpu.VMEM((nb, g, b2, 1, h), jnp.float32),
        ),
        xs_dtype=jnp.float32,
        active_len=max(1, pl.cdiv(experts, g) * g),
        ring_sems=(nb, 4 * g),
        kernel_kwargs=dict(
            local_experts=experts,
            topk=topk,
            routed_scale=float(routed_scale),
            ar_spec=ar_spec,
            fast=fast,
            weight_format=Nvfp4Format(group=g, bk2=bk2, w2_n_chunk=w2_n_chunk),
        ),
    )

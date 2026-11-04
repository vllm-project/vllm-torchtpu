# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 W4A8 decode MoE layer of one TP or EPxTP rank in one pallas_call.

Follows the single-program structure of ``moe_tp_decode.kernel`` (reusing its
in-kernel ``all_reduce``, ``_route`` / ``_route_fast``, ``_compact_vector``,
online-FP8 ``_shared_expert`` and ``_combine``) while replacing the routed
expert weight ring with grouped whole-expert FP4 slots:

  1. Routed ``w13 [El, H, 2I]`` and ``w2 [El, I, H]`` (``float4_e2m1fn``) are
     viewed in HBM as ``uint32 [El, H // 8, 2I]`` and ``uint32 [El, I // 8, H]``
     (8 K-rows per 32-bit word) and streamed whole into a 3-slot VMEM ring.
  2. Each ring slot groups ``g = group_size(I, El)`` active local experts side
     by side along ``N`` (``g = 4`` at TP8 ``I = 256``, ``g = 2`` at EP2xTP4
     ``I = 512``, capping ``g * 2I <= 2048``), so each 512-row ``w13`` K-block
     widens ``fp4 -> fp8`` in VMEM and executes one wide ``[M, 512] x [512,
     g * 2I]`` FP8 MXU dot, scaled in ``f32`` by the ``x`` and ``w13`` block
     scales and summed in ``f32``. Tail lanes when ``active_count % g != 0``
     clamp their HBM source index to ``active[0]`` and mask their routing
     coefficient to ``0.0``.
  3. Each expert in the group runs ``silu(gate) * up`` in ``f32``, masks rows
     where ``coeff == 0`` (preventing non-finite padded rows from propagating
     NaNs), dynamically quantizes per row to ``FP8``, and accumulates its
     block-scaled ``w2`` down projection (``fp4 -> fp8`` widen per ``bk2``
     K-block) into ``owner_ref [M, H] f32``.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.moe_tp_decode import all_reduce
from vllm_torchtpu.kernels.experimental.moe_tp_decode.kernel import (
    FP8,
    FP8_MAX,
    PAIR,
    ROW_BUCKETS,
    _combine,
    _compact_vector,
    _Meta,
    _route,
    _route_fast,
    _shared_expert,
    _unused,
    exact_in_bf16,
)
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_block

FP4 = jnp.float4_e2m1fn
_PACK4 = 8
_Q_BLOCK = 512
_H_ALIGN = 512
_W2_N_CHUNK = 2048
_GROUP_COLS = 2048
_MAX_GROUP = 4
_NUM_BUFFERS = 3
VMEM_HEADROOM = 1.25
VMEM_FRACTION = 0.875
_REF_BITCAST_JAX_VERSION = "0.9.1"


def group_size(i: int, local_experts: int) -> int:
    """Experts grouped per gate/up dot slot (4 at I<=256, 2 at I=512)."""
    return min(_MAX_GROUP, local_experts, max(1, _GROUP_COLS // (2 * i)))


def _split_entry(n_valid_ref, rank_ref, e0_ref, *refs, **kwargs):
    return _kernel(_Meta((n_valid_ref, rank_ref, e0_ref)), *refs, **kwargs)


def _vmem_plan(m, h, i, b13, b2, group, num_experts, shared_inter, info, spec):
    """(need_bytes, vmem_limit_bytes), or None if limit exceeds budget."""
    t = functools.partial(all_reduce.tiled_bytes, info=info)
    e_pad = pl.cdiv(num_experts, info.num_lanes) * info.num_lanes
    f32, bf16, u32 = jnp.float32, jnp.bfloat16, jnp.uint32
    fixed = (
        2 * t((m, h), bf16)  # x in, out
        + t((m, e_pad), f32)  # scores
        + t((1, e_pad), f32)  # bias
        + t((h, 2 * shared_inter), FP8)  # shared gate_up
        + t((1, 2 * shared_inter), f32)
        + t((shared_inter, h), FP8)  # shared down
        + t((1, h), f32)
        + 2 * t((m, h), FP8)  # sxq (per row), xq (per 512-col block)
        + t((h // _Q_BLOCK, m, info.num_lanes), f32)  # xs
        + 2 * t((m, h), f32)  # owner, shared
        + t((m, e_pad), f32)  # wtab
        + t((4, e_pad), jnp.int32)  # hits_v
    )
    slot = (
        t((h // _PACK4, group * 2 * i), u32)
        + t((group, i // _PACK4, h), u32)
        + t((b13, group * 2 * i), f32)
        + t((group, b2, h), f32)
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
    if info.num_cores != 1 or not info.is_matmul_supported(FP8, FP8):
        return False
    if w2_n_chunk <= 0 or w2_n_chunk % info.num_lanes != 0:
        return False
    if x.ndim != 2 or scores.ndim != 2 or bias.ndim != 1:
        return False
    if w13.ndim != 3 or w2.ndim != 3 or shared_w13.ndim != 2 or shared_w2.ndim != 2:
        return False
    if s13.ndim != 4 or s2.ndim != 4:
        return False
    m, h = x.shape
    experts, i, out = w2.shape
    num_experts, shared_inter = scores.shape[1], shared_w2.shape[0]
    if m not in ROW_BUCKETS or scores.shape != (m, num_experts):
        return False
    if bias.shape != (num_experts,) or experts <= 0 or num_experts % experts:
        return False
    if not 0 < topk <= num_experts or i <= 0 or i % info.num_lanes:
        return False
    packed_row_tile = info.num_sublanes * _PACK4
    if h % _H_ALIGN or h % _Q_BLOCK or h % packed_row_tile or i % packed_row_tile:
        return False
    if shared_inter <= 0 or shared_inter % PAIR:
        return False
    if out != h or w13.shape != (experts, h, 2 * i):
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
    shared = (shared_w13.shape, shared_s13.shape, shared_w2.shape, shared_s2.shape)
    fs = shared_inter
    if shared != ((h, 2 * fs), (2 * fs,), (fs, h), (h,)):
        return False
    if x.dtype != jnp.bfloat16 or scores.dtype != jnp.float32:
        return False
    if w13.dtype != FP4 or w2.dtype != FP4:
        return False
    if shared_w13.dtype != FP8 or shared_w2.dtype != FP8:
        return False
    if any(s.dtype != jnp.float32 for s in (bias, s13, s2, shared_s13, shared_s2)):
        return False
    if not all_reduce.supported(n_ranks, m, h, info):
        return False
    spec = all_reduce.make_spec(n_ranks, m, h)
    g = group_size(i, experts)
    plan = _vmem_plan(m, h, i, b13, b2, g, num_experts, shared_inter, info, spec)
    return plan is not None


def _quantize_blocks_f32(x, q_ref, scale_ref):
    """Quantize ``x [m, h]`` bf16 to FP8 per ``_Q_BLOCK`` columns with f32 scales."""
    m, h = x.shape
    lanes = scale_ref.shape[-1]
    for k in range(0, h, _Q_BLOCK):
        block = x[:, k : k + _Q_BLOCK].astype(jnp.float32)
        scale = jnp.max(jnp.abs(block), axis=1, keepdims=True) / FP8_MAX
        inv = jnp.where(scale == 0, 0.0, 1.0 / scale)
        q_ref[:, k : k + _Q_BLOCK] = (block * inv).astype(FP8)
        scale_ref[k // _Q_BLOCK] = jnp.broadcast_to(scale, (m, lanes))


def _kernel(
    meta_ref,
    x_ref,
    scores_ref,
    bias_ref,
    w1_hbm,
    w2_hbm,
    s1_hbm,
    s2_hbm,
    sw13_hbm,
    ss13_hbm,
    sw2_hbm,
    ss2_hbm,
    out_ref,
    w1_ref,
    w2_ref,
    s1_ref,
    s2_ref,
    sw13_ref,
    ss13_ref,
    sw2_ref,
    ss2_ref,
    sxq_ref,
    xq_ref,
    xs_ref,
    owner_ref,
    shared_ref,
    wtab_ref,
    hits_v_ref,
    hits_s_ref,
    active_ids_ref,
    ring_sem,
    shared_sem,
    hit_sem,
    *ar_scr,
    local_experts: int,
    group: int,
    bk2: int,
    topk: int,
    routed_scale: float,
    ar_spec: all_reduce.AllReduceSpec,
    fast: bool = False,
    w2_n_chunk: int = _W2_N_CHUNK,
):
    all_reduce.prologue(ar_spec)
    if not hasattr(w1_hbm, "bitcast"):
        raise NotImplementedError(
            f"moe_tp_decode_nvfp4 views 4-bit weight refs as packed uint32 words "
            f"via {type(w1_hbm).__name__}.bitcast, missing in jax {jax.__version__} "
            f"(built against jax {_REF_BITCAST_JAX_VERSION})."
        )
    w1_u32 = w1_hbm.bitcast(jnp.uint32)
    w2_u32 = w2_hbm.bitcast(jnp.uint32)

    m, h = x_ref.shape
    two_i = s1_ref.shape[-1] // group
    i = two_i // 2
    g = group
    nb = _NUM_BUFFERS
    n_valid = meta_ref[0]
    rank = meta_ref[1]
    e0 = meta_ref[2]

    shared_copies = (
        pltpu.make_async_copy(sw13_hbm, sw13_ref, shared_sem.at[0]),
        pltpu.make_async_copy(ss13_hbm, ss13_ref, shared_sem.at[1]),
        pltpu.make_async_copy(sw2_hbm, sw2_ref, shared_sem.at[2]),
        pltpu.make_async_copy(ss2_hbm, ss2_ref, shared_sem.at[3]),
    )
    for copy in shared_copies:
        copy.start()

    def store_hits():
        hit = jnp.any(wtab_ref[...] != 0, axis=0, keepdims=True)
        hits_v_ref[0:1, :] = jnp.where(hit, 1, 0).astype(jnp.int32)

    def copy_hits():
        hit_copy = pltpu.make_async_copy(hits_v_ref, hits_s_ref, hit_sem.at[0])
        hit_copy.start()
        hit_copy.wait()

    def store_active():
        ids, n_active = _compact_vector(wtab_ref[...], e0, local_experts)
        hits_v_ref[2:3, 0 : ids.shape[1]] = ids.astype(jnp.int32)
        hits_v_ref[3:4, :] = jnp.broadcast_to(
            n_active, (1, hits_v_ref.shape[1])
        ).astype(jnp.int32)

    if fast:
        wtab, flag = _route_fast(scores_ref[...], bias_ref[...], n_valid, topk=topk)
        wtab_ref[...] = wtab
        store_active()
        hits_v_ref[1:2, :] = jnp.broadcast_to(flag, (1, hits_v_ref.shape[1])).astype(
            jnp.int32
        )
        copy_hits()

        @pl.when(hits_s_ref[1, 0] != 0)
        def _exact_routing():
            wtab_ref[...] = _route(scores_ref[...], bias_ref[...], n_valid, topk=topk)
            store_active()
            copy_hits()

    else:
        wtab_ref[...] = _route(scores_ref[...], bias_ref[...], n_valid, topk=topk)
        store_hits()
        copy_hits()

    def compact(e, cnt):
        e_pad = hits_s_ref.shape[1]
        col = jnp.minimum(e0 + e, e_pad - 1)
        hit = jnp.logical_and(e0 + e < e_pad, hits_s_ref[0, col] != 0)
        active_ids_ref[cnt] = e
        return cnt + hit.astype(jnp.int32)

    if fast:
        count = hits_s_ref[3, 0]
    else:
        active_ids_ref[0] = jnp.int32(0)
        count = lax.fori_loop(0, local_experts, compact, jnp.int32(0))

    ngroups = lax.div(count + (g - 1), g)

    def active_clamped(j):
        """Local expert id for compacted index j; tail indices >= count clamp
        to active[0] so group DMAs stay in bounds while finish_group zeroes coeff."""
        j_safe = jnp.where(j < count, j, jnp.int32(0))
        return hits_s_ref[2, j_safe] if fast else active_ids_ref[j_safe]

    def transfers(q):
        slot = q % nb
        copies = []
        for k in range(g):
            e = active_clamped(q * g + k)
            lanes_slice = pl.ds(k * two_i, two_i)
            copies.extend(
                (
                    pltpu.make_async_copy(
                        w1_u32.at[e],
                        w1_ref.at[slot, :, lanes_slice],
                        ring_sem.at[slot, 4 * k],
                    ),
                    pltpu.make_async_copy(
                        w2_u32.at[e],
                        w2_ref.at[slot, k],
                        ring_sem.at[slot, 4 * k + 1],
                    ),
                    pltpu.make_async_copy(
                        s1_hbm.at[e],
                        s1_ref.at[slot, :, lanes_slice],
                        ring_sem.at[slot, 4 * k + 2],
                    ),
                    pltpu.make_async_copy(
                        s2_hbm.at[e],
                        s2_ref.at[slot, k],
                        ring_sem.at[slot, 4 * k + 3],
                    ),
                )
            )
        return copies

    if fast:

        def prime_fast(item, carry):
            for copy in transfers(item):
                copy.start()
            return carry

        lax.fori_loop(0, jnp.minimum(ngroups, nb), prime_fast, 0)
    else:

        def prime(step):
            @pl.when(step < ngroups)
            def start():
                for copy in transfers(step):
                    copy.start()

        for s in range(nb):
            prime(jnp.int32(s))

    for copy in shared_copies:
        copy.wait()

    owner_ref[...] = jnp.zeros((m, h), jnp.float32)
    sxq, sxs = quantize_block(x_ref[...], axis=1, target_dtype=FP8)
    sxq_ref[...] = sxq
    _shared_expert(sxq_ref, sxs, sw13_ref, ss13_ref, sw2_ref, ss2_ref, shared_ref)
    _quantize_blocks_f32(x_ref[...], xq_ref, xs_ref)

    expert_columns = lax.broadcasted_iota(jnp.int32, wtab_ref.shape, 1)
    lanes = xs_ref.shape[-1]
    cols = g * two_i
    rows1_u32 = _Q_BLOCK // _PACK4
    rows2_u32 = bk2 // _PACK4
    n_chunk = min(h, w2_n_chunk)

    def gate_up(slot):
        acc = jnp.zeros((m, cols), jnp.float32)
        for b in range(h // _Q_BLOCK):
            wb = pltpu.bitcast(
                w1_ref[slot, pl.ds(b * rows1_u32, rows1_u32), :], FP4
            ).astype(FP8)
            part = jnp.matmul(
                xq_ref[:, b * _Q_BLOCK : (b + 1) * _Q_BLOCK],
                wb,
                preferred_element_type=jnp.float32,
            )
            xs_b = jnp.concatenate([xs_ref[b]] * (cols // lanes), axis=1)
            acc = acc + part * xs_b * s1_ref[slot, b : b + 1, :]
        return acc

    def finish_group(q, slot, acc1):
        for k in range(g):
            j = q * g + k
            e = active_clamped(j)
            coeff = jnp.where(
                j < count,
                jnp.sum(
                    jnp.where(expert_columns == e0 + e, wtab_ref[...], 0.0),
                    axis=1,
                    keepdims=True,
                ),
                0.0,
            )
            ak = acc1[:, k * two_i : (k + 1) * two_i]
            act = jax.nn.silu(ak[:, :i]) * ak[:, i:]
            mid = jnp.where(coeff != 0, act * coeff, 0.0)
            ms = jnp.max(jnp.abs(mid), axis=1, keepdims=True) / FP8_MAX
            mq = (mid * jnp.where(ms == 0, 0.0, 1.0 / ms)).astype(FP8)
            for n0 in range(0, h, n_chunk):
                width = min(n_chunk, h - n0)
                n1 = n0 + width
                acc2 = jnp.zeros((m, width), jnp.float32)
                for c in range(i // bk2):
                    wb2 = pltpu.bitcast(
                        w2_ref[
                            slot,
                            k,
                            pl.ds(c * rows2_u32, rows2_u32),
                            pl.ds(n0, width),
                        ],
                        FP4,
                    ).astype(FP8)
                    part2 = jnp.matmul(
                        mq[:, c * bk2 : (c + 1) * bk2],
                        wb2,
                        preferred_element_type=jnp.float32,
                    )
                    acc2 = acc2 + part2 * s2_ref[slot, k, c : c + 1, n0:n1]
                owner_ref[:, n0:n1] += acc2 * ms

    def ring_step(step, acc1, refill):
        nxt = step + 1
        for copy in transfers(nxt):
            copy.wait()
        acc1_next = gate_up(nxt % nb)
        finish_group(step, step % nb, acc1)
        if refill:
            for copy in transfers(step + nb):
                copy.start()
        return acc1_next

    @(pl.when(ngroups > 0) if fast else _unused)
    def run_experts_fast():
        for copy in transfers(jnp.int32(0)):
            copy.wait()
        n_refill = jnp.maximum(ngroups - nb, 0)
        acc1 = lax.fori_loop(
            0, n_refill, lambda s, a: ring_step(s, a, True), gate_up(0)
        )
        acc1 = lax.fori_loop(
            n_refill, ngroups - 1, lambda s, a: ring_step(s, a, False), acc1
        )
        finish_group(ngroups - 1, (ngroups - 1) % nb, acc1)

    @(_unused if fast else pl.when(ngroups > 0))
    def run_experts():
        for copy in transfers(jnp.int32(0)):
            copy.wait()

        def group_body(step, acc1):
            nxt = step + 1
            for copy in transfers(nxt):
                copy.wait()
            acc1_next = gate_up(nxt % nb)
            finish_group(step, step % nb, acc1)

            @pl.when(step + nb < ngroups)
            def refill():
                for copy in transfers(step + nb):
                    copy.start()

            return acc1_next

        acc1 = lax.fori_loop(0, ngroups - 1, group_body, gate_up(0))
        finish_group(ngroups - 1, (ngroups - 1) % nb, acc1)

    all_reduce.epilogue(
        ar_spec,
        rank,
        n_valid,
        _combine(owner_ref[...], shared_ref[...], routed_scale),
        out_ref,
        ar_scr,
    )


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
    plan = _vmem_plan(m, h, i, b13, b2, g, num_experts, shared_inter, info, ar_spec)
    if plan is None:
        raise ValueError(
            "moe_tp_decode_nvfp4 does not fit VMEM; check can_use_moe_tp_decode_nvfp4"
        )
    _, limit = plan
    nb = _NUM_BUFFERS
    e_pad = pl.cdiv(num_experts, info.num_lanes) * info.num_lanes
    padded_active = max(1, pl.cdiv(experts, g) * g)
    scores = scores.astype(jnp.float32)
    bias = bias.astype(jnp.float32)
    if e_pad != num_experts:
        scores = jnp.pad(scores, ((0, 0), (0, e_pad - num_experts)))
        bias = jnp.pad(bias, (0, e_pad - num_experts), constant_values=-jnp.inf)
    if split_meta:
        metas = tuple(
            jnp.asarray(v, jnp.int32).reshape(1) for v in (n_valid, rank, experts_start)
        )
    else:
        metas = (
            jnp.stack(
                [
                    jnp.asarray(n_valid, jnp.int32).reshape(()),
                    jnp.asarray(rank, jnp.int32).reshape(()),
                    jnp.asarray(experts_start, jnp.int32).reshape(()),
                ]
            ),
        )
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    weights = (
        w13,
        w2,
        s13.reshape(experts, b13, 2 * i),
        s2.reshape(experts, b2, h),
        shared_w13,
        shared_s13.reshape(1, 2 * shared_inter),
        shared_w2,
        shared_s2.reshape(1, h),
    )
    if hbm_operands:
        weights = tuple(
            pltpu.with_memory_space_constraint(w, pltpu.HBM) for w in weights
        )
    return pl.pallas_call(
        functools.partial(
            _split_entry if split_meta else _kernel,
            local_experts=experts,
            group=g,
            bk2=bk2,
            topk=topk,
            routed_scale=float(routed_scale),
            ar_spec=ar_spec,
            fast=fast,
            w2_n_chunk=w2_n_chunk,
        ),
        out_shape=jax.ShapeDtypeStruct((m, h), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(metas),
            in_specs=[vmem, vmem, vmem, *[hbm] * 8],
            out_specs=vmem,
            scratch_shapes=[
                pltpu.VMEM((nb, h // _PACK4, g * 2 * i), jnp.uint32),
                pltpu.VMEM((nb, g, i // _PACK4, h), jnp.uint32),
                pltpu.VMEM((nb, b13, g * 2 * i), jnp.float32),
                pltpu.VMEM((nb, g, b2, h), jnp.float32),
                pltpu.VMEM((h, 2 * shared_inter), FP8),
                pltpu.VMEM((1, 2 * shared_inter), jnp.float32),
                pltpu.VMEM((shared_inter, h), FP8),
                pltpu.VMEM((1, h), jnp.float32),
                pltpu.VMEM((m, h), FP8),
                pltpu.VMEM((m, h), FP8),
                pltpu.VMEM((h // _Q_BLOCK, m, info.num_lanes), jnp.float32),
                pltpu.VMEM((m, h), jnp.float32),
                pltpu.VMEM((m, h), jnp.float32),
                pltpu.VMEM((m, e_pad), jnp.float32),
                pltpu.VMEM((4 if fast else 1, e_pad), jnp.int32),
                pltpu.SMEM((4 if fast else 1, e_pad), jnp.int32),
                pltpu.SMEM((padded_active,), jnp.int32),
                pltpu.SemaphoreType.DMA((nb, 4 * g)),
                pltpu.SemaphoreType.DMA((4,)),
                pltpu.SemaphoreType.DMA((1,)),
                *all_reduce.scratch_shapes(ar_spec),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=limit, collective_id=all_reduce.COLLECTIVE_ID
        ),
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
    )(
        *metas,
        x,
        scores,
        bias.reshape(1, e_pad),
        *weights,
    )

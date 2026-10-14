# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Decode MoE layer of one TP rank, all-reduce included, in one pallas_call.

For DeepSeek-V3-style MoE layers served with MoE tensor parallelism (every
rank holds all E experts with I = moe_intermediate_size / n columns each, and
its slice of the shared expert), at decode row counts (ROW_BUCKETS). From the
f32 router scores to the layer's bf16 [N, D] output, summed over the n ranks
inside the kernel, so no XLA all-reduce follows:

  1. all_reduce.prologue (barrier signal to every rank);
  2. the shared expert's weight DMAs start;
  3. routing: top-k of scores + bias (router_topk's clamp and lowest-column
     tie rule), weights = the picked unbiased scores over their sum clamped
     to 1e-20, rounded to bf16; rows >= n_valid get weight 0;
  4. the experts any row picked go VMEM -> SMEM and are compacted by a
     branch-free scan; the first _NUM_BUFFERS of them then start their weight
     DMAs, and the shared expert's weights are waited for;
  5. x quantized to FP8 twice, each copy by the rule its matmuls follow with
     this kernel off: per row with quantize_block for the shared expert (its
     linears run xla_quantized_matmul), per 512-column block with bf16
     scales for the routed experts (they run gmm_v2); this rank's slice of
     the shared expert (the usual TP path's bf16 roundings);
  6. the routed experts, streamed through a ring of _NUM_BUFFERS weight slots
     with each gate_up matmul issued one expert ahead; gate_up and its SiLU
     as gmm_v2 computes them: one partial product per 512-column block,
     scaled and summed in bf16;
  7. bf16(bf16(shared) + bf16(bf16(routed) * routed_scale)), the usual TP
     path's combine, all-reduced by all_reduce.epilogue in place of the store.

Rows >= n_valid come out exactly +0; n_valid also picks the all-reduce path,
which sets how many rows travel (all_reduce: zero, fold, trim, full).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.moe_tp_decode import all_reduce
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_block

FP8 = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(FP8).max)
ROW_BUCKETS = (16, 32, 64)
# router_topk's clamp floor, above the finfo.min that masks a pick.
ROUTER_NEG = -3.0e38
# moe_routing: weights / clamp(sum, min=1e-20).
SUM_FLOOR = 1e-20
PAIR = 128
# VMEM limit = need x headroom (Mosaic temporaries), refused above a fraction
# of the capacity so neighbouring XLA ops keep room.
VMEM_HEADROOM = 1.25
VMEM_FRACTION = 0.75
# Output columns per second-matmul chunk; bounds the live f32 result.
_N_CHUNK = 512
# gmm_v2's activation quantization block: one scale per row per this many
# columns of x (its quant_block_size).
_Q_BLOCK = 512
# Weight ring depth. One slot holds the expert whose first matmul is in
# flight, so num_buffers - 2 transfers run ahead of compute.
_NUM_BUFFERS = 6


def exact_in_bf16(value: float) -> bool:
    """True if `value` is a bf16 number, so bf16 and f32 products agree."""
    rounded = np.asarray(value, dtype=jnp.bfloat16).astype(np.float32)
    return float(rounded) == float(value)


def _vmem_plan(m, h, i, num_experts, shared_inter, info, spec):
    """(need_bytes, vmem_limit_bytes), or None if the limit would exceed
    VMEM_FRACTION of the capacity. Whole-array blocks (grid=()), tiles padded."""
    t = functools.partial(all_reduce.tiled_bytes, info=info)
    e_pad = pl.cdiv(num_experts, info.num_lanes) * info.num_lanes
    f32, bf16 = jnp.float32, jnp.bfloat16
    fixed = (
        2 * t((m, h), bf16)  # x in, out
        + t((m, e_pad), f32)  # scores
        + t((1, e_pad), f32)  # bias
        + t((h, 2 * shared_inter), FP8)  # shared gate_up
        + t((1, 2 * shared_inter), f32)
        + t((shared_inter, h), FP8)  # shared down
        + t((1, h), f32)
        + 2 * t((m, h), FP8)  # sxq (per row), xq (per block)
        + t((h // _Q_BLOCK, m, info.num_lanes), bf16)  # xs
        + 2 * t((m, h), f32)  # owner, shared
        + t((m, e_pad), f32)  # wtab
        + t((1, e_pad), jnp.int32)  # hits_v
    )
    slot = t((h, 2 * i), FP8) + t((i, h), FP8) + t((1, 2 * i), f32) + t((1, h), f32)
    need = fixed + _NUM_BUFFERS * slot + all_reduce.vmem_bytes(spec, info)
    limit = int(need * VMEM_HEADROOM)
    if limit > int(info.vmem_capacity_bytes * VMEM_FRACTION):
        return None
    return need, min(limit, info.vmem_capacity_bytes)


def kernel_name(experts, m, h, i, shared_inter, n_ranks) -> str:
    return (
        f"moe_tp_decode-e_{experts}-m_{m}-h_{h}-i_{i}-fs_{shared_inter}"
        f"-nb_{_NUM_BUFFERS}-n_{n_ranks}"
    )


def can_use_moe_tp_decode(
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
) -> bool:
    """Trace-time admission; False means the caller keeps its own path.

    Hardware (a TPU with one TensorCore per device and an FP8 x FP8 MXU),
    shapes, dtypes, the all-reduce geometry and the VMEM plan. The routing
    semantics (sigmoid scores with a correction bias, renormalised ungrouped
    top-k, SiLU, no scale folded into the weights) are the caller's to check.
    """
    info = pltpu.get_tpu_info()
    if info.num_cores != 1 or not info.is_matmul_supported(FP8, FP8):
        return False
    if x.ndim != 2 or scores.ndim != 2 or bias.ndim != 1:
        return False
    if w13.ndim != 3 or w2.ndim != 3 or shared_w13.ndim != 2 or shared_w2.ndim != 2:
        return False
    m, h = x.shape
    experts, i, out = w2.shape
    num_experts, shared_inter = scores.shape[1], shared_w2.shape[0]
    if m not in ROW_BUCKETS or scores.shape != (m, num_experts):
        return False
    if bias.shape != (num_experts,) or experts != num_experts:
        return False
    if not 0 < topk <= num_experts or i <= 0 or i % info.num_lanes:
        return False
    if h % _N_CHUNK or h % _Q_BLOCK or shared_inter <= 0 or shared_inter % PAIR:
        return False
    if out != h or w13.shape != (experts, h, 2 * i):
        return False
    if s13.shape != (experts, 1, 1, 2 * i) or s2.shape != (experts, 1, 1, h):
        return False
    shared = (shared_w13.shape, shared_s13.shape, shared_w2.shape, shared_s2.shape)
    fs = shared_inter
    if shared != ((h, 2 * fs), (2 * fs,), (fs, h), (h,)):
        return False
    if x.dtype != jnp.bfloat16 or scores.dtype != jnp.float32:
        return False
    if any(w.dtype != FP8 for w in (w13, w2, shared_w13, shared_w2)):
        return False
    if any(s.dtype != jnp.float32 for s in (bias, s13, s2, shared_s13, shared_s2)):
        return False
    if not all_reduce.supported(n_ranks, m, h, info):
        return False
    spec = all_reduce.make_spec(n_ranks, m, h)
    return _vmem_plan(m, h, i, num_experts, shared_inter, info, spec) is not None


def _route(scores, bias, n_valid, *, topk):
    """Routing weights [rows, E] f32: router_topk.select on scores + bias
    (clamp, max, lowest attaining column, finfo.min mask), the unbiased scores
    of the picks over their pick-order sum clamped to 1e-20 (NaN stays NaN),
    rounded through bf16; unpicked columns and rows >= n_valid are 0."""
    rows, n = scores.shape
    lane = lax.broadcasted_iota(jnp.int32, (rows, n), 1)
    masked = float(jnp.finfo(jnp.float32).min)
    x = scores + bias
    x = jnp.where(x > ROUTER_NEG, x, ROUTER_NEG)
    wsum = jnp.zeros((rows, 1), jnp.float32)
    for _ in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        idx = jnp.min(jnp.where(x == m, lane, n - 1), axis=1, keepdims=True)
        pick = lane == idx
        # Exactly one nonzero term: the unbiased score of this pick.
        wsum = wsum + jnp.sum(jnp.where(pick, scores, 0.0), axis=1, keepdims=True)
        x = jnp.where(pick, masked, x)
    chosen = x == masked  # unpicked values are >= ROUTER_NEG > masked
    denom = jnp.where(wsum < SUM_FLOOR, SUM_FLOOR, wsum)
    w = (jnp.where(chosen, scores, 0.0) / denom).astype(jnp.bfloat16)
    valid = lax.broadcasted_iota(jnp.int32, (rows, n), 0) < n_valid
    return jnp.where(jnp.logical_and(chosen, valid), w.astype(jnp.float32), 0.0)


def _quantize_blocks(x, q_ref, scale_ref):
    """x [m, h] bf16 to FP8 in q_ref as gmm_v2 quantizes its LHS: per row per
    _Q_BLOCK columns, scale = block amax / FP8 max and its inverse (0 for an
    all-zero block) in bf16, x * inverse in bf16, no clip (as
    moe_dense_expert._quantize). Block k's scales go to scale_ref[k] as an
    [m, lanes] plane, so each read is one contiguous tile."""
    m, h = x.shape
    lanes = scale_ref.shape[-1]
    for k in range(0, h, _Q_BLOCK):
        block = x[:, k : k + _Q_BLOCK]
        scale = jnp.max(jnp.abs(block), axis=1, keepdims=True) / FP8_MAX
        inv = jnp.where(scale == 0, 0, 1 / scale)
        q_ref[:, k : k + _Q_BLOCK] = (block * inv).astype(FP8)
        scale_ref[k // _Q_BLOCK] = jnp.broadcast_to(scale, (m, lanes))


def _shared_expert(xq_ref, xs, sw13_ref, ss13_ref, sw2_ref, ss2_ref, out_ref):
    """This rank's slice of the shared expert into out_ref (f32), as
    xla_quantized_matmul twice on x already quantized by quantize_block
    (xq_ref, row scales xs): f32 dot, x scale then weight scale; the usual TP
    path's bf16 roundings between the projections; the down projection stays
    f32."""
    shared_inter = sw2_ref.shape[0]
    gu = jnp.dot(xq_ref[...], sw13_ref[...], preferred_element_type=jnp.float32)
    gu = (gu * xs * ss13_ref[...]).astype(jnp.bfloat16).astype(jnp.float32)
    gate, up = gu[:, :shared_inter], gu[:, shared_inter:]
    act = jax.nn.silu(gate).astype(jnp.bfloat16).astype(jnp.float32) * up
    act_q, act_scale = quantize_block(
        act.astype(jnp.bfloat16), axis=1, target_dtype=FP8
    )
    y = jnp.dot(act_q, sw2_ref[...], preferred_element_type=jnp.float32)
    out_ref[...] = y * act_scale * ss2_ref[...]


def _combine(routed_sum, shared_sum, routed_scale):
    """The usual TP path's combine, roundings included:
    bf16(bf16(shared) + bf16(bf16(routed) * scale)), f32 arithmetic between the
    roundings (routed_scale is exact in bf16)."""
    routed = routed_sum.astype(jnp.bfloat16).astype(jnp.float32)
    scaled = (routed * routed_scale).astype(jnp.bfloat16).astype(jnp.float32)
    shared = shared_sum.astype(jnp.bfloat16).astype(jnp.float32)
    return (shared + scaled).astype(jnp.bfloat16)


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
    local_experts,
    topk,
    routed_scale,
    ar_spec,
):
    all_reduce.prologue(ar_spec)
    m, h = x_ref.shape
    i = w2_hbm.shape[1]
    nb = _NUM_BUFFERS
    n_valid = meta_ref[0]
    rank = meta_ref[1]

    # The shared slice does not depend on routing: load it first.
    shared_copies = (
        pltpu.make_async_copy(sw13_hbm, sw13_ref, shared_sem.at[0]),
        pltpu.make_async_copy(ss13_hbm, ss13_ref, shared_sem.at[1]),
        pltpu.make_async_copy(sw2_hbm, sw2_ref, shared_sem.at[2]),
        pltpu.make_async_copy(ss2_hbm, ss2_ref, shared_sem.at[3]),
    )
    for copy in shared_copies:
        copy.start()

    wtab_ref[...] = _route(scores_ref[...], bias_ref[...], n_valid, topk=topk)
    hit = jnp.any(wtab_ref[...] != 0, axis=0, keepdims=True)
    hits_v_ref[...] = jnp.where(hit, 1, 0).astype(jnp.int32)
    hit_copy = pltpu.make_async_copy(hits_v_ref, hits_s_ref, hit_sem.at[0])
    hit_copy.start()
    hit_copy.wait()

    def transfers(step):
        # Active expert `step`'s weights and scales into ring slot step % nb.
        e, slot = active_ids_ref[step], step % nb
        return [
            pltpu.make_async_copy(w1_hbm.at[e], w1_ref.at[slot], ring_sem.at[slot, 0]),
            pltpu.make_async_copy(w2_hbm.at[e], w2_ref.at[slot], ring_sem.at[slot, 1]),
            pltpu.make_async_copy(
                s1_hbm.at[e, 0], s1_ref.at[slot], ring_sem.at[slot, 2]
            ),
            pltpu.make_async_copy(
                s2_hbm.at[e, 0], s2_ref.at[slot], ring_sem.at[slot, 3]
            ),
        ]

    def compact(e, count):
        # Appends expert e to active_ids_ref if any row picked it. The store
        # is unconditional, so the loop body has no branch: count <= e, a
        # later hit overwrites the entry, and no read touches one >= count.
        hit = hits_s_ref[0, e] != 0
        active_ids_ref[count] = e
        return count + hit.astype(jnp.int32)

    count = lax.fori_loop(0, local_experts, compact, jnp.int32(0))

    def prime(step):
        @pl.when(step < count)
        def start():
            for copy in transfers(step):
                copy.start()

    # The first nb active experts start their weight copies.
    for s in range(nb):
        prime(jnp.int32(s))

    # The shared expert's copies started at entry. Waiting for them here,
    # before the zeroing and the x quantization, leaves no DMA wait between
    # those steps and the shared expert's matmuls.
    for copy in shared_copies:
        copy.wait()

    # Zeroing and x quantization follow the compaction, so the first weight
    # copies are already in flight while they run. x is quantized twice on
    # purpose, each copy as its matmuls quantize it with this kernel off: per
    # row for the shared expert (xla_quantized_matmul), per _Q_BLOCK columns
    # for the routed experts (gmm_v2).
    owner_ref[...] = jnp.zeros((m, h), jnp.float32)
    sxq, sxs = quantize_block(x_ref[...], axis=1, target_dtype=FP8)
    sxq_ref[...] = sxq
    _shared_expert(sxq_ref, sxs, sw13_ref, ss13_ref, sw2_ref, ss2_ref, shared_ref)
    _quantize_blocks(x_ref[...], xq_ref, xs_ref)

    expert_columns = lax.broadcasted_iota(jnp.int32, wtab_ref.shape, 1)
    lanes = xs_ref.shape[-1]

    def gate_up(slot):
        # gmm_v2's quantized matmul: one partial product per _Q_BLOCK rows of
        # w1, rounded to bf16, times its x block's scales and then the weight
        # scales in bf16, summed in bf16 (moe_dense_expert's pattern).
        w_scale = s1_ref[slot].astype(jnp.bfloat16)
        acc = jnp.zeros((m, 2 * i), jnp.bfloat16)
        for k in range(0, h, _Q_BLOCK):
            part = jnp.matmul(
                xq_ref[:, k : k + _Q_BLOCK],
                w1_ref[slot, k : k + _Q_BLOCK, :],
                preferred_element_type=jnp.float32,
            ).astype(jnp.bfloat16)
            part *= jnp.concatenate([xs_ref[k // _Q_BLOCK]] * (2 * i // lanes), axis=1)
            part *= w_scale
            acc += part
        return acc

    def finish(step, slot, acc1):
        e = active_ids_ref[step]
        # Each row's routing weight for expert e (column e of the routing
        # table, 0 for rows that did not pick it).
        coeff = jnp.sum(
            jnp.where(expert_columns == e, wtab_ref[...], 0.0),
            axis=1,
            keepdims=True,
        )
        # gmm_v2's fused SiLU, on the bf16 gate_up result.
        act = jax.nn.silu(acc1[:, :i]) * acc1[:, i:]
        # Mask, not multiply: a padded row with non-finite hidden would give
        # NaN * 0 = NaN. Rows this expert does not own contribute exactly 0.
        mid = jnp.where(coeff != 0, act.astype(jnp.float32) * coeff, 0.0)
        ms = jnp.max(jnp.abs(mid), axis=1, keepdims=True) / FP8_MAX
        mq = (mid * jnp.where(ms == 0, 0.0, 1.0 / ms)).astype(FP8)
        for n in range(0, h, _N_CHUNK):
            acc2 = jnp.matmul(
                mq,
                w2_ref[slot, :, n : n + _N_CHUNK],
                preferred_element_type=jnp.float32,
            )
            owner_ref[:, n : n + _N_CHUNK] += (
                acc2 * s2_ref[slot, :, n : n + _N_CHUNK] * ms
            )

    @pl.when(count > 0)
    def run_experts():
        # The prime loop after the compaction started the first nb experts'
        # copies.
        for copy in transfers(jnp.int32(0)):
            copy.wait()

        def expert_body(step, acc1):
            nxt = step + 1

            @pl.when(nxt < count)
            def wait_next():
                for copy in transfers(nxt):
                    copy.wait()

            # Issued before this expert's elementwise chain so the MXU and VPU
            # overlap. On the last step it reads a stale slot; the result is
            # dropped.
            acc1_next = gate_up(nxt % nb)
            finish(step, step % nb, acc1)

            # This expert's slot is free once its second matmul has read it.
            @pl.when(step + nb < count)
            def refill():
                for copy in transfers(step + nb):
                    copy.start()

            return acc1_next

        lax.fori_loop(0, count, expert_body, gate_up(0))

    # Outside pl.when(count > 0): every rank must take part in the exchange
    # (its peers wait for its chunks), consume the barrier signals sent at
    # entry and write out_ref.
    all_reduce.epilogue(
        ar_spec,
        rank,
        n_valid,
        _combine(owner_ref[...], shared_ref[...], routed_scale),
        out_ref,
        ar_scr,
    )


@functools.partial(jax.jit, static_argnames=("routed_scale", "topk", "n_ranks"))
def moe_tp_decode(
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
    *,
    routed_scale: float,
    topk: int,
    n_ranks: int,
):
    """The layer's bf16 [N, D] output, summed over the n_ranks TP ranks.

    Args:
      x: [N, D] bf16 hidden states (the same on every rank).
      scores: [N, E] f32 router scores (sigmoid of the logits).
      bias: [E] f32 e_score_correction_bias.
      n_valid: int32 scalar, rows [0, n_valid) are real tokens; the same on
        every rank.
      rank: int32 scalar, this device's index in the 1-D mesh of n_ranks.
      w13, w2: [E, D, 2I] / [E, I, D] fp8, this rank's columns of every
        routed expert ([gate | up] for w13).
      s13, s2: [E, 1, 1, 2I] / [E, 1, 1, D] f32 per-output-channel scales.
      shared_w13, shared_w2: [D, 2Fs] / [Fs, D] fp8, this rank's slice of the
        shared expert; shared_s13, shared_s2: [2Fs] / [D] f32 scales.
      routed_scale: routed_scaling_factor, exact in bf16.
      topk: experts per token.
      n_ranks: TP ranks.

    Returns:
      bf16(sum over ranks of each rank's shared + routed_scale * routed),
      the same on every rank, rows >= n_valid exactly +0.

    Callers check can_use_moe_tp_decode first, call this under shard_map
    over the 1-D mesh (one call per rank), and run the all-reduce self-test
    once at start-up: a wrong rank hangs the kernel.
    """
    if not exact_in_bf16(routed_scale):
        raise ValueError(f"routed_scale {routed_scale!r} is not exact in bf16")
    m, h = x.shape
    experts, i, _ = w2.shape
    num_experts = scores.shape[1]
    shared_inter = shared_w2.shape[0]
    info = pltpu.get_tpu_info()
    ar_spec = all_reduce.make_spec(n_ranks, m, h)
    plan = _vmem_plan(m, h, i, num_experts, shared_inter, info, ar_spec)
    if plan is None:
        raise ValueError("moe_tp_decode does not fit VMEM; check can_use_moe_tp_decode")
    _, limit = plan
    nb = _NUM_BUFFERS
    # Lane-dense routing table. Padded experts never win: their biased score
    # is -inf, clamped to ROUTER_NEG, and every real column has a lower index.
    e_pad = pl.cdiv(num_experts, info.num_lanes) * info.num_lanes
    scores = scores.astype(jnp.float32)
    bias = bias.astype(jnp.float32)
    if e_pad != num_experts:
        scores = jnp.pad(scores, ((0, 0), (0, e_pad - num_experts)))
        bias = jnp.pad(bias, (0, e_pad - num_experts), constant_values=-jnp.inf)
    meta = jnp.stack(
        [
            jnp.asarray(n_valid, jnp.int32).reshape(()),
            jnp.asarray(rank, jnp.int32).reshape(()),
        ]
    )
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        functools.partial(
            _kernel,
            local_experts=experts,
            topk=topk,
            routed_scale=float(routed_scale),
            ar_spec=ar_spec,
        ),
        out_shape=jax.ShapeDtypeStruct((m, h), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vmem, vmem, vmem, *[hbm] * 8],
            out_specs=vmem,
            scratch_shapes=[
                pltpu.VMEM((nb, h, 2 * i), FP8),
                pltpu.VMEM((nb, i, h), FP8),
                pltpu.VMEM((nb, 1, 2 * i), jnp.float32),
                pltpu.VMEM((nb, 1, h), jnp.float32),
                pltpu.VMEM((h, 2 * shared_inter), FP8),
                pltpu.VMEM((1, 2 * shared_inter), jnp.float32),
                pltpu.VMEM((shared_inter, h), FP8),
                pltpu.VMEM((1, h), jnp.float32),
                pltpu.VMEM((m, h), FP8),
                pltpu.VMEM((m, h), FP8),
                pltpu.VMEM((h // _Q_BLOCK, m, info.num_lanes), jnp.bfloat16),
                pltpu.VMEM((m, h), jnp.float32),
                pltpu.VMEM((m, h), jnp.float32),
                pltpu.VMEM((m, e_pad), jnp.float32),
                pltpu.VMEM((1, e_pad), jnp.int32),
                pltpu.SMEM((1, e_pad), jnp.int32),
                pltpu.SMEM((experts,), jnp.int32),
                pltpu.SemaphoreType.DMA((nb, 4)),
                pltpu.SemaphoreType.DMA((4,)),
                pltpu.SemaphoreType.DMA((1,)),
                *all_reduce.scratch_shapes(ar_spec),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=limit, collective_id=all_reduce.COLLECTIVE_ID
        ),
        name=kernel_name(experts, m, h, i, shared_inter, n_ranks),
    )(
        meta,
        x,
        scores,
        bias.reshape(1, e_pad),
        w13,
        w2,
        s13,
        s2,
        shared_w13,
        shared_s13.reshape(1, 2 * shared_inter),
        shared_w2,
        shared_s2.reshape(1, h),
    )

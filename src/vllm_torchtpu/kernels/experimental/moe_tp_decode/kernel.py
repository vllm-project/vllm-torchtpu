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

Fast schedule (fast=True, TPU_MOE_TP_DECODE_FAST): the same arithmetic and the
same output bits, scheduled for the hardware. Routing reduces one per-row
unique key per pick (biased score, ties to the lower column, packed in one f32)
instead of the max, the lowest attaining column and the pick's score (one
dependent cross-lane reduction per pick instead of three); only a row whose
picks reach below the key's window (2^22 f32 ordinals under the row maximum)
reruns the exact routing. The
compaction runs on the vector units (_compact_vector) and travels to the scalar
core with the hit table, and one loop after it starts the first ring items: the
incumbent's scalar compaction costs ~20 bundles per local expert, and its DMA
starts inside a pl.when compile to predicated straight-line code on every hit
expert that does not overlap the work after it. The ring loop runs in two parts
with unconditional waits and refills, and the last item is peeled, so no ring
DMA sits under a pl.when and no matmul reads a stale slot.

Expert range (EP x TP weights, e.g. TPU_MOE_EP_TP_SIZE): a rank may hold only
the experts [experts_start, experts_start + local) of the E the router scores,
each at an I-column TP slice. Routing still runs over all E columns; only the
local experts are compacted, and a ring item is one (expert, _I_CHUNK-column
chunk of I) pair: the gate and up chunks of w13 and the matching w2 rows, so
VMEM per slot is the TP16 slot whatever I is. The all-reduce still sums every
rank, so each rank's partial covers its experts x its columns.
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
# _route_fast: routing keys are positive normal f32 bit patterns from here.
KEY_BASE = 0x00800000
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
# Intermediate columns per ring item (one MXU-wide gate and up chunk).
_I_CHUNK = 128


def exact_in_bf16(value: float) -> bool:
    """True if `value` is a bf16 number, so bf16 and f32 products agree."""
    rounded = np.asarray(value, dtype=jnp.bfloat16).astype(np.float32)
    return float(rounded) == float(value)


def _split_meta():
    """TPU_MOE_TP_DECODE_SPLIT_META: n_valid / rank / e0 as [1] operands."""
    from vllm_torchtpu import envs

    return bool(envs.TPU_MOE_TP_DECODE_SPLIT_META)


class _Meta:
    """meta_ref[i] over three [1] scalar refs (n_valid, rank, e0)."""

    def __init__(self, refs):
        self.refs = refs

    def __getitem__(self, i):
        return self.refs[i][0]


def _split_entry(n_valid_ref, rank_ref, e0_ref, *refs, **kwargs):
    return _kernel(_Meta((n_valid_ref, rank_ref, e0_ref)), *refs, **kwargs)


def _i_chunk(i):
    """Intermediate columns per ring item for a rank holding I = i."""
    return min(i, _I_CHUNK)


def _vmem_plan(m, h, i, num_experts, shared_inter, info, spec):
    """(need_bytes, vmem_limit_bytes), or None if the limit would exceed
    VMEM_FRACTION of the capacity. Whole-array blocks (grid=()), tiles padded.
    `i` is the ring item's column count (_i_chunk of the rank's I)."""
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


def kernel_name(
    experts,
    m,
    h,
    i,
    shared_inter,
    n_ranks,
    num_experts=None,
    fast=False,
    ar_one_shot=False,
) -> str:
    name = (
        f"moe_tp_decode-e_{experts}-m_{m}-h_{h}-i_{i}-fs_{shared_inter}"
        f"-nb_{_NUM_BUFFERS}-n_{n_ranks}"
    )
    if _split_meta():
        name += "-sm"
    if num_experts is not None and num_experts != experts:
        # Expert range: `experts` of the router's num_experts.
        name += f"-ne_{num_experts}"
    if fast:
        name += "-fast"
    if ar_one_shot:
        name += "-os"
    return name


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
    ar_one_shot: bool = False,
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
    # This rank's experts are a linear block of the router's num_experts
    # (experts == num_experts under plain TP).
    if bias.shape != (num_experts,) or experts <= 0 or num_experts % experts:
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
    if ar_one_shot and not all_reduce.one_shot_supported(h, info):
        return False
    spec = all_reduce.make_spec(n_ranks, m, h, one_shot=ar_one_shot)
    plan = _vmem_plan(m, h, _i_chunk(i), num_experts, shared_inter, info, spec)
    return plan is not None


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


def _route_fast(scores, bias, n_valid, *, topk):
    """_route with one cross-lane reduction per pick instead of three, ties
    included. Each column gets a key that is unique in its row and orders
    columns as _route picks them -- larger biased score first, the lower
    column on equal scores -- so a pick is the one column equal to the row's
    largest key and the picks come in _route's order (same wsum additions).

    The key is a positive normal f32 whose bits are KEY_BASE + (rank << lb) +
    (n - 1 - column), lb = bits of a column index: `rank` is the biased score's
    distance in f32 ordinals below the row maximum, inverted, over a window of
    2^(30 - lb) - 2 ordinals; a column beyond the window gets rank 0. Returns
    (weights, flag); the weights equal _route's bit for bit unless some valid
    row's topk-th pick lies beyond the window (flag ([1, 1] f32) = 1, the
    caller must use _route): its picks would need ordinals the key drops. With
    E = 256 the window is 2^22 ordinals (e.g. 4.0 below a row maximum in
    [8, 16), 0.25 in [0.5, 1)). NaN scores clamp to ROUTER_NEG as in _route."""
    rows, n = scores.shape
    lb = max(1, (n - 1).bit_length())
    window = (1 << (30 - lb)) - 1  # ranks 1 .. window - 1 in the window, 0 beyond
    x = scores + bias
    x = jnp.where(x > ROUTER_NEG, x, ROUTER_NEG)
    x = jnp.where(x == 0.0, 0.0, x)  # -0.0 == +0.0 in _route: one ordinal
    xm = jnp.max(x, axis=1, keepdims=True)

    def ordinal(v):  # monotone f32 -> i32, equal floats <-> equal ordinals
        b = lax.bitcast_convert_type(v, jnp.int32)
        return jnp.where(b < 0, b ^ 0x7FFFFFFF, b)

    d = ordinal(xm) - ordinal(x)  # >= 0; wraps negative beyond 2^31
    inside = jnp.logical_and(d >= 0, d < window - 1)
    rank = jnp.where(inside, (window - 1) - d, 0)
    lane = lax.broadcasted_iota(jnp.int32, (rows, n), 1)
    key = lax.bitcast_convert_type(
        KEY_BASE + jnp.left_shift(rank, lb) + ((n - 1) - lane), jnp.float32
    )
    wsum = jnp.zeros((rows, 1), jnp.float32)
    km = None
    for _ in range(topk):
        km = jnp.max(key, axis=1, keepdims=True)
        pick = key == km
        # Exactly one nonzero term: the unbiased score of this pick (_route's).
        wsum = wsum + jnp.sum(jnp.where(pick, scores, 0.0), axis=1, keepdims=True)
        key = jnp.where(pick, 0.0, key)  # below every key (all >= 2^-126)
    chosen = key == 0.0
    denom = jnp.where(wsum < SUM_FLOOR, SUM_FLOOR, wsum)
    w = (jnp.where(chosen, scores, 0.0) / denom).astype(jnp.bfloat16)
    valid = lax.broadcasted_iota(jnp.int32, (rows, n), 0) < n_valid
    # The last pick's rank is 0 iff it (and so some pick) lay beyond the window.
    assert km is not None  # topk >= 1
    last_bits = lax.bitcast_convert_type(km, jnp.int32)
    beyond = last_bits < KEY_BASE + (1 << lb)
    valid_row = lax.broadcasted_iota(jnp.int32, (rows, 1), 0) < n_valid
    bad = jnp.where(jnp.logical_and(valid_row, beyond), 1.0, 0.0)
    flag = jnp.max(bad, axis=0, keepdims=True)
    return jnp.where(jnp.logical_and(chosen, valid), w.astype(jnp.float32), 0.0), flag


def _compact_vector(wtab, e0, local):
    """The compaction on the vector units: ([1, W] f32, [1, 1] f32) = the local
    ids (0 <= id < local) of the experts any row of wtab picked, ascending, in
    lanes [0, count), and count; W = local rounded up to the lane count.

    A one-hot matmul moves the window [e0, e0 + local) of the hit row to lanes
    [0, local) (e0 is runtime data); expert e is the j-th active one when j
    experts below it are hit and it is hit. All sums are of 0/1 values or of
    one id, so f32 holds them exactly."""
    f32 = jnp.float32
    e_pad = wtab.shape[1]
    lanes = 128
    w = pl.cdiv(local, lanes) * lanes
    hit = jnp.any(wtab != 0, axis=0, keepdims=True).astype(f32)
    rows = lax.broadcasted_iota(jnp.int32, (e_pad, w), 0)
    cols = lax.broadcasted_iota(jnp.int32, (e_pad, w), 1)
    window = jnp.where(jnp.logical_and(rows == cols + e0, cols < local), 1.0, 0.0)
    loc = jnp.dot(
        jnp.broadcast_to(hit, (8, e_pad)), window, preferred_element_type=f32
    )[0:1]
    sub = lax.broadcasted_iota(jnp.int32, (w, w), 0)
    lane = lax.broadcasted_iota(jnp.int32, (w, w), 1)
    loc_b = jnp.broadcast_to(loc, (w, w))
    below = jnp.sum(jnp.where(lane < sub, loc_b, 0.0), axis=1, keepdims=True)
    own = jnp.sum(jnp.where(lane == sub, loc_b, 0.0), axis=1, keepdims=True)
    nth = jnp.logical_and(below == lane.astype(f32), own > 0)
    ids = jnp.sum(jnp.where(nth, sub.astype(f32), 0.0), axis=0, keepdims=True)
    return ids, jnp.sum(loc, axis=1, keepdims=True)


def _unused(fn):
    """Decorator that leaves fn untraced (the schedule not selected)."""
    return fn


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
    n_chunks,
    topk,
    routed_scale,
    ar_spec,
    fast=False,
):
    all_reduce.prologue(ar_spec)
    m, h = x_ref.shape
    i = w2_hbm.shape[1]  # this rank's I
    ic = w2_ref.shape[1]  # columns per ring item; i == ic * n_chunks
    nc = n_chunks
    nb = _NUM_BUFFERS
    n_valid = meta_ref[0]
    rank = meta_ref[1]
    # First global expert this rank holds (0 under plain TP).
    e0 = meta_ref[2]

    # The shared slice does not depend on routing: load it first.
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
        # Rows 2 and 3 of the hit table: the active local experts, ascending,
        # and their count (the vector compaction replaces the scalar one).
        ids, n_active = _compact_vector(wtab_ref[...], e0, local_experts)
        hits_v_ref[2:3, 0 : ids.shape[1]] = ids.astype(jnp.int32)
        hits_v_ref[3:4, :] = jnp.broadcast_to(
            n_active, (1, hits_v_ref.shape[1])
        ).astype(jnp.int32)

    if fast:
        # Row 1 of the hit table carries _route_fast's flag to the scalar core.
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

    def active(j):
        """Local id of the j-th active expert."""
        return hits_s_ref[2, j] if fast else active_ids_ref[j]

    def transfers(step):
        if nc > 1:
            return chunk_transfers(step)
        # Active expert `step`'s weights and scales into ring slot step % nb.
        e, slot = active(step), step % nb
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

    def chunk_transfers(step):
        # Ring item `step` is chunk step % nc of active expert step // nc:
        # w13's gate and up column chunks side by side, the matching w2 rows,
        # their scales, and the expert's w2 scale.
        e = active(lax.div(step, nc))
        c0 = pl.multiple_of(lax.rem(step, nc) * ic, ic)
        slot = step % nb
        sem = ring_sem.at[slot]
        lo, hi = pl.ds(0, ic), pl.ds(ic, ic)
        gate, up = pl.ds(c0, ic), pl.ds(i + c0, ic)
        return [
            pltpu.make_async_copy(
                w1_hbm.at[e, :, gate], w1_ref.at[slot, :, lo], sem.at[0]
            ),
            pltpu.make_async_copy(
                w1_hbm.at[e, :, up], w1_ref.at[slot, :, hi], sem.at[1]
            ),
            pltpu.make_async_copy(w2_hbm.at[e, gate], w2_ref.at[slot], sem.at[2]),
            pltpu.make_async_copy(
                s1_hbm.at[e, 0, :, gate], s1_ref.at[slot, :, lo], sem.at[3]
            ),
            pltpu.make_async_copy(
                s1_hbm.at[e, 0, :, up], s1_ref.at[slot, :, hi], sem.at[4]
            ),
            pltpu.make_async_copy(s2_hbm.at[e, 0], s2_ref.at[slot], sem.at[5]),
        ]

    def compact(e, count):
        # Appends expert e to active_ids_ref if any row picked it. The store
        # is unconditional, so the loop body has no branch: count <= e, a
        # later hit overwrites the entry, and no read touches one >= count.
        # Local expert e is global column e0 + e.
        hit = hits_s_ref[0, e0 + e] != 0
        active_ids_ref[count] = e
        return count + hit.astype(jnp.int32)

    if fast:
        # The vector compaction's count, row 3 of the hit table.
        count = hits_s_ref[3, 0]
    else:
        count = lax.fori_loop(0, local_experts, compact, jnp.int32(0))
    # Ring items, nc per active expert.
    items = count * nc

    if fast:
        # One loop starts the first nb ring items, with no DMA under a
        # pl.when.

        def prime_fast(item, carry):
            for copy in transfers(item):
                copy.start()
            return carry

        lax.fori_loop(0, jnp.minimum(items, nb), prime_fast, 0)
    else:

        def prime(step):
            @pl.when(step < items)
            def start():
                for copy in transfers(step):
                    copy.start()

        # The first nb ring items start their weight copies.
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
        acc = jnp.zeros((m, 2 * ic), jnp.bfloat16)
        for k in range(0, h, _Q_BLOCK):
            part = jnp.matmul(
                xq_ref[:, k : k + _Q_BLOCK],
                w1_ref[slot, k : k + _Q_BLOCK, :],
                preferred_element_type=jnp.float32,
            ).astype(jnp.bfloat16)
            part *= jnp.concatenate([xs_ref[k // _Q_BLOCK]] * (2 * ic // lanes), axis=1)
            part *= w_scale
            acc += part
        return acc

    def finish(step, slot, acc1):
        e = active(lax.div(step, nc)) if nc > 1 else active(step)
        # Each row's routing weight for expert e (global column e0 + e of the
        # routing table, 0 for rows that did not pick it).
        coeff = jnp.sum(
            jnp.where(expert_columns == e0 + e, wtab_ref[...], 0.0),
            axis=1,
            keepdims=True,
        )
        # gmm_v2's fused SiLU, on the bf16 gate_up result.
        act = jax.nn.silu(acc1[:, :ic]) * acc1[:, ic:]
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

    def ring_step(step, acc1, refill):
        """Wait for item step + 1, issue its gate_up, finish item step, and
        (refill) start item step + nb into the slot item step leaves."""
        nxt = step + 1
        for copy in transfers(nxt):
            copy.wait()
        acc1_next = gate_up(nxt % nb)
        finish(step, step % nb, acc1)
        if refill:
            for copy in transfers(step + nb):
                copy.start()
        return acc1_next

    # Only one schedule is traced: the other function is defined, not called.
    @(pl.when(items > 0) if fast else _unused)
    def run_experts_fast():
        for copy in transfers(jnp.int32(0)):
            copy.wait()
        # Steps [0, items - nb) refill; the rest only drain; the last item is
        # finished outside the loops (no gate_up past it).
        n_refill = jnp.maximum(items - nb, 0)
        acc1 = lax.fori_loop(
            0, n_refill, lambda s, a: ring_step(s, a, True), gate_up(0)
        )
        acc1 = lax.fori_loop(
            n_refill, items - 1, lambda s, a: ring_step(s, a, False), acc1
        )
        finish(items - 1, (items - 1) % nb, acc1)

    @(_unused if fast else pl.when(items > 0))
    def run_experts():
        # The prime loop after the compaction started the first nb experts'
        # copies.
        for copy in transfers(jnp.int32(0)):
            copy.wait()

        def expert_body(step, acc1):
            nxt = step + 1

            @pl.when(nxt < items)
            def wait_next():
                for copy in transfers(nxt):
                    copy.wait()

            # Issued before this expert's elementwise chain so the MXU and VPU
            # overlap. On the last step it reads a stale slot; the result is
            # dropped.
            acc1_next = gate_up(nxt % nb)
            finish(step, step % nb, acc1)

            # This expert's slot is free once its second matmul has read it.
            @pl.when(step + nb < items)
            def refill():
                for copy in transfers(step + nb):
                    copy.start()

            return acc1_next

        lax.fori_loop(0, items, expert_body, gate_up(0))

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


@functools.partial(
    jax.jit,
    static_argnames=(
        "routed_scale",
        "topk",
        "n_ranks",
        "fast",
        "hbm_operands",
        "ar_one_shot",
    ),
)
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
    experts_start=0,
    *,
    routed_scale: float,
    topk: int,
    n_ranks: int,
    fast: bool = False,
    hbm_operands: bool = False,
    ar_one_shot: bool = False,
):
    """The layer's bf16 [N, D] output, summed over the n_ranks TP ranks.

    Args:
      x: [N, D] bf16 hidden states (the same on every rank).
      scores: [N, E] f32 router scores (sigmoid of the logits).
      bias: [E] f32 e_score_correction_bias.
      n_valid: int32 scalar, rows [0, n_valid) are real tokens; the same on
        every rank.
      rank: int32 scalar, this device's index in the 1-D mesh of n_ranks.
      w13, w2: [El, D, 2I] / [El, I, D] fp8, this rank's columns of routed
        experts [experts_start, experts_start + El) ([gate | up] for w13);
        El == E under plain TP.
      s13, s2: [El, 1, 1, 2I] / [El, 1, 1, D] f32 per-output-channel scales.
      shared_w13, shared_w2: [D, 2Fs] / [Fs, D] fp8, this rank's slice of the
        shared expert; shared_s13, shared_s2: [2Fs] / [D] f32 scales.
      experts_start: int32 scalar, the global id of this rank's first expert
        (runtime data, so every rank compiles the same program).
      routed_scale: routed_scaling_factor, exact in bf16.
      topk: experts per token.
      n_ranks: TP ranks.
      fast: the fast schedule (module docstring); the same output bits.
      hbm_operands: constrain the 8 weight operands to HBM
        (``pltpu.with_memory_space_constraint``). The in_specs only type the
        kernel refs; without the constraint XLA may stage a whole weight into
        VMEM with a copy-start/-done before the call (exposed in the MTP
        drafter's one-layer program), though the kernel DMAs the rows it uses
        itself. Same bytes read, same output bits.
      ar_one_shot: the all-reduce's one-row path (all_reduce.py): a step with
        one valid row exchanges it in one round instead of two. Same bits.

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
    ar_spec = all_reduce.make_spec(n_ranks, m, h, one_shot=ar_one_shot)
    ic = _i_chunk(i)
    plan = _vmem_plan(m, h, ic, num_experts, shared_inter, info, ar_spec)
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
    split = _split_meta()
    if split:
        # UPDATE (TPU_MOE_TP_DECODE_SPLIT_META): three [1] scalar operands, as
        # small_all_reduce_valid_rows passes n_valid and rank, instead of one
        # stacked int32[3] that XLA materializes with its own fusion per call.
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
        s13,
        s2,
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
            _split_entry if split else _kernel,
            local_experts=experts,
            n_chunks=i // ic,
            topk=topk,
            routed_scale=float(routed_scale),
            ar_spec=ar_spec,
            fast=fast,
        ),
        out_shape=jax.ShapeDtypeStruct((m, h), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(metas),
            in_specs=[vmem, vmem, vmem, *[hbm] * 8],
            out_specs=vmem,
            scratch_shapes=[
                pltpu.VMEM((nb, h, 2 * ic), FP8),
                pltpu.VMEM((nb, ic, h), FP8),
                pltpu.VMEM((nb, 1, 2 * ic), jnp.float32),
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
                pltpu.VMEM((4 if fast else 1, e_pad), jnp.int32),
                pltpu.SMEM((4 if fast else 1, e_pad), jnp.int32),
                pltpu.SMEM((experts,), jnp.int32),
                pltpu.SemaphoreType.DMA((nb, 4 if i == ic else 6)),
                pltpu.SemaphoreType.DMA((4,)),
                pltpu.SemaphoreType.DMA((1,)),
                *all_reduce.scratch_shapes(ar_spec),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=limit, collective_id=all_reduce.COLLECTIVE_ID
        ),
        name=kernel_name(
            experts, m, h, i, shared_inter, n_ranks, num_experts, fast, ar_one_shot
        ),
    )(
        *metas,
        x,
        scores,
        bias.reshape(1, e_pad),
        *weights,
    )

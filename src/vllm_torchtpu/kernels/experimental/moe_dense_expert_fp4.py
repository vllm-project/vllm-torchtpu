# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Small-batch NVFP4 (W4A16) MoE without token permutation.

The decode-sized counterpart of moe_dense_expert for fp4 experts with fp32
[128, 1] block scales. Each nonempty local expert is streamed once, in units of
`chunk` intermediate columns (its gate, up and down slices), through a ring of
VMEM buffers (a TP-sliced expert narrow enough is one unit, its [gate | up]
columns one copy); every token is evaluated in its original order and the weighted
expert outputs accumulate in fp32 VMEM. This replaces the routing sort, the
one-hot permute, the grouped GEMM and the combine with one call; the routing
coefficients and the set of nonempty experts are derived in the kernel from the
raw top-k ids and weights (an expert counts as nonempty when any id names it,
even if its routing weights are zero). The fp4 weights feed the MXU as fp8
(exact) against bf16 activations, and each 128-row block scale is applied to
the block's partial product, as gmm_v2's decode tiles do. Rounding follows the
GMM path: the activation is rounded to bf16 before the down projection, each
expert's output to bf16 before its routing weight is applied.

Optionally the call also evaluates an always-on shared expert (`SharedExpert`)
and returns routed + shared: its e4m3 weights are copied into VMEM alongside
the routed ring, so their DMA overlaps the routed experts' instead of following
them as separate XLA matmuls.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox.gmm_v2 import apply_act_fn, interleave_lane

_QUANT_BLOCK = 128
# Weight-ring slots (units in flight = slots - 1). Whole TP-sliced experts
# are small units: one more slot keeps more of their DMAs in flight (c4 twin:
# -3.5 % kernel time; 6 slots no better), where the width allows it.
_NUM_BUFFERS = 3
_WHOLE_UNIT_BUFFERS = 4
_ACTIVATIONS = ("silu", "swigluoai")
# Rows of the in-kernel token buffer: bf16 MXU operands pack 16 sublanes.
_ROW_ALIGN = 16
_SHARED_K_BLOCK = 128
# Output columns per shared-expert down-projection block in sparse mode
# (bounds its f32 temporary).
_SHARED_COLS = 512
# Rows per block of the sparse mode's f32 token copy.
_TOKEN_COPY_ROWS = 64
# Sparse mode groups tokens by expert on one vreg's lanes (`_group_by_expert`).
_SPARSE_MAX_EXPERTS = 128
# Dense mode with more local experts than this (and at most one vreg's lanes)
# finds the nonempty experts with vector ops and one SMEM copy instead of
# scalar loops over every route and every expert, which sit on the critical
# path ahead of the first weight copy (TP-sliced 128 experts: 2.5-3.8 us less
# per call at 8-32 tokens; 16 EP experts: within 1 us either way).
_SCALAR_SELECT_MAX_EXPERTS = 32
# MiniMax-M3's shared-expert SwiGLU-OAI clamp and sigmoid scale (1.702 in
# bf16); dense_fp4_shared admits only shared experts with these values.
_SWIGLU_LIMIT = 7.0
_SWIGLU_ALPHA = 1.703125
# VMEM for the kernel's small operands.
_VMEM_HEADROOM = 1 << 20
# The call's VMEM limit over its planned bytes: the rest stays free for XLA
# to keep the next ops' prefetches (the next layer's qkv weight) in VMEM
# across the call; a limit of all of VMEM left it none (c4 twin j-de6ab234).
_VMEM_MARGIN = 4 << 20


class Route(NamedTuple):
    """Sigmoid-score routing evaluated in the kernel (`dense_expert_moe_fp4`'s
    ``route``), as moe_routing.select_experts computes it with the rowmax
    top-k: the top ``topk`` of the scores plus correction bias (ties to the
    lowest id), their scores renormalized to sum 1, times ``scale``, rounded to
    ``weight_dtype``."""

    topk: int
    scale: float
    weight_dtype: str


# router_topk.NEG: below every routing score, above the mask of a taken column.
_ROUTE_NEG = -3.0e38


def _route(scores, choice, start, num_local, rows_real, route: Route):
    """The kernel's routing of [rows, E] f32 scores / scores + bias: per route,
    the local expert id ([rows, 1] int32, -1 off this rank or past
    ``rows_real``) and its routing weight ([rows, 1] f32, 0 where the id is
    -1). The selection is router_topk's max-and-mask."""
    f32 = jnp.float32
    rows, n = choice.shape
    x = jnp.where(choice > _ROUTE_NEG, choice, _ROUTE_NEG)
    masked = jnp.finfo(f32).min
    lanes = lax.broadcasted_iota(jnp.int32, (rows, n), 1)
    ids, weights = [], []
    for _ in range(route.topk):
        top = jnp.max(x, axis=1, keepdims=True)
        idx = jnp.min(jnp.where(x == top, lanes, n - 1), axis=1, keepdims=True)
        hit = lanes == idx
        weights.append(jnp.sum(jnp.where(hit, scores, 0), axis=1, keepdims=True))
        ids.append(idx)
        x = jnp.where(hit, masked, x)
    # XLA's order for the sum over the top-k axis on TPU: halves added
    # pairwise ((w0 + w2) + (w1 + w3) at k = 4).
    parts = list(weights)
    while len(parts) > 1:
        half = (len(parts) + 1) // 2
        parts = [p + q for p, q in zip(parts[:half], parts[half:])] + parts[
            len(parts) - half : half
        ]
    total = jnp.maximum(parts[0], 1e-20)
    dtype = jnp.dtype(route.weight_dtype)
    row = lax.broadcasted_iota(jnp.int32, (rows, 1), 0)
    out_ids, out_weights = [], []
    for idx, w in zip(ids, weights):
        w = w / total
        if route.scale != 1.0:
            w = w * route.scale
        w = w.astype(dtype).astype(f32)
        local = idx - start
        valid = (local >= 0) & (local < num_local) & (row < rows_real)
        out_ids.append(jnp.where(valid, local, -1))
        out_weights.append(jnp.where(valid, w, 0))
    return out_ids, out_weights


class SharedExpert(NamedTuple):
    """A rank's slice of an always-on SwiGLU-OAI MLP with e4m3 K-major weights
    and one f32 scale per output column, rounded as torch_tpu evaluates
    MiniMax-M3's shared expert on bf16 tensors: each projection
    ``bf16(bf16(x @ bf16(w)) * s)``, the activation per op
    (`shared_activation`), and the bf16 sum with the routed output."""

    w1: jax.Array  # [h, 2 * i_s] e4m3: [gate | up]
    s1: jax.Array  # [1, 2 * i_s] f32
    w2: jax.Array  # [i_s, h] e4m3
    s2: jax.Array  # [1, h] f32


def shared_activation(gate: jax.Array, up: jax.Array) -> jax.Array:
    """SiluAndMulWithClamp(7, 1.702, 1) on bf16 values held in f32, rounded to
    bf16 after every op as torch_tpu does (alpha too): bit-identical to the
    torch path on tpu7x."""

    def bf16(v):
        return v.astype(jnp.bfloat16).astype(jnp.float32)

    gate = jnp.minimum(gate, _SWIGLU_LIMIT)
    up = jnp.clip(up, -_SWIGLU_LIMIT, _SWIGLU_LIMIT)
    sig = bf16(jax.nn.sigmoid(bf16(_SWIGLU_ALPHA * gate)))
    return bf16(bf16(gate * sig) * bf16(up + 1.0))


def _chunk(
    m: int, h: int, i: int, i_s: int, vmem_capacity: int, sparse: bool = False
) -> int:
    """The widest unit of intermediate columns whose buffers fit in VMEM, or 0."""
    budget = vmem_capacity - _VMEM_HEADROOM
    for c in (1024, 768, 512, 384, 256, 128):
        if i % c == 0 and _vmem_bytes(m, h, c, i_s, sparse=sparse) <= budget:
            return c
    return 0


def _shared_ok(shared: SharedExpert | None, h: int, activation: str) -> bool:
    if shared is None:
        return True
    fp8 = jnp.float8_e4m3fn
    i_s = shared.w2.shape[0]
    return (
        activation == "swigluoai"
        and i_s > 0
        and i_s % _QUANT_BLOCK == 0
        and shared.w1.shape == (h, 2 * i_s)
        and shared.w2.shape == (i_s, h)
        and shared.s1.shape == (1, 2 * i_s)
        and shared.s2.shape == (1, h)
        and shared.w1.dtype == fp8
        and shared.w2.dtype == fp8
        and shared.s1.dtype == jnp.float32
        and shared.s2.dtype == jnp.float32
    )


def can_use_dense_expert_fp4(
    hidden,
    w1,
    w2,
    s1,
    s2,
    *,
    activation,
    shared: SharedExpert | None = None,
    sparse_rows: bool = False,
) -> bool:
    """Layout and arithmetic support; the token threshold is the caller's."""
    if activation not in _ACTIVATIONS or hidden.ndim != 2:
        return False
    m, h = hidden.shape
    if w1.ndim != 3 or w2.ndim != 3 or s1 is None or s2 is None:
        return False
    e, i, h2 = w2.shape
    fp4 = jnp.float4_e2m1fn
    if (
        hidden.dtype != jnp.bfloat16
        or w1.dtype != fp4
        or w2.dtype != fp4
        or h2 != h
        or w1.shape != (e, h, 2 * i)
        or min(m, h, i, e) <= 0
        or h % _QUANT_BLOCK
        or not _shared_ok(shared, h, activation)
        or (sparse_rows and e > _SPARSE_MAX_EXPERTS)
    ):
        return False
    if s1.shape != (e, h // _QUANT_BLOCK, 1, 2 * i) or s2.shape != (
        e,
        i // _QUANT_BLOCK,
        1,
        h,
    ):
        return False
    if s1.dtype != jnp.float32 or s2.dtype != jnp.float32:
        return False
    try:
        info = pltpu.get_tpu_info()
    except ValueError:  # not a TPU (e.g. a CPU host)
        return False
    i_s = 0 if shared is None else shared.w2.shape[0]
    return info.is_matmul_supported(jnp.bfloat16, jnp.float8_e4m3fn) and bool(
        _chunk(m, h, i, i_s, info.vmem_capacity_bytes, sparse_rows)
    )


def _vmem_bytes(
    m: int,
    h: int,
    chunk: int,
    i_s: int,
    slots: int = _NUM_BUFFERS,
    sparse: bool = False,
) -> int:
    """VMEM of one call: the weight ring (VMEM holds fp4 one byte per
    element), its scales, the shared expert's weights, the token-sized scratch
    and the compiler's temporaries (tpu7x AOT compiles: ~16 bytes per token
    element when a unit evaluates every token). With ``sparse`` the token
    scratch is the bf16 input and output, f32 tokens, expert output and sum,
    and the temporaries are a few 16-row tiles plus the shared expert's
    column blocks. Whole-expert units hold gate|up in one ring, so the bytes
    are the same in both modes."""
    weights = 3 * h * chunk
    scales = (2 * (h // _QUANT_BLOCK) * chunk + (chunk // _QUANT_BLOCK) * h) * 4
    mp = m + (-m % _ROW_ALIGN)
    if sparse:
        tokens = mp * h * (2 + 4 + 4 + 4 + 2) + 16 * _ROW_ALIGN * h
        # Temporaries live one phase at a time: `_group_by_expert`'s (the
        # triangular matrix, the one-hot, pos and hit), then the shared
        # expert's column blocks.
        width = mp + (-mp % _SPARSE_MAX_EXPERTS)
        grouping = mp * mp * 2 + mp * width * 6 + mp * _SPARSE_MAX_EXPERTS * 8
        # (AOT tpu7x, 768-column units: 128 tokens fit, 144 exceed by 1 MB.)
        tokens += max(grouping, mp * (2 * i_s + _SHARED_COLS) * 4)
        return slots * (weights + scales) + 3 * h * i_s + tokens
    scratch = 2 + 4 + 4 + 2 + (2 if i_s else 0)
    tokens = mp * h * scratch + 16 * mp * h
    return slots * (weights + scales) + 3 * h * i_s + tokens


def _group_by_expert(id_cols, m, mp, experts, stage, smem, sem):
    """Each local expert's tokens, ascending, without a scalar loop over the
    routes: ``smem[e, k]`` the k-th token routing to expert e and
    ``smem[experts, e]`` their count (a token naming e twice counts once).
    ``stage`` is the [rows, width] int32 VMEM it is built in."""
    f32, bf16 = jnp.float32, jnp.bfloat16
    lanes = _SPARSE_MAX_EXPERTS
    width = stage.shape[1]
    columns = lax.broadcasted_iota(jnp.int32, (m, lanes), 1)
    hit = jnp.zeros((m, lanes), jnp.bool_)
    for col in id_cols:
        hit |= (col == columns) & (col >= 0) & (col < experts)
    hit = hit.astype(f32)
    if mp != m:
        hit = jnp.concatenate([hit, jnp.zeros((mp - m, lanes), f32)], 0)
    # Position of each token among its expert's tokens: a strictly
    # lower-triangular ones matrix sums the earlier rows (exact in f32).
    row = lax.broadcasted_iota(jnp.int32, (mp, mp), 0)
    earlier = (lax.broadcasted_iota(jnp.int32, (mp, mp), 1) < row).astype(bf16)
    pos = jnp.dot(earlier, hit.astype(bf16), preferred_element_type=f32)
    counts = jnp.sum(hit, axis=0, keepdims=True)  # [1, lanes]
    slots = lax.broadcasted_iota(jnp.int32, (mp, width), 1).astype(f32)
    tokens = lax.broadcasted_iota(jnp.int32, (8, mp), 1)
    # Token ids as two bf16-exact parts (ids above 256 are not bf16-exact).
    sub = lax.broadcasted_iota(jnp.int32, (8, mp), 0)
    parts = jnp.where(sub == 0, tokens // 16, jnp.where(sub == 1, tokens % 16, 0))
    parts = parts.astype(bf16)
    expert_lanes = lax.broadcasted_iota(jnp.int32, (1, lanes), 1)

    def per_expert(e, carry):
        sel = expert_lanes == e
        pos_e = jnp.sum(jnp.where(sel, pos, 0), axis=1, keepdims=True)  # [mp, 1]
        hit_e = jnp.sum(jnp.where(sel, hit, 0), axis=1, keepdims=True)
        onehot = ((pos_e == slots) & (hit_e > 0)).astype(bf16)  # [mp, width]
        r = jnp.dot(parts, onehot, preferred_element_type=f32)
        stage[pl.ds(e, 1), :] = (16 * r[0:1] + r[1:2]).astype(jnp.int32)
        return carry

    lax.fori_loop(0, experts, per_expert, 0)
    count_row = counts.astype(jnp.int32)
    if width > lanes:
        count_row = jnp.concatenate(
            [count_row, jnp.zeros((1, width - lanes), jnp.int32)], 1
        )
    stage[experts : experts + 1, :] = count_row[:, :width]
    copy = pltpu.make_async_copy(stage, smem, sem.at[0])
    copy.start()
    copy.wait()


# Rows of `_select_experts`' staging: the nonempty local experts ascending
# (lanes 0..count-1), each expert's row-tile bits (`mark`'s encoding), the count.
_SELECT_IDS_ROW, _SELECT_TILES_ROW, _SELECT_COUNT_ROW = 0, 1, 2


def _select_experts(id_cols, experts, stage, smem, sem):
    """`mark` and `compact` with vector ops: the nonempty local experts, their
    row-tile bits and their count, staged in VMEM and copied to ``smem``
    (rows `_SELECT_*_ROW`). An expert's position in the list is the number of
    nonempty experts below it (an MXU prefix sum, exact in f32), and the list
    is placed by a one-hot matmul of the expert ids (exact: ids < 256)."""
    f32, bf16 = jnp.float32, jnp.bfloat16
    m = id_cols[0].shape[0]
    lanes = stage.shape[1]
    expert_lanes = lax.broadcasted_iota(jnp.int32, (m, lanes), 1)
    hit = jnp.zeros((m, lanes), jnp.bool_)
    for col in id_cols:
        hit |= (col == expert_lanes) & (col >= 0) & (col < experts)
    hit = hit.astype(f32)
    rows = lax.broadcasted_iota(jnp.int32, (m, lanes), 0)
    first = jnp.max(jnp.where(rows < _ROW_ALIGN, hit, 0), axis=0, keepdims=True)
    later = jnp.max(jnp.where(rows >= _ROW_ALIGN, hit, 0), axis=0, keepdims=True)
    active = jnp.maximum(first, later)  # [1, lanes]
    sub = lax.broadcasted_iota(jnp.int32, (lanes, lanes), 0)
    lane = lax.broadcasted_iota(jnp.int32, (lanes, lanes), 1)
    below = (sub < lane).astype(bf16)
    rank = jnp.dot(
        jnp.broadcast_to(active, (8, lanes)).astype(bf16),
        below,
        preferred_element_type=f32,
    )[0:1]
    # place[n, e]: expert e is the n-th nonempty one.
    place = jnp.where((rank == sub.astype(f32)) & (active > 0), 1.0, 0.0)
    ids = lax.dot_general(
        lax.broadcasted_iota(jnp.int32, (8, lanes), 1).astype(bf16),
        place.astype(bf16),
        (((1,), (1,)), ((), ())),
        preferred_element_type=f32,
    )[0:1]
    count = jnp.sum(active, axis=1, keepdims=True)
    stage[_SELECT_IDS_ROW : _SELECT_IDS_ROW + 1, :] = ids.astype(jnp.int32)
    stage[_SELECT_TILES_ROW : _SELECT_TILES_ROW + 1, :] = (first + 2 * later).astype(
        jnp.int32
    )
    stage[_SELECT_COUNT_ROW : _SELECT_COUNT_ROW + 1, :] = jnp.broadcast_to(
        count, (1, lanes)
    ).astype(jnp.int32)
    copy = pltpu.make_async_copy(stage, smem, sem.at[0])
    copy.start()
    copy.wait()


def _kernel(
    prefetch_smem,  # SMEM [m * topk] local expert ids, -1 off this rank;
    # with ``route``, [1] this rank's first expert id
    x_ref,  # VMEM [m, h] bf16
    ids_ref,  # VMEM [m, topk] int32: the ids as rows; with ``route``, [m, E]
    # f32 scores + correction bias
    tw_ref,  # VMEM [m, topk] f32 routing weights; with ``route``, [m, E] f32
    # routing scores
    w1_hbm,  # [E, h, 2i] fp4
    w2_hbm,  # [E, i, h] fp4
    s1_hbm,  # [E, h / 128, 1, 2i] f32
    s2_hbm,  # [E, i / 128, 1, h] f32
    *refs,
    h,
    i,
    chunk,
    experts,
    activation,
    i_s,
    sparse,
    route,
    rows_real,
    vector_select,
):
    if i_s:
        sw1_hbm, ss1_ref, sw2_hbm, ss2_ref, *refs = refs
    (
        out_ref,  # VMEM [m, h] bf16
        wg_ref,
        wu_ref,
        wd_ref,
        sg_ref,
        su_ref,
        sd_ref,
        x_pad_ref,  # VMEM [mp, h] bf16: the tokens, padded with zero rows (sparse:
        # f32, non-finite values zeroed, for single-row gathers)
        coeff_ref,  # VMEM [mp, E_pad] f32: each token's weight per local expert
        expert_acc_ref,  # VMEM [mp, h] f32: the current expert's down projection
        acc_ref,  # VMEM [mp, h] f32: the routing-weighted sum
        active_ref,  # SMEM [E]: nonzero for an expert some token routes to
        active_ids_ref,  # SMEM [E]
        sem_ref,
        *extra_scratch,
    ) = refs
    shared_scratch = extra_scratch[:5] if i_s else ()
    if sparse:
        # `_group_by_expert`'s VMEM staging, its SMEM copy and DMA semaphore;
        # VMEM [16, h] f32 one tile's rows.
        group_vmem, sorted_ref, group_sem, rows_ref = extra_scratch[
            len(shared_scratch) : len(shared_scratch) + 4
        ]
    if (route is not None or vector_select) and not sparse:
        # VMEM [8, 128] int32 and its SMEM copy: route j's local ids, row j;
        # with ``vector_select``, the nonempty experts (`_select_experts`).
        ids_vmem, ids_route_smem, route_sem = extra_scratch[-3:]
    fp8 = jnp.float8_e4m3fn
    f32 = jnp.float32
    m = x_ref.shape[0]
    mp = acc_ref.shape[0]
    slots = sem_ref.shape[0]
    num_chunks = i // chunk
    # A whole expert per unit (TP-sliced experts): one contiguous w13 copy.
    whole = chunk == i

    if i_s:
        swg_ref, swu_ref, swd_ref, shared_out_ref, shared_sem = shared_scratch
        shared_copies = [
            pltpu.make_async_copy(
                sw1_hbm.at[:, pl.ds(0, i_s)], swg_ref, shared_sem.at[0]
            ),
            pltpu.make_async_copy(
                sw1_hbm.at[:, pl.ds(i_s, i_s)], swu_ref, shared_sem.at[1]
            ),
            pltpu.make_async_copy(sw2_hbm, swd_ref, shared_sem.at[2]),
        ]
        for copy in shared_copies:
            copy.start()

    def clear(e, carry):
        active_ref[e] = 0
        return carry

    if not vector_select:
        lax.fori_loop(0, experts, clear, 0)
    if route is None:
        topk = ids_ref.shape[1]
        ids_smem = prefetch_smem
        ids = ids_ref[...]
        tw = tw_ref[...]
        id_cols = [ids[:, j : j + 1] for j in range(topk)]
        tw_cols = [tw[:, j : j + 1] for j in range(topk)]

        def route_id(r):
            return ids_smem[r]

    else:
        # Routing while the shared expert's weights fly; the scalar loops read
        # the ids from SMEM.
        topk = route.topk
        scores_ref, choice_ref = tw_ref, ids_ref
        id_cols, tw_cols = _route(
            scores_ref[...],
            choice_ref[...],
            prefetch_smem[0],
            experts,
            rows_real,
            route,
        )
        # Sparse rows group the id columns themselves; vector selection reads
        # them as vectors.
        if not sparse and not vector_select:
            tokens = lax.broadcasted_iota(jnp.int32, (m, ids_vmem.shape[1]), 0)
            lanes = lax.broadcasted_iota(jnp.int32, (m, ids_vmem.shape[1]), 1)
            for j, col in enumerate(id_cols):
                ids_vmem[j : j + 1, :] = jnp.sum(
                    jnp.where(tokens == lanes, col, 0), axis=0, keepdims=True
                )
            route_copy = pltpu.make_async_copy(
                ids_vmem, ids_route_smem, route_sem.at[0]
            )
            route_copy.start()
            route_copy.wait()

        def route_id(r):
            return ids_route_smem[r % topk, r // topk]

    routes = m * topk

    def mark(r, carry):
        e = route_id(r)

        @pl.when((e >= 0) & (e < experts))
        def local():
            # Bit h: a token of the h-th _ROW_ALIGN-row tile routes to e.
            active_ref[e] = active_ref[e] | jnp.where(
                r // topk >= _ROW_ALIGN, 2, 1
            ).astype(jnp.int32)

        return carry

    if sparse:
        _group_by_expert(id_cols, m, mp, experts, group_vmem, sorted_ref, group_sem)

        def mark_counts(e, carry):
            active_ref[e] = sorted_ref[experts, e]
            return carry

        lax.fori_loop(0, experts, mark_counts, 0)
    elif not vector_select:
        lax.fori_loop(0, routes, mark, 0)

    def compact(e, count):
        @pl.when(active_ref[e] != 0)
        def record():
            active_ids_ref[count] = e

        return count + (active_ref[e] != 0).astype(jnp.int32)

    if vector_select:
        _select_experts(id_cols, experts, ids_vmem, ids_route_smem, route_sem)
        count = ids_route_smem[_SELECT_COUNT_ROW, 0]

        def active_id(n):
            return ids_route_smem[_SELECT_IDS_ROW, n]

        def tile_bits(e):
            return ids_route_smem[_SELECT_TILES_ROW, e]

    else:
        count = lax.fori_loop(0, experts, compact, jnp.int32(0))

        def active_id(n):
            return active_ids_ref[n]

        def tile_bits(e):
            return active_ref[e]

    units = count * num_chunks

    def transfers(unit):
        e = active_id(unit // num_chunks)
        col = pl.multiple_of((unit % num_chunks) * chunk, chunk)
        blk = (unit % num_chunks) * (chunk // _QUANT_BLOCK)
        slot = unit % slots
        sem = sem_ref.at[slot]
        if whole:
            return [
                pltpu.make_async_copy(w1_hbm.at[e], wg_ref.at[slot], sem.at[0]),
                pltpu.make_async_copy(w2_hbm.at[e], wd_ref.at[slot], sem.at[2]),
                pltpu.make_async_copy(s1_hbm.at[e], sg_ref.at[slot], sem.at[3]),
                pltpu.make_async_copy(s2_hbm.at[e], sd_ref.at[slot], sem.at[5]),
            ]
        return [
            pltpu.make_async_copy(
                w1_hbm.at[e, :, pl.ds(col, chunk)], wg_ref.at[slot], sem.at[0]
            ),
            pltpu.make_async_copy(
                w1_hbm.at[e, :, pl.ds(i + col, chunk)], wu_ref.at[slot], sem.at[1]
            ),
            pltpu.make_async_copy(
                w2_hbm.at[e, pl.ds(col, chunk), :], wd_ref.at[slot], sem.at[2]
            ),
            pltpu.make_async_copy(
                s1_hbm.at[e, :, :, pl.ds(col, chunk)], sg_ref.at[slot], sem.at[3]
            ),
            pltpu.make_async_copy(
                s1_hbm.at[e, :, :, pl.ds(i + col, chunk)], su_ref.at[slot], sem.at[4]
            ),
            pltpu.make_async_copy(
                s2_hbm.at[e, pl.ds(blk, chunk // _QUANT_BLOCK), :],
                sd_ref.at[slot],
                sem.at[5],
            ),
        ]

    for initial in range(slots - 1):

        @pl.when(units > initial)
        def prime(*, initial=initial):
            for copy in transfers(jnp.int32(initial)):
                copy.start()

    # Token-side setup overlaps the first weight copies.
    def pad_rows(v):
        if mp == m:
            return v
        return jnp.concatenate([v, jnp.zeros((mp - m, v.shape[1]), v.dtype)], 0)

    expert_columns = lax.broadcasted_iota(jnp.int32, (m, coeff_ref.shape[1]), 1)
    coeff = jnp.zeros((m, coeff_ref.shape[1]), f32)
    for col, w in zip(id_cols, tw_cols):
        # Zero routing must suppress even a non-finite routing weight.
        w = jnp.where(jnp.isfinite(w), w, 0)
        coeff += jnp.where(col == expert_columns, w, 0)
    coeff_ref[...] = pad_rows(coeff)
    acc_ref[...] = jnp.zeros_like(acc_ref)

    def matmul(lhs, w_ref, rows):
        """``lhs(k)``: the [rows, 128] K block k of the left operand; K blocks
        keep the MXU operands' temporaries small."""
        y = jnp.zeros((rows, w_ref.shape[1]), f32)
        for k in range(0, w_ref.shape[0], _SHARED_K_BLOCK):
            y += jnp.matmul(
                lhs(k), w_ref[k : k + _SHARED_K_BLOCK, :], preferred_element_type=f32
            )
        return y

    def projection(lhs, w_ref, scale, rows):
        y = matmul(lhs, w_ref, rows).astype(jnp.bfloat16).astype(f32) * scale
        return y.astype(jnp.bfloat16)

    if sparse:
        # f32 tokens (single rows load only from 32-bit refs), written in row
        # blocks so no [mp, h] temporary exists.
        for r0 in range(0, mp, _TOKEN_COPY_ROWS):
            r1 = min(r0 + _TOKEN_COPY_ROWS, mp)
            if r0 >= m:
                x_pad_ref[r0:r1, :] = jnp.zeros((r1 - r0, h), f32)
                continue
            v = x_ref[r0 : min(r1, m), :].astype(f32)
            x_pad_ref[r0 : min(r1, m), :] = jnp.where(jnp.isfinite(v), v, 0)
            if r1 > m:
                x_pad_ref[m:r1, :] = jnp.zeros((r1 - m, h), f32)
        if i_s:
            # The shared expert, on the tokens as given, goes to the output
            # now; the routed sum is added to it at the end.
            for copy in shared_copies:
                copy.wait()

            def x_block(k):
                return x_ref[:, k : k + _SHARED_K_BLOCK]

            gate = projection(x_block, swg_ref, ss1_ref[:, :i_s], m).astype(f32)
            up = projection(x_block, swu_ref, ss1_ref[:, i_s:], m).astype(f32)
            act = shared_activation(gate, up).astype(jnp.bfloat16)

            def act_block(k):
                return act[:, k : k + _SHARED_K_BLOCK]

            for c0 in range(0, h, _SHARED_COLS):
                cols = slice(c0, min(c0 + _SHARED_COLS, h))
                y = matmul(act_block, swd_ref.at[:, cols], m)
                y = y.astype(jnp.bfloat16).astype(f32) * ss2_ref[:, cols]
                out_ref[:, cols] = y.astype(jnp.bfloat16)
    else:
        x = pad_rows(x_ref[...].astype(f32))
        if i_s:
            # The shared expert sees the tokens as given, the routed experts
            # with non-finite values zeroed (as the GMM path's one-hot permute
            # does).
            x_pad_ref[...] = x.astype(jnp.bfloat16)
            for copy in shared_copies:
                copy.wait()
            xs = x_pad_ref[...]

            def xs_block(k):
                return xs[:, k : k + _SHARED_K_BLOCK]

            gate = projection(xs_block, swg_ref, ss1_ref[:, :i_s], mp).astype(f32)
            up = projection(xs_block, swu_ref, ss1_ref[:, i_s:], mp).astype(f32)
            act = shared_activation(gate, up).astype(jnp.bfloat16)

            def act_block(k):
                return act[:, k : k + _SHARED_K_BLOCK]

            shared_out_ref[...] = projection(act_block, swd_ref, ss2_ref[...], mp)
        x_pad_ref[...] = jnp.where(jnp.isfinite(x), x, 0).astype(jnp.bfloat16)

    def expert_mlp(x, slot):
        """The unit's gate, up and down columns on rows ``x``: f32 [rows, h]."""
        gate = jnp.zeros((x.shape[0], chunk), f32)
        up = jnp.zeros((x.shape[0], chunk), f32)
        for k in range(0, h, _QUANT_BLOCK):
            b = k // _QUANT_BLOCK
            xk = x[:, k : k + _QUANT_BLOCK]
            if whole:
                gu = (
                    jnp.matmul(
                        xk,
                        wg_ref[slot, k : k + _QUANT_BLOCK, :].astype(fp8),
                        preferred_element_type=f32,
                    )
                    * sg_ref[slot, b]
                )
                gate += gu[:, :chunk]
                up += gu[:, chunk:]
                continue
            gate += (
                jnp.matmul(
                    xk,
                    wg_ref[slot, k : k + _QUANT_BLOCK, :].astype(fp8),
                    preferred_element_type=f32,
                )
                * sg_ref[slot, b]
            )
            up += (
                jnp.matmul(
                    xk,
                    wu_ref[slot, k : k + _QUANT_BLOCK, :].astype(fp8),
                    preferred_element_type=f32,
                )
                * su_ref[slot, b]
            )
        # apply_act_fn takes [gate | up] lane-interleaved (its dispatch is reused).
        act = apply_act_fn(interleave_lane(gate, up), activation).astype(x.dtype)
        down = jnp.zeros((x.shape[0], h), f32)
        for k in range(0, chunk, _QUANT_BLOCK):
            b = k // _QUANT_BLOCK
            down += (
                jnp.matmul(
                    act[:, k : k + _QUANT_BLOCK],
                    wd_ref[slot, k : k + _QUANT_BLOCK, :].astype(fp8),
                    preferred_element_type=f32,
                )
                * sd_ref[slot, b]
            )
        return down

    tile = _ROW_ALIGN
    expert_lanes = lax.broadcasted_iota(jnp.int32, (1, coeff_ref.shape[1]), 1)

    def compute_sparse(unit):
        """The unit on its expert's routes only, 16 rows at a time: each tile's
        rows load and its weighted outputs add back one token row at a time,
        so the cost does not grow with the step's token count."""
        slot = unit % slots
        c = unit % num_chunks
        e = active_id(unit // num_chunks)
        end = sorted_ref[experts, e]
        tiles = (end + tile - 1) // tile

        def run(t, carry):
            base = t * tile
            rows = pl.ds(pl.multiple_of(t * tile, tile), tile)
            for j in range(tile):
                # Rows past the expert's routes repeat a real token; their
                # outputs are never added.
                tok = sorted_ref[e, jnp.minimum(base + j, end - 1)]
                rows_ref[j : j + 1, :] = x_pad_ref[pl.ds(tok, 1), :]
            down = expert_mlp(rows_ref[...].astype(jnp.bfloat16), slot)

            @pl.when(c == 0)
            def first():
                expert_acc_ref[rows, :] = down

            @pl.when(c > 0)
            def rest():
                expert_acc_ref[rows, :] += down

            @pl.when(c == num_chunks - 1)
            def finish():
                # The expert's output rounds to bf16 before its routing weight
                # (the GMM path's rounding); a non-finite one is dropped.
                # (isfinite takes f32 in Mosaic: test the bf16 round trip.)
                out_e = expert_acc_ref[rows, :].astype(jnp.bfloat16).astype(f32)
                out_e = jnp.where(jnp.isfinite(out_e), out_e, 0)
                for j in range(tile):

                    @pl.when(base + j < end)
                    def add(j=j):
                        tok = sorted_ref[e, base + j]
                        coeff = jnp.sum(
                            jnp.where(
                                expert_lanes == e, coeff_ref[pl.ds(tok, 1), :], 0
                            ),
                            axis=1,
                            keepdims=True,
                        )
                        acc_ref[pl.ds(tok, 1), :] += coeff * out_e[j : j + 1, :]

            return carry

        lax.fori_loop(0, tiles, run, 0)

    def compute(unit):
        if sparse:
            compute_sparse(unit)
            return
        slot = unit % slots
        c = unit % num_chunks
        e = active_id(unit // num_chunks)

        def run(rows):
            down = expert_mlp(x_pad_ref[rows, :], slot)

            @pl.when(c == 0)
            def first():
                expert_acc_ref[rows, :] = down

            @pl.when(c > 0)
            def rest():
                expert_acc_ref[rows, :] += down

            @pl.when(c == num_chunks - 1)
            def finish():
                columns = lax.broadcasted_iota(jnp.int32, coeff_ref.shape, 1)
                columns = columns[rows, :]
                coeff = jnp.sum(
                    jnp.where(columns == e, coeff_ref[rows, :], 0),
                    axis=1,
                    keepdims=True,
                )
                out_e = expert_acc_ref[rows, :].astype(out_ref.dtype).astype(f32)
                # Zero routing must suppress even a non-finite unused output.
                acc_ref[rows, :] += jnp.where(coeff != 0, out_e * coeff, 0)

        if mp != 2 * _ROW_ALIGN:
            run(slice(None))
            return
        # Two row tiles: an expert whose routes all lie in one tile runs only
        # that tile (the other's rows would all carry zero coefficients).
        halves = tile_bits(e)
        for bits, rows in (
            (3, slice(None)),
            (1, slice(0, _ROW_ALIGN)),
            (2, slice(_ROW_ALIGN, mp)),
        ):
            pl.when(halves == bits)(functools.partial(run, rows))

    def body(unit, carry):
        for copy in transfers(unit):
            copy.wait()

        @pl.when(unit + slots - 1 < units)
        def prefetch():
            for copy in transfers(unit + slots - 1):
                copy.start()

        compute(unit)
        return carry

    lax.fori_loop(0, units, body, 0)

    if sparse:
        # Column blocks: no [m, h] temporary.
        for c0 in range(0, h, _SHARED_COLS):
            cols = slice(c0, min(c0 + _SHARED_COLS, h))
            out = acc_ref[:m, cols]
            if i_s:
                routed = out.astype(out_ref.dtype).astype(f32)
                out = routed + out_ref[:, cols].astype(f32)
            out_ref[:, cols] = out.astype(out_ref.dtype)
        return
    out = acc_ref[:m, :]
    if i_s:
        routed = out.astype(out_ref.dtype).astype(f32)
        out = routed + shared_out_ref[:m, :].astype(f32)
    out_ref[...] = out.astype(out_ref.dtype)


@functools.partial(jax.jit, static_argnames=("activation", "sparse_rows", "route"))
def dense_expert_moe_fp4(
    hidden,
    w1,
    w2,
    s1,
    s2,
    topk_weights,
    local_ids,
    *,
    activation="swigluoai",
    shared: SharedExpert | None = None,
    sparse_rows: bool = False,
    route: Route | None = None,
    experts_start=None,
):
    """Local routed contribution of fp4 experts, [tokens, hidden] bf16, as
    fused_moe_func's GMM path returns it, plus the shared expert's output
    when `shared` is given. IDs outside this rank must be -1 (TP-sliced
    experts: every id is local). Callers check can_use_dense_expert_fp4 first.
    Non-finite hidden states and weights are zeroed for the routed experts, as
    the GMM path's one-hot permute does on SparseCore EP steps; elsewhere
    (TP-MoE, no SparseCore) the GMM path would keep a non-finite token
    non-finite.

    With ``route``, the kernel routes the tokens itself: ``topk_weights`` are
    then the [tokens, E] f32 routing scores of all E global experts,
    ``local_ids`` the scores plus correction bias, and ``experts_start`` this
    rank's first global expert id (EP; None: every expert is local)."""
    m, h = hidden.shape
    experts, i, _ = w2.shape
    # `_group_by_expert` holds the local experts on one vreg's lanes.
    assert not sparse_rows or experts <= _SPARSE_MAX_EXPERTS, experts
    i_s = 0 if shared is None else shared.w2.shape[0]
    info = pltpu.get_tpu_info()
    chunk = _chunk(m, h, i, i_s, info.vmem_capacity_bytes, sparse_rows)
    whole = chunk == i
    slots = _NUM_BUFFERS
    budget = info.vmem_capacity_bytes - _VMEM_HEADROOM
    deeper = _vmem_bytes(m, h, chunk, i_s, _WHOLE_UNIT_BUFFERS, sparse_rows)
    if whole and deeper <= budget:
        slots = _WHOLE_UNIT_BUFFERS
    # Whole f32 sublane tiles of tokens; the kernel pads to _ROW_ALIGN itself.
    pad = -m % 8
    if pad:
        hidden = jnp.pad(hidden, ((0, pad), (0, 0)))
        local_ids = jnp.pad(
            local_ids, ((0, pad), (0, 0)), constant_values=0 if route else -1
        )
        topk_weights = jnp.pad(topk_weights, ((0, pad), (0, 0)))
    mk = m + pad
    mp = mk + (-mk % _ROW_ALIGN)
    if route is None:
        prefetch = local_ids.astype(jnp.int32).reshape(-1)
        local_ids = local_ids.astype(jnp.int32)
        topk_weights = topk_weights.astype(jnp.float32)
    else:
        start = 0 if experts_start is None else experts_start
        prefetch = jnp.asarray(start, jnp.int32).reshape(1)
    lanes = info.num_lanes
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    shared_operands, shared_specs, shared_scratch = (), [], []
    if shared is not None:
        shared_operands = tuple(shared)
        shared_specs = [hbm, vmem, hbm, vmem]
        shared_scratch = [
            pltpu.VMEM((h, i_s), shared.w1.dtype),
            pltpu.VMEM((h, i_s), shared.w1.dtype),
            pltpu.VMEM((i_s, h), shared.w2.dtype),
            # Sparse mode stages the shared expert's output in the output.
            pltpu.VMEM((8, 128) if sparse_rows else (mp, h), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((3,)),
        ]
    sparse_scratch = []
    if sparse_rows:
        # `_group_by_expert`: rows 0..experts-1 the tokens, row experts the counts.
        group_shape = (
            experts + 1 + (-(experts + 1)) % 8,
            max(-(-mp // lanes), 1) * lanes,
        )
        sparse_scratch = [
            pltpu.VMEM(group_shape, jnp.int32),
            pltpu.SMEM(group_shape, jnp.int32),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.VMEM((_ROW_ALIGN, h), jnp.float32),
        ]
    vector_select = (
        not sparse_rows and _SCALAR_SELECT_MAX_EXPERTS < experts <= info.num_lanes
    )
    route_scratch = []
    if (route is not None or vector_select) and not sparse_rows:
        route_scratch = [
            pltpu.VMEM((8, lanes), jnp.int32),
            pltpu.SMEM((8, lanes), jnp.int32),
            pltpu.SemaphoreType.DMA((1,)),
        ]
    gate_cols = 2 * chunk if whole else chunk
    out = pl.pallas_call(
        functools.partial(
            _kernel,
            h=h,
            i=i,
            chunk=chunk,
            experts=experts,
            activation=activation,
            i_s=i_s,
            sparse=sparse_rows,
            route=route,
            rows_real=m,
            vector_select=vector_select,
        ),
        out_shape=jax.ShapeDtypeStruct((mk, h), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vmem, vmem, vmem, *[hbm] * 4, *shared_specs],
            out_specs=vmem,
            scratch_shapes=[
                # Whole-expert units keep gate|up in the first ring: no up ring.
                pltpu.VMEM((slots, h, gate_cols), w1.dtype),
                pltpu.VMEM((1, 128, 128) if whole else (slots, h, chunk), w1.dtype),
                pltpu.VMEM((slots, chunk, h), w2.dtype),
                pltpu.VMEM((slots, h // _QUANT_BLOCK, 1, gate_cols), jnp.float32),
                pltpu.VMEM(
                    (1, 1, 1, 128) if whole else (slots, h // _QUANT_BLOCK, 1, chunk),
                    jnp.float32,
                ),
                pltpu.VMEM((slots, chunk // _QUANT_BLOCK, 1, h), jnp.float32),
                pltpu.VMEM((mp, h), jnp.float32 if sparse_rows else jnp.bfloat16),
                pltpu.VMEM((mp, experts + (-experts) % lanes), jnp.float32),
                pltpu.VMEM((mp, h), jnp.float32),
                pltpu.VMEM((mp, h), jnp.float32),
                pltpu.SMEM((experts,), jnp.int32),
                pltpu.SMEM((experts,), jnp.int32),
                pltpu.SemaphoreType.DMA((slots, 6)),
                *shared_scratch,
                *sparse_scratch,
                *route_scratch,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=min(
                info.vmem_capacity_bytes,
                _vmem_bytes(mk, h, chunk, i_s, slots, sparse_rows) + _VMEM_MARGIN,
            )
        ),
        name=f"dense_expert_moe_fp4-e_{experts}-m_{mp}-h_{h}-i_{i}-c_{chunk}"
        + (f"-s_{i_s}" if i_s else "")
        + ("-sparse" if sparse_rows else "")
        + ("-route" if route else ""),
    )(
        prefetch,
        hidden,
        local_ids,
        topk_weights,
        w1,
        w2,
        # 4-D as the GMM path holds them: a reshape that drops the size-1
        # axis is a relayout copy of every scale on each call.
        s1,
        s2,
        *shared_operands,
    )
    return out[:m] if pad else out

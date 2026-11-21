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
"""Experimental small-batch FP8 MoE: one Pallas call, no sort or permute.

Each token's routing is a dense weight table over experts. The kernel collects
the shard's hit experts, streams their weights through a ring of VMEM slots in
units (the gate|up halves and the down projection of one expert), and applies
each expert to every row with that row's routing weight as the mask. The
output is the shard's partial sum; the combine across shards stays the
caller's. Activations are quantized to FP8 before each matmul (W8A8), one
scale per row and 512-column block as in gmm_v2.

The routing arrives either as a table built from the caller's local top-k ids
(`routing_table`) or as the router's softmax scores, with the top-k selection
run in the kernel.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs

# Rows per call at or below which the MoE entry selects this kernel: the
# measured crossover vs dense-expert MoE (j-a6af2846, j-d8af40dc). At 128 rows
# it wins 1.14x (scores) / 1.09x (ids) on recorded skew; at 256 it is parity
# on recorded skew (1.01x / 0.93x) and loses on uniform routing (0.95x / 0.90x).
MOE_LOWTOK_MAX_TOKENS = 128
_MAX_LOCAL_EXPERTS = 128

_ROW_TILE = 32  # FP8 sublane packing
_NEG = -3.0e38  # router scores at or below this never win a pass
_LANE = 128
_VMEM_FRACTION = 0.75
_VMEM_MARGIN = 1.25  # over declared scratch and operands, for compiler tiles
_FP8 = jnp.float8_e4m3fn
_FP8_MAX = 448.0
_QUANT_BLOCK = 512  # activation scale block along K, as gmm_v2
HOT_SLOTS = 128  # copied-expert tables' fixed length (-1 padded)


def can_use_moe_lowtok(hidden, w1, w2, s1, s2, b1, b2, *, activation, rhs_quant_dtype):
    """Check arithmetic/layout support; the caller applies the token cap."""
    if activation != "silu" or rhs_quant_dtype is not None:
        return False
    if b1 is not None or b2 is not None or s1 is None or s2 is None:
        return False
    if hidden.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        return False
    h = hidden.shape[1]
    e, i, out = w2.shape
    if not 0 < e <= _MAX_LOCAL_EXPERTS or out != h or w1.shape != (e, h, 2 * i):
        return False
    if (2 * i) % 512 or h % _quant_block(h) or i % _quant_block(i):
        return False
    if (
        hidden.dtype != jnp.bfloat16
        or w1.dtype != jnp.float8_e4m3fn
        or w2.dtype != jnp.float8_e4m3fn
    ):
        return False
    return s1.shape == (e, 1, 1, 2 * i) and s2.shape == (e, 1, 1, h)


def _vmem_budget(rows, hidden, inter, w13_parts, lanes, nc=0):
    """Declared scratch and VMEM operands plus a compiler margin, capped.

    A shape-dependent reservation leaves room for neighbouring ops to
    prefetch their weights.
    """
    unit = w13_parts + 1
    weights = unit * (hidden * 2 * (inter // w13_parts) + inter * hidden)
    scales = unit * 4 * (2 * (inter // w13_parts) + hidden)
    accumulators = rows * 4 * (2 * inter + hidden)
    # x in and out, scores, tables, one-hot.
    operands = (
        rows * 2 * hidden * 2
        + rows * 4 * lanes
        + 4 * lanes * 8
        + 2 * lanes * lanes
        + (2 * lanes * nc if nc else 0)
    )
    need = int((weights + scales + accumulators + operands) * _VMEM_MARGIN)
    capacity = int(pltpu.get_tpu_info().vmem_capacity_bytes * _VMEM_FRACTION)
    return min(need, capacity)


def _quant_block(k):
    """Activation scale block along a contraction of width k."""
    qblock = envs.TPU_MOE_LOWTOK_QUANT_BLOCK
    return k if qblock <= 0 else min(k, qblock)


def _quantize_blocks(v):
    """FP8 quantization of an f32 block per row and K block, as gmm_v2.

    Returns (values, restoring scale [rows, 1]) per block.
    """
    block = _quant_block(v.shape[1])
    blocks = []
    for k in range(0, v.shape[1], block):
        b = v[:, k : k + block]
        scale = jnp.max(jnp.abs(b), axis=1, keepdims=True) / _FP8_MAX
        inv = jnp.where(scale == 0, 0, 1 / scale)
        blocks.append(((b * inv).astype(_FP8), scale))
    return blocks


def _blocked_dot(blocks, w_ref, slot):
    """Sum over K blocks of each FP8 block's matmul with w_ref[slot], rescaled."""
    out = None
    k = 0
    for q, scale in blocks:
        n = q.shape[1]
        part = jax.lax.dot_general(
            q,
            w_ref[slot, pl.ds(k, n), :],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        part = part * scale
        out = part if out is None else out + part
        k += n
    assert out is not None
    return out


def _stream(
    gid_s,
    hits_s,
    wtab_ref,
    x_ref,
    s13_hbm,
    s2_hbm,
    w13_hbm,
    w2_hbm,
    out_ref,
    ids,
    w13_buf,
    w2_buf,
    s13_buf,
    s2_buf,
    w13_sem,
    w2_sem,
    h_acc,
    acc,
    *,
    n_experts,
    inter,
    w13_parts,
    choice=None,
    gid_of=None,
):
    """Stream the hit experts of this shard and accumulate their outputs.

    gid_s [1, E] (SMEM) is the table lane of each local expert; hits_s
    [1, lanes] (SMEM) is 1 on lanes some row routes to; wtab_ref [rows, lanes]
    holds the routing weights. Each expert is `w13_parts` gate|up units then
    one down unit; the ring has one slot per unit, so slot == part.
    choice: optional hook called with the fixed hit count after fixed hits are
    primed; it appends copied experts served by this rank and returns the new
    hit count.
    """
    if gid_of is None:
        gid_of = lambda e: gid_s[0, e]  # noqa: E731
    half = inter // w13_parts
    units = w13_parts + 1
    slots = units
    lane = jax.lax.broadcasted_iota(jnp.int32, wtab_ref.shape, 1)

    def collect(e, n):
        h = hits_s[0, gid_of(e)]

        @pl.when(h == 1)
        def _():
            ids[n] = e

        return n + h

    n_hit = jax.lax.fori_loop(0, n_experts, collect, jnp.int32(0))
    n_units = n_hit * units

    def unit_copies(e_idx, part):
        expert = ids[e_idx]
        slot = part
        # The down unit builds w13 descriptors it never starts; clamp them.
        col = pl.multiple_of(jnp.minimum(part, w13_parts - 1) * half, half)
        w13_copies = (
            pltpu.make_async_copy(
                w13_hbm.at[expert, :, pl.ds(col, half)],
                w13_buf.at[slot, :, pl.ds(0, half)],
                w13_sem.at[slot],
            ),
            pltpu.make_async_copy(
                w13_hbm.at[expert, :, pl.ds(inter + col, half)],
                w13_buf.at[slot, :, pl.ds(half, half)],
                w13_sem.at[slot],
            ),
            pltpu.make_async_copy(
                s13_hbm.at[expert, :, pl.ds(col, half)],
                s13_buf.at[slot, :, pl.ds(0, half)],
                w13_sem.at[slot],
            ),
            pltpu.make_async_copy(
                s13_hbm.at[expert, :, pl.ds(inter + col, half)],
                s13_buf.at[slot, :, pl.ds(half, half)],
                w13_sem.at[slot],
            ),
        )
        w2_copies = (
            pltpu.make_async_copy(w2_hbm.at[expert], w2_buf.at[slot], w2_sem.at[slot]),
            pltpu.make_async_copy(s2_hbm.at[expert], s2_buf.at[slot], w2_sem.at[slot]),
        )
        return part, w13_copies, w2_copies

    def advance(e_idx, part, by):
        """The unit `by` (at most `units`) ahead, without a division."""
        p = part + by
        wrap = p >= units
        return jnp.where(wrap, e_idx + 1, e_idx), jnp.where(wrap, p - units, p)

    def start(e_idx, part):
        part, w13_copies, w2_copies = unit_copies(e_idx, part)

        @pl.when(part < w13_parts)
        def _():
            for c in w13_copies:
                c.start()

        @pl.when(part == w13_parts)
        def _():
            for c in w2_copies:
                c.start()

    def wait(e_idx, part):
        part, w13_copies, w2_copies = unit_copies(e_idx, part)

        @pl.when(part < w13_parts)
        def _():
            for c in w13_copies:
                c.wait()

        @pl.when(part == w13_parts)
        def _():
            for c in w2_copies:
                c.wait()

    acc[...] = jnp.zeros_like(acc)
    x_blocks = _quantize_blocks(x_ref[...].astype(jnp.float32))

    for ahead in range(slots - 1):

        @pl.when(ahead < n_units)
        def _prime(ahead=ahead):
            start(jnp.int32(0), jnp.int32(ahead))

    if choice is not None:
        n_before = n_hit
        n_hit = choice(n_hit)
        n_units = n_hit * units
        # If this rank had no fixed hits, nothing was primed above; prime now
        # from the appended copied hits.
        for ahead in range(slots - 1):

            @pl.when((n_before == 0) & (ahead < n_units))
            def _prime_late(ahead=ahead):
                start(jnp.int32(0), jnp.int32(ahead))

    def body(u, carry):
        e_idx, part = carry
        slot = part
        expert = ids[e_idx]

        @pl.when(u + slots - 1 < n_units)
        def _prefetch():
            start(*advance(e_idx, part, slots - 1))

        wait(e_idx, part)

        # One static branch per half: the lane offsets must be static.
        for p in range(w13_parts):

            @pl.when(part == p)
            def _gate_up_half(p=p):
                h = _blocked_dot(x_blocks, w13_buf, slot)
                scales = s13_buf[slot]
                col = p * half
                h_acc[:, col : col + half] = h[:, :half] * scales[:, :half]
                h_acc[:, inter + col : inter + col + half] = (
                    h[:, half:] * scales[:, half:]
                )

        @pl.when(part == w13_parts)
        def _down():
            gate = h_acc[:, pl.ds(0, inter)]
            up = h_acc[:, pl.ds(inter, inter)]
            a = gate * jax.nn.sigmoid(gate) * up
            o = _blocked_dot(_quantize_blocks(a), w2_buf, slot)
            o = o * s2_buf[slot]
            # Zero weight masks the rows this expert does not serve.
            w = jnp.sum(
                jnp.where(lane == gid_of(expert), wtab_ref[...], 0.0),
                axis=1,
                keepdims=True,
            )
            acc[...] += w * o

        return advance(e_idx, part, 1)

    jax.lax.fori_loop(0, n_units, body, (jnp.int32(0), jnp.int32(0)))
    out_ref[...] = acc[...].astype(out_ref.dtype)


def _select_topk(scores, topk):
    """The served router as a routing table over global expert ids.

    `topk` max-and-mask passes over the softmax scores (ties to the lowest
    id), weights renormalized and rounded through bf16.
    """
    lane = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    x = jnp.where(scores > _NEG, scores, _NEG)
    masked = jnp.finfo(jnp.float32).min
    sel = jnp.zeros_like(x)
    wsum = jnp.zeros((x.shape[0], 1), jnp.float32)
    for _ in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        idx = jnp.min(jnp.where(x == m, lane, x.shape[1] - 1), axis=1, keepdims=True)
        pick = lane == idx
        sel = jnp.where(pick, m, sel)
        wsum = wsum + m
        x = jnp.where(pick, masked, x)
    wt = jnp.where(wsum > 0, sel / wsum, 0.0)
    return wt.astype(jnp.bfloat16).astype(jnp.float32)


def _kernel(
    gid_s,
    x_ref,
    route_ref,
    s13_hbm,
    s2_hbm,
    w13_hbm,
    w2_hbm,
    out_ref,
    ids,
    hits_v,
    hits_s,
    hsem,
    *scratch,
    n_experts,
    inter,
    w13_parts,
    topk,
):
    if topk is None:
        wtab_ref = route_ref
    else:
        wtab_ref, *scratch = scratch
        wtab_ref[...] = _select_topk(route_ref[...], topk)
    # Lanes some row routes to; `collect` keeps the ones in this shard's gid.
    hit = jnp.any(wtab_ref[...] != 0, axis=0, keepdims=True)
    hits_v[...] = jnp.where(hit, 1, 0).astype(jnp.int32)
    copy = pltpu.make_async_copy(hits_v, hits_s, hsem.at[0])
    copy.start()
    copy.wait()
    _stream(
        gid_s,
        hits_s,
        wtab_ref,
        x_ref,
        s13_hbm,
        s2_hbm,
        w13_hbm,
        w2_hbm,
        out_ref,
        ids,
        *scratch,
        n_experts=n_experts,
        inter=inter,
        w13_parts=w13_parts,
    )


def _layout(x, w13):
    tokens, _ = x.shape
    n_experts, _, two_inter = w13.shape
    inter = two_inter // 2
    if inter % _LANE:
        raise ValueError(
            f"the intermediate width {inter} must be a multiple of {_LANE} lanes"
        )
    w13_parts = 2 if (inter // 2) % _LANE == 0 else 1
    rows = pl.cdiv(tokens, _ROW_TILE) * _ROW_TILE
    return n_experts, inter, w13_parts, rows


def _stream_scratch(rows, hidden, inter, w13_parts):
    slots = w13_parts + 1
    half = inter // w13_parts
    return (
        pltpu.VMEM((slots, hidden, 2 * half), _FP8),
        pltpu.VMEM((slots, inter, hidden), _FP8),
        pltpu.VMEM((slots, 1, 2 * half), jnp.float32),
        pltpu.VMEM((slots, 1, hidden), jnp.float32),
        pltpu.SemaphoreType.DMA((slots,)),
        pltpu.SemaphoreType.DMA((slots,)),
        pltpu.VMEM((rows, 2 * inter), jnp.float32),
        pltpu.VMEM((rows, hidden), jnp.float32),
    )


def routing_table(topk_ids_local, topk_weights, n_experts):
    """[tokens, n_experts] routing weights from local top-k ids (-1: remote)."""
    valid = topk_ids_local >= 0
    onehot = jax.nn.one_hot(
        jnp.where(valid, topk_ids_local, 0), n_experts, dtype=jnp.float32
    )
    weights = jnp.where(valid, topk_weights, 0).astype(jnp.float32)
    return jnp.einsum("tke,tk->te", onehot, weights)


@functools.partial(jax.jit, static_argnames=("topk",))
def moe_lowtok(x, route, w13, s13, w2, s2, experts_start=0, *, topk=None):
    """This shard's MoE output for a few tokens.

    With `topk` None, `route` is the routing table over local experts (see
    routing_table). With `topk`, `route` is the router's softmax scores over
    all experts, the top-k runs in the kernel, and this shard holds
    experts [experts_start, experts_start + E); a zero score row (a padded
    token) selects nothing. Scales use [E, 1, 1, N]. Callers must first check
    can_use_moe_lowtok.
    """
    tokens, hidden = x.shape
    n_experts, inter, w13_parts, rows = _layout(x, w13)
    # The table's lanes are local expert ids, or global ones for the scores.
    lanes = pl.cdiv(route.shape[1], _LANE) * _LANE
    x_rows = jnp.pad(x, ((0, rows - tokens), (0, 0)))
    route = jnp.pad(
        route.astype(jnp.float32), ((0, rows - tokens), (0, lanes - route.shape[1]))
    )
    s13 = s13.reshape(n_experts, 1, 2 * inter).astype(jnp.float32)
    s2 = s2.reshape(n_experts, 1, hidden).astype(jnp.float32)
    gid = (experts_start + jnp.arange(n_experts, dtype=jnp.int32)).reshape(1, n_experts)
    wtab = () if topk is None else (pltpu.VMEM((rows, lanes), jnp.float32),)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    out = pl.pallas_call(
        functools.partial(
            _kernel,
            n_experts=n_experts,
            inter=inter,
            w13_parts=w13_parts,
            topk=topk,
        ),
        out_shape=jax.ShapeDtypeStruct((rows, hidden), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vmem, vmem, hbm, hbm, hbm, hbm],
            out_specs=vmem,
            scratch_shapes=(
                pltpu.SMEM((n_experts,), jnp.int32),
                pltpu.VMEM((1, lanes), jnp.int32),
                pltpu.SMEM((1, lanes), jnp.int32),
                pltpu.SemaphoreType.DMA((1,)),
                *wtab,
                *_stream_scratch(rows, hidden, inter, w13_parts),
            ),
            grid=(),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=_vmem_budget(rows, hidden, inter, w13_parts, lanes)
        ),
        name="moe_lowtok",
    )(gid, x_rows, route, s13, s2, w13, w2)
    return out[:tokens]


def table_offsets(n_experts, nc, lanes=512):
    """Layout of the flat packed table for `n_experts` local slots and `nc` hot entries.

    Returns (gid, hot, hprim, hcopy, slot) offsets and total padded length.
    """
    o_gid = 8
    o_hot = o_gid + n_experts
    o_hprim = o_hot + nc
    o_hcopy = o_hprim + nc
    o_slot = o_hcopy + nc
    length = pl.cdiv(o_slot + lanes, _LANE) * _LANE
    return o_gid, o_hot, o_hprim, o_hcopy, o_slot, length


def pack_tables(rank, gid, hot, hprim, hcopy, slot):
    """Pack per-rank expert assignment tables into one flat int32 array."""
    gid = np.asarray(gid, np.int32).reshape(-1)
    hot, hprim, hcopy = (
        np.asarray(t, np.int32).reshape(-1) for t in (hot, hprim, hcopy)
    )
    slot = np.asarray(slot, np.int32).reshape(-1)
    if slot.size == 0 or slot.size % _LANE != 0:
        raise ValueError(
            f"the slot table must be a multiple of {_LANE} lanes, got {slot.size}"
        )
    if not (hprim.size == hot.size == hcopy.size):
        raise ValueError("hot, hprim and hcopy must be the same length")
    o_gid, o_hot, o_hprim, o_hcopy, o_slot, length = table_offsets(
        gid.size, hot.size, lanes=slot.size
    )
    tab = np.full((length,), -1, np.int32)
    tab[0] = int(rank)
    tab[o_gid : o_gid + gid.size] = gid
    tab[o_hot : o_hot + hot.size] = hot
    tab[o_hprim : o_hprim + hot.size] = hprim
    tab[o_hcopy : o_hcopy + hot.size] = hcopy
    tab[o_slot : o_slot + slot.size] = slot
    return tab


def slot_table(gid, n_experts, lanes=512):
    """Map global expert ids to this rank's local slot index (-1 if absent)."""
    gid = jnp.asarray(gid, jnp.int32).reshape(-1)
    return (
        jnp.full((lanes,), -1, jnp.int32)
        .at[gid]
        .set(jnp.arange(n_experts, dtype=jnp.int32))
    )


def hot_onehot(hot, lanes=512):
    """One-hot [lanes, len(hot)] bf16 gather matrix (-1 gives all-zero columns)."""
    hot = jnp.asarray(hot, jnp.int32).reshape(-1)
    iota = jax.lax.broadcasted_iota(jnp.int32, (lanes, hot.shape[0]), 0)
    return jnp.where(iota == hot.reshape(1, -1), 1.0, 0.0).astype(jnp.bfloat16)


def hot_tables(prim, copy, order=None, slots=HOT_SLOTS):
    """Build copied-expert (hot, hprim, hcopy) tables padded to `slots` with -1."""
    prim = jnp.asarray(prim, jnp.int32).reshape(-1)
    copy = jnp.asarray(copy, jnp.int32).reshape(-1)
    if order is not None:
        o = [int(e) for e in order]
        if len(o) > slots:
            raise ValueError(
                f"{len(o)} copied experts exceed the {slots}-entry hot table"
            )
        if len(set(o)) != len(o) or any(e < 0 or e >= prim.shape[0] for e in o):
            raise ValueError("hot order must contain unique, valid expert ids")
        hot = np.full(slots, -1, np.int32)
        hot[: len(o)] = o
        hot = jnp.asarray(hot)
    else:
        hot = jnp.nonzero(copy >= 0, size=slots, fill_value=-1)[0].astype(jnp.int32)
    safe = jnp.where(hot >= 0, hot, 0)
    hprim = jnp.where(hot >= 0, prim[safe], -1).astype(jnp.int32)
    hcopy = jnp.where((hot >= 0) & (copy[safe] >= 0), copy[safe], -1).astype(jnp.int32)
    return hot, hprim, hcopy


def _kernel_scores(
    tab_s,
    prim_ref,
    copy_ref,
    hot1h_ref,
    x_hbm,
    scores_ref,
    s13_hbm,
    s2_hbm,
    w13_hbm,
    w2_hbm,
    out_ref,
    ids,
    wtab_ref,
    hits_v,
    hits_s,
    cmp_v,
    cmp_s,
    hsem,
    x_vmem,
    xsem,
    w13_buf,
    w2_buf,
    s13_buf,
    s2_buf,
    w13_sem,
    w2_sem,
    h_acc,
    acc,
    *,
    n_experts,
    inter,
    w13_parts,
    topk,
    ranks,
    owned=1,
    has_copies=True,
    offsets,
):
    """Scores entry with replicated-expert load balancing across ranks."""
    rows = x_vmem.shape[0]
    lanes = scores_ref.shape[1]
    x_copy = pltpu.make_async_copy(x_hbm, x_vmem, xsem.at[0])
    x_copy.start()
    lane = jax.lax.broadcasted_iota(jnp.int32, (rows, lanes), 1)
    o_gid, _o_hot, o_hprim, o_hcopy, o_slot = offsets
    my_rank = tab_s[0] // owned

    wt = _select_topk(scores_ref[...], topk)
    wtab_ref[...] = wt
    hit = jnp.any(wt != 0, axis=0, keepdims=True)
    prim = prim_ref[...]
    copy = copy_ref[...]
    has_copy = copy >= 0
    fixed = hit & jnp.logical_not(has_copy)
    lane1 = lane[:1]
    if not has_copies:
        hits_v[...] = jnp.where(fixed & (prim == my_rank), 1, 0).astype(jnp.int32)
        cp1 = pltpu.make_async_copy(hits_v, hits_s, hsem.at[1])
        cp1.start()
        cp1.wait()
        x_copy.wait()
        _stream(
            tab_s,
            hits_s,
            wtab_ref,
            x_vmem,
            s13_hbm,
            s2_hbm,
            w13_hbm,
            w2_hbm,
            out_ref,
            ids,
            w13_buf,
            w2_buf,
            s13_buf,
            s2_buf,
            w13_sem,
            w2_sem,
            h_acc,
            acc,
            n_experts=n_experts,
            inter=inter,
            w13_parts=w13_parts,
            gid_of=lambda e: tab_s[o_gid + e],
        )
        return

    loads = jnp.zeros_like(lane1)
    for r in range(ranks):
        load_r = jnp.sum(jnp.where(fixed & (prim == r), 1, 0), axis=1, keepdims=True)
        loads = jnp.where(lane1 == r, load_r, loads)

    # Stream this rank's fixed experts to SMEM first so weight prefetching
    # overlaps with dynamic copy assignment below.
    hits_v[...] = jnp.where(fixed & (prim == my_rank), 1, 0).astype(jnp.int32)
    cp1 = pltpu.make_async_copy(hits_v, hits_s, hsem.at[1])
    cp1.start()

    nc = hot1h_ref.shape[1]
    hit8 = jnp.broadcast_to(jnp.where(hit, 1.0, 0.0).astype(jnp.bfloat16), (8, lanes))
    hh = jnp.dot(hit8, hot1h_ref[...], preferred_element_type=jnp.float32)[:1]
    lane_c = jax.lax.broadcasted_iota(jnp.int32, (1, nc), 1)
    ps = hh
    k = 1
    while k < nc:
        ps = ps + jnp.where(lane_c >= k, pltpu.roll(ps, k, axis=1), 0.0)
        k *= 2
    n_active = ps[:, nc - 1 : nc].astype(jnp.int32)
    pos = jnp.where(hh > 0.5, ps - 1.0, -1.0)
    posm = jnp.transpose(jnp.broadcast_to(pos, (nc, nc)))
    row_i = jax.lax.broadcasted_iota(jnp.int32, (nc, nc), 0)
    lane_p = jax.lax.broadcasted_iota(jnp.int32, (nc, nc), 1).astype(jnp.float32)
    comp = jnp.sum(jnp.where(posm == lane_p, row_i, 0), axis=0, keepdims=True)

    cmp_v[...] = jnp.zeros_like(cmp_v)
    cmp_v[0:1, :] = comp
    cmp_v[1:2, :] = jnp.broadcast_to(n_active, (1, nc))
    cmp_v[2:3, :] = loads[:, :nc].astype(jnp.int32)
    cp2 = pltpu.make_async_copy(cmp_v, cmp_s, hsem.at[2])
    cp2.start()
    cp1.wait()

    def choice(n_fixed):
        cp2.wait()

        def assign(j, n):
            i = cmp_s[0, j]
            e = tab_s[_o_hot + i]
            o = tab_s[o_hprim + i]
            c = tab_s[o_hcopy + i]
            pick = jnp.where(cmp_s[2, c] < cmp_s[2, o], c, o)
            cmp_s[2, pick] = cmp_s[2, pick] + 1
            mine = pick == my_rank

            @pl.when(mine)
            def _():
                hits_s[0, e] = 1
                ids[n] = tab_s[o_slot + e]

            return n + jnp.where(mine, 1, 0)

        return jax.lax.fori_loop(0, cmp_s[1, 0], assign, n_fixed)

    x_copy.wait()
    _stream(
        tab_s,
        hits_s,
        wtab_ref,
        x_vmem,
        s13_hbm,
        s2_hbm,
        w13_hbm,
        w2_hbm,
        out_ref,
        ids,
        w13_buf,
        w2_buf,
        s13_buf,
        s2_buf,
        w13_sem,
        w2_sem,
        h_acc,
        acc,
        n_experts=n_experts,
        inter=inter,
        w13_parts=w13_parts,
        choice=choice,
        gid_of=lambda e: tab_s[o_gid + e],
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "topk",
        "ranks",
        "owned",
        "vmem_limit_bytes",
        "has_copies",
        "hot_slots",
    ),
)
def moe_lowtok_scores(
    x,
    scores,
    rank,
    gid,
    prim,
    copy,
    w13,
    s13,
    w2,
    s2,
    *,
    topk,
    ranks,
    owned=1,
    vmem_limit_bytes=None,
    hot=None,
    hprim=None,
    hcopy=None,
    hot1h=None,
    slot=None,
    has_copies=None,
    tables=None,
    hot_slots=HOT_SLOTS,
):
    """Scores entry supporting replicated experts and dynamic holder balancing."""
    tokens, hidden = x.shape
    n_experts, _, two_inter = w13.shape
    inter = two_inter // 2
    n_scores = scores.shape[1]
    if prim.shape[0] != n_scores or copy.shape[0] != n_scores:
        raise ValueError(
            f"scores width {n_scores} must match "
            f"prim {prim.shape[0]} and copy {copy.shape[0]}"
        )
    if inter % _LANE:
        raise ValueError(
            f"the intermediate width {inter} must be a multiple of {_LANE} lanes"
        )
    lanes = pl.cdiv(n_scores, _LANE) * _LANE
    w13_parts = 2 if (inter // 2) % _LANE == 0 else 1
    rows = pl.cdiv(tokens, _ROW_TILE) * _ROW_TILE
    x_rows = jnp.pad(x, ((0, rows - tokens), (0, 0)))
    sc = jnp.pad(
        scores.astype(jnp.float32),
        ((0, rows - tokens), (0, lanes - n_scores)),
        constant_values=_NEG,
    )
    s13 = s13.reshape(n_experts, 1, two_inter).astype(jnp.float32)
    s2 = s2.reshape(n_experts, 1, hidden).astype(jnp.float32)
    prim = jnp.pad(
        jnp.asarray(prim, jnp.int32).reshape(-1),
        (0, lanes - n_scores),
        constant_values=-1,
    ).reshape(1, lanes)
    copy = jnp.pad(
        jnp.asarray(copy, jnp.int32).reshape(-1),
        (0, lanes - n_scores),
        constant_values=-1,
    ).reshape(1, lanes)
    if has_copies is None:
        try:
            has_copies = bool(np.any(np.asarray(copy) >= 0))
        except Exception:
            has_copies = True
    has_copies = bool(has_copies)

    if tables is None:
        rk = jnp.asarray(rank, jnp.int32).reshape(1)
        gid = jnp.asarray(gid, jnp.int32).reshape(1, n_experts)
        if hot is None:
            hot, hprim, hcopy = hot_tables(
                prim.reshape(lanes), copy.reshape(lanes), slots=hot_slots
            )
        hot = jnp.asarray(hot, jnp.int32).reshape(1, -1)
        hprim = jnp.asarray(hprim, jnp.int32).reshape(1, -1)
        hcopy = jnp.asarray(hcopy, jnp.int32).reshape(1, -1)
        nc = hot.shape[1]
        if hot1h is None:
            hot1h = hot_onehot(hot.reshape(-1), lanes=lanes)
        if slot is None:
            slot = slot_table(gid.reshape(-1), n_experts, lanes=lanes)
        slot = jnp.asarray(slot, jnp.int32).reshape(1, lanes)
        o_gid, o_hot, o_hprim, o_hcopy, o_slot, length = table_offsets(
            n_experts, nc, lanes=lanes
        )
        tab = jnp.full((length,), -1, jnp.int32)
        tab = tab.at[0].set(rk[0]).at[o_gid : o_gid + n_experts].set(gid[0])
        tab = (
            tab.at[o_hot : o_hot + nc]
            .set(hot[0])
            .at[o_hprim : o_hprim + nc]
            .set(hprim[0])
        )
        tab = (
            tab.at[o_hcopy : o_hcopy + nc]
            .set(hcopy[0])
            .at[o_slot : o_slot + lanes]
            .set(slot[0])
        )
    else:
        nc = hot_slots if hot1h is None else int(hot1h.shape[-1])
        o_gid, o_hot, o_hprim, o_hcopy, o_slot, length = table_offsets(
            n_experts, nc, lanes=lanes
        )
        tab = jnp.asarray(tables, jnp.int32)
        if tab.shape != (length,):
            raise ValueError(
                f"packed tables for {n_experts} slots, {nc} hot entries "
                f"and {lanes} lanes are [{length}], got {tab.shape}"
            )
        if hot1h is None:
            hot1h = hot_onehot(tab[o_hot : o_hot + nc], lanes=lanes)
    if nc % _LANE:
        raise ValueError(f"the hot table's length {nc} must be a multiple of {_LANE}")
    hot1h = jnp.asarray(hot1h, jnp.bfloat16).reshape(lanes, nc)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    out = pl.pallas_call(
        functools.partial(
            _kernel_scores,
            n_experts=n_experts,
            inter=inter,
            w13_parts=w13_parts,
            topk=topk,
            ranks=ranks,
            owned=owned,
            has_copies=has_copies,
            offsets=(o_gid, o_hot, o_hprim, o_hcopy, o_slot),
        ),
        out_shape=jax.ShapeDtypeStruct((rows, hidden), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vmem, vmem, vmem, hbm, vmem, hbm, hbm, hbm, hbm],
            out_specs=vmem,
            scratch_shapes=(
                pltpu.SMEM((n_experts,), jnp.int32),
                pltpu.VMEM((rows, lanes), jnp.float32),
                pltpu.VMEM((1, lanes), jnp.int32),
                pltpu.SMEM((1, lanes), jnp.int32),
                pltpu.VMEM((8, nc), jnp.int32),
                pltpu.SMEM((8, nc), jnp.int32),
                pltpu.SemaphoreType.DMA((3,)),
                pltpu.VMEM((rows, hidden), x_rows.dtype),
                pltpu.SemaphoreType.DMA((1,)),
                *_stream_scratch(rows, hidden, inter, w13_parts),
            ),
            grid=(),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes
            or _vmem_budget(rows, hidden, inter, w13_parts, lanes, nc=nc)
        ),
        name="moe_lowtok",
    )(tab, prim, copy, hot1h, x_rows, sc, s13, s2, w13, w2)
    return out[:tokens]

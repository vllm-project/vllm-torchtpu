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
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

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


def _vmem_budget(rows, hidden, inter, w13_parts, lanes):
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
        rows * 2 * hidden * 2 + rows * 4 * lanes + 4 * lanes * 8 + 2 * lanes * lanes
    )
    need = int((weights + scales + accumulators + operands) * _VMEM_MARGIN)
    capacity = int(pltpu.get_tpu_info().vmem_capacity_bytes * _VMEM_FRACTION)
    return min(need, capacity)


def _quant_block(k):
    """Activation scale block along a contraction of width k."""
    return min(k, _QUANT_BLOCK)


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
):
    """Stream the hit experts of this shard and accumulate their outputs.

    gid_s [1, E] (SMEM) is the table lane of each local expert; hits_s
    [1, lanes] (SMEM) is 1 on lanes some row routes to; wtab_ref [rows, lanes]
    holds the routing weights. Each expert is `w13_parts` gate|up units then
    one down unit; the ring has one slot per unit, so slot == part.
    """
    half = inter // w13_parts
    units = w13_parts + 1
    slots = units
    lane = jax.lax.broadcasted_iota(jnp.int32, wtab_ref.shape, 1)

    def collect(e, n):
        h = hits_s[0, gid_s[0, e]]

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
                jnp.where(lane == gid_s[0, expert], wtab_ref[...], 0.0),
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

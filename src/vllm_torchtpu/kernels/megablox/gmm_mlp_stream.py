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
"""Decode-sized routed-expert MLP (gate_up, activation, down) as one weight stream.

A decode step gives each local expert a few rows, so the call is bound by
streaming the experts' fp4 weights. gmm_v2's decode tiles fetch gate_up in
[K, tile_n] column blocks, i.e. many short strided rows, and stream measurably
below what contiguous transfers of the same bytes reach. Here each nonempty
expert is three contiguous row chunks of equal size: gate_up's two K halves
``w1[e, :K/2]`` and ``w1[e, K/2:]`` (all [gate | up] columns) and ``w2[e]``,
streamed through a VMEM ring with their 128-row block scales. gate_up
accumulates over both halves in f32, the activation and the down projection
follow while the next chunks stream in.

The arithmetic is gmm_v2's fp8 rhs feed (`FP8_RHS_MAX_BUCKET_M`): bf16 rows
against the fp4 weights cast to e4m3 (exact), one f32 product per 128-row
quant block scaled by its block scale, accumulated in K order, activation in
f32, the activation rounded to bf16 before down. An expert's rows run in
windows of at most 48 rows; gmm_v2 runs groups that fill more of a tile on
bf16-dequantized weights instead, so only those groups round differently.

Given the routing's inverse permutation, the kernel gathers the routed rows
from the tokens itself (no permuted copy in HBM): each routed row's packed bf16
row pair is one DMA, and the token's half of each word is selected in VMEM.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox.gmm_v2 import (
    FP8_RHS_MAX_BUCKET_M,
    _get_lhs_sublane_size,
    align_to,
    swigluoai,
)

_QUANT_BLOCK = 128
# Ring slots: two chunks in flight while the third is computed on.
_SLOTS = 3
# Row windows per pass over an expert's chunks (acc / activation scratch);
# an expert with more rows streams its chunks once per pass.
_PASS_WINDOWS = 2
# Row tiles whose write-back may be in flight at once.
_OUT_SLOTS = 8
_ACTIVATIONS = {"swigluoai": swigluoai}
# Routed rows per block of the in-kernel inverse permutation (one vreg of lanes).
_SLOT_BLOCK = 128
# Lanes per step of the routed-row selection (bounds its temporaries).
_SELECT_COLS = 1024


def supported(
    lhs: jax.Array,
    w1: jax.Array,
    s1: jax.Array | None,
    w2: jax.Array,
    s2: jax.Array | None,
    fuse_act: str | None,
) -> bool:
    """Shapes and dtypes this kernel serves: bf16 rows, fp4 K-major experts
    with f32 128-row block scales, ``w1`` [G, K, 2I] holding [gate | up] and
    ``w2`` [G, I, K] with I = K / 2 (so all three chunks are one shape)."""
    if s1 is None or s2 is None or fuse_act not in _ACTIVATIONS:
        return False
    if lhs.dtype != jnp.bfloat16 or w1.dtype != w2.dtype:
        return False
    if w1.dtype != jnp.float4_e2m1fn or not (s1.dtype == s2.dtype == jnp.float32):
        return False
    info = pltpu.get_tpu_info()
    if not info.is_matmul_supported(jnp.bfloat16, jnp.float8_e4m3fn):
        return False
    g, k, two_i = w1.shape
    i = two_i // 2
    sub = info.get_sublane_tiling(lhs.dtype)
    return (
        lhs.ndim == 2
        # Row tiles of whole packed sublane tiles (decode buffers are larger).
        and lhs.shape[0] >= sub
        and FP8_RHS_MAX_BUCKET_M % sub == 0
        # A lower bound: the shapes it serves compile within the capacity.
        and _vmem_bytes(i, k, sub) <= info.vmem_capacity_bytes
        and lhs.shape[1] == k
        and w2.shape == (g, i, k)
        and 2 * i == k
        and i % (2 * _QUANT_BLOCK) == 0
        and s1.shape == (g, k // _QUANT_BLOCK, 1, two_i)
        and s2.shape == (g, i // _QUANT_BLOCK, 1, k)
    )


def gathers(num_tokens: int, topk: int, w1: jax.Array) -> bool:
    """Whether the kernel can gather the routed rows of ``num_tokens`` bf16
    tokens with ``topk`` routes each itself (``gmm_mlp_stream(revert=...)``):
    whole row pairs and sublane tiles of tokens, within VMEM."""
    info = pltpu.get_tpu_info()
    _, k, two_i = w1.shape
    sub = info.get_sublane_tiling(jnp.bfloat16)
    return (
        num_tokens % sub == 0
        and topk > 0
        and k % _SELECT_COLS == 0
        and _vmem_bytes(two_i // 2, k, sub, (topk, num_tokens))
        <= info.vmem_capacity_bytes
    )


def _vmem_bytes(i: int, k: int, sub: int, routes: tuple[int, int] = (0, 0)) -> int:
    """The kernel's scratch, a lower bound on its VMEM (fp4 takes a byte per
    element in VMEM); ``routes``: (topk, tokens) it gathers rows by, if any."""
    win = FP8_RHS_MAX_BUCKET_M
    ring = _SLOTS * (i * k + (i // _QUANT_BLOCK) * k * 4)
    # x_buf (gather: one pass's 32-bit row pairs, the same bytes), plus the
    # inverse permutation it gathers by (int32 [topk, T] in (8, 128) tiles).
    topk, tokens = routes
    rows = (
        2 * _PASS_WINDOWS * win * k * 2 + align_to(topk, 8) * align_to(tokens, 128) * 4
    )
    rows += _PASS_WINDOWS * win * (2 * i * 4 + i * 2)  # acc_buf, h_buf
    rows += (win + (_OUT_SLOTS + 1) * sub) * k * 2  # o_buf, o_ring, p_buf
    return ring + rows


def _kernel(
    gs_ref,  # SMEM int32 [G]
    *refs,
    groups: int,
    sub: int,
    act: str,
    gather: bool,
):
    if gather:
        # VMEM int32 [topk, T]: each route's routed row (the inverse permutation),
        # slot-major as the routing prep builds it.
        revert_ref, *refs = refs
    (
        lhs_hbm,  # [M, K] bf16 routed rows, or [T, K] bf16 tokens (gather)
        w1_hbm,  # [G, K, 2I] fp4
        s1_hbm,  # [G, K / 128, 1, 2I] f32
        w2_hbm,  # [G, I, K] fp4
        s2_hbm,  # [G, I / 128, 1, K] f32
        out_hbm,  # [M, K] bf16
        w_ring,  # [SLOTS, I, K] fp4 (every chunk is [K / 2, 2I] = [I, K])
        s_ring,  # [SLOTS, I / 128, 1, K] f32
        x_buf,  # [2, PW, WIN, K] bf16: rows of the current and the next pair;
        # gather: [PW, 2, WIN / 2, K] int32, routed row i's token row pair at
        # [., i % 2, i // 2] (even and odd rows apart, no strided loads), the
        # even half then overwritten in place by the selected bf16 rows
        acc_buf,  # [PW, WIN, 2I] f32
        h_buf,  # [PW, WIN, I] bf16
        o_buf,  # [WIN, K] bf16
        o_ring,  # [OUT_SLOTS, SUB, K] bf16: row tiles being written back
        p_buf,  # [SUB, K] bf16: the last tile of a group that ends inside it
        tab,  # SMEM int32 [5, max_pairs]: expert, window start, windows, row0, row1
        st,  # SMEM int32 [2]: pending tile's first row (-1: none), tiles written
        w_sem,
        s_sem,
        x_sem,
        o_sem,
        *gather_scratch,
    ) = refs
    if gather:
        (
            tok_v,  # VMEM int32 [ceil(M / 128) padded to 8, 128]: routed row -> token
            tok_s,  # SMEM, its copy
            shift_s,  # SMEM int32 [PW * WIN]: the token's half in its pair (0 / 16)
            tok_sem,
        ) = gather_scratch
        # A packed bf16 row pair is one 32-bit row, the smallest DMA unit.
        lhs_words = lhs_hbm.bitcast(jnp.int32)
    fp8 = jnp.float8_e4m3fn
    f32 = jnp.float32
    win = FP8_RHS_MAX_BUCKET_M
    half, cols = w_ring.shape[1], w_ring.shape[2]  # I, K (== 2I)
    nb = half // _QUANT_BLOCK
    lanes = 2 * _QUANT_BLOCK
    buckets = tuple(b for b in range(sub, win + 1, sub))
    out_slots = o_ring.shape[0]

    # (expert, pass) pairs over nonempty experts, rows in ascending order.
    def plan(e, carry):
        off, npairs = carry
        n = gs_ref[e]
        lo = off % sub
        nwin = jnp.where(n > 0, (lo + n + win - 1) // win, 0)
        npass = (nwin + _PASS_WINDOWS - 1) // _PASS_WINDOWS

        def one(p, np_):
            tab[0, np_] = e
            tab[1, np_] = off - lo + p * (_PASS_WINDOWS * win)
            tab[2, np_] = jnp.minimum(_PASS_WINDOWS, nwin - p * _PASS_WINDOWS)
            tab[3, np_] = off
            tab[4, np_] = off + n
            return np_ + 1

        return off + n, lax.fori_loop(0, npass, one, npairs)

    rows_total, npairs = lax.fori_loop(0, groups, plan, (jnp.int32(0), jnp.int32(0)))
    units = npairs * 3
    st[0] = -1
    st[1] = 0

    # Weight chunks: unit u is part u % 3 of pair u // 3.
    def copies(u):
        e = tab[0, u // 3]
        slot = u % _SLOTS
        w_dst, s_dst = w_ring.at[slot], s_ring.at[slot]
        return (
            (
                pltpu.make_async_copy(
                    w1_hbm.at[e, pl.ds(0, half), :], w_dst, w_sem.at[slot]
                ),
                pltpu.make_async_copy(
                    s1_hbm.at[e, pl.ds(0, nb)], s_dst, s_sem.at[slot]
                ),
            ),
            (
                pltpu.make_async_copy(
                    w1_hbm.at[e, pl.ds(half, half), :], w_dst, w_sem.at[slot]
                ),
                pltpu.make_async_copy(
                    s1_hbm.at[e, pl.ds(nb, nb)], s_dst, s_sem.at[slot]
                ),
            ),
            (
                pltpu.make_async_copy(w2_hbm.at[e], w_dst, w_sem.at[slot]),
                pltpu.make_async_copy(s2_hbm.at[e], s_dst, s_sem.at[slot]),
            ),
        )

    def on_chunk(u, method):
        part = u % 3
        for p, pair in enumerate(copies(u)):

            @pl.when(part == p)
            def _(pair=pair):
                for c in pair:
                    getattr(c, method)()

    def pair_copy(src_pair, w, i):
        return pltpu.make_async_copy(
            lhs_words.at[pl.ds(src_pair, 1)],
            x_buf.at[w, i % 2, pl.ds(i // 2, 1)],
            x_sem.at[0],
        )

    # A pair's rows: its windows' sublane tiles up to the group's end (gather:
    # each of the group's rows as its token's row pair).
    def on_rows(pair, method):
        parity = pair % 2
        start, nwin, row1 = tab[1, pair], tab[2, pair], tab[4, pair]
        if gather:
            row0 = tab[3, pair]
            for w in range(_PASS_WINDOWS):
                w0 = start + w * win

                def one(r, c, w=w, w0=w0):
                    if method == "start":
                        tok = tok_s[r // _SLOT_BLOCK, r % _SLOT_BLOCK]
                        shift_s[w * win + r - w0] = (tok % 2) * 16
                        pair_copy(tok // 2, w, r - w0).start()
                    else:
                        pair_copy(0, w, 0).wait()
                    return c

                @pl.when(w < nwin)
                def _(w0=w0, one=one):
                    hi = jnp.minimum(w0 + win, row1)
                    lax.fori_loop(jnp.maximum(w0, row0), hi, one, 0)

            return
        for w in range(_PASS_WINDOWS):
            for t in range(win // sub):
                t0 = pl.multiple_of(start + w * win + t * sub, sub)

                @pl.when(jnp.logical_and(w < nwin, t0 < row1))
                def _(w=w, t=t, t0=t0):
                    getattr(
                        pltpu.make_async_copy(
                            lhs_hbm.at[pl.ds(t0, sub)],
                            x_buf.at[parity, w, pl.ds(t * sub, sub)],
                            x_sem.at[parity],
                        ),
                        method,
                    )()

    # Gather: the first pair's row pairs queue right behind the first chunk.
    for u0 in range(1 if gather else _SLOTS - 1):

        @pl.when(u0 < units)
        def _(u0=u0):
            on_chunk(jnp.int32(u0), "start")

    if gather:
        # Routed row -> token for the rows the groups hold, under the first
        # weight copies: one 128-row block of the inverse permutation at a time.
        topk, tokens = revert_ref.shape
        slot = lax.broadcasted_iota(jnp.int32, (_SLOT_BLOCK, tokens), 0)
        token = lax.broadcasted_iota(jnp.int32, (_SLOT_BLOCK, tokens), 1)
        diagonal = lax.broadcasted_iota(
            jnp.int32, (_SLOT_BLOCK, _SLOT_BLOCK), 0
        ) == lax.broadcasted_iota(jnp.int32, (_SLOT_BLOCK, _SLOT_BLOCK), 1)
        for b in range(pl.cdiv(out_hbm.shape[0], _SLOT_BLOCK)):

            @pl.when(b * _SLOT_BLOCK < rows_total)
            def _(b=b):
                hit = jnp.zeros((_SLOT_BLOCK, tokens), jnp.int32)
                for k in range(topk):
                    route = revert_ref[k : k + 1, :] == slot + b * _SLOT_BLOCK
                    hit = jnp.where(route, token, hit)
                # The block's tokens from slots on sublanes to slots on lanes.
                column = jnp.sum(hit, axis=1, keepdims=True)
                row = jnp.sum(jnp.where(diagonal, column, 0), axis=0, keepdims=True)
                tok_v[b : b + 1, :] = row

        tok_copy = pltpu.make_async_copy(tok_v, tok_s, tok_sem.at[0])
        tok_copy.start()
        tok_copy.wait()

    @pl.when(npairs > 0)
    def _():
        on_rows(jnp.int32(0), "start")

    for u0 in range(1 if gather else _SLOTS - 1, _SLOTS - 1):

        @pl.when(u0 < units)
        def _(u0=u0):
            on_chunk(jnp.int32(u0), "start")

    def out_copy(slot, t0):
        return pltpu.make_async_copy(
            o_ring.at[slot],
            out_hbm.at[pl.ds(pl.multiple_of(t0, sub), sub)],
            o_sem.at[slot],
        )

    def write_tile(tile, t0):
        n = st[1]
        slot = n % out_slots

        @pl.when(n >= out_slots)
        def _():
            out_copy(slot, 0).wait()  # that slot's previous write

        o_ring[slot] = tile
        out_copy(slot, t0).start()
        st[1] = n + 1

    def project(lhs_ref, k0, slot, acc_ref, first, bm):
        """acc_ref[:bm] = (first ? 0 : acc_ref[:bm]) + lhs_ref[:bm, k0 : k0 + I]
        @ chunk, each 128-row quant block's product scaled by its block scale,
        in K order (gmm_v2's fp8 rhs feed). Unrolled: the scheduler overlaps
        one block's weight push with the previous block's pops."""
        w_ref, s_ref = w_ring.at[slot], s_ring.at[slot]
        for n0 in range(0, cols, lanes):
            prev = acc_ref[pl.ds(0, bm), pl.ds(n0, lanes)]
            acc = jnp.where(first, jnp.zeros_like(prev), prev)
            for b in range(nb):
                k = b * _QUANT_BLOCK
                blk = jnp.matmul(
                    lhs_ref[
                        pl.ds(0, bm),
                        pl.ds(pl.multiple_of(k0 + k, _QUANT_BLOCK), _QUANT_BLOCK),
                    ],
                    w_ref[pl.ds(k, _QUANT_BLOCK), pl.ds(n0, lanes)].astype(fp8),
                    preferred_element_type=f32,
                )
                acc += blk * s_ref[b, :, pl.ds(n0, lanes)]
            acc_ref[pl.ds(0, bm), pl.ds(n0, lanes)] = acc

    def select(w, bm):
        """Window w's first bm routed rows as bf16 rows, in place of its even
        row pairs: each pair's half that holds the routed row (a packed bf16
        word holds rows 2j and 2j + 1 in its low and high half)."""
        h2 = bm // 2
        row = lax.broadcasted_iota(jnp.int32, (h2, 1), 0)
        shifts = [jnp.zeros((h2, 1), jnp.int32), jnp.zeros((h2, 1), jnp.int32)]
        for j in range(bm):
            v = shift_s[w * win + j]
            shifts[j % 2] = jnp.where(row == j // 2, v, shifts[j % 2])
        for c0 in range(0, x_buf.shape[-1], _SELECT_COLS):
            c = pl.ds(c0, _SELECT_COLS)
            even = x_buf[w, 0, pl.ds(0, h2), c]
            odd = x_buf[w, 1, pl.ds(0, h2), c]
            lo = lax.shift_right_logical(even, jnp.broadcast_to(shifts[0], even.shape))
            hi = lax.shift_right_logical(odd, jnp.broadcast_to(shifts[1], odd.shape))
            words = (lo & 0xFFFF) | lax.shift_left(hi, 16)
            # Zero a non-finite value (all-ones exponent) as the one-hot
            # permute does: the combine's one-hot matmul would carry it from
            # its routed row to every token (0 * NaN).
            for exponent, value in ((0x7F80, 0xFFFF), (0x7F800000, -0x10000)):
                words = jnp.where((words & exponent) == exponent, words & ~value, words)
            x_buf[w, 0, pl.ds(0, h2), c] = words

    def bucketed(rows, fn):
        """fn(bm) on the smallest row bucket bm >= rows."""
        idx = sum((rows > b).astype(jnp.int32) for b in buckets[:-1])
        lax.switch(idx, [functools.partial(fn, bm=b) for b in buckets])

    def store_rows(w0, row0, row1):
        """Writes the window's rows of this group. A tile the group shares with
        the previous group merges into the pending tile that group left; the
        group's own last tile, when it ends inside it, becomes the pending one."""
        for t in range(win // sub):
            t0 = pl.multiple_of(w0 + t * sub, sub)

            @pl.when(t0 < row1)
            def _(t=t, t0=t0):
                r = t0 + lax.broadcasted_iota(jnp.int32, p_buf.shape, 0)
                keep = jnp.logical_and(st[0] == t0, r < row0)
                tile = jnp.where(keep, p_buf[...], o_buf[pl.ds(t * sub, sub), :])

                @pl.when(t0 + sub > row1)
                def _():
                    p_buf[...] = tile
                    st[0] = t0

                @pl.when(t0 + sub <= row1)
                def _():
                    write_tile(tile, t0)

                    @pl.when(st[0] == t0)
                    def _():
                        st[0] = -1

    def body(u, carry):
        pair, part, slot = u // 3, u % 3, u % _SLOTS
        parity = pair % 2
        start, nwin, row0, row1 = tab[1, pair], tab[2, pair], tab[3, pair], tab[4, pair]

        if gather:
            # The next pair's row pairs land in the buffer once this pair's
            # gate_up is done with it, queued ahead of the next weight chunk.
            @pl.when((part == 2) & (pair + 1 < npairs))
            def _():
                on_rows(pair + 1, "start")

        @pl.when(u + _SLOTS - 1 < units)
        def _():
            on_chunk(u + _SLOTS - 1, "start")

        @pl.when(part == 0)
        def _():
            on_rows(pair, "wait")

            if not gather:

                @pl.when(pair + 1 < npairs)
                def _():
                    on_rows(pair + 1, "start")

        on_chunk(u, "wait")

        def window(w, c):
            w0 = pl.multiple_of(start + w * win, sub)
            rows = jnp.minimum(win, row1 - w0)  # window rows up to the group's end

            if gather:

                @pl.when(part == 0)
                def _():
                    bucketed(rows, lambda bm: select(w, bm))

            @pl.when(part < 2)
            def _():
                # gate_up over K half ``part``, accumulated across both halves.
                k0 = pl.multiple_of(part * half, _QUANT_BLOCK)
                bucketed(
                    rows,
                    lambda bm: project(
                        x_buf.at[w, 0].bitcast(jnp.bfloat16)
                        if gather
                        else x_buf.at[parity, w],
                        k0,
                        slot,
                        acc_buf.at[w],
                        part == 0,
                        bm,
                    ),
                )

                @pl.when(part == 1)
                def _():
                    acc = acc_buf[w]
                    h_buf[w] = _ACTIVATIONS[act](acc[:, :half], acc[:, half:]).astype(
                        h_buf.dtype
                    )

            @pl.when(part == 2)
            def _():
                # down into acc_buf[w] (free after the activation).
                bucketed(
                    rows,
                    lambda bm: project(h_buf.at[w], 0, slot, acc_buf.at[w], True, bm),
                )
                o_buf[...] = acc_buf[w][:, :cols].astype(o_buf.dtype)
                store_rows(w0, row0, row1)

            return c

        lax.fori_loop(0, nwin, window, 0)
        return carry

    lax.fori_loop(0, units, body, 0)

    @pl.when(st[0] >= 0)
    def _():
        write_tile(p_buf[...], st[0])

    for slot in range(out_slots):

        @pl.when(slot < st[1])
        def _(slot=slot):
            out_copy(slot, 0).wait()


@functools.partial(jax.jit, static_argnames=["fuse_act"])
def gmm_mlp_stream(
    lhs: jax.Array,  # [M, K] bf16, rows grouped by expert (or [T, K] tokens)
    w1: jax.Array,  # [G, K, 2I] fp4, [gate | up]
    w2: jax.Array,  # [G, I, K] fp4
    group_sizes: jax.Array,  # int32 [G]
    s1: jax.Array,  # [G, K / 128, 1, 2I] f32
    s2: jax.Array,  # [G, I / 128, 1, K] f32
    *,
    fuse_act: str,
    revert: jax.Array | None = None,  # int32 [T * topk]
) -> jax.Array:
    """down(act(gate_up(lhs))) per group, as gmm_v2_mlp on decode tiles: the
    first ``sum(group_sizes)`` rows of the [M, K] bf16 result are defined.

    With ``revert`` (fused_moe_gmm's ``argsort_revert_indices``: route k of
    token t is routed row ``revert[t * topk + k]``), ``lhs`` holds the [T, K]
    tokens and routed row r is the row of the token whose route maps to r
    (M = T * topk), gathered in the kernel with non-finite values zeroed, as
    bounded_onehot_permute gives them."""
    g, k, two_i = w1.shape
    i = two_i // 2
    gather = revert is not None
    m = revert.size if gather else lhs.shape[0]
    sub = _get_lhs_sublane_size(lhs.dtype, m)
    win = FP8_RHS_MAX_BUCKET_M
    assert win % sub == 0, (win, sub)
    mp = align_to(m, sub)
    gather_operands, gather_specs, gather_scratch = (), [], []
    if gather:
        tokens = lhs.shape[0]
        assert tokens % sub == 0 and m % tokens == 0, (tokens, m, sub)
        # Slot-major: the counting sort's layout, so the transposes cancel.
        gather_operands = (revert.reshape(tokens, m // tokens).T.astype(jnp.int32),)
        gather_specs = [pl.BlockSpec(memory_space=pltpu.VMEM)]
        blocks = align_to(pl.cdiv(mp, _SLOT_BLOCK), 8)
        gather_scratch = [
            pltpu.VMEM((blocks, _SLOT_BLOCK), jnp.int32),
            pltpu.SMEM((blocks, _SLOT_BLOCK), jnp.int32),
            pltpu.SMEM((_PASS_WINDOWS * win,), jnp.int32),
            pltpu.SemaphoreType.DMA((1,)),
        ]
    elif mp != m:
        lhs = jnp.pad(lhs, ((0, mp - m), (0, 0)))
    # Every group but the last can add one window (sublane offset) and a pass.
    max_pairs = g + pl.cdiv(mp + g * win, win * _PASS_WINDOWS)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem_limit = pltpu.get_tpu_info().vmem_capacity_bytes
    out = pl.pallas_call(
        functools.partial(_kernel, groups=g, sub=sub, act=fuse_act, gather=gather),
        out_shape=jax.ShapeDtypeStruct((mp, k), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[*gather_specs, *[hbm] * 5],
            out_specs=hbm,
            scratch_shapes=[
                pltpu.VMEM((_SLOTS, i, k), w1.dtype),
                pltpu.VMEM((_SLOTS, i // _QUANT_BLOCK, 1, k), jnp.float32),
                pltpu.VMEM((_PASS_WINDOWS, 2, win // 2, k), jnp.int32)
                if gather
                else pltpu.VMEM((2, _PASS_WINDOWS, win, k), jnp.bfloat16),
                pltpu.VMEM((_PASS_WINDOWS, win, two_i), jnp.float32),
                pltpu.VMEM((_PASS_WINDOWS, win, i), jnp.bfloat16),
                pltpu.VMEM((win, k), jnp.bfloat16),
                pltpu.VMEM((_OUT_SLOTS, sub, k), jnp.bfloat16),
                pltpu.VMEM((sub, k), jnp.bfloat16),
                pltpu.SMEM((5, max_pairs), jnp.int32),
                pltpu.SMEM((2,), jnp.int32),
                pltpu.SemaphoreType.DMA((_SLOTS,)),
                pltpu.SemaphoreType.DMA((_SLOTS,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((_OUT_SLOTS,)),
                *gather_scratch,
            ],
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit),
        name=f"gmm_mlp_stream-g_{g}-m_{mp}-k_{k}-i_{i}" + ("-gather" if gather else ""),
    )(group_sizes.astype(jnp.int32), *gather_operands, lhs, w1, s1, w2, s2)
    return out[:m]

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


def _vmem_bytes(i: int, k: int, sub: int) -> int:
    """The kernel's scratch, a lower bound on its VMEM (fp4 takes a byte per
    element in VMEM)."""
    win = FP8_RHS_MAX_BUCKET_M
    ring = _SLOTS * (i * k + (i // _QUANT_BLOCK) * k * 4)
    rows = 2 * _PASS_WINDOWS * win * k * 2  # x_buf
    rows += _PASS_WINDOWS * win * (2 * i * 4 + i * 2)  # acc_buf, h_buf
    rows += (win + (_OUT_SLOTS + 1) * sub) * k * 2  # o_buf, o_ring, p_buf
    return ring + rows


def _kernel(
    gs_ref,  # SMEM int32 [G]
    lhs_hbm,  # [M, K] bf16
    w1_hbm,  # [G, K, 2I] fp4
    s1_hbm,  # [G, K / 128, 1, 2I] f32
    w2_hbm,  # [G, I, K] fp4
    s2_hbm,  # [G, I / 128, 1, K] f32
    out_hbm,  # [M, K] bf16
    w_ring,  # [SLOTS, I, K] fp4 (every chunk is [K / 2, 2I] = [I, K])
    s_ring,  # [SLOTS, I / 128, 1, K] f32
    x_buf,  # [2, PW, WIN, K] bf16: rows of the current and the next pair
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
    *,
    groups: int,
    sub: int,
    act: str,
):
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

    _, npairs = lax.fori_loop(0, groups, plan, (jnp.int32(0), jnp.int32(0)))
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

    # A pair's rows: its windows' sublane tiles up to the group's end.
    def on_rows(pair, method):
        parity = pair % 2
        start, nwin, row1 = tab[1, pair], tab[2, pair], tab[4, pair]
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

    for u0 in range(_SLOTS - 1):

        @pl.when(u0 < units)
        def _(u0=u0):
            on_chunk(jnp.int32(u0), "start")

    @pl.when(npairs > 0)
    def _():
        on_rows(jnp.int32(0), "start")

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
        @pl.when(u + _SLOTS - 1 < units)
        def _():
            on_chunk(u + _SLOTS - 1, "start")

        pair, part, slot = u // 3, u % 3, u % _SLOTS
        parity = pair % 2
        start, nwin, row0, row1 = tab[1, pair], tab[2, pair], tab[3, pair], tab[4, pair]

        @pl.when(part == 0)
        def _():
            on_rows(pair, "wait")

            @pl.when(pair + 1 < npairs)
            def _():
                on_rows(pair + 1, "start")

        on_chunk(u, "wait")

        def window(w, c):
            w0 = pl.multiple_of(start + w * win, sub)
            rows = jnp.minimum(win, row1 - w0)  # window rows up to the group's end

            @pl.when(part < 2)
            def _():
                # gate_up over K half ``part``, accumulated across both halves.
                k0 = pl.multiple_of(part * half, _QUANT_BLOCK)
                bucketed(
                    rows,
                    lambda bm: project(
                        x_buf.at[parity, w], k0, slot, acc_buf.at[w], part == 0, bm
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
    lhs: jax.Array,  # [M, K] bf16, rows grouped by expert
    w1: jax.Array,  # [G, K, 2I] fp4, [gate | up]
    w2: jax.Array,  # [G, I, K] fp4
    group_sizes: jax.Array,  # int32 [G]
    s1: jax.Array,  # [G, K / 128, 1, 2I] f32
    s2: jax.Array,  # [G, I / 128, 1, K] f32
    *,
    fuse_act: str,
) -> jax.Array:
    """down(act(gate_up(lhs))) per group, as gmm_v2_mlp on decode tiles: the
    first ``sum(group_sizes)`` rows of the [M, K] bf16 result are defined."""
    g, k, two_i = w1.shape
    i = two_i // 2
    m = lhs.shape[0]
    sub = _get_lhs_sublane_size(lhs.dtype, m)
    win = FP8_RHS_MAX_BUCKET_M
    assert win % sub == 0, (win, sub)
    mp = align_to(m, sub)
    if mp != m:
        lhs = jnp.pad(lhs, ((0, mp - m), (0, 0)))
    # Every group but the last can add one window (sublane offset) and a pass.
    max_pairs = g + pl.cdiv(mp + g * win, win * _PASS_WINDOWS)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem_limit = pltpu.get_tpu_info().vmem_capacity_bytes
    out = pl.pallas_call(
        functools.partial(_kernel, groups=g, sub=sub, act=fuse_act),
        out_shape=jax.ShapeDtypeStruct((mp, k), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[hbm] * 5,
            out_specs=hbm,
            scratch_shapes=[
                pltpu.VMEM((_SLOTS, i, k), w1.dtype),
                pltpu.VMEM((_SLOTS, i // _QUANT_BLOCK, 1, k), jnp.float32),
                pltpu.VMEM((2, _PASS_WINDOWS, win, k), jnp.bfloat16),
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
            ],
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit),
        name=f"gmm_mlp_stream-g_{g}-m_{mp}-k_{k}-i_{i}",
    )(group_sizes.astype(jnp.int32), lhs, w1, s1, w2, s2)
    return out[:m]

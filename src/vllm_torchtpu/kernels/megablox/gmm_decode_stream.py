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
"""Streamed decode MoE expert MLP: gate_up + act + down in one weight stream.

Decode-sized grouped matmuls read each hit expert's weights once and are
bound by that read. gmm_v2_mlp runs a pipeline whose grid step is one whole
expert block (9.4-19 MB), so every call exposes one block: the first block's
DMA before any compute and the last block's compute after the last DMA. At
decode that is ~11 us per MoE layer per core, as much as one expert.

This kernel streams every hit expert's gate_up rows and then its down rows as
one sequence of equal k-slices ("chunks", chunk_rows x N fp4 + its block
scales) through a ring of VMEM slots, slots - 1 chunks ahead. Only one chunk's
DMA is exposed at the start and one chunk's compute at the end, and the down
weights of an expert stream in under its gate_up compute.

With ``shared`` the call also computes a TP slice of a dense shared expert
(e4m3 weights, one scale per output column) on the unpermuted token rows and
returns it as a second output. Its weights are copied to VMEM in three DMAs
issued before the routed chunks, and it is computed at the first routed chunk,
while the next chunks stream, so the shared expert costs its weight bytes in
the same stream and no separate weight-bound call. Its roundings follow the
torch shared MLP: bf16 matmul results times the power-of-two column scale, the
SwiGLU-OAI activation in bf16, bf16 out.

Numerics equal gmm_v2's fp8 rhs feed (decode tiles, at most
FP8_RHS_MAX_BUCKET_M... per tile there; here any bucket, rows are independent):
per output column, the f32 sum over 128-deep quant blocks runs in the same
order with the same block products (bf16 lhs x fp8(fp4) rhs, f32 result times
the f32 block scale), carried in VMEM between chunks; the activation and the
bf16 roundings of h and out are the same. Rows outside the group are masked
the same way.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import tight_vmem
from vllm_torchtpu.kernels.megablox import gmm_v2 as _gmm

_QB = 128  # quant block rows (NVFP4 requantized to block 128)
_NCOL = 256  # output columns per matmul (MXU width)
_SUB = 16  # bf16 sublane tiling of lhs / out rows
_ACT_COLS = 512  # columns per activation / output-merge step
_VMEM_MARGIN = 2 << 20  # Mosaic internal scratch over vmem_bytes()


def plan(size_m: int, max_group_size: int | None) -> tuple[tuple[int, ...], int | None]:
    """(row buckets of one entry, first bucket fed bf16-dequantized or None).

    An entry is a group's rows, split as gmm_v2 splits groups into tiles of
    at most buckets[-1] rows counted from the 16-row slab start. Up to
    FP8_SMALL_DECODE_MAX_M rows gmm_v2 feeds every tile fp8 (rows are
    independent, so any bucket gives its results). Above it, gmm_v2's tiles
    are 128 rows with buckets 16/32/48 on the fp8 feed and 128 dequantized to
    bf16 before the matmul; the same split and feeds are used here.
    """
    max_rows = min(max_group_size or size_m, size_m)
    if size_m <= _gmm.FP8_SMALL_DECODE_MAX_M:
        cap = min(_gmm.align_to(max_rows + _SUB - 1, _SUB), 128)
        return (*[b for b in (16, 32, 64) if b < cap], cap), None
    return tuple(_gmm.FP8_DECODE_BUCKETS), _gmm.FP8_DECODE_TILE_M


def parse_spec(raw: str) -> tuple[int, int, int] | None:
    """'<chunk_rows>:<slots>[@<max size_m>]' -> (chunk_rows, slots, max_m).

    'on' / '1' mean 768:4@512. Empty / '0' = off.
    """
    raw = (raw or "").strip()
    if raw in ("", "0", "off", "false", "False"):
        return None
    if raw in ("1", "on", "true", "True"):
        raw = "768:4@512"
    spec, _, max_m = raw.partition("@")
    parts = [int(x) for x in spec.split(":") if x.strip()]
    chunk = parts[0] if parts else 768
    slots = parts[1] if len(parts) > 1 else 4
    if chunk % _QB or slots < 2:
        raise ValueError(f"TPU_GMM_V2_DECODE_STREAM={raw!r}: chunk % 128, slots >= 2")
    max_m = int(max_m) if max_m else 512
    if max_m > _gmm.FP8_DECODE_MAX_M:
        raise ValueError(f"TPU_GMM_V2_DECODE_STREAM={raw!r}: max size_m <= 2048")
    return chunk, slots, max_m


def eligible(
    lhs, rhs1, rhs2, rhs1_scale, rhs2_scale, fuse_act, chunk_rows, max_m
) -> str | None:
    """None if the streamed kernel serves this call, else why not."""
    fp4 = jnp.float4_e2m1fn
    if fuse_act is None:
        return "no fused activation"
    if rhs1.dtype != fp4 or rhs2.dtype != fp4:
        return "rhs not fp4"
    if rhs1_scale is None or rhs2_scale is None:
        return "no scales"
    if lhs.dtype != jnp.bfloat16:
        return "lhs not bf16"
    size_m, k1 = lhs.shape
    g, k1w, n1 = rhs1.shape
    g2, f, n2 = rhs2.shape
    if size_m > max_m or size_m % _SUB:
        return "size_m"
    if k1w != k1 or g2 != g or n1 != 2 * f or n2 != n1:
        return "shapes"
    if k1 % chunk_rows or f % chunk_rows:
        return "chunk rows"
    if rhs1_scale.shape != (g, k1 // _QB, 1, n1) or rhs2_scale.shape != (
        g, f // _QB, 1, n2
    ):
        return "scale block"
    if n1 % _NCOL or f % _ACT_COLS:
        return "lanes"
    if not pltpu.get_tpu_info().is_matmul_supported(
        jnp.bfloat16, jnp.float8_e4m3fn
    ):
        return "no bf16 x fp8 matmul"
    return None


def _kernel(
    # Scalar prefetch
    gs_ref,  # int32[size_lhs_group]
    go_ref,  # int32[1]
    # In (HBM)
    lhs_hbm,  # [Mp, K1] bf16
    w1_hbm,  # [G, K1, N] fp4
    s1_hbm,  # [G, K1 / 128, 1, N] f32
    w2_hbm,  # [G, F, N] fp4
    s2_hbm,  # [G, F / 128, 1, N] f32
    *refs,
    num_groups: int,
    chunk: int,
    slots: int,
    buckets: tuple[int, ...],
    bf16_from: int | None,
    fuse_act: str,
    w_load: str = "tile",
    has_shared: bool = False,
):
    if has_shared:
        (xt_hbm, sw1_hbm, ss1_ref, sw2_hbm, ss2_ref, out_hbm, so_hbm,
         *scratch) = refs
        *scratch, xt_v, sw1_v, sw2_v, so_v, ssem = scratch
    else:
        out_hbm, *scratch = refs
    _routed_kernel(
        gs_ref, go_ref, lhs_hbm, w1_hbm, s1_hbm, w2_hbm, s2_hbm, out_hbm, *scratch,
        num_groups=num_groups, chunk=chunk, slots=slots, buckets=buckets,
        bf16_from=bf16_from, fuse_act=fuse_act, w_load=w_load,
        shared=(
            (xt_hbm, sw1_hbm, ss1_ref, sw2_hbm, ss2_ref, so_hbm, xt_v, sw1_v,
             sw2_v, so_v, ssem)
            if has_shared else None
        ),
    )


def _shared_expert(sh, fuse_act: str):
    """Copies and compute of the shared expert slice (see module doc)."""
    (xt_hbm, sw1_hbm, ss1_ref, sw2_hbm, ss2_ref, so_hbm, xt_v, sw1_v, sw2_v,
     so_v, ssem) = sh
    copies = [
        pltpu.make_async_copy(sw1_hbm, sw1_v, ssem.at[0]),
        pltpu.make_async_copy(sw2_hbm, sw2_v, ssem.at[1]),
        pltpu.make_async_copy(xt_hbm, xt_v, ssem.at[2]),
    ]
    out_copy = pltpu.make_async_copy(so_v, so_hbm, ssem.at[3])

    def start():
        for cp in copies:
            cp.start()

    def compute():
        if fuse_act != "swigluoai":
            raise NotImplementedError(fuse_act)
        for cp in copies:
            cp.wait()
        x = xt_v[...]
        k1, n1 = sw1_v.shape
        fs = n1 // 2
        n = sw2_v.shape[1]
        kstep = math.gcd(k1, 512)

        def proj(c0, width):
            acc = jnp.zeros((x.shape[0], width), jnp.float32)
            for k in range(0, k1, kstep):
                acc = acc + jnp.matmul(
                    x[:, k : k + kstep],
                    sw1_v[k : k + kstep, c0 : c0 + width],
                    preferred_element_type=jnp.float32,
                )
            # torch: bf16 matmul result, times the power-of-two column scale.
            return acc.astype(jnp.bfloat16) * ss1_ref[:, c0 : c0 + width].astype(
                jnp.bfloat16
            )

        gate = jnp.minimum(proj(0, fs), jnp.bfloat16(7.0))
        up = jnp.clip(proj(fs, fs), jnp.bfloat16(-7.0), jnp.bfloat16(7.0))
        glu = gate * jax.nn.sigmoid(jnp.bfloat16(1.702) * gate)
        h = glu * (up + jnp.bfloat16(1.0))
        ncol = math.gcd(n, _ACT_COLS)
        for c0 in range(0, n, ncol):
            y = jnp.matmul(
                h, sw2_v[:, c0 : c0 + ncol], preferred_element_type=jnp.float32
            )
            so_v[:, c0 : c0 + ncol] = (
                y.astype(jnp.bfloat16)
                * ss2_ref[:, c0 : c0 + ncol].astype(jnp.bfloat16)
            ).astype(so_v.dtype)
        out_copy.start()

    return start, compute, out_copy.wait


def _routed_kernel(
    gs_ref, go_ref, lhs_hbm, w1_hbm, s1_hbm, w2_hbm, s2_hbm,
    out_hbm,  # [Mp, N] out dtype
    # Scratch
    x_ref,  # VMEM [3, K1 / chunk, MCAP, chunk] bf16: lhs rows of 2 entries, h
    a_ref,  # VMEM [2, MCAP, N] f32: gate_up acc, down acc
    o_ref,  # VMEM [2, MCAP + 16, N] out dtype: output rows of 2 entries
    cs_ref,  # VMEM [16, N] out dtype: the open 16-row output slab
    w_ref,  # VMEM [slots, chunk, N] fp4
    s_ref,  # VMEM [slots, chunk / 128, N] f32
    ent_ref,  # SMEM [3, max entries] int32: group, first row, rows
    cnt_ref,  # SMEM [3] int32: entries, out DMA rows of entry slot 0 / 1
    wsem,  # DMA [slots, 2]
    xsem,  # DMA [2]
    osem,  # DMA [2]
    *,
    num_groups: int,
    chunk: int,
    slots: int,
    buckets: tuple[int, ...],
    bf16_from: int | None,
    fuse_act: str,
    w_load: str = "tile",
    shared=None,
):
    k1 = lhs_hbm.shape[1]
    n = w1_hbm.shape[2]
    f = w2_hbm.shape[1]
    mcap = buckets[-1]
    c1 = k1 // chunk
    ce = c1 + f // chunk
    kbc = chunk // _QB
    act_cols = math.gcd(chunk, _ACT_COLS)
    h_slot = 2

    # Entries in row order: each hit group split as gmm_v2's fill_metadata
    # splits it into tiles of at most mcap rows from the slab start.
    group_offset = go_ref[0]
    cnt_ref[0] = 0
    cnt_ref[1] = 0
    cnt_ref[2] = 0

    def scan(lg, m_off):
        g = lg - group_offset
        size = gs_ref[lg]

        @pl.when(jnp.logical_and(size > 0, g >= 0))
        def _():
            lo = m_off % _SUB
            first = jnp.minimum(mcap - lo, size)
            pieces = 1 + pl.cdiv(size - first, mcap)

            def piece(p, cur):
                rows = jnp.minimum(mcap - cur % _SUB, m_off + size - cur)
                k = cnt_ref[0]
                ent_ref[0, k] = g
                ent_ref[1, k] = cur
                ent_ref[2, k] = rows
                cnt_ref[0] = k + 1
                return cur + rows

            lax.fori_loop(0, pieces, piece, m_off)

        return m_off + size

    lax.fori_loop(0, group_offset + num_groups, scan, jnp.int32(0))
    num_ent = cnt_ref[0]
    num_chunks = num_ent * ce

    def ent(j):
        m_off = ent_ref[1, j]
        rows = ent_ref[2, j]
        a = pl.multiple_of(m_off - m_off % _SUB, _SUB)
        return ent_ref[0, j], m_off, rows, a

    def x_copies(j):
        # Entry j's lhs rows from the slab start, one k-slice per copy, so
        # compute indexes k-slices on a leading dim.
        _, m_off, rows, a = ent(j)
        lr = pl.multiple_of(pl.cdiv(m_off - a + rows, _SUB) * _SUB, _SUB)
        slot = j % 2
        return [
            pltpu.make_async_copy(
                lhs_hbm.at[pl.ds(a, lr), pl.ds(i * chunk, chunk)],
                x_ref.at[slot, i, pl.ds(0, lr)],
                xsem.at[slot],
            )
            for i in range(c1)
        ]

    def chunk_copies(t, w1_side: bool):
        """The weight and scale copies of chunk t from rhs1 or rhs2."""
        slot = t % slots
        c = t % ce
        e = ent_ref[0, t // ce]
        if w1_side:
            w_hbm, s_hbm, r = w1_hbm, s1_hbm, jnp.where(c < c1, c, 0)
        else:
            w_hbm, s_hbm, r = w2_hbm, s2_hbm, jnp.where(c < c1, 0, c - c1)
        return (
            pltpu.make_async_copy(
                w_hbm.at[e, pl.ds(pl.multiple_of(r * chunk, chunk), chunk)],
                w_ref.at[slot],
                wsem.at[slot, 0],
            ),
            pltpu.make_async_copy(
                s_hbm.at[e, pl.ds(r * kbc, kbc), 0], s_ref.at[slot], wsem.at[slot, 1]
            ),
        )

    def start_chunk(t):
        is_w1 = t % ce < c1

        @pl.when(is_w1)
        def _():
            for cp in chunk_copies(t, True):
                cp.start()

        @pl.when(jnp.logical_not(is_w1))
        def _():
            for cp in chunk_copies(t, False):
                cp.start()

    def wait_chunk(t):
        # Same byte counts from either side.
        for cp in chunk_copies(t, True):
            cp.wait()

    def out_copy(slot, a, size):
        return pltpu.make_async_copy(
            o_ref.at[slot, pl.ds(0, size)], out_hbm.at[pl.ds(a, size)], osem.at[slot]
        )

    def wait_out(slot):
        size = pl.multiple_of(cnt_ref[1 + slot], _SUB)

        @pl.when(size > 0)
        def _():
            out_copy(slot, 0, size).wait()
            cnt_ref[1 + slot] = 0

    def compute(t, mb):
        slot = t % slots
        j = t // ce
        c = t % ce
        _, m_off, rows, a = ent(j)
        is_w2 = c >= c1
        cc = jnp.where(is_w2, c - c1, c)
        xs = jnp.where(is_w2, h_slot, j % 2)
        ai = is_w2.astype(jnp.int32)
        bf16 = bf16_from is not None and mb >= bf16_from

        @pl.when(cc == 0)
        def _():
            a_ref[ai, 0:mb, :] = jnp.zeros((mb, n), jnp.float32)

        x = x_ref[xs, cc, 0:mb, :]
        w_slot = w_ref.at[slot]
        s_slot = s_ref.at[slot]
        w_all = w_slot[...] if w_load == "full" else None
        for c0 in range(0, n, _NCOL):
            acc = a_ref[ai, 0:mb, c0 : c0 + _NCOL]
            if bf16:
                # gmm_v2's dequantize-before-matmul: bf16(w) * bf16(scale)
                # in bf16, one matmul over the slice (f32 sums per slice).
                w = w_slot[:, pl.ds(c0, _NCOL)].astype(x.dtype)
                sc = s_slot[:, pl.ds(c0, _NCOL)].astype(x.dtype)
                w = (w.reshape(kbc, _QB, _NCOL) * sc[:, None, :]).reshape(chunk, _NCOL)
                acc = acc + jnp.matmul(x, w, preferred_element_type=jnp.float32)
            else:
                for kb in range(kbc):
                    if w_all is None:
                        w = w_slot[pl.ds(kb * _QB, _QB), pl.ds(c0, _NCOL)]
                    else:
                        w = w_all[kb * _QB : (kb + 1) * _QB, c0 : c0 + _NCOL]
                    blk = jnp.matmul(
                        x[:, kb * _QB : (kb + 1) * _QB],
                        w.astype(jnp.float8_e4m3fn),
                        preferred_element_type=jnp.float32,
                    )
                    # [1, 256] row, broadcast over the rows (same products).
                    acc = acc + blk * s_slot[kb : kb + 1, pl.ds(c0, _NCOL)]
            a_ref[ai, 0:mb, c0 : c0 + _NCOL] = acc

        @pl.when(c == c1 - 1)
        def _():
            # h = act(gate, up), rounded to the lhs dtype (gmm_v2's h).
            for c0 in range(0, f, act_cols):
                gate = a_ref[0, 0:mb, c0 : c0 + act_cols]
                up = a_ref[0, 0:mb, f + c0 : f + c0 + act_cols]
                lo = c0 % chunk
                x_ref[h_slot, c0 // chunk, 0:mb, lo : lo + act_cols] = (
                    _gmm.gate_up_act(gate, up, fuse_act).astype(x_ref.dtype)
                )

        @pl.when(c == ce - 1)
        def _():
            os = j % 2
            wait_out(os)  # entry j - 2's rows have left this slot
            i = lax.broadcasted_iota(jnp.int32, (mb, _ACT_COLS), 0)
            lo = m_off - a
            hi = lo + rows
            for c0 in range(0, n, _ACT_COLS):
                y = a_ref[1, 0:mb, c0 : c0 + _ACT_COLS].astype(o_ref.dtype)
                y = jnp.where(i < hi, y, jnp.zeros_like(y))
                # Rows before the entry in its first slab: earlier entries'.
                carry = cs_ref[:, c0 : c0 + _ACT_COLS]
                o_ref[os, 0:_SUB, c0 : c0 + _ACT_COLS] = jnp.where(
                    i[:_SUB] < lo, carry, y[:_SUB]
                )
                if mb > _SUB:
                    o_ref[os, _SUB:mb, c0 : c0 + _ACT_COLS] = y[_SUB:]
            # Write the slabs no later entry touches; keep the open one.
            end = m_off + rows
            is_last = j == num_ent - 1
            final = jnp.where(is_last, pl.cdiv(end, _SUB), end // _SUB) * _SUB
            size = pl.multiple_of(final - a, _SUB)

            @pl.when(size > 0)
            def _():
                out_copy(os, a, size).start()
                cnt_ref[1 + os] = size

            cs_ref[...] = o_ref[os, pl.ds(size, _SUB), :]

    branches = [functools.partial(lambda t, mb: compute(t, mb), mb=mb) for mb in buckets]

    if shared is not None:
        sh_start, sh_compute, sh_wait = _shared_expert(shared, fuse_act)
        # Ahead of the routed chunks: computed at the first chunk.
        sh_start()

        @pl.when(num_chunks == 0)
        def _():
            sh_compute()

    @pl.when(num_chunks > 0)
    def _():
        for cp in x_copies(0):
            cp.start()
        for i in range(slots - 1):

            @pl.when(i < num_chunks)
            def _():
                start_chunk(i)

        def body(t, carry):
            nxt = t + slots - 1

            @pl.when(nxt < num_chunks)
            def _():
                start_chunk(nxt)

            if shared is not None:
                # Chunks t..t+slots-1 stream in under the shared expert.
                @pl.when(t == 0)
                def _():
                    sh_compute()

            j = t // ce

            @pl.when(t % ce == 0)
            def _():
                for cp in x_copies(j):
                    cp.wait()

                @pl.when(j + 1 < num_ent)
                def _():
                    for cp in x_copies(j + 1):
                        cp.start()

            wait_chunk(t)
            _, m_off, rows, a = ent(j)
            need = m_off - a + rows
            bidx = sum((need > b).astype(jnp.int32) for b in buckets[:-1])
            if len(buckets) == 1:
                branches[0](t)
            else:
                lax.switch(bidx, branches, t)
            return carry

        lax.fori_loop(0, num_chunks, body, 0)
        wait_out(0)
        wait_out(1)

    if shared is not None:
        sh_wait()


def vmem_bytes(k1, f, n, chunk, slots, mcap, out_itemsize=2, shared=None) -> int:
    x = 3 * mcap * k1 * 2
    acc = 2 * mcap * n * 4
    o = (2 * (mcap + _SUB) + _SUB) * n * out_itemsize
    w = slots * chunk * n  # fp4 takes a byte per element in VMEM
    s = slots * _gmm.align_to(chunk // _QB, 8) * n * 4
    sh = 0
    if shared is not None:
        tp, n1s, fs = shared  # token rows, 2 * shared width, shared width
        # e4m3 weights, token rows and out, 8-row padded column scales, and
        # the f32 / bf16 temporaries of the projections.
        sh = (k1 * n1s + fs * n) + tp * (k1 + n) * 2 + 8 * (n1s + n) * 4
        sh += tp * (n1s * 6 + n * 4)
    return x + acc + o + w + s + sh


def shared_eligible(lhs_tok, sw1, ss1, sw2, ss2, k1, n) -> str | None:
    """None if (token rows, e4m3 [K1, 2Fs], f32 [1, 2Fs], e4m3 [Fs, N],
    f32 [1, N]) is a shared expert slice this kernel computes, else why not."""
    e4m3 = jnp.float8_e4m3fn
    tp, k = lhs_tok.shape
    if lhs_tok.dtype != jnp.bfloat16 or k != k1 or tp % _SUB:
        return "token rows"
    if sw1.dtype != e4m3 or sw2.dtype != e4m3:
        return "shared not e4m3"
    k1s, n1s = sw1.shape
    fs, ns = sw2.shape
    if k1s != k1 or n1s != 2 * fs or ns != n or fs % 128:
        return "shared shapes"
    if ss1.shape != (1, n1s) or ss2.shape != (1, n):
        return "shared scales"
    return None


def gmm_decode_stream_mlp(
    lhs: jax.Array,  # [Mp, K1] bf16, Mp % 16 == 0
    rhs1: jax.Array,  # [G, K1, 2F] fp4
    rhs2: jax.Array,  # [G, F, N] fp4
    group_sizes: jax.Array,  # int32[size_lhs_group]
    rhs1_scale: jax.Array,  # [G, K1/128, 1, 2F]
    rhs2_scale: jax.Array,  # [G, F/128, 1, N]
    group_offset: jax.Array,  # int32[1]
    *,
    fuse_act: str,
    out_dtype,
    chunk_rows: int,
    slots: int,
    max_group_size: int | None = None,
    interpret: bool = False,
    w_load: str = "tile",
    shared: tuple | None = None,
):
    """Routed rows' expert MLP outputs; with ``shared`` = (token rows [Tp, K1]
    bf16, e4m3 [K1, 2Fs], f32 [1, 2Fs], e4m3 [Fs, N], f32 [1, N]) also the
    shared expert slice on the token rows, as (routed, shared)."""
    size_m, k1 = lhs.shape
    g, _, n = rhs1.shape
    f = rhs2.shape[1]
    buckets, bf16_from = plan(size_m, max_group_size)
    mcap = buckets[-1]
    max_ent = g + pl.cdiv(size_m, _SUB)
    out_dtype = jnp.dtype(out_dtype or lhs.dtype)
    sh_dims = None
    if shared is not None:
        sh_dims = (shared[0].shape[0], shared[1].shape[1], shared[3].shape[0])
    need = vmem_bytes(
        k1, f, n, chunk_rows, slots, mcap, out_dtype.itemsize, shared=sh_dims
    )
    capacity = 64 << 20 if interpret else pltpu.get_tpu_info().vmem_capacity_bytes
    vmem_limit = min(
        int(capacity * 0.9),
        _gmm.align_to(need + _VMEM_MARGIN, 1 << 20),
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    kernel = functools.partial(
        _kernel,
        num_groups=g,
        chunk=chunk_rows,
        slots=slots,
        buckets=buckets,
        bf16_from=bf16_from,
        fuse_act=fuse_act,
        w_load=w_load,
        has_shared=shared is not None,
    )
    w_bytes = g * (k1 * n + f * n) // 2 + g * (k1 + f) // _QB * n * 4
    in_specs = [hbm, hbm, hbm, hbm, hbm]
    out_shape = jax.ShapeDtypeStruct((size_m, n), out_dtype)
    out_specs = hbm
    sh_scratch = []
    sh_args = ()
    flops = 2 * size_m * (k1 * n + f * n)
    name_sh = ""
    if shared is not None:
        xt, sw1, ss1, sw2, ss2 = shared
        tp = xt.shape[0]
        n1s = sw1.shape[1]
        fs = sw2.shape[0]
        vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
        in_specs += [hbm, hbm, vmem, hbm, vmem]
        out_shape = (out_shape, jax.ShapeDtypeStruct((tp, n), jnp.bfloat16))
        out_specs = (hbm, hbm)
        sh_scratch = [
            pltpu.VMEM((tp, k1), jnp.bfloat16),
            pltpu.VMEM((k1, n1s), sw1.dtype),
            pltpu.VMEM((fs, n), sw2.dtype),
            pltpu.VMEM((tp, n), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((4,)),
        ]
        sh_args = (
            xt,
            sw1,
            ss1.astype(jnp.float32),
            sw2,
            ss2.astype(jnp.float32),
        )
        w_bytes += k1 * n1s + fs * n
        flops += 2 * tp * (k1 * n1s + fs * n)
        name_sh = f"-sh_{fs}"
    call = pl.pallas_call if interpret else tight_vmem.pallas_call
    extra = {"interpret": pltpu.InterpretParams()} if interpret else {}
    return call(
        kernel,
        **extra,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=[
                pltpu.VMEM((3, k1 // chunk_rows, mcap, chunk_rows), lhs.dtype),
                pltpu.VMEM((2, mcap, n), jnp.float32),
                pltpu.VMEM((2, mcap + _SUB, n), out_dtype),
                pltpu.VMEM((_SUB, n), out_dtype),
                pltpu.VMEM((slots, chunk_rows, n), rhs1.dtype),
                pltpu.VMEM((slots, chunk_rows // _QB, n), jnp.float32),
                pltpu.SMEM((3, max_ent), jnp.int32),
                pltpu.SMEM((3,), jnp.int32),
                pltpu.SemaphoreType.DMA((slots, 2)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                *sh_scratch,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit,
            disable_bounds_checks=True,
        ),
        name=f"gmm_decode_stream-g_{g}-m_{size_m}-k_{k1}-f_{f}-c_{chunk_rows}x{slots}-{w_load}"
        f"-b_{'_'.join(map(str, buckets))}{name_sh}",
        cost_estimate=pl.CostEstimate(
            flops=flops,
            bytes_accessed=w_bytes + size_m * (k1 + n) * 2,
            transcendentals=0,
        ),
    )(
        group_sizes,
        group_offset,
        lhs,
        rhs1,
        rhs1_scale.astype(jnp.float32),
        rhs2,
        rhs2_scale.astype(jnp.float32),
        *sh_args,
    )

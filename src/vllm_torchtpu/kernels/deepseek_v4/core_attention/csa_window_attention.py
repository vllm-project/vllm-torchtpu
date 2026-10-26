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
"""TensorCore half of the union-of-window CSA attention (`csa_window_union`).

One grid step attends every live window: an in-kernel loop over the windows
(nw from meta row 0), and per window a loop over its union tiles of 1024 rows.
The tile stream is flattened across windows: while tile j computes, tile j + 1
-- or, at a window's last tile, the next window's tile 0 -- is in flight
(2-deep). Per tile the MXU sees all the window's tokens at once (M = 8 x heads
rows, token-major), each token keeping only its own ranks via its mask bits.
The online softmax starts from the SWA partials (acc, l, m); the sink joins
the denominator at the end. Rows outside every window return q (q aliases the
output).

Inputs follow the frozen interface in `csa_window_union`'s module docstring.
`version=0`: q / SWA of each window loaded at its start (synchronous).
`version=1`: q / SWA of window w + 1 prefetched under window w (double
buffered).
`version=2`: v1 + the next tile dequantized (to a bf16 VMEM tile) in the same
basic block as this tile's QK / softmax / PV (`_kernel_v2`).
`zero_tail_rows=False` is a removal arm for tests only (unwritten tail rows
then reach the MXU).
`version=3`: v1 with tiles taken in pairs, both tiles' QK / softmax / PV in one
basic block (an odd last tile runs alone).
`version=4`: v1 with a narrow tail tile: a last tile of <= 128 / 256 / 512
valid rows DMAs and attends only that many rows (static body per size).
`version=5`: v4 with each tile (and narrow tail) done in column sub-tiles of
`sub` rows (dequant, QK, mask, online softmax, bf16 PV per sub-tile).
`hbm_pin`: constrain the HBM operands to HBM (no XLA VMEM staging copies).
`version=6`: v4's narrow tail DMA + every tile in column sub-tiles of `sub`
rows under one fori_loop body (trip = ceil(valid / sub)), bf16 P per sub-tile.
`version=7`: v3 + v4's narrow body for the window's last tile when it has
<= 512 valid rows (full pairs run v3's branch unchanged).
`tails`: the narrow tail sizes of v4 / v7 (v7 with (512,) keeps one narrow
size: a smaller program).
`lm_raw`: SWA l / m DMA'd as [qn, H] rows and lane-broadcast in the kernel
(no XLA per-row broadcast in front of the call).
`pv_bf16`: P pushed as bf16 (default f32 P x bf16 KV, as SparseMLA / union-dense).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla import (
    NOPE_DIM,
    _dequant_dsv4_fp8_sub_rows,
    prepare_outputs,
    prepare_q_inputs,
    prepare_swa_inputs,
)

LANES = 128
TILE = 1024  # union rows per KV tile (the gathered buffer's period)
WQ = 8  # tokens per window
MAX_TILES = 8  # union tiles per window (U <= 8192)
TAIL_ROWS = (128, 256, 512)  # v4 narrow tail-tile bodies (static row counts)
WORD_BITS = 32
SUPER_BLOCK = WORD_BITS * LANES  # 4096 ranks per 128-word mask column
DEFAULT_VMEM_LIMIT_BYTES = 48 * 1024 * 1024  # <= 64 MiB physical (v7x)

# SMEM state slots
_KV_SLOT, _BO_SLOT, _BO_Q0, _BO_QN, _PENDING = 0, 1, 2, 4, 6


def cdiv(a, b):
    return (a + b - 1) // b


def _kernel(
    # Scalar prefetch
    meta_ref,  # i32[(1 + nw_cap + workers) * L]
    ucnt_ref,  # i32[nw_cap * L]
    # Inputs
    sinks_ref,  # VMEM f32[WQ * H, 128]
    q_hbm_ref,  # [T, H, D]
    nope_hbm_ref,  # i32[T * TILE, 128]
    rope_hbm_ref,  # i32[T * TILE // 4, 128] (viewed bf16[T * TILE // 2, 128])
    mask_hbm_ref,  # i32[T, W // 128, 128]
    swa_acc_hbm_ref,  # [T * H, D]
    swa_l_hbm_ref,  # f32[T * H, 128] (lm_raw: f32[T, H])
    swa_m_hbm_ref,  # f32[T * H, 128] (lm_raw: f32[T, H])
    # Output (aliases q)
    o_hbm_ref,
    # Scratch
    nope_x2_ref,  # i32[2, TILE, 128]
    rope_x2_ref,  # bf16[2, TILE // 2, 128]
    mask_x2_ref,  # i32[2, WQ, 1, 128]
    bq_ref,  # [NQ, WQ, H, D]
    bo_x2_ref,  # [2, WQ, H, D]
    swa_acc_ref,  # [NQ, WQ * H, D]
    swa_l_ref,  # f32[NQ, WQ * H, 128]
    swa_m_ref,  # f32[NQ, WQ * H, 128]
    sems,  # DMA[8, 2]
    l_ref,  # f32[WQ * H, 128]
    m_ref,  # f32[WQ * H, 128]
    acc_ref,  # f32[WQ * H, D]
    st_ref,  # SMEM i32[8]
    *probe_refs,  # probe only: bf16[TILE, 512], f32[WQ * H, TILE]
    sm_scale: float,
    L: int,
    version: int,
    pv_bf16: bool,
    zero_tail_rows: bool,
    probe: str = "",
    lm_raw: bool = False,
    sub: int = 256,
    mask_sb: bool = False,
    tails: tuple = TAIL_ROWS,
    tslot_ref=None,
    wq: int = WQ,
):
    WQ = wq  # noqa: N806 -- tokens per window (8, or 16 on the mixed route); closures read this
    _, num_q_heads, head_dim = q_hbm_ref.shape
    rows = WQ * num_q_heads
    q_dtype = q_hbm_ref.dtype
    q_packing = 32 // jax.dtypes.itemsize_bits(q_dtype)
    heads_per_packing = num_q_heads // q_packing
    blocks_per_super = SUPER_BLOCK // TILE
    bits_per_block = TILE // LANES
    half = TILE // 2
    rope_dim = LANES // 2
    assert rope_dim == NOPE_DIM % LANES
    rope_hbm_ref = rope_hbm_ref.bitcast(jnp.bfloat16)

    nw = meta_ref[0]

    def win(w):
        base = (1 + w) * L
        q0, qn = meta_ref[base], meta_ref[base + 1]
        U = ucnt_ref[w * L]
        return q0, qn, U, (U + TILE - 1) // TILE

    def copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def fetch_tile(q0, qn, j, slot, *, wait=False, n=TILE):
        """Union tile j of the window at q0 (gathered tile q0 + j) + its mask
        words. n < TILE (<= TILE / 2): a narrow tail -- nope rows [0, n) and
        rope bf16 rows [0, n) (ranks i < TILE / 2 live in lanes 0:64)."""
        nr = half if n == TILE else n
        nope_v = nope_x2_ref.at[slot, pl.ds(0, n)]
        rope_v = rope_x2_ref.at[slot, pl.ds(0, nr)]
        mask_v = mask_x2_ref.at[slot, pl.ds(0, qn), 0, pl.ds(0, LANES)]
        if wait:
            copy(nope_v, nope_v, sems.at[0, slot], True)
            copy(rope_v, rope_v, sems.at[1, slot], True)
            copy(mask_v, mask_v, sems.at[2, slot], True)
            return
        g = (q0 if tslot_ref is None else tslot_ref[q0]) + j
        copy(nope_hbm_ref.at[pl.ds(g * TILE, n)], nope_v, sems.at[0, slot], False)
        copy(rope_hbm_ref.at[pl.ds(g * half, nr)], rope_v, sems.at[1, slot], False)
        col = j // blocks_per_super
        if mask_sb:  # super-block-major [W / 128, T, 128] (L1 mask_sb_major=True)
            msrc = mask_hbm_ref.at[col, pl.ds(q0, qn), pl.ds(0, LANES)]
        else:  # [T, W / 128, 128]
            msrc = mask_hbm_ref.at[pl.ds(q0, qn), col, pl.ds(0, LANES)]
        copy(msrc, mask_v, sems.at[2, slot], False)

    def fetch_q_swa(q0, qn, qs, *, wait=False):
        hs, hn = q0 * num_q_heads, qn * num_q_heads
        ls, ln = (q0, qn) if lm_raw else (hs, hn)  # lm_raw: [T, H] rows
        pairs = (
            (q_hbm_ref.at[pl.ds(q0, qn)], bq_ref.at[qs, pl.ds(0, qn)]),
            (swa_acc_hbm_ref.at[pl.ds(hs, hn)], swa_acc_ref.at[qs, pl.ds(0, hn)]),
            (swa_l_hbm_ref.at[pl.ds(ls, ln)], swa_l_ref.at[qs, pl.ds(0, ln)]),
            (swa_m_hbm_ref.at[pl.ds(ls, ln)], swa_m_ref.at[qs, pl.ds(0, ln)]),
        )
        for k, (src, dst) in enumerate(pairs):
            if wait:
                copy(dst, dst, sems.at[4 + k, qs], True)
            else:
                copy(src, dst, sems.at[4 + k, qs], False)

    def send_bo(q0, qn, slot, *, wait=False):
        copy(bo_x2_ref.at[slot, pl.ds(0, qn)], o_hbm_ref.at[pl.ds(q0, qn)],
             sems.at[3, slot], wait)  # fmt: skip

    def wait_send_bo(slot):
        q0 = st_ref[_BO_Q0 + slot]
        qn = st_ref[_BO_QN + slot]

        @pl.when(q0 >= 0)
        def _():
            send_bo(q0, qn, slot, wait=True)

    def load_bq(qs):
        q_ref = bq_ref.at[qs].bitcast(jnp.uint32).reshape(WQ * heads_per_packing, head_dim)
        return pltpu.bitcast(q_ref[...], q_dtype).reshape(rows, head_dim)

    def zero_tail(slot, valid):
        """Rows at ranks >= valid of the tail tile are unwritten (any bits):
        zero them in VMEM (zero words dequantize to 0) so none can inject NaN."""
        nope_ref = nope_x2_ref.at[slot]
        row = lax.broadcasted_iota(jnp.int32, nope_ref.shape, 0)
        nope_ref[...] = jnp.where(row < valid, nope_ref[...], 0)
        rope_ref = rope_x2_ref.at[slot]
        r = rope_ref[...]
        rrow = lax.broadcasted_iota(jnp.int32, r.shape, 0)
        lane = lax.broadcasted_iota(jnp.int32, r.shape, 1)
        rank = rrow + jnp.where(lane < rope_dim, 0, half)
        rope_ref[...] = jnp.where(rank < valid, r, jnp.zeros_like(r))

    def load_tile(slot):
        if "nodeq" in probe:  # removal arm: a resident bf16 tile instead of dequant
            return probe_refs[0][...]
        # Sub-row dequant + rope placed in lanes NOPE_DIM % 128: of sub-row 3
        # (sparse_mla.load_bkv): the paired rope block holds rank i in lanes
        # 0:64 and rank i + TILE / 2 in 64:128 of row i.
        nope = _dequant_dsv4_fp8_sub_rows(nope_x2_ref[slot])
        r = rope_x2_ref[slot]
        r = jnp.concatenate([pltpu.roll(r, rope_dim, 1), r], axis=0)
        lane = lax.broadcasted_iota(jnp.int32, r.shape, 1)
        r = jnp.where(lane < rope_dim, nope[-1], r)
        return jnp.concatenate([*nope[:-1], r], axis=-1)  # [TILE, 512]

    def load_mask(slot, j):
        words = mask_x2_ref[slot][:, 0, :]  # [WQ, 128]
        base = (j % blocks_per_super) * bits_per_block
        planes = [
            jnp.bitwise_and(lax.shift_right_logical(words, base + t), 1)
            for t in range(bits_per_block)
        ]
        return jnp.concatenate(planes, axis=1) != 0  # [WQ, TILE]

    def tail_cases(valid):
        """(n, predicate) per tile body: narrow tails 128 / 256 / 512, else full."""
        out, lo = [], 0
        for n in tails:
            out.append((n, jnp.logical_and(valid > lo, valid <= n)))
            lo = n
        out.append((TILE, valid > lo))
        return out

    def fetch_any(q0, qn, j, slot, U, *, wait=False):
        if version not in (4, 5, 6, 7):
            fetch_tile(q0, qn, j, slot, wait=wait)
            return
        valid = U - j * TILE
        if version == 7:  # full tiles: one branch, narrow cases nested behind it
            @pl.when(valid > tails[-1])
            def _():
                fetch_tile(q0, qn, j, slot, wait=wait)

            @pl.when(valid <= tails[-1])
            def _():
                for n, pred in tail_cases(valid)[:-1]:
                    @pl.when(pred)
                    def _():
                        fetch_tile(q0, qn, j, slot, wait=wait, n=n)
            return
        for n, pred in tail_cases(valid):
            @pl.when(pred)
            def _():
                fetch_tile(q0, qn, j, slot, wait=wait, n=n)

    def load_tile_narrow(slot, n, valid):
        nope = _dequant_dsv4_fp8_sub_rows(nope_x2_ref[slot, pl.ds(0, n)])
        r = pltpu.roll(rope_x2_ref[slot, pl.ds(0, n)], rope_dim, 1)  # rank i -> lanes 64:
        lane = lax.broadcasted_iota(jnp.int32, r.shape, 1)
        r = jnp.where(lane < rope_dim, nope[-1], r)
        kv = jnp.concatenate([*nope[:-1], r], axis=-1)  # [n, 512]
        row = lax.broadcasted_iota(jnp.int32, kv.shape, 0)
        return jnp.where(row < valid, kv, jnp.zeros_like(kv))  # unwritten rows

    def load_mask_narrow(slot, j, n):
        words = mask_x2_ref[slot][:, 0, :]
        base = (j % blocks_per_super) * bits_per_block
        planes = [
            jnp.bitwise_and(lax.shift_right_logical(words, base + t), 1)
            for t in range(n // LANES)
        ]
        return (jnp.concatenate(planes, axis=1) if len(planes) > 1 else planes[0]) != 0

    def flash_sub(q, slot, j, valid, n):
        """v5: the n-row tile in column sub-tiles of `sub` rows: per sub-tile
        dequant -> QK ([M, sub] scores) -> mask -> online softmax -> bf16 PV,
        so each sub-tile's working set stays near the vreg file."""
        words = mask_x2_ref[slot][:, 0, :]  # [WQ, 128]
        base = (j % blocks_per_super) * bits_per_block
        sb = min(sub, n)
        for k in range(n // sb):
            r0 = k * sb
            nope = _dequant_dsv4_fp8_sub_rows(nope_x2_ref[slot, pl.ds(r0, sb)])
            r = rope_x2_ref[slot, pl.ds(r0 % half, sb)]
            if r0 < half:  # ranks < TILE / 2 sit in lanes 0:64 of the pair rows
                r = pltpu.roll(r, rope_dim, 1)
            lane = lax.broadcasted_iota(jnp.int32, r.shape, 1)
            r = jnp.where(lane < rope_dim, nope[-1], r)
            kv = jnp.concatenate([*nope[:-1], r], axis=-1)  # [sb, 512]
            row = lax.broadcasted_iota(jnp.int32, kv.shape, 0) + r0
            kv = jnp.where(row < valid, kv, jnp.zeros_like(kv))  # unwritten rows
            planes = [
                jnp.bitwise_and(lax.shift_right_logical(words, base + t), 1)
                for t in range(r0 // LANES, (r0 + sb) // LANES)
            ]
            keep = (jnp.concatenate(planes, axis=1) if len(planes) > 1 else planes[0]) != 0
            flash(q, kv, keep, bf16_p=True)

    def flash_loop(q, slot, j, valid):
        """v6: the tile's valid rows in column sub-tiles of `sub` rows under a
        fori_loop (one body for full and tail tiles; trip = ceil(valid / sub)):
        per sub-tile dequant -> QK -> mask -> online softmax -> bf16 PV."""
        words = mask_x2_ref[slot][:, 0, :]  # [WQ, 128]
        base = (j % blocks_per_super) * bits_per_block
        nsub = (jnp.minimum(valid, TILE) + sub - 1) // sub

        def body(k, _):
            r0 = pl.multiple_of(k * sub, sub)
            nope = _dequant_dsv4_fp8_sub_rows(nope_x2_ref[slot, pl.ds(r0, sub)])
            hr = pl.multiple_of(r0 % half, sub) if sub < half else 0
            r = rope_x2_ref[slot, pl.ds(hr, min(sub, half))]
            if sub > half:  # sub == TILE: both pair halves
                r = jnp.concatenate([pltpu.roll(r, rope_dim, 1), r], axis=0)
            else:  # ranks < TILE / 2 sit in lanes 0:64 of the pair rows
                r = jnp.where(r0 < half, pltpu.roll(r, rope_dim, 1), r)
            lane = lax.broadcasted_iota(jnp.int32, r.shape, 1)
            r = jnp.where(lane < rope_dim, nope[-1], r)
            kv = jnp.concatenate([*nope[:-1], r], axis=-1)  # [sub, 512]
            row = lax.broadcasted_iota(jnp.int32, kv.shape, 0) + r0
            kv = jnp.where(row < valid, kv, jnp.zeros_like(kv))  # unwritten rows
            b0 = base + r0 // LANES
            planes = [
                jnp.bitwise_and(lax.shift_right_logical(words, b0 + t), 1)
                for t in range(sub // LANES)
            ]
            keep = (jnp.concatenate(planes, axis=1) if len(planes) > 1 else planes[0]) != 0
            flash(q, kv, keep, bf16_p=True)
            return 0

        lax.fori_loop(0, nsub, body, 0, unroll=False)

    def lanes(x, n):
        return jnp.concatenate([x] * (n // LANES), axis=1) if n != LANES else x

    def rows_to_lanes(x):
        """f32 [WQ, H] (token t, head h) -> [WQ * H, 128]: row t * H + h holds
        x[t, h] in every lane (replaces the wrapper's per-row broadcast)."""
        sub = lax.broadcasted_iota(jnp.int32, (num_q_heads, num_q_heads), 0)
        lan = lax.broadcasted_iota(jnp.int32, (num_q_heads, num_q_heads), 1)
        cols = []
        for t in range(WQ):
            b = jnp.broadcast_to(x[t:t + 1, :], (num_q_heads, num_q_heads))
            cols.append(jnp.sum(jnp.where(sub == lan, b, 0.0), axis=1, keepdims=True))
        return jnp.broadcast_to(jnp.concatenate(cols, axis=0), (rows, LANES))

    def flash(q, kv, keep, bf16_p=False):
        # `probe` (timing-only removal arms; wrong results): noqk / nomask /
        # nosoftmax / nopv drop that stage.
        if "noqk" in probe:
            s = probe_refs[1][...] + kv[0:1, 0:1].astype(jnp.float32)
        else:
            s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
        s *= sm_scale
        width = kv.shape[0]
        if "nomask" not in probe:
            s3 = s.reshape(WQ, num_q_heads, width)
            s3 = jnp.where(keep[:, None, :], s3, jnp.finfo(jnp.float32).min)
            s = s3.reshape(rows, width)
        m_prev = m_ref[...]
        if "nosoftmax" in probe:
            m_curr = m_prev
            p = s
            psum = l_ref[...]
        else:
            m_curr = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
            p = jnp.exp(s - lanes(m_curr, width))
            psum = jnp.sum(p, axis=1, keepdims=True)
        m_ref[...] = m_curr
        if "nopv" in probe:
            pv = p[:, :head_dim] + q[:, :].astype(jnp.float32)
        else:
            pv = jnp.einsum("nm,md->nd", p.astype(kv.dtype) if (pv_bf16 or bf16_p) else p,
                            kv, preferred_element_type=jnp.float32)  # fmt: skip
        exp_m_diff = jnp.exp(m_prev - m_curr)
        l_ref[...] = exp_m_diff * l_ref[...] + psum
        acc_ref[...] = lanes(exp_m_diff, head_dim) * acc_ref[...] + pv

    st_ref[_KV_SLOT] = 0
    st_ref[_BO_SLOT] = 0
    st_ref[_BO_Q0] = -1
    st_ref[_BO_Q0 + 1] = -1
    st_ref[_BO_QN] = 0
    st_ref[_BO_QN + 1] = 0
    st_ref[_PENDING] = -1

    if version in (1, 3, 4, 5, 6, 7):  # window 0's q / SWA (later windows: prefetched)
        @pl.when(nw > 0)
        def _():
            q0, qn, _, _ = win(0)
            fetch_q_swa(q0, qn, 0)

    def window_body(w, _):
        q0, qn, U, nt = win(w)
        qs = (w % 2) if version in (1, 3, 4, 5, 6, 7) else 0
        if version == 0:
            fetch_q_swa(q0, qn, qs)

        # Tile 0 (v3: tiles 0, 1) unless the previous window's last tile
        # prefetched it.
        @pl.when(jnp.logical_and(nt > 0, st_ref[_PENDING] != w))
        def _():
            fetch_any(q0, qn, 0, st_ref[_KV_SLOT], U)
            if version in (3, 7):
                @pl.when(nt > 1)
                def _():
                    fetch_any(q0, qn, 1, st_ref[_KV_SLOT] + 1, U)

        if version in (1, 3, 4, 5, 6, 7):
            @pl.when(w + 1 < nw)
            def _():
                q0n, qnn, _, _ = win(w + 1)
                fetch_q_swa(q0n, qnn, 1 - qs)

        fetch_q_swa(q0, qn, qs, wait=True)
        if lm_raw:
            m_ref[...] = rows_to_lanes(swa_m_ref[qs])
            l_ref[...] = rows_to_lanes(swa_l_ref[qs])
        else:
            m_ref[...] = swa_m_ref[qs]
            l_ref[...] = swa_l_ref[qs]
        acc_ref[...] = swa_acc_ref[qs].astype(jnp.float32)

        def tile_body(j, _):
            slot = st_ref[_KV_SLOT]
            nslot = 1 - slot

            @pl.when(j + 1 < nt)
            def _():
                fetch_any(q0, qn, j + 1, nslot, U)

            # Last tile: start the next window's tile 0 (flattened stream).
            @pl.when(jnp.logical_and(j + 1 == nt, w + 1 < nw))
            def _():
                q0n, qnn, Un, ntn = win(w + 1)

                @pl.when(ntn > 0)
                def _():
                    fetch_any(q0n, qnn, 0, nslot, Un)
                    st_ref[_PENDING] = w + 1

            fetch_any(q0, qn, j, slot, U, wait=True)
            valid = U - j * TILE

            if version == 6:
                flash_loop(load_bq(qs), slot, j, valid)

            if version == 5:
                for n, pred in tail_cases(valid):
                    @pl.when(pred)
                    def _():
                        flash_sub(load_bq(qs), slot, j, valid, n)

            if version == 4:
                for n, pred in tail_cases(valid):
                    if n == TILE:
                        continue

                    @pl.when(pred)
                    def _():
                        flash(load_bq(qs), load_tile_narrow(slot, n, valid),
                              load_mask_narrow(slot, j, n))  # fmt: skip

            def full_tile():
                if zero_tail_rows:
                    @pl.when(valid < TILE)
                    def _():
                        zero_tail(slot, valid)

                flash(load_bq(qs), load_tile(slot), load_mask(slot, j))

            if version == 4:
                pl.when(valid > TAIL_ROWS[-1])(full_tile)
            elif version not in (5, 6):
                full_tile()

            st_ref[_KV_SLOT] = nslot
            return 0

        def pair_body(pi, _):
            """v3: tiles 2 pi, 2 pi + 1 in one basic block, so the second
            tile's QK can issue under the first tile's softmax / PV."""
            base = st_ref[_KV_SLOT]  # 0 or 2
            nbase = 2 - base
            j0 = 2 * pi
            j1 = j0 + 1

            @pl.when(j1 + 1 < nt)
            def _():
                fetch_tile(q0, qn, j1 + 1, nbase)

                @pl.when(j1 + 2 < nt)
                def _():
                    fetch_tile(q0, qn, j1 + 2, nbase + 1)

            @pl.when(jnp.logical_and(j1 + 1 >= nt, w + 1 < nw))
            def _():
                q0n, qnn, _, ntn = win(w + 1)

                @pl.when(ntn > 0)
                def _():
                    fetch_tile(q0n, qnn, 0, nbase)

                    @pl.when(ntn > 1)
                    def _():
                        fetch_tile(q0n, qnn, 1, nbase + 1)

                    st_ref[_PENDING] = w + 1

            fetch_tile(q0, qn, j0, base, wait=True)
            v0_ = U - j0 * TILE
            v1_ = U - j1 * TILE

            @pl.when(v0_ < TILE)
            def _():
                zero_tail(base, v0_)

            @pl.when(j1 < nt)
            def _():
                fetch_tile(q0, qn, j1, base + 1, wait=True)

                @pl.when(v1_ < TILE)
                def _():
                    zero_tail(base + 1, v1_)

                q = load_bq(qs)
                flash(q, load_tile(base), load_mask(base, j0))
                flash(q, load_tile(base + 1), load_mask(base + 1, j1))

            @pl.when(j1 >= nt)
            def _():
                flash(load_bq(qs), load_tile(base), load_mask(base, j0))

            st_ref[_KV_SLOT] = nbase
            return 0

        def pair_body_v7(pi, _):
            """v7: v3 + a narrow body for the window's last tile when it has
            <= 512 valid rows (DMA and attend only 128 / 256 / 512 rows).
            Full pairs take the v3 branch unchanged."""
            base = st_ref[_KV_SLOT]  # 0 or 2
            nbase = 2 - base
            j0 = 2 * pi
            j1 = j0 + 1

            @pl.when(j1 + 1 < nt)
            def _():
                fetch_any(q0, qn, j1 + 1, nbase, U)

                @pl.when(j1 + 2 < nt)
                def _():
                    fetch_any(q0, qn, j1 + 2, nbase + 1, U)

            @pl.when(jnp.logical_and(j1 + 1 >= nt, w + 1 < nw))
            def _():
                q0n, qnn, Un, ntn = win(w + 1)

                @pl.when(ntn > 0)
                def _():
                    fetch_any(q0n, qnn, 0, nbase, Un)

                    @pl.when(ntn > 1)
                    def _():
                        fetch_any(q0n, qnn, 1, nbase + 1, Un)

                    st_ref[_PENDING] = w + 1

            fetch_any(q0, qn, j0, base, U, wait=True)
            v0_ = U - j0 * TILE
            v1_ = U - j1 * TILE
            lim = tails[-1]
            pair = j1 < nt
            narrow1 = jnp.logical_and(pair, v1_ <= lim)  # j1 is the last tile
            narrow0 = jnp.logical_and(jnp.logical_not(pair), v0_ <= lim)  # j0 last

            @pl.when(jnp.logical_and(v0_ > lim, v0_ < TILE))
            def _():
                zero_tail(base, v0_)

            @pl.when(jnp.logical_and(pair, jnp.logical_not(narrow1)))
            def _():
                fetch_any(q0, qn, j1, base + 1, U, wait=True)

                @pl.when(v1_ < TILE)
                def _():
                    zero_tail(base + 1, v1_)

                q = load_bq(qs)
                flash(q, load_tile(base), load_mask(base, j0))
                flash(q, load_tile(base + 1), load_mask(base + 1, j1))

            @pl.when(narrow1)
            def _():
                fetch_any(q0, qn, j1, base + 1, U, wait=True)
                for n, pred in tail_cases(v1_)[:-1]:
                    @pl.when(pred)
                    def _():
                        q = load_bq(qs)
                        flash(q, load_tile(base), load_mask(base, j0))
                        flash(q, load_tile_narrow(base + 1, n, v1_),
                              load_mask_narrow(base + 1, j1, n))  # fmt: skip

            @pl.when(jnp.logical_and(jnp.logical_not(pair), jnp.logical_not(narrow0)))
            def _():
                flash(load_bq(qs), load_tile(base), load_mask(base, j0))

            for n, pred in tail_cases(v0_)[:-1]:
                @pl.when(jnp.logical_and(narrow0, pred))
                def _():
                    flash(load_bq(qs), load_tile_narrow(base, n, v0_),
                          load_mask_narrow(base, j0, n))  # fmt: skip

            st_ref[_KV_SLOT] = nbase
            return 0

        if version == 3:
            lax.fori_loop(0, (nt + 1) // 2, pair_body, 0, unroll=False)
        elif version == 7:
            lax.fori_loop(0, (nt + 1) // 2, pair_body_v7, 0, unroll=False)
        else:
            lax.fori_loop(0, nt, tile_body, 0, unroll=False)

        l_total = l_ref[...] + jnp.exp(sinks_ref[...] - m_ref[...])
        inv = lanes(pl.reciprocal(l_total, approx=True), head_dim)
        out = (acc_ref[...] * inv).astype(q_dtype)
        bo = st_ref[_BO_SLOT]
        st_ref[_BO_SLOT] = 1 - bo
        wait_send_bo(bo)  # the DMA of window w - 2
        bo_x2_ref.at[bo].bitcast(jnp.int32).reshape(WQ * heads_per_packing, head_dim)[
            ...
        ] = pltpu.bitcast(out, jnp.int32)
        st_ref[_BO_Q0 + bo] = q0
        st_ref[_BO_QN + bo] = qn
        send_bo(q0, qn, bo)
        return 0

    lax.fori_loop(0, nw, window_body, 0, unroll=False)
    for i in range(2):
        wait_send_bo(i)


def _kernel_v2(
    # Scalar prefetch
    meta_ref,  # i32[(1 + nw_cap + workers) * L]
    ucnt_ref,  # i32[nw_cap * L]
    # Inputs
    sinks_ref,  # VMEM f32[WQ * H, 128]
    q_hbm_ref,  # [T, H, D]
    nope_hbm_ref,  # i32[T * TILE, 128]
    rope_hbm_ref,  # i32[T * TILE // 4, 128]
    mask_hbm_ref,  # i32[T, W // 128, 128]
    swa_acc_hbm_ref,  # [T * H, D]
    swa_l_hbm_ref,  # f32[T * H, 128]
    swa_m_hbm_ref,  # f32[T * H, 128]
    # Output (aliases q)
    o_hbm_ref,
    # Scratch
    nope_x2_ref,  # i32[2, TILE, 128]   raw tiles, slot = g % 2
    rope_x2_ref,  # bf16[2, TILE // 2, 128]
    mask_x3_ref,  # i32[3, WQ, 1, 128]  slot = g % 3
    kvb0_ref,  # bf16[TILE, 512]  dequantized tile, even g
    kvb1_ref,  # bf16[TILE, 512]  odd g
    bq_ref,  # [2, WQ, H, D]   window parity
    bo_x2_ref,  # [2, WQ, H, D]
    swa_acc_ref,  # [2, WQ * H, D]
    swa_l_ref,  # f32[2, WQ * H, 128]
    swa_m_ref,  # f32[2, WQ * H, 128]
    sems,  # DMA[8, 3]
    l_ref,  # f32[WQ * H, 128]
    m_ref,  # f32[WQ * H, 128]
    acc_ref,  # f32[WQ * H, D]
    st_ref,  # SMEM i32[8]
    tw_ref,  # SMEM i32[GMAX]  window of flattened entry g
    tj_ref,  # SMEM i32[GMAX]  tile of entry g
    *,
    sm_scale: float,
    L: int,
    pv_bf16: bool,
):
    """v2: the flattened stream as a list of (window, tile) entries (a window
    with U = 0 is one empty entry); entry g + 1 is dequantized into a bf16 VMEM
    tile in the same basic block as entry g's QK / softmax / PV, so the VPU
    dequant can co-issue with the MXU pushes. Raw tiles run 2 entries ahead."""
    _, num_q_heads, head_dim = q_hbm_ref.shape
    rows = WQ * num_q_heads
    q_dtype = q_hbm_ref.dtype
    q_packing = 32 // jax.dtypes.itemsize_bits(q_dtype)
    heads_per_packing = num_q_heads // q_packing
    blocks_per_super = SUPER_BLOCK // TILE
    bits_per_block = TILE // LANES
    half = TILE // 2
    rope_dim = LANES // 2
    rope_hbm_ref = rope_hbm_ref.bitcast(jnp.bfloat16)
    kvb = (kvb0_ref, kvb1_ref)

    nw = meta_ref[0]

    def win(w):
        base = (1 + w) * L
        q0, qn = meta_ref[base], meta_ref[base + 1]
        U = ucnt_ref[w * L]
        return q0, qn, U, (U + TILE - 1) // TILE

    # Flattened entry list.
    def win_entries(w, G):
        _, _, _, nt = win(w)

        def put(j, G):
            tw_ref[G] = w
            tj_ref[G] = j
            return G + 1

        return lax.fori_loop(0, jnp.maximum(nt, 1), put, G)

    tw_ref[0] = 0
    tj_ref[0] = 0
    G = lax.fori_loop(0, nw, win_entries, jnp.int32(0))

    def entry(g):
        gc = jnp.clip(g, 0, jnp.maximum(G - 1, 0))
        w, j = tw_ref[gc], tj_ref[gc]
        q0, qn, U, nt = win(w)
        valid = jnp.where(g < G, U - j * TILE, 0)
        return w, j, q0, qn, nt, valid

    def copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def dma_tile(g, raw, *, wait):
        @pl.when(g < G)
        def _():
            _, j, q0, qn, nt, _ = entry(g)

            @pl.when(nt > 0)
            def _():
                ms = g % 3
                nope_v = nope_x2_ref.at[raw]
                rope_v = rope_x2_ref.at[raw]
                mask_v = mask_x3_ref.at[ms, pl.ds(0, qn), 0, pl.ds(0, LANES)]
                if wait:
                    copy(nope_v, nope_v, sems.at[0, raw], True)
                    copy(rope_v, rope_v, sems.at[1, raw], True)
                    copy(mask_v, mask_v, sems.at[2, ms], True)
                    return
                t = q0 + j
                copy(nope_hbm_ref.at[pl.ds(t * TILE, TILE)], nope_v, sems.at[0, raw], False)
                copy(rope_hbm_ref.at[pl.ds(t * half, half)], rope_v, sems.at[1, raw], False)
                copy(
                    mask_hbm_ref.at[pl.ds(q0, qn), j // blocks_per_super, pl.ds(0, LANES)],
                    mask_v,
                    sems.at[2, ms],
                    False,
                )

    def fetch_q_swa(w, *, wait=False):
        q0, qn, _, _ = win(w)
        qs = w % 2
        hs, hn = q0 * num_q_heads, qn * num_q_heads
        pairs = (
            (q_hbm_ref.at[pl.ds(q0, qn)], bq_ref.at[qs, pl.ds(0, qn)]),
            (swa_acc_hbm_ref.at[pl.ds(hs, hn)], swa_acc_ref.at[qs, pl.ds(0, hn)]),
            (swa_l_hbm_ref.at[pl.ds(hs, hn)], swa_l_ref.at[qs, pl.ds(0, hn)]),
            (swa_m_hbm_ref.at[pl.ds(hs, hn)], swa_m_ref.at[qs, pl.ds(0, hn)]),
        )
        for k, (src, dst) in enumerate(pairs):
            if wait:
                copy(dst, dst, sems.at[4 + k, qs], True)
            else:
                copy(src, dst, sems.at[4 + k, qs], False)

    def send_bo(q0, qn, slot, *, wait=False):
        copy(bo_x2_ref.at[slot, pl.ds(0, qn)], o_hbm_ref.at[pl.ds(q0, qn)],
             sems.at[3, slot], wait)  # fmt: skip

    def wait_send_bo(slot):
        q0 = st_ref[_BO_Q0 + slot]
        qn = st_ref[_BO_QN + slot]

        @pl.when(q0 >= 0)
        def _():
            send_bo(q0, qn, slot, wait=True)

    def load_bq(qs):
        q_ref = bq_ref.at[qs].bitcast(jnp.uint32).reshape(WQ * heads_per_packing, head_dim)
        return pltpu.bitcast(q_ref[...], q_dtype).reshape(rows, head_dim)

    def dequant(raw, valid):
        nope = _dequant_dsv4_fp8_sub_rows(nope_x2_ref[raw])
        r = rope_x2_ref[raw]
        r = jnp.concatenate([pltpu.roll(r, rope_dim, 1), r], axis=0)
        lane = lax.broadcasted_iota(jnp.int32, r.shape, 1)
        r = jnp.where(lane < rope_dim, nope[-1], r)
        kv = jnp.concatenate([*nope[:-1], r], axis=-1)  # [TILE, 512]
        # Ranks >= valid are unwritten (any bits) or absent: zero them so no
        # garbage row can inject NaN through P @ V.
        row = lax.broadcasted_iota(jnp.int32, kv.shape, 0)
        return jnp.where(row < valid, kv, jnp.zeros_like(kv))

    def load_mask(ms, j, valid):
        words = mask_x3_ref[ms][:, 0, :]  # [WQ, 128]
        base = (j % blocks_per_super) * bits_per_block
        planes = [
            jnp.bitwise_and(lax.shift_right_logical(words, base + t), 1)
            for t in range(bits_per_block)
        ]
        keep = jnp.concatenate(planes, axis=1) != 0  # [WQ, TILE]
        # An empty entry (U = 0) has no mask DMA: valid = 0 keeps nothing.
        rank = lax.broadcasted_iota(jnp.int32, keep.shape, 1)
        return jnp.logical_and(keep, rank < valid)

    def lanes(x, n):
        return jnp.concatenate([x] * (n // LANES), axis=1) if n != LANES else x

    def flash(q, kv, keep):
        s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
        s *= sm_scale
        s3 = s.reshape(WQ, num_q_heads, TILE)
        s3 = jnp.where(keep[:, None, :], s3, jnp.finfo(jnp.float32).min)
        s = s3.reshape(rows, TILE)
        m_prev = m_ref[...]
        m_curr = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
        m_ref[...] = m_curr
        p = jnp.exp(s - lanes(m_curr, TILE))
        pv = jnp.einsum("nm,md->nd", p.astype(kv.dtype) if pv_bf16 else p, kv,
                        preferred_element_type=jnp.float32)  # fmt: skip
        exp_m_diff = jnp.exp(m_prev - m_curr)
        l_ref[...] = exp_m_diff * l_ref[...] + jnp.sum(p, axis=1, keepdims=True)
        acc_ref[...] = lanes(exp_m_diff, head_dim) * acc_ref[...] + pv

    st_ref[_BO_SLOT] = 0
    st_ref[_BO_Q0] = -1
    st_ref[_BO_Q0 + 1] = -1
    st_ref[_BO_QN] = 0
    st_ref[_BO_QN + 1] = 0

    @pl.when(G > 0)
    def _():
        fetch_q_swa(0)

    dma_tile(0, 0, wait=False)
    dma_tile(1, 1, wait=False)
    dma_tile(0, 0, wait=True)
    kvb0_ref[...] = dequant(0, entry(0)[5])

    def step(g, par):
        w, j, q0, qn, nt, valid = entry(g)
        qs = w % 2

        @pl.when(j == 0)
        def _():
            @pl.when(w + 1 < nw)
            def _():
                fetch_q_swa(w + 1)

            fetch_q_swa(w, wait=True)
            m_ref[...] = swa_m_ref[qs]
            l_ref[...] = swa_l_ref[qs]
            acc_ref[...] = swa_acc_ref[qs].astype(jnp.float32)

        dma_tile(g + 2, par, wait=False)
        dma_tile(g + 1, 1 - par, wait=True)
        # One basic block: dequant of g + 1 (VPU) next to g's QK / PV (MXU).
        kvb[1 - par][...] = dequant(1 - par, entry(g + 1)[5])
        flash(load_bq(qs), kvb[par][...], load_mask(g % 3, j, valid))

        @pl.when(j == jnp.maximum(nt, 1) - 1)
        def _():
            l_total = l_ref[...] + jnp.exp(sinks_ref[...] - m_ref[...])
            inv = lanes(pl.reciprocal(l_total, approx=True), head_dim)
            out = (acc_ref[...] * inv).astype(q_dtype)
            bo = st_ref[_BO_SLOT]
            st_ref[_BO_SLOT] = 1 - bo
            wait_send_bo(bo)  # the DMA of two windows back
            bo_x2_ref.at[bo].bitcast(jnp.int32).reshape(WQ * heads_per_packing, head_dim)[
                ...
            ] = pltpu.bitcast(out, jnp.int32)
            st_ref[_BO_Q0 + bo] = q0
            st_ref[_BO_QN + bo] = qn
            send_bo(q0, qn, bo)

    def pair(p, _):
        step(2 * p, 0)

        @pl.when(2 * p + 1 < G)
        def _():
            step(2 * p + 1, 1)

        return 0

    lax.fori_loop(0, (G + 1) // 2, pair, 0, unroll=False)
    for i in range(2):
        wait_send_bo(i)


@functools.partial(
    jax.jit, static_argnames=("sm_scale", "nw_cap", "version", "pv_bf16", "zero_tail_rows",
                     "vmem_limit_bytes", "probe", "lm_raw", "sub", "hbm_pin", "tails",
                     "pin_lm", "wq")
)
def window_attention(
    q: jax.Array,  # [T, actual_heads, D] bf16
    nope_g: jax.Array,  # i32[T * TILE, 128]
    rope_g: jax.Array,  # i32[T * TILE // 4, 128]
    mask: jax.Array,  # i32[T, W], or super-block-major i32[W / 128, T, 128]
    meta: jax.Array,  # i32[(1 + nw_cap + workers) * L]
    ucnt: jax.Array,  # i32[nw_cap * L]
    attention_sinks: jax.Array,  # f32[actual_heads]
    swa_acc: jax.Array,  # [T, actual_heads, D]
    swa_l: jax.Array,  # f32[T, actual_heads]
    swa_m: jax.Array,  # f32[T, actual_heads]
    *,
    sm_scale: float,
    nw_cap: int,
    version: int = 0,
    pv_bf16: bool = False,
    zero_tail_rows: bool = True,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    probe: str = "",
    lm_raw: bool = False,
    sub: int = 256,
    hbm_pin: bool = False,
    tails: tuple = TAIL_ROWS,
    pin_lm: bool = True,
    tslot: jax.Array | None = None,
    wq: int = WQ,
) -> jax.Array:
    """`tslot` (i32[T], grouped mixed steps): window at q0's first gathered tile
    is tslot[q0] (default q0).

    Rows of live windows get CSA attention over their union folded with the
    SWA partials and the sink; all other rows return q."""
    WQ = wq  # noqa: N806
    assert wq == 8 or (tslot is not None and version == 7), "wq != 8: mixed route (v7, tile slots) only"
    _, actual_heads, actual_dim = q.shape
    q = prepare_q_inputs(q)
    T, H, D = q.shape
    rows = WQ * H
    L = ucnt.shape[0] // nw_cap
    assert ucnt.shape[0] == nw_cap * L and meta.shape[0] >= (1 + nw_cap) * L
    if tslot is None:
        assert nope_g.shape == (T * TILE, LANES), nope_g.shape
        assert rope_g.shape == (T * TILE // 4, LANES), rope_g.shape
    else:
        assert version != 2 and tslot.shape == (T,), (version, tslot.shape)
        assert rope_g.shape[0] * 4 == nope_g.shape[0] and nope_g.shape[0] % TILE == 0
    if mask.ndim == 3:  # super-block-major [W / 128, T, 128]
        assert mask.shape[1] == T and mask.shape[2] == LANES and version != 2, mask.shape
    else:
        assert mask.shape[0] == T and mask.shape[1] % LANES == 0
    swa_l_raw, swa_m_raw = swa_l.astype(jnp.float32), swa_m.astype(jnp.float32)
    assert not lm_raw or swa_l.shape == (T, H), (swa_l.shape, T, H)
    swa_acc, swa_l, swa_m = prepare_swa_inputs(swa_acc, swa_l, swa_m)
    swa_acc = swa_acc.reshape(T * H, D)

    def per_row(x):
        x = x[:, :H].reshape(T * H, 1)
        return jnp.broadcast_to(x, (T * H, LANES))

    sinks = jnp.pad(attention_sinks, (0, H - actual_heads),
                    constant_values=jnp.finfo(attention_sinks.dtype).min)  # fmt: skip
    sinks = jnp.broadcast_to(jnp.tile(sinks, WQ)[:, None], (rows, LANES)).astype(jnp.float32)
    nq = 2 if version in (1, 3, 4, 5, 6, 7) else 1
    ns = 4 if version in (3, 7) else 2  # raw tile slots
    assert not (lm_raw and version == 2), "lm_raw: v0 / v1 only"
    lm_shape = (nq, WQ, H) if lm_raw else (nq, rows, LANES)
    if version == 2:
        kernel = functools.partial(_kernel_v2, sm_scale=sm_scale, L=L, pv_bf16=pv_bf16)
        scratch = [
            pltpu.VMEM((2, TILE, LANES), jnp.int32),
            pltpu.VMEM((2, TILE // 2, LANES), jnp.bfloat16),
            pltpu.VMEM((3, WQ, 1, LANES), jnp.int32),
            pltpu.VMEM((TILE, D), jnp.bfloat16),
            pltpu.VMEM((TILE, D), jnp.bfloat16),
            pltpu.VMEM((2, WQ, H, D), q.dtype),
            pltpu.VMEM((2, WQ, H, D), q.dtype),
            pltpu.VMEM((2, rows, D), swa_acc.dtype),
            pltpu.VMEM((2, rows, LANES), jnp.float32),
            pltpu.VMEM((2, rows, LANES), jnp.float32),
            pltpu.SemaphoreType.DMA((8, 3)),
            pltpu.VMEM((rows, LANES), jnp.float32),
            pltpu.VMEM((rows, LANES), jnp.float32),
            pltpu.VMEM((rows, D), jnp.float32),
            pltpu.SMEM((8,), jnp.int32),
            pltpu.SMEM((nw_cap * MAX_TILES,), jnp.int32),
            pltpu.SMEM((nw_cap * MAX_TILES,), jnp.int32),
        ]
    else:
        kernel = functools.partial(_kernel, sm_scale=sm_scale, L=L, version=version,
                                   pv_bf16=pv_bf16, zero_tail_rows=zero_tail_rows,
                                   probe=probe, lm_raw=lm_raw, sub=sub,
                                   mask_sb=mask.ndim == 3, tails=tuple(tails),
                                   **({} if wq == 8 else {"wq": wq}))
        scratch = [
            pltpu.VMEM((ns, TILE, LANES), jnp.int32),
            pltpu.VMEM((ns, TILE // 2, LANES), jnp.bfloat16),
            pltpu.VMEM((ns, WQ, 1, LANES), jnp.int32),
            pltpu.VMEM((nq, WQ, H, D), q.dtype),
            pltpu.VMEM((2, WQ, H, D), q.dtype),
            pltpu.VMEM((nq, rows, D), swa_acc.dtype),
            pltpu.VMEM(lm_shape, jnp.float32),
            pltpu.VMEM(lm_shape, jnp.float32),
            pltpu.SemaphoreType.DMA((8, ns)),
            pltpu.VMEM((rows, LANES), jnp.float32),
            pltpu.VMEM((rows, LANES), jnp.float32),
            pltpu.VMEM((rows, D), jnp.float32),
            pltpu.SMEM((8,), jnp.int32),
        ]
        if probe:
            scratch += [pltpu.VMEM((TILE, D), jnp.bfloat16),
                        pltpu.VMEM((rows, TILE), jnp.float32)]
    prefetch = (meta.astype(jnp.int32), ucnt.astype(jnp.int32))
    if tslot is not None:
        prefetch += (tslot.astype(jnp.int32),)
        inner = kernel

        def kernel(meta_ref, ucnt_ref, tslot_ref, *refs):
            return inner(meta_ref, ucnt_ref, *refs, tslot_ref=tslot_ref)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def pin(x):
        # The HBM BlockSpec only types the ref inside the kernel; without the
        # constraint XLA may stage the operand in VMEM with a copy-start/done
        # pair in front of the call (pallas_hbm_pin.py).
        return pltpu.with_memory_space_constraint(x, pltpu.HBM) if hbm_pin else x
    out = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(prefetch),
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] + [hbm] * 7,
            out_specs=hbm,
            grid=(1,),
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        input_output_aliases={len(prefetch) + 1: 0},
        name=f"CsaWindowAttn-v{version}-h{H}",
    )(
        *prefetch,
        sinks,
        *map(pin, (
            q,
            nope_g,
            rope_g,
            mask if mask.ndim == 3 else mask.reshape(T, -1, LANES),
            swa_acc,
        )),
        # [T, H] f32 l / m pinned to HBM get a relayout copy (~0.67 us each at
        # T32 H16): pinned only when pin_lm.
        *map(pin if pin_lm else (lambda x: x), (
            swa_l_raw if lm_raw else per_row(swa_l),
            swa_m_raw if lm_raw else per_row(swa_m),
        )),
    )
    return prepare_outputs(out, actual_heads, actual_dim)

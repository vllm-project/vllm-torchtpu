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
"""Stateless output/LSE storage, send-layout packing, and DCP RDMA exchange."""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla.sparse.config import (
    SEND_PAD_ROWS,
    SEND_SLAB_ROW0,
    AttentionKernelConfig,
)
from vllm_torchtpu.kernels.mla.sparse.kernel_context import KernelGeometry
from vllm_torchtpu.kernels.mla.sparse.ref_registry import AttentionRefs
from vllm_torchtpu.kernels.mla.sparse.testing_faults import FaultHooks


def store_out(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    hooks: FaultHooks,
    bo_sem_idx: Any,
    batch_idx: Any,
    out: jax.Array,
) -> None:
    """Stores one token's ``[num_q_heads, >= o_dim]`` output into the bo tile."""
    if not cfg.send_enabled:
        refs.bo_x2.at[bo_sem_idx, batch_idx][...] = out[:, : geom.o_dim]
        return
    # Send layout: head slab w (L rows) to rows batch_idx*L of destination
    # w's slab, as int16 words (a bitcast: bits unchanged).
    words = pltpu.bitcast(out[:, : geom.o_dim], jnp.int16)
    if not isinstance(batch_idx, int):
        for b in range(cfg.batch_size):
            pl.when(batch_idx == b)(
                functools.partial(store_out, refs, cfg, geom, hooks, bo_sem_idx, b, out)
            )
        return
    rows = pl.ds(cfg.slab_row0 + batch_idx * geom.send_per, geom.send_per)
    assert refs.bo_x2 is not None
    for w in range(cfg.send_world):
        dest = hooks.send_dest_rank(w, cfg.send_world)
        refs.bo_x2[bo_sem_idx, dest, rows, :] = words[
            w * geom.send_per : (w + 1) * geom.send_per
        ]


def store_blse(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    bo_sem_idx: Any,
    batch_idx: Any,
    lse_col: jax.Array,
) -> None:
    """Stores one token's ``f32[num_q_heads, 1]`` log-sum-exp column."""
    assert refs.blse_x2 is not None
    if not cfg.lse_lane_dense:
        refs.blse_x2.at[bo_sem_idx, batch_idx][...] = lse_col
        return
    # Heads from sublanes to lanes, exactly: each column of the masked
    # [heads, heads] tile holds one live entry, the rest -inf.
    num_q_heads = geom.num_q_heads
    if cfg.send_enabled:
        # Send layout: head d*L+l to lane (W-1-d)*L + l of a send_lanes row
        # (the word build's stride-L roll then needs no negative stride);
        # lanes past the heads stay -inf.
        eye_shape = (num_q_heads, geom.send_lanes)
        h = lax.broadcasted_iota(jnp.int32, eye_shape, 0)
        target = (
            (cfg.send_world - 1) * geom.send_per - h + 2 * (h & (geom.send_per - 1))
        )
        eye = lax.broadcasted_iota(jnp.int32, eye_shape, 1) == target
    else:
        eye_shape = (num_q_heads, num_q_heads)
        eye = lax.broadcasted_iota(jnp.int32, eye_shape, 0) == (
            lax.broadcasted_iota(jnp.int32, eye_shape, 1)
        )
    row = jnp.max(
        jnp.where(eye, jnp.broadcast_to(lse_col, eye_shape), -jnp.inf),
        axis=0,
        keepdims=True,
    )
    tile_ref = refs.blse_x2.at[bo_sem_idx]  # [bsz, num_q_heads]
    if isinstance(batch_idx, int):
        tile_ref[pl.ds(batch_idx, 1), :] = row
    else:
        pick = lax.broadcasted_iota(jnp.int32, tile_ref.shape, 0) == batch_idx
        tile_ref[...] = jnp.where(pick, row, tile_ref[...])


def write_blse(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    bo_sem_idx: Any,
    batch_idx: Any,
    m: jax.Array,
    l_sum: jax.Array,
) -> None:
    """Computes unshifted LSE and stores it into ``refs.blse_x2``."""
    # Log-sum-exp in the *unshifted* frame, so a peer holding a disjoint slice
    # of the same token's top-k can merge with o = sum_r softmax(lse_r) * o_r
    # without knowing our max.
    if not cfg.return_lse:
        return
    lse_col = m + jnp.log(l_sum)
    if cfg.is_dcp_mode or cfg.lse_mask_empty:
        kv_len = refs.kv_lens[geom.batch_start_seq_idx + batch_idx]
        lse_col = jnp.where(kv_len > 0, lse_col, -jnp.inf)
    store_blse(refs, cfg, geom, bo_sem_idx, batch_idx, lse_col)


def build_send_lse_words(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    hooks: FaultHooks,
) -> None:
    """Packs every destination's LSE word row from ``fill`` into int16 header.

    Writes the ``[W, SEND_PAD_ROWS, o_dim]`` int16 header into either ``bo_x2``
    slot 0 (fused exchange, row ``SEND_SLAB_ROW0 - 1``) or ``send_wbuf``
    (non-fused send layout, row ``SEND_PAD_ROWS - 1``).
    """
    assert refs.lse_fill is not None
    fill_ref = refs.lse_fill.fill
    lanes = geom.send_lanes
    send_rows = geom.send_rows
    send_per = geom.send_per
    send_world = cfg.send_world
    num_tokens = geom.num_tokens
    num_q_heads = geom.num_q_heads
    o_dim = geom.o_dim

    x = jnp.concatenate(
        [fill_ref[i] for i in range(num_tokens // cfg.batch_size)], axis=0
    )  # f32 [tokens, lanes], block-reversed heads
    bits = lax.bitcast_convert_type(x, jnp.int32)
    hi = lax.shift_right_arithmetic(bits, jnp.int32(16))
    # The low half as a signed 16-bit value (the packed merge's words).
    lo = ((bits & 0xFFFF) ^ 0x8000) - 0x8000
    lane = lax.broadcasted_iota(jnp.int32, x.shape, 1)
    lo_src = pltpu.roll(lo, send_rows % lanes, 1) if send_rows % lanes else lo
    lo_src = hooks.transform_send_lse_lo(lo_src)
    src = jnp.where(
        lane < num_q_heads,
        hi,
        jnp.where((lane >= send_rows) & (lane < send_rows + num_q_heads), lo_src, 0),
    )
    # Sublane d of the result is rolled by d*L once at the end (one strided
    # roll, shifts below a vreg width); before it, lane j of sublane d holds
    # what lands at J = (j + d*L) mod lanes. Token t's source row, rolled by
    # t*L - (W-1)*L, puts head d*L+l (lane (W-1-d)*L + l) at J = t*L + l and
    # its lo half at rows + t*L + l, so each lane takes the covering token.
    shape = (send_world, lanes)
    big_j = (
        lax.broadcasted_iota(jnp.int32, shape, 1)
        + lax.broadcasted_iota(jnp.int32, shape, 0) * send_per
    ) & (lanes - 1)
    tok = lax.shift_right_logical(
        big_j & (send_rows - 1), jnp.int32(send_per.bit_length() - 1)
    )
    tok = jnp.where(big_j < 2 * send_rows, tok, -1)
    acc = jnp.zeros(shape, jnp.int32)
    for t in range(num_tokens):
        shift = (t * send_per - (send_world - 1) * send_per) % lanes
        row = src[t : t + 1, :]
        if shift:
            row = pltpu.roll(row, shift, 1)
        acc = jnp.where(tok == t, jnp.broadcast_to(row, shape), acc)
    acc = pltpu.roll(acc, 0, 1, stride=send_per, stride_axis=0)
    if lanes < o_dim:
        acc = jnp.concatenate(
            [acc, jnp.zeros((send_world, o_dim - lanes), jnp.int32)], axis=1
        )
    # int16 row 2r+1 is the high half of int32 row r: the word row goes to
    # int16 row SEND_PAD_ROWS - 1, every other head row is zero.
    tile_shape = (send_world, SEND_PAD_ROWS // 2, o_dim)
    r = lax.broadcasted_iota(jnp.int32, tile_shape, 1)
    word = jnp.broadcast_to(lax.shift_left(acc, jnp.int32(16))[:, None, :], tile_shape)
    if cfg.xchg_enabled:
        assert refs.bo_x2 is not None
        # Fused exchange: the word row is int16 row SEND_SLAB_ROW0 - 1 of each
        # slab, rows before it zero; the slab's first tile also holds partial
        # rows from SEND_SLAB_ROW0 on, so it is read back (int32 row r = int16
        # rows 2r, 2r + 1) and only rows < SEND_SLAB_ROW0 are replaced.
        head = refs.bo_x2[0, :, pl.ds(0, SEND_PAD_ROWS), :]
        old = pltpu.bitcast(head, jnp.int32)
        tile = jnp.where(
            r == SEND_SLAB_ROW0 // 2 - 1,
            word,
            jnp.where(r < SEND_SLAB_ROW0 // 2, 0, old),
        )
        refs.bo_x2[0, :, pl.ds(0, SEND_PAD_ROWS), :] = pltpu.bitcast(tile, jnp.int16)
        return
    tile = jnp.where(r == SEND_PAD_ROWS // 2 - 1, word, 0)
    send_wbuf = refs.lse_fill.send_wbuf
    assert send_wbuf is not None
    send_wbuf[...] = pltpu.bitcast(tile, jnp.int16)


def make_send_words_copy(refs: AttentionRefs) -> Any:
    """Builds the async DMA descriptor sending the packed LSE word header."""
    assert refs.lse_fill is not None
    assert refs.lse_fill.send_wbuf is not None and refs.lse is not None
    return pltpu.make_async_copy(
        refs.lse_fill.send_wbuf,
        refs.lse.at[:, pl.ds(0, SEND_PAD_ROWS), :],
        refs.lse_fill.sem,
    )


def xchg_peer(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    peer_idx: int,
) -> dict[str, Any]:
    """Returns the MESH device-id dict for peer ``peer_idx`` in this lane."""
    assert cfg.send_exchange is not None
    # Inside a (lane, axis) mesh a peer is (my lane, j); both as data.
    ids: dict[str, Any] = {cfg.send_exchange.axis: jnp.int32(peer_idx)}
    if cfg.send_exchange.lane_axis is not None:
        ids[cfg.send_exchange.lane_axis] = refs.bounds.peer_lane
    return ids


def xchg_signal(refs: AttentionRefs, cfg: AttentionKernelConfig) -> None:
    """Signals every peer's barrier semaphore once (self included) at entry.

    The corresponding wait sits after the attention body in ``xchg_round``.
    """
    bsem = pltpu.get_barrier_semaphore()
    for j in range(cfg.send_world):
        pl.semaphore_signal(
            bsem,
            1,
            device_id=xchg_peer(refs, cfg, j),
            device_id_type=pl.DeviceIdType.MESH,
        )


def xchg_round(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    hooks: FaultHooks,
) -> None:
    """Runs the fused small all-to-all RDMA exchange round on the slabs.

    Pushes slab ``d`` into rows ``[me * block + SEND_SLAB_ROW0, (me + 1) *
    block)`` of rank ``d``'s output (or into ``recv_v[me]`` in merge mode),
    copies the own slab locally, and waits for both arrivals and sends.
    """
    n = cfg.send_world
    me = refs.bounds.peer_rank
    assert refs.lse_fill is not None and refs.lse_fill.xchg_sems is not None
    send_sem, recv_sem, local_sem = (refs.lse_fill.xchg_sems.at[i] for i in range(3))
    if hooks.wait_exchange_barrier_in_round():
        pl.semaphore_wait(pltpu.get_barrier_semaphore(), n)
    expect = hooks.exchange_expect_peers(n)
    slot = hooks.exchange_dst_slot(me, n)

    def rows_of(k: Any) -> Any:
        if cfg.xchg_merge:
            # Merge mode: slot k of the receive scratch (VMEM).
            assert refs.recv_v is not None
            return refs.recv_v.at[k]
        assert refs.lse is not None
        start = k * geom.xchg_block + SEND_SLAB_ROW0
        if not isinstance(k, int):
            start = pl.multiple_of(start, 8)
        return refs.lse.at[pl.ds(start, geom.xchg_rows)]

    mine = rows_of(slot)
    for j in range(n):
        skip = hooks.exchange_should_skip_peer(j, me, n)

        @pl.when(jnp.logical_not(skip))
        def _send_peer(*, j: int = j) -> None:
            if hooks.exchange_local_only_round():
                # Timing only: the round without the peers -- slab j copied
                # into this rank's own slot j.
                pltpu.make_async_copy(refs.bo_x2.at[0, j], rows_of(j), recv_sem).start()
                return
            pltpu.make_async_remote_copy(
                src_ref=refs.bo_x2.at[0, j],
                dst_ref=mine,
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=xchg_peer(refs, cfg, j),
                device_id_type=pl.DeviceIdType.MESH,
            ).start()

    local = pltpu.make_async_copy(refs.bo_x2.at[0, me], rows_of(me), local_sem)
    local.start()
    if cfg.xchg_merge:
        assert refs.recv_v is not None
        others = refs.recv_v.at[pl.ds(0, expect)]
    else:
        assert refs.lse is not None
        others = refs.lse.at[pl.ds(0, expect * geom.xchg_rows)]
    pltpu.make_async_copy(others, others, recv_sem).wait()
    local.wait()
    if not hooks.exchange_local_only_round():
        pltpu.make_async_copy(others, others, send_sem).wait()


def xchg_merge_store(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    hooks: FaultHooks,
) -> None:
    """Merges all received slabs in VMEM and DMAs the bf16 result to HBM.

    Runs in two passes to bound live register pressure: LSE columns and
    softmax weights first, then each sender's partial rows accumulated in
    compensated high/low float32 accumulators.
    """
    n = cfg.send_world
    tl = geom.send_rows
    o_dim = geom.o_dim
    assert refs.bo_x2 is not None and refs.recv_v is not None
    assert refs.lse is not None and refs.lse_fill is not None
    # Trace-time: the lse word row carries row r's hi/lo halves in lanes r and
    # tl + r of one o_dim-wide row, and the lane->sublane select reads the
    # first 128 lanes -- both need these static bounds.
    assert tl <= 128, ("in-kernel merge needs send_rows <= 128", tl)
    assert 2 * tl <= o_dim, (
        "in-kernel merge needs 2*send_rows <= o_dim",
        tl,
        o_dim,
    )
    hdr = SEND_PAD_ROWS
    live = refs.bounds.real_tokens * geom.send_per  # real tokens x L
    r128 = lax.broadcasted_iota(jnp.int32, (tl, 128), 0)
    c128 = lax.broadcasted_iota(jnp.int32, (tl, 128), 1)
    l_cols = []
    for w in range(n):
        # int16 words to int32 lanes (sign-extended): row SEND_SLAB_ROW0 - 1
        # of the slab is the lse word row.
        head = refs.recv_v[w, pl.ds(0, hdr), :].astype(jnp.int32)
        words = head[SEND_SLAB_ROW0 - 1 : SEND_SLAB_ROW0]  # [1, o_dim]
        lo_src = pltpu.roll(words, o_dim - tl, 1)  # lane r <- lane tl + r
        bits = lax.shift_left(words, jnp.int32(16)) | (lo_src & 0xFFFF)
        l_row = lax.bitcast_convert_type(bits, jnp.float32)[:, :128]
        # Lanes to sublanes exactly: row r keeps only lane r (a masked max).
        l_cols.append(
            jnp.max(
                jnp.where(r128 == c128, jnp.broadcast_to(l_row, (tl, 128)), -jnp.inf),
                axis=1,
                keepdims=True,
            )
        )  # f32 [tl, 1]
    finite = [jnp.isfinite(c) for c in l_cols]
    shift = jnp.where(finite[0], l_cols[0], -jnp.inf)
    for w in range(1, n):
        shift = jnp.maximum(shift, jnp.where(finite[w], l_cols[w], -jnp.inf))
    shift = jnp.where(jnp.isfinite(shift), shift, 0.0)
    weights = [jnp.where(finite[w], jnp.exp(l_cols[w] - shift), 0.0) for w in range(n)]
    denom = weights[0]
    for w in range(1, n):
        denom = denom + weights[w]
    denom = jnp.where(denom > 0, denom, 1.0)
    rows_i = lax.broadcasted_iota(jnp.int32, (tl, o_dim), 0)
    acc_hi = jnp.zeros((tl, o_dim), jnp.float32)
    acc_lo = jnp.zeros((tl, o_dim), jnp.float32)
    for w in range(n):
        wt = weights[w] / denom
        w_hi = wt.astype(jnp.bfloat16).astype(jnp.float32)
        w_lo = hooks.transform_merge_w_lo(wt - w_hi)
        # bf16 -> f32 is the bits shifted up 16 (exact, NaNs kept); the unpack
        # zeroes partial rows past the real ones; NaN partials are zeroed as
        # the merge does.
        xi = refs.recv_v[w].astype(jnp.int32)  # [SEND_SLAB_ROW0 + tl, o_dim]
        part = lax.bitcast_convert_type(
            lax.shift_left(xi[SEND_SLAB_ROW0 : SEND_SLAB_ROW0 + tl], jnp.int32(16)),
            jnp.float32,
        )
        part = jnp.where(rows_i < live, part, 0.0)
        part = jnp.where(jnp.isnan(part), 0.0, part)
        acc_hi = acc_hi + part * w_hi
        acc_lo = acc_lo + part * w_lo
    out = (acc_hi + acc_lo).astype(jnp.bfloat16)
    refs.bo_x2[1, 0, pl.ds(0, tl), :] = pltpu.bitcast(out, jnp.int16)
    copy = pltpu.make_async_copy(
        refs.bo_x2.at[1, 0, pl.ds(0, tl)], refs.lse, refs.lse_fill.sem
    )
    copy.start()
    copy.wait()


def finalize_send_exchange(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    hooks: FaultHooks,
    *,
    is_last: Any | None = None,
) -> None:
    """Orchestrates LSE word packing and fused RDMA exchange or header DMA.

    For fused exchange, packs LSE words into ``bo_x2`` slot 0, runs the RDMA
    exchange round, and optionally merges slabs in VMEM. For non-fused send
    layout, packs LSE words and starts the async header DMA on the last step
    (caller waits via ``make_send_words_copy(refs).wait()`` after epilogue).
    """
    if cfg.xchg_enabled:
        build_send_lse_words(refs, cfg, geom, hooks)
        xchg_round(refs, cfg, geom, hooks)
        if cfg.xchg_merge:
            xchg_merge_store(refs, cfg, geom, hooks)
    elif cfg.send_enabled:
        step_is_last = (
            pl.program_id(0) == pl.num_programs(0) - 1 if is_last is None else is_last
        )

        # Every token's lse is in `fill_ref` now; the words overlap the last
        # partial DMA.
        @pl.when(step_is_last)
        def _send_words() -> None:
            build_send_lse_words(refs, cfg, geom, hooks)
            make_send_words_copy(refs).start()

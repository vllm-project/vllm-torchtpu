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
"""Reduce-scatter of projected attention partials that merges their
log-sum-exps on the way, over the 16 cores of a 2x2x2 slice of 2-core chips
(rotated halving-doubling by remote DMAs, as
``rotated_collectives.reduce_scatter``).

Each core holds, for every row of every head, a partial output ``v`` (bf16,
normalized by its own softmax sum) and its log-sum-exp ``l``. Two partials of
the same row combine as

    l = logaddexp(l1, l2)
    v = v1 * exp(l1 - l) + v2 * exp(l2 - l)

which is associative and commutative, so the halving-doubling can merge them
in any order and the result is the softmax merge over every core: no core
needs the other cores' log-sum-exps first, and the all-gather of them that the
weighted reduce-scatter needs goes away. A partial with ``l = -inf`` (a core
that owns no key of the row) gets weight 0 whatever its value.

Each core's partial is projected through W_UV inside the kernel, head block by
head block, into the slot the reduce-scatter sends it from, so the V-space
partial never goes to HBM. The rows of each head block are split into three
parts, each running its own halving-doubling over the chip-mate link and then
the x, y and z chip links in another axis order, so every chip link carries a
part in every phase. Each part's log-sum-exps travel beside its rows as one
[16, 128] f32 tile (up to 2,048 rows). Each combine reads two bf16 blocks,
computes in float32 and rounds to bf16, as the bf16 reduce-scatter rounds
every partial sum.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.rotated_collectives import (
    _BASE,
    _MATE,
    _PARTNER,
    _PRIORITY,
    _RS_OWN,
    _RS_TO_MATE,
    _SENDS,
    _SLOTS,
    CORES,
    LANES,
    MESH,
    ROTATIONS,
    TABLE_WIDTH,
    _wait,
)

COLLECTIVE_ID = 15
ROW_TILE = LANES
LSE_ROWS = 16
MAX_PART_ROWS = LSE_ROWS * LANES
ROW_UNIT = 32
HEAD_GROUP = 16


def _column(w):
    """[1, 128] f32 -> [128, 128] f32 whose row i holds w[i] in every lane."""
    return jnp.transpose(jnp.broadcast_to(w, (ROW_TILE, ROW_TILE)))


def head_part_rows(tokens):
    """Rows per head of each part of `tokens` rows: whole 32-row units, as
    even as they go, so a part of 4 heads fills whole 128-row tiles."""
    parts = len(ROTATIONS)
    units = tokens // ROW_UNIT
    return [ROW_UNIT * (units // parts + (j < units % parts)) for j in range(parts)]


def _chunk_supported(tokens, heads, latent, vdim):
    """Whether a chunk of `tokens` rows splits into 16 head blocks of whole
    128-row part tiles (at most 2,048 rows each), with whole lane tiles."""
    if heads % CORES or tokens % ROW_UNIT or tokens < len(ROTATIONS) * ROW_UNIT:
        return False
    hb = heads // CORES
    rows = [hb * p for p in head_part_rows(tokens)]
    return (
        all(r % ROW_TILE == 0 for r in rows)
        and max(rows) <= MAX_PART_ROWS
        and latent % LANES == 0
        and vdim % LANES == 0
    )


def head_lse_tiles(lse, hb):
    """[t, H] f32 log-sum-exps -> [16 * 3, 16, 128] f32: per head block and
    part, the values of its heads' part rows in (head, row) order, padded with
    -inf."""
    tokens, heads = lse.shape
    blocks = jnp.transpose(lse).reshape(CORES, hb, tokens)
    tiles = []
    start = 0
    for size in head_part_rows(tokens):
        part = blocks[:, :, start : start + size].reshape(CORES, hb * size)
        part = jnp.pad(
            part, ((0, 0), (0, MAX_PART_ROWS - hb * size)), constant_values=-jnp.inf
        )
        tiles.append(part.reshape(CORES, 1, LSE_ROWS, LANES))
        start += size
    return jnp.concatenate(tiles, axis=1).reshape(
        CORES * len(ROTATIONS), LSE_ROWS, LANES
    )


def _merge_rows(dst, ldst, src, lsrc, rows, H):
    """(dst, ldst) = the merge of (dst, ldst) and (src, lsrc) over `rows` rows;
    a partial of weight 0 is dropped whatever its value (it may be NaN)."""
    CC = 512 if H % 512 == 0 else LANES

    def tile(a, carry):
        l1 = ldst[pl.ds(a, 1), :]
        l2 = lsrc[pl.ds(a, 1), :]
        m = jnp.maximum(l1, l2)
        shift = jnp.where(m == -jnp.inf, 0.0, m)
        e1 = jnp.exp(l1 - shift)
        e2 = jnp.exp(l2 - shift)
        s = e1 + e2
        live = s > 0
        safe = jnp.where(live, s, 1.0)
        ldst[pl.ds(a, 1), :] = jnp.where(live, shift + jnp.log(safe), -jnp.inf)
        w1 = _column(jnp.where(live, e1 / safe, 0.0))
        w2 = _column(jnp.where(live, e2 / safe, 0.0))
        if CC != ROW_TILE:
            w1 = jnp.tile(w1, (1, CC // ROW_TILE))
            w2 = jnp.tile(w2, (1, CC // ROW_TILE))
        rs = pl.ds(pl.multiple_of(a * ROW_TILE, ROW_TILE), ROW_TILE)

        def cols(b, carry2):
            cs = pl.ds(pl.multiple_of(b * CC, CC), CC)
            v1 = dst[rs, cs].astype(jnp.float32)
            v2 = src[rs, cs].astype(jnp.float32)
            dst[rs, cs] = (
                jnp.where(w1 > 0, v1 * w1, 0.0) + jnp.where(w2 > 0, v2 * w2, 0.0)
            ).astype(jnp.bfloat16)
            return carry2

        return lax.fori_loop(0, H // CC, cols, carry)

    lax.fori_loop(0, rows // ROW_TILE, tile, 0)


CHUNK_TOKENS = 256
VMEM_CAP = 63 << 20
# Chunks in flight: chunk c is projected (and its phase 2 sent) in loop step
# c, takes phase 1 in step c + 1 and phase 0 in step c + 2, so its slot
# buffers, receive semaphores and log-sum-exp tiles rotate over 3 sets.
DEPTH = 3
# o group buffers: group n's copies start 2 groups before its projection.
OBUFS = 3


def _chunks_need(chunk_tokens, heads, latent, vdim, w_bytes):
    """Scoped VMEM bytes of `lse_matmul_reduce_scatter_chunks`."""
    nsub = len(ROTATIONS)
    per_head = head_part_rows(chunk_tokens)
    RS = heads // CORES * max(per_head)
    return (
        (DEPTH * nsub * _SLOTS + 2 * 8 * nsub) * RS * vdim * 2
        + DEPTH * (nsub * _SLOTS + CORES * nsub) * LSE_ROWS * LANES * 4
        + heads * latent * vdim * w_bytes
        + heads * vdim * 4
        + OBUFS * max(per_head) * HEAD_GROUP * latent * 2
    )


def chunks_supports(tokens, heads, latent, vdim, w_bytes=1, chunk_tokens=CHUNK_TOKENS):
    """Shapes `lse_matmul_reduce_scatter_chunks` is written for: whole chunks
    of whole head-block part tiles, in 16-head groups of whole head blocks,
    with every buffer in scoped VMEM."""
    hb = heads // CORES
    if not (
        _chunk_supported(chunk_tokens, heads, latent, vdim)
        and heads % HEAD_GROUP == 0
        and (heads // HEAD_GROUP) % 2 == 0
        and HEAD_GROUP % hb == 0
        and hb % 2 == 0
        and _chunks_need(chunk_tokens, heads, latent, vdim, w_bytes) + (4 << 20)
        <= VMEM_CAP
    ):
        return False
    return tokens >= chunk_tokens and tokens % chunk_tokens == 0


def _head_pair(obuf, b, g, rows, latent, interpret):
    """Heads g and g + 1 (g even) of the o group in obuf[b], [rows, latent]
    f32 each. The bf16 copy packs heads g and g + 1 of a token into one 32-bit
    word, so one strided load of words reads both, and their halves are the
    two heads' values exactly."""
    RH = obuf.shape[2]
    low, high = [], []
    for t in range(latent // LANES):
        if interpret is None:
            words = obuf.at[b, t].reshape(RH * HEAD_GROUP, LANES).bitcast(jnp.uint32)
            w32 = words[pl.ds(g // 2, rows, stride=HEAD_GROUP // 2), :]
        else:
            # Interpret mode cannot reshape a ref: the same words by
            # pltpu.bitcast's semantics on the loaded tile.
            tile = obuf[b, t, pl.ds(0, rows)].reshape(rows * HEAD_GROUP, LANES)
            w32 = pltpu.bitcast(tile, jnp.uint32)[g // 2 :: HEAD_GROUP // 2]
        low.append(lax.bitcast_convert_type(w32 << 16, jnp.float32))
        high.append(lax.bitcast_convert_type(w32 & jnp.uint32(0xFFFF0000), jnp.float32))
    return jnp.concatenate(low, axis=1), jnp.concatenate(high, axis=1)


def lse_matmul_reduce_scatter_chunks(
    o_parts,
    w,
    w_scale,
    lse,
    rank,
    table,
    *,
    axis,
    other_ids=None,
    chunk_tokens=CHUNK_TOKENS,
    interpret=None,
    name=None,
):
    """The softmax merge over the cores of this core's V-space partials, as
    a reduce-scatter that merges log-sum-exps: o_parts [span, H, C] bf16 (the
    attention calls' latent partials of every head, in row order; the last
    may repeat to fill a fixed count), w [H, C, V] (W_UV, fp8 or bf16),
    w_scale [H, V] f32 or None, lse [t, H] f32 for rows [0, t). Returns
    [H / 16, t, V] bf16: the merge of head block `rank`, heads-major. rank
    int32 [1] (this core's mesh index), table int32 [CORES * TABLE_WIDTH]
    (``rotated_collectives.peer_table`` "reduce_scatter"). other_ids: this
    core's index on each other mesh axis, if the mesh has any.

    The rows run as t / chunk_tokens chunks. Part by part, each group of 16
    heads of a chunk is copied once and projected (o_h @ W_UV_h on the MXU,
    f32 accumulation, times the scale, rounded once to bf16) into the slot its
    head block takes in the reduce-scatter; a block for the chip-mate is sent
    while the next group is projected. The attention outputs come as separate
    operands, since one array of all of them is a concatenation XLA copies
    before the call. One barrier and one W_UV load serve every chunk; the W_UV
    load and chunk 0's first o copies start before the barrier, whose wait
    comes just before chunk 0's first mate send. Each o group's copies start 2
    groups before its projection (3 o buffers).

    Loop step i merges chunk i - 1's phase 2 (sending its phase 1), merges
    chunk i - 2's phase 1 (sending its phase 0), projects chunk i (sending its
    mate blocks and phase 2), then merges and writes out chunk i - 2's phase
    0: 3 chunks are in flight, and every chip-link wait is on a copy issued a
    step earlier or before a whole projection, so the links run while the
    TensorCore projects and merges. Chunk buffers, their semaphores and the
    log-sum-exp tiles rotate over 3 sets by c mod 3; a core writes chunk c
    into a peer's buffers only after the peer reports chunk c - 3 done with
    them. The projected blocks keep 1 set: chunk c's kept blocks are merged
    before chunk c + 1 is projected, and chunk c + 1's projection first waits
    for chunk c's mate sends."""
    nparts = len(o_parts)
    if not nparts or len({x.shape for x in o_parts}) != 1:
        raise ValueError(
            "lse_matmul_reduce_scatter_chunks takes equal o parts, got "
            f"{[x.shape for x in o_parts]}"
        )
    if o_parts[0].dtype != jnp.bfloat16:
        raise ValueError(
            "lse_matmul_reduce_scatter_chunks reads 2 bf16 heads per 32-bit "
            f"word and takes bf16 o parts, got {o_parts[0].dtype}"
        )
    span, H, C = o_parts[0].shape
    tokens = lse.shape[0]
    V = w.shape[2]
    w_bytes = jnp.dtype(w.dtype).itemsize
    if (
        tokens > nparts * span
        or span % chunk_tokens
        or not chunks_supports(tokens, H, C, V, w_bytes, chunk_tokens)
    ):
        raise ValueError(
            "lse_matmul_reduce_scatter_chunks does not take o parts "
            f"[{span}, {H}, {C}], "
            f"{tokens} rows in chunks of {chunk_tokens}, V {V}, W {w.dtype}"
        )
    per_part = span // chunk_tokens
    chunks = tokens // chunk_tokens
    hb = H // CORES
    nsub = len(ROTATIONS)
    per_head = head_part_rows(chunk_tokens)
    starts = [sum(per_head[:j]) for j in range(nsub)]
    sizes = [hb * p for p in per_head]
    RS = max(sizes)
    RH = max(per_head)
    scaled = w_scale is not None
    groups = [(j, q) for j in range(nsub) for q in range(H // HEAD_GROUP)]
    # Group n of every chunk uses o buffer n mod OBUFS, so the rotation runs
    # on across chunks.
    assert len(groups) % OBUFS == 0 and len(groups) >= 2
    ltiles = CORES * nsub
    SENT, KEPT = 0, 1

    def peer(p):
        return {**(other_ids or {}), axis: p}

    def kernel(rank_ref, t_ref, *refs):
        o_refs = refs[:nparts]
        (
            w_ref,
            s_ref,
            l_ref,
            out_ref,
            buf,
            lbuf,
            pair,
            lown,
            wbuf,
            sbuf,
            obuf,
            wsem,
            osem_in,
            lsem,
            send,
            recv,
            msend,
            osem,
            free,
        ) = refs[nparts:]
        base = rank_ref[0] * TABLE_WIDTH

        def at(column):
            return t_ref[base + column]

        mate = at(_MATE)
        peers = [mate] + [at(_PARTNER + k) for k in range(3)]
        barrier = pltpu.get_barrier_semaphore()

        weights = [pltpu.make_async_copy(w_ref, wbuf, wsem.at[0])]
        if scaled:
            weights.append(pltpu.make_async_copy(s_ref, sbuf, wsem.at[1]))
        for c in weights:
            c.start()

        def slot(b, j, s):
            return buf.at[(b * nsub + j) * _SLOTS + s, pl.ds(0, sizes[j])]

        def lslot(b, j, s):
            return lbuf.at[(b * nsub + j) * _SLOTS + s]

        def rsem(b, j, s):
            return recv.at[(b * nsub + j) * _SLOTS + s]

        def o_group(o_ref, row, n):
            j, q = groups[n]
            return [
                pltpu.make_async_copy(
                    o_ref.at[
                        pl.ds(row + starts[j], per_head[j]),
                        pl.ds(q * HEAD_GROUP, HEAD_GROUP),
                        pl.ds(t * LANES, LANES),
                    ],
                    obuf.at[n % OBUFS, t, pl.ds(0, per_head[j])],
                    osem_in.at[n % OBUFS],
                )
                for t in range(C // LANES)
            ]

        def o_start(c, n):
            c = jnp.int32(c)
            row = lax.rem(c, per_part) * chunk_tokens
            for a, o_ref in enumerate(o_refs):

                @pl.when(c // per_part == a)
                def _(*, o_ref=o_ref):
                    for d in o_group(o_ref, row, n):
                        d.start()

        def o_wait(n):
            for d in o_group(o_refs[0], 0, n):
                d.wait()

        def own_lses(c, b):
            return pltpu.make_async_copy(
                l_ref.at[pl.ds(c * ltiles, ltiles)],
                lown.at[pl.ds(b * ltiles, ltiles)],
                lsem.at[b],
            )

        def place(block, j):
            """(slot i, SENT or KEPT) of head block `block` in part j."""
            i = jnp.int32(0)
            side = jnp.int32(KEPT)
            for k in range(8):
                to_mate = at(_RS_TO_MATE + j * 8 + k) == block
                kept = at(_RS_OWN + j * 8 + k) == block
                i = jnp.where(to_mate | kept, k, i)
                side = jnp.where(to_mate, SENT, side)
            return i, side

        def mate_sends_done(b):
            """Wait for the mate sends (blocks and log-sum-exps) of the chunk
            in set b."""
            for j in range(nsub):
                _wait(
                    buf.at[pl.ds((b * nsub + j) * _SLOTS, 8), pl.ds(0, sizes[j])],
                    msend.at[b],
                )
            _wait(lbuf.at[pl.ds(b * nsub * _SLOTS, 8 * nsub)], msend.at[b])

        def project(c, n):
            b = lax.rem(c, DEPTH)
            j, q = groups[n]
            rows = per_head[j]
            if n == 0:
                own_lses(c, b).start()

                # The projected blocks keep 1 set: chunk c - 1's mate sends
                # read them until they complete (its kept blocks are merged
                # by now).
                @pl.when(c >= 1)
                def _():
                    mate_sends_done(lax.rem(c + DEPTH - 1, DEPTH))

            # Group n + 2 goes into the buffer group n - 1 was projected from.
            if n + 2 < len(groups):
                o_start(c, n + 2)
            else:

                @pl.when(c + 1 < chunks)
                def _():
                    o_start(c + 1, n + 2 - len(groups))

            o_wait(n)
            for blk in range(HEAD_GROUP // hb):
                block = q * (HEAD_GROUP // hb) + blk
                i, side = place(block, j)
                dst = pair.at[j * 8 + i, side]
                for h in range(0, hb, 2):
                    g = blk * hb + h
                    for k, x in enumerate(
                        _head_pair(obuf, n % OBUFS, g, rows, C, interpret)
                    ):
                        head = q * HEAD_GROUP + g + k
                        y = lax.dot_general(
                            x.astype(jnp.bfloat16),
                            wbuf[head].astype(jnp.bfloat16),
                            (((1,), (0,)), ((), ())),
                            preferred_element_type=jnp.float32,
                        )
                        if scaled:
                            y = y * sbuf[pl.ds(head, 1), :]
                        dst[pl.ds((h + k) * rows, rows), :] = y.astype(jnp.bfloat16)

                if n == 0 and blk == 0:
                    # The entry barrier: every peer is in this call before
                    # the first copy or signal to it.
                    @pl.when(c == 0)
                    def _():
                        pl.semaphore_wait(barrier, len(peers))

                    @pl.when(c >= DEPTH)
                    def _():
                        pl.semaphore_wait(free.at[b], len(peers))

                @pl.when(side == SENT)
                def _(*, i=i, block=block):
                    rv = rsem(b, j, i)
                    for src, dst_ in (
                        (pair.at[j * 8 + i, SENT, pl.ds(0, sizes[j])], slot(b, j, i)),
                        (l_ref.at[(c * CORES + block) * nsub + j], lslot(b, j, i)),
                    ):
                        pltpu.make_async_remote_copy(
                            src,
                            dst_,
                            msend.at[b],
                            rv,
                            device_id=peer(mate),
                            device_id_type=MESH,
                        ).start()

        def arrived(b, j, s):
            _wait(slot(b, j, s), rsem(b, j, s))
            _wait(lslot(b, j, s), rsem(b, j, s))

        def complete(c, j, i):
            """Slot i of part j = the merge of the mate's partial and this
            core's own block."""
            b = lax.rem(c, DEPTH)
            arrived(b, j, i)
            own = lown.at[b * ltiles + at(_RS_OWN + j * 8 + i) * nsub + j]
            _merge_rows(
                slot(b, j, i),
                lslot(b, j, i),
                pair.at[j * 8 + i, KEPT, pl.ds(0, sizes[j])],
                own,
                sizes[j],
                V,
            )

        def send_slot(c, j, k, i):
            """Send slot i of part j, chip phase k, to its partner."""
            b = lax.rem(c, DEPTH)
            dst = _BASE[k] + i - 2**k
            partner = at(_PARTNER + j * 3 + k)
            for src, dst_ in (
                (slot(b, j, i), slot(b, j, dst)),
                (lslot(b, j, i), lslot(b, j, dst)),
            ):
                pltpu.make_async_remote_copy(
                    src,
                    dst_,
                    send.at[b],
                    rsem(b, j, dst),
                    device_id=peer(partner),
                    device_id_type=MESH,
                ).start()

        def send_phase2(c, j):
            if j == 0:
                own_lses(c, lax.rem(c, DEPTH)).wait()
            for i in _SENDS[2]:
                complete(c, j, i)
                send_slot(c, j, 2, i)

        def merge_phase(c, j, k):
            """Phase k's merges of part j; for k > 0 each slot that phase
            k - 1 sends goes out as soon as it and the slots sent before it
            are merged (the send order stays _SENDS[k - 1])."""
            b = lax.rem(c, DEPTH)
            pending = list(_SENDS[k - 1]) if k else []
            merged = set()
            for i in _PRIORITY[k]:
                if k == 2:
                    complete(c, j, i)
                s = _BASE[k] + i
                arrived(b, j, s)
                _merge_rows(
                    slot(b, j, i),
                    lslot(b, j, i),
                    slot(b, j, s),
                    lslot(b, j, s),
                    sizes[j],
                    V,
                )
                merged.add(i)
                while pending and pending[0] in merged:
                    send_slot(c, j, k - 1, pending.pop(0))
            assert not pending

        def finish(c):
            b = lax.rem(c, DEPTH)
            outs = [
                pltpu.make_async_copy(
                    buf.at[
                        b * nsub * _SLOTS + j * _SLOTS,
                        pl.ds(h * per_head[j], per_head[j]),
                    ],
                    out_ref.at[h, pl.ds(c * chunk_tokens + starts[j], per_head[j])],
                    osem,
                )
                for j in range(nsub)
                for h in range(hb)
            ]
            for d in outs:
                d.start()
            for j in range(nsub):
                first = (b * nsub + j) * _SLOTS
                _wait(buf.at[pl.ds(first, 7), pl.ds(0, sizes[j])], send.at[b])
            _wait(lbuf.at[pl.ds(b * nsub * _SLOTS, 7 * nsub)], send.at[b])

            # The next chunk's projection waited for these mate sends.
            @pl.when(c + 1 >= chunks)
            def _():
                mate_sends_done(b)

            for d in outs:
                d.wait()

            @pl.when(c + DEPTH < chunks)
            def _():
                for p in peers:
                    pl.semaphore_signal(
                        free.at[b], 1, device_id=peer(p), device_id_type=MESH
                    )

        def proj_steps(c):
            steps = []
            for n, (j, _) in enumerate(groups):

                def step(n=n, j=j):
                    project(c, n)
                    if n + 1 == len(groups) or groups[n + 1][0] != j:
                        send_phase2(c, j)

                steps.append(step)
            return steps

        # The W_UV load and chunk 0's first 2 o groups need no peer: they
        # start before the barrier, and its wait is in chunk 0's projection,
        # just before the first mate send.
        o_start(0, 0)
        o_start(0, 1)
        for p in peers:
            pl.semaphore_signal(barrier, 1, device_id=peer(p), device_id_type=MESH)
        for d in weights:
            d.wait()

        # Every core runs the same program, and every wait in it follows, in
        # program order, the send or signal it waits for: step i waits for
        # chunk i - 1's phase 2 and chunk i - 2's phase 1, sent in step
        # i - 1; chunk i's mate blocks, sent earlier in the same part's
        # projection; chunk i - 2's phase 0, sent before the projection; and
        # the buffer-free signal of chunk i - 3, sent at the end of step
        # i - 1. So no core waits on a peer that is itself waiting for this
        # core. A peer signals chunk c's set free only after it has merged
        # chunk c, which needs this core's chunk c sends, which follow this
        # core's wait for chunk c - 3's signals: the free counts never run
        # ahead.
        def body(i, carry):
            for j in range(nsub):
                pl.when((i >= 1) & (i <= chunks))(lambda j=j: merge_phase(i - 1, j, 2))
            for j in range(nsub):
                pl.when(i >= 2)(lambda j=j: merge_phase(i - 2, j, 1))
            for step in proj_steps(i):
                pl.when(i < chunks)(step)
            for j in range(nsub):
                pl.when(i >= 2)(lambda j=j: merge_phase(i - 2, j, 0))
            pl.when(i >= 2)(lambda: finish(i - 2))
            return carry

        lax.fori_loop(0, chunks + DEPTH - 1, body, 0)

    need = _chunks_need(chunk_tokens, H, C, V, w_bytes)
    kw = {"interpret": interpret} if interpret is not None else {}
    scale = (w_scale if scaled else jnp.zeros((H, V), jnp.float32)).astype(jnp.float32)
    lses = jax.vmap(lambda x: head_lse_tiles(x, hb))(
        lse.astype(jnp.float32).reshape(chunks, chunk_tokens, H)
    ).reshape(chunks * ltiles, LSE_ROWS, LANES)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((hb, tokens, V), jnp.bfloat16),
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)] * 2
        + [pl.BlockSpec(memory_space=pl.ANY)] * (nparts + 3),
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        scratch_shapes=[
            pltpu.VMEM((DEPTH * nsub * _SLOTS, RS, V), jnp.bfloat16),
            pltpu.VMEM((DEPTH * nsub * _SLOTS, LSE_ROWS, LANES), jnp.float32),
            pltpu.VMEM((8 * nsub, 2, RS, V), jnp.bfloat16),
            pltpu.VMEM((DEPTH * ltiles, LSE_ROWS, LANES), jnp.float32),
            pltpu.VMEM((H, C, V), w.dtype),
            pltpu.VMEM((H, V), jnp.float32),
            pltpu.VMEM((OBUFS, C // LANES, RH, HEAD_GROUP, LANES), o_parts[0].dtype),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((OBUFS,)),
            pltpu.SemaphoreType.DMA((DEPTH,)),
            pltpu.SemaphoreType.DMA((DEPTH,)),
            pltpu.SemaphoreType.DMA((DEPTH * nsub * _SLOTS,)),
            pltpu.SemaphoreType.DMA((DEPTH,)),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.REGULAR((DEPTH,)),
        ],
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID,
            vmem_limit_bytes=min(max(32 << 20, need + (8 << 20)), VMEM_CAP),
        ),
        name=name
        or f"lse_matmul_reduce_scatter_chunks_t{tokens}_c{chunk_tokens}_h{H}_c{C}_v{V}",
        **kw,
    )(
        rank.reshape(1).astype(jnp.int32),
        table,
        *o_parts,
        w,
        scale,
        lses,
    )

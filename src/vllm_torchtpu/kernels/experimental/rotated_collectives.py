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
"""Reduce-scatter and all-gather of a bf16 [rows, hidden] block over the 16
cores of a 2x2x2 slice of 2-core chips, as rotated halving-doubling by remote
DMAs.

Today's XLA collectives over the 16 cores run one ring in mesh order, which
keeps one chip link per direction busy. Halving-doubling crosses the chip-mate
link and the x, y and z chip links one after another. The rotated form splits
the columns into three parts and runs three halving-doublings at once, each
taking the chip axes in another order (x y z, y z x, z x y), so every chip link
carries a part in every phase.

    reduce_scatter  The chip-mate phase first (each core sends its mate the 8
                    row blocks the mate's chip half owns and adds the 8 it
                    receives, each block's first chip phase sent as soon as its
                    sum is done), then three chip phases, each sending half of
                    the blocks still held to the axis neighbour (same core
                    index) and adding the half it receives. Each add reads two
                    bf16 blocks, adds in float32 and rounds to bf16, as XLA's
                    bf16 reduce-scatter rounds every partial sum.
    all_gather      Three chip phases, each sending every block held so far to
                    the axis neighbour, each block forwarded to the chip-mate
                    as it lands. No arithmetic: the output is bitwise the
                    tiled all-gather's.

Row block m of the [16 r, hidden] block is the sum (reduce-scatter) or the
shard (all-gather) of the core at mesh index m, as ``lax.psum_scatter`` and
``lax.all_gather`` with ``tiled=True`` over the mesh axis leave them. The mesh
may list the cores in any order: ``peer_table`` reads each core's chip
coordinates and core index and writes, per mesh index, the peers and the row
blocks of every phase.

Under the torch bridge the core's own index arrives as data (an int32 [1] in
SMEM) and peers take the dict form keyed by the axis name; every wait goes on a
local descriptor of the same bytes (small_collectives/common.py).
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CORES = 16
LANES = 128
# Barrier semaphore group; the other collective kernels use 0, 5, 7, 9, 11 and
# 13. Both kernels here signal the same four peers, so they share one group.
COLLECTIVE_ID = 15
ROTATIONS = (("x", "y", "z"), ("y", "z", "x"), ("z", "x", "y"))
AXES = ("x", "y", "z")
TABLE_WIDTH = 64
MESH = pl.DeviceIdType.MESH

# Reduce-scatter scratch per part: 8 slots for the chip-mate's partial sums
# (they become the sums), then 4, 2 and 1 slots for the blocks received in the
# three chip phases (phase k, the part's k-th gather axis, receives 2^k).
_SLOTS = 15
_BASE = {2: 8, 1: 12, 0: 14}
# Within chip phase k: the received slots in the order the next phase needs
# them, and the sends in the order that fills them.
_PRIORITY = {2: (2, 3, 1, 0), 1: (1, 0), 0: (0,)}
_SENDS = {k: tuple(s + 2**k for s in _PRIORITY[k]) for k in _PRIORITY}
_MATE_ORDER = _SENDS[2] + _PRIORITY[2]

# Table columns, per part j and chip phase k.
_PARTNER = 0
_MATE = 3 * len(ROTATIONS)
_RS_TO_MATE = _MATE + 1
_RS_OWN = _RS_TO_MATE + 8 * len(ROTATIONS)
_AG_SEND = _MATE + 1
_AG_RECV = _AG_SEND + 7 * len(ROTATIONS)


class Layout:
    """The cores of a 2x2x2 slice of 2-core chips by mesh index: each core's
    peer on every chip axis (same core index) and its chip-mate."""

    def __init__(self, places):
        """``places``: (chip coords (x, y, z), core on chip) per mesh index."""
        self.places = [(tuple(int(c) for c in coords[:3]), int(core)) for coords, core in places]
        self.index = {p: m for m, p in enumerate(self.places)}
        chips = {coords for coords, _ in self.places}
        if (
            len(self.places) != CORES
            or len(self.index) != CORES
            or chips != {(x, y, z) for x in range(2) for y in range(2) for z in range(2)}
        ):
            raise ValueError(
                f"the cores {self.places} are not a 2x2x2 slice of 2-core chips"
            )

    def peer(self, m, axis):
        coords, core = self.places[m]
        if axis == "c":
            return self.index[(coords, 1 - core)]
        moved = list(coords)
        moved[AXES.index(axis)] ^= 1
        return self.index[(tuple(moved), core)]

    def flipped(self, m, axes):
        for axis in axes:
            m = self.peer(m, axis)
        return m

    def reduce_scatter_row(self, m):
        """Partner of each (part, chip phase), the mate, then per (part, slot i)
        the row block it sends the mate and its own row block that slot i sums.
        Slot i of a part holds the block of the core whose chip differs from
        the holder's on the axes of i's set bits (bit t: the part's t-th gather
        axis); in chip phase k the holder sends slot i (bit k set) to its
        axis-k partner, whose slot i - 2^k names the same block."""
        mate = self.peer(m, "c")
        partners, to_mate, own = [], [], []
        for order in ROTATIONS:
            partners += [self.peer(m, order[k]) for k in range(3)]
            to_mate += [self.flipped(mate, _bits(order, i)) for i in range(8)]
            own += [self.flipped(m, _bits(order, i)) for i in range(8)]
        return partners + [mate] + to_mate + own

    def all_gather_row(self, m):
        """Partner of each (part, chip phase), the mate, then per (part, chip
        phase k, i < 2^k) the block it sends and the block it receives. Block i
        of phase k is the core whose chip differs from the holder's on the axes
        of i's set bits; the partner receives the list in the same order."""
        partners, sends, recvs = [], [], []
        for order in ROTATIONS:
            for k in range(3):
                partner = self.peer(m, order[k])
                partners.append(partner)
                for i in range(2**k):
                    bits = [order[t] for t in range(k) if i >> t & 1]
                    sends.append(self.flipped(m, bits))
                    recvs.append(self.flipped(partner, bits))
        return partners + [self.peer(m, "c")] + sends + recvs


def _bits(order, i):
    return [order[t] for t in range(3) if i >> t & 1]


def peer_table(places, kind):
    """int32 [CORES * TABLE_WIDTH]: row m is mesh index m's table for `kind`
    ("reduce_scatter" or "all_gather")."""
    layout = Layout(places)
    row_of = {
        "reduce_scatter": layout.reduce_scatter_row,
        "all_gather": layout.all_gather_row,
    }[kind]
    rows = []
    for m in range(CORES):
        row = row_of(m)
        rows.append(row + [0] * (TABLE_WIDTH - len(row)))
    return np.asarray(rows, dtype=np.int32).reshape(-1)


def part_widths(hidden, parts=len(ROTATIONS)):
    """Column widths of the parts: whole 128-lane tiles, as even as they go."""
    if hidden % LANES:
        raise ValueError(f"hidden {hidden} is not a multiple of {LANES} lanes")
    tiles = hidden // LANES
    return [LANES * (tiles // parts + (j < tiles % parts)) for j in range(parts)]


def reduce_scatter_supports(rows, hidden):
    """Shapes the reduce-scatter is written for: 16-row blocks of whole tiles,
    three equal parts."""
    return rows % (CORES * 16) == 0 and hidden % (len(ROTATIONS) * LANES) == 0


def all_gather_supports(rows, hidden, dtype=jnp.bfloat16):
    """Whether all_gather takes [rows, hidden] of `dtype`: whole packed row
    tiles (16 bf16 rows, 32 fp8 rows) and at least one lane tile per part."""
    tile_rows = 8 * 4 // jnp.dtype(dtype).itemsize
    return (
        rows % max(tile_rows, 16) == 0
        and hidden % LANES == 0
        and hidden >= len(ROTATIONS) * LANES
    )


def _peer(p, axis, dict_ids):
    return {axis: p} if dict_ids else (p,)


def _wait(ref, sem):
    pltpu.make_async_copy(ref, ref, sem).wait()


def _barrier(peers, axis, dict_ids):
    sem = pltpu.get_barrier_semaphore()
    for p in peers:
        pl.semaphore_signal(sem, 1, device_id=_peer(p, axis, dict_ids), device_id_type=MESH)
    pl.semaphore_wait(sem, len(peers))


def reduce_scatter(x, rank, table, *, axis, dict_ids=True, interpret=None, name=None):
    """x [16 r, hidden] bf16 (this core's partial sum) -> [r, hidden] bf16, the
    sum over the cores of row block `rank`. rank int32 [1] (this core's mesh
    index), table int32 [CORES * TABLE_WIDTH] (peer_table "reduce_scatter")."""
    T, H = x.shape
    if not reduce_scatter_supports(T, H):
        raise ValueError(f"reduce_scatter does not take [{T}, {H}]")
    r = T // CORES
    nsub = len(ROTATIONS)
    W = H // nsub
    RC = 16
    CC = 512 if W % 512 == 0 else LANES

    def slot(buf, j, s):
        return buf.at[j * _SLOTS + s]

    def add_into(dst, src):
        """dst += src over [r, W] bf16 VMEM blocks: float32 add, rounded to bf16."""

        def rows(a, carry):
            rs = pl.ds(pl.multiple_of(a * RC, RC), RC)

            def cols(b, carry2):
                cs = pl.ds(pl.multiple_of(b * CC, CC), CC)
                dst[rs, cs] = (
                    dst[rs, cs].astype(jnp.float32) + src[rs, cs].astype(jnp.float32)
                ).astype(jnp.bfloat16)
                return carry2

            return lax.fori_loop(0, W // CC, cols, carry)

        lax.fori_loop(0, r // RC, rows, 0)

    def kernel(rank_ref, t_ref, x_ref, o_ref, buf, tmp, lsem, send, recv, msend, osem):
        base = rank_ref[0] * TABLE_WIDTH

        def at(column):
            return t_ref[base + column]

        mate = at(_MATE)
        _barrier([mate] + [at(_PARTNER + k) for k in range(3)], axis, dict_ids)

        def remote(src, dst, s, rv, p):
            pltpu.make_async_remote_copy(
                src, dst, s, rv, device_id=_peer(p, axis, dict_ids), device_id_type=MESH
            ).start()

        def rows_of(ref, block, j):
            return ref.at[pl.ds(block * r, r), pl.ds(j * W, W)]

        seq = [(j, i) for i in _MATE_ORDER for j in range(nsub)]

        def own_copy(n):
            j, i = seq[n]
            return pltpu.make_async_copy(
                rows_of(x_ref, at(_RS_OWN + j * 8 + i), j), tmp.at[n % 2], lsem.at[n % 2]
            )

        own_copy(0).start()
        for i in _MATE_ORDER:
            for j in range(nsub):
                remote(
                    rows_of(x_ref, at(_RS_TO_MATE + j * 8 + i), j),
                    slot(buf, j, i),
                    msend,
                    recv.at[j * _SLOTS + i],
                    mate,
                )

        done = []

        def complete(j, i):
            """Slot i of part j = the mate's partial sum + this core's own block."""
            n = len(done)
            assert seq[n] == (j, i), (seq[n], (j, i))
            if n + 1 < len(seq):
                own_copy(n + 1).start()
            own_copy(n).wait()
            _wait(slot(buf, j, i), recv.at[j * _SLOTS + i])
            add_into(slot(buf, j, i), tmp.at[n % 2])
            done.append((j, i))

        for k in (2, 1, 0):
            for i in _SENDS[k]:
                for j in range(nsub):
                    if k == 2:
                        complete(j, i)
                    dst = _BASE[k] + i - 2**k
                    remote(
                        slot(buf, j, i),
                        slot(buf, j, dst),
                        send,
                        recv.at[j * _SLOTS + dst],
                        at(_PARTNER + j * 3 + k),
                    )
            for i in _PRIORITY[k]:
                for j in range(nsub):
                    if k == 2:
                        complete(j, i)
                    _wait(slot(buf, j, _BASE[k] + i), recv.at[j * _SLOTS + _BASE[k] + i])
                    add_into(slot(buf, j, i), slot(buf, j, _BASE[k] + i))
        outs = [
            pltpu.make_async_copy(slot(buf, j, 0), o_ref.at[:, pl.ds(j * W, W)], osem)
            for j in range(nsub)
        ]
        for c in outs:
            c.start()
        _wait(x_ref.at[pl.ds(0, 8 * r)], msend)
        _wait(x_ref.at[pl.ds(0, 7 * r)], send)
        for c in outs:
            c.wait()

    need = (nsub * _SLOTS + 2) * r * W * 2
    kw = {"interpret": interpret} if interpret is not None else {}
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((r, H), jnp.bfloat16),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        scratch_shapes=[
            pltpu.VMEM((nsub * _SLOTS, r, W), jnp.bfloat16),
            pltpu.VMEM((2, r, W), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((nsub * _SLOTS,)),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID,
            vmem_limit_bytes=max(32 << 20, need + (8 << 20)),
        ),
        name=name or f"rotated_reduce_scatter_t{T}_h{H}",
        **kw,
    )(rank.reshape(1).astype(jnp.int32), table, x.astype(jnp.bfloat16))


def all_gather(x, rank, table, *, axis, dict_ids=True, interpret=None, name=None):
    """x [r, hidden] bf16 (this core's shard) -> [16 r, hidden] bf16, row
    block m the shard of mesh index m. rank int32 [1] (this core's mesh index),
    table int32 [CORES * TABLE_WIDTH] (peer_table "all_gather")."""
    (out,) = all_gather_many(
        (x.astype(jnp.bfloat16),),
        rank,
        table,
        axis=axis,
        dict_ids=dict_ids,
        interpret=interpret,
        name=name,
    )
    return out


def all_gather_many(xs, rank, table, *, axis, dict_ids=True, interpret=None, name=None):
    """`all_gather` of several arrays of the same row count, each in its own
    type, in one kernel: xs [r, hidden_i] -> [16 r, hidden_i], every array
    sent part by part along the same rotations as one block would be."""
    r = xs[0].shape[0]
    for x in xs:
        if x.shape[0] != r or not all_gather_supports(r, x.shape[1], x.dtype):
            raise ValueError(f"all_gather does not take {x.dtype}[{x.shape[0]}, {x.shape[1]}]")
    T = CORES * r
    nsub = len(ROTATIONS)
    count = len(xs)
    widths = [part_widths(x.shape[1], nsub) for x in xs]
    starts = [[sum(w[:j]) for j in range(nsub)] for w in widths]

    def kernel(rank_ref, t_ref, *refs):
        x_refs, o_refs = refs[:count], refs[count : 2 * count]
        stages = refs[2 * count : 3 * count]
        lsem, send, recv, msend, mrecv = refs[3 * count :]
        me = rank_ref[0]
        base = me * TABLE_WIDTH

        def at(column):
            return t_ref[base + column]

        def blk(a, block, j):
            return o_refs[a].at[pl.ds(block * r, r), pl.ds(starts[a][j], widths[a][j])]

        def own(a, j):
            return x_refs[a].at[:, pl.ds(starts[a][j], widths[a][j])]

        mate = at(_MATE)
        peers = [at(_PARTNER + j * 3 + k) for j in range(nsub) for k in range(3)]
        _barrier([mate] + peers[:3], axis, dict_ids)

        local_in = [
            pltpu.make_async_copy(x_refs[a], stages[a], lsem.at[a]) for a in range(count)
        ]
        for copy in local_in:
            copy.start()

        def remote(src, dst, s, rv, p):
            pltpu.make_async_remote_copy(
                src, dst, s, rv, device_id=_peer(p, axis, dict_ids), device_id_type=MESH
            ).start()

        for j in range(nsub):
            for a in range(count):
                remote(own(a, j), blk(a, me, j), msend, mrecv, mate)
        for k in range(4):
            for j in range(nsub):
                if k > 0:
                    for i in range(2 ** (k - 1)):
                        n = j * 7 + 2 ** (k - 1) - 1 + i
                        block = at(_AG_RECV + n)
                        for a in range(count):
                            _wait(blk(a, block, j), recv.at[a * 7 * nsub + n])
                        for a in range(count):
                            remote(blk(a, block, j), blk(a, block, j), msend, mrecv, mate)
                if k < 3:
                    partner = peers[j * 3 + k]
                    for i in range(2**k):
                        n = j * 7 + 2**k - 1 + i
                        block = at(_AG_SEND + n)
                        for a in range(count):
                            src = own(a, j) if i == 0 else blk(a, block, j)
                            remote(src, blk(a, block, j), send, recv.at[a * 7 * nsub + n], partner)
        local_out = []
        for a in range(count):
            local_in[a].wait()
            copy = pltpu.make_async_copy(
                stages[a], o_refs[a].at[pl.ds(me * r, r)], lsem.at[count + a]
            )
            copy.start()
            local_out.append(copy)
        for a in range(count):
            _wait(o_refs[a].at[pl.ds(0, 7 * r)], send)
            _wait(o_refs[a].at[pl.ds(0, 8 * r)], msend)
            _wait(o_refs[a].at[pl.ds(0, 8 * r)], mrecv)
        for copy in local_out:
            copy.wait()

    kw = {"interpret": interpret} if interpret is not None else {}
    hidden = "_".join(str(x.shape[1]) for x in xs)
    return pl.pallas_call(
        kernel,
        out_shape=tuple(jax.ShapeDtypeStruct((T, x.shape[1]), x.dtype) for x in xs),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            *(pl.BlockSpec(memory_space=pl.ANY) for _ in xs),
        ],
        out_specs=tuple(pl.BlockSpec(memory_space=pl.ANY) for _ in xs),
        scratch_shapes=[
            *(pltpu.VMEM(x.shape, x.dtype) for x in xs),
            pltpu.SemaphoreType.DMA((2 * count,)),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((7 * nsub * count,)),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID, vmem_limit_bytes=32 << 20
        ),
        name=name or f"rotated_all_gather_t{T}_h{hidden}",
        **kw,
    )(rank.reshape(1).astype(jnp.int32), table, *xs)


def _chunked(x, rows, call):
    """`call` on each `rows`-row chunk of x, in order, concatenated."""
    outs = [call(x[c * rows : (c + 1) * rows]) for c in range(x.shape[0] // rows)]
    return outs[0] if len(outs) == 1 else jnp.concatenate(outs, 0)


def reduce_scatter_chunks(x, rank, table, *, axis, chunk, dict_ids=True):
    """One reduce_scatter per `chunk`-row chunk of x, the outputs in chunk
    order: `lax.psum_scatter(tiled=True)` of each chunk, concatenated."""
    return _chunked(
        x, chunk, lambda c: reduce_scatter(c, rank, table, axis=axis, dict_ids=dict_ids)
    )


def all_gather_chunks(x, rank, table, *, axis, chunk, dict_ids=True):
    """One all_gather per `chunk`-row chunk of this core's rows x, the outputs
    in chunk order: `lax.all_gather(tiled=True)` of each chunk, concatenated."""
    return _chunked(
        x, chunk, lambda c: all_gather(c, rank, table, axis=axis, dict_ids=dict_ids)
    )


def all_gather_many_chunks(xs, rank, table, *, axis, chunk, dict_ids=True):
    """`all_gather_chunks` of several arrays of the same rows in one kernel
    per chunk: the outputs of each array in chunk order."""
    calls = [
        all_gather_many(
            [x[c * chunk : (c + 1) * chunk] for x in xs],
            rank,
            table,
            axis=axis,
            dict_ids=dict_ids,
        )
        for c in range(xs[0].shape[0] // chunk)
    ]
    if len(calls) == 1:
        return calls[0]
    return tuple(jnp.concatenate(parts, 0) for parts in zip(*calls))


_FUSED_VMEM_MAX = 56 << 20


def matmul_reduce_scatter_supports(rows, k, hidden, w_dtype):
    """Shapes the fused projection and reduce-scatter is written for: the
    reduce-scatter's shapes, an fp8 weight of whole lane tiles, and the weight,
    the quantized rows and the scratch inside the core's VMEM."""
    if not reduce_scatter_supports(rows, hidden) or w_dtype != jnp.float8_e4m3fn:
        return False
    if k % LANES:
        return False
    return _matmul_reduce_scatter_vmem(rows, k, hidden) <= _FUSED_VMEM_MAX


def _matmul_reduce_scatter_vmem(rows, k, hidden):
    r = rows // CORES
    nsub = len(ROTATIONS)
    W = hidden // nsub
    blocks = (nsub * _SLOTS + 8 * nsub + 2) * r * W * 2
    padded_r = -(-r // 32) * 32
    return blocks + k * hidden + CORES * padded_r * k + CORES * padded_r * LANES * 4 + 8 * hidden * 4


def matmul_reduce_scatter(x_q, x_scale, w_q, w_s, rank, table, *, axis, dict_ids=True, interpret=None, name=None):
    """This core's rows of the sum over the cores of the per-channel fp8
    projection of x, in one kernel: the projection's row blocks are computed
    in the order the rotated reduce-scatter sends them, each sent while the
    next is computed.

    x_q [16 r, k] fp8 and x_scale [16 r, 1] float32 are ``quantize_tensor``
    of the core's activations x; w_q [k, hidden] fp8 and w_s [hidden] float32
    the weight and its per-channel scale. Each block is
    bf16(float32(x_q . w_q) * x_scale * w_s), ``xla_quantized_matmul``'s
    expression; the sum is ``reduce_scatter``'s. Returns [r, hidden] bf16:
    row block `rank` of the sum."""
    T, K = x_q.shape
    H = w_q.shape[1]
    if not matmul_reduce_scatter_supports(T, K, H, w_q.dtype):
        raise ValueError(f"matmul_reduce_scatter does not take [{T}, {K}] x [{K}, {H}]")
    r = T // CORES
    nsub = len(ROTATIONS)
    W = H // nsub
    RC = 16
    CC = 512 if W % 512 == 0 else LANES

    def slot(buf, j, s):
        return buf.at[j * _SLOTS + s]

    def add_into(dst, src):
        def rows(a, carry):
            rs = pl.ds(pl.multiple_of(a * RC, RC), RC)

            def cols(b, carry2):
                cs = pl.ds(pl.multiple_of(b * CC, CC), CC)
                dst[rs, cs] = (
                    dst[rs, cs].astype(jnp.float32) + src[rs, cs].astype(jnp.float32)
                ).astype(jnp.bfloat16)
                return carry2

            return lax.fori_loop(0, W // CC, cols, carry)

        lax.fori_loop(0, r // RC, rows, 0)

    def kernel(rank_ref, t_ref, xq_ref, xs_ref, w_ref, ws_ref, o_ref, buf, stage, tmp, send, recv, msend, osem):
        base = rank_ref[0] * TABLE_WIDTH

        def at(column):
            return t_ref[base + column]

        def project(dst, block, j):
            """dst [r, W] = row block `block` of the partial, column part j."""
            xq = xq_ref[block]
            xs = xs_ref[block]

            def cols(b, carry):
                cs = pl.ds(pl.multiple_of(j * W + b * CC, CC), CC)
                y = lax.dot_general(
                    xq, w_ref[:, cs], (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
                )
                dst[:, pl.ds(pl.multiple_of(b * CC, CC), CC)] = (y * xs * ws_ref[:, cs]).astype(jnp.bfloat16)
                return carry

            lax.fori_loop(0, W // CC, cols, 0)

        mate = at(_MATE)
        _barrier([mate] + [at(_PARTNER + k) for k in range(3)], axis, dict_ids)

        def remote(src, dst, s, rv, p):
            pltpu.make_async_remote_copy(
                src, dst, s, rv, device_id=_peer(p, axis, dict_ids), device_id_type=MESH
            ).start()

        n = 0
        for i in _MATE_ORDER:
            for j in range(nsub):
                project(stage.at[n], at(_RS_TO_MATE + j * 8 + i), j)
                remote(stage.at[n], slot(buf, j, i), msend, recv.at[j * _SLOTS + i], mate)
                n += 1

        done = []

        def complete(j, i):
            """Slot i of part j = the mate's partial sum + this core's own block."""
            k = len(done)
            project(tmp.at[k % 2], at(_RS_OWN + j * 8 + i), j)
            _wait(slot(buf, j, i), recv.at[j * _SLOTS + i])
            add_into(slot(buf, j, i), tmp.at[k % 2])
            done.append((j, i))

        for k in (2, 1, 0):
            for i in _SENDS[k]:
                for j in range(nsub):
                    if k == 2:
                        complete(j, i)
                    dst = _BASE[k] + i - 2**k
                    remote(
                        slot(buf, j, i),
                        slot(buf, j, dst),
                        send,
                        recv.at[j * _SLOTS + dst],
                        at(_PARTNER + j * 3 + k),
                    )
            for i in _PRIORITY[k]:
                for j in range(nsub):
                    if k == 2:
                        complete(j, i)
                    _wait(slot(buf, j, _BASE[k] + i), recv.at[j * _SLOTS + _BASE[k] + i])
                    add_into(slot(buf, j, i), slot(buf, j, _BASE[k] + i))
        outs = [
            pltpu.make_async_copy(slot(buf, j, 0), o_ref.at[:, pl.ds(j * W, W)], osem)
            for j in range(nsub)
        ]
        for c in outs:
            c.start()
        _wait(stage, msend)
        _wait(buf.at[pl.ds(0, 7 * nsub)], send)
        for c in outs:
            c.wait()

    need = _matmul_reduce_scatter_vmem(T, K, H)
    kw = {"interpret": interpret} if interpret is not None else {}
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((r, H), jnp.bfloat16),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        scratch_shapes=[
            pltpu.VMEM((nsub * _SLOTS, r, W), jnp.bfloat16),
            pltpu.VMEM((8 * nsub, r, W), jnp.bfloat16),
            pltpu.VMEM((2, r, W), jnp.bfloat16),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((nsub * _SLOTS,)),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID,
            vmem_limit_bytes=min(max(32 << 20, need + (8 << 20)), 64 << 20),
        ),
        name=name or f"rotated_matmul_reduce_scatter_t{T}_k{K}_h{H}",
        **kw,
    )(
        rank.reshape(1).astype(jnp.int32),
        table,
        x_q.reshape(CORES, r, K),
        x_scale.astype(jnp.float32).reshape(CORES, r, 1),
        w_q,
        w_s.astype(jnp.float32).reshape(1, H),
    )


def matmul_reduce_scatter_chunks(x_q, x_scale, w_q, w_s, rank, table, *, axis, chunk, dict_ids=True):
    """One matmul_reduce_scatter per `chunk`-row chunk of the core's quantized
    activations, the outputs in chunk order: the per-chunk reduce-scatter of
    ``xla_quantized_matmul``, concatenated."""
    outs = [
        matmul_reduce_scatter(
            x_q[c * chunk : (c + 1) * chunk],
            x_scale[c * chunk : (c + 1) * chunk],
            w_q,
            w_s,
            rank,
            table,
            axis=axis,
            dict_ids=dict_ids,
        )
        for c in range(x_q.shape[0] // chunk)
    ]
    return outs[0] if len(outs) == 1 else jnp.concatenate(outs, 0)

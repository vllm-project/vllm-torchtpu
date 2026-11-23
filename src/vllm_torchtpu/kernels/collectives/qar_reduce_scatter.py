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
"""Row reduce-scatter of a large bf16 [T, H] over the 8 TensorCores of a
tpu7x 2x2x1 with an int8 payload on the chip-to-chip links: rank r gets the
sum of rows [r T/8, (r + 1) T/8) of every rank's x.

The reduce-scatter half of qar_all_reduce.py, with rows as the scattered
dimension. A bf16 reduce-scatter of this size is bound by the ICI links, so
its time is its bytes; this one sends half of them:

1. d2d, bf16: each core sends its sibling core the row chunks owned by the
   ranks with the sibling's core index (4 of the 8), on the on-package link;
   the pair's sum is formed in f32 when it is used.
2. ICI ring reduce-scatter, int8: the cores with the same core index on the 4
   chips form a ring (x, y, x, y links). Every chunk's columns split into a
   clockwise and a counter-clockwise half, so both of a chip's links carry
   traffic. Each of 3 ring steps sends int8 with one absmax scale per
   (32 rows, lane); the receiver dequantizes, adds its own f32 pair partial
   and requantizes. After the third step each rank adds its own partial to
   what it received (in f32) and writes its bf16 rows: no all-gather.

One fused Pallas call. The rows of each chunk split into two halves that run
the same steps interleaved, so one half's remote DMA is in flight while the
other half's dequantize / add / quantize (an emit_pipeline over 256-row tiles)
runs. Every ring step has its own receive buffer, so no credits are needed.
Error: ~0.5% rel L2 per call against an f64 sum (bf16: ~0.2%).
"""

import functools
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.collectives import qar_all_reduce as qar

BF16, F32, I8 = jnp.bfloat16, jnp.float32, jnp.int8
COLLECTIVE_ID = qar.COLLECTIVE_ID
QROWS = qar.QROWS
TILE = 256  # rows per pipeline tile (8 f32 scale rows)
_MESH = pl.DeviceIdType.MESH
_N = 8
# Rows of the layout table: qar's five, then OWNER[pos * 2 + core] = the mesh
# index of the core `core` on the chip at ring position `pos`.
SIB, CW_NEXT, CW_PREV, POS, CORE, OWNER = range(6)
# The d2d leg's 4 chunk slots, in the order the ring needs them: chunk
# pos + _REL[s] (ring positions mod 4) of the receiver's core index.
_REL = (-1, 1, 2, 0)
# Slot of the own-partial chunk used at ring step i (3 = the final add):
# clockwise uses chunk pos - 1 - i, counter-clockwise pos + 1 + i.
_CW_SLOT = (0, 2, 1, 3)
_CC_SLOT = (1, 2, 0, 3)
# d2d slots first used at each step.
_D2D_WAIT = ((0, 1), (2,), (), (3,))


def layout_table(devices: Sequence) -> np.ndarray:
    """int32 [6, 8]: qar_all_reduce.layout_table plus the OWNER row."""
    tab = qar.layout_table(devices)
    own = np.zeros((1, _N), np.int32)
    for i in range(_N):
        own[0, tab[POS, i] * 2 + tab[CORE, i]] = i
    return np.concatenate([tab, own]).astype(np.int32)


def chunk_rows(rows: int) -> int:
    """Rows per rank chunk the kernel runs on (padded to whole tiles)."""
    cr = rows // _N
    return -(-cr // TILE) * TILE


def col_block(w: int) -> int:
    """Pipeline tile width (lanes) for a direction's column half of w."""
    for wb in (768, 512, 384, 256, 128):
        if w % wb == 0:
            return wb
    raise ValueError(f"column half {w} is not a multiple of 128")


def _step_body(with_prev: bool, final: bool):
    def body(*refs):
        x_cw, r_cw, x_cc, r_cc = refs[:4]
        rest = refs[4:]
        a_cw = x_cw[...].astype(F32) + r_cw[...].astype(F32)
        a_cc = x_cc[...].astype(F32) + r_cc[...].astype(F32)
        if with_prev:
            pq_cw, ps_cw, pq_cc, ps_cc = rest[:4]
            rest = rest[4:]
            a_cw = a_cw + qar._deq_tile(pq_cw[...], ps_cw[...])
            a_cc = a_cc + qar._deq_tile(pq_cc[...], ps_cc[...])
        if final:
            o_cw, o_cc = rest
            o_cw[...] = a_cw.astype(o_cw.dtype)
            o_cc[...] = a_cc.astype(o_cc.dtype)
        else:
            q_cw, s_cw, q_cc, s_cc = rest
            q_cw[...], s_cw[...] = qar._quant_tile(a_cw)
            q_cc[...], s_cc[...] = qar._quant_tile(a_cc)

    return body


@functools.cache
def _fused_call(cr: int, h: int, nh: int, interpret=None):
    """x [8 cr, h] bf16 -> this rank's reduced chunk [cr, h] (first output)."""
    if h % (2 * 128):
        raise ValueError(f"hidden {h} must split into 2 lane-aligned halves")
    if cr % (nh * TILE):
        raise ValueError(f"chunk rows {cr} must be a multiple of {nh} x {TILE}")
    w = h // 2
    wb = col_block(w)
    nc = w // wb  # column tiles per direction
    th = cr // nh  # rows per half
    nt = th // TILE  # row tiles per half
    ct = cr // TILE  # row tiles per chunk
    sr = TILE // QROWS
    nbuf = 3 * 2 * nh  # (step, direction, half)

    def slot(i, d, hf):
        return (i * 2 + d) * nh + hf

    def kernel(rank_ref, tab_ref, x_ref, out, recv, sq, ss, rq, rs,
               d2d_send, d2d_recv, rs_send, rs_recv):
        me = rank_ref[0]
        core = tab_ref[CORE, me]
        pos = tab_ref[POS, me]
        sib = tab_ref[SIB, me]
        peer = (tab_ref[CW_NEXT, me], tab_ref[CW_PREV, me])
        barrier = pltpu.get_barrier_semaphore()
        for dv in range(_N):
            pl.semaphore_signal(barrier, 1, device_id=(dv,), device_id_type=_MESH)
        pl.semaphore_wait(barrier, _N)

        def owner(k, c):
            return tab_ref[OWNER, lax.rem(k + 8, 4) * 2 + c]

        def remote(src, dst, ssem, rsem, dev):
            return pltpu.make_async_remote_copy(src, dst, ssem, rsem, device_id=(dev,),
                                                device_id_type=_MESH)

        # 1. d2d: the sibling's chunks, full rows, in the order it needs them.
        def d2d(s, hf):
            src = owner(pos + _REL[s], 1 - core) * cr + hf * th
            return remote(x_ref.at[pl.ds(pl.multiple_of(src, TILE), th)],
                          recv.at[pl.ds(s * cr + hf * th, th)],
                          d2d_send.at[s * nh + hf], d2d_recv.at[s * nh + hf], sib)

        d2ds = {(s, hf): d2d(s, hf) for s in range(4) for hf in range(nh)}
        for s in (0, 1):
            for hf in range(nh):
                d2ds[(s, hf)].start()
        sends = list(d2ds.values())

        def tile(rb, cb):
            return pl.BlockSpec((TILE, wb), lambda r, c: (rb + r, cb + c))

        def qspec(j):
            return pl.BlockSpec((TILE, wb), lambda r, c: (j * nt + r, c))

        def sspec(j):
            return pl.BlockSpec((sr, wb), lambda r, c: (j * nt + r, c))

        rs_copies = {}
        # 2. ring reduce-scatter, interleaved over the halves.
        for i in range(4):
            for hf in range(nh):
                for s in _D2D_WAIT[i]:
                    d2ds[(s, hf)].wait_recv()
                if i > 0:
                    for d in range(2):
                        for c in rs_copies[(i - 1, d, hf)]:
                            c.wait_recv()
                xb_cw = owner(pos - 1 - i, core) * ct + hf * nt
                xb_cc = owner(pos + 1 + i, core) * ct + hf * nt
                in_specs = [tile(xb_cw, 0), tile(_CW_SLOT[i] * ct + hf * nt, 0),
                            tile(xb_cc, nc), tile(_CC_SLOT[i] * ct + hf * nt, nc)]
                args = [x_ref, recv, x_ref, recv]
                if i > 0:
                    for d in range(2):
                        j = slot(i - 1, d, hf)
                        in_specs += [qspec(j), sspec(j)]
                        args += [rq, rs]
                if i < 3:
                    out_specs, outs = [], []
                    for d in range(2):
                        j = slot(i, d, hf)
                        out_specs += [qspec(j), sspec(j)]
                        outs += [sq, ss]
                else:
                    out_specs = [tile(hf * nt, 0), tile(hf * nt, nc)]
                    outs = [out, out]
                pltpu.emit_pipeline(_step_body(i > 0, i == 3), grid=(nt, nc),
                                    in_specs=in_specs, out_specs=out_specs)(*args, *outs)
                if i < 3:
                    for d in range(2):
                        j = slot(i, d, hf)
                        qs = pl.ds(j * th, th)
                        ssl = pl.ds(j * (th // QROWS), th // QROWS)
                        pair = (remote(sq.at[qs], rq.at[qs], rs_send.at[2 * j],
                                       rs_recv.at[2 * j], peer[d]),
                                remote(ss.at[ssl], rs.at[ssl], rs_send.at[2 * j + 1],
                                       rs_recv.at[2 * j + 1], peer[d]))
                        for c in pair:
                            c.start()
                        rs_copies[(i, d, hf)] = pair
                        sends += list(pair)
                if i == 0 and hf == 0:  # the rest of the d2d leg behind step 0
                    for s in (2, 3):
                        for h2 in range(nh):
                            d2ds[(s, h2)].start()
        for c in sends:
            c.wait_send()

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    outs = (
        jax.ShapeDtypeStruct((cr, h), BF16),                     # out
        jax.ShapeDtypeStruct((4 * cr, h), BF16),                 # recv (d2d)
        jax.ShapeDtypeStruct((nbuf * th, w), I8),                # sq
        jax.ShapeDtypeStruct((nbuf * th // QROWS, w), F32),      # ss
        jax.ShapeDtypeStruct((nbuf * th, w), I8),                # rq
        jax.ShapeDtypeStruct((nbuf * th // QROWS, w), F32),      # rs
    )
    kw = {} if interpret is None else dict(interpret=interpret)
    return pl.pallas_call(
        kernel,
        out_shape=outs,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[hbm],
            out_specs=[hbm] * len(outs),
            scratch_shapes=[
                pltpu.SemaphoreType.DMA((4 * nh,)), pltpu.SemaphoreType.DMA((4 * nh,)),
                pltpu.SemaphoreType.DMA((2 * nbuf,)), pltpu.SemaphoreType.DMA((2 * nbuf,)),
            ],
        ),
        # No has_side_effects: under the serving shard_map (torch_tpu jax_op)
        # the SPMD partitioner refuses a side-effecting custom call without a
        # sharding; the used output keeps the call alive (as qar_all_reduce).
        compiler_params=pltpu.CompilerParams(collective_id=COLLECTIVE_ID),
        name=f"qrs_fused-cr_{cr}-h_{h}-nh_{nh}",
        **kw,
    )


def _pad_chunks(x: jax.Array, cr: int, crp: int) -> jax.Array:
    if crp == cr:
        return x
    h = x.shape[1]
    return jnp.pad(x.reshape(_N, cr, h), ((0, 0), (0, crp - cr), (0, 0))).reshape(_N * crp, h)


def reduce_scatter(x: jax.Array, me: jax.Array, tab: np.ndarray, interpret=None) -> jax.Array:
    """Row reduce-scatter (TPU; `interpret` for CPU tests): bf16 x [rows, h]
    per rank -> rank me's chunk [rows / 8, h] of the sum. Call under
    shard_map over the 1-D mesh whose order `tab` (layout_table) describes."""
    rows, h = x.shape
    if rows % _N:
        raise ValueError(f"rows {rows} must be a multiple of {_N}")
    cr = rows // _N
    crp = chunk_rows(rows)
    nh = 2 if crp % (2 * TILE) == 0 else 1
    x = _pad_chunks(x.astype(BF16), cr, crp)
    rank = me.reshape(1).astype(jnp.int32)
    out = _fused_call(crp, h, nh, interpret)(rank, jnp.asarray(tab), x)[0]
    return out[:cr]


def reduce_scatter_ref(x: jax.Array, me: jax.Array, tab: np.ndarray, exchange) -> jax.Array:
    """The same steps and rounding over `exchange` (qar.ppermute_exchange on
    CPU): the reference for the fused call."""
    rows, h = x.shape
    cr = rows // _N
    crp = chunk_rows(rows)
    w = h // 2
    xr = _pad_chunks(x.astype(BF16), cr, crp).reshape(_N, crp, h)
    tab_j = jnp.asarray(tab)
    me = me.reshape(()).astype(jnp.int32)
    core, pos = tab_j[CORE, me], tab_j[POS, me]

    def owner(k, c):
        return tab_j[OWNER, ((k + 8) % 4) * 2 + c]

    recv = exchange([xr[owner(pos + _REL[s], 1 - core)] for s in range(4)], [SIB] * 4)
    prev = None
    for i in range(4):
        x_cw = xr[owner(pos - 1 - i, core)][:, :w]
        x_cc = xr[owner(pos + 1 + i, core)][:, w:]
        a_cw = x_cw.astype(F32) + recv[_CW_SLOT[i]][:, :w].astype(F32)
        a_cc = x_cc.astype(F32) + recv[_CC_SLOT[i]][:, w:].astype(F32)
        if prev is not None:
            a_cw = a_cw + qar.dequantize(prev[0], prev[1])
            a_cc = a_cc + qar.dequantize(prev[2], prev[3])
        if i == 3:
            return jnp.concatenate([a_cw, a_cc], axis=1).astype(BF16)[:cr]
        q_cw, s_cw = qar.quantize(a_cw)
        q_cc, s_cc = qar.quantize(a_cc)
        prev = exchange([q_cw, s_cw, q_cc, s_cc], [CW_NEXT, CW_NEXT, CW_PREV, CW_PREV])

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
"""All-reduce of a large bf16 [T, H] over the 8 TensorCores of a tpu7x 2x2x1
with an int8 payload on the chip-to-chip links.

A bf16 all-reduce of this size is bound by the ICI links (one link per
neighbour direction per chip, ~88 GB/s each way; the two cores of a chip share
it), so its time is its bytes. This one moves half of them:

1. d2d reduce-scatter, bf16: core c of each chip keeps column half c and sends
   the other half to its sibling core (the on-package link, ~500 GB/s/core);
   the pair's sum is kept in f32.
2. ICI reduce-scatter, int8: the cores with the same c on the 4 chips form a
   ring (x, y, x, y links). The half splits into a clockwise and a
   counter-clockwise quarter so both of a chip's links carry traffic, each
   quarter into 4 chunks; 3 ring steps leave each chip one fully reduced chunk
   per direction. Every hop sends int8 with one absmax scale per (32 rows,
   lane), and the receiver dequantizes, adds its own f32 partial and
   requantizes.
3. ICI all-gather, int8: the owned chunk is quantized once and forwarded
   unchanged around the ring, so every rank dequantizes the same bytes (TP
   replicas must agree bit for bit; the owner uses the dequantized value too).
4. d2d all-gather, bf16: each core writes its half into its own and its
   sibling's output.

The data movement is Pallas (remote DMA, one call per leg and ring step); the
dequantize / add / quantize between legs is plain XLA. Error: ~0.8-0.9% rel
L2 per call against an f64 sum (bf16 all-reduce: 0.17%).
"""

import functools
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BF16, F32, I8 = jnp.bfloat16, jnp.float32, jnp.int8
COLLECTIVE_ID = 3  # tp_all_reduce uses 0
QROWS = 32  # rows per quantization scale (int8 sublane tile)
_MESH = pl.DeviceIdType.MESH
_N = 8
# Rows of the layout table.
SIB, CW_NEXT, CW_PREV, POS, CORE, DIAG = range(6)


def layout_table(devices: Sequence) -> np.ndarray:
    """int32 [6, 8] over the mesh order: sibling core, clockwise ring next and
    previous (same core index on the neighbouring chip), ring position, core
    index, and the diagonal chip's same core. The chip ring is (x0,y0) -> (x1,y0) -> (x1,y1) -> (x0,y1)."""
    if len(devices) != _N:
        raise ValueError(f"qar_all_reduce needs 8 TensorCores, got {len(devices)}")
    devices = [_physical(d) for d in devices]
    chips = sorted({tuple(d.coords[:2]) for d in devices})
    if len(chips) != 4 or {d.core_on_chip for d in devices} != {0, 1}:
        raise ValueError(f"qar_all_reduce needs 4 chips x 2 cores: {chips}")
    xs = sorted({c[0] for c in chips})
    ys = sorted({c[1] for c in chips})
    ring = [(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])]
    index = {(tuple(d.coords[:2]), d.core_on_chip): i for i, d in enumerate(devices)}
    tab = np.zeros((6, _N), np.int32)
    for i, d in enumerate(devices):
        chip, core = tuple(d.coords[:2]), d.core_on_chip
        p = ring.index(chip)
        tab[SIB, i] = index[(chip, 1 - core)]
        tab[CW_NEXT, i] = index[(ring[(p + 1) % 4], core)]
        tab[CW_PREV, i] = index[(ring[(p - 1) % 4], core)]
        tab[POS, i] = p
        tab[CORE, i] = core
        tab[DIAG, i] = index[(ring[(p + 2) % 4], core)]
    return tab


class _Placed:
    """A device's id with the physical coordinates layout_table needs."""

    def __init__(self, id_: int, coords: tuple, core_on_chip: int):
        self.id, self.coords, self.core_on_chip = id_, coords, core_on_chip


def _physical(d):
    """torch_tpu's Pallas mesh holds abstract devices with only an id. Take the
    coordinates from jax.devices() when this process sees that id, else from a
    tpu7x 2x2x1 host's numbering: id = 2 * chip + core, chip c at (c % 2, c // 2)
    (read from jax.devices() on such a host)."""
    if hasattr(d, "coords") and hasattr(d, "core_on_chip"):
        return d
    did = int(d.id)
    for real in jax.devices():
        if int(real.id) == did and hasattr(real, "coords"):
            return real
    if not 0 <= did < _N:
        raise ValueError(f"qar_all_reduce: no coordinates for device id {did}")
    chip = did // 2
    return _Placed(did, (chip % 2, chip // 2, 0), did % 2)


def padded_rows(rows: int) -> int:
    # f32 scales [rows / 32, w] keep an 8-row tile: rows multiple of 256.
    return -(-rows // (QROWS * 8)) * (QROWS * 8)


def quantize(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    """f32 [T, W] -> int8 [T, W], f32 scales [T / 32, W] (absmax / 127)."""
    t, w = a.shape
    b = a.reshape(t // QROWS, QROWS, w)
    s = jnp.max(jnp.abs(b), axis=1) / 127.0
    s = jnp.where(s == 0, 1.0, s)
    q = jnp.clip(jnp.round(b / s[:, None, :]), -127, 127).astype(I8)
    return q.reshape(t, w), s


def dequantize(q: jax.Array, s: jax.Array) -> jax.Array:
    t, w = q.shape
    return (q.reshape(t // QROWS, QROWS, w).astype(F32) * s[:, None, :]).reshape(t, w)


# --------------------------------------------------------------------------
# Data movement. `Exchange(bufs, rows)` sends bufs[i] to the device in table
# row rows[i] and returns what this device received in the same slot; every
# row used is a permutation of the mesh, so each device receives one per slot.
Exchange = Callable[[Sequence[jax.Array], Sequence[int]], list]

_HBM = pl.BlockSpec(memory_space=pltpu.HBM)


@functools.cache
def _exchange_call(avals: tuple, rows: tuple, tab_key: bytes):
    nb = len(avals)

    def kernel(rank_ref, tab_ref, *refs):
        ins, outs = refs[:nb], refs[nb : 2 * nb]
        send_sem, recv_sem = refs[2 * nb], refs[2 * nb + 1]
        me = rank_ref[0]
        barrier = pltpu.get_barrier_semaphore()
        for d in range(_N):
            pl.semaphore_signal(barrier, 1, device_id=(d,), device_id_type=_MESH)
        pl.semaphore_wait(barrier, _N)
        copies = []
        for b in range(nb):
            copy = pltpu.make_async_remote_copy(
                ins[b],
                outs[b],
                send_sem.at[b],
                recv_sem.at[b],
                device_id=(tab_ref[rows[b], me],),
                device_id_type=_MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait_recv()
        for copy in copies:
            copy.wait_send()

    return pl.pallas_call(
        kernel,
        out_shape=tuple(jax.ShapeDtypeStruct(s, d) for s, d in avals),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[_HBM] * nb,
            out_specs=[_HBM] * nb,
            scratch_shapes=[pltpu.SemaphoreType.DMA((nb,)), pltpu.SemaphoreType.DMA((nb,))],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID
        ),
        name="qar_exchange-" + "-".join(f"{d}{'x'.join(map(str, s))}" for s, d in avals),
    )


def pallas_exchange(rank: jax.Array, tab: np.ndarray) -> Exchange:
    tab_j = jnp.asarray(tab)

    def exchange(bufs, rows):
        avals = tuple((tuple(b.shape), jnp.dtype(b.dtype).name) for b in bufs)
        call = _exchange_call(avals, tuple(rows), tab.tobytes())
        out = call(rank.reshape(1).astype(jnp.int32), tab_j, *bufs)
        return list(out)

    return exchange


def ppermute_exchange(axis: str, tab: np.ndarray) -> Exchange:
    """Reference data movement over lax.ppermute (CPU tests)."""

    def exchange(bufs, rows):
        return [
            lax.ppermute(b, axis, [(i, int(tab[r, i])) for i in range(_N)])
            for b, r in zip(bufs, rows)
        ]

    return exchange


# --------------------------------------------------------------------------
def all_reduce(x: jax.Array, me: jax.Array, tab: np.ndarray, exchange: Exchange) -> jax.Array:
    """Sum of bf16 x [rows, hidden] over the 8 ranks; call under shard_map over
    the 1-D mesh whose order `tab` describes. me: this rank's mesh index."""
    rows, h = x.shape
    hh = h // 2
    if h % (2 * 2 * 4 * 128):
        raise ValueError(f"hidden {h} must split into 16 lane-aligned chunks")
    w = hh // 8  # chunk width: 2 directions x 4 chunks per half
    t = padded_rows(rows)
    if t != rows:
        x = jnp.pad(x, ((0, t - rows), (0, 0)))
    tab_j = jnp.asarray(tab)
    me = me.reshape(()).astype(jnp.int32)
    core = tab_j[CORE, me]
    pos = tab_j[POS, me]

    # 1. d2d reduce-scatter (bf16).
    mine = lax.dynamic_slice_in_dim(x, core * hh, hh, axis=1)
    other = lax.dynamic_slice_in_dim(x, (1 - core) * hh, hh, axis=1)
    (recv,) = exchange([other], [SIB])
    part = (mine.astype(F32) + recv.astype(F32)).reshape(t, 2, 4, w)

    def chunk(d, k):
        return lax.dynamic_index_in_dim(part[:, d], k % 4, axis=1, keepdims=False)

    # 2. ICI ring reduce-scatter (int8). Clockwise: step i sends chunk pos - i;
    # counter-clockwise: pos + i. Ends owning chunk pos + 1 / pos - 1.
    q_cw, s_cw = quantize(chunk(0, pos))
    q_cc, s_cc = quantize(chunk(1, pos))
    for i in range(3):
        rq_cw, rs_cw, rq_cc, rs_cc = exchange(
            [q_cw, s_cw, q_cc, s_cc], [CW_NEXT, CW_NEXT, CW_PREV, CW_PREV]
        )
        acc_cw = dequantize(rq_cw, rs_cw) + chunk(0, pos - i - 1)
        acc_cc = dequantize(rq_cc, rs_cc) + chunk(1, pos + i + 1)
        q_cw, s_cw = quantize(acc_cw)
        q_cc, s_cc = quantize(acc_cc)

    # 3. ICI ring all-gather (int8, forwarded unchanged).
    out = jnp.zeros((t, 2, 4, w), BF16)

    def put(out, d, k, q, s):
        block = dequantize(q, s).astype(BF16)[:, None, None, :]
        return lax.dynamic_update_slice(out, block, (0, d, k % 4, 0))

    out = put(out, 0, pos + 1, q_cw, s_cw)
    out = put(out, 1, pos - 1, q_cc, s_cc)
    for i in range(3):
        q_cw, s_cw, q_cc, s_cc = exchange(
            [q_cw, s_cw, q_cc, s_cc], [CW_NEXT, CW_NEXT, CW_PREV, CW_PREV]
        )
        out = put(out, 0, pos - i, q_cw, s_cw)
        out = put(out, 1, pos + i, q_cc, s_cc)
    half = out.reshape(t, hh)

    # 4. d2d all-gather (bf16).
    (sib_half,) = exchange([half], [SIB])
    full = jnp.zeros((t, h), BF16)
    full = lax.dynamic_update_slice_in_dim(full, half, core * hh, axis=1)
    full = lax.dynamic_update_slice_in_dim(full, sib_half, (1 - core) * hh, axis=1)
    return full[:rows]


# --------------------------------------------------------------------------
# v1: the same algorithm with the glue in Pallas. Every chunk is addressed by
# a scalar-prefetched column-block index in a BlockSpec index_map (no dynamic
# slices, no 4-D views), quantize / dequantize / add run per 256-row tile, and
# the all-gather writes each chunk in place.
_TILE = 256


def _quant_tile(a):
    r, w = a.shape
    b = a.reshape(r // QROWS, QROWS, w)
    s = jnp.max(jnp.abs(b), axis=1) / 127.0
    s = jnp.where(s == 0, 1.0, s)
    q = jnp.clip(jnp.round(b / s[:, None, :]), -127.0, 127.0)
    return q.reshape(r, w).astype(I8), s


def _deq_tile(q, s):
    r, w = q.shape
    return (q.astype(F32).reshape(r // QROWS, QROWS, w) * s[:, None, :]).reshape(r, w)


@functools.cache
def _step_call(t: int, h: int, w: int, with_prev: bool):
    """(x, recv, [q_cw, s_cw, q_cc, s_cc]) -> q/s of (x_chunk + recv_chunk
    [+ deq(prev)]) for both directions. blk = [x block cw, x block cc, recv
    block cw, recv block cc] in units of w columns."""
    sr = _TILE // QROWS

    def kernel(blk_ref, *refs):
        if with_prev:
            x_cw, x_cc, r_cw, r_cc, pq_cw, ps_cw, pq_cc, ps_cc = refs[:8]
            outs = refs[8:]
        else:
            x_cw, x_cc, r_cw, r_cc = refs[:4]
            outs = refs[4:]
        q_cw, s_cw, q_cc, s_cc = outs
        a_cw = x_cw[...].astype(F32) + r_cw[...].astype(F32)
        a_cc = x_cc[...].astype(F32) + r_cc[...].astype(F32)
        if with_prev:
            a_cw = a_cw + _deq_tile(pq_cw[...], ps_cw[...])
            a_cc = a_cc + _deq_tile(pq_cc[...], ps_cc[...])
        q_cw[...], s_cw[...] = _quant_tile(a_cw)
        q_cc[...], s_cc[...] = _quant_tile(a_cc)

    tile = lambda j: pl.BlockSpec((_TILE, w), lambda i, blk: (i, blk[j]))
    qspec = pl.BlockSpec((_TILE, w), lambda i, blk: (i, 0))
    sspec = pl.BlockSpec((sr, w), lambda i, blk: (i, 0))
    in_specs = [tile(0), tile(1), tile(2), tile(3)]
    if with_prev:
        in_specs += [qspec, sspec, qspec, sspec]
    q_shape = jax.ShapeDtypeStruct((t, w), I8)
    s_shape = jax.ShapeDtypeStruct((t // QROWS, w), F32)
    return pl.pallas_call(
        kernel,
        out_shape=(q_shape, s_shape, q_shape, s_shape),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(t // _TILE,),
            in_specs=in_specs,
            out_specs=[qspec, sspec, qspec, sspec],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name=f"qar_step-t_{t}-w_{w}-prev_{int(with_prev)}",
    )


@functools.cache
def _put_call(t: int, w: int):
    """Dequantize q/s of both directions into blocks blk[0] / blk[1] of the
    per-direction halves [t, 4w], in place."""
    sr = _TILE // QROWS

    def kernel(blk_ref, q_cw, s_cw, q_cc, s_cc, h_cw_in, h_cc_in, h_cw, h_cc):
        del h_cw_in, h_cc_in
        h_cw[...] = _deq_tile(q_cw[...], s_cw[...]).astype(BF16)
        h_cc[...] = _deq_tile(q_cc[...], s_cc[...]).astype(BF16)

    qspec = pl.BlockSpec((_TILE, w), lambda i, blk: (i, 0))
    sspec = pl.BlockSpec((sr, w), lambda i, blk: (i, 0))
    hspec = lambda j: pl.BlockSpec((_TILE, w), lambda i, blk: (i, blk[j]))
    half = jax.ShapeDtypeStruct((t, 4 * w), BF16)
    return pl.pallas_call(
        kernel,
        out_shape=(half, half),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(t // _TILE,),
            in_specs=[qspec, sspec, qspec, sspec, hspec(0), hspec(1)],
            out_specs=[hspec(0), hspec(1)],
        ),
        input_output_aliases={5: 0, 6: 1},  # operand index counts the prefetch arg
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name=f"qar_put-t_{t}-w_{w}",
    )


@functools.cache
def _d2d_rs_call(t: int, h: int):
    """x [t, h] -> the sibling's column half (1 - its core) of x, bf16 [t, h/2]:
    each core sends x[:, (1 - core) * h/2 :] straight from x."""
    hh = h // 2

    def kernel(rank_ref, tab_ref, x_ref, o_ref, send_sem, recv_sem):
        me = rank_ref[0]
        barrier = pltpu.get_barrier_semaphore()
        for d in range(_N):
            pl.semaphore_signal(barrier, 1, device_id=(d,), device_id_type=_MESH)
        pl.semaphore_wait(barrier, _N)
        off = pl.multiple_of((1 - tab_ref[CORE, me]) * hh, 128)
        copy = pltpu.make_async_remote_copy(
            x_ref.at[:, pl.ds(off, hh)],
            o_ref,
            send_sem.at[0],
            recv_sem.at[0],
            device_id=(tab_ref[SIB, me],),
            device_id_type=_MESH,
        )
        copy.start()
        copy.wait_recv()
        copy.wait_send()

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((t, hh), BF16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[_HBM],
            out_specs=_HBM,
            scratch_shapes=[pltpu.SemaphoreType.DMA((1,)), pltpu.SemaphoreType.DMA((1,))],
        ),
        compiler_params=pltpu.CompilerParams(collective_id=COLLECTIVE_ID),
        name=f"qar_d2d_rs-t_{t}-h_{h}",
    )


@functools.cache
def _d2d_ag_call(t: int, h: int):
    """Halves h_cw, h_cc [t, h/4] -> full [t, h]: each core writes its two
    quarters at columns core * h/2 (+ h/4) of its own and its sibling's output."""
    hh, hq = h // 2, h // 4

    def kernel(rank_ref, tab_ref, h_cw, h_cc, o_ref, send_sem, recv_sem, local_sem):
        me = rank_ref[0]
        barrier = pltpu.get_barrier_semaphore()
        for d in range(_N):
            pl.semaphore_signal(barrier, 1, device_id=(d,), device_id_type=_MESH)
        pl.semaphore_wait(barrier, _N)
        base = tab_ref[CORE, me] * hh
        sib = tab_ref[SIB, me]
        copies, locals_ = [], []
        for j, src in enumerate((h_cw, h_cc)):
            dst = o_ref.at[:, pl.ds(pl.multiple_of(base + j * hq, 128), hq)]
            copy = pltpu.make_async_remote_copy(
                src, dst, send_sem.at[j], recv_sem.at[j], device_id=(sib,), device_id_type=_MESH
            )
            copy.start()
            copies.append(copy)
            local = pltpu.make_async_copy(src, dst, local_sem.at[j])
            local.start()
            locals_.append(local)
        for c in locals_:
            c.wait()
        for c in copies:
            c.wait_recv()
        for c in copies:
            c.wait_send()

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((t, h), BF16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[_HBM, _HBM],
            out_specs=_HBM,
            scratch_shapes=[pltpu.SemaphoreType.DMA((2,))] * 3,
        ),
        compiler_params=pltpu.CompilerParams(collective_id=COLLECTIVE_ID),
        name=f"qar_d2d_ag-t_{t}-h_{h}",
    )


def all_reduce_v1(x: jax.Array, me: jax.Array, tab: np.ndarray) -> jax.Array:
    """v1 of `all_reduce` (TPU only): same algorithm and bits, glue in Pallas."""
    rows, h = x.shape
    hh = h // 2
    if h % (2 * 2 * 4 * 128):
        raise ValueError(f"hidden {h} must split into 16 lane-aligned chunks")
    w = hh // 8
    t = padded_rows(rows)
    if t != rows:
        x = jnp.pad(x, ((0, t - rows), (0, 0)))
    tab_j = jnp.asarray(tab)
    rank = me.reshape(1).astype(jnp.int32)
    me = rank[0]
    core = tab_j[CORE, me]
    pos = tab_j[POS, me]
    exchange = pallas_exchange(rank, tab)
    recv = _d2d_rs_call(t, h)(rank, tab_j, x)

    def blocks(k_cw, k_cc):
        # x column blocks (units of w): core*8 + d*4 + k; recv: d*4 + k.
        k_cw, k_cc = k_cw % 4, k_cc % 4
        return jnp.stack([core * 8 + k_cw, core * 8 + 4 + k_cc, k_cw, 4 + k_cc]).astype(jnp.int32)

    q_cw, s_cw, q_cc, s_cc = _step_call(t, h, w, False)(blocks(pos, pos), x, x, recv, recv)
    for i in range(3):
        prev = exchange([q_cw, s_cw, q_cc, s_cc], [CW_NEXT, CW_NEXT, CW_PREV, CW_PREV])
        q_cw, s_cw, q_cc, s_cc = _step_call(t, h, w, True)(
            blocks(pos - i - 1, pos + i + 1), x, x, recv, recv, *prev
        )
    h_cw = jnp.zeros((t, 4 * w), BF16)
    h_cc = jnp.zeros((t, 4 * w), BF16)
    put = _put_call(t, w)

    def put_blocks(k_cw, k_cc):
        return jnp.stack([k_cw % 4, k_cc % 4]).astype(jnp.int32)

    h_cw, h_cc = put(put_blocks(pos + 1, pos - 1), q_cw, s_cw, q_cc, s_cc, h_cw, h_cc)
    for i in range(3):
        q_cw, s_cw, q_cc, s_cc = exchange(
            [q_cw, s_cw, q_cc, s_cc], [CW_NEXT, CW_NEXT, CW_PREV, CW_PREV]
        )
        h_cw, h_cc = put(put_blocks(pos - i, pos + i), q_cw, s_cw, q_cc, s_cc, h_cw, h_cc)
    full = _d2d_ag_call(t, h)(rank, tab_j, h_cw, h_cc)
    return full[:rows]


# --------------------------------------------------------------------------
# v1b: the all-gather puts write straight into the full [t, h] output (column
# block core*8 + d*4 + k) and the d2d all-gather is one remote DMA of this
# core's half into the sibling's output (6 KB row segments; v1's four 3 KB
# strided writes took 1.5 ms at t = 8192).
@functools.cache
def _put_full_call(t: int, h: int, w: int):
    sr = _TILE // QROWS

    # One grid axis over the direction: both write disjoint column blocks of
    # the one aliased output.
    def kernel(blk_ref, q, s, full_in, o):
        del full_in
        o[...] = _deq_tile(q[...], s[...]).astype(BF16)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((t, h), BF16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(2, t // _TILE),
            in_specs=[
                pl.BlockSpec((None, _TILE, w), lambda d, i, blk: (d, i, 0)),
                pl.BlockSpec((None, sr, w), lambda d, i, blk: (d, i, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec((_TILE, w), lambda d, i, blk: (i, blk[d])),
        ),
        input_output_aliases={3: 0},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary", "parallel")),
        name=f"qar_put_full-t_{t}-w_{w}",
    )


@functools.cache
def _d2d_ag_full_call(t: int, h: int):
    hh = h // 2

    def kernel(rank_ref, tab_ref, full_in, o_ref, send_sem, recv_sem):
        del full_in
        me = rank_ref[0]
        barrier = pltpu.get_barrier_semaphore()
        for d in range(_N):
            pl.semaphore_signal(barrier, 1, device_id=(d,), device_id_type=_MESH)
        pl.semaphore_wait(barrier, _N)
        cols = pl.ds(pl.multiple_of(tab_ref[CORE, me] * hh, 128), hh)
        copy = pltpu.make_async_remote_copy(
            o_ref.at[:, cols], o_ref.at[:, cols], send_sem.at[0], recv_sem.at[0],
            device_id=(tab_ref[SIB, me],), device_id_type=_MESH,
        )
        copy.start()
        copy.wait_recv()
        copy.wait_send()

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((t, h), BF16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[_HBM],
            out_specs=_HBM,
            scratch_shapes=[pltpu.SemaphoreType.DMA((1,)), pltpu.SemaphoreType.DMA((1,))],
        ),
        input_output_aliases={2: 0},
        compiler_params=pltpu.CompilerParams(collective_id=COLLECTIVE_ID),
        name=f"qar_d2d_ag_full-t_{t}-h_{h}",
    )


def all_reduce_v1b(x: jax.Array, me: jax.Array, tab: np.ndarray) -> jax.Array:
    """v1 with in-place puts into the full output and a one-DMA d2d all-gather."""
    rows, h = x.shape
    hh = h // 2
    w = hh // 8
    t = padded_rows(rows)
    if t != rows:
        x = jnp.pad(x, ((0, t - rows), (0, 0)))
    tab_j = jnp.asarray(tab)
    rank = me.reshape(1).astype(jnp.int32)
    me = rank[0]
    core = tab_j[CORE, me]
    pos = tab_j[POS, me]
    exchange = pallas_exchange(rank, tab)
    recv = _d2d_rs_call(t, h)(rank, tab_j, x)

    def blocks(k_cw, k_cc):
        k_cw, k_cc = k_cw % 4, k_cc % 4
        return jnp.stack([core * 8 + k_cw, core * 8 + 4 + k_cc, k_cw, 4 + k_cc]).astype(jnp.int32)

    q_cw, s_cw, q_cc, s_cc = _step_call(t, h, w, False)(blocks(pos, pos), x, x, recv, recv)
    for i in range(3):
        prev = exchange([q_cw, s_cw, q_cc, s_cc], [CW_NEXT, CW_NEXT, CW_PREV, CW_PREV])
        q_cw, s_cw, q_cc, s_cc = _step_call(t, h, w, True)(
            blocks(pos - i - 1, pos + i + 1), x, x, recv, recv, *prev
        )
    full = jnp.zeros((t, h), BF16)
    put = _put_full_call(t, h, w)

    def put_blocks(k_cw, k_cc):
        return jnp.stack([core * 8 + k_cw % 4, core * 8 + 4 + k_cc % 4]).astype(jnp.int32)

    def put_both(full, k_cw, k_cc, q_cw, s_cw, q_cc, s_cc):
        return put(put_blocks(k_cw, k_cc), jnp.stack([q_cw, q_cc]), jnp.stack([s_cw, s_cc]), full)

    full = put_both(full, pos + 1, pos - 1, q_cw, s_cw, q_cc, s_cc)
    for i in range(3):
        q_cw, s_cw, q_cc, s_cc = exchange(
            [q_cw, s_cw, q_cc, s_cc], [CW_NEXT, CW_NEXT, CW_PREV, CW_PREV]
        )
        full = put_both(full, pos - i, pos + i, q_cw, s_cw, q_cc, s_cc)
    full = _d2d_ag_full_call(t, h)(rank, tab_j, full)
    return full[:rows]


# --------------------------------------------------------------------------
# v2: one fused call. The rows split into two halves that run the same
# algorithm interleaved, so one half's remote DMA is in flight while the other
# half's dequantize / add / quantize (an emit_pipeline over 256-row tiles)
# runs. Every ring step has its own receive buffer: nothing is reused, so no
# credits are needed, only one receive semaphore per buffer. Bits are v0's.
def _rs_body(with_prev: bool, with_put: bool):
    def body(*refs):
        x_cw, r_cw, x_cc, r_cc = refs[:4]
        rest = refs[4:]
        if with_prev:
            pq_cw, ps_cw, pq_cc, ps_cc = rest[:4]
            rest = rest[4:]
        q_cw, s_cw, q_cc, s_cc = rest[:4]
        a_cw = x_cw[...].astype(F32) + r_cw[...].astype(F32)
        a_cc = x_cc[...].astype(F32) + r_cc[...].astype(F32)
        if with_prev:
            a_cw = a_cw + _deq_tile(pq_cw[...], ps_cw[...])
            a_cc = a_cc + _deq_tile(pq_cc[...], ps_cc[...])
        qa, sa = _quant_tile(a_cw)
        qb, sb = _quant_tile(a_cc)
        q_cw[...], s_cw[...], q_cc[...], s_cc[...] = qa, sa, qb, sb
        if with_put:
            f_cw, f_cc = rest[4:6]
            f_cw[...] = _deq_tile(qa, sa).astype(BF16)
            f_cc[...] = _deq_tile(qb, sb).astype(BF16)

    return body


def _put_body(q_cw, s_cw, q_cc, s_cc, f_cw, f_cc):
    f_cw[...] = _deq_tile(q_cw[...], s_cw[...]).astype(BF16)
    f_cc[...] = _deq_tile(q_cc[...], s_cc[...]).astype(BF16)


@functools.cache
def _fused_call(t: int, h: int):
    hh = h // 2
    w = hh // 8
    th = t // 2  # rows per half
    nt = th // _TILE  # tiles per half
    sr = _TILE // QROWS
    nbuf = 3 * 2 * 2  # (step, direction, half)

    def slot(i, d, hf):
        return (i * 2 + d) * 2 + hf

    def kernel(rank_ref, tab_ref, x_ref, full, recv, sq, ss, rq, rs, fq, fs, aq, as_,
               d2d_send, d2d_recv, rs_send, rs_recv, ag_send, ag_recv, fin_send, fin_recv):
        me = rank_ref[0]
        core = tab_ref[CORE, me]
        pos = tab_ref[POS, me]
        sib = tab_ref[SIB, me]
        peer = (tab_ref[CW_NEXT, me], tab_ref[CW_PREV, me])
        barrier = pltpu.get_barrier_semaphore()
        for dv in range(_N):
            pl.semaphore_signal(barrier, 1, device_id=(dv,), device_id_type=_MESH)
        pl.semaphore_wait(barrier, _N)

        def rows(hf):
            return pl.ds(hf * th, th)

        def remote(src, dst, ssem, rsem, dev):
            return pltpu.make_async_remote_copy(src, dst, ssem, rsem, device_id=(dev,),
                                                device_id_type=_MESH)

        # 1. d2d reduce-scatter of both halves (x's other column half, strided).
        other = pl.ds(pl.multiple_of((1 - core) * hh, 128), hh)
        d2d = [remote(x_ref.at[rows(hf), other], recv.at[rows(hf)], d2d_send.at[hf],
                      d2d_recv.at[hf], sib) for hf in range(2)]
        for c in d2d:
            c.start()

        def kblk(d, k):  # chunk k of direction d: column block in units of w
            return d * 4 + lax.rem(k + 8, 4)

        def tile_spec(base_blk, col_blk, rows_per=_TILE):
            return pl.BlockSpec((rows_per, w), lambda r: (base_blk + r, col_blk))

        def buf_q(i, d, hf):
            return slot(i, d, hf) * nt

        def buf_s(i, d, hf):
            return slot(i, d, hf) * nt

        sends = list(d2d)
        rs_copies = {}

        def rs_copy(i, d, hf):
            j = slot(i, d, hf)
            qs = pl.ds(j * th, th)
            ssl = pl.ds(j * (th // QROWS), th // QROWS)
            return (remote(sq.at[qs], rq.at[qs], rs_send.at[2 * j], rs_recv.at[2 * j], peer[d]),
                    remote(ss.at[ssl], rs.at[ssl], rs_send.at[2 * j + 1], rs_recv.at[2 * j + 1], peer[d]))

        # 2. ring reduce-scatter, interleaved over the halves.
        for i in range(4):
            for hf in range(2):
                if i == 0:
                    d2d[hf].wait_recv()
                else:
                    for d in range(2):
                        for c in rs_copies[(i - 1, d, hf)]:
                            c.wait_recv()
                k = (pos - i, pos + i)  # chunk index per direction at this step
                hb = hf * nt
                in_specs = [tile_spec(hb, core * 8 + kblk(0, k[0])), tile_spec(hb, kblk(0, k[0])),
                            tile_spec(hb, core * 8 + kblk(1, k[1])), tile_spec(hb, kblk(1, k[1]))]
                args = [x_ref, recv, x_ref, recv]
                if i > 0:
                    for d in range(2):
                        j = slot(i - 1, d, hf)
                        in_specs += [pl.BlockSpec((_TILE, w), lambda r, j=j: (j * nt + r, 0)),
                                     pl.BlockSpec((sr, w), lambda r, j=j: (j * nt + r, 0))]
                        args += [rq, rs]
                if i < 3:
                    out_specs, outs = [], []
                    for d in range(2):
                        j = slot(i, d, hf)
                        out_specs += [pl.BlockSpec((_TILE, w), lambda r, j=j: (j * nt + r, 0)),
                                      pl.BlockSpec((sr, w), lambda r, j=j: (j * nt + r, 0))]
                        outs += [sq, ss]
                else:
                    out_specs, outs = [], []
                    for d in range(2):
                        j = d * 2 + hf
                        out_specs += [pl.BlockSpec((_TILE, w), lambda r, j=j: (j * nt + r, 0)),
                                      pl.BlockSpec((sr, w), lambda r, j=j: (j * nt + r, 0))]
                        outs += [fq, fs]
                    out_specs += [tile_spec(hb, core * 8 + kblk(0, k[0])),
                                  tile_spec(hb, core * 8 + kblk(1, k[1]))]
                    outs += [full, full]
                pltpu.emit_pipeline(_rs_body(i > 0, i == 3), grid=(nt,), in_specs=in_specs,
                                    out_specs=out_specs)(*args, *outs)
                if i < 3:
                    for d in range(2):
                        rs_copies[(i, d, hf)] = rs_copy(i, d, hf)
                        for c in rs_copies[(i, d, hf)]:
                            c.start()
                        sends += list(rs_copies[(i, d, hf)])
                else:  # owned chunk quantized once: start the all-gather
                    for d in range(2):
                        j = d * 2 + hf
                        qs = pl.ds(j * th, th)
                        ssl = pl.ds(j * (th // QROWS), th // QROWS)
                        g = slot(0, d, hf)
                        gq = pl.ds(g * th, th)
                        gs = pl.ds(g * (th // QROWS), th // QROWS)
                        cq = remote(fq.at[qs], aq.at[gq], fin_send.at[2 * j], ag_recv.at[2 * g], peer[d])
                        cs = remote(fs.at[ssl], as_.at[gs], fin_send.at[2 * j + 1], ag_recv.at[2 * g + 1], peer[d])
                        cq.start()
                        cs.start()
                        sends += [cq, cs]

        # 3. ring all-gather: receive, forward unchanged, put in place.
        def ag_pair(g, d, hf, dev, ssem_base):
            gq = pl.ds(g * th, th)
            gs = pl.ds(g * (th // QROWS), th // QROWS)
            return (remote(aq.at[gq], aq.at[gq], ssem_base.at[2 * g], ag_recv.at[2 * g], dev),
                    remote(as_.at[gs], as_.at[gs], ssem_base.at[2 * g + 1], ag_recv.at[2 * g + 1], dev))

        fin_sends = []
        for jstep in range(3):
            for hf in range(2):
                for d in range(2):
                    g = slot(jstep, d, hf)
                    for c in ag_pair(g, d, hf, peer[d], ag_send):
                        c.wait_recv()
                    if jstep < 2:  # forward the same bytes into the next ring slot
                        g2 = slot(jstep + 1, d, hf)
                        src_q = pl.ds(g * th, th)
                        src_s = pl.ds(g * (th // QROWS), th // QROWS)
                        dst_q = pl.ds(g2 * th, th)
                        dst_s = pl.ds(g2 * (th // QROWS), th // QROWS)
                        fq_ = remote(aq.at[src_q], aq.at[dst_q], ag_send.at[2 * g], ag_recv.at[2 * g2], peer[d])
                        fs_ = remote(as_.at[src_s], as_.at[dst_s], ag_send.at[2 * g + 1], ag_recv.at[2 * g2 + 1], peer[d])
                        fq_.start()
                        fs_.start()
                        sends += [fq_, fs_]
                k = (pos - jstep, pos + jstep)
                hb = hf * nt
                gs_ = [slot(jstep, d, hf) for d in range(2)]
                in_specs = []
                for g in gs_:
                    in_specs += [pl.BlockSpec((_TILE, w), lambda r, g=g: (g * nt + r, 0)),
                                 pl.BlockSpec((sr, w), lambda r, g=g: (g * nt + r, 0))]
                out_specs = [tile_spec(hb, core * 8 + kblk(0, k[0])),
                             tile_spec(hb, core * 8 + kblk(1, k[1]))]
                pltpu.emit_pipeline(_put_body, grid=(nt,), in_specs=in_specs,
                                    out_specs=out_specs)(aq, as_, aq, as_, full, full)
                if jstep == 2:  # half done: 4. d2d all-gather of this half
                    mine = pl.ds(pl.multiple_of(core * hh, 128), hh)
                    c = remote(full.at[rows(hf), mine], full.at[rows(hf), mine],
                               fin_send.at[8 + hf], fin_recv.at[hf], sib)
                    c.start()
                    sends.append(c)
                    fin_sends.append(c)
        for c in fin_sends:
            c.wait_recv()
        for c in sends:
            c.wait_send()

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    outs = (
        jax.ShapeDtypeStruct((t, h), BF16),                      # full
        jax.ShapeDtypeStruct((t, hh), BF16),                     # recv (d2d)
        jax.ShapeDtypeStruct((nbuf * th, w), I8),                # sq
        jax.ShapeDtypeStruct((nbuf * th // QROWS, w), F32),      # ss
        jax.ShapeDtypeStruct((nbuf * th, w), I8),                # rq
        jax.ShapeDtypeStruct((nbuf * th // QROWS, w), F32),      # rs
        jax.ShapeDtypeStruct((4 * th, w), I8),                   # fq (owned, per (d, hf))
        jax.ShapeDtypeStruct((4 * th // QROWS, w), F32),         # fs
        jax.ShapeDtypeStruct((nbuf * th, w), I8),                # aq (all-gather slots)
        jax.ShapeDtypeStruct((nbuf * th // QROWS, w), F32),      # as
    )
    return pl.pallas_call(
        kernel,
        out_shape=outs,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[hbm],
            out_specs=[hbm] * len(outs),
            scratch_shapes=[
                pltpu.SemaphoreType.DMA((2,)), pltpu.SemaphoreType.DMA((2,)),          # d2d rs
                pltpu.SemaphoreType.DMA((2 * nbuf,)), pltpu.SemaphoreType.DMA((2 * nbuf,)),  # rs
                pltpu.SemaphoreType.DMA((2 * nbuf,)), pltpu.SemaphoreType.DMA((2 * nbuf,)),  # ag
                pltpu.SemaphoreType.DMA((10,)), pltpu.SemaphoreType.DMA((2,)),          # owned AG start + d2d ag
            ],
        ),
        compiler_params=pltpu.CompilerParams(collective_id=COLLECTIVE_ID),
        name=f"qar_fused-t_{t}-h_{h}",
    )


def all_reduce_v2(x: jax.Array, me: jax.Array, tab: np.ndarray) -> jax.Array:
    """v2 of `all_reduce` (TPU only): one fused call, two interleaved row halves."""
    rows, h = x.shape
    if h % (2 * 2 * 4 * 128):
        raise ValueError(f"hidden {h} must split into 16 lane-aligned chunks")
    t = -(-rows // (2 * _TILE)) * (2 * _TILE)
    if t != rows:
        x = jnp.pad(x, ((0, t - rows), (0, 0)))
    rank = me.reshape(1).astype(jnp.int32)
    full = _fused_call(t, h)(rank, jnp.asarray(tab), x)[0]
    return full[:rows]


# --------------------------------------------------------------------------
# v3: small all-reduces (T <= 512 rows). v2's 6 ring steps and 2 row halves
# cost ~52 us of fixed latency, more than XLA's all-reduce below ~256 rows.
# v3 keeps the d2d legs but replaces the ring by a one-shot scatter: each
# chip's pair sum is quantized once per destination chunk and sent straight
# to the chunk's owner (the diagonal chip is 2 hops, routed), the owner sums
# the 4 contributions in f32, quantizes once and sends its chunk straight to
# the 3 others. 4 serial transfers instead of 8. Whole working set in VMEM.
V3_MAX_ROWS = 512
_OFFS = ((1, CW_NEXT), (3, CW_PREV), (2, DIAG))  # (ring offset, table row)


def all_reduce_v3_ref(x: jax.Array, me: jax.Array, tab: np.ndarray, exchange: Exchange) -> jax.Array:
    """v3's algorithm with pluggable data movement (CPU reference)."""
    rows, h = x.shape
    hh = h // 2
    w4 = hh // 4
    t = padded_rows(rows)
    if t != rows:
        x = jnp.pad(x, ((0, t - rows), (0, 0)))
    tab_j = jnp.asarray(tab)
    me = me.reshape(()).astype(jnp.int32)
    core, pos = tab_j[CORE, me], tab_j[POS, me]
    mine = lax.dynamic_slice_in_dim(x, core * hh, hh, axis=1)
    other = lax.dynamic_slice_in_dim(x, (1 - core) * hh, hh, axis=1)
    (recv,) = exchange([other], [SIB])
    part = mine.astype(F32) + recv.astype(F32)

    def chunk(k):
        return lax.dynamic_slice_in_dim(part, ((pos + k) % 4) * w4, w4, axis=1)

    sends = []
    for off, _ in _OFFS:
        sends += list(quantize(chunk(off)))
    got = exchange(sends, [r for _, r in _OFFS for _ in (0, 1)])
    acc = chunk(0)
    for j in range(3):
        acc = acc + dequantize(got[2 * j], got[2 * j + 1])
    q, s_ = quantize(acc)
    ag = exchange([q, s_] * 3, [r for _, r in _OFFS for _ in (0, 1)])
    half = jnp.zeros((t, hh), F32)
    half = lax.dynamic_update_slice_in_dim(half, dequantize(q, s_), pos * w4, axis=1)
    for j, (off, _) in enumerate(_OFFS):
        # what arrived from offset `off` was sent by the chip at pos - off,
        # which owns chunk pos - off
        half = lax.dynamic_update_slice_in_dim(
            half, dequantize(ag[2 * j], ag[2 * j + 1]), ((pos - off) % 4) * w4, axis=1)
    half = half.astype(BF16)
    (sib_half,) = exchange([half], [SIB])
    full = jnp.zeros((t, h), BF16)
    full = lax.dynamic_update_slice_in_dim(full, half, core * hh, axis=1)
    full = lax.dynamic_update_slice_in_dim(full, sib_half, (1 - core) * hh, axis=1)
    return full[:rows]


@functools.cache
def _v3_call(t: int, h: int):
    hh, w4, sr = h // 2, h // 8, t // QROWS
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)

    def kernel(rank_ref, tab_ref, x_ref, out_ref, rd2d, sq, ss, rq, rs, oq, os_, aq, as_,
               d2d_s, d2d_r, sc_s, sc_r, ag_s, ag_r, fin_s, fin_r):
        me = rank_ref[0]
        core, pos, sib = tab_ref[CORE, me], tab_ref[POS, me], tab_ref[SIB, me]
        peers = [tab_ref[row, me] for _, row in _OFFS]
        barrier = pltpu.get_barrier_semaphore()
        for d in range(_N):
            pl.semaphore_signal(barrier, 1, device_id=(d,), device_id_type=_MESH)
        pl.semaphore_wait(barrier, _N)

        def remote(src, dst, ssem, rsem, dev):
            return pltpu.make_async_remote_copy(src, dst, ssem, rsem, device_id=(dev,),
                                                device_id_type=_MESH)

        mine = pl.ds(pl.multiple_of(core * hh, 128), hh)
        other = pl.ds(pl.multiple_of((1 - core) * hh, 128), hh)
        sends = []
        # 1. d2d reduce-scatter
        c = remote(x_ref.at[:, other], rd2d, d2d_s.at[0], d2d_r.at[0], sib)
        c.start()
        sends.append(c)
        c.wait_recv()

        def part(k):  # f32 pair sum of chunk (pos + k) % 4 of this core's half
            col = pl.multiple_of(lax.rem(pos + k, 4) * w4, 128)
            return (x_ref[:, pl.ds(pl.multiple_of(core * hh + col, 128), w4)].astype(F32)
                    + rd2d[:, pl.ds(col, w4)].astype(F32))

        # 2. one-shot scatter of the 3 non-owned chunks
        for j, (off, _) in enumerate(_OFFS):
            sq[j], ss[j] = _quant_tile(part(off))
            cq = remote(sq.at[j], rq.at[j], sc_s.at[2 * j], sc_r.at[2 * j], peers[j])
            cs = remote(ss.at[j], rs.at[j], sc_s.at[2 * j + 1], sc_r.at[2 * j + 1], peers[j])
            cq.start()
            cs.start()
            sends += [cq, cs]
        # 3. owner: sum, quantize once, gather
        acc = part(0)
        for j in range(3):
            remote(sq.at[j], rq.at[j], sc_s.at[2 * j], sc_r.at[2 * j], peers[j]).wait_recv()
            remote(ss.at[j], rs.at[j], sc_s.at[2 * j + 1], sc_r.at[2 * j + 1], peers[j]).wait_recv()
            acc = acc + _deq_tile(rq[j], rs[j])
        oq[...], os_[...] = _quant_tile(acc)
        for j in range(3):
            cq = remote(oq, aq.at[j], ag_s.at[2 * j], ag_r.at[2 * j], peers[j])
            cs = remote(os_, as_.at[j], ag_s.at[2 * j + 1], ag_r.at[2 * j + 1], peers[j])
            cq.start()
            cs.start()
            sends += [cq, cs]
        own = pl.multiple_of(core * hh + pos * w4, 128)
        out_ref[:, pl.ds(own, w4)] = _deq_tile(oq[...], os_[...]).astype(BF16)
        for j, (off, _) in enumerate(_OFFS):
            remote(oq, aq.at[j], ag_s.at[2 * j], ag_r.at[2 * j], peers[j]).wait_recv()
            remote(os_, as_.at[j], ag_s.at[2 * j + 1], ag_r.at[2 * j + 1], peers[j]).wait_recv()
            col = pl.multiple_of(core * hh + lax.rem(pos - off + 4, 4) * w4, 128)
            out_ref[:, pl.ds(col, w4)] = _deq_tile(aq[j], as_[j]).astype(BF16)
        # 4. d2d all-gather of this core's half into the sibling's output
        c = remote(out_ref.at[:, mine], out_ref.at[:, mine], fin_s.at[0], fin_r.at[0], sib)
        c.start()
        sends.append(c)
        c.wait_recv()
        for c in sends:
            c.wait_send()

    work = t * h * 2 * 2 + t * hh * 2 + 13 * t * w4 + 13 * sr * w4 * 4
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((t, h), BF16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[vmem],
            out_specs=vmem,
            scratch_shapes=[
                pltpu.VMEM((t, hh), BF16),                                   # rd2d
                pltpu.VMEM((3, t, w4), I8), pltpu.VMEM((3, sr, w4), F32),    # scatter send
                pltpu.VMEM((3, t, w4), I8), pltpu.VMEM((3, sr, w4), F32),    # scatter recv
                pltpu.VMEM((t, w4), I8), pltpu.VMEM((sr, w4), F32),          # owned
                pltpu.VMEM((3, t, w4), I8), pltpu.VMEM((3, sr, w4), F32),    # gather recv
                pltpu.SemaphoreType.DMA((1,)), pltpu.SemaphoreType.DMA((1,)),
                pltpu.SemaphoreType.DMA((6,)), pltpu.SemaphoreType.DMA((6,)),
                pltpu.SemaphoreType.DMA((6,)), pltpu.SemaphoreType.DMA((6,)),
                pltpu.SemaphoreType.DMA((1,)), pltpu.SemaphoreType.DMA((1,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID, vmem_limit_bytes=int(work * 1.5) + (8 << 20)
        ),
        name=f"qar_small-t_{t}-h_{h}",
    )


def all_reduce_v3(x: jax.Array, me: jax.Array, tab: np.ndarray) -> jax.Array:
    """v3 (TPU, VMEM-resident one-shot scatter) for rows <= V3_MAX_ROWS; bits are
    all_reduce_v3_ref's."""
    rows, h = x.shape
    if h % (2 * 4 * 128):
        raise ValueError(f"hidden {h} must split into 8 lane-aligned chunks")
    t = padded_rows(rows)
    if t > V3_MAX_ROWS:
        raise ValueError(f"all_reduce_v3 takes <= {V3_MAX_ROWS} rows (padded {t})")
    if t != rows:
        x = jnp.pad(x, ((0, t - rows), (0, 0)))
    rank = me.reshape(1).astype(jnp.int32)
    return _v3_call(t, h)(rank, jnp.asarray(tab), x)[:rows]

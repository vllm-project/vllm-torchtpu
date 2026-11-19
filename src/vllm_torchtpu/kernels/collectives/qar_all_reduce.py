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
SIB, CW_NEXT, CW_PREV, POS, CORE = range(5)


def layout_table(devices: Sequence) -> np.ndarray:
    """int32 [5, 8] over the mesh order: sibling core, clockwise ring next and
    previous (same core index on the neighbouring chip), ring position and
    core index. The chip ring is (x0,y0) -> (x1,y0) -> (x1,y1) -> (x0,y1)."""
    if len(devices) != _N:
        raise ValueError(f"qar_all_reduce needs 8 TensorCores, got {len(devices)}")
    chips = sorted({tuple(d.coords[:2]) for d in devices})
    if len(chips) != 4 or {d.core_on_chip for d in devices} != {0, 1}:
        raise ValueError(f"qar_all_reduce needs 4 chips x 2 cores: {chips}")
    xs = sorted({c[0] for c in chips})
    ys = sorted({c[1] for c in chips})
    ring = [(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])]
    index = {(tuple(d.coords[:2]), d.core_on_chip): i for i, d in enumerate(devices)}
    tab = np.zeros((5, _N), np.int32)
    for i, d in enumerate(devices):
        chip, core = tuple(d.coords[:2]), d.core_on_chip
        p = ring.index(chip)
        tab[SIB, i] = index[(chip, 1 - core)]
        tab[CW_NEXT, i] = index[(ring[(p + 1) % 4], core)]
        tab[CW_PREV, i] = index[(ring[(p - 1) % 4], core)]
        tab[POS, i] = p
        tab[CORE, i] = core
    return tab


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
            collective_id=COLLECTIVE_ID, has_side_effects=True
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

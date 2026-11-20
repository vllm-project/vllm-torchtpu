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
"""The whole-step call's pool side of the GDN window log (VLLM_TPU_GDN_WINDOW_LOG).

Pool layout of a window-log checkpoint block 0 (manager block rows, 1 KiB each;
the stock kernel's `pool_adapters.v3_state_source(log_positions=K)` regions):

* rows 0..383: f32 recurrent state `[vh, KD, VD]` after window position 0;
* rows 384..399: f32 conv history `[3, QKV]` after window position 0 (15 rows + pad);
* rows 400..413: the log's typed rows (`megakernel/blocks/gdn/window_log.py`), two
  128-lane rows per 256-lane pool row;
* rows 414..428: the conv inputs `[K, QKV]` of draft positions 1..K.

`fold_gdn_window_log` runs before the kernel: for each live request with a read
offset `r > 0` it applies the first `r` logged positions to checkpoint 0 and writes
the result over the block's state and conv rows, so the kernel reads block 0 as
a full checkpoint (`gdn_blocks(window_log=True)` names block 0 and passes the
offsets). The log rows are left as they are; the step's `scatter_gdn_log_fast`
rewrites the whole block (new checkpoint 0 + log) for every live request.

The fold is `fold_window_log`'s arithmetic (state * g, then + k^T v_new; the conv
window shifts in `r` inputs) in f32.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.blocks.gdn.window_log import LOG_ROWS, log_rows_used
from vllm_torchtpu.megakernel.models.qwen38_gdn_pool import (
    CONV_ROWS,
    SSM_BLOCKS,
    SSM_ROWS,
    pool_rows,
)

from .qwen38_gdn_state import Qwen38GdnPoolLayout

LOG_ROW0 = SSM_BLOCKS * SSM_ROWS + CONV_ROWS  # 400
RING, PREFETCH = 4, 2


def _regions(layout: Qwen38GdnPoolLayout, K: int) -> tuple[int, int, int, int]:
    """`(log_used, log_rows, xin_row0, xin_rows)`: typed log rows, their pool rows, and
    the conv-input region's first pool row and pool row count."""
    *_, used = log_rows_used(K, layout.n_k, layout.n_v)
    assert used <= LOG_ROWS, (used, LOG_ROWS)
    log_rows = -(-used * layout.d_v * 4 // 1024)
    xin_rows = -(-K * layout.conv_dim * 4 // 1024)
    xin_row0 = LOG_ROW0 + log_rows
    assert xin_row0 + xin_rows <= layout.manager_block_tokens, (xin_row0, xin_rows)
    return used, log_rows, xin_row0, xin_rows


def _view(ref: Any, lanes: int, interpret: bool) -> Any:
    """f32 values `(-1, lanes)` of a whole FP8 byte scratch ref `[rows, 1, 4, 256]`."""
    if interpret:
        raw = ref[...]
        return pltpu.bitcast(raw.reshape(-1, raw.shape[-1]), jnp.float32).reshape(-1, lanes)
    return ref.bitcast(jnp.float32).reshape(-1, lanes)[...]


def _store(ref: Any, vals: Any, interpret: bool) -> None:
    """f32 `vals` `(-1, lanes)` (row-major) -> whole FP8 byte scratch ref."""
    if interpret:
        words = vals.reshape(-1, ref.shape[-1])
        ref[...] = pltpu.bitcast(words, ref.dtype).reshape(ref.shape)
        return
    ref.bitcast(jnp.float32).reshape(-1, vals.shape[-1])[...] = vals


def _conv_rows(cv: Any, r0: int, n: int, per: int) -> Any:
    """Conv rows `r0 .. r0 + n` of `cv [*, QKV]` as `n * per` 256-lane rows (row-major)."""
    return jnp.concatenate(
        [cv[r : r + 1, c * 256 : (c + 1) * 256] for r in range(r0, r0 + n) for c in range(per)],
        axis=0,
    )


def fold_gdn_window_log(
    pool: Any,
    read_blocks: Any,
    read_offsets: Any,
    *,
    layout: Qwen38GdnPoolLayout,
    K: int,
    interpret: bool = False,
) -> Any:
    """Fold each `(read_blocks, read_offsets)` [NL, NS] block's log in place (per rank)."""
    vh, KD, VD, kh = layout.n_v, layout.d_k, layout.d_v, layout.n_k
    QKV = layout.conv_dim
    per = QKV // 256
    expand = vh // kh
    keys0, upd0, _ = log_rows_used(K, kh, vh)
    _, LR, XR0, XR = _regions(layout, K)
    assert per * 3 <= CONV_ROWS and K * per == XR, (per, XR)
    ids = read_blocks.reshape(-1).astype(jnp.int32)
    ro = read_offsets.reshape(-1).astype(jnp.int32)
    n = ids.shape[0]
    active = (ids != 0) & (ro > 0)
    order = jnp.argsort(jnp.logical_not(active).astype(jnp.int32), stable=True)
    ids, ro = ids[order], ro[order]
    cnt = jnp.sum(active.astype(jnp.int32)).reshape(1)
    f8 = pool.dtype

    def cols(x):  # [kh, KD] -> [vh, KD, VD], value head h <- key head h // expand
        xc = jnp.swapaxes(jnp.broadcast_to(x[:, None, :], (kh, VD, KD)), 1, 2)
        return jnp.broadcast_to(xc[:, None], (kh, expand, KD, VD)).reshape(vh, KD, VD)

    def kernel(ids_ref, ro_ref, cnt_ref, pool_in, pool_out, vb, cb, lb, xb, sin, sout):
        count = cnt_ref[0]

        def in_copies(i, slot):
            m = ids_ref[i]
            cps = [
                pltpu.make_async_copy(
                    pool_rows(pool_in, m, j * SSM_ROWS, SSM_ROWS), vb.at[slot, j], sin.at[slot, j]
                )
                for j in range(SSM_BLOCKS)
            ]
            cps += [
                pltpu.make_async_copy(
                    pool_rows(pool_in, m, SSM_BLOCKS * SSM_ROWS, CONV_ROWS),
                    cb.at[slot],
                    sin.at[slot, SSM_BLOCKS],
                ),
                pltpu.make_async_copy(
                    pool_rows(pool_in, m, LOG_ROW0, LR), lb.at[slot], sin.at[slot, SSM_BLOCKS + 1]
                ),
                pltpu.make_async_copy(
                    pool_rows(pool_in, m, XR0, XR), xb.at[slot], sin.at[slot, SSM_BLOCKS + 2]
                ),
            ]
            return cps

        def out_copies(slot, m):
            cps = [
                pltpu.make_async_copy(
                    vb.at[slot, j], pool_rows(pool_out, m, j * SSM_ROWS, SSM_ROWS), sout.at[slot, j]
                )
                for j in range(SSM_BLOCKS)
            ]
            cps.append(
                pltpu.make_async_copy(
                    cb.at[slot],
                    pool_rows(pool_out, m, SSM_BLOCKS * SSM_ROWS, CONV_ROWS),
                    sout.at[slot, SSM_BLOCKS],
                )
            )
            return cps

        for k in range(min(PREFETCH, n)):

            @pl.when(k < count)
            def _(k=k):
                for c in in_copies(k, k):
                    c.start()

        def body(i, carry):
            slot = lax.rem(i, RING)
            ahead = i + PREFETCH
            nslot = lax.rem(ahead, RING)
            prev = jnp.clip(ahead - RING, 0, n - 1)

            @pl.when((ahead >= RING) & (ahead - RING < count))
            def _():
                for c in out_copies(nslot, ids_ref[prev]):
                    c.wait()

            @pl.when(ahead < count)
            def _():
                for c in in_copies(jnp.minimum(ahead, n - 1), nslot):
                    c.start()

            @pl.when(i < count)
            def _():
                for c in in_copies(i, slot):
                    c.wait()
                r = ro_ref[i]
                st = jnp.concatenate(
                    [_view(vb.at[slot, j], VD, interpret) for j in range(SSM_BLOCKS)], axis=0
                )[: vh * KD].reshape(vh, KD, VD)
                lg = _view(lb.at[slot], VD, interpret)
                for j in range(K):
                    g = lg[j, :vh]
                    kk = lg[keys0 + j * kh : keys0 + (j + 1) * kh, :]
                    u = lg[upd0 + j * vh : upd0 + (j + 1) * vh, :]
                    new = st * g[:, None, None] + cols(kk) * u[:, None, :]
                    st = jnp.where(j < r, new, st)
                flat = st.reshape(vh * KD, VD)
                rows = flat.shape[0] // SSM_BLOCKS
                for j in range(SSM_BLOCKS):
                    _store(vb.at[slot, j], flat[j * rows : (j + 1) * rows], interpret)
                cw = _view(cb.at[slot], 256, interpret)[: 3 * per]
                hist = jnp.concatenate([cw, _view(xb.at[slot], 256, interpret)], axis=0)
                win = cw
                for rr in range(1, K + 1):
                    win = jnp.where(r == rr, hist[rr * per : rr * per + 3 * per], win)
                pad = jnp.zeros((CONV_ROWS - 3 * per, 256), jnp.float32)
                _store(cb.at[slot], jnp.concatenate([win, pad], axis=0), interpret)
                for c in out_copies(slot, ids_ref[i]):
                    c.start()

            return carry

        lax.fori_loop(0, n, body, 0)
        for k in range(max(0, n - RING + PREFETCH), n):

            @pl.when(k < count)
            def _(k=k):
                for c in out_copies(k % RING, ids_ref[k]):
                    c.wait()

    any_ = pltpu.HBM
    tail = tuple(pool.shape[2:])
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(),
            in_specs=[pl.BlockSpec(memory_space=any_)],
            out_specs=pl.BlockSpec(memory_space=any_),
            scratch_shapes=[
                pltpu.VMEM((RING, SSM_BLOCKS, SSM_ROWS) + tail, f8),
                pltpu.VMEM((RING, CONV_ROWS) + tail, f8),
                pltpu.VMEM((RING, LR) + tail, f8),
                pltpu.VMEM((RING, XR) + tail, f8),
                pltpu.SemaphoreType.DMA((RING, SSM_BLOCKS + 3)),
                pltpu.SemaphoreType.DMA((RING, SSM_BLOCKS + 1)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={3: 0},
        interpret=interpret,
    )(ids, ro, cnt, pool)


def scatter_gdn_log_fast(
    pool: Any,
    conv_ckpt: Any,
    rec_ckpt: Any,
    log_ckpt: Any,
    write_blocks: Any,
    *,
    layout: Qwen38GdnPoolLayout,
    T: int,
    interpret: bool = False,
) -> Any:
    """Write checkpoint 0 + draft log + conv inputs of each request into its block 0.

    `write_blocks [NL, NS]` (0 = no write), `conv_ckpt [NL, NS, HR, QKV]` (rows 0..2
    the old history, 3 .. 3 + T the window's inputs), `rec_ckpt [NL, NS * T * vh, KD,
    VD]`, `log_ckpt [NL, NS, LOG_ROWS, VD]` f32 (per rank)."""
    NL, NS = write_blocks.shape
    n = NL * NS
    vh, KD, VD = layout.n_v, layout.d_k, layout.d_v
    QKV = layout.conv_dim
    per = QKV // 256
    K = T - 1
    used, LR, XR0, XR = _regions(layout, K)
    HR = conv_ckpt.shape[-2]
    assert HR >= 3 + T, (HR, T)
    rec = rec_ckpt.reshape(n, T * vh, KD, VD)
    conv = conv_ckpt.reshape(n, HR, QKV)
    log = log_ckpt.reshape(n, LOG_ROWS, VD)
    ids = write_blocks.reshape(-1).astype(jnp.int32)
    ring = 16 if n >= 64 else RING
    P = ring - 4 if n >= 64 else 1
    f8 = pool.dtype

    def kernel(ids_ref, rec_ref, conv_ref, log_ref, pool_in, pool_out,
               vin, cin, lin, vout, cout, lout, xout, sin, sout):
        del pool_in

        def in_copies(i, slot):
            return (
                pltpu.make_async_copy(rec_ref.at[i, pl.ds(0, vh)], vin.at[slot], sin.at[slot, 0]),
                pltpu.make_async_copy(conv_ref.at[i], cin.at[slot], sin.at[slot, 1]),
                pltpu.make_async_copy(log_ref.at[i], lin.at[slot], sin.at[slot, 2]),
            )

        def out_copies(slot, m):
            cps = [
                pltpu.make_async_copy(
                    vout.at[slot, j], pool_rows(pool_out, m, j * SSM_ROWS, SSM_ROWS), sout.at[slot, j]
                )
                for j in range(SSM_BLOCKS)
            ]
            cps += [
                pltpu.make_async_copy(
                    cout.at[slot],
                    pool_rows(pool_out, m, SSM_BLOCKS * SSM_ROWS, CONV_ROWS),
                    sout.at[slot, SSM_BLOCKS],
                ),
                pltpu.make_async_copy(
                    lout.at[slot], pool_rows(pool_out, m, LOG_ROW0, LR), sout.at[slot, SSM_BLOCKS + 1]
                ),
                pltpu.make_async_copy(
                    xout.at[slot], pool_rows(pool_out, m, XR0, XR), sout.at[slot, SSM_BLOCKS + 2]
                ),
            ]
            return cps

        for k in range(min(P, n)):
            for c in in_copies(k, k):
                c.start()

        def body(i, carry):
            slot = lax.rem(i, ring)
            ahead = i + P
            nslot = lax.rem(ahead, ring)
            prev = ids_ref[jnp.clip(ahead - ring, 0, n - 1)]

            @pl.when((ahead >= ring) & (ahead < n + ring) & (prev != 0))
            def _():
                for c in out_copies(nslot, prev):
                    c.wait()

            @pl.when(ahead < n)
            def _():
                for c in in_copies(jnp.minimum(ahead, n - 1), nslot):
                    c.start()

            for c in in_copies(i, slot):
                c.wait()
            vals = vin[slot].astype(jnp.float32).reshape(vh * KD, VD)
            rows = vals.shape[0] // SSM_BLOCKS
            for j in range(SSM_BLOCKS):
                _store(vout.at[slot, j], vals[j * rows : (j + 1) * rows], interpret)
            cv = cin[slot].astype(jnp.float32)
            # Checkpoint 0's conv history = rows 1..3; the draft inputs = rows 4 .. 3 + T.
            crow = jnp.concatenate(
                [_conv_rows(cv, 1, 3, per), jnp.zeros((CONV_ROWS - 3 * per, 256), jnp.float32)],
                axis=0,
            )
            _store(cout.at[slot], crow, interpret)
            _store(xout.at[slot], _conv_rows(cv, 4, K, per), interpret)
            lg = lin[slot]
            _store(lout.at[slot], lg[: 2 * LR], interpret)

            @pl.when(ids_ref[i] != 0)
            def _():
                for c in out_copies(slot, ids_ref[i]):
                    c.start()

            return carry

        lax.fori_loop(0, n, body, 0)
        for k in range(max(0, n - ring + P), n):

            @pl.when(ids_ref[k] != 0)
            def _(k=k):
                for c in out_copies(k % ring, ids_ref[k]):
                    c.wait()

    any_ = pltpu.HBM
    tail = tuple(pool.shape[2:])
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(),
            in_specs=[pl.BlockSpec(memory_space=any_)] * 4,
            out_specs=pl.BlockSpec(memory_space=any_),
            scratch_shapes=[
                pltpu.VMEM((ring, vh, KD, VD), rec.dtype),
                pltpu.VMEM((ring, HR, QKV), conv.dtype),
                pltpu.VMEM((ring, LOG_ROWS, VD), jnp.float32),
                pltpu.VMEM((ring, SSM_BLOCKS, SSM_ROWS) + tail, f8),
                pltpu.VMEM((ring, CONV_ROWS) + tail, f8),
                pltpu.VMEM((ring, LR) + tail, f8),
                pltpu.VMEM((ring, XR) + tail, f8),
                pltpu.SemaphoreType.DMA((ring, 3)),
                pltpu.SemaphoreType.DMA((ring, SSM_BLOCKS + 3)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={4: 0},
        interpret=interpret,
    )(ids, rec, conv, log, pool)

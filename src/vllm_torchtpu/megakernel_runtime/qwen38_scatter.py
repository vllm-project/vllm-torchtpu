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
"""Write Qwen3.8 GDN verify checkpoints into the unified pool with manual DMAs.

`scatter_gdn_ckpts_fast` has the contract of `qwen38_gdn_state.scatter_gdn_ckpts`
but writes only the bytes that change: per checkpoint the 384 rows of the f32
recurrent state and the 16 conv rows after them (any kernel block size), into the
aliased pool (HBM refs, no BlockSpec windows), through a ring of VMEM
staging buffers so several copies are in flight. The region adapters instead
rewrite whole 512-token manager blocks once per region (full-write rule).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.models.qwen38_gdn_pool import (
    CONV_ROWS,
    SSM_BLOCKS,
    SSM_ROWS,
    pool_rows,
)

from .qwen38_gdn_state import Qwen38GdnPoolLayout, _conv_windows

RING = 4


def _store_typed(ref: Any, vals: Any, interpret: bool) -> None:
    """f32 `vals` (row-major, 128 lanes) -> whole FP8 byte ref `[rows, 1, 4, 256]`."""
    if interpret:
        rows, _, pack, lanes = ref.shape
        words = vals.reshape(-1, lanes)
        ref[...] = pltpu.bitcast(words, ref.dtype).reshape(ref.shape)
        return
    view = ref.bitcast(jnp.float32)
    view.reshape(-1, vals.shape[-1])[...] = vals


def scatter_gdn_ckpts_fast(
    pool: Any,
    conv_ckpt: Any,
    rec_ckpt: Any,
    write_blocks: Any,
    *,
    layout: Qwen38GdnPoolLayout,
    interpret: bool = False,
    ring: int = RING,
    prefetch: int = 1,
) -> Any:
    """Same contract as `scatter_gdn_ckpts` (per-rank arrays); returns the pool.

    `ring` staging slots: checkpoints `i + 1 .. i + prefetch` are loaded while `i` is
    converted, and up to `ring` pool writes are in flight (`prefetch < ring`; 1 keeps one
    load in flight, which leaves each step waiting on its own load's latency). Checkpoints addressed to the
    null block 0 (rows with no request, every row of a shadow call) are not
    written, as the stock pooled GDN kernel moves no bytes for padded rows:
    the null block's pages are also read (masked) as KV, and f32 state bytes
    there would carry FP8 NaN encodings.
    """
    RING_ = ring
    P = prefetch
    assert 1 <= P < RING_, (prefetch, ring)
    NL, NS, T = write_blocks.shape
    n = NL * NS * T
    vh, KD, VD, QKV = layout.n_v, layout.d_k, layout.d_v, layout.conv_dim
    rec = rec_ckpt.reshape(n, vh, KD, VD)  # row block (l, s, t), [layout: (s*T+t)*vh]
    win = _conv_windows(conv_ckpt.astype(jnp.float32), T).reshape(n, 3, QKV)
    ids = write_blocks.reshape(-1).astype(jnp.int32)
    per = QKV // 256
    f8 = pool.dtype

    def kernel(ids_ref, rec_ref, win_ref, pool_in, pool_out, vin, cin, vout, cout, sin, sout):
        del pool_in

        def out_copies(slot, m):
            cps = [
                pltpu.make_async_copy(
                    vout.at[slot, j],
                    pool_rows(pool_out, m, j * SSM_ROWS, SSM_ROWS),
                    sout.at[slot, j],
                )
                for j in range(SSM_BLOCKS)
            ]
            cps.append(
                pltpu.make_async_copy(
                    cout.at[slot],
                    pool_rows(pool_out, m, SSM_BLOCKS * SSM_ROWS, CONV_ROWS),
                    sout.at[slot, SSM_BLOCKS],
                )
            )
            return cps

        def in_copies(i, slot):
            return (
                pltpu.make_async_copy(rec_ref.at[i], vin.at[slot], sin.at[slot, 0]),
                pltpu.make_async_copy(win_ref.at[i], cin.at[slot], sin.at[slot, 1]),
            )

        for k in range(min(P, n)):
            for c in in_copies(k, k):
                c.start()

        def body(i, carry):
            slot = lax.rem(i, RING_)
            ahead = i + P
            nslot = lax.rem(ahead, RING_)

            # Slot `nslot` is reused by checkpoint i + P: its write from
            # i + P - RING must have landed before the prefetch overwrites it.
            prev = ids_ref[jnp.clip(ahead - RING_, 0, n - 1)]

            @pl.when((ahead >= RING_) & (ahead < n + RING_) & (prev != 0))
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
                _store_typed(vout.at[slot, j], vals[j * rows : (j + 1) * rows], interpret)
            cv = cin[slot]
            # 256-lane rows of the conv history in row-major order, padded to 16.
            crow = jnp.concatenate(
                [cv[r : r + 1, c * 256 : (c + 1) * 256] for r in range(3) for c in range(per)]
                + [jnp.zeros((CONV_ROWS - 3 * per, 256), jnp.float32)],
                axis=0,
            )
            _store_typed(cout.at[slot], crow, interpret)

            @pl.when(ids_ref[i] != 0)
            def _():
                for c in out_copies(slot, ids_ref[i]):
                    c.start()

            return carry

        lax.fori_loop(0, n, body, 0)
        # Writes not yet waited on: the last RING - P checkpoints (the loop waits
        # for i + P - RING when it reuses a slot).
        for k in range(max(0, n - RING_ + P), n):

            @pl.when(ids_ref[k] != 0)
            def _():
                for c in out_copies(k % RING_, ids_ref[k]):
                    c.wait()

    any_ = pltpu.HBM
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(),
            in_specs=[
                pl.BlockSpec(memory_space=any_),
                pl.BlockSpec(memory_space=any_),
                pl.BlockSpec(memory_space=any_),
            ],
            out_specs=pl.BlockSpec(memory_space=any_),
            scratch_shapes=[
                pltpu.VMEM((RING_, vh, KD, VD), rec.dtype),
                pltpu.VMEM((RING_, 3, QKV), jnp.float32),
                pltpu.VMEM((RING_, SSM_BLOCKS, SSM_ROWS) + tuple(pool.shape[2:]), f8),
                pltpu.VMEM((RING_, CONV_ROWS) + tuple(pool.shape[2:]), f8),
                pltpu.SemaphoreType.DMA((RING_, 2)),
                pltpu.SemaphoreType.DMA((RING_, SSM_BLOCKS + 1)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={3: 0},
        interpret=interpret,
    )(ids, rec, win, pool)

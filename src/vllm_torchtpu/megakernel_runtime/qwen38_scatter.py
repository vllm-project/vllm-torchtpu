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
but writes only the bytes that change: per checkpoint the 3 kernel blocks of
the f32 recurrent state and the 16 conv rows of the 4th block, into the
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

from .qwen38_gdn_state import Qwen38GdnPoolLayout, _conv_windows

RING = 4
SSM_BLOCKS = 3
CONV_ROWS = 16


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
) -> Any:
    """Same contract as `scatter_gdn_ckpts` (per-rank arrays); returns the pool."""
    NL, NS, T = write_blocks.shape
    n = NL * NS * T
    vh, KD, VD, QKV = layout.n_v, layout.d_k, layout.d_v, layout.conv_dim
    split = layout.split
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
                    vout.at[slot, j], pool_out.at[m * split + j], sout.at[slot, j]
                )
                for j in range(SSM_BLOCKS)
            ]
            cps.append(
                pltpu.make_async_copy(
                    cout.at[slot],
                    pool_out.at[m * split + SSM_BLOCKS, pl.ds(0, CONV_ROWS)],
                    sout.at[slot, SSM_BLOCKS],
                )
            )
            return cps

        def body(i, carry):
            slot = lax.rem(i, RING)

            @pl.when(i >= RING)
            def _():
                for c in out_copies(slot, ids_ref[i - RING]):
                    c.wait()

            a = pltpu.make_async_copy(rec_ref.at[i], vin.at[slot], sin.at[slot, 0])
            b = pltpu.make_async_copy(win_ref.at[i], cin.at[slot], sin.at[slot, 1])
            a.start()
            b.start()
            a.wait()
            b.wait()
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
            for c in out_copies(slot, ids_ref[i]):
                c.start()
            return carry

        lax.fori_loop(0, n, body, 0)
        for k in range(min(RING, n)):
            i = n - min(RING, n) + k
            for c in out_copies(i % RING, ids_ref[i]):
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
                pltpu.VMEM((RING, vh, KD, VD), rec.dtype),
                pltpu.VMEM((RING, 3, QKV), jnp.float32),
                pltpu.VMEM((RING, SSM_BLOCKS) + tuple(pool.shape[1:]), f8),
                pltpu.VMEM((RING, CONV_ROWS) + tuple(pool.shape[2:]), f8),
                pltpu.SemaphoreType.DMA((RING, 2)),
                pltpu.SemaphoreType.DMA((RING, SSM_BLOCKS + 1)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={3: 0},
        interpret=interpret,
    )(ids, rec, win, pool)

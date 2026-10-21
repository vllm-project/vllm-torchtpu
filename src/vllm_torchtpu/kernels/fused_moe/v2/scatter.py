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
"""Single SparseCore, disjoint subcore output ownership; exact i32 add/drop."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.compute_on import compute_on
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

_SCATTER_CHUNK = 128


def _scratch_shape(n: int, output_size: int) -> tuple[int, int, int]:
    info = plsc.get_sparse_core_info()
    aligned_n = ((n + _SCATTER_CHUNK - 1) // _SCATTER_CHUNK) * _SCATTER_CHUNK
    alignment = info.dma_granule_size_bytes // jnp.dtype(jnp.int32).itemsize
    partitions = info.num_subcores * alignment
    width = ((output_size + partitions - 1) // partitions) * alignment
    return aligned_n, width, info.num_subcores


def can_scatter_pair(n: int, token_size: int, scale_size: int) -> bool:
    """Keep explicit scratch within half of subcore VMEM for compiler headroom."""
    aligned_n, width, _ = _scratch_shape(n, max(token_size, scale_size))
    scratch_bytes = (3 * aligned_n + 2 * width) * jnp.dtype(jnp.int32).itemsize
    return (
        n > 0 and scratch_bytes <= plsc.get_sparse_core_info().vmem_capacity_bytes // 2
    )


def scatter_pair(
    rows: jax.Array,
    scales: jax.Array,
    token_size: int,
    scale_size: int,
    topk: int,
) -> tuple[jax.Array, jax.Array]:
    """Scatter token IDs and scale bits from nonnegative local slab rows.

    Caller checks scratch capacity with can_scatter_pair.
    Each subcore owns disjoint output rows; duplicate updates add modulo
    int32, and off-shard sentinel rows are dropped. Scale padding is retained.
    """
    n = rows.size
    aligned_n, width, subcores = _scratch_shape(n, max(token_size, scale_size))
    tokens = jnp.arange(n, dtype=jnp.int32) // topk
    padding = aligned_n - n
    rows = jnp.pad(rows, (0, padding), constant_values=max(token_size, scale_size))
    scales = jnp.pad(scales, (0, padding))
    tokens = jnp.pad(tokens, (0, padding))
    size = width * subcores

    def body(r, s, tokens, ot, os, rv, sv, tokens_v, tv, ov, sem):
        pltpu.async_copy(r, rv, sem).wait()
        pltpu.async_copy(s, sv, sem).wait()
        tv[...] = jnp.zeros((width,), jnp.int32)
        ov[...] = jnp.zeros((width,), jnp.int32)
        pltpu.async_copy(tokens, tokens_v, sem).wait()
        base = jax.lax.axis_index("subcore") * width

        @pl.loop(0, aligned_n // _SCATTER_CHUNK)
        def accumulate(i):
            r_chunk = rv[pl.ds(i * _SCATTER_CHUNK, _SCATTER_CHUNK)]
            idx = r_chunk - base
            owned = (idx >= 0) & (idx < width)
            safe = jnp.clip(idx, 0, width - 1)
            plsc.addupdate_scatter(
                tv,
                (safe,),
                tokens_v[pl.ds(i * _SCATTER_CHUNK, _SCATTER_CHUNK)],
                mask=owned & (r_chunk < token_size),
            )
            plsc.addupdate_scatter(
                ov,
                (safe,),
                sv[pl.ds(i * _SCATTER_CHUNK, _SCATTER_CHUNK)],
                mask=owned & (r_chunk < scale_size),
            )

        pltpu.async_copy(tv, ot.at[pl.ds(base, width)], sem).wait()
        pltpu.async_copy(ov, os.at[pl.ds(base, width)], sem).wait()

    fn = pl.kernel(
        body,
        out_type=(jax.ShapeDtypeStruct((size,), jnp.int32),) * 2,
        mesh=plsc.VectorSubcoreMesh(
            core_axis_name="core", subcore_axis_name="subcore", num_cores=1
        ),
        scratch_types=[
            pltpu.VMEM((aligned_n,), jnp.int32),
            pltpu.VMEM((aligned_n,), jnp.int32),
            pltpu.VMEM((aligned_n,), jnp.int32),
            pltpu.VMEM((width,), jnp.int32),
            pltpu.VMEM((width,), jnp.int32),
            pltpu.SemaphoreType.DMA,
        ],
        name="moe_preprocess_sc_scatter",
    )
    t, s = compute_on(
        fn,
        compute_type="tpu_sparsecore",
        out_memory_spaces=jax.memory.Space.Device,
        compiler_options={"sparse_core_config": {"core_ids": [0]}},
    )(rows, scales, tokens)
    return jnp.clip(t[:token_size], 0, n // topk - 1), s[:scale_size]

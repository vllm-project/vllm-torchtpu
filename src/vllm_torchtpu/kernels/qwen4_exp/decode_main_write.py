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
"""Concurrent small-bucket writes with last-token ownership of duplicate rows."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def write_decode_rows(
    pool: jax.Array,
    rows: jax.Array,
    page: jax.Array,
    logical: jax.Array,
    valid: jax.Array,
) -> jax.Array:
    """Write only the final live source row for each destination, bit for bit."""
    count = len(rows)
    offset = logical % pool.shape[1]
    order = jnp.arange(count)
    later = (
        (page[:, None] == page[None, :])
        & (offset[:, None] == offset[None, :])
        & valid[None, :]
        & (order[:, None] < order[None, :])
    )
    live = valid & ~jnp.any(later, axis=1)

    def kernel(pages, offsets, active, source, old, target, semaphores):
        for row in range(count):

            @pl.when(active[row] != 0)
            def start(*, row=row):
                pltpu.make_async_copy(
                    source.at[pl.ds(row, 1)],
                    target.at[pages[row], pl.ds(offsets[row], 1)],
                    semaphores.at[row],
                ).start()

        for row in range(count):

            @pl.when(active[row] != 0)
            def finish(*, row=row):
                pltpu.make_async_copy(
                    source.at[pl.ds(row, 1)],
                    target.at[pages[row], pl.ds(offsets[row], 1)],
                    semaphores.at[row],
                ).wait()

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            in_specs=[
                pl.BlockSpec(rows.shape, lambda *_: (0,) * rows.ndim),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[pltpu.SemaphoreType.DMA((count,))],
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={4: 0},
        name="qsa_decode_main_write",
    )(page, offset, live.astype(jnp.int32), rows, pool)

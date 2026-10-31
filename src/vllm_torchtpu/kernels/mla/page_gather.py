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
"""SparseCore Pallas copy of whole pages out of a paged KV cache.

The cache is read as the flat one-row-per-token view the SparseCore cache
writer (`scatter_utils.scatter_rows`) and gather use, so a step that also
writes or gathers the cache that way keeps one layout of it. Reading pages
with an XLA gather instead makes XLA hold a second, page-tiled layout of the
whole cache and copy it every call.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


def _body(
    ids_hbm: Any,
    src_hbm: Any,
    out_hbm: Any,
    ids_vmem: Any,
    sem: Any,
    *,
    num_pages: int,
    rows: int,
    lanes: int,
    num_subcores: int,
) -> None:
    # Subcore c copies pages [c * lanes, (c + 1) * lanes).
    core = lax.axis_index("core") * num_subcores + lax.axis_index("subcore")
    base = core * lanes

    @pl.when(base < num_pages)
    def _():
        pltpu.sync_copy(ids_hbm.at[pl.ds(pl.multiple_of(base, lanes), lanes)], ids_vmem)
        ids = ids_vmem[...]
        copies = []
        for i in range(lanes):
            live = base + i < num_pages
            copy = pltpu.make_async_copy(
                src_hbm.at[pl.ds(ids[i] * rows, rows)],
                out_hbm.at[pl.ds((base + i) * rows, rows)],
                sem.at[i],
            )
            copies.append((live, copy))

            @pl.when(live)
            def _(copy=copy):
                copy.start()

        for live, copy in copies:

            @pl.when(live)
            def _(copy=copy):
                copy.wait()


def gather_pages(cache: jax.Array, page_ids: jax.Array, row_width: int) -> jax.Array:
    """`cache[page_ids]` for a uint32 `[pages, ...]` SparseCore cache, as
    `[len(page_ids) * rows, row_width]` rows of the `[-1, row_width]` view its
    writer uses (one token per row). `page_ids` must be in range."""
    if cache.dtype != jnp.uint32 or (cache[0].size % row_width):
        raise ValueError(f"expected uint32 pages of {row_width}-word rows: {cache}")
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")
    num_pages = page_ids.shape[0]
    rows, width = cache[0].size // row_width, row_width
    lanes = sc.num_lanes
    if num_pages > lanes * sc.num_cores * sc.num_subcores:
        raise ValueError(f"{num_pages} pages exceed one page per lane per subcore")
    padded = pl.cdiv(num_pages, lanes) * lanes
    ids = jnp.pad(page_ids.astype(jnp.int32), (0, padded - num_pages))
    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    return pl.kernel(
        functools.partial(
            _body,
            num_pages=num_pages,
            rows=rows,
            lanes=lanes,
            num_subcores=sc.num_subcores,
        ),
        out_type=jax.ShapeDtypeStruct((num_pages * rows, width), jnp.uint32),
        scratch_types=(
            pltpu.VMEM((lanes,), jnp.int32),
            pltpu.SemaphoreType.DMA((lanes,)),
        ),
        # Native SparseCore (linear) operand layout, as the cache writer's.
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        mesh=mesh,
        name=f"page_gather_p{num_pages}_r{rows}",
    )(ids, cache.reshape(-1, width))

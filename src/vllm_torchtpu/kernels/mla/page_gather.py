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

_VMEM_STAGING_BUDGET_BYTES = 128 * 1024
_DEFAULT_STRIPE_ROWS = 128
_MIN_SPLIT_STRIPE_ROWS = 64


def _choose_stripe_plan(
    num_pages: int,
    rows: int,
    row_width: int,
    total_subcores: int,
) -> tuple[int, int, int]:
    stripe_rows = rows
    while stripe_rows > _DEFAULT_STRIPE_ROWS and stripe_rows % 2 == 0:
        stripe_rows //= 2
    while (
        stripe_rows % 2 == 0
        and 2 * stripe_rows * row_width * 4 > _VMEM_STAGING_BUDGET_BYTES
    ):
        stripe_rows //= 2
    # Halve once more on small page counts so all subcores stay utilized.
    if (
        num_pages * (rows // stripe_rows) < total_subcores
        and stripe_rows % 2 == 0
        and stripe_rows // 2 >= _MIN_SPLIT_STRIPE_ROWS
    ):
        stripe_rows //= 2
    stripes_per_page = rows // stripe_rows
    total_stripes = num_pages * stripes_per_page
    max_per_subcore = pl.cdiv(total_stripes, total_subcores)
    if max_per_subcore > 1 and (
        2 * stripe_rows * row_width * 4 <= _VMEM_STAGING_BUDGET_BYTES
    ):
        num_buffers = 2
    else:
        num_buffers = 1
    return stripe_rows, stripes_per_page, num_buffers


def _build_subcore_page_table(
    page_ids: jax.Array,
    stripes_per_page: int,
    total_subcores: int,
    lanes: int,
) -> tuple[jax.Array, int, int]:
    num_pages = page_ids.shape[0]
    total_stripes = num_pages * stripes_per_page
    max_per_subcore = pl.cdiv(total_stripes, total_subcores)
    slots = pl.cdiv(max_per_subcore, lanes) * lanes
    subcore_idx = jnp.arange(total_subcores, dtype=jnp.int32)[:, None]
    slot_idx = jnp.arange(slots, dtype=jnp.int32)[None, :]
    stripe_idx = subcore_idx + slot_idx * total_subcores
    page_idx = jnp.minimum(stripe_idx // stripes_per_page, num_pages - 1)
    # Avoid dynamic indexing inside sc_vector_subcore: pre-shuffling page IDs
    # per subcore lets the kernel address each stripe with a static index.
    return page_ids.astype(jnp.int32)[page_idx].reshape(-1), max_per_subcore, slots


def _body(
    ids_hbm: Any,
    src_hbm: Any,
    out_hbm: Any,
    ids_vmem: Any,
    buf_vmem: Any,
    sem_in: Any,
    sem_out: Any,
    *,
    total_stripes: int,
    rows: int,
    stripe_rows: int,
    stripes_per_page: int,
    max_per_subcore: int,
    slots: int,
    lanes: int,
    num_subcores: int,
    total_subcores: int,
    num_buffers: int,
) -> None:
    sc_id = lax.axis_index("core") * num_subcores + lax.axis_index("subcore")
    pltpu.sync_copy(
        ids_hbm.at[pl.ds(pl.multiple_of(sc_id * slots, lanes), slots)],
        ids_vmem,
    )
    ids = ids_vmem[...]

    # Avoid direct HBM-to-HBM make_async_copy on SparseCore: direct HBM copies
    # bottleneck far below HBM bandwidth, whereas staging stripes through SC
    # VMEM across all subcores saturates full SparseCore bandwidth.
    metas = []
    in_copies = []
    out_copies = []
    for k in range(max_per_subcore):
        stripe_idx = sc_id + k * total_subcores
        live = stripe_idx < total_stripes
        safe_stripe = jnp.where(live, stripe_idx, jnp.int32(0))
        if stripes_per_page == 1:
            page_idx = safe_stripe
            row_off = jnp.int32(0)
        else:
            page_idx = lax.div(safe_stripe, jnp.int32(stripes_per_page))
            row_off = lax.rem(safe_stripe, jnp.int32(stripes_per_page)) * jnp.int32(
                stripe_rows
            )
        slot = k % num_buffers
        metas.append(live)
        in_copies.append(
            pltpu.make_async_copy(
                src_hbm.at[pl.ds(ids[k] * rows + row_off, stripe_rows)],
                buf_vmem.at[slot],
                sem_in.at[slot],
            )
        )
        out_copies.append(
            pltpu.make_async_copy(
                buf_vmem.at[slot],
                out_hbm.at[pl.ds(page_idx * rows + row_off, stripe_rows)],
                sem_out.at[slot],
            )
        )

    if num_buffers == 1:
        for k in range(max_per_subcore):

            @pl.when(metas[k])
            def _(ci=in_copies[k], co=out_copies[k]):
                ci.start()
                ci.wait()
                co.start()
                co.wait()

        return

    @pl.when(metas[0])
    def _():
        in_copies[0].start()

    for k in range(max_per_subcore):
        if k + 1 < max_per_subcore:
            if k >= 1:
                # Wait for buf[(k - 1) % 2] == buf[(k + 1) % 2] store before
                # overwriting it with stripe k + 1.
                @pl.when(metas[k - 1])
                def _(co=out_copies[k - 1]):
                    co.wait()

            @pl.when(metas[k + 1])
            def _(ci=in_copies[k + 1]):
                ci.start()

        @pl.when(metas[k])
        def _(ci=in_copies[k], co=out_copies[k]):
            ci.wait()
            co.start()

    for k in range(max(0, max_per_subcore - num_buffers), max_per_subcore):

        @pl.when(metas[k])
        def _(co=out_copies[k]):
            co.wait()


def gather_pages(cache: jax.Array, page_ids: jax.Array, row_width: int) -> jax.Array:
    """`cache[page_ids]` for a uint32 `[pages, ...]` SparseCore cache, as
    `[len(page_ids) * rows, row_width]` rows of the `[-1, row_width]` view its
    writer uses (one token per row). `page_ids` must be in range."""
    if cache.dtype != jnp.uint32 or (cache[0].size % row_width):
        raise ValueError(f"expected uint32 pages of {row_width}-word rows: {cache}")
    rows, width = cache[0].size // row_width, row_width
    num_pages = page_ids.shape[0]
    if num_pages == 0:
        return jnp.zeros((0, width), jnp.uint32)
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")
    lanes = sc.num_lanes
    total_subcores = sc.num_cores * sc.num_subcores
    stripe_rows, stripes_per_page, num_buffers = _choose_stripe_plan(
        num_pages, rows, width, total_subcores
    )
    total_stripes = num_pages * stripes_per_page
    ids_table, max_per_subcore, slots = _build_subcore_page_table(
        page_ids, stripes_per_page, total_subcores, lanes
    )
    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    pl_kernel: Any = pl.kernel
    return pl_kernel(
        functools.partial(
            _body,
            total_stripes=total_stripes,
            rows=rows,
            stripe_rows=stripe_rows,
            stripes_per_page=stripes_per_page,
            max_per_subcore=max_per_subcore,
            slots=slots,
            lanes=lanes,
            num_subcores=sc.num_subcores,
            total_subcores=total_subcores,
            num_buffers=num_buffers,
        ),
        out_type=jax.ShapeDtypeStruct((num_pages * rows, width), jnp.uint32),
        scratch_types=(
            pltpu.VMEM((slots,), jnp.int32),
            pltpu.VMEM((num_buffers, stripe_rows, width), jnp.uint32),
            pltpu.SemaphoreType.DMA((num_buffers,)),
            pltpu.SemaphoreType.DMA((num_buffers,)),
        ),
        # Native SparseCore (linear) operand layout, as the cache writer's.
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        mesh=mesh,
        name=f"page_gather_p{num_pages}_r{rows}",
    )(ids_table, cache.reshape(-1, width))

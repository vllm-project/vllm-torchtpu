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

"""Physical micro-page mapping for sparse decode (no KV materialization)."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.sparse_gqa_decode.planning import VMEM_USABLE_FRACTION


def _vector_gather_rows(
    table: jax.Array | jax.ShapeDtypeStruct,
    logical: jax.Array | jax.ShapeDtypeStruct,
) -> int | None:
    if table.shape[0] == 0 or jax.default_backend() != "tpu":
        return None
    from vllm_torchtpu.kernels.mla.sparse.kernel import _GATHER_PAGE_CHUNK

    info = pltpu.get_tpu_info()
    padded_pages = pl.cdiv(table.shape[1], _GATHER_PAGE_CHUNK) * _GATHER_PAGE_CHUNK
    # Bound the gather kernel's Python-unrolled windows by one vector row.
    if (
        padded_pages > info.num_lanes
        or table.dtype != jnp.int32
        or logical.dtype != jnp.int32
    ):
        return None
    rows = info.num_sublanes
    elements = rows * pl.cdiv(logical.shape[1], info.num_lanes) * info.num_lanes
    table_bytes = rows * padded_pages * table.dtype.itemsize
    index_bytes = elements * logical.dtype.itemsize
    output_bytes = elements * jnp.dtype(jnp.int32).itemsize
    staging_bytes = table_bytes + index_bytes + output_bytes
    if staging_bytes > info.vmem_capacity_bytes * VMEM_USABLE_FRACTION:
        return None
    return rows


def virtual_page_metadata(
    blocks: jax.Array,
    lengths: jax.Array,
    table: jax.Array,
    *,
    ratio: int,
    page_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Map groups of ratio tokens through physical pages of page_size tokens.

    page_size comes from the original cache shape; the returned RPA page has
    ratio tokens. Valid group/page IDs and bounded lengths are caller contracts.
    """
    if ratio <= 0 or page_size <= 0 or page_size % ratio:
        raise ValueError("Physical page size must be positive and divisible by ratio")
    batch, budget = blocks.shape
    if table.ndim != 2 or table.shape[0] != batch or table.shape[1] == 0:
        raise ValueError("Page tables must have one nonempty row per query")
    complete = lengths // ratio
    count = jnp.minimum(complete, budget)
    columns = jnp.arange(budget + 1)[None, :]
    groups = jnp.concatenate((blocks, jnp.zeros((batch, 1), jnp.int32)), axis=1)
    groups = jnp.where(columns == count[:, None], complete[:, None], groups)
    valid = (columns < count[:, None]) | (
        (columns == count[:, None]) & ((lengths % ratio)[:, None] > 0)
    )
    safe = jnp.where(valid, groups, 0)
    logical = jnp.clip(safe // (page_size // ratio), 0, table.shape[1] - 1)
    gather_rows = _vector_gather_rows(table, logical)
    if gather_rows is None:
        physical = jnp.take_along_axis(table, logical, axis=1)
    else:
        from vllm_torchtpu.kernels.mla.sparse.kernel import gather_page_ids

        physical = gather_page_ids(
            table.reshape(-1),
            logical,
            jnp.arange(batch, dtype=jnp.int32),
            batch,
            block_tokens=gather_rows,
        )
    pages = jnp.where(
        valid, physical * (page_size // ratio) + safe % (page_size // ratio), 0
    )
    kv_lens = count * ratio + lengths % ratio
    return pages, kv_lens

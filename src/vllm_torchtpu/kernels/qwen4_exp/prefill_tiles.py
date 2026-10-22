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
"""QSA prefill tiles selected from per-rank layout and resource bounds."""

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

# Scoped compiler budgets, below TPU7x's 64 MiB per core. The reference
# Q256/H24/D256 paired kernel compiles to 61.13 MiB with FP8 KV.
PAIRED_PAGE_VMEM_MIB = {128: 61, 256: 62}


def _paired_working_set(
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    packing: int,
    item_bytes: int,
    query_block: int,
) -> int:
    # FP32 Q/accumulator, softmax scores/probabilities and max/mass, BF16 Q,
    # and two native KV DMA buffers. This is a relative resource bound, not
    # an exact compiler VMEM prediction (layouts/temporaries add overhead).
    rows = query_block * query_heads
    query_state = rows * ((2 * head_dim + 2 * 256 + 2) * 4 + head_dim * 2)
    native_heads = (2 * kv_heads + packing - 1) // packing * packing
    return query_state + 2 * 256 * native_heads * head_dim * item_bytes


def supports_paired_prefill(
    query: jax.Array, cache: jax.Array, *, kv_heads: int, ratio: int, query_block: int
) -> bool:
    """Check segment addressing, packing and the validated VMEM footprint.

    Token count is not a tile resource. Head counts are local geometry, with
    no assumption about tensor-parallel degree or the GQA ratio. Other shapes
    retain the general one-selection-word TC kernel. SparseCore's calibrated
    costs are deliberately separate from this TC layout/resource policy.
    """
    if query.ndim != 3 or cache.ndim != 5 or kv_heads <= 0:
        return False
    _, heads, dim = query.shape
    if (
        heads <= 0
        or heads % kv_heads
        or dim <= 0
        or dim % 128
        or query.dtype != jnp.bfloat16
        or cache.dtype not in (jnp.bfloat16, jnp.float8_e4m3fn)
        or ratio != 4
        or query_block not in PAIRED_PAGE_VMEM_MIB
        or cache.shape[1] <= 0
        or cache.shape[1] % 128
        or pltpu.get_tpu_info().chip_version != pltpu.ChipVersion.TPU_7X
    ):
        return False
    item_bytes = jax.dtypes.itemsize_bits(cache.dtype) // 8
    packing = 4 // item_bytes
    if cache.shape[2:] != ((2 * kv_heads + packing - 1) // packing, packing, dim):
        return False
    reference = _paired_working_set(24, 2, 256, 2, 2, 256)
    return (
        _paired_working_set(heads, kv_heads, dim, packing, item_bytes, query_block)
        <= reference
    )


def get_sparse_prefill_tiles(
    query: jax.Array, cache: jax.Array, *, kv_heads: int, ratio: int
) -> tuple[int, int]:
    """Use paired tiles within their footprint, independent of token bucket."""
    for query_block in (256, 128):
        if supports_paired_prefill(
            query, cache, kv_heads=kv_heads, ratio=ratio, query_block=query_block
        ):
            return query_block, 256
    # Bound query/state windows by the Q24/D256 footprint for other layouts.
    rows = max(8, 256 * 24 * 256 // (query.shape[1] * cache.shape[-1]))
    query_block = min(256, 1 << (rows.bit_length() - 1))
    return query_block, ratio * 32

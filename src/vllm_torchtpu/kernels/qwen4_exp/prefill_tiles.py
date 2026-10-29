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

# Manual soft logical-array targets, independent of VMEM and compiler peaks.
TC_SELECTION_TARGET_BYTES = 256 * 1024**2
TC_PACKING_TARGET_BYTES = 256 * 1024**2
TC_METADATA_BITSET_TARGET_BYTES = 16 * 1024**2
TC_METADATA_MAX_BATCH_BLOCKS = 8


def prefill_vmem_limit() -> int:
    """Reserve one thirty-second of physical VMEM for compiler overhead."""
    return pltpu.get_tpu_info().vmem_capacity_bytes * 31 // 32


def _paired_working_set(
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    packing: int,
    item_bytes: int,
    query_block: int,
    kv_tile: int = 256,
) -> int:
    # FP32 Q/accumulator, scores/probabilities, max/mass and BF16 I/O.
    rows = query_block * query_heads
    query_state = rows * ((2 * head_dim + 2 * kv_tile + 2) * 4 + head_dim * 4)
    native_heads = (2 * kv_heads + packing - 1) // packing * packing
    # Allow another copy of query state for layout conversions/temporaries.
    return 2 * query_state + 2 * kv_tile * native_heads * head_dim * item_bytes


def supports_paired_prefill(
    query: jax.Array | jax.ShapeDtypeStruct,
    cache: jax.Array | jax.ShapeDtypeStruct,
    *,
    kv_heads: int,
    ratio: int,
    query_block: int,
) -> bool:
    """Check paired mask geometry, native packing and VMEM working set."""
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
        or query_block <= 0
        or query_block > 256
        or query_block % 8
        or cache.shape[1] <= 0
        or cache.shape[1] % 128
    ):
        return False
    item_bytes = jax.dtypes.itemsize_bits(cache.dtype) // 8
    packing = 4 // item_bytes
    if cache.shape[2:] != ((2 * kv_heads + packing - 1) // packing, packing, dim):
        return False
    return (
        _paired_working_set(heads, kv_heads, dim, packing, item_bytes, query_block)
        <= prefill_vmem_limit()
    )


def get_sparse_prefill_tiles(
    query: jax.Array | jax.ShapeDtypeStruct,
    cache: jax.Array | jax.ShapeDtypeStruct,
    *,
    kv_heads: int,
    ratio: int,
) -> tuple[int, int]:
    """Choose power-of-two query tiles from the affine VMEM forecast."""
    if (
        query.ndim != 3
        or cache.ndim != 5
        or kv_heads <= 0
        or ratio <= 0
        or min(query.shape[1:]) <= 0
        or min(cache.shape[1:]) <= 0
    ):
        raise ValueError(
            "Sparse prefill requires positive head, page, packing and ratio dimensions"
        )
    item_bytes = jax.dtypes.itemsize_bits(cache.dtype) // 8
    limit = prefill_vmem_limit()

    def capacity(kv_tile: int) -> int:
        fixed = _paired_working_set(
            query.shape[1],
            kv_heads,
            cache.shape[-1],
            cache.shape[3],
            item_bytes,
            0,
            kv_tile,
        )
        per_row = (
            _paired_working_set(
                query.shape[1],
                kv_heads,
                cache.shape[-1],
                cache.shape[3],
                item_bytes,
                1,
                kv_tile,
            )
            - fixed
        )
        # Automatic query tiles use powers of two, capped at one 256-row subtile.
        affordable = min(256, (limit - fixed) // per_row)
        # The caller and kernel require positive query tiles divisible by eight.
        return 1 << (affordable.bit_length() - 1) if affordable >= 8 else 0

    query_block = capacity(ratio * 32)
    paired_query_block = capacity(256)
    if paired_query_block >= query_block and supports_paired_prefill(
        query, cache, kv_heads=kv_heads, ratio=ratio, query_block=paired_query_block
    ):
        return paired_query_block, 256
    if query_block:
        return query_block, ratio * 32
    raise ValueError("Sparse prefill's smallest query tile exceeds available VMEM")

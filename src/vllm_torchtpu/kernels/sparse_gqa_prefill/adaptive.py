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
"""Normal cost-based prefill dispatch and bounded SC-window capability checks."""

import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as candidate
from vllm_torchtpu.kernels.qwen4_exp.prefill_tiles import (
    _paired_working_set,
    prefill_vmem_limit,
)
from vllm_torchtpu.kernels.sparse_gqa_prefill.cost_model import CostModel
from vllm_torchtpu.kernels.sparse_gqa_prefill.tuned_block_sizes import (
    SC_GATHER_MAX_CACHE_BYTES,
)


def adaptive_prefill(
    q,
    gate,
    blocks,
    lengths,
    table,
    cache,
    starts,
    *,
    kv_heads: int,
    ratio: int,
    k_scale=1.0,
    v_scale=1.0,
    cost_model: CostModel | None = None,
    sc_min_context: int | None = None,
):
    """Choose TC/SC from work estimates and an optional context preference."""
    from vllm_torchtpu.kernels.sparse_gqa_prefill.selector import select_prefill

    output, _route = select_prefill(
        q,
        gate,
        blocks,
        lengths,
        table,
        cache,
        starts,
        kv_heads=kv_heads,
        ratio=ratio,
        k_scale=k_scale,
        v_scale=v_scale,
        cost_model=cost_model,
        sc_min_context=sc_min_context,
    )
    return output, cache


def _direct_tc(
    query,
    gate,
    blocks,
    lengths,
    table,
    cache,
    *,
    kv_heads: int,
    ratio: int,
    kv_tile: int,
):
    output, _ = candidate.block_sparse_gqa_attention(
        query,
        gate,
        blocks,
        lengths,
        table,
        cache,
        jnp.array([0, len(query)], jnp.int32),
        kv_heads=kv_heads,
        ratio=ratio,
        query_block=256,
        kv_tile=kv_tile,
    )
    return output


def _ragged_tc_write(
    arrays,
    cache,
    starts,
    output,
    offset,
    size,
    *,
    output_base=0,
    kv_heads: int,
    ratio: int,
    kv_tile: int,
):
    query, gate, blocks, lengths, table = arrays
    return candidate._ragged_prefill(
        query,
        gate,
        blocks,
        lengths,
        table,
        cache,
        starts,
        output,
        offset,
        size,
        ratio=ratio,
        kv_heads=kv_heads,
        query_block=256,
        kv_tile=kv_tile,
        k_scale=1.0,
        v_scale=1.0,
        output_base=output_base,
    )


def _nonempty_request_starts_are_q256_aligned(starts):
    """Ignore padded empty requests when checking Q256 start alignment."""
    return jnp.all((starts[:-1] % 256 == 0) | (jnp.diff(starts) == 0))


def _tc(
    query,
    gate,
    blocks,
    lengths,
    table,
    cache,
    starts,
    *,
    kv_heads: int,
    ratio: int,
    kv_tile: int,
):
    aligned = _nonempty_request_starts_are_q256_aligned(starts)
    arrays = (query, gate, blocks, lengths, table)
    return lax.cond(
        aligned,
        lambda _: _direct_tc(
            query,
            gate,
            blocks,
            lengths,
            table,
            cache,
            kv_heads=kv_heads,
            ratio=ratio,
            kv_tile=kv_tile,
        ),
        lambda _: _ragged_tc_write(
            arrays,
            cache,
            starts,
            jnp.zeros_like(query),
            0,
            len(query),
            kv_heads=kv_heads,
            ratio=ratio,
            kv_tile=kv_tile,
        ),
        None,
    )


def is_supported_by_sc_window(
    query: jax.Array | jax.ShapeDtypeStruct,
    gate: jax.Array | jax.ShapeDtypeStruct,
    blocks: jax.Array | jax.ShapeDtypeStruct,
    lengths: jax.Array | jax.ShapeDtypeStruct,
    table: jax.Array | jax.ShapeDtypeStruct,
    cache: jax.Array | jax.ShapeDtypeStruct,
    starts: jax.Array | jax.ShapeDtypeStruct,
    *,
    kv_heads: int,
    ratio: int,
    k_scale: float,
    v_scale: float,
) -> bool:
    """Static SC-window eligibility; the persistent owner is staged separately."""
    # Keep these stages consecutive: early failures must not inspect later inputs.
    if not _has_query_and_request_geometry(query, gate, kv_heads, ratio, starts):
        return False
    if not _has_supported_cache_and_index_metadata(
        query, blocks, lengths, table, cache, starts
    ):
        return False
    if cache.shape[1] % ratio:
        return False
    if not _has_unit_scales(k_scale, v_scale):
        return False
    packing = 32 // jax.dtypes.itemsize_bits(cache.dtype)
    if cache.shape[2:] != (
        (2 * kv_heads + packing - 1) // packing,
        packing,
        query.shape[2],
    ):
        return False
    info = pltpu.get_tpu_info()
    return (
        info.sparse_core is not None
        and _paired_working_set(
            query.shape[1],
            kv_heads,
            query.shape[2],
            packing,
            jax.dtypes.itemsize_bits(cache.dtype) // 8,
            8,
            256,
        )
        <= prefill_vmem_limit()
    )


def _has_query_and_request_geometry(query, gate, kv_heads, ratio, starts):
    """Check query geometry before request offsets."""
    if (  # noqa: SIM103 - Keep one truth test per predicate.
        query.ndim != 3
        or min(query.shape) <= 0
        or kv_heads <= 0
        or query.shape[1] % kv_heads
        or query.shape[2] % 128
        or gate.shape != query.shape
        or query.dtype != jnp.bfloat16
        or gate.dtype != jnp.bfloat16
        or ratio <= 0
        or 256 % ratio
        or (ratio * 32) % 128
        or starts.ndim != 1
        or starts.shape[0] < 2
    ):
        return False
    return True


def _has_supported_cache_and_index_metadata(
    query, blocks, lengths, table, cache, starts
):
    """Check cache layout and budget before inspecting index metadata."""
    if (  # noqa: SIM103 - Keep one truth test per predicate.
        cache.ndim != 5
        or cache.shape[0] == 0
        or cache.shape[1] <= 0
        or cache.shape[1] % 128
        or cache.dtype not in (jnp.bfloat16, jnp.float8_e4m3fn)
        or cache.shape[0] * cache.shape[1] * cache.shape[2] > jnp.iinfo(jnp.int32).max
        # SparseCore receives this bounded window as one parameter. The native
        # persistent pool is not subject to this parameter-size bound.
        or math.prod(cache.shape) * jnp.dtype(cache.dtype).itemsize
        > SC_GATHER_MAX_CACHE_BYTES
        or blocks.ndim != 2
        or blocks.shape[0] != len(query)
        or blocks.shape[1] <= 0
        or lengths.shape != (len(query),)
        or table.ndim != 2
        or table.shape[0] != len(query)
        or table.shape[1] <= 0
        or any(x.dtype != jnp.int32 for x in (blocks, lengths, table, starts))
    ):
        return False
    return True


def _has_unit_scales(k_scale, v_scale):
    """Check both scale types before comparing their values."""
    if (  # noqa: SIM103 - Keep one truth test per predicate.
        not isinstance(k_scale, (int, float))
        or not isinstance(v_scale, (int, float))
        or k_scale != 1.0
        or v_scale != 1.0
    ):
        return False
    return True

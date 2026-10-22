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

import math

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from .abi import RPA_WAIT_ROW_WORDS
from .planning import DecodeLaunch, _plan_launch, make_config
from .tuned_block_sizes import DEFAULT_POLICY, DecodePolicy


def plan_decode(
    q: jax.Array,
    gate: jax.Array,
    blocks: jax.Array,
    lengths: jax.Array,
    table: jax.Array,
    cache: jax.Array,
    *,
    kv_heads: int,
    ratio: int,
    k_scale: float,
    v_scale: float,
    policy: DecodePolicy = DEFAULT_POLICY,
) -> DecodeLaunch | None:
    """Static eligibility; callers supply valid per-query selected groups."""
    if (
        q.ndim != 3
        or q.shape[1] <= 0
        or q.shape[2] <= 0
        or kv_heads <= 0
        or q.shape[1] % kv_heads != 0
        or len(q) == 0
        or gate.shape != q.shape
        or q.dtype != jnp.bfloat16
        or gate.dtype != jnp.bfloat16
        or ratio <= 0
        or policy.bkv_tokens <= 0
        or policy.bkv_tokens % ratio != 0
        or blocks.ndim != 2
        or blocks.shape[0] != len(q)
        or blocks.shape[1] <= 0
        or lengths.shape != (len(q),)
        or table.ndim != 2
        or table.shape[0] != len(q)
        or cache.ndim != 5
        or cache.shape[0] <= 0
        or cache.shape[1] <= 0
        or cache.shape[1] % ratio != 0
        or cache.shape[0] * cache.shape[1] > jnp.iinfo(jnp.int32).max
        or cache.dtype not in (jnp.bfloat16, jnp.float8_e4m3fn)
        or any(x.dtype != jnp.int32 for x in (blocks, lengths, table))
        or not isinstance(k_scale, (int, float))
        or not isinstance(v_scale, (int, float))
        or not math.isfinite(k_scale)
        or not math.isfinite(v_scale)
        or k_scale <= 0
        or v_scale <= 0
    ):
        return None
    info = pltpu.get_tpu_info()
    if (
        info.chip_version != pltpu.ChipVersion.TPU_7X
        or info.num_lanes != RPA_WAIT_ROW_WORDS
        or policy.bkv_tokens % info.num_lanes
        or not 0 < policy.pipeline_lanes <= info.num_lanes
        or policy.schedule_copy_words <= 0
        or policy.schedule_copy_words % info.num_lanes
    ):
        return None
    packing = jax.dtypes.itemsize_bits(jnp.uint32) // jax.dtypes.itemsize_bits(
        cache.dtype
    )
    # Physical token offsets, logical context lengths and flattened page-table
    # lookups use int32. These limits follow the indexing contract, not a
    # benchmark's largest batch or context.
    index_max = jnp.iinfo(jnp.int32).max
    if (
        q.shape[2] % info.num_lanes != 0
        or cache.shape[2:] != (math.ceil(2 * kv_heads / packing), packing, q.shape[2])
        or (blocks.shape[1] + 1) * ratio > index_max
        or len(q) * (blocks.shape[1] + 1) > index_max
        or table.shape[1] == 0
        or (table.shape[1] - 1) * cache.shape[1] >= index_max
        or table.size > index_max
    ):
        return None
    cfgs = make_config(
        1 << (q.shape[0] - 1).bit_length(),
        q.shape[1],
        q.shape[2],
        kv_heads,
        ratio,
        blocks.shape[1],
        cache.dtype,
        info=info,
        policy=policy,
    )
    return _plan_launch(cfgs, info, policy)


def is_supported_by_fused_decode(
    q: jax.Array,
    gate: jax.Array,
    blocks: jax.Array,
    lengths: jax.Array,
    table: jax.Array,
    cache: jax.Array,
    *,
    kv_heads: int,
    ratio: int,
    k_scale: float,
    v_scale: float,
) -> bool:
    """Shape/static envelope only; callers guarantee metadata contents valid."""
    return (
        plan_decode(
            q,
            gate,
            blocks,
            lengths,
            table,
            cache,
            kv_heads=kv_heads,
            ratio=ratio,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        is not None
    )

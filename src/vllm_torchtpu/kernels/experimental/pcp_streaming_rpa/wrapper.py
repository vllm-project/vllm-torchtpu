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
"""Composite wrappers for PCP streaming RPA."""

import math

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs as batched_rpa_configs,
)
from vllm_torchtpu.kernels.experimental.batched_rpa.utils import (
    align_to,
    get_dtype_packing,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
    splash_attention_kernel as _splash_kernel_mod,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
    stage_plan as _stage_plan_mod,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.kernel import (
    pcp_streaming_attention_page_groups_packed_local_from_metadata,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.stage_plan import (
    PCP_STREAMING_RPA_VPAGE_GROUP_SIZE,
    PCPStageConfig,
    PCPStagePlan,
)

NUM_LANES = _splash_kernel_mod.NUM_LANES
NUM_SUBLANES = _splash_kernel_mod.NUM_SUBLANES

PCP_AXIS_NAME = _stage_plan_mod.PCP_AXIS_NAME
TP_AXIS_NAME = "tp"
# Pallas Splash sublane/lane tile layout requires head_dim >= NUM_LANES (128).
_UNSUPPORTED_HEAD_DIM_64 = 64
_EXPECTED_KV_CACHE_NDIM = 5

KVLayout = batched_rpa_configs.KVLayout

__all__ = [
    "PCP_AXIS_NAME",
    "TP_AXIS_NAME",
    "KVLayout",
    "PCPStagePlan",
    "compute_pcp_local_mapping",
    "sharded_pcp_ragged_paged_attention",
]


def _pack_kv_for_cache(
    k: jax.Array,
    v: jax.Array,
    cache_dtype: jnp.dtype,
    *,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
    page_size: int | None = None,
) -> jax.Array:
    """Pack K and V together before converting to the cache dtype."""
    max_num_tokens, num_kv_heads, head_dim = k.shape
    packing = get_dtype_packing(cache_dtype)
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        if page_size is None or page_size <= 0:
            msg = "PCP SEQ_ALONG_LANE KV packing requires page_size > 0."
            raise ValueError(msg)
        head_dim_padded = align_to(head_dim, NUM_SUBLANES * packing)
        actual_num_kv_heads = num_kv_heads * 2
        padded_tokens = align_to(max_num_tokens, page_size)
        kv = jnp.concatenate([k, v], axis=-1).reshape(
            max_num_tokens, actual_num_kv_heads, head_dim
        )
        if padded_tokens != max_num_tokens or head_dim_padded != head_dim:
            kv = jnp.pad(
                kv,
                (
                    (0, padded_tokens - max_num_tokens),
                    (0, 0),
                    (0, head_dim_padded - head_dim),
                ),
                constant_values=0,
            )
        packed_kv = kv.reshape(
            padded_tokens,
            actual_num_kv_heads,
            head_dim_padded // packing,
            packing,
        )
        return jnp.transpose(packed_kv, (1, 2, 3, 0))

    head_dim_padded = align_to(head_dim, NUM_LANES)
    actual_num_kv_heads = num_kv_heads * 2
    num_kv_heads_padded = align_to(actual_num_kv_heads, packing)
    if head_dim_padded != head_dim:
        k = jnp.pad(
            k,
            ((0, 0), (0, 0), (0, head_dim_padded - head_dim)),
            constant_values=0,
        )
        v = jnp.pad(
            v,
            ((0, 0), (0, 0), (0, head_dim_padded - head_dim)),
            constant_values=0,
        )
    kv = jnp.concatenate([k, v], axis=-1).reshape(
        max_num_tokens, actual_num_kv_heads, head_dim_padded
    )
    if num_kv_heads_padded != actual_num_kv_heads:
        kv = jnp.pad(
            kv,
            (
                (0, 0),
                (0, num_kv_heads_padded - actual_num_kv_heads),
                (0, 0),
            ),
            constant_values=0,
        )
    return kv.reshape(
        max_num_tokens,
        num_kv_heads_padded // packing,
        packing,
        head_dim_padded // NUM_LANES,
        NUM_LANES,
    )


def _kv_cache_page_size(kv_cache: jax.Array, kv_layout: KVLayout) -> int:
    """Return the token page size of a 5D packed PCP KV cache."""
    if kv_cache.ndim != _EXPECTED_KV_CACHE_NDIM:
        msg = f"PCP KV cache must be {_EXPECTED_KV_CACHE_NDIM}D, got {kv_cache.shape}."
        raise ValueError(msg)
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        return int(kv_cache.shape[-1])
    return int(kv_cache.shape[1])


def _prepare_packed_kv_for_cache(
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    *,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
) -> jax.Array:
    """Pack K/V once and adapt the packed tail to the cache layout."""
    page_size = _kv_cache_page_size(kv_cache, kv_layout)
    packed_kv = _pack_kv_for_cache(
        k, v, kv_cache.dtype, kv_layout=kv_layout, page_size=page_size
    )
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        if packed_kv.shape[:-1] != kv_cache.shape[1:-1]:
            msg = (
                "SEQ_ALONG_LANE fresh KV prefix must match the cache head "
                f"layout: {packed_kv.shape[:-1]} vs {kv_cache.shape[1:-1]}."
            )
            raise ValueError(msg)
        if packed_kv.shape[-1] < k.shape[0]:
            msg = (
                "SEQ_ALONG_LANE fresh KV token extent must cover all local "
                f"tokens: {packed_kv.shape[-1]} vs {k.shape[0]}."
            )
            raise ValueError(msg)
        return packed_kv.astype(kv_cache.dtype)

    if packed_kv.shape[1:] != kv_cache.shape[2:]:
        packed_tail = 1
        for dim in packed_kv.shape[1:]:
            packed_tail *= dim
        cache_tail = 1
        for dim in kv_cache.shape[2:]:
            cache_tail *= dim
        if packed_tail != cache_tail:
            msg = (
                "packed KV update shape is incompatible with KV cache layout: "
                f"{packed_kv.shape[1:]} vs {kv_cache.shape[2:]}."
            )
            raise ValueError(msg)
        packed_kv = packed_kv.reshape((packed_kv.shape[0], *kv_cache.shape[2:]))
    return packed_kv.astype(kv_cache.dtype)


def _reshape_packed_kv_cache_for_attention(
    kv_cache: jax.Array,
) -> jax.Array:
    """Validate that kv_cache packing matches its element dtype."""
    expected_packing = get_dtype_packing(kv_cache.dtype)
    if kv_cache.shape[3] == expected_packing:
        return kv_cache
    msg = (
        "KV cache layout packing does not match dtype packing: "
        f"shape={kv_cache.shape}, "
        f"expected_packing={expected_packing}."
    )
    raise ValueError(msg)


def compute_pcp_local_mapping(
    positions: jax.Array,
    token_req_indices: jax.Array,
    block_tables: jax.Array,
    block_size: int,
    cp_size: int,
    cp_rank: int,
    interleave_size: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Compute vLLM-compatible local CP slot mapping for one PCP rank."""
    if block_size <= 0:
        raise ValueError(f"Expected positive block_size, got {block_size}.")
    if cp_size <= 0:
        raise ValueError(f"Expected positive cp_size, got {cp_size}.")
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"Expected cp_rank in [0, {cp_size}), got {cp_rank}.")
    if interleave_size <= 0:
        raise ValueError(f"Expected positive interleave_size, got {interleave_size}.")

    positions = positions.astype(jnp.int32)
    token_req_indices = token_req_indices.astype(jnp.int32)
    virtual_block_size = block_size * cp_size
    block_indices = positions // virtual_block_size
    block_numbers = block_tables[token_req_indices, block_indices].astype(jnp.int32)

    virtual_offsets = positions - block_indices * virtual_block_size
    is_local = ((virtual_offsets // interleave_size) % cp_size) == cp_rank
    local_offsets = (
        virtual_offsets // (cp_size * interleave_size)
    ) * interleave_size + (virtual_offsets % interleave_size)
    slot_ids = block_numbers * block_size + local_offsets
    slot_ids = jnp.where(is_local, slot_ids, -1)
    return is_local, slot_ids.astype(jnp.int32)


def sharded_pcp_ragged_paged_attention(
    mesh: Mesh,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sink: jax.Array | None,
    sm_scale: float,
    attention_chunk_size: int | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    update_kv_cache: bool = True,
    cp_kv_cache_interleave_size: int = 0,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
) -> tuple[jax.Array, jax.Array]:
    """Run streaming PCP RPA over local Q/K/V shards."""
    pcp_size = _validate_pcp_streaming_unsupported_features(
        q,
        mesh,
        attention_sink=attention_sink,
        attention_chunk_size=attention_chunk_size,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
    )
    _validate_pcp_streaming_scales_and_cache_flag(
        q_scale,
        k_scale,
        v_scale,
        update_kv_cache=update_kv_cache,
    )
    page_size = _validate_pcp_streaming_layout_and_sharding(
        q,
        kv_cache,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        kv_layout=kv_layout,
        pcp_size=pcp_size,
    )

    if TP_AXIS_NAME in mesh.axis_names:
        qkv_spec = P(PCP_AXIS_NAME, TP_AXIS_NAME)
        kv_cache_spec = P(PCP_AXIS_NAME, None, TP_AXIS_NAME)
    else:
        qkv_spec = P(PCP_AXIS_NAME)
        kv_cache_spec = P(PCP_AXIS_NAME)
    metadata_spec = P()
    in_specs = (
        qkv_spec,
        qkv_spec,
        qkv_spec,
        kv_cache_spec,
        metadata_spec,
        metadata_spec,
        metadata_spec,
        metadata_spec,
    )
    out_specs = (qkv_spec, kv_cache_spec)
    args = (
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    )

    def _pcp_ragged_paged_attention(
        q_local: jax.Array,
        k_local: jax.Array,
        v_local: jax.Array,
        kv_cache: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Build the local stage plan and run PCP unified Splash on one shard."""
        stage_config = PCPStageConfig(
            local_tokens=q_local.shape[0],
            local_kv_cache_num_blocks=kv_cache.shape[0],
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=cp_kv_cache_interleave_size,
            vpage_group_size=PCP_STREAMING_RPA_VPAGE_GROUP_SIZE,
            kv_layout=kv_layout,
            pcp_axis_name=PCP_AXIS_NAME,
        )
        local_stage_plan = _stage_plan_mod.build_pcp_local_stage_plan(
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            config=stage_config,
        )
        query_row_valid = local_stage_plan.query_row_valid
        segment_descriptors = local_stage_plan.segment_descriptors
        num_segments = local_stage_plan.num_segments
        packed_current_kv = _prepare_packed_kv_for_cache(
            k_local, v_local, kv_cache, kv_layout=kv_layout
        )
        q_streaming, q_per_kv = _pad_local_q_heads_for_streaming(
            q_local, k_local.shape[1]
        )
        attention_kv_cache = _reshape_packed_kv_cache_for_attention(kv_cache)
        output, kv_cache = (
            pcp_streaming_attention_page_groups_packed_local_from_metadata(
                q_streaming,
                packed_current_kv,
                attention_kv_cache,
                kv_lens,
                cu_q_lens,
                distribution,
                segment_descriptors,
                num_segments,
                stage_plan=local_stage_plan,
                vpage_group_size=PCP_STREAMING_RPA_VPAGE_GROUP_SIZE,
                pcp_size=pcp_size,
                interleave_size=cp_kv_cache_interleave_size,
                sm_scale=sm_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                kv_layout=kv_layout,
                pcp_axis_name=PCP_AXIS_NAME,
            )
        )
        output = output[:, :, :q_per_kv, :]
        valid_output_rows = query_row_valid.reshape(
            (query_row_valid.shape[0],) + (1,) * (output.ndim - 1)
        )
        output = jnp.where(valid_output_rows, output, jnp.zeros_like(output))
        output = output.reshape(q_local.shape)
        return output, kv_cache

    return jax.shard_map(
        _pcp_ragged_paged_attention,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(*args)


def _validate_pcp_streaming_unsupported_features(
    q: jax.Array,
    mesh: Mesh,
    *,
    attention_sink: jax.Array | None,
    attention_chunk_size: int | None,
    cp_kv_cache_interleave_size: int,
) -> int:
    """Validate mesh axis and unsupported attention features, returning pcp_size."""
    if attention_sink is not None:
        msg = "PCP RPA does not support attention sinks."
        raise NotImplementedError(msg)
    if q.shape[-1] == _UNSUPPORTED_HEAD_DIM_64:
        msg = "PCP RPA does not support head_dim==64."
        raise NotImplementedError(msg)
    if cp_kv_cache_interleave_size <= 0:
        msg = "PCP RPA requires cp_kv_cache_interleave_size > 0."
        raise ValueError(msg)
    if PCP_AXIS_NAME not in mesh.axis_names:
        msg = "PCP requires a mesh with a pcp axis."
        raise NotImplementedError(msg)
    if attention_chunk_size is not None:
        msg = "PCP streaming RPA supports full attention only."
        raise NotImplementedError(msg)
    return int(mesh.shape[PCP_AXIS_NAME])


def _validate_pcp_streaming_scales_and_cache_flag(
    q_scale: jax.Array | float | None,
    k_scale: jax.Array | float | None,
    v_scale: jax.Array | float | None,
    *,
    update_kv_cache: bool,
) -> None:
    """Validate Q/K/V quantization scales and KV cache update flag."""
    if q_scale is not None:
        msg = "PCP streaming RPA does not support quantized Q scales."
        raise NotImplementedError(msg)
    if (k_scale is None) != (v_scale is None):
        msg = (
            "PCP streaming RPA requires k_scale and v_scale to be both "
            "None or both non-None."
        )
        raise ValueError(msg)
    if not update_kv_cache:
        msg = "PCP streaming RPA requires update_kv_cache=True."
        raise NotImplementedError(msg)


def _validate_pcp_streaming_layout_and_sharding(
    q: jax.Array,
    kv_cache: jax.Array,
    *,
    cp_kv_cache_interleave_size: int,
    kv_layout: KVLayout,
    pcp_size: int,
) -> int:
    """Validate page size divisibility and even PCP sharding, returning page_size."""
    page_size = _kv_cache_page_size(kv_cache, kv_layout)
    if kv_layout == KVLayout.SEQ_ALONG_LANE and (
        page_size % NUM_LANES != 0 or cp_kv_cache_interleave_size % NUM_LANES != 0
    ):
        msg = (
            "PCP SEQ_ALONG_LANE UT path requires page_size and "
            f"cp_kv_cache_interleave_size to be multiples of {NUM_LANES}, got "
            f"{page_size=} {cp_kv_cache_interleave_size=}."
        )
        raise NotImplementedError(msg)
    if (
        cp_kv_cache_interleave_size > page_size
        or page_size % cp_kv_cache_interleave_size != 0
    ):
        msg = (
            "PCP streaming RPA requires page_size to be divisible by "
            "cp_kv_cache_interleave_size: "
            f"page_size={page_size} "
            f"{cp_kv_cache_interleave_size=}."
        )
        raise NotImplementedError(msg)
    if q.shape[0] % pcp_size != 0:
        msg = (
            "PCP streaming RPA requires q tokens to be evenly "
            f"sharded across PCP ranks: {q.shape[0]=} "
            f"{pcp_size=}."
        )
        raise ValueError(msg)
    if kv_cache.shape[0] % pcp_size != 0:
        msg = (
            "PCP streaming RPA requires KV cache blocks to be "
            "evenly sharded across PCP ranks: "
            f"{kv_cache.shape[0]=} {pcp_size=}."
        )
        raise ValueError(msg)
    return page_size


def _pad_local_q_heads_for_streaming(
    q_local: jax.Array,
    num_kv_heads: int,
) -> tuple[jax.Array, int]:
    """Pad Q heads to a multiple of dtype packing for streaming attention."""
    if q_local.shape[1] % num_kv_heads != 0:
        msg = "Q heads must be divisible by KV heads."
        raise ValueError(msg)
    q_per_kv = q_local.shape[1] // num_kv_heads
    q_streaming = q_local.reshape(
        q_local.shape[0], num_kv_heads, q_per_kv, q_local.shape[2]
    )
    q_packing = get_dtype_packing(q_local.dtype)
    q_per_kv_padded = math.ceil(q_per_kv / q_packing) * q_packing
    if q_per_kv_padded != q_per_kv:
        q_streaming = jnp.pad(
            q_streaming,
            (
                (0, 0),
                (0, 0),
                (0, q_per_kv_padded - q_per_kv),
                (0, 0),
            ),
            constant_values=0,
        )
    return q_streaming, q_per_kv

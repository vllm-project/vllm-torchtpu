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
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    wrapper as batched_rpa_wrapper,
)
from vllm_torchtpu.kernels.experimental.batched_rpa.utils import get_dtype_packing
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
    stage_plan as _stage_plan_mod,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.kernel import (
    PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE,
    pcp_streaming_attention_page_groups_packed_local_from_metadata,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
    _reshape_metadata_page_indices_jax,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.stage_plan import (
    _build_writeback_segment_descriptors,
    _compact_writeback_slot_ids_for_cache_rank,
    _compute_cache_owner_and_slot_ids_for_rows,
    _compute_pcp_kv_writeback_mapping,
    _compute_pcp_local_query_row_mapping,
    _compute_pcp_local_query_row_mapping_from_metadata,
    _compute_pcp_rank_major_query_row_mapping_from_metadata,
    _pcp_rank_token_count_before,
    _PCPKVWritebackMapping,
    _PCPQueryRowMapping,
    get_pcp_rank_without_axis_index,
)

PCP_AXIS_NAME = _stage_plan_mod.PCP_AXIS_NAME
TP_AXIS_NAME = "tp"

KVLayout = batched_rpa_configs.KVLayout

__all__ = [
    "PCP_AXIS_NAME",
    "TP_AXIS_NAME",
    "KVLayout",
    "_PCPKVWritebackMapping",
    "_PCPQueryRowMapping",
    "_build_writeback_segment_descriptors",
    "_compact_writeback_slot_ids_for_cache_rank",
    "_compute_cache_owner_and_slot_ids_for_rows",
    "_compute_pcp_kv_writeback_mapping",
    "_compute_pcp_local_query_row_mapping",
    "_compute_pcp_local_query_row_mapping_from_metadata",
    "_compute_pcp_rank_major_query_row_mapping_from_metadata",
    "_pcp_rank_token_count_before",
    "compute_pcp_local_mapping",
    "get_pcp_rank_without_axis_index",
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
    dummy_q = jnp.zeros_like(k)
    _, packed_kv = batched_rpa_wrapper.prepare_inputs(
        dummy_q,
        k,
        v,
        k.dtype,
        cache_dtype,
        kv_layout=kv_layout,
    )
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        if page_size is None or page_size <= 0:
            raise ValueError("PCP SEQ_ALONG_LANE KV packing requires page_size > 0.")
        padded_tokens = math.ceil(packed_kv.shape[-1] / page_size) * page_size
        packed_kv = jnp.pad(
            packed_kv,
            ((0, 0), (0, 0), (0, 0), (0, padded_tokens - packed_kv.shape[-1])),
            constant_values=0,
        )
    return packed_kv


def _kv_cache_page_size(kv_cache: jax.Array, kv_layout: KVLayout) -> int:
    if kv_cache.ndim != 5:
        raise ValueError(f"PCP KV cache must be 5D, got {kv_cache.shape}.")
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
            raise ValueError(
                "SEQ_ALONG_LANE fresh KV prefix must match the cache head "
                f"layout: {packed_kv.shape[:-1]} vs {kv_cache.shape[1:-1]}."
            )
        if packed_kv.shape[-1] < k.shape[0]:
            raise ValueError(
                "SEQ_ALONG_LANE fresh KV token extent must cover all local "
                f"tokens: {packed_kv.shape[-1]} vs {k.shape[0]}."
            )
        return packed_kv.astype(kv_cache.dtype)

    if packed_kv.shape[1:] != kv_cache.shape[2:]:
        packed_tail = 1
        for dim in packed_kv.shape[1:]:
            packed_tail *= dim
        cache_tail = 1
        for dim in kv_cache.shape[2:]:
            cache_tail *= dim
        if packed_tail != cache_tail:
            raise ValueError(
                "packed KV update shape is incompatible with KV cache layout: "
                f"{packed_kv.shape[1:]} vs {kv_cache.shape[2:]}."
            )
        packed_kv = packed_kv.reshape((packed_kv.shape[0], *kv_cache.shape[2:]))
    return packed_kv.astype(kv_cache.dtype)


def _reshape_packed_kv_cache_for_attention(
    kv_cache: jax.Array,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
) -> jax.Array:
    del kv_layout
    expected_packing = get_dtype_packing(kv_cache.dtype)
    if kv_cache.shape[3] == expected_packing:
        return kv_cache
    raise ValueError(
        "KV cache layout packing does not match dtype packing: "
        f"shape={kv_cache.shape}, "
        f"expected_packing={expected_packing}."
    )


def _select_replicated_shard_for_pcp_rank(
    tensor: jax.Array,
    pcp_axis: str,
    pcp_size: int,
    *,
    axis: int,
) -> jax.Array:
    """Select this PCP rank shard from replicated data without axis_index."""
    if axis < 0:
        axis += tensor.ndim
    if tensor.shape[axis] % pcp_size != 0:
        raise ValueError(
            "Replicated PCP tensor axis must be divisible by "
            f"pcp_size: shape={tensor.shape}, axis={axis}, "
            f"pcp_size={pcp_size}."
        )
    shard_size = tensor.shape[axis] // pcp_size
    exchanged = jax.lax.all_to_all(
        tensor, axis_name=pcp_axis, split_axis=axis, concat_axis=axis, tiled=True
    )
    return jax.lax.dynamic_slice_in_dim(exchanged, 0, shard_size, axis=axis)


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


def _select_replicated_query_row_mapping_for_pcp_rank(
    mapping: _PCPQueryRowMapping,
    pcp_axis: str,
    pcp_size: int,
) -> _PCPQueryRowMapping:
    """Select a PCP rank's query rows with one collective selection."""
    packed_mapping = jnp.stack(
        (
            mapping.absolute_positions,
            mapping.request_indices,
            mapping.source_ranks,
            mapping.is_valid.astype(jnp.int32),
        ),
        axis=1,
    )
    local_mapping = _select_replicated_shard_for_pcp_rank(
        packed_mapping,
        pcp_axis,
        pcp_size,
        axis=0,
    )
    return _PCPQueryRowMapping(
        absolute_positions=local_mapping[:, 0],
        request_indices=local_mapping[:, 1],
        source_ranks=local_mapping[:, 2],
        is_valid=local_mapping[:, 3].astype(jnp.bool_),
    )


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
    q_block_size: int = PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE,
    q_compute_size: int | None = None,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
):
    """Runs streaming PCP RPA over local Q/K/V shards."""
    if attention_sink is not None:
        raise NotImplementedError("PCP RPA does not support attention sinks.")
    if q.shape[-1] == 64:
        raise NotImplementedError("PCP RPA does not support head_dim==64.")
    if cp_kv_cache_interleave_size <= 0:
        raise ValueError("PCP RPA requires cp_kv_cache_interleave_size > 0.")
    if PCP_AXIS_NAME not in mesh.axis_names:
        raise NotImplementedError("PCP requires a mesh with a pcp axis.")
    pcp_size = mesh.shape[PCP_AXIS_NAME]
    if attention_chunk_size is not None:
        raise NotImplementedError("PCP streaming RPA supports full attention only.")
    if q_scale is not None:
        raise NotImplementedError(
            "PCP streaming RPA does not support quantized Q scales."
        )
    if (k_scale is None) != (v_scale is None):
        raise ValueError(
            "PCP streaming RPA requires k_scale and v_scale to be both "
            "None or both non-None."
        )
    if not update_kv_cache:
        raise NotImplementedError("PCP streaming RPA requires update_kv_cache=True.")
    page_size = _kv_cache_page_size(kv_cache, kv_layout)
    if kv_layout == KVLayout.SEQ_ALONG_LANE and (
        page_size % 128 != 0 or cp_kv_cache_interleave_size % 128 != 0
    ):
        raise NotImplementedError(
            "PCP SEQ_ALONG_LANE UT path requires page_size and "
            "cp_kv_cache_interleave_size to be multiples of 128, got "
            f"{page_size=} {cp_kv_cache_interleave_size=}."
        )
    if (
        cp_kv_cache_interleave_size > page_size
        or page_size % cp_kv_cache_interleave_size != 0
    ):
        raise NotImplementedError(
            "PCP streaming RPA requires page_size to be divisible by "
            "cp_kv_cache_interleave_size: "
            f"page_size={page_size} "
            f"{cp_kv_cache_interleave_size=}."
        )
    if q.shape[0] % pcp_size != 0:
        raise ValueError(
            "PCP streaming RPA requires q tokens to be evenly "
            f"sharded across PCP ranks: {q.shape[0]=} "
            f"{pcp_size=}."
        )
    if kv_cache.shape[0] % pcp_size != 0:
        raise ValueError(
            "PCP streaming RPA requires KV cache blocks to be "
            f"evenly sharded across PCP ranks: "
            f"{kv_cache.shape[0]=} {pcp_size=}."
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
        q_local,
        k_local,
        v_local,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    ):
        rank_major_query_rows = _compute_pcp_rank_major_query_row_mapping_from_metadata(
            kv_lens,
            cu_q_lens,
            distribution,
            local_padded_tokens=q_local.shape[0],
            pcp_size=pcp_size,
            interleave_size=cp_kv_cache_interleave_size,
        )
        local_query_rows = _select_replicated_query_row_mapping_for_pcp_rank(
            rank_major_query_rows,
            PCP_AXIS_NAME,
            pcp_size,
        )
        query_row_valid = local_query_rows.is_valid
        cache_rank = local_query_rows.source_ranks[0]
        block_tables = _reshape_metadata_page_indices_jax(
            page_indices, kv_lens.shape[0]
        )
        writeback_mapping = _compute_pcp_kv_writeback_mapping(
            rank_major_query_rows,
            block_tables,
            local_kv_cache_num_blocks=kv_cache.shape[0],
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=cp_kv_cache_interleave_size,
        )
        compact_slot_ids = _compact_writeback_slot_ids_for_cache_rank(
            writeback_mapping, cache_rank
        )
        writeback_count = writeback_mapping.writeback_counts[cache_rank]
        packed_current_kv = _prepare_packed_kv_for_cache(
            k_local, v_local, kv_cache, kv_layout=kv_layout
        )
        if q_local.shape[1] % k_local.shape[1] != 0:
            raise ValueError("Q heads must be divisible by KV heads.")
        q_per_kv = q_local.shape[1] // k_local.shape[1]
        q_streaming = q_local.reshape(
            q_local.shape[0], k_local.shape[1], q_per_kv, q_local.shape[2]
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
        attention_kv_cache = _reshape_packed_kv_cache_for_attention(kv_cache, kv_layout)
        segment_descriptors, num_segments = _build_writeback_segment_descriptors(
            compact_slot_ids,
            writeback_count,
            kv_layout=kv_layout,
            page_size=page_size,
        )
        output, kv_cache = (
            pcp_streaming_attention_page_groups_packed_local_from_metadata(
                q_streaming,
                packed_current_kv,
                attention_kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                segment_descriptors,
                num_segments,
                pcp_size=pcp_size,
                interleave_size=cp_kv_cache_interleave_size,
                sm_scale=sm_scale,
                q_block_size=q_block_size,
                q_compute_size=q_compute_size,
                k_scale=k_scale,
                v_scale=v_scale,
                kv_layout=kv_layout,
                mesh_axis_names=tuple(mesh.axis_names),
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

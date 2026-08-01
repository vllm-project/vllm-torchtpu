# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Composite wrappers for PCP streaming RPA."""

import math

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.batched_rpa import \
    wrapper as batched_rpa_wrapper
from vllm_torchtpu.kernels.experimental.batched_rpa.utils import \
    get_dtype_packing
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.kernel import (
    PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE,
    pcp_streaming_attention_page_groups_packed_local_from_metadata)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import \
    _reshape_metadata_page_indices_jax

PCP_AXIS_NAME = "pcp"


def _update_local_paged_kv_cache(
    kv_cache: jax.Array,
    k: jax.Array,
    v: jax.Array,
    slot_ids: jax.Array,
) -> jax.Array:
    """Write local K/V rows into the local paged KV cache shard."""
    dummy_q = jnp.zeros_like(k)
    _, packed_kv = batched_rpa_wrapper.prepare_inputs(dummy_q, k, v, k.dtype,
                                                      kv_cache.dtype)
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
                f"{packed_kv.shape[1:]} vs {kv_cache.shape[2:]}.")
        packed_kv = packed_kv.reshape(
            (packed_kv.shape[0], *kv_cache.shape[2:]))
    packed_kv = packed_kv.astype(kv_cache.dtype)
    page_size = kv_cache.shape[1]
    valid = slot_ids >= 0
    page = jnp.where(valid, slot_ids // page_size, kv_cache.shape[0])
    offset = jnp.where(valid, slot_ids % page_size, 0)
    return kv_cache.at[page, offset].set(packed_kv, mode="drop")


def _reshape_packed_kv_cache_for_attention(kv_cache: jax.Array) -> jax.Array:
    expected_packing = get_dtype_packing(kv_cache.dtype)
    if kv_cache.shape[3] == expected_packing:
        return kv_cache
    raise ValueError("KV cache layout packing does not match dtype packing: "
                     f"shape={kv_cache.shape}, "
                     f"expected_packing={expected_packing}.")


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
        raise ValueError("Replicated PCP tensor axis must be divisible by "
                         f"pcp_size: shape={tensor.shape}, axis={axis}, "
                         f"pcp_size={pcp_size}.")
    shard_size = tensor.shape[axis] // pcp_size
    exchanged = jax.lax.all_to_all(tensor,
                                   axis_name=pcp_axis,
                                   split_axis=axis,
                                   concat_axis=axis,
                                   tiled=True)
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
        raise ValueError(
            f"Expected positive interleave_size, got {interleave_size}.")

    positions = positions.astype(jnp.int32)
    token_req_indices = token_req_indices.astype(jnp.int32)
    virtual_block_size = block_size * cp_size
    block_indices = positions // virtual_block_size
    block_numbers = block_tables[token_req_indices,
                                 block_indices].astype(jnp.int32)

    virtual_offsets = positions - block_indices * virtual_block_size
    is_local = ((virtual_offsets // interleave_size) % cp_size) == cp_rank
    local_offsets = ((virtual_offsets //
                      (cp_size * interleave_size)) * interleave_size +
                     (virtual_offsets % interleave_size))
    slot_ids = block_numbers * block_size + local_offsets
    slot_ids = jnp.where(is_local, slot_ids, -1)
    return is_local, slot_ids.astype(jnp.int32)


def _pcp_rank_token_count_before(position: jax.Array, pcp_rank: jax.Array,
                                 pcp_size: int,
                                 interleave_size: int) -> jax.Array:
    cycle = pcp_size * interleave_size
    full_cycles = position // cycle
    cycle_offset = position - full_cycles * cycle
    rank_start = pcp_rank * interleave_size
    rank_tokens_in_partial = jnp.clip(cycle_offset - rank_start, 0,
                                      interleave_size)
    return full_cycles * interleave_size + rank_tokens_in_partial


def compute_pcp_local_slot_ids_from_metadata(
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    local_padded_tokens: int,
    local_kv_cache_num_blocks: int,
    page_size: int,
    pcp_size: int,
    pcp_rank: int | jax.Array,
    interleave_size: int,
) -> jax.Array:
    """Rebuild local KV-cache slot ids from standard PCP attention metadata."""
    if local_padded_tokens <= 0:
        raise ValueError("local_padded_tokens must be positive.")
    if local_kv_cache_num_blocks <= 0:
        raise ValueError("local_kv_cache_num_blocks must be positive.")
    if page_size <= 0:
        raise ValueError("page_size must be positive.")
    if pcp_size <= 0:
        raise ValueError("pcp_size must be positive.")
    if interleave_size <= 0:
        raise ValueError("interleave_size must be positive.")
    if interleave_size > page_size or page_size % interleave_size != 0:
        raise NotImplementedError(
            "PCP slot id reconstruction requires page_size to be divisible "
            f"by interleave_size: {page_size=} {interleave_size=}.")

    kv_lens = jnp.asarray(kv_lens, dtype=jnp.int32)
    cu_q_lens = jnp.asarray(cu_q_lens, dtype=jnp.int32)
    page_indices = jnp.asarray(page_indices, dtype=jnp.int32)
    distribution = jnp.asarray(distribution, dtype=jnp.int32)
    max_num_reqs = int(kv_lens.shape[0])
    block_tables = _reshape_metadata_page_indices_jax(page_indices,
                                                      max_num_reqs)
    block_tables = jnp.mod(
        block_tables,
        jnp.asarray(local_kv_cache_num_blocks, dtype=jnp.int32),
    )
    pages_per_seq = block_tables.shape[1]

    q_lens_raw = cu_q_lens[1:max_num_reqs + 1] - cu_q_lens[:max_num_reqs]
    active_num_reqs = jnp.clip(distribution[2], 0, max_num_reqs)
    req_indices = jnp.arange(max_num_reqs, dtype=jnp.int32)
    active_req_mask = jnp.logical_and(req_indices < active_num_reqs, q_lens_raw
                                      > 0)
    q_lens = jnp.where(active_req_mask, q_lens_raw, 0)
    q_starts = jnp.where(active_req_mask, kv_lens - q_lens, 0)
    q_ends = q_starts + q_lens
    pcp_rank = jnp.asarray(pcp_rank, dtype=jnp.int32)

    local_counts = (_pcp_rank_token_count_before(q_ends, pcp_rank, pcp_size,
                                                 interleave_size) -
                    _pcp_rank_token_count_before(q_starts, pcp_rank, pcp_size,
                                                 interleave_size))
    local_counts = jnp.where(q_lens > 0, local_counts, 0)
    req_ends = jnp.cumsum(local_counts)
    total_local_tokens = req_ends[-1]

    local_token_indices = jnp.arange(local_padded_tokens, dtype=jnp.int32)
    token_req_indices = jnp.sum(
        local_token_indices[:, None] >= req_ends[None, :],
        axis=1,
        dtype=jnp.int32,
    )
    valid = local_token_indices < total_local_tokens
    safe_req_indices = jnp.minimum(token_req_indices, max_num_reqs - 1)
    req_local_starts = req_ends[safe_req_indices] - local_counts[
        safe_req_indices]
    req_rank_offsets = local_token_indices - req_local_starts

    cycle = pcp_size * interleave_size
    req_q_starts = q_starts[safe_req_indices]
    req_q_ends = q_ends[safe_req_indices]
    rank_chunk_offset = pcp_rank * interleave_size
    first_chunk = ((req_q_starts // cycle) * cycle + rank_chunk_offset)
    first_chunk = jnp.where(first_chunk + interleave_size <= req_q_starts,
                            first_chunk + cycle, first_chunk)
    first_overlap_start = jnp.maximum(first_chunk, req_q_starts)
    first_overlap_end = jnp.minimum(first_chunk + interleave_size, req_q_ends)
    first_overlap_len = jnp.maximum(first_overlap_end - first_overlap_start, 0)

    after_first = req_rank_offsets - first_overlap_len
    positions_in_first = first_overlap_start + req_rank_offsets
    remaining = jnp.maximum(after_first, 0)
    positions_after_first = (first_chunk + cycle +
                             (remaining // interleave_size) * cycle +
                             (remaining % interleave_size))
    positions = jnp.where(req_rank_offsets < first_overlap_len,
                          positions_in_first, positions_after_first)
    positions = jnp.where(valid, positions, 0)

    virtual_block_size = page_size * pcp_size
    block_indices = positions // virtual_block_size
    safe_block_indices = jnp.clip(block_indices, 0, pages_per_seq - 1)
    block_numbers = block_tables[safe_req_indices, safe_block_indices]
    virtual_offsets = positions - block_indices * virtual_block_size
    local_offsets = ((virtual_offsets //
                      (pcp_size * interleave_size)) * interleave_size +
                     (virtual_offsets % interleave_size))
    slot_ids = block_numbers * page_size + local_offsets
    return jnp.where(valid, slot_ids, -1).astype(jnp.int32)


def compute_pcp_rank_major_slot_ids_from_metadata(
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    local_padded_tokens: int,
    local_kv_cache_num_blocks: int,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
) -> jax.Array:
    """Return rank-major PCP slot ids shaped like the global sharded Q axis."""
    ranks = jnp.arange(pcp_size, dtype=jnp.int32)
    slot_ids_by_rank = jax.vmap(
        lambda pcp_rank: compute_pcp_local_slot_ids_from_metadata(
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            local_padded_tokens=local_padded_tokens,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            interleave_size=interleave_size,
        ))(ranks)
    return slot_ids_by_rank.reshape((pcp_size * local_padded_tokens, ))


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
    max_context_tokens: int | None = None,
    update_kv_cache: bool = True,
    cp_kv_cache_interleave_size: int = 0,
    q_block_size: int = PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE,
    q_compute_size: int | None = None,
    return_local_shards: bool = False,
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
        raise NotImplementedError(
            "PCP streaming RPA supports full attention only.")
    if q_scale is not None:
        raise NotImplementedError(
            "PCP streaming RPA does not support quantized Q scales.")
    if (k_scale is None) != (v_scale is None):
        raise ValueError(
            "PCP streaming RPA requires k_scale and v_scale to be both "
            "None or both non-None.")
    if not update_kv_cache:
        raise NotImplementedError(
            "PCP streaming RPA requires update_kv_cache=True.")
    if (cp_kv_cache_interleave_size > kv_cache.shape[1]
            or kv_cache.shape[1] % cp_kv_cache_interleave_size != 0):
        raise NotImplementedError(
            "PCP streaming RPA requires page_size to be divisible by "
            "cp_kv_cache_interleave_size: "
            f"page_size={kv_cache.shape[1]} "
            f"{cp_kv_cache_interleave_size=}.")
    if q.shape[0] % pcp_size != 0:
        raise ValueError("PCP streaming RPA requires q tokens to be evenly "
                         f"sharded across PCP ranks: {q.shape[0]=} "
                         f"{pcp_size=}.")
    if kv_cache.shape[0] % pcp_size != 0:
        raise ValueError("PCP streaming RPA requires KV cache blocks to be "
                         f"evenly sharded across PCP ranks: "
                         f"{kv_cache.shape[0]=} {pcp_size=}.")
    qkv_spec = P(PCP_AXIS_NAME)
    metadata_spec = P()
    in_specs = (
        qkv_spec,
        qkv_spec,
        qkv_spec,
        qkv_spec,
        metadata_spec,
        metadata_spec,
        metadata_spec,
        metadata_spec,
    )
    out_specs = (qkv_spec, qkv_spec)
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

    def _pcp_ragged_paged_attention(q_local, k_local, v_local, kv_cache,
                                    kv_lens, page_indices, cu_q_lens,
                                    distribution):
        rank_major_slot_ids = compute_pcp_rank_major_slot_ids_from_metadata(
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            local_padded_tokens=q_local.shape[0],
            local_kv_cache_num_blocks=kv_cache.shape[0],
            page_size=kv_cache.shape[1],
            pcp_size=pcp_size,
            interleave_size=cp_kv_cache_interleave_size,
        )
        slot_ids = _select_replicated_shard_for_pcp_rank(rank_major_slot_ids,
                                                         PCP_AXIS_NAME,
                                                         pcp_size,
                                                         axis=0)
        kv_cache = _update_local_paged_kv_cache(kv_cache, k_local, v_local,
                                                slot_ids)
        if q_local.shape[1] % k_local.shape[1] != 0:
            raise ValueError("Q heads must be divisible by KV heads.")
        q_per_kv = q_local.shape[1] // k_local.shape[1]
        q_streaming = q_local.reshape(q_local.shape[0], k_local.shape[1],
                                      q_per_kv, q_local.shape[2])
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
        attention_kv_cache = _reshape_packed_kv_cache_for_attention(kv_cache)
        output = pcp_streaming_attention_page_groups_packed_local_from_metadata(
            q_streaming,
            attention_kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            pcp_size=pcp_size,
            interleave_size=cp_kv_cache_interleave_size,
            sm_scale=sm_scale,
            max_context_tokens=max_context_tokens,
            q_block_size=q_block_size,
            q_compute_size=q_compute_size,
            kv_pages_per_block=1,
            k_scale=k_scale,
            v_scale=v_scale,
            mesh_axis_names=tuple(mesh.axis_names),
            pcp_axis_name=PCP_AXIS_NAME,
        )
        output = output[:, :, :q_per_kv, :]
        valid_output_rows = (slot_ids >= 0).reshape((slot_ids.shape[0], ) +
                                                    (1, ) * (output.ndim - 1))
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

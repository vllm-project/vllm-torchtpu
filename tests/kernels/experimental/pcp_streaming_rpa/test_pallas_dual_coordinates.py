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
"""Real Pallas coverage for separate PCP owner and absolute coordinates."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import kernel
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import wrapper as pcp_wrapper
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    build_pcp_rank_major_token_order,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import TilePlanField

pytestmark = pytest.mark.multichip

P = jax.sharding.PartitionSpec
AXIS = "pcp"
PCP_SIZE = 2
PAGE_SIZE = 128
INTERLEAVE_SIZE = 32
Q_BLOCK_SIZE = 128
LOCAL_BUCKET_TOKENS = 128
GLOBAL_BUCKET_TOKENS = PCP_SIZE * LOCAL_BUCKET_TOKENS
HEAD_DIM = 128


def _require_tpu_devices():
    devices = jax.local_devices()
    if len(devices) < PCP_SIZE or devices[0].platform != "tpu":
        pytest.skip("Dual-coordinate PCP Pallas test requires two TPU devices.")
    return devices


def _pack_query_rank_major(q_request_major, token_owner_start):
    q_len = q_request_major.shape[0]
    token_order, inverse_order = build_pcp_rank_major_token_order(
        [q_len],
        pcp_size=PCP_SIZE,
        interleave_size=INTERLEAVE_SIZE,
        padded_num_tokens=GLOBAL_BUCKET_TOKENS,
        token_owner_start_offsets_per_req=[token_owner_start],
    )
    packed = np.zeros(
        (GLOBAL_BUCKET_TOKENS,) + q_request_major.shape[1:], dtype=np.float32
    )
    valid = token_order >= 0
    packed[valid] = q_request_major[token_order[valid]]
    return packed, token_order, inverse_order


def _pack_multi_request_queries(q_requests, token_owner_starts):
    q_lens = [request.shape[0] for request in q_requests]
    request_major = np.concatenate(q_requests, axis=0)
    token_order, inverse_order = build_pcp_rank_major_token_order(
        q_lens,
        pcp_size=PCP_SIZE,
        interleave_size=INTERLEAVE_SIZE,
        padded_num_tokens=GLOBAL_BUCKET_TOKENS,
        token_owner_start_offsets_per_req=token_owner_starts,
    )
    packed = np.zeros(
        (GLOBAL_BUCKET_TOKENS,) + request_major.shape[1:], dtype=np.float32
    )
    valid = token_order >= 0
    packed[valid] = request_major[token_order[valid]]
    return packed, token_order, inverse_order


def _pack_current_kv_rank_major(k_requests, v_requests, absolute_starts, token_order):
    current_k = np.concatenate(
        [k_request[start:] for k_request, start in zip(k_requests, absolute_starts)]
    )
    current_v = np.concatenate(
        [v_request[start:] for v_request, start in zip(v_requests, absolute_starts)]
    )
    packed = np.zeros((GLOBAL_BUCKET_TOKENS, 1, 2, HEAD_DIM), dtype=np.float32)
    valid = token_order >= 0
    packed[valid, :, 0, :] = current_k[token_order[valid]]
    packed[valid, :, 1, :] = current_v[token_order[valid]]
    return packed


def _pack_kv_cache(k_request, v_request, num_local_pages):
    cache = np.zeros(
        (PCP_SIZE, num_local_pages, PAGE_SIZE, 1, 2, HEAD_DIM),
        dtype=np.float32,
    )
    virtual_page_size = PCP_SIZE * PAGE_SIZE
    for absolute_pos in range(k_request.shape[0]):
        local_page = absolute_pos // virtual_page_size
        position_in_virtual_page = absolute_pos % virtual_page_size
        chunk_index = position_in_virtual_page // INTERLEAVE_SIZE
        owner_rank = chunk_index % PCP_SIZE
        local_chunk_index = chunk_index // PCP_SIZE
        local_offset = (
            local_chunk_index * INTERLEAVE_SIZE + absolute_pos % INTERLEAVE_SIZE
        )
        cache[owner_rank, local_page, local_offset, 0, 0] = k_request[absolute_pos, 0]
        cache[owner_rank, local_page, local_offset, 0, 1] = v_request[absolute_pos, 0]
    return cache.reshape((PCP_SIZE * num_local_pages, PAGE_SIZE, 1, 2, HEAD_DIM))


def _pack_multi_request_kv_cache(k_requests, v_requests):
    pages_per_request = [
        (request.shape[0] + PCP_SIZE * PAGE_SIZE - 1) // (PCP_SIZE * PAGE_SIZE)
        for request in k_requests
    ]
    request_page_starts = np.cumsum([0] + pages_per_request[:-1])
    total_local_pages = sum(pages_per_request)
    cache = np.zeros(
        (PCP_SIZE, total_local_pages, PAGE_SIZE, 1, 2, HEAD_DIM),
        dtype=np.float32,
    )
    block_tables = np.zeros((len(k_requests), max(pages_per_request)), dtype=np.int32)
    virtual_page_size = PCP_SIZE * PAGE_SIZE
    for request_idx, (k_request, v_request) in enumerate(zip(k_requests, v_requests)):
        request_page_start = int(request_page_starts[request_idx])
        block_tables[request_idx, : pages_per_request[request_idx]] = (
            request_page_start + np.arange(pages_per_request[request_idx])
        )
        for absolute_pos in range(k_request.shape[0]):
            local_page = request_page_start + absolute_pos // virtual_page_size
            position_in_virtual_page = absolute_pos % virtual_page_size
            chunk_index = position_in_virtual_page // INTERLEAVE_SIZE
            owner_rank = chunk_index % PCP_SIZE
            local_chunk_index = chunk_index // PCP_SIZE
            local_offset = (
                local_chunk_index * INTERLEAVE_SIZE + absolute_pos % INTERLEAVE_SIZE
            )
            cache[owner_rank, local_page, local_offset, 0, 0] = k_request[
                absolute_pos, 0
            ]
            cache[owner_rank, local_page, local_offset, 0, 1] = v_request[
                absolute_pos, 0
            ]
    return (
        cache.reshape((PCP_SIZE * total_local_pages, PAGE_SIZE, 1, 2, HEAD_DIM)),
        block_tables,
        total_local_pages,
    )


def _naive_causal_attention(q, k, v, request_absolute_start, sm_scale):
    q_positions = request_absolute_start + np.arange(q.shape[0])
    scores = np.einsum("thqd,shd->thqs", q, k) * sm_scale
    causal_mask = q_positions[:, None] >= np.arange(k.shape[0])[None, :]
    scores = np.where(causal_mask[:, None, None, :], scores, -np.inf)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    return np.einsum("thqs,shd->thqd", probabilities, v)


def _run_real_pallas(
    q, current_kv, kv_cache, kv_lens, page_indices, cu_q_lens, distribution, mesh
):
    pallas_attention = (
        kernel.pcp_streaming_attention_page_groups_packed_local_from_metadata
    )

    def _attention_only(
        q_local,
        current_kv_local,
        kv_cache_local,
        kv_lens_local,
        page_indices_local,
        cu_q_lens_local,
        distribution_local,
    ):
        rank_major_query_rows = (
            pcp_wrapper._compute_pcp_rank_major_query_row_mapping_from_metadata(
                kv_lens_local,
                cu_q_lens_local,
                distribution_local,
                local_padded_tokens=q_local.shape[0],
                pcp_size=PCP_SIZE,
                interleave_size=INTERLEAVE_SIZE,
            )
        )
        block_tables = pcp_wrapper._reshape_metadata_page_indices_jax(
            page_indices_local, kv_lens_local.shape[0]
        )
        writeback_mapping = pcp_wrapper._compute_pcp_kv_writeback_mapping(
            rank_major_query_rows,
            block_tables,
            local_kv_cache_num_blocks=kv_cache_local.shape[0],
            page_size=kv_cache_local.shape[1],
            pcp_size=PCP_SIZE,
            interleave_size=INTERLEAVE_SIZE,
        )
        cache_rank = jax.lax.axis_index(AXIS)
        compact_slot_ids = pcp_wrapper._compact_writeback_slot_ids_for_cache_rank(
            writeback_mapping, cache_rank
        )
        segment_descriptors, num_segments = (
            pcp_wrapper._build_writeback_segment_descriptors(
                compact_slot_ids,
                writeback_mapping.writeback_counts[cache_rank],
            )
        )
        return pallas_attention(
            q_local,
            current_kv_local,
            kv_cache_local,
            kv_lens_local,
            page_indices_local,
            cu_q_lens_local,
            distribution_local,
            segment_descriptors,
            num_segments,
            pcp_size=PCP_SIZE,
            interleave_size=INTERLEAVE_SIZE,
            q_block_size=Q_BLOCK_SIZE,
            q_compute_size=Q_BLOCK_SIZE,
            sm_scale=1.0 / math.sqrt(HEAD_DIM),
            mesh_axis_names=tuple(mesh.axis_names),
            pcp_axis_name=AXIS,
        )

    mapped = jax.shard_map(
        _attention_only,
        mesh=mesh,
        in_specs=(
            P(AXIS, None, None, None),
            P(AXIS, None, None, None),
            P(AXIS, None, None, None, None),
            P(),
            P(),
            P(),
            P(),
        ),
        out_specs=(
            P(AXIS, None, None, None),
            P(AXIS, None, None, None),
        ),
        check_vma=False,
    )
    return jax.jit(mapped)(
        q, current_kv, kv_cache, kv_lens, page_indices, cu_q_lens, distribution
    )


def _cache_row_coordinates(absolute_pos, num_local_pages):
    local_page = absolute_pos // (PCP_SIZE * PAGE_SIZE)
    position_in_virtual_page = absolute_pos % (PCP_SIZE * PAGE_SIZE)
    chunk_index = position_in_virtual_page // INTERLEAVE_SIZE
    owner_rank = chunk_index % PCP_SIZE
    local_chunk_index = chunk_index // PCP_SIZE
    local_offset = local_chunk_index * INTERLEAVE_SIZE + absolute_pos % INTERLEAVE_SIZE
    return owner_rank * num_local_pages + local_page, local_offset


def test_real_pallas_uses_batch_flat_owner_with_long_history_and_padding(
    release_jax_backend,
):
    devices = _require_tpu_devices()
    q_len = 192
    request_absolute_start = 301
    token_owner_start = 0
    kv_len = request_absolute_start + q_len
    assert request_absolute_start % INTERLEAVE_SIZE != 0

    rng = np.random.default_rng(123)
    q_request_major = rng.normal(size=(q_len, 1, 1, HEAD_DIM)).astype(np.float32) * 0.1
    k_request = rng.normal(size=(kv_len, 1, HEAD_DIM)).astype(np.float32) * 0.1
    v_request = rng.normal(size=(kv_len, 1, HEAD_DIM)).astype(np.float32) * 0.1
    q_packed, token_order, inverse_order = _pack_query_rank_major(
        q_request_major, token_owner_start
    )
    num_local_pages = (kv_len + PCP_SIZE * PAGE_SIZE - 1) // (PCP_SIZE * PAGE_SIZE)
    kv_cache = _pack_kv_cache(k_request, v_request, num_local_pages)

    schedule_inputs = kernel.build_pcp_streaming_schedule_inputs_from_metadata_jax(
        kv_lens=np.array([kv_len], dtype=np.int32),
        page_indices=np.arange(num_local_pages, dtype=np.int32),
        cu_q_lens=np.array([0, q_len], dtype=np.int32),
        distribution=np.array([0, 0, 1], dtype=np.int32),
        global_bucket_tokens=GLOBAL_BUCKET_TOKENS,
        local_kv_cache_num_blocks=num_local_pages,
        page_size=PAGE_SIZE,
        pcp_size=PCP_SIZE,
        interleave_size=INTERLEAVE_SIZE,
        q_block_size=Q_BLOCK_SIZE,
    )
    tile_plan = np.asarray(jax.device_get(schedule_inputs[0]))
    logical_plan = tile_plan[:, 0, : PCP_SIZE * TilePlanField.NUM_FIELDS].reshape(
        tile_plan.shape[0], PCP_SIZE, TilePlanField.NUM_FIELDS
    )
    current_meta = np.asarray(jax.device_get(schedule_inputs[4]))
    history_meta = np.asarray(jax.device_get(schedule_inputs[7]))
    assert int(current_meta[1]) > 0
    assert int(history_meta[1]) > 0
    assert int(logical_plan[..., TilePlanField.Q_TILE_SIZE].max()) > (INTERLEAVE_SIZE)

    q_bf16 = jnp.asarray(q_packed, dtype=jnp.bfloat16)
    current_kv = _pack_current_kv_rank_major(
        [k_request], [v_request], [request_absolute_start], token_order
    )
    current_kv_bf16 = jnp.asarray(current_kv, dtype=jnp.bfloat16)
    for absolute_pos in range(request_absolute_start, kv_len):
        cache_block, local_offset = _cache_row_coordinates(
            absolute_pos, num_local_pages
        )
        kv_cache[cache_block, local_offset] = 0
    kv_cache_bf16 = jnp.asarray(kv_cache, dtype=jnp.bfloat16)
    kv_lens = jnp.asarray([kv_len], dtype=jnp.int32)
    page_indices = jnp.arange(num_local_pages, dtype=jnp.int32)
    cu_q_lens = jnp.asarray([0, q_len], dtype=jnp.int32)
    distribution = jnp.asarray([0, 0, 1], dtype=jnp.int32)
    mesh = jax.sharding.Mesh(np.asarray(devices[:PCP_SIZE]), (AXIS,))

    output, new_cache = _run_real_pallas(
        q_bf16,
        current_kv_bf16,
        kv_cache_bf16,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        mesh,
    )
    output.block_until_ready()
    new_cache.block_until_ready()

    output_np = np.asarray(jax.device_get(output), dtype=np.float32)
    output_request_major = output_np[inverse_order]
    q_bf16_np = np.asarray(jax.device_get(q_bf16), dtype=np.float32)
    expected = _naive_causal_attention(
        np.asarray(q_request_major, dtype=jnp.bfloat16).astype(np.float32),
        np.asarray(k_request, dtype=jnp.bfloat16).astype(np.float32),
        np.asarray(v_request, dtype=jnp.bfloat16).astype(np.float32),
        request_absolute_start,
        1.0 / math.sqrt(HEAD_DIM),
    )
    np.testing.assert_allclose(output_request_major, expected, rtol=2e-2, atol=5e-4)

    local_valid_rows = np.count_nonzero(
        token_order.reshape(PCP_SIZE, LOCAL_BUCKET_TOKENS) >= 0, axis=1
    )
    output_by_rank = output_np.reshape((PCP_SIZE, LOCAL_BUCKET_TOKENS, 1, 1, HEAD_DIM))
    for rank, valid_rows in enumerate(local_valid_rows):
        np.testing.assert_array_equal(output_by_rank[rank, valid_rows:], 0)
    assert np.count_nonzero(token_order < 0) == (GLOBAL_BUCKET_TOKENS - q_len)
    assert np.all(q_bf16_np[token_order < 0] == 0)

    new_cache_np = np.asarray(jax.device_get(new_cache))
    for absolute_pos in range(request_absolute_start, kv_len):
        cache_block, local_offset = _cache_row_coordinates(
            absolute_pos, num_local_pages
        )
        expected_row = np.stack(
            (k_request[absolute_pos], v_request[absolute_pos]),
            axis=1,
        ).astype(jnp.bfloat16)
        np.testing.assert_array_equal(
            new_cache_np[cache_block, local_offset], expected_row
        )


def test_real_pallas_no_q_rank_captures_current_kv_for_cache_owner(release_jax_backend):
    devices = _require_tpu_devices()
    q_len = 1
    request_absolute_start = INTERLEAVE_SIZE + 1
    token_owner_start = 0
    kv_len = request_absolute_start + q_len

    # Batch-flat ownership puts the only fresh token on rank 0, while its
    # request-absolute cache position belongs to rank 1.
    assert token_owner_start // INTERLEAVE_SIZE % PCP_SIZE == 0
    assert request_absolute_start // INTERLEAVE_SIZE % PCP_SIZE == 1

    rng = np.random.default_rng(456)
    q_request_major = rng.normal(size=(q_len, 1, 1, HEAD_DIM)).astype(np.float32) * 0.1
    k_request = rng.normal(size=(kv_len, 1, HEAD_DIM)).astype(np.float32) * 0.1
    v_request = rng.normal(size=(kv_len, 1, HEAD_DIM)).astype(np.float32) * 0.1
    q_packed, token_order, inverse_order = _pack_query_rank_major(
        q_request_major, token_owner_start
    )
    num_local_pages = (kv_len + PCP_SIZE * PAGE_SIZE - 1) // (PCP_SIZE * PAGE_SIZE)
    kv_cache = _pack_kv_cache(k_request, v_request, num_local_pages)
    cache_block, local_offset = _cache_row_coordinates(
        request_absolute_start, num_local_pages
    )
    kv_cache[cache_block, local_offset] = 0
    current_kv = _pack_current_kv_rank_major(
        [k_request], [v_request], [request_absolute_start], token_order
    )

    mesh = jax.sharding.Mesh(np.asarray(devices[:PCP_SIZE]), (AXIS,))
    output, new_cache = _run_real_pallas(
        jnp.asarray(q_packed, dtype=jnp.bfloat16),
        jnp.asarray(current_kv, dtype=jnp.bfloat16),
        jnp.asarray(kv_cache, dtype=jnp.bfloat16),
        jnp.asarray([kv_len], dtype=jnp.int32),
        jnp.arange(num_local_pages, dtype=jnp.int32),
        jnp.asarray([0, q_len], dtype=jnp.int32),
        jnp.asarray([0, 0, 1], dtype=jnp.int32),
        mesh,
    )
    output.block_until_ready()
    new_cache.block_until_ready()

    output_np = np.asarray(jax.device_get(output), dtype=np.float32)
    expected = _naive_causal_attention(
        np.asarray(q_request_major, dtype=jnp.bfloat16).astype(np.float32),
        np.asarray(k_request, dtype=jnp.bfloat16).astype(np.float32),
        np.asarray(v_request, dtype=jnp.bfloat16).astype(np.float32),
        request_absolute_start,
        1.0 / math.sqrt(HEAD_DIM),
    )
    np.testing.assert_allclose(output_np[inverse_order], expected, rtol=2e-2, atol=5e-4)

    output_by_rank = output_np.reshape((PCP_SIZE, LOCAL_BUCKET_TOKENS, 1, 1, HEAD_DIM))
    np.testing.assert_array_equal(output_by_rank[1], 0)

    expected_current = np.stack(
        (k_request[request_absolute_start], v_request[request_absolute_start]),
        axis=1,
    )
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(new_cache))[cache_block, local_offset],
        expected_current.astype(jnp.bfloat16),
    )


def test_real_wrapper_writes_ring_captured_kv_before_history(release_jax_backend):
    devices = _require_tpu_devices()
    q_len = 1
    request_absolute_start = INTERLEAVE_SIZE + 1
    kv_len = request_absolute_start + q_len

    rng = np.random.default_rng(789)
    q_request_major = rng.normal(size=(q_len, 1, 1, HEAD_DIM)).astype(np.float32) * 0.1
    k_request = rng.normal(size=(kv_len, 1, HEAD_DIM)).astype(np.float32) * 0.1
    v_request = rng.normal(size=(kv_len, 1, HEAD_DIM)).astype(np.float32) * 0.1
    q_packed, token_order, inverse_order = _pack_query_rank_major(
        q_request_major, token_owner_start=0
    )

    current_k = k_request[request_absolute_start:]
    current_v = v_request[request_absolute_start:]
    k_packed = np.zeros((GLOBAL_BUCKET_TOKENS, 1, HEAD_DIM), dtype=np.float32)
    v_packed = np.zeros_like(k_packed)
    valid = token_order >= 0
    k_packed[valid] = current_k[token_order[valid]]
    v_packed[valid] = current_v[token_order[valid]]

    num_local_pages = (kv_len + PCP_SIZE * PAGE_SIZE - 1) // (PCP_SIZE * PAGE_SIZE)
    kv_cache = _pack_kv_cache(k_request, v_request, num_local_pages)
    cache_owner = request_absolute_start // INTERLEAVE_SIZE % PCP_SIZE
    local_offset = request_absolute_start % INTERLEAVE_SIZE
    cache_block = cache_owner * num_local_pages
    kv_cache[cache_block, local_offset] = 0
    cache_before = kv_cache.copy()

    mesh = jax.sharding.Mesh(np.asarray(devices[:PCP_SIZE]), (AXIS,))
    output, new_cache = pcp_wrapper.sharded_pcp_ragged_paged_attention(
        mesh=mesh,
        q=jnp.asarray(q_packed[:, :, 0, :], dtype=jnp.bfloat16),
        k=jnp.asarray(k_packed, dtype=jnp.bfloat16),
        v=jnp.asarray(v_packed, dtype=jnp.bfloat16),
        kv_cache=jnp.asarray(kv_cache, dtype=jnp.bfloat16),
        kv_lens=jnp.asarray([kv_len], dtype=jnp.int32),
        page_indices=jnp.arange(num_local_pages, dtype=jnp.int32),
        cu_q_lens=jnp.asarray([0, q_len], dtype=jnp.int32),
        distribution=jnp.asarray([0, 0, 1], dtype=jnp.int32),
        attention_sink=None,
        sm_scale=1.0 / math.sqrt(HEAD_DIM),
        cp_kv_cache_interleave_size=INTERLEAVE_SIZE,
        q_block_size=Q_BLOCK_SIZE,
        q_compute_size=Q_BLOCK_SIZE,
    )
    output.block_until_ready()
    new_cache.block_until_ready()

    output_np = np.asarray(jax.device_get(output), dtype=np.float32)
    expected_output = _naive_causal_attention(
        np.asarray(q_request_major, dtype=jnp.bfloat16).astype(np.float32),
        np.asarray(k_request, dtype=jnp.bfloat16).astype(np.float32),
        np.asarray(v_request, dtype=jnp.bfloat16).astype(np.float32),
        request_absolute_start,
        1.0 / math.sqrt(HEAD_DIM),
    )
    np.testing.assert_allclose(
        output_np[inverse_order], expected_output[:, :, 0, :], rtol=2e-2, atol=5e-4
    )

    new_cache_np = np.asarray(jax.device_get(new_cache))
    expected_row = np.stack(
        (k_request[request_absolute_start], v_request[request_absolute_start]),
        axis=1,
    ).astype(jnp.bfloat16)
    np.testing.assert_array_equal(new_cache_np[cache_block, local_offset], expected_row)
    unchanged = np.ones(new_cache_np.shape[:2], dtype=bool)
    unchanged[cache_block, local_offset] = False
    np.testing.assert_array_equal(
        new_cache_np[unchanged], cache_before.astype(jnp.bfloat16)[unchanged]
    )


def test_real_pallas_preserves_state_when_first_ring_page_is_fully_masked(
    release_jax_backend,
):
    devices = _require_tpu_devices()
    q_lens = [96, 96]
    token_owner_starts = [0, q_lens[0]]
    request_absolute_starts = [0, 0]
    # Request 1 starts one ownership chunk out of phase with its absolute KV
    # coordinates.  On rank 1 its first local Q chunk is abs[0:32], while the
    # first ring source only contains KV abs[32:64] and is fully causal-masked.
    assert token_owner_starts[1] % (PCP_SIZE * INTERLEAVE_SIZE) == (INTERLEAVE_SIZE)

    rng = np.random.default_rng(321)
    q_requests = [
        rng.normal(size=(q_len, 1, 1, HEAD_DIM)).astype(np.float32) * 0.1
        for q_len in q_lens
    ]
    k_requests = [
        rng.normal(size=(q_len, 1, HEAD_DIM)).astype(np.float32) * 0.1
        for q_len in q_lens
    ]
    v_requests = [
        rng.normal(size=(q_len, 1, HEAD_DIM)).astype(np.float32) * 0.1
        for q_len in q_lens
    ]
    q_packed, token_order, inverse_order = _pack_multi_request_queries(
        q_requests, token_owner_starts
    )
    kv_cache, block_tables, total_local_pages = _pack_multi_request_kv_cache(
        k_requests, v_requests
    )

    q_bf16 = jnp.asarray(q_packed, dtype=jnp.bfloat16)
    current_kv = _pack_current_kv_rank_major(
        k_requests, v_requests, request_absolute_starts, token_order
    )
    current_kv_bf16 = jnp.asarray(current_kv, dtype=jnp.bfloat16)
    kv_cache_bf16 = jnp.asarray(kv_cache, dtype=jnp.bfloat16)
    kv_lens = jnp.asarray(q_lens, dtype=jnp.int32)
    page_indices = jnp.asarray(block_tables, dtype=jnp.int32)
    cu_q_lens = jnp.asarray([0, q_lens[0], sum(q_lens)], dtype=jnp.int32)
    distribution = jnp.asarray([0, 0, len(q_lens)], dtype=jnp.int32)
    mesh = jax.sharding.Mesh(np.asarray(devices[:PCP_SIZE]), (AXIS,))

    assert total_local_pages == 2
    output, new_cache = _run_real_pallas(
        q_bf16,
        current_kv_bf16,
        kv_cache_bf16,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        mesh,
    )
    output.block_until_ready()
    new_cache.block_until_ready()

    output_np = np.asarray(jax.device_get(output), dtype=np.float32)
    output_request_major = output_np[inverse_order]
    q_offset = 0
    for request_idx, q_len in enumerate(q_lens):
        expected = _naive_causal_attention(
            np.asarray(q_requests[request_idx], dtype=jnp.bfloat16).astype(np.float32),
            np.asarray(k_requests[request_idx], dtype=jnp.bfloat16).astype(np.float32),
            np.asarray(v_requests[request_idx], dtype=jnp.bfloat16).astype(np.float32),
            request_absolute_starts[request_idx],
            1.0 / math.sqrt(HEAD_DIM),
        )
        actual = output_request_major[q_offset : q_offset + q_len]
        np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=7e-4)
        q_offset += q_len

    # This exact slice was all zeros before empty-page online-softmax updates
    # were made a no-op.
    request_1 = output_request_major[q_lens[0] : sum(q_lens)]
    assert np.count_nonzero(request_1[:INTERLEAVE_SIZE]) > 0
    assert np.count_nonzero(token_order < 0) == (GLOBAL_BUCKET_TOKENS - sum(q_lens))

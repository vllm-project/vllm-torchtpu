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
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
    stage_plan as pcp_stage_plan,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import wrapper as pcp_wrapper
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    build_pcp_rank_major_token_order,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
    TilePlanField,
    build_pcp_streaming_schedule_inputs_from_metadata_jax,
)

pytestmark = pytest.mark.multichip

P = jax.sharding.PartitionSpec
AXIS = "pcp"


def _generate_multi_request_arrays(
    q_lens: list[int],
    kv_lens_list: list[int],
    request_absolute_starts: list[int],
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    """Generate two-request Q, K, V arrays and zeroed current-tail history views."""
    rng = np.random.default_rng(2028)
    total_q = sum(q_lens)
    total_kv = sum(kv_lens_list)
    global_q = rng.normal(size=(total_q, 1, 1, HEAD_DIM)).astype(np.float32)
    global_k = rng.normal(size=(total_kv, 1, HEAD_DIM)).astype(np.float32)
    global_v = rng.normal(size=(total_kv, 1, HEAD_DIM)).astype(np.float32)

    q_requests = [global_q[: q_lens[0]], global_q[q_lens[0] :]]
    k_requests = [global_k[: kv_lens_list[0]], global_k[kv_lens_list[0] :]]
    v_requests = [global_v[: kv_lens_list[0]], global_v[kv_lens_list[0] :]]

    k_history_only = []
    v_history_only = []
    for k_req, v_req, start in zip(
        k_requests, v_requests, request_absolute_starts, strict=True
    ):
        k_hist = k_req.copy()
        v_hist = v_req.copy()
        k_hist[start:] = 0.0
        v_hist[start:] = 0.0
        k_history_only.append(k_hist)
        v_history_only.append(v_hist)
    return q_requests, k_requests, v_requests, k_history_only, v_history_only


def _pack_multi_request_kv_cache_vectorized(
    k_requests: list[np.ndarray],
    v_requests: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pack multi-request K/V history into striped local PCP pages using NumPy."""
    virtual_page_size = PCP_SIZE * PAGE_SIZE
    chunks_per_page = PAGE_SIZE // INTERLEAVE_SIZE
    pages_per_request = [
        (req.shape[0] + virtual_page_size - 1) // virtual_page_size
        for req in k_requests
    ]
    request_page_starts = np.cumsum([0, *pages_per_request[:-1]])
    total_local_pages = sum(pages_per_request)
    cache = np.zeros(
        (PCP_SIZE, total_local_pages, PAGE_SIZE, 1, 2, HEAD_DIM),
        dtype=np.float32,
    )
    block_tables = np.zeros((len(k_requests), max(pages_per_request)), dtype=np.int32)
    for req_idx, (k_req, v_req) in enumerate(zip(k_requests, v_requests, strict=True)):
        num_pages = pages_per_request[req_idx]
        req_page_start = int(request_page_starts[req_idx])
        block_tables[req_idx, :num_pages] = req_page_start + np.arange(
            num_pages, dtype=np.int32
        )
        padded_tokens = num_pages * virtual_page_size
        k_pad = np.zeros((padded_tokens, HEAD_DIM), dtype=np.float32)
        v_pad = np.zeros((padded_tokens, HEAD_DIM), dtype=np.float32)
        k_pad[: k_req.shape[0]] = k_req[:, 0, :]
        v_pad[: v_req.shape[0]] = v_req[:, 0, :]
        k_grid = k_pad.reshape(
            num_pages, chunks_per_page, PCP_SIZE, INTERLEAVE_SIZE, HEAD_DIM
        )
        v_grid = v_pad.reshape(
            num_pages, chunks_per_page, PCP_SIZE, INTERLEAVE_SIZE, HEAD_DIM
        )
        cache[:, req_page_start : req_page_start + num_pages, :, 0, 0, :] = (
            np.transpose(k_grid, (2, 0, 1, 3, 4)).reshape(
                PCP_SIZE, num_pages, PAGE_SIZE, HEAD_DIM
            )
        )
        cache[:, req_page_start : req_page_start + num_pages, :, 0, 1, :] = (
            np.transpose(v_grid, (2, 0, 1, 3, 4)).reshape(
                PCP_SIZE, num_pages, PAGE_SIZE, HEAD_DIM
            )
        )
    return (
        cache.reshape((PCP_SIZE * total_local_pages, PAGE_SIZE, 1, 2, HEAD_DIM)),
        block_tables,
        total_local_pages,
    )


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
    q: jax.Array,
    current_kv: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    mesh: jax.sharding.Mesh,
    *,
    vpage_group_size: int = kernel.PCP_STREAMING_RPA_VPAGE_GROUP_SIZE,
) -> tuple[jax.Array, jax.Array]:
    pallas_attention = (
        kernel.pcp_streaming_attention_page_groups_packed_local_from_metadata
    )

    def _attention_only(
        q_local: jax.Array,
        current_kv_local: jax.Array,
        kv_cache_local: jax.Array,
        kv_lens_local: jax.Array,
        page_indices_local: jax.Array,
        cu_q_lens_local: jax.Array,
        distribution_local: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Build PCPStagePlan and invoke metadata Splash entry on one PCP shard."""
        local_stage_plan = pcp_stage_plan.build_pcp_local_stage_plan(
            kv_lens_local,
            page_indices_local,
            cu_q_lens_local,
            distribution_local,
            local_tokens=q_local.shape[0],
            local_kv_cache_num_blocks=kv_cache_local.shape[0],
            page_size=kv_cache_local.shape[1],
            pcp_size=PCP_SIZE,
            interleave_size=INTERLEAVE_SIZE,
            vpage_group_size=vpage_group_size,
            pcp_axis_name=AXIS,
        )
        return pallas_attention(
            q_local,
            current_kv_local,
            kv_cache_local,
            kv_lens_local,
            cu_q_lens_local,
            distribution_local,
            local_stage_plan.segment_descriptors,
            local_stage_plan.num_segments,
            stage_plan=local_stage_plan,
            vpage_group_size=vpage_group_size,
            pcp_size=PCP_SIZE,
            interleave_size=INTERLEAVE_SIZE,
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

    schedule_inputs = build_pcp_streaming_schedule_inputs_from_metadata_jax(
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


@pytest.mark.parametrize("request_length", [95, 96, 97, 127])
def test_real_pallas_multi_request_causal_positions(
    release_jax_backend, request_length
):
    devices = _require_tpu_devices()
    q_lens = [request_length, request_length]
    token_owner_starts = [0, q_lens[0]]
    request_absolute_starts = [0, 0]
    # Request 1 starts one ownership chunk out of phase with its absolute KV
    # coordinates.  On rank 1 its first local Q chunk is abs[0:32], while the
    # first ring source only contains KV abs[32:64] and is fully causal-masked.
    # 95/97 additionally exercise packed rows that cross a partial
    # interleave; the 96 case retains the fully masked first-page regression.

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


EXPECTED_NUM_CHUNKS = 6
EXPECTED_ACTIVE_VPAGES_516 = 516


def _assert_two_request_pallas_parity(
    output_request_major: np.ndarray,
    q_requests: list[np.ndarray],
    k_requests: list[np.ndarray],
    v_requests: list[np.ndarray],
    group0_cutoff_tokens: int,
) -> None:
    """Assert bf16 accuracy against FP64 reference and group-0 truncation control."""
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    u_bf16 = 2.0**-8
    u_f32 = 2.0**-24
    q_offset = 0
    for req_idx, q_req in enumerate(q_requests):
        q_len = int(q_req.shape[0])
        start = int(k_requests[req_idx].shape[0]) - q_len
        q_bf16_f64 = np.asarray(q_req, dtype=jnp.bfloat16).astype(np.float64)
        k_bf16_f64 = np.asarray(k_requests[req_idx], dtype=jnp.bfloat16).astype(
            np.float64
        )
        v_bf16_f64 = np.asarray(v_requests[req_idx], dtype=jnp.bfloat16).astype(
            np.float64
        )
        expected = _naive_causal_attention(
            q_bf16_f64, k_bf16_f64, v_bf16_f64, start, sm_scale
        )
        scores = np.einsum("thqd,shd->thqs", q_bf16_f64, k_bf16_f64) * sm_scale
        max_abs_score = float(np.max(np.abs(scores)))
        rms_v = float(np.sqrt(np.mean(v_bf16_f64**2)))
        max_abs_v = float(np.max(np.abs(v_bf16_f64)))
        max_abs_expected = float(np.max(np.abs(expected)))
        tol_bf16 = (
            2.0 * u_bf16 * max_abs_score * rms_v
            + u_bf16 * max_abs_v
            + 16.0 * u_f32 * max_abs_v
            + u_bf16 * max_abs_expected
        )
        actual = output_request_major[q_offset : q_offset + q_len]
        assert float(np.max(np.abs(actual - expected))) <= tol_bf16
        group0_only = (
            _naive_causal_attention(
                q_bf16_f64,
                k_bf16_f64[:group0_cutoff_tokens],
                v_bf16_f64[:group0_cutoff_tokens],
                start,
                sm_scale,
            )
            if req_idx == 0
            else np.zeros_like(expected)
        )
        assert float(np.max(np.abs(group0_only - expected))) > tol_bf16
        q_offset += q_len


@pytest.mark.usefixtures("release_jax_backend")
def test_real_pallas_multi_group_merge_with_two_requests() -> None:
    """Verify real Pallas multi-group LSE merge with vpage_group_size=2."""
    devices = _require_tpu_devices()
    vpage_group_size = 2
    q_lens = [120, 90]
    request_absolute_starts = [580, 560]
    kv_lens_list = [
        start + q_len
        for start, q_len in zip(request_absolute_starts, q_lens, strict=True)
    ]
    (
        q_requests,
        k_requests,
        v_requests,
        k_history_only,
        v_history_only,
    ) = _generate_multi_request_arrays(q_lens, kv_lens_list, request_absolute_starts)
    q_packed, token_order, inverse_order = _pack_multi_request_queries(
        q_requests, [0, q_lens[0]]
    )
    kv_cache_hist, block_tables, total_local_pages = _pack_multi_request_kv_cache(
        k_history_only, v_history_only
    )
    assert total_local_pages == EXPECTED_NUM_CHUNKS
    current_kv = _pack_current_kv_rank_major(
        k_requests, v_requests, request_absolute_starts, token_order
    )
    mesh = jax.sharding.Mesh(np.asarray(devices[:PCP_SIZE]), (AXIS,))
    output, new_cache = _run_real_pallas(
        jnp.asarray(q_packed, dtype=jnp.bfloat16),
        jnp.asarray(current_kv, dtype=jnp.bfloat16),
        jnp.asarray(kv_cache_hist, dtype=jnp.bfloat16),
        jnp.asarray(kv_lens_list, dtype=jnp.int32),
        jnp.asarray(block_tables, dtype=jnp.int32),
        jnp.asarray([0, q_lens[0], sum(q_lens)], dtype=jnp.int32),
        jnp.asarray([0, 0, len(q_lens)], dtype=jnp.int32),
        mesh,
        vpage_group_size=vpage_group_size,
    )
    output.block_until_ready()
    new_cache.block_until_ready()
    output_request_major = np.asarray(jax.device_get(output), dtype=np.float64)[
        inverse_order
    ]
    _assert_two_request_pallas_parity(
        output_request_major,
        q_requests,
        k_requests,
        v_requests,
        group0_cutoff_tokens=vpage_group_size * PCP_SIZE * PAGE_SIZE,
    )


@pytest.mark.usefixtures("release_jax_backend")
def test_real_pallas_rung_ladder_fallback_to_multi_group_128() -> None:
    """Verify production vpage_group_size=128 rung ladder fallback above 512 vpages."""
    devices = _require_tpu_devices()
    vpage_group_size = kernel.PCP_STREAMING_RPA_VPAGE_GROUP_SIZE
    virtual_page_size = PCP_SIZE * PAGE_SIZE
    q_lens = [120, 90]
    kv_lens_list = [260 * virtual_page_size, 256 * virtual_page_size]
    request_absolute_starts = [
        kv_len - q_len for kv_len, q_len in zip(kv_lens_list, q_lens, strict=True)
    ]
    (
        q_requests,
        k_requests,
        v_requests,
        k_history_only,
        v_history_only,
    ) = _generate_multi_request_arrays(q_lens, kv_lens_list, request_absolute_starts)
    q_packed, token_order, inverse_order = _pack_multi_request_queries(
        q_requests, [0, q_lens[0]]
    )
    kv_cache_hist, block_tables, total_local_pages = (
        _pack_multi_request_kv_cache_vectorized(k_history_only, v_history_only)
    )
    assert total_local_pages == EXPECTED_ACTIVE_VPAGES_516
    current_kv = _pack_current_kv_rank_major(
        k_requests, v_requests, request_absolute_starts, token_order
    )
    mesh = jax.sharding.Mesh(np.asarray(devices[:PCP_SIZE]), (AXIS,))
    output, new_cache = _run_real_pallas(
        jnp.asarray(q_packed, dtype=jnp.bfloat16),
        jnp.asarray(current_kv, dtype=jnp.bfloat16),
        jnp.asarray(kv_cache_hist, dtype=jnp.bfloat16),
        jnp.asarray(kv_lens_list, dtype=jnp.int32),
        jnp.asarray(block_tables, dtype=jnp.int32),
        jnp.asarray([0, q_lens[0], sum(q_lens)], dtype=jnp.int32),
        jnp.asarray([0, 0, len(q_lens)], dtype=jnp.int32),
        mesh,
        vpage_group_size=vpage_group_size,
    )
    output.block_until_ready()
    new_cache.block_until_ready()
    output_request_major = np.asarray(jax.device_get(output), dtype=np.float64)[
        inverse_order
    ]
    _assert_two_request_pallas_parity(
        output_request_major,
        q_requests,
        k_requests,
        v_requests,
        group0_cutoff_tokens=vpage_group_size * virtual_page_size,
    )

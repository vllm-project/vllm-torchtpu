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
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

sys.path.insert(0, str(Path(__file__).parent))

from schedule_reference import \
    generate_pcp_streaming_schedule_reference as \
    generate_pcp_streaming_schedule  # noqa: E402

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import \
    kernel as pcp_kernel
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.kernel import (
    pcp_streaming_attention_page_groups,
    pcp_streaming_attention_page_groups_local,
    pcp_streaming_attention_page_groups_packed_local,
    pcp_streaming_attention_single_page_group)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.reference import \
    execute_pcp_streaming_reference
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
    PcpStreamingSchedule, build_pcp_streaming_active_page_groups)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import \
    generate_pcp_streaming_schedule as generate_production_pcp_schedule
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
    generate_pcp_streaming_schedule_from_metadata_host,
    pack_pcp_streaming_schedule_fields)

PCP_SIZE = 4
Q_TILE = 16
PAGE_SIZE = 128
HEAD_DIM = 128
P = jax.sharding.PartitionSpec
AXIS = "pcp"


def _make_single_page_group_schedule():
    shape = (PCP_SIZE, PCP_SIZE, 1)
    req_id = np.zeros(shape, dtype=np.int32)
    kv_page_rank = np.zeros(shape, dtype=np.int32)
    kv_page_idx = np.zeros(shape, dtype=np.int32)
    is_first_kv = np.zeros(shape, dtype=np.int32)
    is_last_kv = np.zeros(shape, dtype=np.int32)
    load_q = np.zeros(shape, dtype=np.int32)
    q_global_start = np.full(shape, 3 * PAGE_SIZE, dtype=np.int32)
    kv_global_start = np.zeros(shape, dtype=np.int32)
    kv_valid_len = np.full(shape, PAGE_SIZE, dtype=np.int32)
    q_hbm_offset = np.zeros(shape, dtype=np.int32)
    q_tile_size = np.full(shape, Q_TILE, dtype=np.int32)
    o_hbm_offset = np.zeros(shape, dtype=np.int32)

    for step in range(PCP_SIZE):
        kv_page_rank[:, step, 0] = step
        kv_global_start[:, step, 0] = step * PAGE_SIZE
    is_first_kv[:, 0, 0] = 1
    is_last_kv[:, PCP_SIZE - 1, 0] = 1
    load_q[:, 0, 0] = 1

    packed_schedule = pack_pcp_streaming_schedule_fields(
        req_id=req_id,
        kv_page_rank=kv_page_rank,
        kv_page_idx=kv_page_idx,
        is_first_kv=is_first_kv,
        is_last_kv=is_last_kv,
        load_q=load_q,
        q_global_start=q_global_start,
        kv_global_start=kv_global_start,
        kv_valid_len=kv_valid_len,
        q_hbm_offset=q_hbm_offset,
        q_tile_size=q_tile_size,
        o_hbm_offset=o_hbm_offset,
    )
    return PcpStreamingSchedule(
        req_id=req_id,
        kv_page_rank=kv_page_rank,
        kv_page_idx=kv_page_idx,
        is_first_kv=is_first_kv,
        is_last_kv=is_last_kv,
        load_q=load_q,
        q_global_start=q_global_start,
        kv_global_start=kv_global_start,
        kv_valid_len=kv_valid_len,
        q_hbm_offset=q_hbm_offset,
        q_tile_size=q_tile_size,
        o_hbm_offset=o_hbm_offset,
        packed_schedule=packed_schedule,
        actual_steps=np.full((PCP_SIZE, ), PCP_SIZE, dtype=np.int32),
        global_actual_steps=np.array([PCP_SIZE], dtype=np.int32),
    )


def _make_two_page_group_schedule():
    steps = 2 * PCP_SIZE
    shape = (PCP_SIZE, steps, 1)
    req_id = np.zeros(shape, dtype=np.int32)
    kv_page_rank = np.zeros(shape, dtype=np.int32)
    kv_page_idx = np.zeros(shape, dtype=np.int32)
    is_first_kv = np.zeros(shape, dtype=np.int32)
    is_last_kv = np.zeros(shape, dtype=np.int32)
    load_q = np.zeros(shape, dtype=np.int32)
    q_global_start = np.full(shape, steps * PAGE_SIZE, dtype=np.int32)
    kv_global_start = np.zeros(shape, dtype=np.int32)
    kv_valid_len = np.full(shape, PAGE_SIZE, dtype=np.int32)
    q_hbm_offset = np.zeros(shape, dtype=np.int32)
    q_tile_size = np.full(shape, Q_TILE, dtype=np.int32)
    o_hbm_offset = np.zeros(shape, dtype=np.int32)

    for step in range(steps):
        kv_page_rank[:, step, 0] = step % PCP_SIZE
        kv_page_idx[:, step, 0] = step // PCP_SIZE
        kv_global_start[:, step, 0] = step * PAGE_SIZE
    is_first_kv[:, 0, 0] = 1
    is_last_kv[:, steps - 1, 0] = 1
    load_q[:, 0, 0] = 1

    packed_schedule = pack_pcp_streaming_schedule_fields(
        req_id=req_id,
        kv_page_rank=kv_page_rank,
        kv_page_idx=kv_page_idx,
        is_first_kv=is_first_kv,
        is_last_kv=is_last_kv,
        load_q=load_q,
        q_global_start=q_global_start,
        kv_global_start=kv_global_start,
        kv_valid_len=kv_valid_len,
        q_hbm_offset=q_hbm_offset,
        q_tile_size=q_tile_size,
        o_hbm_offset=o_hbm_offset,
    )
    return PcpStreamingSchedule(
        req_id=req_id,
        kv_page_rank=kv_page_rank,
        kv_page_idx=kv_page_idx,
        is_first_kv=is_first_kv,
        is_last_kv=is_last_kv,
        load_q=load_q,
        q_global_start=q_global_start,
        kv_global_start=kv_global_start,
        kv_valid_len=kv_valid_len,
        q_hbm_offset=q_hbm_offset,
        q_tile_size=q_tile_size,
        o_hbm_offset=o_hbm_offset,
        packed_schedule=packed_schedule,
        actual_steps=np.full((PCP_SIZE, ), steps, dtype=np.int32),
        global_actual_steps=np.array([steps], dtype=np.int32),
    )


def _make_compact_overlap_two_tile_schedule():
    """Build a compact local layout schedule that exposes full-block stores."""
    steps = 2 * PCP_SIZE
    shape = (PCP_SIZE, steps, 1)
    req_id = np.full(shape, -1, dtype=np.int32)
    kv_page_rank = np.full(shape, -1, dtype=np.int32)
    kv_page_idx = np.zeros(shape, dtype=np.int32)
    is_first_kv = np.zeros(shape, dtype=np.int32)
    is_last_kv = np.zeros(shape, dtype=np.int32)
    load_q = np.zeros(shape, dtype=np.int32)
    q_global_start = np.full(shape, 3 * PAGE_SIZE, dtype=np.int32)
    kv_global_start = np.zeros(shape, dtype=np.int32)
    kv_valid_len = np.full(shape, PAGE_SIZE, dtype=np.int32)
    q_hbm_offset = np.zeros(shape, dtype=np.int32)
    q_tile_size = np.zeros(shape, dtype=np.int32)
    o_hbm_offset = np.zeros(shape, dtype=np.int32)

    # The full tile writes rows [3, 11) first. The later partial tile writes
    # rows [0, 3). A full-block partial store would corrupt rows [3, 8).
    tile_specs = (
        (1, 3, Q_TILE),
        (0, 0, 3),
    )
    for group_idx, (tile_req_id, q_offset, tile_size) in enumerate(tile_specs):
        group_start = group_idx * PCP_SIZE
        for step in range(PCP_SIZE):
            schedule_step = group_start + step
            req_id[:, schedule_step, 0] = tile_req_id
            kv_page_rank[:, schedule_step, 0] = step
            kv_global_start[:, schedule_step, 0] = step * PAGE_SIZE
            q_hbm_offset[:, schedule_step, 0] = q_offset
            q_tile_size[:, schedule_step, 0] = tile_size
            o_hbm_offset[:, schedule_step, 0] = q_offset
        is_first_kv[:, group_start, 0] = 1
        is_last_kv[:, group_start + PCP_SIZE - 1, 0] = 1
        load_q[:, group_start, 0] = 1

    packed_schedule = pack_pcp_streaming_schedule_fields(
        req_id=req_id,
        kv_page_rank=kv_page_rank,
        kv_page_idx=kv_page_idx,
        is_first_kv=is_first_kv,
        is_last_kv=is_last_kv,
        load_q=load_q,
        q_global_start=q_global_start,
        kv_global_start=kv_global_start,
        kv_valid_len=kv_valid_len,
        q_hbm_offset=q_hbm_offset,
        q_tile_size=q_tile_size,
        o_hbm_offset=o_hbm_offset,
    )
    return PcpStreamingSchedule(
        req_id=req_id,
        kv_page_rank=kv_page_rank,
        kv_page_idx=kv_page_idx,
        is_first_kv=is_first_kv,
        is_last_kv=is_last_kv,
        load_q=load_q,
        q_global_start=q_global_start,
        kv_global_start=kv_global_start,
        kv_valid_len=kv_valid_len,
        q_hbm_offset=q_hbm_offset,
        q_tile_size=q_tile_size,
        o_hbm_offset=o_hbm_offset,
        packed_schedule=packed_schedule,
        actual_steps=np.full((PCP_SIZE, ), steps, dtype=np.int32),
        global_actual_steps=np.array([steps], dtype=np.int32),
    )


def _truncate_schedule(schedule, steps, actual_steps):
    kv_page_indices = None
    if schedule.kv_page_indices is not None:
        kv_page_indices = schedule.kv_page_indices[:, :steps].copy()
    return PcpStreamingSchedule(
        req_id=schedule.req_id[:, :steps].copy(),
        kv_page_rank=schedule.kv_page_rank[:, :steps].copy(),
        kv_page_idx=schedule.kv_page_idx[:, :steps].copy(),
        is_first_kv=schedule.is_first_kv[:, :steps].copy(),
        is_last_kv=schedule.is_last_kv[:, :steps].copy(),
        load_q=schedule.load_q[:, :steps].copy(),
        q_global_start=schedule.q_global_start[:, :steps].copy(),
        kv_global_start=schedule.kv_global_start[:, :steps].copy(),
        kv_valid_len=schedule.kv_valid_len[:, :steps].copy(),
        q_hbm_offset=schedule.q_hbm_offset[:, :steps].copy(),
        q_tile_size=schedule.q_tile_size[:, :steps].copy(),
        o_hbm_offset=schedule.o_hbm_offset[:, :steps].copy(),
        packed_schedule=schedule.packed_schedule[:steps].copy(),
        actual_steps=np.asarray(actual_steps, dtype=np.int32),
        global_actual_steps=np.array([steps], dtype=np.int32),
        kv_page_indices=kv_page_indices,
    )


def _run_page_groups_local(q_global,
                           kv_cache_by_rank,
                           packed_schedule,
                           *,
                           sm_scale,
                           collective_id,
                           active_page_groups=None):
    mesh = jax.sharding.Mesh(jax.local_devices()[:PCP_SIZE], (AXIS, ))
    if active_page_groups is None:
        active_page_groups = jnp.array([packed_schedule.shape[0] // PCP_SIZE],
                                       dtype=jnp.int32)

    def _call(q_local, kv_cache_local, schedule, active_groups):
        return pcp_streaming_attention_page_groups_local(
            q_local,
            kv_cache_local[0],
            schedule,
            active_groups,
            pcp_size=PCP_SIZE,
            q_block_size=Q_TILE,
            sm_scale=sm_scale,
            collective_id=collective_id,
        )

    fn = jax.jit(
        jax.shard_map(
            _call,
            mesh=mesh,
            in_specs=(
                P(AXIS, None, None, None),
                P(AXIS, None, None, None, None, None),
                P(None, None, None, None),
                P(None),
            ),
            out_specs=P(AXIS, None, None, None),
            check_vma=False,
        ))
    return fn(q_global, kv_cache_by_rank, packed_schedule, active_page_groups)


def _run_page_groups_packed_local(q_global,
                                  kv_cache_by_rank,
                                  packed_schedule,
                                  *,
                                  sm_scale,
                                  collective_id,
                                  active_page_groups=None,
                                  k_scale=None,
                                  v_scale=None):
    mesh = jax.sharding.Mesh(jax.local_devices()[:PCP_SIZE], (AXIS, ))
    if active_page_groups is None:
        active_page_groups = jnp.array([packed_schedule.shape[0] // PCP_SIZE],
                                       dtype=jnp.int32)

    def _call(q_local, kv_cache_local, schedule, active_groups):
        return pcp_streaming_attention_page_groups_packed_local(
            q_local,
            kv_cache_local[0],
            schedule,
            active_groups,
            pcp_size=PCP_SIZE,
            q_block_size=Q_TILE,
            sm_scale=sm_scale,
            collective_id=collective_id,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    fn = jax.jit(
        jax.shard_map(
            _call,
            mesh=mesh,
            in_specs=(
                P(AXIS, None, None, None),
                P(AXIS, None, None, None, None, None),
                P(None, None, None, None),
                P(None),
            ),
            out_specs=P(AXIS, None, None, None),
            check_vma=False,
        ))
    return fn(q_global, kv_cache_by_rank, packed_schedule, active_page_groups)


def _pack_native_kv_cache(kv_cache, kv_packing):
    pcp_size, pages, page_size, kv_heads, kv_pair, head_dim = kv_cache.shape
    if kv_pair != 2:
        raise ValueError("native KV cache must have a K/V pair axis.")
    aligned_kv_heads_x2 = math.ceil(kv_heads * 2 / kv_packing) * kv_packing
    flat = np.zeros(
        (pcp_size, pages, page_size, aligned_kv_heads_x2, head_dim),
        dtype=kv_cache.dtype)
    flat[..., :kv_heads * 2, :] = kv_cache.reshape(pcp_size, pages, page_size,
                                                   kv_heads * 2, head_dim)
    return flat.reshape(pcp_size, pages, page_size,
                        aligned_kv_heads_x2 // kv_packing, kv_packing,
                        head_dim)


def _quantize_native_kv_cache_fp8(kv_cache, *, k_scale, v_scale):
    k_q = jnp.asarray(kv_cache[..., 0, :] / k_scale, dtype=jnp.float8_e4m3fn)
    v_q = jnp.asarray(kv_cache[..., 1, :] / v_scale, dtype=jnp.float8_e4m3fn)
    quantized = jnp.stack((k_q, v_q), axis=-2)
    dequantized = jnp.stack(
        (
            k_q.astype(jnp.float32) * k_scale,
            v_q.astype(jnp.float32) * v_scale,
        ),
        axis=-2,
    )
    return quantized, np.asarray(jax.device_get(dequantized))


def _make_tail_partial_metadata_schedule():
    pcp_size = 8
    page_size = 256
    q_block_size = 256
    q_len = 1025
    kv_len = 5121
    local_kv_cache_num_blocks = 3
    schedule = generate_pcp_streaming_schedule_from_metadata_host(
        kv_lens=np.array([kv_len], dtype=np.int32),
        page_indices=np.arange(local_kv_cache_num_blocks,
                               dtype=np.int32)[None, :],
        cu_q_lens=np.array([0, q_len], dtype=np.int32),
        distribution=np.array([0, 0, 1], dtype=np.int32),
        global_bucket_tokens=pcp_size * q_block_size,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=page_size,
        q_block_size=q_block_size,
        num_lanes=1,
        kv_pages_per_block=1,
    )
    active_page_groups = build_pcp_streaming_active_page_groups(schedule)
    return schedule, active_page_groups


def _assert_tail_partial_metadata_schedule(schedule, active_page_groups):
    np.testing.assert_array_equal(active_page_groups,
                                  np.array([3], dtype=np.int32))
    np.testing.assert_array_equal(schedule.global_actual_steps,
                                  np.array([24], dtype=np.int32))
    np.testing.assert_array_equal(
        schedule.q_tile_size[:, 16, 0],
        np.array([256, 256, 256, 256, 1, 0, 0, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        schedule.req_id[:, 16, 0],
        np.array([0, 0, 0, 0, 0, -1, -1, -1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        schedule.req_id[:, 21:24, 0],
        np.full((8, 3), -1, dtype=np.int32),
    )


def _make_delayed_second_group_rank3_schedule_loader(original_load):

    def _delayed_load(packed_schedule_ref, sched_vmem_ref, sem, step):
        original_load(packed_schedule_ref, sched_vmem_ref, sem, step)
        should_delay = jnp.logical_and(
            lax.axis_index(AXIS) == PCP_SIZE - 1,
            step == 2 * PCP_SIZE - 1,
        )

        @pl.when(should_delay)
        def _delay_rank3_before_round_send():
            for _ in range(128):
                delay_load = pltpu.make_async_copy(
                    src_ref=packed_schedule_ref.at[step],
                    dst_ref=sched_vmem_ref.at[:, :, :],
                    sem=sem,
                )
                delay_load.start()
                delay_load.wait()

    return _delayed_load


def _make_uniform_two_group_attention_inputs():
    q_by_rank = np.zeros((PCP_SIZE, Q_TILE, 1, 1, HEAD_DIM), dtype=np.float32)
    kv_cache = np.zeros((PCP_SIZE, 2, PAGE_SIZE, 1, 2, HEAD_DIM),
                        dtype=np.float32)
    for local_page in range(2):
        for rank in range(PCP_SIZE):
            global_page = local_page * PCP_SIZE + rank
            kv_cache[rank, local_page, :, 0, 1, :] = global_page + 1
    return q_by_rank, kv_cache


def _require_tpu_devices(min_count, reason):
    devices = jax.local_devices()
    if len(devices) < min_count or devices[0].platform != "tpu":
        pytest.skip(reason)
    return devices


@pytest.fixture(autouse=True)
def _require_pcp_tpu_devices():
    _require_tpu_devices(
        PCP_SIZE, "PCP streaming kernel test requires four TPU devices.")


def test_single_page_group_kernel_rejects_sub_hbm_row_q_tile():
    q_by_rank = np.zeros((PCP_SIZE, 8, 1, 1, HEAD_DIM), dtype=np.float32)
    kv_cache = np.zeros((PCP_SIZE, 1, PAGE_SIZE, 1, 2, HEAD_DIM),
                        dtype=np.float32)
    schedule = _make_single_page_group_schedule()

    with pytest.raises(NotImplementedError,
                       match="q_block_size.*multiple.*16"):
        pcp_streaming_attention_single_page_group(
            jnp.asarray(q_by_rank),
            jnp.asarray(kv_cache),
            jnp.asarray(schedule.packed_schedule),
            pcp_size=PCP_SIZE,
            sm_scale=1.0,
        )


def test_single_page_group_kernel_matches_reference():
    rng = np.random.default_rng(1234)
    q_by_rank = rng.normal(size=(PCP_SIZE, Q_TILE, 1, 1, HEAD_DIM)).astype(
        np.float32) * 0.1
    kv_cache = rng.normal(size=(PCP_SIZE, 1, PAGE_SIZE, 1, 2,
                                HEAD_DIM)).astype(np.float32) * 0.1
    schedule = _make_single_page_group_schedule()
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = pcp_streaming_attention_single_page_group(
        jnp.asarray(q_by_rank),
        jnp.asarray(kv_cache),
        jnp.asarray(schedule.packed_schedule),
        pcp_size=PCP_SIZE,
        sm_scale=sm_scale,
        collective_id=14,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               kv_cache,
                                               schedule,
                                               sm_scale=sm_scale)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-4,
                               atol=5e-5)


def test_single_page_group_kernel_handles_idle_consumer_ranks():
    rng = np.random.default_rng(5678)
    q_by_rank = rng.normal(size=(PCP_SIZE, Q_TILE, 1, 1, HEAD_DIM)).astype(
        np.float32) * 0.1
    kv_cache = rng.normal(size=(PCP_SIZE, 1, PAGE_SIZE, 1, 2,
                                HEAD_DIM)).astype(np.float32) * 0.1
    generated = generate_pcp_streaming_schedule(
        kv_lens=[PCP_SIZE * PAGE_SIZE],
        cu_q_lens=[0, 32],
        q_start_offsets=[3 * PAGE_SIZE],
        block_tables=np.array([[0]], dtype=np.int32),
        page_size=PAGE_SIZE,
        pcp_size=PCP_SIZE,
        interleave_size=PAGE_SIZE,
        num_lanes=1,
        bq_sz=Q_TILE,
    )
    schedule = _truncate_schedule(generated,
                                  steps=PCP_SIZE,
                                  actual_steps=[0, 0, 0, PCP_SIZE])
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = pcp_streaming_attention_single_page_group(
        jnp.asarray(q_by_rank),
        jnp.asarray(kv_cache),
        jnp.asarray(schedule.packed_schedule),
        pcp_size=PCP_SIZE,
        sm_scale=sm_scale,
        collective_id=15,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               kv_cache,
                                               schedule,
                                               sm_scale=sm_scale)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-4,
                               atol=5e-5)


def test_page_group_kernel_matches_generated_multi_tile_schedule():
    rng = np.random.default_rng(9012)
    q_by_rank = rng.normal(size=(PCP_SIZE, 4 * Q_TILE, 1, 1, HEAD_DIM)).astype(
        np.float32) * 0.1
    kv_cache = rng.normal(size=(PCP_SIZE, 1, PAGE_SIZE, 1, 2,
                                HEAD_DIM)).astype(np.float32) * 0.1
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[PCP_SIZE * PAGE_SIZE],
        cu_q_lens=[0, 32],
        q_start_offsets=[3 * PAGE_SIZE],
        block_tables=np.array([[0]], dtype=np.int32),
        page_size=PAGE_SIZE,
        pcp_size=PCP_SIZE,
        interleave_size=PAGE_SIZE,
        num_lanes=1,
        bq_sz=Q_TILE,
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = pcp_streaming_attention_page_groups(
        jnp.asarray(q_by_rank),
        jnp.asarray(kv_cache),
        jnp.asarray(schedule.packed_schedule),
        pcp_size=PCP_SIZE,
        q_block_size=Q_TILE,
        sm_scale=sm_scale,
        collective_id=16,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               kv_cache,
                                               schedule,
                                               sm_scale=sm_scale)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-4,
                               atol=5e-5)


def test_page_group_kernel_handles_partial_q_tile_and_partial_kv_page():
    rng = np.random.default_rng(3456)
    q_by_rank = rng.normal(size=(PCP_SIZE, 4 * Q_TILE, 1, 1, HEAD_DIM)).astype(
        np.float32) * 0.1
    kv_cache = rng.normal(size=(PCP_SIZE, 1, PAGE_SIZE, 1, 2,
                                HEAD_DIM)).astype(np.float32) * 0.1
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[3 * PAGE_SIZE + 116],
        cu_q_lens=[0, 28],
        q_start_offsets=[3 * PAGE_SIZE],
        block_tables=np.array([[0]], dtype=np.int32),
        page_size=PAGE_SIZE,
        pcp_size=PCP_SIZE,
        interleave_size=PAGE_SIZE,
        num_lanes=1,
        bq_sz=Q_TILE,
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = pcp_streaming_attention_page_groups(
        jnp.asarray(q_by_rank),
        jnp.asarray(kv_cache),
        jnp.asarray(schedule.packed_schedule),
        pcp_size=PCP_SIZE,
        q_block_size=Q_TILE,
        sm_scale=sm_scale,
        collective_id=17,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               kv_cache,
                                               schedule,
                                               sm_scale=sm_scale)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-4,
                               atol=5e-5)


def test_page_group_kernel_matches_production_unaligned_multi_request_schedule(
):
    rng = np.random.default_rng(2468)
    q_block_size = PAGE_SIZE
    local_tokens = 2 * q_block_size
    q_by_rank = rng.normal(size=(PCP_SIZE, local_tokens, 1, 1,
                                 HEAD_DIM)).astype(np.float32) * 0.1
    kv_cache = rng.normal(size=(PCP_SIZE, 1, PAGE_SIZE, 1, 2,
                                HEAD_DIM)).astype(np.float32) * 0.1
    q_lens = np.array([17, 20], dtype=np.int32)
    q_starts = np.array([120, 250], dtype=np.int32)
    schedule = generate_production_pcp_schedule(
        kv_lens=q_starts + q_lens,
        cu_q_lens=np.array([0, q_lens[0], q_lens.sum()], dtype=np.int32),
        q_start_offsets=q_starts,
        block_tables=np.array([[0], [0]], dtype=np.int32),
        page_size=PAGE_SIZE,
        pcp_size=PCP_SIZE,
        interleave_size=PAGE_SIZE,
        num_lanes=1,
        bq_sz=q_block_size,
        pad_kv_pages_to_pcp_group=True,
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = pcp_streaming_attention_page_groups(
        jnp.asarray(q_by_rank),
        jnp.asarray(kv_cache),
        jnp.asarray(schedule.packed_schedule),
        pcp_size=PCP_SIZE,
        q_block_size=q_block_size,
        sm_scale=sm_scale,
        collective_id=33,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               kv_cache,
                                               schedule,
                                               sm_scale=sm_scale)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-4,
                               atol=1e-4)


def test_page_group_kernel_preserves_compact_rows_after_later_partial_tile():
    rng = np.random.default_rng(1122)
    local_tokens = 2 * Q_TILE
    q_by_rank = rng.normal(size=(PCP_SIZE, local_tokens, 1, 1,
                                 HEAD_DIM)).astype(np.float32) * 0.1
    kv_cache = rng.normal(size=(PCP_SIZE, 1, PAGE_SIZE, 1, 2,
                                HEAD_DIM)).astype(np.float32) * 0.1
    schedule = _make_compact_overlap_two_tile_schedule()
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = pcp_streaming_attention_page_groups(
        jnp.asarray(q_by_rank),
        jnp.asarray(kv_cache),
        jnp.asarray(schedule.packed_schedule),
        pcp_size=PCP_SIZE,
        q_block_size=Q_TILE,
        sm_scale=sm_scale,
        collective_id=30,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               kv_cache,
                                               schedule,
                                               sm_scale=sm_scale)
    assert np.max(np.abs(expected[:, 3:3 + Q_TILE])) > 1e-6
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-4,
                               atol=5e-5)


def test_page_group_kernel_carries_online_state_across_kv_groups():
    rng = np.random.default_rng(7890)
    q_by_rank = rng.normal(size=(PCP_SIZE, Q_TILE, 1, 1, HEAD_DIM)).astype(
        np.float32) * 0.1
    kv_cache = rng.normal(size=(PCP_SIZE, 2, PAGE_SIZE, 1, 2,
                                HEAD_DIM)).astype(np.float32) * 0.1
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[5 * PAGE_SIZE],
        cu_q_lens=[0, Q_TILE],
        q_start_offsets=[4 * PAGE_SIZE],
        block_tables=np.array([[0, 1]], dtype=np.int32),
        page_size=PAGE_SIZE,
        pcp_size=PCP_SIZE,
        interleave_size=PAGE_SIZE,
        num_lanes=1,
        bq_sz=Q_TILE,
        pad_kv_pages_to_pcp_group=True,
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = pcp_streaming_attention_page_groups(
        jnp.asarray(q_by_rank),
        jnp.asarray(kv_cache),
        jnp.asarray(schedule.packed_schedule),
        pcp_size=PCP_SIZE,
        q_block_size=Q_TILE,
        sm_scale=sm_scale,
        collective_id=18,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               kv_cache,
                                               schedule,
                                               sm_scale=sm_scale)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-4,
                               atol=5e-5)


def test_page_group_kernel_waits_for_delayed_incoming_kv_page(monkeypatch):
    q_by_rank, kv_cache = _make_uniform_two_group_attention_inputs()
    schedule = _make_two_page_group_schedule()
    sm_scale = 1.0
    original_load = pcp_kernel._load_schedule_step
    monkeypatch.setattr(
        pcp_kernel,
        "_load_schedule_step",
        _make_delayed_second_group_rank3_schedule_loader(original_load),
    )

    out = pcp_streaming_attention_page_groups(
        jnp.asarray(q_by_rank),
        jnp.asarray(kv_cache),
        jnp.asarray(schedule.packed_schedule),
        pcp_size=PCP_SIZE,
        q_block_size=Q_TILE,
        sm_scale=sm_scale,
        collective_id=28,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               kv_cache,
                                               schedule,
                                               sm_scale=sm_scale)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-4,
                               atol=5e-5)


def test_page_group_kernel_handles_two_lanes():
    rng = np.random.default_rng(2468)
    q_by_rank = rng.normal(size=(PCP_SIZE, 4 * Q_TILE, 1, 1, HEAD_DIM)).astype(
        np.float32) * 0.1
    kv_cache = rng.normal(size=(PCP_SIZE, 2, PAGE_SIZE, 1, 2,
                                HEAD_DIM)).astype(np.float32) * 0.1
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[5 * PAGE_SIZE],
        cu_q_lens=[0, 4 * Q_TILE],
        q_start_offsets=[4 * PAGE_SIZE],
        block_tables=np.array([[0, 1]], dtype=np.int32),
        page_size=PAGE_SIZE,
        pcp_size=PCP_SIZE,
        interleave_size=PAGE_SIZE,
        num_lanes=2,
        bq_sz=Q_TILE,
        pad_kv_pages_to_pcp_group=True,
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = pcp_streaming_attention_page_groups(
        jnp.asarray(q_by_rank),
        jnp.asarray(kv_cache),
        jnp.asarray(schedule.packed_schedule),
        pcp_size=PCP_SIZE,
        q_block_size=Q_TILE,
        sm_scale=sm_scale,
        collective_id=19,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               kv_cache,
                                               schedule,
                                               sm_scale=sm_scale)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-4,
                               atol=5e-5)


def test_page_group_kernel_handles_multiple_kv_and_q_heads():
    rng = np.random.default_rng(1357)
    q_by_rank = rng.normal(size=(PCP_SIZE, Q_TILE, 2, 2, HEAD_DIM)).astype(
        np.float32) * 0.1
    kv_cache = rng.normal(size=(PCP_SIZE, 1, PAGE_SIZE, 2, 2,
                                HEAD_DIM)).astype(np.float32) * 0.1
    schedule = _make_single_page_group_schedule()
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = pcp_streaming_attention_page_groups(
        jnp.asarray(q_by_rank),
        jnp.asarray(kv_cache),
        jnp.asarray(schedule.packed_schedule),
        pcp_size=PCP_SIZE,
        q_block_size=Q_TILE,
        sm_scale=sm_scale,
        collective_id=20,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               kv_cache,
                                               schedule,
                                               sm_scale=sm_scale)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-4,
                               atol=5e-5)


def test_page_group_local_kernel_runs_inside_existing_pcp_shard_map():
    rng = np.random.default_rng(9753)
    q_by_rank = rng.normal(size=(PCP_SIZE, Q_TILE, 1, 1, HEAD_DIM)).astype(
        np.float32) * 0.1
    q_global = q_by_rank.reshape(PCP_SIZE * Q_TILE, 1, 1, HEAD_DIM)
    kv_cache = rng.normal(size=(PCP_SIZE, 1, PAGE_SIZE, 1, 2,
                                HEAD_DIM)).astype(np.float32) * 0.1
    schedule = _make_single_page_group_schedule()
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = _run_page_groups_local(
        jnp.asarray(q_global),
        jnp.asarray(kv_cache),
        jnp.asarray(schedule.packed_schedule),
        sm_scale=sm_scale,
        collective_id=21,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               kv_cache,
                                               schedule,
                                               sm_scale=sm_scale)
    expected = expected.reshape(PCP_SIZE * Q_TILE, 1, 1, HEAD_DIM)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-4,
                               atol=5e-5)


def test_page_group_local_kernel_uses_dynamic_active_page_groups():
    rng = np.random.default_rng(2468)
    q_by_rank = rng.normal(size=(PCP_SIZE, Q_TILE, 1, 1, HEAD_DIM)).astype(
        np.float32) * 0.1
    q_global = q_by_rank.reshape(PCP_SIZE * Q_TILE, 1, 1, HEAD_DIM)
    kv_cache = rng.normal(size=(PCP_SIZE, 1, PAGE_SIZE, 1, 2,
                                HEAD_DIM)).astype(np.float32) * 0.1
    schedule = _make_single_page_group_schedule()
    padded_schedule = np.zeros(
        (PCP_SIZE * 2, PCP_SIZE, 1, schedule.packed_schedule.shape[-1]),
        dtype=np.int32)
    padded_schedule[..., 0] = -1
    padded_schedule[:PCP_SIZE] = schedule.packed_schedule
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = _run_page_groups_local(
        jnp.asarray(q_global),
        jnp.asarray(kv_cache),
        jnp.asarray(padded_schedule),
        sm_scale=sm_scale,
        collective_id=24,
        active_page_groups=jnp.array([1], dtype=jnp.int32),
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               kv_cache,
                                               schedule,
                                               sm_scale=sm_scale)
    expected = expected.reshape(PCP_SIZE * Q_TILE, 1, 1, HEAD_DIM)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-4,
                               atol=5e-5)


def test_page_group_packed_local_kernel_consumes_batched_rpa_kv_layout():
    rng = np.random.default_rng(8642)
    q_by_rank = rng.normal(size=(PCP_SIZE, Q_TILE, 2, 2, HEAD_DIM)).astype(
        np.float32) * 0.1
    q_global = q_by_rank.reshape(PCP_SIZE * Q_TILE, 2, 2, HEAD_DIM)
    native_kv_np = rng.normal(size=(PCP_SIZE, 1, PAGE_SIZE, 2, 2,
                                    HEAD_DIM)).astype(np.float32) * 0.1
    native_kv = jnp.asarray(native_kv_np, dtype=jnp.bfloat16)
    native_kv_host = np.asarray(jax.device_get(native_kv))
    packed_kv = jnp.asarray(_pack_native_kv_cache(native_kv_host, 2),
                            dtype=jnp.bfloat16)
    schedule = _make_single_page_group_schedule()
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = _run_page_groups_packed_local(
        jnp.asarray(q_global),
        packed_kv,
        jnp.asarray(schedule.packed_schedule),
        sm_scale=sm_scale,
        collective_id=22,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               native_kv_host,
                                               schedule,
                                               sm_scale=sm_scale)
    expected = expected.reshape(PCP_SIZE * Q_TILE, 2, 2, HEAD_DIM)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-3,
                               atol=5e-4)


def test_page_group_packed_local_multi_head_preserves_compact_partial_rows():
    rng = np.random.default_rng(3344)
    local_tokens = 2 * Q_TILE
    kv_heads = 2
    q_per_kv = 2
    q_by_rank = rng.normal(size=(PCP_SIZE, local_tokens, kv_heads, q_per_kv,
                                 HEAD_DIM)).astype(np.float32) * 0.1
    q_global = q_by_rank.reshape(PCP_SIZE * local_tokens, kv_heads, q_per_kv,
                                 HEAD_DIM)
    native_kv_np = rng.normal(size=(PCP_SIZE, 1, PAGE_SIZE, kv_heads, 2,
                                    HEAD_DIM)).astype(np.float32) * 0.1
    native_kv = jnp.asarray(native_kv_np, dtype=jnp.bfloat16)
    native_kv_host = np.asarray(jax.device_get(native_kv))
    packed_kv = jnp.asarray(_pack_native_kv_cache(native_kv_host, 2),
                            dtype=jnp.bfloat16)
    schedule = _make_compact_overlap_two_tile_schedule()
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = _run_page_groups_packed_local(
        jnp.asarray(q_global, dtype=jnp.bfloat16),
        packed_kv,
        jnp.asarray(schedule.packed_schedule),
        sm_scale=sm_scale,
        collective_id=31,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               native_kv_host,
                                               schedule,
                                               sm_scale=sm_scale)
    expected = expected.reshape(q_global.shape)
    assert np.max(np.abs(expected[3:3 + Q_TILE])) > 1e-6
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-3,
                               atol=5e-4)


def test_page_group_packed_local_kernel_consumes_fp8_kv_packing4_layout():
    rng = np.random.default_rng(97531)
    q_by_rank = rng.normal(size=(PCP_SIZE, Q_TILE, 2, 2, HEAD_DIM)).astype(
        np.float32) * 0.1
    q_global = q_by_rank.reshape(PCP_SIZE * Q_TILE, 2, 2, HEAD_DIM)
    native_kv_np = rng.normal(size=(PCP_SIZE, 1, PAGE_SIZE, 2, 2,
                                    HEAD_DIM)).astype(np.float32) * 0.1
    k_scale = 0.125
    v_scale = 0.25
    native_kv_q, native_kv_dequant = _quantize_native_kv_cache_fp8(
        native_kv_np, k_scale=k_scale, v_scale=v_scale)
    native_kv_q_host = np.asarray(jax.device_get(native_kv_q))
    packed_kv = jnp.asarray(_pack_native_kv_cache(native_kv_q_host, 4),
                            dtype=jnp.float8_e4m3fn)
    schedule = _make_single_page_group_schedule()
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = _run_page_groups_packed_local(
        jnp.asarray(q_global),
        packed_kv,
        jnp.asarray(schedule.packed_schedule),
        sm_scale=sm_scale,
        collective_id=29,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               native_kv_dequant,
                                               schedule,
                                               sm_scale=sm_scale)
    expected = expected.reshape(PCP_SIZE * Q_TILE, 2, 2, HEAD_DIM)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=2e-2,
                               atol=2e-2)


def test_page_group_packed_local_kernel_tail_partial_metadata_schedule_shape():
    schedule, active_page_groups = _make_tail_partial_metadata_schedule()
    _assert_tail_partial_metadata_schedule(schedule, active_page_groups)


def test_page_group_packed_local_kernel_reproduces_tail_partial_pcp8_fp8():
    if os.environ.get("VLLM_TPU_RUN_PCP_TAIL_REPRO") != "1":
        pytest.skip("Set VLLM_TPU_RUN_PCP_TAIL_REPRO=1 to run this "
                    "expected PCP8 tail-partial core-halt reproducer.")
    if jax.local_device_count() < 8:
        pytest.skip("PCP8 tail-partial reproducer requires 8 devices.")

    pcp_size = 8
    q_tile = 256
    page_size = 256
    kv_heads = 2
    q_per_kv = 2
    local_q_len = q_tile
    local_kv_cache_num_blocks = 3
    rng = np.random.default_rng(397081)
    q_by_rank = rng.normal(size=(pcp_size, local_q_len, kv_heads, q_per_kv,
                                 HEAD_DIM)).astype(np.float32) * 0.1
    q_global = q_by_rank.reshape(pcp_size * local_q_len, kv_heads, q_per_kv,
                                 HEAD_DIM)
    native_kv_np = rng.normal(size=(pcp_size, local_kv_cache_num_blocks,
                                    page_size, kv_heads, 2, HEAD_DIM)).astype(
                                        np.float32) * 0.1
    k_scale = 0.125
    v_scale = 0.25
    native_kv_q, native_kv_dequant = _quantize_native_kv_cache_fp8(
        native_kv_np, k_scale=k_scale, v_scale=v_scale)
    native_kv_q_host = np.asarray(jax.device_get(native_kv_q))
    packed_kv = jnp.asarray(_pack_native_kv_cache(native_kv_q_host, 4),
                            dtype=jnp.float8_e4m3fn)
    schedule, active_page_groups = _make_tail_partial_metadata_schedule()
    _assert_tail_partial_metadata_schedule(schedule, active_page_groups)

    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    mesh = jax.sharding.Mesh(jax.local_devices()[:pcp_size], (AXIS, ))

    def _call(q_local, kv_cache_local, packed_schedule, active_groups):
        return pcp_streaming_attention_page_groups_packed_local(
            q_local,
            kv_cache_local[0],
            packed_schedule,
            active_groups,
            pcp_size=pcp_size,
            q_block_size=q_tile,
            sm_scale=sm_scale,
            collective_id=34,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    fn = jax.jit(
        jax.shard_map(
            _call,
            mesh=mesh,
            in_specs=(
                P(AXIS, None, None, None),
                P(AXIS, None, None, None, None, None),
                P(None, None, None, None),
                P(None),
            ),
            out_specs=P(AXIS, None, None, None),
            check_vma=False,
        ))
    out = fn(jnp.asarray(q_global, dtype=jnp.bfloat16), packed_kv,
             jnp.asarray(schedule.packed_schedule),
             jnp.asarray(active_page_groups))
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               native_kv_dequant,
                                               schedule,
                                               sm_scale=sm_scale)
    expected = expected.reshape(q_global.shape)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=2e-2,
                               atol=2e-2)


def test_page_group_packed_local_kernel_handles_qwen_pcp8_tile_shape():
    if jax.local_device_count() < 8:
        pytest.skip("Qwen-shaped PCP streaming regression requires 8 devices.")

    pcp_size = 8
    q_tile = 256
    page_size = 512
    kv_heads = 2
    q_per_kv = 2
    rng = np.random.default_rng(97531)
    q_by_rank = rng.normal(size=(pcp_size, page_size, kv_heads, q_per_kv,
                                 HEAD_DIM)).astype(np.float32) * 0.1
    q_global = q_by_rank.reshape(pcp_size * page_size, kv_heads, q_per_kv,
                                 HEAD_DIM)
    native_kv_np = rng.normal(size=(pcp_size, 1, page_size, kv_heads, 2,
                                    HEAD_DIM)).astype(np.float32) * 0.1
    native_kv = jnp.asarray(native_kv_np, dtype=jnp.bfloat16)
    native_kv_host = np.asarray(jax.device_get(native_kv))
    packed_kv = jnp.asarray(_pack_native_kv_cache(native_kv_host, 2),
                            dtype=jnp.bfloat16)
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[pcp_size * page_size],
        cu_q_lens=[0, pcp_size * page_size],
        q_start_offsets=[0],
        block_tables=np.array([[0]], dtype=np.int32),
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=page_size,
        num_lanes=1,
        bq_sz=q_tile,
        pad_kv_pages_to_pcp_group=True,
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    mesh = jax.sharding.Mesh(jax.local_devices()[:pcp_size], (AXIS, ))

    def _call(q_local, kv_cache_local, packed_schedule):
        return pcp_streaming_attention_page_groups_packed_local(
            q_local,
            kv_cache_local[0],
            packed_schedule,
            pcp_size=pcp_size,
            q_block_size=q_tile,
            sm_scale=sm_scale,
            collective_id=24,
        )

    fn = jax.jit(
        jax.shard_map(
            _call,
            mesh=mesh,
            in_specs=(
                P(AXIS, None, None, None),
                P(AXIS, None, None, None, None, None),
                P(None, None, None, None),
            ),
            out_specs=P(AXIS, None, None, None),
            check_vma=False,
        ))
    out = fn(jnp.asarray(q_global, dtype=jnp.bfloat16), packed_kv,
             jnp.asarray(schedule.packed_schedule))
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               native_kv_host,
                                               schedule,
                                               sm_scale=sm_scale)
    expected = expected.reshape(q_global.shape)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-3,
                               atol=5e-4)


def test_page_group_packed_local_kernel_handles_page32_interleave():
    if jax.local_device_count() < 8:
        pytest.skip("page32 PCP streaming regression requires 8 devices.")

    pcp_size = 8
    q_tile = 32
    page_size = 32
    kv_heads = 1
    q_per_kv = 1
    q_len = 1024
    local_q_len = q_len // pcp_size
    local_pages = q_len // (pcp_size * page_size)
    rng = np.random.default_rng(64208)
    q_by_rank = rng.normal(size=(pcp_size, local_q_len, kv_heads, q_per_kv,
                                 HEAD_DIM)).astype(np.float32) * 0.1
    q_global = q_by_rank.reshape(pcp_size * local_q_len, kv_heads, q_per_kv,
                                 HEAD_DIM)
    native_kv_np = rng.normal(size=(pcp_size, local_pages, page_size, kv_heads,
                                    2, HEAD_DIM)).astype(np.float32) * 0.1
    native_kv = jnp.asarray(native_kv_np, dtype=jnp.bfloat16)
    native_kv_host = np.asarray(jax.device_get(native_kv))
    packed_kv = jnp.asarray(_pack_native_kv_cache(native_kv_host, 2),
                            dtype=jnp.bfloat16)
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[q_len],
        cu_q_lens=[0, q_len],
        q_start_offsets=[0],
        block_tables=np.arange(local_pages, dtype=np.int32)[None, :],
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=page_size,
        num_lanes=1,
        bq_sz=q_tile,
        pad_kv_pages_to_pcp_group=True,
    )
    assert schedule.packed_schedule.shape[0] > pcp_size

    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    mesh = jax.sharding.Mesh(jax.local_devices()[:pcp_size], (AXIS, ))

    def _call(q_local, kv_cache_local, packed_schedule):
        return pcp_streaming_attention_page_groups_packed_local(
            q_local,
            kv_cache_local[0],
            packed_schedule,
            pcp_size=pcp_size,
            q_block_size=q_tile,
            sm_scale=sm_scale,
            collective_id=25,
        )

    fn = jax.jit(
        jax.shard_map(
            _call,
            mesh=mesh,
            in_specs=(
                P(AXIS, None, None, None),
                P(AXIS, None, None, None, None, None),
                P(None, None, None, None),
            ),
            out_specs=P(AXIS, None, None, None),
            check_vma=False,
        ))
    out = fn(jnp.asarray(q_global, dtype=jnp.bfloat16), packed_kv,
             jnp.asarray(schedule.packed_schedule))
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               native_kv_host,
                                               schedule,
                                               sm_scale=sm_scale)
    expected = expected.reshape(q_global.shape)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-3,
                               atol=5e-4)


def test_page_group_packed_local_kernel_merges_page32_q_chunks():
    if jax.local_device_count() < 8:
        pytest.skip(
            "merged page32 PCP streaming regression requires 8 devices.")

    pcp_size = 8
    q_tile = 256
    page_size = 32
    kv_heads = 1
    q_per_kv = 1
    q_len = 2048
    local_q_len = q_len // pcp_size
    local_pages = q_len // (pcp_size * page_size)
    rng = np.random.default_rng(64209)
    q_by_rank = rng.normal(size=(pcp_size, local_q_len, kv_heads, q_per_kv,
                                 HEAD_DIM)).astype(np.float32) * 0.1
    q_global = q_by_rank.reshape(pcp_size * local_q_len, kv_heads, q_per_kv,
                                 HEAD_DIM)
    native_kv_np = rng.normal(size=(pcp_size, local_pages, page_size, kv_heads,
                                    2, HEAD_DIM)).astype(np.float32) * 0.1
    native_kv = jnp.asarray(native_kv_np, dtype=jnp.bfloat16)
    native_kv_host = np.asarray(jax.device_get(native_kv))
    packed_kv = jnp.asarray(_pack_native_kv_cache(native_kv_host, 2),
                            dtype=jnp.bfloat16)
    merged_schedule = generate_pcp_streaming_schedule(
        kv_lens=[q_len],
        cu_q_lens=[0, q_len],
        q_start_offsets=[0],
        block_tables=np.arange(local_pages, dtype=np.int32)[None, :],
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=page_size,
        num_lanes=1,
        bq_sz=q_tile,
        pad_kv_pages_to_pcp_group=True,
    )
    unmerged_schedule = generate_pcp_streaming_schedule(
        kv_lens=[q_len],
        cu_q_lens=[0, q_len],
        q_start_offsets=[0],
        block_tables=np.arange(local_pages, dtype=np.int32)[None, :],
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=page_size,
        num_lanes=1,
        bq_sz=page_size,
        pad_kv_pages_to_pcp_group=True,
    )
    assert int(merged_schedule.global_actual_steps[0]) < int(
        unmerged_schedule.global_actual_steps[0])
    assert np.max(merged_schedule.q_tile_size) == q_tile

    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    mesh = jax.sharding.Mesh(jax.local_devices()[:pcp_size], (AXIS, ))

    def _call(q_local, kv_cache_local, packed_schedule):
        return pcp_streaming_attention_page_groups_packed_local(
            q_local,
            kv_cache_local[0],
            packed_schedule,
            pcp_size=pcp_size,
            q_block_size=q_tile,
            sm_scale=sm_scale,
            collective_id=27,
        )

    fn = jax.jit(
        jax.shard_map(
            _call,
            mesh=mesh,
            in_specs=(
                P(AXIS, None, None, None),
                P(AXIS, None, None, None, None, None),
                P(None, None, None, None),
            ),
            out_specs=P(AXIS, None, None, None),
            check_vma=False,
        ))
    out = fn(jnp.asarray(q_global, dtype=jnp.bfloat16), packed_kv,
             jnp.asarray(merged_schedule.packed_schedule))
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               native_kv_host,
                                               merged_schedule,
                                               sm_scale=sm_scale)
    expected = expected.reshape(q_global.shape)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-3,
                               atol=5e-4)


def test_page_group_packed_local_kernel_handles_kv_page_blocks():
    if jax.local_device_count() < 8:
        pytest.skip(
            "KV page block PCP streaming regression requires 8 devices.")

    pcp_size = 8
    q_tile = 32
    page_size = 32
    kv_pages_per_block = 4
    kv_heads = 2
    q_per_kv = 2
    q_len = 2048
    local_q_len = q_len // pcp_size
    local_pages = q_len // (pcp_size * page_size)
    rng = np.random.default_rng(24680)
    q_by_rank = rng.normal(size=(pcp_size, local_q_len, kv_heads, q_per_kv,
                                 HEAD_DIM)).astype(np.float32) * 0.1
    q_global = q_by_rank.reshape(pcp_size * local_q_len, kv_heads, q_per_kv,
                                 HEAD_DIM)
    native_kv_np = rng.normal(size=(pcp_size, local_pages, page_size, kv_heads,
                                    2, HEAD_DIM)).astype(np.float32) * 0.1
    native_kv = jnp.asarray(native_kv_np, dtype=jnp.bfloat16)
    native_kv_host = np.asarray(jax.device_get(native_kv))
    packed_kv = jnp.asarray(_pack_native_kv_cache(native_kv_host, 2),
                            dtype=jnp.bfloat16)
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[q_len],
        cu_q_lens=[0, q_len],
        q_start_offsets=[0],
        block_tables=np.arange(local_pages, dtype=np.int32)[None, :],
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=page_size,
        num_lanes=1,
        bq_sz=q_tile,
        pad_kv_pages_to_pcp_group=True,
        kv_pages_per_block=kv_pages_per_block,
    )
    assert schedule.kv_page_indices is not None
    single_page_schedule = generate_pcp_streaming_schedule(
        kv_lens=[q_len],
        cu_q_lens=[0, q_len],
        q_start_offsets=[0],
        block_tables=np.arange(local_pages, dtype=np.int32)[None, :],
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=page_size,
        num_lanes=1,
        bq_sz=q_tile,
        pad_kv_pages_to_pcp_group=True,
    )
    assert int(schedule.global_actual_steps[0]) < int(
        single_page_schedule.global_actual_steps[0])

    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    mesh = jax.sharding.Mesh(jax.local_devices()[:pcp_size], (AXIS, ))

    def _call(q_local, kv_cache_local, packed_schedule):
        return pcp_streaming_attention_page_groups_packed_local(
            q_local,
            kv_cache_local[0],
            packed_schedule,
            pcp_size=pcp_size,
            q_block_size=q_tile,
            sm_scale=sm_scale,
            collective_id=26,
            kv_pages_per_block=kv_pages_per_block,
        )

    fn = jax.jit(
        jax.shard_map(
            _call,
            mesh=mesh,
            in_specs=(
                P(AXIS, None, None, None),
                P(AXIS, None, None, None, None, None),
                P(None, None, None, None),
            ),
            out_specs=P(AXIS, None, None, None),
            check_vma=False,
        ))
    out = fn(jnp.asarray(q_global, dtype=jnp.bfloat16), packed_kv,
             jnp.asarray(schedule.packed_schedule))
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               native_kv_host,
                                               schedule,
                                               sm_scale=sm_scale)
    expected = expected.reshape(q_global.shape)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-3,
                               atol=5e-4)

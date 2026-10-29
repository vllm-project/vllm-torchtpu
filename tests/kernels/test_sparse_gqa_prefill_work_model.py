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
"""Shape, traffic, peak-rate and schedule-bound checks for adaptive prefill."""

import functools
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from tests.kernels.prefill_work_model_helpers import work_model
from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as tc
from vllm_torchtpu.kernels.sparse_gqa_prefill import adaptive
from vllm_torchtpu.kernels.sparse_gqa_prefill.cost_model import (
    PrefillCostModel,
    prefill_cost_model,
    sc_cost,
    tc_cost,
)


def shapes(*, dtype=jnp.bfloat16, heads=4, dim=128, budget=1, page_size=128, columns=1):
    shape = jax.ShapeDtypeStruct
    packing = 4 // jnp.dtype(dtype).itemsize
    return (
        shape((256, heads, dim), jnp.bfloat16),
        shape((256, budget), jnp.int32),
        shape((256, columns), jnp.int32),
        shape((4, page_size, 4 // packing, packing, dim), dtype),
    )


@pytest.fixture
def unit_rates(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    info = replace(info, mem_bw_bytes_per_second=1000, bf16_ops_per_second=1000)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    return info


def test_tiny_tc_work_is_hand_counted_in_bytes_and_two_matrix_products(unit_rates):
    model = prefill_cost_model(*shapes(), kv_heads=2, ratio=4, kv_tile=256)
    # Three BF16 query-sized boundaries plus one ID, table entry and length.
    assert model.tc_query_ms == 3084
    assert model.tc_mask_word_ms == 8
    assert model.tc_fixed_ms == 12
    # KV256 has 262144 bytes and 134217728 QK/PV FLOPs.
    assert model.tc_tile_ms == 262144 + 134217728


@pytest.mark.parametrize(
    "page_size,columns", [(128, 13), (256, 7), (896, 2), (1664, 1)]
)
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
def test_pages_use_actual_extent_and_table_bytes(unit_rates, page_size, columns, dtype):
    args = shapes(dtype=dtype, page_size=page_size, columns=columns)
    model = prefill_cost_model(*args, kv_heads=2, ratio=4, kv_tile=256)
    tiles = (page_size * columns + 255) // 256
    value = tc_cost(model, queries=256, num_tiles=tiles, tile_visits=jnp.int32(1))
    assert float(value) > 0 and model.budget == 1
    # Ratio4 copies are 128 rows for every legal 128-aligned physical page.
    row_bytes = 1024 if dtype == jnp.bfloat16 else 512
    assert model.tc_tile_ms == 256 * row_bytes + 134217728
    assert model.tc_query_ms == 3072 + (columns + 2) * 4


def test_same_logical_extent_with_smaller_pages_reads_more_table_metadata(unit_rates):
    small = prefill_cost_model(
        *shapes(page_size=128, columns=7), kv_heads=2, ratio=4, kv_tile=256
    )
    large = prefill_cost_model(
        *shapes(page_size=896, columns=1), kv_heads=2, ratio=4, kv_tile=256
    )
    assert small.tc_tile_ms == large.tc_tile_ms
    assert small.tc_query_ms - large.tc_query_ms == 24
    assert small.tc_fixed_ms - large.tc_fixed_ms == 24


def test_sc_index_chunks_follow_padded_table_width(unit_rates):
    narrow = prefill_cost_model(*shapes(columns=128), kv_heads=2, ratio=4, kv_tile=256)
    wide = prefill_cost_model(*shapes(columns=129), kv_heads=2, ratio=4, kv_tile=256)
    # K1 still gathers capacity256; another table128 chunk reads 512 bytes/row.
    assert wide.sc_query_ms - narrow.sc_query_ms == 512


def test_actual_cache_dtype_changes_traffic_without_changing_qk_pv_work(unit_rates):
    bf16 = prefill_cost_model(*shapes(), kv_heads=2, ratio=4, kv_tile=256)
    fp8 = prefill_cost_model(
        *shapes(dtype=jnp.float8_e4m3fn), kv_heads=2, ratio=4, kv_tile=256
    )
    assert bf16.tc_tile_ms - fp8.tc_tile_ms == 131072
    assert bf16.tc_query_ms == fp8.tc_query_ms
    assert bf16.sc_query_ms > fp8.sc_query_ms


def test_public_peak_rates_scale_work_and_do_not_gate_chip_identity(
    monkeypatch, unit_rates
):
    base = prefill_cost_model(*shapes(), kv_heads=2, ratio=4, kv_tile=256)
    info = replace(unit_rates, mem_bw_bytes_per_second=2000, bf16_ops_per_second=2000)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    faster = prefill_cost_model(*shapes(), kv_heads=2, ratio=4, kv_tile=256)
    np.testing.assert_allclose(faster[:-5], np.asarray(base[:-5]) / 2)
    assert faster.budget == base.budget


def test_lengths_tail_zero_and_actual_budget_drive_sc_work():
    model = PrefillCostModel(
        0, 0, 0, 0, 0, 1, 10, 1, kv_heads=2, ratio=4, kv_tile=256, sc_chunk=32
    )
    lengths = jnp.array([0, 1, 4, 5, 7, 8, 1024, 0], jnp.int32)
    assert float(sc_cost(model, lengths)) == 18
    assert float(sc_cost(model, jnp.zeros(8, jnp.int32))) == 18
    assert float(sc_cost(model._replace(budget=256), lengths)) == 48


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
def test_both_routes_exist_for_actual_work_instead_of_a_profile(dtype):
    model = work_model(dtype)
    lengths = jnp.full(256, 64, jnp.int32)
    target = float(sc_cost(model, lengths)) * 1.02
    aligned = float(tc_cost(model, queries=256, num_tiles=7, tile_visits=jnp.int32(1)))
    ragged = float(
        tc_cost(
            model,
            queries=13 * 20 * 256,
            num_tiles=7,
            tile_visits=jnp.int32(256),
            launches=13,
        )
    )
    assert aligned < target < ragged


def test_resource_limited_q256_uses_legal_smaller_tc_tiles(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    args = shapes(heads=48, dim=256, budget=16, page_size=896)
    query, blocks, table, cache = args
    calls = []

    def fallback(*arrays, **kwargs):
        calls.append(kwargs)
        return arrays[0], arrays[5]

    def forbidden(*args):
        raise AssertionError("Q256 route entered despite its VMEM requirement")

    monkeypatch.setattr(tc, "block_sparse_gqa_attention", fallback)
    monkeypatch.setattr(adaptive, "_one_chunk", forbidden)
    result = jax.eval_shape(
        functools.partial(adaptive.adaptive_prefill, kv_heads=2, ratio=4),
        query,
        query,
        blocks,
        jax.ShapeDtypeStruct((256,), jnp.int32),
        table,
        cache,
        jax.ShapeDtypeStruct((2,), jnp.int32),
    )
    assert result == (query, cache)
    assert len(calls) == 1
    assert calls[0]["query_block"] < 256


@pytest.mark.parametrize("starts", [[0, 256], [0, 17, 17, 128, 256], list(range(257))])
def test_live_query_visit_lower_bound_is_within_independent_request_tile_work(starts):
    lengths = np.where(np.arange(256) % 11 == 0, 0, 1 + np.arange(256) % 7)
    ids = np.full((256, 2), -1, np.int32)
    for row, length in enumerate(lengths):
        ids[row, : min(int(length) // 4, 2)] = np.arange(min(int(length) // 4, 2))
    exact = 0
    lower = 0
    for lo, hi in zip(starts[:-1], starts[1:]):
        lower += (int(np.count_nonzero(lengths[lo:hi])) + 255) // 256
        for first in range(lo, hi, 256):
            tiles = set()
            for row in range(first, min(first + 256, hi)):
                if lengths[row] > 0:
                    tiles.update(int(g) // 64 for g in ids[row] if g >= 0)
                    if lengths[row] % 4:
                        tiles.add(int(lengths[row]) // 256)
            exact += len(tiles)
    assert lower <= exact
    assert lower > 0
    model = work_model(budget=2)
    cheap = tc_cost(model, queries=256, num_tiles=4, tile_visits=jnp.int32(lower))
    fine = tc_cost(model, queries=256, num_tiles=4, tile_visits=jnp.int32(exact))
    assert float(cheap) <= float(fine)


@pytest.mark.parametrize("kv_heads,ratio", [(1, 4), (4, 4), (2, 8)])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
def test_rank_local_geometry_admitted_and_used_by_work_model(
    monkeypatch, kv_heads, ratio, dtype
):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    shape = jax.ShapeDtypeStruct
    heads, dim = 8, 128
    packing = 4 // jnp.dtype(dtype).itemsize
    query = shape((256, heads, dim), jnp.bfloat16)
    blocks = shape((256, 64), jnp.int32)
    lengths = shape((256,), jnp.int32)
    table = shape((256, 3), jnp.int32)
    cache = shape(
        (4, 128, (2 * kv_heads + packing - 1) // packing, packing, dim), dtype
    )
    starts = shape((2,), jnp.int32)
    assert adaptive.is_supported_by_adaptive_prefill(
        query,
        query,
        blocks,
        lengths,
        table,
        cache,
        starts,
        kv_heads=kv_heads,
        ratio=ratio,
        k_scale=1.0,
        v_scale=1.0,
    )
    model = prefill_cost_model(
        query, blocks, table, cache, kv_heads=kv_heads, ratio=ratio, kv_tile=256
    )
    assert (model.kv_heads, model.ratio, model.kv_tile) == (kv_heads, ratio, 256)
    assert model.tc_tile_ms > 0 and model.sc_query_ms > 0


@pytest.mark.parametrize("ratio,kv_tile", [(4, 128), (4, 256), (8, 256)])
def test_parameterized_occupancy_matches_independent_token_union(ratio, kv_tile):
    from vllm_torchtpu.kernels.sparse_gqa_prefill.occupancy import (
        active_counts,
        bucket_upper_counts,
        exact_counts,
        span_counts,
    )

    lengths = np.zeros(256, np.int32)
    ids = np.full((256, 4), -1, np.int32)
    lengths[:6] = [1, ratio, kv_tile + 1, 2 * kv_tile + ratio - 1, 0, 4 * kv_tile]
    ids[1, 0] = 0
    ids[2, :2] = [0, kv_tile // ratio - 1]
    ids[3, :3] = [1, kv_tile // ratio, 2 * kv_tile // ratio - 1]
    ids[5] = [0, 31, 32, 63]
    touched = set()
    for row, length in enumerate(lengths):
        if length <= 0:
            continue
        for group in ids[row]:
            if group >= 0:
                touched.update(
                    token // kv_tile
                    for token in range(int(group) * ratio, (int(group) + 1) * ratio)
                )
        touched.update(
            token // kv_tile
            for token in range(int(length) // ratio * ratio, int(length))
        )
    kwargs = dict(num_tiles=16, ratio=ratio, kv_tile=kv_tile)
    np.testing.assert_array_equal(
        active_counts(jnp.asarray(ids), jnp.asarray(lengths), **kwargs), [len(touched)]
    )
    np.testing.assert_array_equal(
        exact_counts(jnp.asarray(ids), jnp.asarray(lengths), **kwargs), [len(touched)]
    )
    assert int(
        bucket_upper_counts(jnp.asarray(ids), jnp.asarray(lengths), **kwargs)[0]
    ) >= len(touched)
    assert int(
        span_counts(
            jnp.asarray(ids), jnp.asarray(lengths), ratio=ratio, kv_tile=kv_tile
        )[0]
    ) >= len(touched)


@pytest.mark.parametrize(
    "kv_heads,ratio,dtype,expected_chunk",
    [
        (2, 4, jnp.bfloat16, 32),
        (2, 4, jnp.float8_e4m3fn, 64),
        (4, 4, jnp.bfloat16, 16),
        (2, 8, jnp.bfloat16, 16),
    ],
)
def test_compact_workspace_tracks_actual_geometry(
    monkeypatch, kv_heads, ratio, dtype, expected_chunk
):
    import math

    from vllm_torchtpu.kernels.sparse_gqa_prefill.tuned_block_sizes import (
        SC_COMPACT_TARGET_BYTES,
        sc_chunk_size,
    )

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    packing = 4 // jnp.dtype(dtype).itemsize
    head_axes = (2 * kv_heads + packing - 1) // packing
    cache = jax.ShapeDtypeStruct((4, 256, head_axes, packing, 256), dtype)
    chunk = sc_chunk_size(cache, budget=512, ratio=ratio)
    assert chunk == expected_chunk and chunk % 8 == 0
    sparse = info.sparse_core
    assert sparse is not None
    granularity = math.lcm(
        sparse.num_lanes * sparse.num_cores * sparse.num_subcores * 4, head_axes
    )
    capacity = 2304 if ratio == 4 else 4352
    rows = math.ceil(chunk * capacity * head_axes / granularity) * granularity
    assert rows * packing * 256 * jnp.dtype(dtype).itemsize <= SC_COMPACT_TARGET_BYTES


@pytest.mark.parametrize(
    "kv_heads,ratio,kv_tile",
    [(0, 4, 256), (2, 0, 256), (3, 4, 256), (2, 4, 0), (2, 8, 128)],
)
def test_invalid_work_geometry_is_rejected_before_division(kv_heads, ratio, kv_tile):
    with pytest.raises(ValueError, match="geometry"):
        prefill_cost_model(*shapes(), kv_heads=kv_heads, ratio=ratio, kv_tile=kv_tile)


def test_compact_chunks_divide_dispatch_and_overflow_uses_tc_policy(monkeypatch):
    from vllm_torchtpu.kernels.sparse_gqa_prefill.tuned_block_sizes import sc_chunk_size

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    cache = jax.ShapeDtypeStruct((4, 256, 2, 2, 256), jnp.bfloat16)
    assert sc_chunk_size(cache, budget=768, ratio=4) == 16
    assert sc_chunk_size(cache, budget=2**24, ratio=4) == 0


@pytest.mark.parametrize("ratio,kv_tile", [(0, 256), (-1, 256), (4, 0), (8, 130)])
def test_invalid_occupancy_geometry_is_rejected_before_division(ratio, kv_tile):
    from vllm_torchtpu.kernels.sparse_gqa_prefill.occupancy import active_counts

    with pytest.raises(ValueError, match="whole number of groups"):
        active_counts(
            jnp.zeros((256, 1), jnp.int32),
            jnp.ones(256, jnp.int32),
            num_tiles=1,
            ratio=ratio,
            kv_tile=kv_tile,
        )

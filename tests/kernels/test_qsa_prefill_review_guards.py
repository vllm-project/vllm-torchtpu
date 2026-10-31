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
"""CPU regressions for live sparse-prefill addressing and bounded dispatch."""

import functools
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tests.kernels.prefill_work_model_helpers import work_model
from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as tc
from vllm_torchtpu.kernels.qwen4_exp import prefill_tiles
from vllm_torchtpu.kernels.sparse_gqa_prefill import adaptive, grouped
from vllm_torchtpu.kernels.sparse_gqa_prefill.cost_model import tc_cost


@pytest.fixture
def tpu_info(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    return info


def metadata_inputs():
    blocks = np.full((128, 4), -1, np.int32)
    lengths = np.zeros(128, np.int32)
    lengths[:6] = [0, 1, 4, 5, 8, 129]
    blocks[0] = [0, 1, 2, 3]
    blocks[1] = [0, 1, 2, 3]
    blocks[2] = [0, 1, 2, 3]
    blocks[3] = [0, 1, -1, -1]
    blocks[4] = [0, 2, 1, 3]
    blocks[5] = [0, 1, 2, 3]
    table = np.broadcast_to(np.arange(4, dtype=np.int32), (128, 4)).copy()
    return jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table)


def test_bounded_metadata_filters_inactive_prefix_and_implicit_tail():
    blocks, lengths, table = metadata_inputs()
    _, tiles, counts, bits = tc._prepare_metadata_bounded(
        blocks, lengths, table, query_block=128, physical_page_size=128
    )
    actual = np.asarray(bits)[0, :, :128].T
    expected = np.zeros((128, 128), np.int32)
    expected[:6, 0] = [0, 1, 1, 3, 1, 15]
    expected[5, 1] = 1
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.asarray(counts), [1])
    assert int(tiles[0, 0]) == 0


def test_bounded_metadata_duplicate_bits_cannot_carry_to_another_group():
    blocks = jnp.full((128, 4), -1, jnp.int32).at[0, :2].set(0)
    lengths = jnp.zeros(128, jnp.int32).at[0].set(16)
    table = jnp.zeros((128, 1), jnp.int32)
    _, _, _, bits = tc._prepare_metadata_bounded(
        blocks, lengths, table, query_block=128, physical_page_size=128
    )
    assert int(bits[0, 0, 0]) == 1
    np.testing.assert_array_equal(np.asarray(bits)[0, 1:], 0)


def capture_tc(
    monkeypatch,
    query,
    blocks,
    lengths,
    table,
    cache,
    *,
    ratio=4,
    kv_tile=256,
    bf16_io=True,
):
    captured = {}

    def call(kernel, *, out_shape, **kwargs):
        captured.update(kwargs)

        def run(*args):
            captured["args"] = args
            return jnp.zeros(out_shape.shape, out_shape.dtype)

        return run

    monkeypatch.setattr(pl, "pallas_call", call)
    with jax.disable_jit():
        result = tc.block_sparse_gqa_prefill(
            query,
            blocks,
            lengths,
            table,
            cache,
            ratio=ratio,
            kv_heads=2,
            query_block=128,
            kv_tile=kv_tile,
            bf16_io=bf16_io,
        )
    return captured, result


def test_full_metadata_matches_bounded_and_prefetches_last_live_page(
    monkeypatch, tpu_info
):
    blocks, lengths, table = metadata_inputs()
    query = jnp.zeros((128, 4, 128), jnp.bfloat16)
    cache = jnp.zeros((4, 128, 2, 2, 128), jnp.bfloat16)
    captured, _ = capture_tc(monkeypatch, query, blocks, lengths, table, cache)
    args = captured["args"]
    np.testing.assert_array_equal(args[3], [1])
    _, _, _, expected = tc._prepare_metadata_bounded(
        blocks, lengths, table, query_block=128, physical_page_size=128
    )
    np.testing.assert_array_equal(args[5], expected)
    cost = captured["cost_estimate"]
    assert cost.flops > 0 and cost.transcendentals > 0
    assert cost.bytes_accessed >= query.nbytes + cache.nbytes
    assert captured["compiler_params"].vmem_limit_bytes < tpu_info.vmem_capacity_bytes


@pytest.mark.parametrize("length,last", [(0, 0), (1664, 0), (1665, 1)])
def test_prefetch_bound_ignores_padded_table_capacity(
    monkeypatch, tpu_info, length, last
):
    query = jnp.zeros((128, 4, 128), jnp.bfloat16)
    blocks = jnp.full((128, 4), -1, jnp.int32)
    lengths = jnp.full(128, length, jnp.int32)
    table = jnp.zeros((128, 158), jnp.int32).at[:, 2:].set(999999)
    cache = jnp.zeros((2, 1664, 2, 2, 128), jnp.bfloat16)
    captured, _ = capture_tc(monkeypatch, query, blocks, lengths, table, cache)
    assert int(captured["args"][3][0]) == last
    if length == 0:
        np.testing.assert_array_equal(captured["args"][2], 0)


def test_ratio8_page1664_general_tile_is_legal(monkeypatch, tpu_info):
    query = jnp.zeros((128, 4, 128), jnp.bfloat16)
    blocks = jnp.full((128, 4), -1, jnp.int32)
    lengths = jnp.full(128, 1664, jnp.int32)
    table = jnp.zeros((128, 3), jnp.int32)
    cache = jnp.zeros((1, 1664, 2, 2, 128), jnp.bfloat16)
    captured, result = capture_tc(
        monkeypatch, query, blocks, lengths, table, cache, ratio=8
    )
    assert result.shape == query.shape
    assert int(captured["args"][3][0]) == 0


def test_ragged_schedule_may_select_sc_and_counts_request_local_unions():
    rows = 256
    lengths = jnp.full(rows, 2048, jnp.int32)
    ids = jnp.broadcast_to(jnp.arange(512, dtype=jnp.int32), (rows, 512))
    # Identical logical IDs belong to 256 independent request tables.
    starts = jnp.arange(rows + 1, dtype=jnp.int32)
    assert bool(
        adaptive._select_sc_for_chunk(
            ids,
            lengths,
            starts,
            0,
            rows,
            1024,
            jnp.bfloat16,
            tc._ragged_plan(
                rows, 24, 256, 2048, 65536, 158, 512, 256, request_capacity=rows
            ),
            work_model(jnp.bfloat16),
        )
    )
    model = work_model(jnp.bfloat16)
    one = tc_cost(
        model, queries=256, num_tiles=1024, tile_visits=jnp.int32(8), launches=1
    )
    many = tc_cost(
        model,
        queries=256 * rows,
        num_tiles=1024,
        tile_visits=jnp.int32(8 * rows),
        launches=rows,
    )
    np.testing.assert_allclose(many, one * rows)


def test_ragged_general_tc_uses_bounded_pack_and_each_requests_table(
    monkeypatch, tpu_info
):
    sizes = []

    def fake(query, blocks, lengths, table, cache, bitset, **kwargs):
        sizes.append(len(query))
        return jnp.broadcast_to(table[:, :1, None], query.shape).astype(query.dtype)

    monkeypatch.setattr(tc, "_prefill", fake)
    query = jnp.zeros((17, 4, 128), jnp.bfloat16)
    lengths = jnp.ones(17, jnp.int32)
    table = jnp.concatenate((jnp.full((3, 1), 2), jnp.full((14, 1), 7)))
    args = (
        query,
        query,
        jnp.zeros((17, 4), jnp.int32),
        lengths,
        table,
        jnp.zeros((8, 128, 2, 2, 128), jnp.bfloat16),
    )
    short = jnp.array([0, 3, 17], jnp.int32)
    padded = jnp.pad(short, (0, 253), constant_values=17)
    run = functools.partial(
        tc.block_sparse_gqa_attention, kv_heads=2, ratio=4, query_block=128, kv_tile=256
    )
    actual, _ = run(*args, padded)
    expected, _ = run(*args, short)
    plan = tc._ragged_plan(17, 4, 128, 128, 32, 1, 4, 128, request_capacity=17)
    assert sizes and set(sizes) == {plan[0] * 128}
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.asarray(actual[:, 0, 0]), [1] * 3 + [3.5] * 14)


@pytest.mark.parametrize(
    "page_size,columns,heads,chip",
    [
        (896, 293, 12, pltpu.ChipVersion.TPU_7X),
        (1664, 159, 24, pltpu.ChipVersion.TPU_7),
    ],
)
def test_adaptive_admission_uses_resources_not_calibration_geometry(
    monkeypatch, page_size, columns, heads, chip
):
    info = pltpu.get_tpu_info_for_chip(chip, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    shape = jax.ShapeDtypeStruct
    q = shape((256, heads, 256), jnp.bfloat16)
    assert adaptive.is_supported_by_adaptive_prefill(
        q,
        q,
        shape((256, 512), jnp.int32),
        shape((256,), jnp.int32),
        shape((256, columns), jnp.int32),
        shape((columns, page_size, 2, 2, 256), jnp.bfloat16),
        shape((257,), jnp.int32),
        kv_heads=2,
        ratio=4,
        k_scale=1.0,
        v_scale=1.0,
    )


def test_resource_cap_and_working_set_select_smaller_tiles(monkeypatch, tpu_info):
    shape = jax.ShapeDtypeStruct
    query = shape((256, 48, 256), jnp.bfloat16)
    cache = shape((2, 1664, 4, 2, 256), jnp.bfloat16)
    assert prefill_tiles.get_sparse_prefill_tiles(
        query, cache, kv_heads=4, ratio=4
    ) == (128, 128)
    small = replace(tpu_info, vmem_capacity_bytes=16 * 1024**2)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: small)
    qb, kt = prefill_tiles.get_sparse_prefill_tiles(query, cache, kv_heads=4, ratio=4)
    assert qb < 128
    footprint = prefill_tiles._paired_working_set(48, 4, 256, 2, 2, qb, kt)
    assert footprint <= prefill_tiles.prefill_vmem_limit() < small.vmem_capacity_bytes


@pytest.mark.parametrize("tokens", [1, 17, 128])
def test_tiny_ragged_selector_traces_both_branches(tokens):
    ids = jnp.full((tokens, 512), -1, jnp.int32)
    lengths = jnp.zeros(tokens, jnp.int32)
    starts = jnp.array([0, min(1, tokens), tokens], jnp.int32)
    choose = jax.jit(
        functools.partial(
            adaptive._select_sc_for_chunk,
            offset=0,
            size=tokens,
            num_tiles=1024,
            dtype=jnp.bfloat16,
            plan=tc._ragged_plan(
                tokens, 24, 256, 2048, 65536, 158, 512, 256, request_capacity=tokens
            ),
            model=work_model(),
        )
    )(ids, lengths, starts)
    assert choose.shape == ()


@pytest.mark.parametrize(
    "heads,page_size,budget",
    [(12, 128, 512), (24, 896, 512), (24, 256, 16), (24, 2048, 256)],
)
def test_geometry_and_page_use_the_work_model_route(
    monkeypatch, tpu_info, heads, page_size, budget
):
    recorded = []

    def selected(arrays, cache, starts, model):
        recorded.append(model)
        return jnp.zeros_like(arrays[0]), cache

    def forbidden(*args, **kwargs):
        raise AssertionError("Legal geometry bypassed the work model")

    monkeypatch.setattr(tc, "block_sparse_gqa_attention", forbidden)
    monkeypatch.setattr(adaptive, "_one_chunk", selected)
    query = jnp.zeros((256, heads, 256), jnp.bfloat16)
    cache = jnp.zeros((1, page_size, 2, 2, 256), jnp.bfloat16)
    output, returned = adaptive.adaptive_prefill(
        query,
        query,
        jnp.full((256, budget), -1, jnp.int32),
        jnp.zeros(256, jnp.int32),
        jnp.zeros((256, 1), jnp.int32),
        cache,
        jnp.array([0, 256], jnp.int32),
        kv_heads=2,
        ratio=4,
    )
    assert output.shape == query.shape and returned is cache
    assert len(recorded) == 1 and recorded[0].budget == budget


@pytest.mark.parametrize("bf16_io", [False, True])
def test_tc_hint_counts_cast_boundaries_and_main_matrix_products(
    monkeypatch, tpu_info, bf16_io
):
    query = jnp.zeros((128, 4, 128), jnp.bfloat16)
    blocks, lengths, table = metadata_inputs()
    cache = jnp.zeros((4, 128, 2, 2, 128), jnp.bfloat16)
    captured, output = capture_tc(
        monkeypatch,
        query,
        blocks,
        lengths,
        table,
        cache,
        kv_tile=128,
        bf16_io=bf16_io,
    )
    args = captured["args"]
    boundary_dtype = jnp.bfloat16 if bf16_io else jnp.float32
    assert args[4].dtype == output.dtype == boundary_dtype
    repeated_kv_bytes = 4 * 128 * 2 * 2 * 128 * 2
    cost = captured["cost_estimate"]
    assert cost.flops == 4 * 128 * 4 * 128 * 128 * 4
    assert cost.transcendentals == 262144 + 2048
    assert (
        cost.bytes_accessed
        == sum(x.nbytes for x in args[:-1]) + output.nbytes + repeated_kv_bytes
    )


def test_sc_indices_keep_row_ownership_and_only_gather_has_a_byte_hint(
    monkeypatch, tpu_info
):
    original_call = pl.pallas_call
    costs: list[bool | pl.CostEstimate] = []

    def interpreted(kernel, **kwargs):
        costs.append("cost_estimate" in kwargs)
        return original_call(kernel, **kwargs, interpret=True)

    def gather(kernel, **kwargs):
        costs.append(kwargs["cost_estimate"])

        def run(cache, indices):
            return cache.reshape(-1, *cache.shape[3:])[indices]

        return run

    monkeypatch.setattr(pl, "pallas_call", interpreted)
    monkeypatch.setattr(pl, "kernel", gather)
    blocks = jnp.tile(jnp.arange(4, dtype=jnp.int32), (8, 1))
    lengths = jnp.arange(9, 17, dtype=jnp.int32)
    table = jnp.arange(8, dtype=jnp.int32)[:, None] % 2
    cache = jnp.broadcast_to(
        jnp.array([1, 9], jnp.bfloat16)[:, None, None, None, None], (2, 896, 2, 2, 128)
    )
    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("device",),
        abstract_device=jax.sharding.AbstractDevice(
            device_kind="TPU7x", num_cores=1, platform="tpu"
        ),
    )
    original_mesh = grouped.plsc.VectorSubcoreMesh

    def target_mesh(**kwargs):
        with jax.sharding.use_abstract_mesh(mesh):
            return original_mesh(**kwargs)

    monkeypatch.setattr(grouped.plsc, "VectorSubcoreMesh", target_mesh)
    compact, capacity = grouped.gather_interleaved(
        blocks, lengths, table, cache, query_group=8, tile_k=256, ratio=4
    )
    values = np.asarray(compact).reshape(8, capacity, 2, 2, 128)
    for row in range(8):
        np.testing.assert_array_equal(
            values[row, : int(lengths[row])], 1 if row % 2 == 0 else 9
        )
    assert len(costs) == 2 and costs[0] is False
    gather_cost = costs[1]
    assert isinstance(gather_cost, pl.CostEstimate)
    assert gather_cost.flops == gather_cost.transcendentals == 0
    assert gather_cost.bytes_accessed == compact.nbytes * 2 + compact.shape[0] * 2 * 4


def test_sc_attention_hint_separates_score_alpha_and_gate_transcendentals(
    monkeypatch, tpu_info
):
    captured = []

    def compact(*args, **kwargs):
        return jnp.zeros((8 * 256, 2, 2, 128), jnp.bfloat16), 256

    def call(kernel, *, out_shape, **kwargs):
        def run(*args):
            captured.append((kwargs["cost_estimate"], args))
            return jnp.zeros(out_shape.shape, out_shape.dtype)

        return run

    monkeypatch.setattr(grouped, "gather_interleaved", compact)
    monkeypatch.setattr(pl, "pallas_call", call)
    query = jnp.zeros((8, 12, 128), jnp.bfloat16)
    output, _ = grouped.sc_grouped_attention(
        query,
        query,
        jnp.zeros((8, 4), jnp.int32),
        jnp.ones(8, jnp.int32),
        jnp.zeros((8, 1), jnp.int32),
        jnp.zeros((1, 896, 2, 2, 128), jnp.bfloat16),
        kv_heads=2,
        tile_k=256,
    )
    cost, args = captured[0]
    assert args[1].nbytes > query.nbytes
    assert cost.flops == 134217728
    assert cost.transcendentals == 262144 + 128 + 16384
    assert cost.bytes_accessed == sum(x.nbytes for x in args) + args[1].nbytes
    assert args[1].dtype == args[2].dtype == output.dtype == jnp.bfloat16
    assert output.shape == query.shape


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
def test_zero_visit_tc_cost_bounds_all_nonnegative_visit_counts(dtype):
    model = work_model(dtype)
    assert all(coefficient >= 0 for coefficient in model)
    launches = jnp.array([0, 1, 3, 256], jnp.int32)
    visits = jnp.array([0, 1, 23, 1024], jnp.int32)
    lower = tc_cost(
        model,
        queries=256 * launches,
        num_tiles=1024,
        tile_visits=jnp.zeros_like(visits),
        launches=launches,
    )
    complete = tc_cost(
        model,
        queries=256 * launches,
        num_tiles=1024,
        tile_visits=visits,
        launches=launches,
    )
    assert np.all(np.asarray(lower) <= np.asarray(complete))


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
@pytest.mark.parametrize(
    "requests,expected_sc,expected_fine_calls", [(2, False, 2), (256, True, 0)]
)
def test_ragged_lower_bound_skips_only_resolved_sc_work(
    monkeypatch, dtype, requests, expected_sc, expected_fine_calls
):
    calls = []
    original = adaptive.active_counts

    def observed(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(adaptive, "active_counts", observed)
    ids = jnp.full((256, 512), -1, jnp.int32).at[:, :16].set(jnp.arange(16))
    lengths = jnp.full(256, 64, jnp.int32)
    starts = (
        jnp.array([0, 127, 256], jnp.int32)
        if requests == 2
        else jnp.arange(257, dtype=jnp.int32)
    )
    model = work_model(dtype)
    plan = tc._ragged_plan(256, 24, 256, 128, 416, 1, 512, 256, request_capacity=256)
    tile_count = jnp.sum((jnp.diff(starts) + 255) // 256, dtype=jnp.int32)
    launches = (tile_count + plan[0] - 1) // plan[0]
    live_prefix = jnp.concatenate((jnp.zeros(1, jnp.int32), jnp.cumsum(lengths > 0)))
    live_rows = live_prefix[starts[1:]] - live_prefix[starts[:-1]]
    minimum_visits = jnp.sum((live_rows + 255) // 256)
    lower = tc_cost(
        model,
        queries=256 * plan[0] * launches,
        num_tiles=7,
        tile_visits=minimum_visits,
        launches=launches,
    )
    target = adaptive.sc_cost(model, lengths) * 1.02
    assert bool(lower > target) == expected_sc
    # disable_jit executes the selected branch rather than observing tracing.
    with jax.disable_jit():
        actual = adaptive._select_sc_for_chunk(
            ids,
            lengths,
            starts,
            0,
            256,
            7,
            dtype,
            tc._ragged_plan(256, 24, 256, 128, 416, 1, 512, 256, request_capacity=256),
            work_model(dtype),
        )
    assert bool(actual) == expected_sc
    assert len(calls) == expected_fine_calls

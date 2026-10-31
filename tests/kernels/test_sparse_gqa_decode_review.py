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
"""Physical-page lookup, actual-batch planning and SMEM boundaries."""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import schedule as rpa_schedule
from vllm_torchtpu.kernels.sparse_gqa_decode import metadata, planning
from vllm_torchtpu.kernels.sparse_gqa_decode.batching import rpa_query_batch_size
from vllm_torchtpu.kernels.sparse_gqa_decode.eligibility import plan_decode


@pytest.fixture
def info(monkeypatch):
    value = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: value)
    return value


@pytest.mark.parametrize("page,ratio", [(128, 4), (896, 4), (136, 8), (1664, 8)])
def test_physical_pages_and_implicit_tail_match_token_addresses(page, ratio):
    lengths = np.array([page + ratio + 1, 0, 2 * page], np.int32)
    blocks = np.array([[0, 3, 1, 2, 4], [-1] * 5, [5, 2, 3, 0, 1]], np.int32)
    table = np.array([[7, 2, 9], [3, 0, 4], [8, 1, 6]], np.int32)
    expected = np.zeros((3, 6), np.int32)
    wanted_lengths = []
    for row, length in enumerate(lengths):
        count = min(int(length) // ratio, blocks.shape[1])
        groups = list(blocks[row, :count])
        if length % ratio:
            groups.append(int(length) // ratio)
        for col, group in enumerate(groups):
            token = int(group) * ratio
            physical = int(table[row, token // page]) * page + token % page
            expected[row, col] = physical // ratio
        wanted_lengths.append(count * ratio + int(length) % ratio)
    actual, actual_lengths = jax.jit(
        lambda b, lens, pages: metadata.virtual_page_metadata(
            b, lens, pages, ratio=ratio, page_size=page
        )
    )(jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_lengths, wanted_lengths)


def test_wide_table_does_not_stage_a_whole_vmem_window(monkeypatch, info):
    from vllm_torchtpu.kernels.mla.sparse import kernel

    columns = info.vmem_capacity_bytes // (8 * np.dtype(np.int32).itemsize) + 1
    page, ratio = 896, 4
    table = jnp.zeros((1, columns), jnp.int32).at[0, -1].set(7)
    group = (columns - 1) * (page // ratio)

    def reject_oversized_window(*args, **kwargs):
        staged = 8 * ((columns + 127) // 128 * 128) * 4
        assert staged <= info.vmem_capacity_bytes, (staged, info.vmem_capacity_bytes)
        return jnp.zeros((1, 2), jnp.int32)

    monkeypatch.setattr(kernel, "gather_page_ids", reject_oversized_window)
    pages, lengths = metadata.virtual_page_metadata(
        jnp.array([[group]], jnp.int32),
        jnp.array([(group + 1) * ratio], jnp.int32),
        table,
        ratio=ratio,
        page_size=page,
    )
    np.testing.assert_array_equal(pages, [[7 * (page // ratio), 0]])
    np.testing.assert_array_equal(lengths, [ratio])


def test_repeated_ids_and_empty_rows_preserve_existing_decode_slots():
    pages, lengths = metadata.virtual_page_metadata(
        jnp.array([[0, 0, 1], [-1, -1, -1]], jnp.int32),
        jnp.array([13, 0], jnp.int32),
        jnp.array([[4], [2]], jnp.int32),
        ratio=4,
        page_size=128,
    )
    np.testing.assert_array_equal(pages, [[128, 128, 129, 131], [0, 0, 0, 0]])
    np.testing.assert_array_equal(lengths, [13, 0])


def specs(batch, groups, dtype=jnp.bfloat16, kv_heads=2, ratio=4, page=896):
    packing = 32 // jax.dtypes.itemsize_bits(dtype)
    return (
        jax.ShapeDtypeStruct((batch, 12, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((batch, 12, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((batch, groups), jnp.int32),
        jax.ShapeDtypeStruct((batch,), jnp.int32),
        jax.ShapeDtypeStruct((batch, 33), jnp.int32),
        jax.ShapeDtypeStruct(
            (4, page, (2 * kv_heads + packing - 1) // packing, packing, 128), dtype
        ),
    )


def test_actual_33_fits_before_rounded_64_exceeds_schedule(info):
    args = specs(33, 1024)
    launch = plan_decode(*args, kv_heads=2, ratio=4, k_scale=1.0, v_scale=1.0)
    assert launch is not None and launch.chunk_queries == 33
    assert launch.resources.worst_steps == 71
    too_large = planning.make_config(64, 12, 128, 2, 4, 1024, jnp.dtype(jnp.bfloat16))
    assert planning.resource_usage(too_large).worst_steps == 136
    assert planning.resource_usage(too_large).schedule_steps == 128


def test_smaller_power_of_two_fallback_keeps_32(monkeypatch, info):
    seen = []
    original = planning.resource_usage

    def record(cfg):
        seen.append(cfg.serve.num_seqs)
        return original(cfg)

    monkeypatch.setattr(planning, "resource_usage", record)
    launch = plan_decode(
        *specs(65, 1024), kv_heads=2, ratio=4, k_scale=1.0, v_scale=1.0
    )
    assert launch is not None and launch.chunk_queries == 32
    assert seen == [65, 64, 32]


@pytest.mark.parametrize(
    "dtype,kv,ratio,page", [(jnp.bfloat16, 1, 4, 896), (jnp.float8_e4m3fn, 4, 8, 136)]
)
def test_actual_packing_kv_heads_and_page_geometry(info, dtype, kv, ratio, page):
    launch = plan_decode(
        *specs(3, 17, dtype, kv, ratio, page),
        kv_heads=kv,
        ratio=ratio,
        k_scale=1.0,
        v_scale=1.0,
    )
    assert launch is not None and launch.chunk_queries == 3
    assert launch.config.model.num_kv_heads == kv
    assert launch.config.serve.page_size == ratio


@pytest.mark.parametrize("lanes", [128, 64])
def test_rpa_one_row_exact_smem_capacity(monkeypatch, info, lanes):
    columns = 513
    hardware = dataclasses.replace(info, num_lanes=lanes)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: hardware)
    query, _, _, _, _, cache = specs(1, 512)
    cfg = planning.make_config(1, 12, 128, 2, 4, 512, cache.dtype, info=hardware)
    scratch = rpa_schedule.BaseMetadataComputer.get_rpa_schedule(cfg).scratch_shapes()
    scratch_bytes = sum(
        math.prod(ref.shape) * ref.dtype.itemsize for ref in jax.tree.leaves(scratch)
    )
    # Attention: cu_q_lens, offsets, kv_lens, q_lens, virtual page table.
    scalars = tuple(
        jax.ShapeDtypeStruct(shape, jnp.int32)
        for shape in ((2,), (1,), (1,), (1,), (columns,))
    )
    # Schedule generation replaces the virtual page table by distribution[3].
    generation = (*scalars[:-1], jax.ShapeDtypeStruct((3,), jnp.int32))
    scalar_bytes = max(
        sum(ref.size * ref.dtype.itemsize for ref in scalars),
        sum(ref.size * ref.dtype.itemsize for ref in generation),
    )
    required = scratch_bytes + scalar_bytes
    hardware = dataclasses.replace(hardware, smem_capacity_bytes=required)
    assert (
        rpa_query_batch_size(columns, query=query, cache=cache, kv_heads=2, ratio=4)
        == 1
    )
    hardware = dataclasses.replace(hardware, smem_capacity_bytes=required - 1)
    with pytest.raises(ValueError, match="One QSA page-table row.*SMEM"):
        rpa_query_batch_size(columns, query=query, cache=cache, kv_heads=2, ratio=4)


@pytest.mark.parametrize("columns", [0, -1, 1_000_000])
def test_bad_width_and_unfittable_row_fail_before_launch(info, columns):
    q, _, _, _, _, cache = specs(1, 1)
    with pytest.raises(ValueError, match="positive|One QSA page-table row"):
        rpa_query_batch_size(columns, query=q, cache=cache, kv_heads=2, ratio=4)


@pytest.mark.parametrize("page,route", [(136, "rpa"), (256, "tc")])
def test_prefill_dma_layout_selects_bounded_fallback(monkeypatch, info, page, route):
    from vllm_torchtpu.kernels.qwen4_exp import (
        block_sparse_gqa_prefill,
        prefill_tiles,
        qsa_rpa,
    )
    from vllm_torchtpu.kernels.sparse_gqa_prefill import adaptive

    query = jnp.arange(3 * 12 * 128, dtype=jnp.float32).reshape(3, 12, 128)
    query = (query / query.size).astype(jnp.bfloat16)
    gate = -query
    blocks = jnp.zeros((3, 17), jnp.int32)
    lengths = jnp.array([137, 0, 139], jnp.int32)
    table = jnp.array([[3, 1], [2, 0], [0, 2]], jnp.int32)
    cache = jnp.zeros((4, page, 1, 4, 128), jnp.float8_e4m3fn)
    operands = (query, gate, blocks, lengths, table, cache)
    options = dict(kv_heads=2, ratio=4, k_scale=0.375, v_scale=2.5)
    expected = jnp.where((lengths > 0)[:, None, None], query + gate * 2, 0)
    calls = []

    def rpa(*args, **kwargs):
        assert len(args) == len(operands)
        assert all(actual is original for actual, original in zip(args, operands))
        assert int(kwargs.pop("active_tokens")) == 3
        assert kwargs == options
        calls.append("rpa")
        return expected, cache

    def tiles(q, c, *, kv_heads, ratio):
        assert q is query and c is cache
        assert (kv_heads, ratio) == (2, 4)
        assert route == "tc"
        return 128, 128

    def tc(*args, **kwargs):
        assert len(args) == 7 and args[-1] is starts
        assert all(actual is original for actual, original in zip(args, operands))
        assert kwargs == dict(options, query_block=128, kv_tile=128)
        calls.append("tc")
        return expected, cache

    starts = jnp.array([0, 1, 1, 3], jnp.int32)
    monkeypatch.setattr(
        adaptive, "is_supported_by_adaptive_prefill", lambda *a, **k: False
    )
    monkeypatch.setattr(qsa_rpa, "_rpa_attention_in_chunks", rpa)
    monkeypatch.setattr(prefill_tiles, "get_sparse_prefill_tiles", tiles)
    monkeypatch.setattr(block_sparse_gqa_prefill, "block_sparse_gqa_attention", tc)
    output, owner = qsa_rpa._prefill_attention(*operands, starts, **options)
    np.testing.assert_array_equal(output, expected)
    assert owner is cache
    assert calls == [route]


@pytest.mark.parametrize("page", [896, 1664])
def test_small_vector_gather_keeps_physical_page_mapping(monkeypatch, info, page):
    from vllm_torchtpu.kernels.mla.sparse import kernel

    ratio, budget = 4, 512
    lengths = np.array([page + 5, 0], np.int32)
    blocks = np.full((2, budget), -1, np.int32)
    blocks[0, : lengths[0] // ratio] = np.arange(lengths[0] // ratio)
    table = np.array([[3, 1], [2, 0]], np.int32)
    calls = []

    def vector_gather(page_indices, logical, seq_ids, max_seqs, *, block_tokens):
        calls.append(block_tokens)
        windows = page_indices.reshape(max_seqs, -1)[seq_ids]
        return jnp.take_along_axis(windows, logical, axis=1)

    monkeypatch.setattr(kernel, "gather_page_ids", vector_gather)
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    actual, kv_lens = metadata.virtual_page_metadata(
        jnp.asarray(blocks),
        jnp.asarray(lengths),
        jnp.asarray(table),
        ratio=ratio,
        page_size=page,
    )
    expected = np.zeros((2, budget + 1), np.int32)
    for row, length in enumerate(lengths):
        selected = list(blocks[row, : min(int(length) // ratio, budget)])
        if length % ratio:
            selected.append(int(length) // ratio)
        for col, group in enumerate(selected):
            token = int(group) * ratio
            expected[row, col] = (
                int(table[row, token // page]) * page + token % page
            ) // ratio
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(kv_lens, lengths)
    assert calls == [info.num_sublanes]


def test_vector_gather_cpu_dispatch_never_queries_hardware(monkeypatch):
    def forbidden():
        raise AssertionError("CPU metadata must use the generic gather")

    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(pltpu, "get_tpu_info", forbidden)
    table = jax.ShapeDtypeStruct((8, 1), jnp.int32)
    logical = jax.ShapeDtypeStruct((8, 513), jnp.int32)
    assert metadata._vector_gather_rows(table, logical) is None


def test_vector_gather_empty_metadata_never_launches(monkeypatch):
    from vllm_torchtpu.kernels.mla.sparse import kernel

    def forbidden(*args, **kwargs):
        raise AssertionError("Empty metadata must not launch the vector gather")

    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(pltpu, "get_tpu_info", forbidden)
    monkeypatch.setattr(kernel, "gather_page_ids", forbidden)
    pages, lengths = metadata.virtual_page_metadata(
        jnp.zeros((0, 512), jnp.int32),
        jnp.zeros((0,), jnp.int32),
        jnp.zeros((0, 1), jnp.int32),
        ratio=4,
        page_size=896,
    )
    assert pages.shape == (0, 513) and pages.dtype == jnp.int32
    assert lengths.shape == (0,) and lengths.dtype == jnp.int32


@pytest.mark.parametrize("oversized", ["window", "selected", "capacity", "dtype"])
def test_vector_gather_rejects_unbounded_footprint(monkeypatch, info, oversized):
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    columns, selected, dtype = 1, 513, jnp.int32
    if oversized == "window":
        columns = info.num_lanes + 1
    elif oversized == "selected":
        selected = info.vmem_capacity_bytes // (info.num_sublanes * 4) + 1
    elif oversized == "capacity":
        hardware = dataclasses.replace(info, vmem_capacity_bytes=4)
        monkeypatch.setattr(pltpu, "get_tpu_info", lambda: hardware)
    else:
        dtype = jnp.float32
    table = jax.ShapeDtypeStruct((8, columns), dtype)
    logical = jax.ShapeDtypeStruct((8, selected), jnp.int32)
    assert metadata._vector_gather_rows(table, logical) is None


@pytest.mark.parametrize("active", [0, 1, 7, 13])
@pytest.mark.parametrize("fill", [0, 19])
def test_runtime_prefix_skips_empty_chunks_and_preserves_rows(active, fill):
    from vllm_torchtpu.kernels.sparse_gqa_decode.batching import attend_in_chunks

    query = jnp.arange(13 * 128, dtype=jnp.float32).reshape(13, 1, 128)
    lengths = jnp.full((13,), 17, jnp.int32)
    ids = jnp.arange(13, dtype=jnp.int32)[:, None]
    cache = jnp.arange(256, dtype=jnp.float32).reshape(2, 128)
    visited = []

    def attend(q, gate, blocks, lens, table, owner):
        visited.extend(np.asarray(blocks)[np.asarray(lens) > 0, 0].tolist())
        return jnp.where((lens > 0)[:, None, None], q + table[:, :1, None], 0), owner

    with jax.disable_jit():
        output, owner = attend_in_chunks(
            attend,
            query,
            query,
            ids,
            lengths,
            ids,
            cache,
            batch_size=4,
            active_tokens=jnp.array(active, jnp.int32),
            initial_output=jnp.full_like(query, fill),
        )
    expected = np.full(query.shape, fill, np.float32)
    expected[:active] = np.asarray(query[:active]) + np.arange(active)[:, None, None]
    np.testing.assert_array_equal(output, expected)
    assert visited == list(range(active))
    assert owner is cache

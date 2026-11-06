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
"""Transfer concurrency must depend on live blocks, not allocator page length."""

import dataclasses

import jax
import jax.numpy as jnp
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import tuned_block_sizes as policy


@pytest.fixture(autouse=True)
def hardware_descriptor(monkeypatch):
    from vllm_torchtpu.kernels.qwen4_exp import tuned_block_sizes

    def descriptor():
        return tuned_block_sizes.DmaHardware.from_tpu_info(pltpu.get_tpu_info())

    monkeypatch.setattr(tuned_block_sizes, "_hardware", descriptor)


def hardware():
    return policy.DmaHardware.from_tpu_info(
        pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    )


@pytest.mark.parametrize(
    "case",
    [
        "decode",
        "slots32",
        "large",
        "requests",
        "width",
        "ratio",
        "fp8",
        "short",
        "chip",
    ],
)
def test_fused_state_derives_layout_and_capacity(case, monkeypatch):
    chip = pltpu.ChipVersion.TPU_V5P if case == "chip" else pltpu.ChipVersion.TPU_7X
    monkeypatch.setattr(
        policy.pltpu, "get_tpu_info", lambda: pltpu.get_tpu_info_for_chip(chip, 1)
    )
    tokens = 128 if case == "large" else 16
    requests = 8 if case == "requests" else 32 if case == "slots32" else 16
    width = 256 if case == "width" else 128
    ratio = 8 if case == "ratio" else 4
    dtype = jnp.float8_e4m3fn if case == "fp8" else jnp.bfloat16
    pool = carrier(4 if case == "short" else 208, dtype, width)
    raw = jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16)
    assert policy.use_decode_state_fusion(
        raw, pool, requests=requests, ratio=ratio
    ) == (case not in ("fp8", "short", "large", "requests"))


@pytest.mark.parametrize(
    "tokens,requests,width,ratio,expected",
    [
        (8192, 16, 128, 4, False),
        (128, 16, 128, 4, False),
        (17, 16, 128, 4, False),
        (16, 16, 128, 4, True),
        (3, 3, 256, 8, True),
        (5, 8, 128, 2, True),
        (33, 64, 128, 4, True),
    ],
)
def test_state_schedule_routes_whole_query_bucket_or_bounded_writer(
    monkeypatch, tokens, requests, width, ratio, expected
):
    from jax.experimental import pallas as pl

    from vllm_torchtpu.kernels.qwen4_exp import compressed_prepare, pooled_qsa

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(policy.pltpu, "get_tpu_info", lambda: info)
    arr = jax.ShapeDtypeStruct
    pool = carrier(208, jnp.bfloat16, width)
    keys = arr((tokens, width), jnp.bfloat16)
    fused = policy.use_decode_state_fusion(
        keys, pool, requests=requests, ratio=ratio, position_axes=3
    )
    assert fused == expected
    routes = []

    def whole_batch(owner, values, *args, **kwargs):
        assert expected
        assert values.shape == (tokens, 2, width)
        routes.append("whole_query_bucket")
        return owner

    def tiled(kernel, **kwargs):
        assert not expected
        assert kwargs["name"] == "qwen4_exp_compressed_kv_page_tiled_scatter"
        assert kwargs["grid_spec"].in_specs[0].block_shape == (256, width)
        routes.append("bounded_writer")
        return lambda *args: args[-1]

    monkeypatch.setattr(compressed_prepare, "write_prepared_pairs", whole_batch)
    monkeypatch.setattr(pl, "pallas_call", tiled)
    args = (
        pool,
        keys,
        arr((tokens,), jnp.bool_),
        arr((tokens,), jnp.int32),
        arr((tokens,), jnp.int32),
        arr((requests + 1,), jnp.int32),
        arr((requests,), jnp.int32),
        arr((requests, 21), jnp.int32),
    )
    result = jax.eval_shape(
        lambda *arrays: pooled_qsa.write_compressed(*arrays, ratio, batch_pairs=fused),
        *args,
    )
    assert result == pool
    assert routes == ["whole_query_bucket" if expected else "bounded_writer"]


@pytest.mark.parametrize("chip", [pltpu.ChipVersion.TPU_7X, pltpu.ChipVersion.TPU_V5P])
def test_decode_index_tile_uses_native_layout_and_resource_budget(chip, monkeypatch):
    monkeypatch.setattr(
        policy.pltpu, "get_tpu_info", lambda: pltpu.get_tpu_info_for_chip(chip, 1)
    )
    query = jax.ShapeDtypeStruct((16, 4, 128), jnp.bfloat16)
    for dtype, packing in ((jnp.bfloat16, 2), (jnp.uint8, 4)):
        cache = jax.ShapeDtypeStruct((225069, 16, packing, 128), dtype)
        pages = policy.get_decode_index_pages(
            query, cache, pages_per_query=533, ratio=4, topk=512
        )
        assert pages == (32 if dtype == jnp.bfloat16 else 8)
        assert policy.get_decode_index_pages(
            query, cache, pages_per_query=16, ratio=4, topk=512
        ) == (16 if dtype == jnp.bfloat16 else 8)


def carrier(page=384, dtype=jnp.float8_e4m3fn, width=256, heads=1):
    return jax.ShapeDtypeStruct(
        (20, page, heads, 4 // jnp.dtype(dtype).itemsize, width), dtype
    )


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
@pytest.mark.parametrize("page", [128, 256, 384, 896, 1664, 1792])
def test_same_payload_uses_batch_independently_of_page_stride(dtype, page, monkeypatch):
    monkeypatch.setattr(
        policy.pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    pool = carrier(page, dtype)
    rows = jax.ShapeDtypeStruct((128, *pool.shape[2:]), dtype)
    assert (
        policy.get_ring_tag_dma_batch(pool, 8, ring_elements=1024, tag_elements=16) == 8
    )
    assert policy.get_main_kv_dma_batch(pool, rows) == 8


@pytest.mark.parametrize(
    "requests,batch", [(1, 1), (6, 2), (7, 1), (8, 8), (12, 4), (16, 8)]
)
def test_request_divisibility_does_not_drop_tail_requests(requests, batch):
    assert (
        policy.ring_tag_dma_batch(
            carrier(),
            requests,
            ring_elements=1024,
            tag_elements=16,
            hardware=hardware(),
        )
        == batch
    )


def test_padded_live_memory_and_boundary_admission():
    hw = hardware()
    pool = carrier()
    memory = policy.ring_tag_dma_memory(pool, 8, 8, 1024, 16, hw)
    # Logical byte counts miss padding: three FP8 token rows
    # each occupy a32-row x256-column plane, and both typed planes pad too.
    assert memory.scratch == 8 * 3 * 32 * 256
    assert memory.buffers == 2 * (8 * 16 * 256 * 2 + 8 * 8 * 256 * 4)
    assert memory.headroom == memory.scratch + memory.buffers // 2 + 8 * 128 * 4

    def choose(facts):
        return policy.ring_tag_dma_batch(
            pool, 8, ring_elements=1024, tag_elements=16, hardware=facts
        )

    assert choose(dataclasses.replace(hw, vmem_bytes=memory.vmem_bytes)) == 8
    assert choose(dataclasses.replace(hw, vmem_bytes=memory.vmem_bytes - 1)) == 4
    assert choose(dataclasses.replace(hw, smem_bytes=memory.scalar)) == 8
    assert choose(dataclasses.replace(hw, smem_bytes=memory.scalar - 1)) == 4
    assert choose(dataclasses.replace(hw, vmem_bytes=1)) == 1
    assert choose(None) == 1


@pytest.mark.parametrize(
    "pool",
    [
        jax.ShapeDtypeStruct((20, 384, 1, 3, 256), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((20, 384, 1, 4, 192), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((20, 384, 1, 4, 256), jnp.float32),
        jax.ShapeDtypeStruct((20, 0, 1, 4, 256), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((2**31, 384, 1, 4, 256), jnp.float8_e4m3fn),
    ],
)
def test_invalid_layout_does_not_enable_batched_typed_io(pool):
    assert (
        policy.ring_tag_dma_batch(
            pool, 8, ring_elements=1024, tag_elements=16, hardware=hardware()
        )
        == 1
    )


def test_insufficient_region_capacity_and_main_partial_tiles():
    hw = hardware()
    assert (
        policy.ring_tag_dma_batch(
            carrier(page=2), 8, ring_elements=1024, tag_elements=16, hardware=hw
        )
        == 1
    )
    for count, page in [(129, 384), (128, 129), (0, 384)]:
        pool = carrier(page)
        rows = jax.ShapeDtypeStruct((count, *pool.shape[2:]), pool.dtype)
        assert policy.main_kv_dma_batch(pool, rows, hardware=hw) == 1
    pool = carrier()
    rows = jax.ShapeDtypeStruct((128, *pool.shape[2:]), jnp.bfloat16)
    assert policy.main_kv_dma_batch(pool, rows, hardware=hw) == 1
    rows = jax.ShapeDtypeStruct((128, *pool.shape[2:]), pool.dtype)
    memory = policy.main_kv_dma_memory(pool, rows, 8, hw)
    assert memory.buffers == 2 * 128 * 32 * 256
    assert (
        policy.main_kv_dma_batch(
            pool,
            rows,
            hardware=dataclasses.replace(hw, vmem_bytes=memory.vmem_bytes - 1),
        )
        == 1
    )


@pytest.mark.parametrize(
    "error", [ValueError("unsupported device"), RuntimeError("backend failed")]
)
def test_public_hardware_errors_propagate(monkeypatch, error):
    def broken():
        raise error

    monkeypatch.setattr(policy.pltpu, "get_tpu_info", broken)
    rows = jax.ShapeDtypeStruct((128, *carrier().shape[2:]), carrier().dtype)
    with pytest.raises(type(error)) as caught:
        policy.get_main_kv_dma_batch(carrier(), rows)
    assert caught.value is error
    with pytest.raises(type(error)) as caught:
        policy.get_ring_tag_dma_batch(carrier(), 8, ring_elements=1024, tag_elements=16)
    assert caught.value is error


def test_pure_planning_never_queries_backend(monkeypatch):
    def forbidden():
        pytest.fail("pure planning queried a device")

    monkeypatch.setattr(policy.pltpu, "get_tpu_info", forbidden)
    hw = hardware()
    for width, heads in [(128, 2), (256, 1), (384, 2)]:
        pool = carrier(width=width, heads=heads)
        assert (
            policy.ring_tag_dma_batch(
                pool, 8, ring_elements=1536, tag_elements=32, hardware=hw
            )
            == 8
        )
        rows = jax.ShapeDtypeStruct((128, *pool.shape[2:]), pool.dtype)
        assert policy.main_kv_dma_batch(pool, rows, hardware=hw) == 8


@pytest.mark.parametrize(
    "chip",
    [pltpu.ChipVersion.TPU_V5P, pltpu.ChipVersion.TPU_V6E, pltpu.ChipVersion.TPU_7X],
)
def test_public_capacities_and_layout_determine_concurrency(chip):
    hw = policy.DmaHardware.from_tpu_info(pltpu.get_tpu_info_for_chip(chip, 1))
    pool = carrier()
    rows = jax.ShapeDtypeStruct((128, *pool.shape[2:]), pool.dtype)
    assert (
        policy.ring_tag_dma_batch(
            pool, 16, ring_elements=1024, tag_elements=16, hardware=hw
        )
        == 8
    )
    assert policy.main_kv_dma_batch(pool, rows, hardware=hw) == 8
    constrained = dataclasses.replace(hw, vmem_bytes=1)
    assert (
        policy.ring_tag_dma_batch(
            pool, 16, ring_elements=1024, tag_elements=16, hardware=constrained
        )
        == 1
    )
    assert policy.main_kv_dma_batch(pool, rows, hardware=constrained) == 1


@pytest.mark.parametrize("chip", [pltpu.ChipVersion.TPU_7X, pltpu.ChipVersion.TPU_V5P])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
@pytest.mark.parametrize("count", [8, 16, 32])
def test_main_decode_policy_fits_all_resident_small_buckets(chip, dtype, count):
    hw = policy.DmaHardware.from_tpu_info(pltpu.get_tpu_info_for_chip(chip, 1))
    pool = carrier(1664, dtype)
    rows = jax.ShapeDtypeStruct((count, *pool.shape[2:]), dtype)
    assert (
        policy.main_kv_dma_batch(
            pool, rows, hardware=dataclasses.replace(hw, smem_bytes=1)
        )
        == 1
    )
    assert (
        policy.main_kv_dma_batch(
            pool, rows, hardware=dataclasses.replace(hw, vmem_bytes=1)
        )
        == 1
    )


def test_main_decode_counts_all_sixteen_dma_semaphores():
    hw = hardware()
    pool = carrier(1664)
    rows = jax.ShapeDtypeStruct((16, *pool.shape[2:]), pool.dtype)
    memory = policy.main_kv_dma_memory(pool, rows, 8, hw)
    assert memory.scalar == 4 * 16 * 4 + policy._COMPILER_SMEM_RESERVE
    assert (
        policy.main_kv_dma_batch(
            pool, rows, hardware=dataclasses.replace(hw, smem_bytes=memory.scalar - 1)
        )
        == 1
    )


@pytest.mark.parametrize(
    "shape,dtype,rotary,expected",
    [
        ((16, 24, 512), jnp.bfloat16, 64, True),
        ((8, 24, 512), jnp.bfloat16, 64, True),
        ((16, 24, 512), jnp.float32, 64, False),
        ((16, 24, 512), jnp.bfloat16, 128, True),
    ],
)
def test_query_prepare_derives_norm_rope_layout_and_dtype(
    shape, dtype, rotary, expected, monkeypatch
):
    monkeypatch.setattr(
        policy.pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    q = jax.ShapeDtypeStruct(shape, dtype)
    norm = jax.ShapeDtypeStruct((256,), jnp.bfloat16)
    trig = jax.ShapeDtypeStruct((67584, 128), jnp.bfloat16)
    assert policy.use_query_prepare(q, norm, trig, rotary) == expected


@pytest.mark.parametrize(
    "count,ratio,rotary,expected",
    [(16, 4, 64, True), (8, 4, 64, True), (16, 8, 64, True), (16, 4, 128, True)],
)
def test_compressed_preparation_uses_resident_layout(
    count, ratio, rotary, expected, monkeypatch
):
    monkeypatch.setattr(policy, "_hardware", hardware)
    raw = jax.ShapeDtypeStruct((count, 128), jnp.bfloat16)
    trig = jax.ShapeDtypeStruct((67584, 128), jnp.bfloat16)
    assert policy.use_compressed_preparation(raw, trig, ratio, rotary) == expected

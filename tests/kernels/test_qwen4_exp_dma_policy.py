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


def hardware():
    return policy.DmaHardware.from_tpu_info(
        pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    )


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
    for count, page in [(127, 384), (128, 129), (0, 384)]:
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


def test_unknown_hardware_falls_back_but_unrelated_errors_propagate(monkeypatch):
    def unsupported():
        raise ValueError(
            "Unsupported TPU device kind: TPU-future. If you are not running a TPU"
        )

    monkeypatch.setattr(policy.pltpu, "get_tpu_info", unsupported)
    assert (
        policy.get_ring_tag_dma_batch(carrier(), 8, ring_elements=1024, tag_elements=16)
        == 1
    )
    rows = jax.ShapeDtypeStruct((128, *carrier().shape[2:]), carrier().dtype)
    assert policy.get_main_kv_dma_batch(carrier(), rows) == 1

    def broken():
        raise RuntimeError("backend session failed")

    monkeypatch.setattr(policy.pltpu, "get_tpu_info", broken)
    with pytest.raises(RuntimeError, match="backend session failed"):
        policy.get_main_kv_dma_batch(carrier(), rows)


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


def test_unmeasured_chip_keeps_serial_policy_with_known_resource_facts():
    hw = policy.DmaHardware.from_tpu_info(
        pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1)
    )
    pool = carrier()
    rows = jax.ShapeDtypeStruct((128, *pool.shape[2:]), pool.dtype)
    assert (
        policy.ring_tag_dma_batch(
            pool, 16, ring_elements=1024, tag_elements=16, hardware=hw
        )
        == 1
    )
    assert policy.main_kv_dma_batch(pool, rows, hardware=hw) == 1


def test_registered_unknown_chip_variant_does_not_acquire_measured_policy():
    from jax._src.tpu_info import ChipVersionBase

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    facts = policy.DmaHardware.from_tpu_info(
        dataclasses.replace(info, chip_version=ChipVersionBase())
    )
    assert facts.chip_version is None
    assert (
        policy.ring_tag_dma_batch(
            carrier(), 16, ring_elements=1024, tag_elements=16, hardware=facts
        )
        == 1
    )

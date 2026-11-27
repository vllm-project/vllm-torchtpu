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
"""Tests for the shared-memory KV staging pool."""

import queue
import uuid

import pytest
import torch

from vllm_torchtpu.distributed.kv_transfer.host_kv_shm import HostKVShmPool, PoolSpec


def _pool_spec() -> PoolSpec:
    return PoolSpec(
        num_slots=1,
        tp_size=2,
        num_layers=3,
        max_blocks=4,
        layer_shard_shape=(4, 2),
        dtype=torch.uint8,
    )


def test_unpack_rank_layers_copies_used_bytes_and_preserves_padding():
    spec = _pool_spec()
    pool = HostKVShmPool.create(spec, f"test_host_kv_{uuid.uuid4().hex}")
    try:
        pool._shm.buf[:] = b"\xee" * spec.total_bytes
        num_blocks = 2
        used = pool._used_per_layer(num_blocks)
        payloads = [
            memoryview(bytes([layer + 1]) * used) for layer in range(spec.num_layers)
        ]

        pool.unpack_rank_layers(
            slot_idx=0, rank=1, num_blocks=num_blocks, layer_buffers=payloads
        )

        rank_start = spec.per_rank_bytes
        for layer in range(spec.num_layers):
            layer_off = rank_start + layer * spec.per_layer_bytes
            assert bytes(pool._shm.buf[layer_off : layer_off + used]) == (
                bytes([layer + 1]) * used
            )
            assert bytes(
                pool._shm.buf[layer_off + used : layer_off + spec.per_layer_bytes]
            ) == (b"\xee" * (spec.per_layer_bytes - used))
    finally:
        pool.close()


def test_unpack_rank_layers_rejects_wrong_layer_size():
    spec = _pool_spec()
    pool = HostKVShmPool.create(spec, f"test_host_kv_{uuid.uuid4().hex}")
    try:
        with pytest.raises(RuntimeError, match="layer 0 size"):
            pool.unpack_rank_layers(
                slot_idx=0, rank=0, num_blocks=2, layer_buffers=[b"x"] * spec.num_layers
            )
    finally:
        pool.close()


def test_stale_release_after_reacquire_does_not_free_a_held_slot():
    """A duplicate release that arrives after the slot was handed to another
    transfer must raise and must not put the slot back on the free list.
    """
    pool = HostKVShmPool.create(_pool_spec(), f"test_host_kv_{uuid.uuid4().hex}")
    try:
        first = pool.acquire_slot(timeout=0, owner="req-a")
        pool.release_slot(first, owner="req-a")
        second = pool.acquire_slot(timeout=0, owner="req-b")
        assert second == first

        with pytest.raises(ValueError, match="held by owner='req-b'"):
            pool.release_slot(first, owner="req-a")

        with pytest.raises(queue.Empty):
            pool.acquire_slot(timeout=0, owner="req-c")

        # The rightful owner can still release, after which a new transfer acquires.
        pool.release_slot(second, owner="req-b")
        third = pool.acquire_slot(timeout=0, owner="req-c")
        assert third == second
    finally:
        pool.close()


def test_double_release_of_free_slot_raises_and_does_not_duplicate():
    pool = HostKVShmPool.create(_pool_spec(), f"test_host_kv_{uuid.uuid4().hex}")
    try:
        # Use falsy owner=0 to verify ownership checks use `is None`, not truthiness.
        slot = pool.acquire_slot(timeout=0, owner=0)
        pool.release_slot(slot, owner=0)

        with pytest.raises(ValueError, match="not currently held"):
            pool.release_slot(slot, owner=0)

        # Free list must contain exactly one copy of the single slot.
        reacquired = pool.acquire_slot(timeout=0, owner=102)
        assert reacquired == slot
        with pytest.raises(queue.Empty):
            pool.acquire_slot(timeout=0, owner=103)
    finally:
        pool.close()


def test_release_slot_rejects_none_owner_and_unknown_slot():
    pool = HostKVShmPool.create(_pool_spec(), f"test_host_kv_{uuid.uuid4().hex}")
    try:
        slot = pool.acquire_slot(timeout=0, owner=1)
        with pytest.raises(ValueError, match="non-None owner"):
            pool.release_slot(slot, owner=None)
        with pytest.raises(ValueError, match="not currently held"):
            pool.release_slot(99, owner=1)
    finally:
        pool.close()


def test_acquire_without_owner_uses_unforgeable_sentinel():
    pool = HostKVShmPool.create(_pool_spec(), f"test_host_kv_{uuid.uuid4().hex}")
    try:
        slot = pool.acquire_slot(timeout=0)
        with pytest.raises(ValueError, match="held by owner="):
            pool.release_slot(slot, owner="arbitrary")
        with pytest.raises(queue.Empty):
            pool.acquire_slot(timeout=0)
    finally:
        pool.close()

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
"""Tests for the HMA shared-memory KV staging pool."""

import uuid

import torch

from vllm_torchtpu.distributed.kv_transfer.host_kv_shm_hma import (
    HostKVShmPoolHMA, PoolSpecHMA)


def _pool_spec() -> PoolSpecHMA:
    return PoolSpecHMA(
        num_slots=2,
        tp_size=2,
        num_arrays=3,
        array_inner_shape=((2, 2), (3, ), (2, )),
        array_dtype=(torch.uint8, torch.int16, torch.uint8),
        array_max_blocks=(4, 2, 4),
        array_to_group=(0, 1, 0),
    )


def _make_pool() -> HostKVShmPoolHMA:
    return HostKVShmPoolHMA.create(_pool_spec(),
                                   f"test_host_kv_hma_{uuid.uuid4().hex}")


def test_spec_byte_accounting():
    spec = _pool_spec()
    # array 0: 4 blocks * (2*2) * 1 byte = 16
    # array 1: 2 blocks * (3)   * 2 byte = 12
    # array 2: 4 blocks * (2)   * 1 byte = 8
    assert spec.array_max_bytes(0) == 16
    assert spec.array_max_bytes(1) == 12
    assert spec.array_max_bytes(2) == 8
    assert spec.array_offset_in_rank(0) == 0
    assert spec.array_offset_in_rank(1) == 16
    assert spec.array_offset_in_rank(2) == 28
    assert spec.per_rank_bytes == 36
    assert spec.per_slot_bytes == 72
    assert spec.total_bytes == 144


def test_unpack_rank_layers_copies_used_bytes_and_preserves_padding():
    spec = _pool_spec()
    pool = _make_pool()
    try:
        pool._shm.buf[:] = b"\xee" * spec.total_bytes
        blocks = [2, 1]
        payloads = [
            memoryview(bytes([a + 1]) * spec.array_used_bytes(a, blocks))
            for a in range(spec.num_arrays)
        ]

        pool.unpack_rank_layers(slot_idx=0,
                                rank=1,
                                blocks=blocks,
                                layer_buffers=payloads)

        for a in range(spec.num_arrays):
            off = pool._array_offset(0, 1, a)
            used = spec.array_used_bytes(a, blocks)
            full = spec.array_max_bytes(a)
            # Used prefix carries the payload...
            assert bytes(pool._shm.buf[off:off +
                                       used]) == bytes([a + 1]) * used
            # ...and the padding tail is untouched.
            assert bytes(pool._shm.buf[off + used:off +
                                       full]) == (b"\xee" * (full - used))
    finally:
        pool.close()

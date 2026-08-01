# SPDX-License-Identifier: Apache-2.0
"""Hybrid shared-memory KV staging pool for the HMA TPUConnector.

This is the HMA analogue of ``host_kv_shm.HostKVShmPool``.
Where the uniform pool assumes every layer has the same shard shape, dtype and
block count, this pool stores a flat list of *arrays* whose inner shape, dtype
and per-request block count vary.

Layout (single SharedMemory block per host):

    slot 0: [rank 0 array 0 .. array A-1] [rank 1 array 0 .. A-1] ... [rank N-1]
    slot 1: ...
"""

from dataclasses import dataclass

import torch

from vllm_torchtpu.distributed.kv_transfer.host_kv_shm import (HostKVShmPool,
                                                               _dtype_bytes)


@dataclass(frozen=True)
class PoolSpecHMA:
    """Describes the per-host memory pool. Must match byte-for-byte
    across ranks. ``num_arrays`` is the flattened kv-array count (mamba
    layers contribute 2 arrays each)."""
    num_slots: int
    tp_size: int
    num_arrays: int
    # Per-array inner shape.
    array_inner_shape: tuple
    # Per-array torch dtype.
    array_dtype: tuple
    # Per-array max block count.
    array_max_blocks: tuple
    # Per-array kv-cache-group id (index into the ``blocks`` token).
    array_to_group: tuple

    def _inner_numel(self, a: int) -> int:
        n = 1
        for d in self.array_inner_shape[a]:
            n *= d
        return n

    def array_max_bytes(self, a: int) -> int:
        return (self.array_max_blocks[a] * self._inner_numel(a) *
                _dtype_bytes(self.array_dtype[a]))

    def array_offset_in_rank(self, a: int) -> int:
        off = 0
        for i in range(a):
            off += self.array_max_bytes(i)
        return off

    @property
    def per_rank_bytes(self) -> int:
        return sum(self.array_max_bytes(a) for a in range(self.num_arrays))

    @property
    def per_slot_bytes(self) -> int:
        return self.tp_size * self.per_rank_bytes

    @property
    def total_bytes(self) -> int:
        return self.num_slots * self.per_slot_bytes

    def array_num_blocks(self, a: int, blocks) -> int:
        n = int(blocks[self.array_to_group[a]])
        if n > self.array_max_blocks[a]:
            raise RuntimeError(
                f"array {a}: runtime block count {n} exceeds reserved "
                f"capacity array_max_blocks={self.array_max_blocks[a]}")
        return n

    def array_used_bytes(self, a: int, blocks) -> int:
        return (self.array_num_blocks(a, blocks) * self._inner_numel(a) *
                _dtype_bytes(self.array_dtype[a]))


class HostKVShmPoolHMA(HostKVShmPool):
    """Per-host shared-memory pool for HMA kv arrays. Rank 0
    create()`s, others attach().
    """

    def _array_offset(self, slot_idx: int, rank: int, array_idx: int) -> int:
        spec = self.spec
        return (slot_idx * spec.per_slot_bytes + rank * spec.per_rank_bytes +
                spec.array_offset_in_rank(array_idx))

    # ---- Tensor / buffer views -----------------------------------------
    def layer_view(self, slot_idx: int, rank: int, array_idx: int,
                   blocks) -> torch.Tensor:
        """CPU tensor view over the (slot, rank, array) region, sized to the
        runtime block count derived from the per-group ``blocks`` token."""
        spec = self.spec
        num_blocks = spec.array_num_blocks(array_idx, blocks)
        offset = self._array_offset(slot_idx, rank, array_idx)
        count = num_blocks * spec._inner_numel(array_idx)
        flat = torch.frombuffer(self._shm.buf,
                                dtype=spec.array_dtype[array_idx],
                                count=count,
                                offset=offset)
        view_shape = (num_blocks, ) + tuple(spec.array_inner_shape[array_idx])
        return flat.view(view_shape)

    def rank_layer_views(self, slot_idx: int, rank: int,
                         blocks) -> list[memoryview]:
        """Return memoryview per array over the 'used' slice of shm."""
        spec = self.spec
        base = memoryview(self._shm.buf)
        out: list[memoryview] = []
        for a in range(spec.num_arrays):
            off = self._array_offset(slot_idx, rank, a)
            used = spec.array_used_bytes(a, blocks)
            out.append(base[off:off + used])
        return out

    def unpack_rank_layers(self, slot_idx: int, rank: int, blocks,
                           layer_buffers: list) -> None:
        """memcpy each received array buffer into its shm region."""
        spec = self.spec
        if len(layer_buffers) != spec.num_arrays:
            raise RuntimeError(
                f"unpack_rank_layers: got {len(layer_buffers)} arrays, "
                f"expected {spec.num_arrays}")
        for a, buf in enumerate(layer_buffers):
            used = spec.array_used_bytes(a, blocks)
            src = memoryview(buf)
            if src.nbytes != used:
                raise RuntimeError(
                    f"unpack_rank_layers: array {a} size {src.nbytes} != "
                    f"expected {used} (blocks={blocks})")
            off = self._array_offset(slot_idx, rank, a)
            self._copy_into_shm(off, src, used)

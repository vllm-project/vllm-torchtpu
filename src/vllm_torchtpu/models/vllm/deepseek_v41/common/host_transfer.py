# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental raw DMA for packed Engram rows and host hash downloads.

Upload lifecycle adapted from vllm-project/vllm-torchtpu PR #2256,
common/host_transfer.py. Downloads rebind each producer's PJRT buffer.
"""

import ctypes
import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import torch
from torch_tpu._internal import sync
from torch_tpu._internal.compile import tpu_torch_compile
from tpu_sync.api.torch import raw_transfer


def _physical_words(tensor: torch.Tensor) -> int:
    layout = tpu_torch_compile.get_device_layout_if_materialized(tensor)
    if layout is None:
        raise RuntimeError("Engram raw DMA requires a materialized int32 buffer")
    order, tiles, element_bits = layout
    if list(order) != [0] or element_bits not in (0, 32):
        raise NotImplementedError(f"Unsupported Engram raw int32 layout: {layout}")
    words = tensor.numel()
    for tile in tiles:
        if len(tile) != 1 or tile[0] <= 0:
            raise NotImplementedError(f"Unsupported Engram raw 1-D tile: {layout}")
        words = -(-words // tile[0]) * tile[0]
    return words


def _host_words(buffer: Any, num_words: int) -> torch.Tensor:
    owner = (ctypes.c_uint8 * buffer.size_bytes).from_address(buffer.data_ptr)
    return torch.frombuffer(owner, dtype=torch.uint8)[: num_words * 4].view(torch.int32)


def _hash_layout(
    shape: tuple[int, ...], layout: Any
) -> tuple[list[int], list[int], list[int]]:
    order, tiles, element_bits = layout
    rank = len(shape)
    if sorted(order) != list(range(rank)) or element_bits not in (0, 32):
        raise NotImplementedError(f"Unsupported Engram hash layout: {layout}")
    if len(tiles) > 1:
        raise NotImplementedError(f"Unsupported Engram hash tiles: {layout}")
    tile = list(tiles[0]) if tiles else []
    if len(tile) > rank or any(size <= 0 for size in tile):
        raise NotImplementedError(f"Unsupported Engram hash tile: {layout}")
    major = list(reversed(order))
    tiled_axes = list(reversed(order[: len(tile)]))
    sizes = dict(zip(tiled_axes, tile))
    outer = [-(-shape[dim] // sizes.get(dim, 1)) for dim in major]
    permutation = []
    for dim in range(rank):
        permutation.append(major.index(dim))
        if dim in sizes:
            permutation.append(rank + tiled_axes.index(dim))
    padded = [outer[major.index(dim)] * sizes.get(dim, 1) for dim in range(rank)]
    return outer + tile, permutation, padded


class PreparedEngramUpload:
    """One immutable device allocation, its host storage and consumer lifetime."""

    def __init__(self, num_words: int, device: torch.device) -> None:
        probe = torch.arange(num_words, dtype=torch.int32, device="cpu")
        probe.bitwise_xor_(0x13579BDF)
        self.device = probe.to(device)
        self.host_buffer = raw_transfer.RawHostBuffer(_physical_words(self.device) * 4)
        self.transfer = raw_transfer.PreparedTorchRawTransfer(
            self.device, self.host_buffer, unsafe_skip_buffer_lock=False
        )
        self.host_words = _host_words(self.host_buffer, num_words)
        self.transfer.d2h()
        if not torch.equal(self.host_words, probe):
            raise RuntimeError("Engram raw layout validation failed")
        self.pending: Any = None
        self.consumers: list[torch.Tensor] = []
        self.needs_consumer = False

    def acquire(self) -> torch.Tensor:
        self.wait()
        return self.host_words

    def release_consumers(self) -> None:
        if self.consumers:
            with torch.profiler.record_function("engram::slot_consumer_wait"):
                sync.synchronize(tensors=self.consumers, wait=True)
            self.consumers = []

    def start(self) -> None:
        if self.needs_consumer:
            raise RuntimeError("Record the Engram consumer before overwriting its slot")
        self.release_consumers()
        with torch.profiler.record_function("engram::h2d_submit"):
            self.pending = self.transfer.h2d_async()

    def wait(self) -> None:
        if self.pending is not None:
            with torch.profiler.record_function("engram::h2d_wait"):
                self.pending.wait()
            self.pending = None

    def activate(self) -> None:
        self.wait()
        self.needs_consumer = True

    def record_consumers(self, tensors: list[torch.Tensor]) -> None:
        if self.needs_consumer:
            if not tensors:
                raise ValueError("Engram slot reuse requires a consumer output")
            self.consumers = tensors
            self.needs_consumer = False

    def copy_from(self, words: torch.Tensor) -> None:
        if words.shape != self.host_words.shape:
            raise ValueError("Engram upload must match the slot's token bucket")
        self.acquire().copy_(words)
        self.start()
        self.wait()


@dataclass
class _DownloadBuffer:
    buffer: Any
    physical_words: int
    layout: Any

    @cached_property
    def words(self) -> torch.Tensor:
        return _host_words(self.buffer, self.physical_words)


class EngramHashDownload:
    """Reuse host storage, rebinding the changing device producer on every call.

    The returned CPU view is valid until the next download of the same bucket.
    Model forwards and their host gathers must be submitted serially.
    """

    def __init__(self) -> None:
        self._buffers: dict[tuple[torch.device, tuple[int, ...]], _DownloadBuffer] = {}

    def copy_to_host(self, hashes: torch.Tensor) -> torch.Tensor:
        if hashes.device.type == "cpu":
            return hashes
        if hashes.numel() == 0:
            return torch.empty(hashes.shape, dtype=hashes.dtype, device="cpu")
        if hashes.dtype != torch.int32:
            raise TypeError("Engram raw hash download requires int32 hashes")
        sync.synchronize(tensors=[hashes], wait=True)
        layout = tpu_torch_compile.get_device_layout_if_materialized(hashes)
        if layout is None:
            raise RuntimeError(
                "Engram raw hash download requires a materialized buffer"
            )
        shape = tuple(hashes.shape)
        physical_shape, permutation, padded = _hash_layout(shape, layout)
        physical_words = math.prod(physical_shape)
        key = (hashes.device, shape)
        storage = self._buffers.get(key)
        if storage is None or storage.layout != layout:
            buffer = raw_transfer.RawHostBuffer(physical_words * 4)
            storage = _DownloadBuffer(buffer, physical_words, layout)
            self._buffers[key] = storage
        # Rebind each step: compiled hashes may use a new device allocation.
        # Binding allocates RawHostBuffer before storage.words creates its view.
        transfer = raw_transfer.PreparedTorchRawTransfer(
            hashes, storage.buffer, unsafe_skip_buffer_lock=False
        )
        with torch.profiler.record_function("engram::hash_d2h"):
            transfer.d2h()
        return (
            storage.words.view(physical_shape)
            .permute(permutation)
            .reshape(padded)[tuple(slice(0, size) for size in shape)]
        )

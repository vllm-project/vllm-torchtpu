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
"""Multi-threaded host copies of large multimodal tensors on their way to the workers.

The pixels of a multimodal request are copied twice on the host after the API
server hands them over: EngineCore writes each large item into the shared-memory
tensor arena (`ShmTensorArena.write_tensor`), and every TP worker concatenates a
batch's items into one tensor before its upload (`MultiModalFlatField.
_reduce_data`). Both are one-thread `copy_`s into 4 KiB pages, which run at about
5 GB/s: 6.0 ms to write a 32 MB item, 3-5 ms for each worker to concatenate a
36 MB batch. Here both run as 4 MiB chunks on a small thread pool: 1.3 and 2.3 ms.
The pool is plain Python threads, not torch's OpenMP pool, whose threads would
spin after every parallel region in the engine and worker loops (the reason
vLLM keeps those processes at one torch thread). The copies move bytes, so the
tensors are unchanged.

The arena's creator also touches every slot once at creation (`_init_arena`).
A slot's tmpfs pages are allocated by their first write, and the arena hands
out its 8 slots in turn, so each of a fresh server's first 8 large items paid
that: 12.5 ms to write a 32 MB item into a new slot against 1.3-1.5 into a
used one. The slots keep upstream's order, so a slot freed by its readers is
written again only after the 7 others: TPU slots are not pinned, and the arena
gates a release on an upload's completion only for pinned ones.

`write_tensor` is a drop-in for its namesake in `vllm.distributed.
device_communicators.shm_tensor_arena` of vLLM 0.33.0 (the tests pin the
sources), written to move there unchanged; this module goes once the pooled
copies and the slot touch are upstream.
"""

from __future__ import annotations

import concurrent.futures
import functools
import os
from collections.abc import Sequence

import torch
from vllm.distributed.device_communicators import shm_tensor_arena
from vllm.multimodal import inputs

# The upstream code this module reproduces or wraps; the tests pin their sources.
UPSTREAM_ARENA_INIT = shm_tensor_arena.ShmTensorArena.__init__
UPSTREAM_WRITE_TENSOR = shm_tensor_arena.ShmTensorArena.write_tensor
UPSTREAM_FLAT_REDUCE_DATA = inputs.MultiModalFlatField._reduce_data

# Knee of a 4/8/16-thread sweep with 8 concurrent readers: the 32 MB arena write
# takes 1.81, 1.28 and 1.27 ms.
COPY_THREADS = 8
CHUNK_BYTES = 4 << 20
# Smaller copies stay on the caller's thread: the arena takes items from 8 MB.
MIN_PARALLEL_BYTES = 8 << 20


def host_threads() -> int:
    """The host pool's threads: the knee, within this process's CPUs."""
    return min(COPY_THREADS, len(os.sched_getaffinity(0)))


@functools.cache
def host_pool() -> concurrent.futures.ThreadPoolExecutor:
    """The process's thread pool for host-side byte work on multimodal data."""
    return concurrent.futures.ThreadPoolExecutor(host_threads(), "mm_host_copy")


# A forked child (EngineCore, TP workers) has none of its parent's pool threads.
os.register_at_fork(after_in_child=host_pool.cache_clear)


def copy_bytes_(pairs: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> None:
    """`dst.copy_(src)` for each pair of equal-length 1-D uint8 CPU tensors.

    Large totals are split into `CHUNK_BYTES` pieces on the copy pool; `copy_`
    releases the GIL, so the pieces run in parallel.
    """
    if sum(src.numel() for _, src in pairs) < MIN_PARALLEL_BYTES:
        for dst, src in pairs:
            dst.copy_(src)
        return
    futures = [
        host_pool().submit(dst[i : i + CHUNK_BYTES].copy_, src[i : i + CHUNK_BYTES])
        for dst, src in pairs
        for i in range(0, src.numel(), CHUNK_BYTES)
    ]
    for future in futures:
        future.result()


def _init_arena(self: shm_tensor_arena.ShmTensorArena, *args, **kwargs) -> None:
    """Upstream's constructor; the creator then writes every slot once, which
    allocates their tmpfs pages before a request needs them."""
    UPSTREAM_ARENA_INIT(self, *args, **kwargs)
    if self.is_creator:

        def touch(idx: int) -> None:
            with self._slot(idx, self.slot_bytes) as slot_mv:
                slot = torch.frombuffer(slot_mv, dtype=torch.uint8)
                slot.fill_(0)
                del slot

        list(host_pool().map(touch, range(self.n_slots)))


def write_tensor(self: shm_tensor_arena.ShmTensorArena, t: torch.Tensor) -> int | None:
    """Copy tensor bytes into a free slot; return slot idx or None
    (caller must then fall back to the default pickle path)."""
    nbytes = t.numel() * t.element_size()
    if nbytes > self.slot_bytes:
        return None
    try:
        # Computed before claiming a slot: exotic tensors (e.g. conjugate
        # bit set) raise RuntimeError here, same as `_reduce_tensor`'s
        # identical view -- fail before claiming, not after.
        src = t.detach().reshape(-1).view(torch.uint8)
    except RuntimeError:
        return None
    shm_tensor_arena.memory_fence()
    for probe in range(self.n_slots):
        idx = (self._next_slot + probe) % self.n_slots
        with self._meta(idx) as meta:
            free = meta[0] == 0 or sum(meta[1:]) == self.n_reader
            if not free:
                continue
            meta[0] = 0  # claim
        slot_mv = self._slot(idx, nbytes)
        try:
            dst = torch.frombuffer(slot_mv, dtype=torch.uint8, count=nbytes)
            copy_bytes_([(dst, src)])
        finally:
            slot_mv.release()
        with self._meta(idx) as meta:
            for i in range(1, self.n_reader + 1):
                meta[i] = 0
            shm_tensor_arena.memory_fence()
            meta[0] = 1
            shm_tensor_arena.memory_fence()
        self._next_slot = (idx + 1) % self.n_slots
        return idx
    self._fallbacks += 1
    if self._fallbacks == 1 or self._fallbacks % 100 == 0:
        shm_tensor_arena.logger.info(
            "ShmTensorArena: no free slot (%d bytes, %d fallbacks so far); "
            "falling back to the out-of-band pickle path.",
            nbytes,
            self._fallbacks,
        )
    return None


def _flat_reduce_data(
    self: inputs.MultiModalFlatField,
    batch: list[inputs.NestedTensors],
    *,
    pin_memory: bool,
) -> inputs.NestedTensors:
    """Upstream's concatenation, with the bytes of contiguous CPU items on the
    copy pool when they are concatenated on their leading dimension."""
    tensors = [x for x in batch if isinstance(x, torch.Tensor)]
    if (
        len(batch) > 1
        and self.dim == 0
        and len(tensors) == len(batch)
        and all(
            x.ndim > 0
            and x.is_cpu
            and x.is_contiguous()
            and x.dtype == tensors[0].dtype
            and x.shape[1:] == tensors[0].shape[1:]
            for x in tensors
        )
    ):
        out = tensors[0].new_empty(
            (sum(x.shape[0] for x in tensors), *tensors[0].shape[1:]),
            pin_memory=pin_memory,
        )
        flat_out = out.view(-1).view(torch.uint8)
        pairs, offset = [], 0
        for x in tensors:
            flat = x.view(-1).view(torch.uint8)
            pairs.append((flat_out[offset : offset + flat.numel()], flat))
            offset += flat.numel()
        copy_bytes_(pairs)
        return out
    return UPSTREAM_FLAT_REDUCE_DATA(self, batch, pin_memory=pin_memory)


def patch_mm_host_copies() -> None:
    """Run the arena write and the workers' batch concatenation on the copy
    pool, with the arena's slots touched at creation."""
    shm_tensor_arena.ShmTensorArena.__init__ = _init_arena
    shm_tensor_arena.ShmTensorArena.write_tensor = write_tensor
    inputs.MultiModalFlatField._reduce_data = _flat_reduce_data

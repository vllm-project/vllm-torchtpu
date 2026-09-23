# SPDX-License-Identifier: Apache-2.0
"""Shared-memory KV staging pool for the TPUConnector TP>1 path.

Layout (single SharedMemory block per host):

    slot 0: [rank 0 layers back-to-back] [rank 1 layers ...] ... [rank N-1 ...]
    slot 1: ...
    ...

Each (slot, rank, layer) is a contiguous region of max_per_layer_bytes; the
actual number of valid blocks per request varies but is always <= max_blocks.
The wire-format header carries the actual block count so only the meaningful
bytes are transferred over the network; padding stays resident.

Free-list is held on rank-0 only. Other ranks compute offsets from
(slot_idx, tp_rank, layer_idx) and never touch the free-list state.
"""

import ctypes
import os
import queue
import threading
import time
from dataclasses import dataclass
from multiprocessing import shared_memory

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is present in normal installs.
    np = None
import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

try:
    _memmove = ctypes.CDLL(None).memmove
    _memmove.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
    _memmove.restype = ctypes.c_void_p
except (AttributeError, OSError):  # pragma: no cover - libc is present on TPU.
    _memmove = None

# mlock(2) / munlock(2) bindings. Used to keep the shm pool resident in RAM
# so PJRT's H2D path does not re-fault pages mid-DMA. mlock alone does not
# tell PJRT the pages are pinned for DMA, but in practice keeping them
# resident is the cheapest first step toward "transfer_h2d_batch from shm
# at line rate" parity with the D2H direction.
try:
    _libc_for_mlock = ctypes.CDLL("libc.so.6", use_errno=True)
    _libc_for_mlock.mlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    _libc_for_mlock.mlock.restype = ctypes.c_int
    _libc_for_mlock.munlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    _libc_for_mlock.munlock.restype = ctypes.c_int
except OSError:  # pragma: no cover - libc is present on TPU.
    _libc_for_mlock = None


def _shm_base_addr(shm: shared_memory.SharedMemory) -> int:
    """Return the raw mmap base address of a SharedMemory block.

    `shm.buf` is a memoryview over the mmapped region; `c_char.from_buffer`
    on it produces a ctypes object aliasing the same memory, whose address
    is the mmap base."""
    return ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))


def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


@dataclass(frozen=True)
class PoolSpec:
    """Describes the per-host pool. Must match byte-for-byte across ranks."""

    num_slots: int
    tp_size: int
    num_layers: int
    max_blocks: int
    # Shape of a single layer's shard for one rank at max_blocks.
    # E.g. [max_blocks, block_size, kv_heads_per_rank, 2, head_dim].
    layer_shard_shape: tuple
    dtype: torch.dtype

    @property
    def per_layer_bytes(self) -> int:
        n = 1
        for d in self.layer_shard_shape:
            n *= d
        return n * _dtype_bytes(self.dtype)

    @property
    def per_rank_bytes(self) -> int:
        return self.num_layers * self.per_layer_bytes

    @property
    def per_slot_bytes(self) -> int:
        return self.tp_size * self.per_rank_bytes

    @property
    def total_bytes(self) -> int:
        return self.num_slots * self.per_slot_bytes


class HostKVShmPool:
    """Per-host shared-memory pool. Rank 0 `create()`s, others `attach()`."""

    def __init__(self, spec: PoolSpec, shm: shared_memory.SharedMemory, owner: bool):
        self.spec = spec
        self._shm = shm
        self._owner = owner
        # Regions locked by mlock_async(); used by close() for clean
        # munlock without re-deriving the (possibly already-closed) shm
        # base address. Each entry is (addr, size).
        self._mlocked_regions: list[tuple[int, int]] = []
        self._mlock_thread: threading.Thread | None = None
        self._mlock_done = threading.Event()
        # Only rank 0 uses the free list.
        self._free: queue.Queue | None = None
        self._lock: threading.Lock | None = None
        if owner:
            self._free = queue.Queue(maxsize=spec.num_slots)
            for i in range(spec.num_slots):
                self._free.put(i)
            self._lock = threading.Lock()

    # ---- mlock / munlock ----------------------------------------------
    def mlock_async(self, my_rank: int | None = None) -> threading.Thread:
        """Run mlock(2) on a daemon thread so init isn't blocked.

        With pool=128 GB and ~1 µs/page, locking on the calling thread
        takes 30+ seconds and trips vLLM's engine-core <-> worker
        heartbeat. Async + per-rank slicing avoids both.

        If `my_rank` is given, locks only this rank's slice across all
        slots (per_rank_bytes * num_slots); since each rank only reads
        its own shard for H2D, locking the other 7/8 is wasted work. If
        `my_rank` is None, locks the whole pool (use only when the caller
        truly reads across all ranks' regions).
        """
        if self._mlock_thread is not None and self._mlock_thread.is_alive():
            return self._mlock_thread
        target = self._mlock_blocking
        name = f"shm-mlock-r{my_rank}" if my_rank is not None else "shm-mlock-all"
        self._mlock_done.clear()
        t = threading.Thread(target=target, args=(my_rank,), name=name, daemon=True)
        t.start()
        self._mlock_thread = t
        return t

    def wait_mlock(self, timeout: float | None = None) -> bool:
        """Block until the async mlock completes (or until timeout). Mainly
        useful for tests / benchmarks that need pages pinned before the
        first request hits."""
        return self._mlock_done.wait(timeout=timeout)

    def _mlock_blocking(self, my_rank: int | None) -> None:
        if _libc_for_mlock is None:
            logger.warning("HostKVShmPool: libc.mlock unavailable")
            self._mlock_done.set()
            return
        try:
            base = _shm_base_addr(self._shm)
        except Exception as e:
            logger.warning("HostKVShmPool: cannot derive shm base addr: %s", e)
            self._mlock_done.set()
            return
        spec = self.spec
        regions: list[tuple[int, int]] = []
        if my_rank is None:
            regions.append((base, self._shm.size))
        else:
            # Per-rank slice: one (per_rank_bytes) region per slot, since
            # ranks are interleaved within a slot. Doing num_slots small
            # mlocks costs the same as one big mlock (kernel paginates
            # internally), so we don't try to coalesce.
            rank_off_in_slot = my_rank * spec.per_rank_bytes
            for s in range(spec.num_slots):
                addr = base + s * spec.per_slot_bytes + rank_off_in_slot
                regions.append((addr, spec.per_rank_bytes))
        t0 = time.perf_counter()
        locked_bytes = 0
        for addr, size in regions:
            rc = _libc_for_mlock.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(size))
            if rc != 0:
                err = ctypes.get_errno()
                logger.warning(
                    "HostKVShmPool: mlock(addr=0x%x size=%d) failed "
                    "errno=%d (%s); raise RLIMIT_MEMLOCK or grant "
                    "CAP_IPC_LOCK to use",
                    addr,
                    size,
                    err,
                    os.strerror(err),
                )
                # Don't bail: keep the regions we did manage to lock so
                # munlock() balances. Just stop trying further regions.
                break
            self._mlocked_regions.append((addr, size))
            locked_bytes += size
        elapsed = time.perf_counter() - t0
        logger.info(
            "HostKVShmPool: mlock done | rank=%s | locked=%.2fGB across "
            "%d regions | elapsed=%.2fs",
            "all" if my_rank is None else str(my_rank),
            locked_bytes / (1024**3),
            len(self._mlocked_regions),
            elapsed,
        )
        self._mlock_done.set()

    def _munlock_pool(self) -> None:
        if _libc_for_mlock is None or not self._mlocked_regions:
            return
        for addr, size in self._mlocked_regions:
            _libc_for_mlock.munlock(ctypes.c_void_p(addr), ctypes.c_size_t(size))
        self._mlocked_regions.clear()

    @classmethod
    def create(cls, spec: PoolSpec, name: str) -> "HostKVShmPool":
        size = spec.total_bytes
        # Unlink any stale block from a previous crashed run. We only do this
        # on create (rank 0); other ranks must never unlink.
        try:
            stale = shared_memory.SharedMemory(name=name, create=False)
            stale.close()
            stale.unlink()
            logger.warning("HostKVShmPool --> unlinked stale block %s", name)
        except FileNotFoundError:
            pass
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        logger.info(
            "%s --> created name=%s size=%.2fGB "
            "(num_slots=%d, tp_size=%d, per_slot=%.2fMB)",
            cls.__name__,
            name,
            size / (1024**3),
            spec.num_slots,
            spec.tp_size,
            spec.per_slot_bytes / (1024**2),
        )
        return cls(spec, shm, owner=True)

    @classmethod
    def attach(cls, spec: PoolSpec, name: str) -> "HostKVShmPool":
        shm = shared_memory.SharedMemory(name=name, create=False)
        if shm.size < spec.total_bytes:
            raise RuntimeError(
                f"{cls.__name__} attach: shm size {shm.size} < expected "
                f"{spec.total_bytes}. Spec mismatch between ranks?"
            )
        logger.info(
            "%s --> attached name=%s size=%.2fGB",
            cls.__name__,
            name,
            shm.size / (1024**3),
        )
        return cls(spec, shm, owner=False)

    # ---- Free-list (rank 0 only) ---------------------------------------
    def acquire_slot(self, timeout: float | None = None) -> int:
        assert self._owner, "acquire_slot: only rank 0 manages the free list"
        return self._free.get(timeout=timeout)

    def release_slot(self, slot_idx: int) -> None:
        assert self._owner, "release_slot: only rank 0 manages the free list"
        # Idempotent: tolerate double-release which can happen if a timeout
        # races a pull-done notification.
        with self._lock:
            if slot_idx in self._free.queue:
                return
            self._free.put_nowait(slot_idx)

    # ---- Tensor views ---------------------------------------------------
    def layer_view(
        self, slot_idx: int, rank: int, layer_idx: int, num_blocks: int
    ) -> torch.Tensor:
        """Return a CPU tensor view over the (slot, rank, layer) region.

        Shape is (num_blocks,) + layer_shard_shape[1:], so callers can copy
        the exact request-sized slice in/out without touching the padding.
        """
        spec = self.spec
        offset = (
            slot_idx * spec.per_slot_bytes
            + rank * spec.per_rank_bytes
            + layer_idx * spec.per_layer_bytes
        )
        count = num_blocks
        for d in spec.layer_shard_shape[1:]:
            count *= d
        # frombuffer returns a flat tensor; reshape to the layer shard shape
        # with the actual block count on dim 0.
        flat = torch.frombuffer(
            self._shm.buf, dtype=spec.dtype, count=count, offset=offset
        )
        view_shape = (num_blocks,) + tuple(spec.layer_shard_shape[1:])
        return flat.view(view_shape)

    def _used_per_layer(self, num_blocks: int) -> int:
        """Bytes used by one layer's shard at the actual block count."""
        bytes_per_block = _dtype_bytes(self.spec.dtype)
        for d in self.spec.layer_shard_shape[1:]:
            bytes_per_block *= d
        return bytes_per_block * num_blocks

    def pack_rank(self, slot_idx: int, rank: int, num_blocks: int) -> bytes:
        """Copy the `num_blocks`-sized shard of each layer into a single
        contiguous bytes buffer (strips per-layer padding). Suitable as a ZMQ
        payload frame."""
        spec = self.spec
        rank_start = slot_idx * spec.per_slot_bytes + rank * spec.per_rank_bytes
        used = self._used_per_layer(num_blocks)
        out = bytearray(used * spec.num_layers)
        pos = 0
        for li in range(spec.num_layers):
            layer_off = rank_start + li * spec.per_layer_bytes
            out[pos : pos + used] = self._shm.buf[layer_off : layer_off + used]
            pos += used
        return bytes(out)

    def rank_layer_views(
        self, slot_idx: int, rank: int, num_blocks: int
    ) -> list[memoryview]:
        """Return one memoryview per layer over the (slot, rank, layer)
        'used' slice of shm. No copy -- ZeroMQ can send these directly
        with copy=False, which turns the 4-copy path
        (shm->bytearray->bytes->zmq->kernel) into a single scatter-gather
        sendmsg from shm straight to the NIC buffer. The returned views
        alias live shm and must not outlive the current slot."""
        spec = self.spec
        rank_start = slot_idx * spec.per_slot_bytes + rank * spec.per_rank_bytes
        used = self._used_per_layer(num_blocks)
        base = memoryview(self._shm.buf)
        out: list[memoryview] = []
        for li in range(spec.num_layers):
            layer_off = rank_start + li * spec.per_layer_bytes
            out.append(base[layer_off : layer_off + used])
        return out

    def unpack_rank_layers(
        self, slot_idx: int, rank: int, num_blocks: int, layer_buffers: list
    ) -> None:
        """Inverse of rank_layer_views. `layer_buffers` is a list of
        buffer-protocol objects (zmq.Frame.buffer, memoryview, bytes)
        sized per-layer `used` bytes, one per layer. Each is memcpy'd into
        its shm slot -- one copy on the consumer instead of two."""
        spec = self.spec
        if len(layer_buffers) != spec.num_layers:
            raise RuntimeError(
                f"unpack_rank_layers: got {len(layer_buffers)} layers, "
                f"expected {spec.num_layers}"
            )
        rank_start = slot_idx * spec.per_slot_bytes + rank * spec.per_rank_bytes
        used = self._used_per_layer(num_blocks)
        for li, buf in enumerate(layer_buffers):
            src = memoryview(buf)
            if src.nbytes != used:
                raise RuntimeError(
                    f"unpack_rank_layers: layer {li} size {src.nbytes} != "
                    f"expected {used} (num_blocks={num_blocks})"
                )
            layer_off = rank_start + li * spec.per_layer_bytes
            self._copy_into_shm(layer_off, src, used)

    def _copy_into_shm(self, offset: int, src: memoryview, nbytes: int) -> None:
        """Copy a contiguous source buffer into shm.

        ctypes CDLL calls release the GIL around the native call, so the large
        memcpy does not block rank-0 progress while channel workers unpack.
        Keep fallbacks for unusual environments without NumPy/libc memmove.
        """
        if np is None:
            self._shm.buf[offset : offset + nbytes] = src
            return

        dst = np.ndarray((nbytes,), dtype=np.uint8, buffer=self._shm.buf, offset=offset)
        src_arr = np.frombuffer(src, dtype=np.uint8, count=nbytes)
        if _memmove is not None:
            _memmove(dst.ctypes.data, src_arr.ctypes.data, nbytes)
        else:
            np.copyto(dst, src_arr, casting="no")

    def unpack_rank(self, slot_idx: int, rank: int, num_blocks: int, payload) -> None:
        """Inverse of pack_rank: scatter a received contiguous payload into
        the per-layer regions of this slot/rank."""
        spec = self.spec
        rank_start = slot_idx * spec.per_slot_bytes + rank * spec.per_rank_bytes
        used = self._used_per_layer(num_blocks)
        expected = used * spec.num_layers
        if len(payload) != expected:
            raise RuntimeError(
                f"unpack_rank: payload size {len(payload)} != expected "
                f"{expected} (num_blocks={num_blocks})"
            )
        mv = memoryview(payload)
        for li in range(spec.num_layers):
            layer_off = rank_start + li * spec.per_layer_bytes
            self._shm.buf[layer_off : layer_off + used] = mv[
                li * used : (li + 1) * used
            ]

    # ---- Lifecycle ------------------------------------------------------
    def close(self) -> None:
        # munlock before closing the mmap; otherwise the kernel still holds
        # the locked pages until munmap, which is a slower path.
        self._munlock_pool()
        try:
            self._shm.close()
        except Exception:
            logger.exception(
                "HostKVShmPool close: error closing shm %s", self._shm.name
            )
        if self._owner:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass

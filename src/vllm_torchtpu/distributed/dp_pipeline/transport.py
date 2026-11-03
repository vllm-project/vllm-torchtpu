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
"""The hand-offs of a pipelined prefill, through host memory.

Dense DP engines run in separate torch_tpu slices: no device collective spans
two of them (worker/tpu_worker.py), so a stage's hidden states reach the next
stage's engine, and a helper's K/V pages reach the owner, through POSIX shared
memory. Each channel of a directed pair of engines is one ring of two slots:
``HIDDEN`` carries hidden states from the sender's TP rank 0 to every TP rank
of the receiver; ``kv_channel(r)`` carries TP rank ``r``'s K/V pages to the
same rank of the owner. The writer copies a hand-off into a slot and marks it
ready; each reader copies it to its device and acknowledges it; a slot is
reused once all readers have. Slots take the hand-offs in turn, so the writer
fills one while the other is being read.

Tensor.copy_ into the slot and ``.to(device)`` out of it are the fast torch_tpu
paths (3.5 and 4.3 GB/s per core measured, unlike batch_transfer into
unlocked memory), and both run while the device computes.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from multiprocessing import resource_tracker, shared_memory

import numpy as np
import torch

_SLOTS = 2
_HEADER_BYTES = 4096
# Per slot: chunk token, rows, ready (the token, once the rows are written),
# then one acknowledgement per reader.
_TOKEN, _ROWS, _READY, _ACK = 0, 1, 2, 3
_POLL_S = 1e-4
# How long the other end of a pair may take to size the ring it created.
_ATTACH_TIMEOUT_S = 10.0


HIDDEN = "h"


def kv_channel(tp_rank: int) -> str:
    return f"kv{tp_rank}"


def ring_name(server_id: int, src_rank: int, dst_rank: int, channel: str) -> str:
    return f"tpu_dp_pipeline_{server_id}_{src_rank}_{dst_rank}_{channel}"


def unlink_rings(server_id: int, num_engines: int, tp_size: int) -> None:
    """Removes the rings of API server ``server_id``, whose engines have
    stopped (engines map them for as long as they serve)."""
    channels = [HIDDEN, *(kv_channel(r) for r in range(tp_size))]
    for src, dst in itertools.permutations(range(num_engines), 2):
        for channel in channels:
            name = ring_name(server_id, src, dst, channel)
            try:
                shm = shared_memory.SharedMemory(name=name)
            except FileNotFoundError:
                continue  # no pipeline used this channel
            shm.close()
            shm.unlink()


class EdgeRing:
    """Hand-off slots on ``channel`` from DP engine ``src_rank`` to DP engine
    ``dst_rank``, read by ``readers`` ranks.

    Rings are named per API server (``server_id``): a restarted server never
    meets a ring a previous one left half-used."""

    def __init__(
        self,
        server_id: int,
        src_rank: int,
        dst_rank: int,
        channel: str,
        max_rows: int,
        row_shape: tuple[int, ...],
        dtype: torch.dtype,
        readers: int,
        materialize: Callable[[torch.Tensor], None],
    ) -> None:
        """``materialize(t)`` returns once device tensor ``t`` holds its data."""
        self.max_rows, self.row_shape, self.dtype = max_rows, row_shape, dtype
        self.readers = readers
        self._materialize = materialize
        row_bytes = (
            int(np.prod(row_shape)) * torch.empty((), dtype=dtype).element_size()
        )
        self.slot_bytes = max_rows * row_bytes
        if (_ACK + readers) * _SLOTS * 8 > _HEADER_BYTES:
            raise ValueError(f"{readers} readers do not fit the slot header")
        size = _HEADER_BYTES + _SLOTS * self.slot_bytes
        name = ring_name(server_id, src_rank, dst_rank, channel)
        try:
            self._shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        except FileExistsError:
            self._shm = self._attach(name, size)
        # Every engine of the server maps the ring for as long as it serves,
        # and the API server unlinks it at shutdown (unlink_rings): no engine
        # may unlink it when it exits.
        resource_tracker.unregister(self._shm._name, "shared_memory")  # noqa: SLF001
        self._header = np.ndarray(
            (_SLOTS, _ACK + readers), dtype=np.int64, buffer=self._shm.buf
        )
        # The writer's hand-offs so far, and the token it last put per slot.
        self._puts = 0
        self._last = [0] * _SLOTS
        self._data = [
            torch.frombuffer(
                self._shm.buf,
                dtype=torch.uint8,
                count=self.slot_bytes,
                offset=_HEADER_BYTES + slot * self.slot_bytes,
            )
            .view(dtype)
            .view(max_rows, *row_shape)
            for slot in range(_SLOTS)
        ]

    @staticmethod
    def _attach(name: str, size: int) -> shared_memory.SharedMemory:
        """Maps a ring another process created. Both ends of a pair create
        their ring on first use, so the creator may still be sizing it: the
        name exists before its bytes do."""
        deadline = time.monotonic() + _ATTACH_TIMEOUT_S
        while True:
            try:
                shm = shared_memory.SharedMemory(name=name)
            except ValueError:  # still empty: "cannot mmap an empty file"
                if time.monotonic() > deadline:
                    raise
                time.sleep(_POLL_S)
                continue
            if shm.size < size:
                shm.close()
                raise RuntimeError(
                    f"/dev/shm/{name} holds {shm.size} bytes, this server needs "
                    f"{size}; it is left from another configuration, remove it"
                )
            return shm

    # -- sender ------------------------------------------------------------------
    def put(
        self, token: int, srcs: torch.Tensor | list[torch.Tensor], timeout_s: float
    ) -> None:
        """Copies ``srcs`` (each ``[rows_i, *row_shape]``, any device) one
        after another into the next slot and marks it ready under ``token``."""
        srcs = [srcs] if isinstance(srcs, torch.Tensor) else srcs
        rows = sum(src.shape[0] for src in srcs)
        if rows > self.max_rows or any(
            tuple(src.shape[1:]) != self.row_shape for src in srcs
        ):
            raise ValueError(
                f"hand-off of shapes {[tuple(t.shape) for t in srcs]} does not fit "
                f"the ring ({self.max_rows} rows of {self.row_shape})"
            )
        slot = self._puts % _SLOTS
        self._wait(
            lambda: slot if self._read_by_all(slot) else None,
            timeout_s,
            f"a free slot for {token}",
        )
        header = self._header[slot]
        header[_READY] = 0
        header[_TOKEN] = token
        header[_ROWS] = rows
        row = 0
        for src in srcs:
            self._data[slot][row : row + src.shape[0]].copy_(src)
            row += src.shape[0]
        header[_READY] = token
        self._last[slot] = token
        self._puts += 1

    def _read_by_all(self, slot: int) -> bool:
        """Whether every reader took what the writer last put in ``slot``."""
        last = self._last[slot]
        header = self._header[slot]
        return last == 0 or all(header[_ACK + r] == last for r in range(self.readers))

    # -- receiver ----------------------------------------------------------------
    def take(
        self,
        token: int,
        rows: int | None,
        device: torch.device,
        reader: int,
        timeout_s: float,
        split: int | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Waits for the hand-off ``token`` of ``rows`` rows (None: as sent),
        copies it to ``device`` (in pieces of ``split`` rows) and acknowledges
        it as reader ``reader``."""
        slot = self._wait(lambda: self._ready_slot(token), timeout_s, f"chunk {token}")
        header = self._header[slot]
        sent = int(header[_ROWS])
        if rows is not None and sent != rows:
            raise RuntimeError(
                f"chunk {token} arrived with {sent} rows, this stage expects {rows}"
            )
        data = self._data[slot]
        if split is None:
            out: torch.Tensor | list[torch.Tensor] = data[:sent].to(device, copy=True)
            pieces = [out]
        else:
            if sent % split:
                raise RuntimeError(
                    f"chunk {token}: {sent} rows are not pieces of {split}"
                )
            pieces = [
                data[row : row + split].to(device, copy=True)
                for row in range(0, sent, split)
            ]
            out = pieces
        # The acknowledgement frees the slot: the copies must have left it first.
        for piece in pieces:
            self._materialize(piece)
        header[_ACK + reader] = token
        return out

    def _ready_slot(self, token: int) -> int | None:
        for slot in range(_SLOTS):
            if self._header[slot][_READY] == token:
                return slot
        return None

    @staticmethod
    def _wait(probe, timeout_s: float, what: str) -> int:
        deadline = time.monotonic() + timeout_s
        while (slot := probe()) is None:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"pipelined prefill: no {what} after {timeout_s:.0f} s"
                )
            time.sleep(_POLL_S)
        return slot

    def close(self) -> None:
        self._data.clear()
        self._header = None  # type: ignore[assignment]
        self._shm.close()

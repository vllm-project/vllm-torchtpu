# SPDX-License-Identifier: Apache-2.0
"""MAX all-reduce of a few int64 over the DP group through shared memory.

The EP-DP engines all-reduce a handful of scheduling integers twice per step on
the DP group's gloo group. When every rank is a process of one host, each call
is TCP rounds and wake-ups, not data. This replaces it with one shared-memory
file: each rank writes its integers into its own row, publishes the call's
sequence number last, waits until every row shows it, and reduces locally.
"""

from __future__ import annotations

import mmap
import os
import platform
import tempfile
import time
import uuid
from typing import Any

import numpy as np
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Waiting ranks yield the CPU (and the GIL) between polls; past these waits they
# sleep instead, so a long prefill or compile wait does not burn a core.
_YIELD_S = 2e-3
_SHORT_SLEEP_UNTIL_S = 0.1
_TIMEOUT_S = 1800.0
_TOKEN_WORDS = 2


class ShmMaxAllReduce:
    """One rank's view of a [2 slots, world, width + 1] int64 shared table.

    A row holds the rank's values, then the call's sequence number. Call c uses
    slot c % 2: a rank can start call c + 2 only after every rank published
    c + 1, so every rank has finished reading call c's slot by then.
    Publishing the sequence number after the values relies on x86's store
    order; the constructor's caller only enables this on x86_64.
    """

    def __init__(self, path: str, rank: int, world: int, width: int):
        self.rank, self.world, self.width = rank, world, width
        with open(path, "r+b") as f:
            self._map = mmap.mmap(f.fileno(), table_bytes(world, width))
        words = np.frombuffer(self._map, dtype=np.int64)
        self.token = words[:_TOKEN_WORDS]
        self._table = words[_TOKEN_WORDS:].reshape(2, world, width + 1)
        self._calls = 0

    def all_reduce_max(self, values: np.ndarray) -> np.ndarray:
        self._calls += 1
        call = self._calls
        slot = self._table[call % 2]
        slot[self.rank, : self.width] = values
        slot[self.rank, self.width] = call
        sequence = slot[:, self.width]
        if (sequence != call).any():
            _wait_for(sequence, call)
        return slot[:, : self.width].max(axis=0)


def table_bytes(world: int, width: int) -> int:
    return 8 * (_TOKEN_WORDS + 2 * world * (width + 1))


def _wait_for(sequence: np.ndarray, call: int) -> None:
    start = time.perf_counter()
    while (sequence != call).any():
        waited = time.perf_counter() - start
        if waited < _YIELD_S:
            os.sched_yield()
        elif waited < _SHORT_SLEEP_UNTIL_S:
            time.sleep(50e-6)
        elif waited < _TIMEOUT_S:
            time.sleep(1e-3)
        else:
            raise TimeoutError(
                f"DP shared-memory all-reduce waited {waited:.0f} s for call {call}"
            )


def _open_shared(group: Any, width: int) -> ShmMaxAllReduce | None:
    """Build the reducer when every rank maps rank 0's file; None otherwise.

    Every rank reaches this at the same call, so the two gathers line up.
    """
    rank, world = group.rank_in_group, group.world_size
    path = token = None
    if rank == 0:
        token = np.frombuffer(uuid.uuid4().bytes, dtype=np.int64).copy()
        fd, path = tempfile.mkstemp(prefix="vllm_tpu_dp_allreduce_", dir="/dev/shm")
        os.ftruncate(fd, table_bytes(world, width))
        os.close(fd)
    shared: list[Any] = [None] * world
    torch.distributed.all_gather_object(shared, (path, token), group=group.cpu_group)
    path, token = shared[0]
    reducer = None
    try:
        reducer = ShmMaxAllReduce(path, rank, world, width)
        if rank == 0:
            reducer.token[:] = token
    except OSError as error:
        logger.info(
            "DP shared-memory all-reduce: rank %d cannot map %s: %s", rank, path, error
        )
    torch.distributed.barrier(group=group.cpu_group)
    ok = reducer is not None and bool((reducer.token == token).all())
    oks: list[Any] = [None] * world
    torch.distributed.all_gather_object(oks, ok, group=group.cpu_group)
    if rank == 0:
        os.unlink(path)  # Every rank has mapped it or given up; the maps stay.
    return reducer if all(oks) else None


_reducers: dict[str, ShmMaxAllReduce | None] = {}


def dp_all_reduce_max(t: torch.Tensor, group: Any, site: str) -> None:
    """In-place MAX all-reduce of a small CPU int64 tensor over the DP group.

    `site` names the call site: each one owns a table, since their call counts
    and widths differ. Falls back to the gloo group when the switch is off, the
    host is not x86_64, or any rank cannot map rank 0's file.
    """
    if site not in _reducers:
        enabled = envs.VLLM_TPU_SHM_DP_ALLREDUCE and platform.machine() == "x86_64"
        _reducers[site] = _open_shared(group, t.numel()) if enabled else None
        logger.info(
            "DP all-reduce %s: %s",
            site,
            "shared memory" if _reducers[site] is not None else "gloo",
        )
    reducer = _reducers[site]
    if reducer is None:
        torch.distributed.all_reduce(
            t, op=torch.distributed.ReduceOp.MAX, group=group.cpu_group
        )
        return
    values = t.numpy()
    values[:] = reducer.all_reduce_max(values)

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
"""Element-wise max of a short int64 vector across the ranks of one host.

The EP-DP lockstep runs one all-reduce of a handful of integers per engine step
(`TPUModelRunner._dp_coordinated_step`). Over vLLM's gloo CPU group that is a
ring of loopback TCP exchanges on the serving thread of every rank. When all
ranks share a host, each rank instead writes its vector into its own row of a
shared memory block and reads every other row.

Rows are double-buffered by step parity: a rank can only start step g + 2 once
every rank has written step g + 1, which each did after reading step g. A rank
publishes its step after its values and readers load the step first; that
order holds without fences only under x86's store and load ordering, so other
CPUs keep the gloo path (`usable`).
"""

from __future__ import annotations

import mmap
import os
import platform
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist

_SPIN_S = 1e-3
_SLEEP_S = 50e-6


def usable(group: dist.ProcessGroup) -> bool:
    """Whether every rank of the (CPU) group can map this host's shared memory,
    on a CPU whose memory order the protocol relies on. Collective."""
    from vllm.distributed.parallel_state import in_the_same_node_as

    same_node = all(in_the_same_node_as(group, source_rank=0))
    return same_node and platform.machine() in ("x86_64", "AMD64")


class HostMaxAllReduce:
    """`all_reduce(MAX)` of a fixed-length int64 vector over a one-host group."""

    def __init__(self, group: dist.ProcessGroup, length: int, timeout_s: float):
        self._rank = dist.get_rank(group)
        self._world = dist.get_world_size(group)
        self._length = length
        self._timeout_s = timeout_s
        # [parity, rank, step + values]; zero-filled, so step 0 is "not written".
        shape = (2, self._world, 1 + length)
        nbytes = int(np.prod(shape)) * 8
        # [path, error]: rank 0's failure reaches every rank at once.
        created: list[str | None] = [None, None]
        if self._rank == 0:
            try:
                fd, created[0] = tempfile.mkstemp(
                    prefix="vllm_tpu_dp_max_", dir="/dev/shm"
                )
                os.ftruncate(fd, nbytes)
                os.close(fd)
            except OSError as e:
                created[1] = repr(e)
        src = dist.get_global_rank(group, 0)
        dist.broadcast_object_list(created, src=src, group=group)
        path, error = created
        if error is not None:
            raise RuntimeError(
                f"DP rank 0 could not create a block in /dev/shm: {error}"
            )
        assert path is not None
        with open(path, "r+b") as f:
            self._mmap = mmap.mmap(f.fileno(), nbytes)
        dist.barrier(group=group)
        if self._rank == 0:
            os.unlink(path)  # the mappings keep the block alive
        self._rows = np.ndarray(shape, dtype=np.int64, buffer=self._mmap)
        self._step = 0

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        """The element-wise max of `values` over the group, as a new tensor."""
        assert values.dtype == torch.int64 and values.shape == (self._length,)
        self._step += 1
        step = self._step
        rows = self._rows[step % 2]
        mine = rows[self._rank]
        mine[1:] = values.numpy()
        mine[0] = step  # publish after the values
        # Early ranks wait here for the slowest one: polling, yielding the CPU
        # (and the GIL), for the usual skew between ranks' steps, then sleeping
        # between polls so a long stall elsewhere does not hold a core.
        start = time.monotonic()
        deadline = start + self._timeout_s
        for r in range(self._world):
            while rows[r, 0] != step:
                now = time.monotonic()
                if now > deadline:
                    raise TimeoutError(
                        f"DP rank {r} did not publish step {step} within "
                        f"{self._timeout_s:.0f} s"
                    )
                if now - start < _SPIN_S:
                    os.sched_yield()
                else:
                    time.sleep(_SLEEP_S)
        return torch.from_numpy(rows[:, 1:].max(axis=0))

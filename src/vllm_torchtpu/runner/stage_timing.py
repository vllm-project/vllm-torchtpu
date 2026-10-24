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
"""Runner stage timings without synchronizing the serving thread."""

from __future__ import annotations

import contextlib
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

from torch_tpu._internal.stream.streams import TpuEvent
from vllm.distributed.kv_transfer import has_kv_transfer_group

from vllm_torchtpu import envs


@dataclass
class _PendingTiming:
    stage: str
    start: TpuEvent
    end: TpuEvent
    end_recorded: bool = False


class RunnerStageTiming:
    """Record and drain stages on the runner thread without device waits."""

    MAX_IN_FLIGHT_DEVICE_TIMINGS = 1024

    def __init__(self, rank: int):
        self._rank = str(rank)
        self._cpu_samples: list[tuple[str, str, float]] = []
        # Includes active stages; saturation keeps CPU samples only.
        self._pending: deque[_PendingTiming] = deque()
        # Whether a KV connector exists to report samples through.
        self._sink = False

    def _has_sink(self) -> bool:
        """Samples only reach the metrics through a KV connector's stats
        (`TPUModelRunner.get_finished_kv_transfers`); without one they are
        discarded every step, so skip recording them -- each device timing
        costs two TpuEvent records on the serving thread."""
        if not envs.VLLM_TPU_STAGE_TIMING_ONLY_WITH_SINK:
            return True
        if not self._sink:
            # Re-checked until true: warmup can run before the connector exists.
            self._sink = has_kv_transfer_group()
        return self._sink

    @contextlib.contextmanager
    def host_stage(self, stage: str) -> Iterator[None]:
        if not self._has_sink():
            yield
            return
        cpu_start = time.perf_counter()
        yield
        self._cpu_samples.append((self._rank, stage, time.perf_counter() - cpu_start))

    @contextlib.contextmanager
    def stage(self, stage: str) -> Iterator[None]:
        if not self._has_sink():
            yield
            return
        if len(self._pending) >= self.MAX_IN_FLIGHT_DEVICE_TIMINGS:
            with self.host_stage(stage):
                yield
            return

        start = TpuEvent(enable_timing=True)
        end = TpuEvent(enable_timing=True)
        start.record()
        pending = _PendingTiming(stage, start, end)
        self._pending.append(pending)
        cpu_start = time.perf_counter()
        # Close the timing even when the stage body raises. An active outer
        # context can remain at the queue head while a nested context drains.
        try:
            yield
        finally:
            cpu_elapsed = time.perf_counter() - cpu_start
            end.record()
            pending.end_recorded = True
            self._cpu_samples.append((self._rank, stage, cpu_elapsed))

    def take_samples(
        self,
    ) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]]]:
        cpu, self._cpu_samples = self._cpu_samples, []
        device = []
        while self._pending:
            pending = self._pending[0]
            if not pending.end_recorded or not pending.end.query():
                break
            self._pending.popleft()
            # TPU events report host completion timestamps in milliseconds,
            # independent of when this runner drains the completed samples.
            seconds = pending.start.elapsed_time(pending.end) / 1000
            device.append((self._rank, pending.stage, seconds))
        return cpu, device

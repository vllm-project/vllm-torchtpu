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
"""Observe target forwards independently of the runner's stage event timing."""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from torch_tpu._internal.stream.streams import TpuEvent

from vllm_torchtpu.runner.forward_metrics import ForwardBatch, ForwardSample


@dataclass
class _PendingTiming:
    start: TpuEvent
    end: TpuEvent
    batch: ForwardBatch
    end_recorded: threading.Event = field(default_factory=threading.Event)
    submitted: float = 0.0
    forward_id: int = -1


class ForwardTiming:
    MAX_IN_FLIGHT_DEVICE_TIMINGS = 1024

    def __init__(self, rank: int):
        self._rank = str(rank)
        self._forward_samples: list[ForwardSample] = []
        self._last_host_end: float | None = None
        self._last_completion_end: float | None = None
        self._forward_id = 0
        self._last_completion_id = -1
        self._pending: queue.Queue[_PendingTiming] = queue.Queue()
        # Bound in-flight device timings; a full collector keeps CPU samples only.
        self._slots = threading.BoundedSemaphore(self.MAX_IN_FLIGHT_DEVICE_TIMINGS)
        self._samples_lock = threading.Lock()
        threading.Thread(target=self._collect_device_time, daemon=True).start()

    def _collect_device_time(self) -> None:
        while True:
            pending = self._pending.get()
            observer_start = time.perf_counter()
            pending.start.synchronize()
            start_time = time.perf_counter()
            pending.end_recorded.wait()
            pending.end.synchronize()
            end_time = time.perf_counter()
            seconds = end_time - start_time
            with self._samples_lock:
                bucket = str(pending.batch.bucket)
                values = {
                    "completion_span_seconds": seconds,
                    "observer_lag_seconds": observer_start - pending.submitted,
                }
                if (
                    self._last_completion_end is not None
                    and pending.forward_id == self._last_completion_id + 1
                ):
                    values["completion_gap_seconds"] = (
                        start_time - self._last_completion_end
                    )
                self._forward_samples.extend(
                    (self._rank, bucket, name, value) for name, value in values.items()
                )
                self._last_completion_end = end_time
                self._last_completion_id = pending.forward_id
            self._slots.release()

    @contextlib.contextmanager
    def stage(self, batch: ForwardBatch) -> Iterator[None]:
        host_start = time.perf_counter()
        self._forward_id += 1
        with self._samples_lock:
            self._forward_samples.extend(batch.samples(self._rank))
            if self._last_host_end is not None:
                self._forward_samples.append(
                    (
                        self._rank,
                        str(batch.bucket),
                        "host_gap_seconds",
                        host_start - self._last_host_end,
                    )
                )
        if not self._slots.acquire(blocking=False):
            try:
                yield
            finally:
                self._finish_forward_host(batch, host_start)
            return

        start = TpuEvent()
        end = TpuEvent()
        start.record()
        pending = _PendingTiming(
            start,
            end,
            batch=batch,
            submitted=time.perf_counter(),
            forward_id=self._forward_id,
        )
        self._pending.put_nowait(pending)
        # Close the queued timing even when the stage body raises.
        try:
            yield
        finally:
            end.record()
            pending.end_recorded.set()
            self._finish_forward_host(batch, host_start)

    def _finish_forward_host(self, batch: ForwardBatch, start: float) -> None:
        end = time.perf_counter()
        with self._samples_lock:
            self._forward_samples.append(
                (self._rank, str(batch.bucket), "host_issue_seconds", end - start)
            )
        self._last_host_end = end

    def take_forward_samples(self) -> list[ForwardSample]:
        with self._samples_lock:
            samples, self._forward_samples = self._forward_samples, []
        return samples

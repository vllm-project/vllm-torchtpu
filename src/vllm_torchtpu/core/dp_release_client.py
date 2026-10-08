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
"""The DP load-balancing client behind DP_PLACEMENT=release (dp_placement.py)."""

import asyncio
import contextlib
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from vllm.config import VllmConfig
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import DPLBAsyncMPClient, EngineIdentity
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.executor import Executor
from vllm.v1.pool.late_interaction import get_late_interaction_engine_index

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@dataclass
class _Held:
    arrival: float
    request: EngineCoreRequest
    sent: asyncio.Future


def _settle(future: asyncio.Future, exc: BaseException | None = None) -> None:
    if future.done():
        return
    if exc is None:
        future.set_result(None)
    else:
        future.set_exception(exc)


class ReleaseDPLBAsyncMPClient(DPLBAsyncMPClient):
    """DP load-balancing client that releases held requests as a group.

    A release fires when as many requests as engines are held, when an
    engine is idle, when the oldest has waited DEADLINE_S, or when none has
    arrived for QUIET_S. Each released request goes to the engine with the
    fewest assignments in this release, then the fewest in flight, ties
    cyclic; a request pinned to a DP rank keeps its engine.
    """

    DEADLINE_S = 5.0
    QUIET_S = 2.0
    POLL_S = 0.005

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> None:
        if client_count > 1:
            # TODO: hold one release group across API servers.
            raise NotImplementedError(
                "DP_PLACEMENT=release groups the arrivals of one API server, "
                f"but {client_count} API servers would each hold a fraction of "
                "them. Pass --api-server-count=1, or unset DP_PLACEMENT."
            )
        self._held: list[_Held] = []
        self._assigned: Counter[int] = Counter()
        self._next_engine = 0
        self._poller: asyncio.Task | None = None
        self._release: asyncio.Task | None = None
        super().__init__(
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )
        logger.info(
            "DP release router: group of %d, deadline %.1f s, quiet %.1f s",
            len(self.core_engines),
            self.DEADLINE_S,
            self.QUIET_S,
        )

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        self._ensure_stats_update_task()
        loop = asyncio.get_running_loop()
        if self._poller is None:
            self._poller = loop.create_task(self._poll())
        held = _Held(time.monotonic(), request, loop.create_future())
        self._held.append(held)
        self._maybe_release()
        # Returns once this request has been sent, or raises its send's error.
        await held.sent

    def get_core_engine_for_request(self, request: EngineCoreRequest) -> EngineIdentity:
        num = len(self.core_engines)
        if (
            request.data_parallel_rank is not None
            or get_late_interaction_engine_index(request.pooling_params, num)
            is not None
        ):
            return super().get_core_engine_for_request(request)
        order = [(self._next_engine + i) % num for i in range(num)]
        idx = min(
            order,
            key=lambda i: (
                self._assigned[i],
                self.engine_inflight[self.core_engines[i]],
            ),
        )
        self._assigned[idx] += 1
        self._next_engine = (idx + 1) % num
        engine = self.core_engines[idx]
        self.lb_engines[idx][0] += self.client_count
        self.reqs_in_flight[request.request_id] = engine
        self.engine_inflight[engine] += 1
        return engine

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        ids = set(request_ids)
        kept = []
        for held in self._held:
            if held.request.request_id in ids:
                held.sent.cancel()
            else:
                kept.append(held)
        # In place: a release in progress holds this same list.
        self._held[:] = kept
        await super().abort_requests_async(request_ids)

    def shutdown(self, timeout: float | None = None) -> None:
        # The engine monitor thread calls shutdown too.
        if self._poller is not None:
            with contextlib.suppress(RuntimeError):  # The loop is closed.
                self._poller.get_loop().call_soon_threadsafe(self._stop_releasing)
        super().shutdown(timeout)

    def _stop_releasing(self) -> None:
        for task in (self._poller, self._release):
            if task is not None:
                task.cancel()
        self._fail_held(EngineDeadError())

    def _fail_held(self, exc: BaseException) -> None:
        for held in self._held:
            _settle(held.sent, exc)
        self._held.clear()

    def _any_engine_idle(self) -> bool:
        return not self.engines_running or any(
            waiting == 0 and running == 0 for waiting, running, _ in self.lb_engines
        )

    def _maybe_release(self) -> None:
        if not self._held:
            return
        if self.resources.engine_dead:
            self._fail_held(EngineDeadError())
            return
        now = time.monotonic()
        if (
            len(self._held) >= len(self.core_engines)
            or now - self._held[0].arrival >= self.DEADLINE_S
            or now - self._held[-1].arrival >= self.QUIET_S
            or self._any_engine_idle()
        ) and (self._release is None or self._release.done()):
            self._release = asyncio.get_running_loop().create_task(self._release_held())
            self._release.add_done_callback(self._release_done)

    async def _poll(self) -> None:
        while True:
            await asyncio.sleep(self.POLL_S)
            try:
                self._maybe_release()
            except Exception:
                logger.exception("DP release router: release poller")

    async def _release_held(self) -> None:
        # Runs in its own task, so a cancelled add_request_async cancels no
        # other request's send.
        self._assigned.clear()
        while self._held:
            # Popped before the send: an abort during the send reaches the
            # engine through the send's registration instead.
            held = self._held.pop(0)
            if held.sent.done():
                continue
            try:
                await self._send(held.request)
            except asyncio.CancelledError:
                held.sent.cancel()
                raise
            except Exception as exc:
                logger.exception(
                    "DP release router: request %s could not be sent",
                    held.request.request_id,
                )
                _settle(held.sent, exc)
                continue
            _settle(held.sent)

    async def _send(self, request: EngineCoreRequest) -> None:
        try:
            await super().add_request_async(request)
        except BaseException:
            # The counts must not claim a request no engine holds.
            engine = self.reqs_in_flight.pop(request.request_id, None)
            if engine is not None:
                self.engine_inflight[engine] -= 1
                idx = self.core_engines.index(engine)
                self.lb_engines[idx][0] -= self.client_count
            raise

    @staticmethod
    def _release_done(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                "DP release router: release task failed", exc_info=task.exception()
            )

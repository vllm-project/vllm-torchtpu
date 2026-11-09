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
"""The data-parallel router's release placement (``DP_PLACEMENT=release``).

vLLM's API-server router (``DPLBAsyncMPClient``) places each request on the
least-loaded engine as it arrives, so arrivals follow completions and the
engines' prefill steps drift apart. With ``DP_PLACEMENT=release`` the router
holds new requests and releases them as a group, one per engine where it can,
so the engines run their prefill steps together. Streaming-input sessions and
late-interaction pooling requests keep vLLM's own engine affinity and are
never held. Off by default; takes effect in the API-server process only.
"""

import asyncio
import time
from collections import Counter

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# How often the release triggers are re-checked while requests are held.
_RELEASE_POLL_INTERVAL_S = 0.005


def install_release(
    client_cls,
    threshold: int | None = None,
    deadline_s: float | None = None,
    idle_trigger: bool | None = None,
    quiet_s: float | None = None,
) -> None:
    """Installs the release placement on vLLM's DP load-balancing client class.

    Arrivals are held in the router (on no engine yet) and released together
    when ``threshold`` are held (the engine count when 0), when any engine is
    idle (by the coordinator's counts, or no wave running), when the oldest
    has waited ``deadline_s``, or when no request has arrived for ``quiet_s``
    (0 disables). At release each request goes to the engine with the fewest
    assignments in this release, then the fewest in flight, ties cyclic, so a
    release of one per engine lands one per engine and an idle engine is fed
    first; a request pinned to a data-parallel rank keeps its engine. The
    sends run in the client's own release task; a request's
    ``add_request_async`` returns once its request has been sent and raises
    its own send's error.

    Requests that vLLM routes by affinity go through the original
    ``add_request_async`` unheld: streaming-input sessions (resumable
    requests, open sessions, and requests already in flight) and
    late-interaction pooling requests.
    """
    import msgspec
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.exceptions import EngineDeadError
    from vllm.v1.pool.late_interaction import get_late_interaction_engine_index

    if getattr(client_cls, "_tpu_dp_release_patch", False):
        return
    if threshold is None:
        threshold = envs.DP_SCHED_BUFFER_THRESHOLD
    if deadline_s is None:
        deadline_s = envs.DP_SCHED_BUFFER_FLUSH_MS / 1000.0
    if idle_trigger is None:
        idle_trigger = envs.DP_SCHED_BUFFER_IDLE_TRIGGER
    if quiet_s is None:
        quiet_s = envs.DP_SCHED_BUFFER_QUIET_MS / 1000.0
    original_add_request = client_cls.add_request_async
    original_abort_requests = client_cls.abort_requests_async
    original_shutdown = client_cls.shutdown

    def _held_requests(self) -> list:
        """(arrival time, request, waiter) for each held request, oldest first."""
        return self.__dict__.setdefault("_dp_release_held", [])

    def _release_size(self) -> int:
        return threshold or len(self.core_engines)

    def _keeps_vllm_routing(self, request) -> bool:
        return (
            request.resumable
            # vLLM releases before v0.33 have no streaming_reqs.
            or request.request_id in getattr(self, "streaming_reqs", ())
            or request.request_id in self.reqs_in_flight
            or get_late_interaction_engine_index(
                request.pooling_params, len(self.core_engines)
            )
            is not None
        )

    def _any_engine_idle(self) -> bool:
        if not self.engines_running:
            return True
        return any(
            int(waiting) == 0 and int(running) == 0
            for waiting, running, _ in self.lb_engines
        )

    def _settle_waiter(waiter, exc=None) -> None:
        # A held request's outcome goes to its own add_request_async, which
        # awaits this future; a waiter that has left (cancelled) is skipped.
        if waiter.done():
            return
        if exc is None:
            waiter.set_result(None)
        else:
            waiter.set_exception(exc)

    def _fail_held_requests(self, exc) -> None:
        held = _held_requests(self)
        for _, _, waiter in held:
            _settle_waiter(waiter, exc)
        held.clear()

    async def _send_held_request(self, request, engine_index, load_balanced):
        engine = self.core_engines[engine_index]
        request.current_wave = self.current_wave
        request.client_index = self.client_index
        engine_counts = self.lb_engines
        self.reqs_in_flight[request.request_id] = engine
        self.engine_inflight[engine] += 1
        if load_balanced:
            # As vLLM's router does: counts until the coordinator's next stats.
            engine_counts[engine_index][0] += self.client_count
        try:
            pending_send = self._send_input(EngineCoreRequestType.ADD, request, engine)
            # vLLM releases before v0.33 wake idle engines from the client by
            # telling the coordinator about the first request; later releases
            # wake them from the receiving engine and have no such socket.
            first_request_socket = getattr(self, "first_req_send_socket", None)
            if first_request_socket is not None and not self.engines_running:
                await first_request_socket.send(
                    msgspec.msgpack.encode(("FIRST_REQ", engine))
                )
            await pending_send
        except BaseException:
            # The counts must not claim a request no engine holds.
            self.reqs_in_flight.pop(request.request_id, None)
            self.engine_inflight[engine] -= 1
            if load_balanced:
                engine_counts[engine_index][0] -= self.client_count
            raise
        self._ensure_output_queue_task()

    async def _release_held_requests(self):
        """Sends every held request. Runs in the client's own release task,
        never in a request's, so a cancelled add_request_async (a client that
        left) cancels no other request's send, and each send's outcome
        reaches the request it belongs to."""
        held = _held_requests(self)
        num_engines = len(self.core_engines)
        assigned = Counter()
        next_engine = self.__dict__.get("_dp_release_next_engine", 0) % num_engines
        while held:
            # The request leaves the held list before its send: an abort that
            # lands during the send finds it gone here and, through the send's
            # reqs_in_flight entry, reaches the engine instead.
            _, request, waiter = held.pop(0)
            if waiter.done():
                continue  # Its add was cancelled before the release reached it.
            engine_index = request.data_parallel_rank
            load_balanced = engine_index is None
            if load_balanced:
                engine_index = min(
                    ((next_engine + i) % num_engines for i in range(num_engines)),
                    key=lambda i: (
                        assigned[i],
                        self.engine_inflight[self.core_engines[i]],
                    ),
                )
                assigned[engine_index] += 1
                next_engine = (engine_index + 1) % num_engines
            try:
                await _send_held_request(self, request, engine_index, load_balanced)
            except asyncio.CancelledError:
                waiter.cancel()  # The release task itself is being cancelled.
                raise
            except Exception as exc:
                # This request's add raises; the rest go out.
                logger.exception(
                    "DP placement: request %s could not be sent to engine index %s",
                    request.request_id,
                    engine_index,
                )
                _settle_waiter(waiter, exc)
                continue
            _settle_waiter(waiter)
        self._dp_release_next_engine = next_engine

    def _log_release_failure(release_task) -> None:
        if not release_task.cancelled() and release_task.exception() is not None:
            logger.error(
                "DP placement: release task failed", exc_info=release_task.exception()
            )

    def _start_release(self) -> None:
        release_task = self.__dict__.get("_dp_release_task")
        if release_task is None or release_task.done():
            release_task = asyncio.get_running_loop().create_task(
                _release_held_requests(self)
            )
            release_task.add_done_callback(_log_release_failure)
            self._dp_release_task = release_task

    def _maybe_release(self) -> None:
        held = _held_requests(self)
        if not held:
            return
        if self.resources.engine_dead:
            _fail_held_requests(self, EngineDeadError())
            return
        now = time.monotonic()
        if (
            len(held) >= _release_size(self)
            or now - held[0][0] >= deadline_s
            or (idle_trigger and _any_engine_idle(self))
            or (quiet_s > 0 and now - held[-1][0] >= quiet_s)
        ):
            _start_release(self)

    async def _poll_release_triggers(self):
        while True:
            await asyncio.sleep(_RELEASE_POLL_INTERVAL_S)
            try:
                _maybe_release(self)
            except Exception:  # The poller stays alive; the failure is logged.
                logger.exception("DP placement: release poller")

    async def add_request_async(self, request):
        if _keeps_vllm_routing(self, request):
            await original_add_request(self, request)
            return
        self._ensure_stats_update_task()
        loop = asyncio.get_running_loop()
        if self.__dict__.get("_dp_release_poller") is None:
            self._dp_release_poller = loop.create_task(_poll_release_triggers(self))
        waiter = loop.create_future()
        _held_requests(self).append((time.monotonic(), request, waiter))
        _maybe_release(self)
        # Returns once this request has been sent, or raises its own send's
        # error; the sends run in the release task.
        await waiter

    async def abort_requests_async(self, request_ids):
        aborted_ids = set(request_ids)
        held = _held_requests(self)
        if held:
            # Filtered in place: a release in progress holds this same list.
            kept = []
            for entry in held:
                if entry[1].request_id in aborted_ids:
                    entry[2].cancel()  # Its add, if still waiting, returns.
                else:
                    kept.append(entry)
            held[:] = kept
        await original_abort_requests(self, request_ids)

    def shutdown(self, *args, **kwargs):
        try:
            for name in ("_dp_release_poller", "_dp_release_task"):
                task = self.__dict__.get(name)
                if task is not None:
                    task.cancel()
            _fail_held_requests(self, EngineDeadError())
        except RuntimeError:  # The event loop is closed: nobody is waiting.
            pass
        return original_shutdown(self, *args, **kwargs)

    client_cls.add_request_async = add_request_async
    client_cls.abort_requests_async = abort_requests_async
    client_cls.shutdown = shutdown
    client_cls._tpu_dp_release_patch = True
    logger.info(
        "DP placement: held in the router and released across the engines "
        "(DP_PLACEMENT=release; threshold %s, deadline %.1f s, idle trigger %s, "
        "quiet release %s)",
        threshold or "engine count",
        deadline_s,
        idle_trigger,
        f"{quiet_s:.1f} s" if quiet_s > 0 else "off",
    )


def install_if_enabled() -> None:
    """Installs the release placement on vLLM's DP load-balancing client when
    ``DP_PLACEMENT=release``."""
    if envs.DP_PLACEMENT != "release":
        return
    from vllm.v1.engine.core_client import DPLBAsyncMPClient

    install_release(DPLBAsyncMPClient)

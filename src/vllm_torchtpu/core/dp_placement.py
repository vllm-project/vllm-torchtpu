"""The data-parallel router's release rules (DP_PLACEMENT=release).

vLLM's router places each arriving request on the least-loaded engine as it
arrives, so arrivals follow completions and the engines' prompt steps drift
apart. With DP_PLACEMENT=release the router holds new requests and releases
them as a group, one per engine where it can, so the engines run their prompt
steps together. Off by default; API-server process only.
"""

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

MODES = ("release", )


def mode() -> str:
    """The DP router's placement mode (DP_PLACEMENT)."""
    from vllm_torchtpu import envs
    return envs.DP_PLACEMENT


def enabled() -> bool:
    """Whether the router's release rules are on."""
    return mode() in MODES


def install_release(client_cls,
                    threshold: int | None = None,
                    deadline_s: float | None = None,
                    idle_trigger: bool | None = None,
                    quiet_s: float | None = None) -> None:
    """Install the release rules on vLLM's DP load-balancing client class.

    Arrivals are held in the router (on no engine yet) and released together
    when `threshold` are pending (the engine count when 0), when any engine is
    idle (by the coordinator's counts, or no wave running), when the oldest
    has waited `deadline_s`, or when no request has arrived for `quiet_s`
    (0 = off). At release each request goes to the engine with the fewest
    assignments in this release, then the fewest in flight, ties cyclic, so a
    release of one per engine lands one per engine and an idle engine is fed
    first; a request pinned to a data-parallel rank keeps its engine. The
    engines then admit at once.
    """
    import asyncio
    import time as _time
    from collections import Counter

    import msgspec  # pyrefly: ignore  # vLLM's own dependency, not on the checker's path
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.exceptions import EngineDeadError

    from vllm_torchtpu import envs
    if getattr(client_cls, "_tpu_dp_release_patch", False):
        return
    threshold = threshold if threshold is not None else envs.DP_SCHED_BUFFER_THRESHOLD
    deadline_s = deadline_s if deadline_s is not None else envs.DP_SCHED_BUFFER_FLUSH_MS / 1000.0
    idle_trigger = idle_trigger if idle_trigger is not None else envs.DP_SCHED_BUFFER_IDLE_TRIGGER
    quiet_s = quiet_s if quiet_s is not None else envs.DP_SCHED_BUFFER_QUIET_MS / 1000.0
    poll_s = 0.005
    orig_abort = client_cls.abort_requests_async
    orig_shutdown = getattr(client_cls, "shutdown", None)

    def _buf(self):
        return self.__dict__.setdefault("_release_buf", [])

    def _thresh(self):
        return threshold or len(self.core_engines)

    def _any_idle(self):
        if not getattr(self, "engines_running", True):
            return True
        counts = getattr(self, "lb_engines", None) or []
        return any(int(c[0]) == 0 and int(c[1]) == 0 for c in counts)

    async def _send_one(self, request, idx):
        chosen = self.core_engines[idx]
        request.current_wave = self.current_wave
        request.client_index = self.client_index
        counts = getattr(self, "lb_engines", None)
        self.reqs_in_flight[request.request_id] = chosen
        self.engine_inflight[chosen] += 1
        if counts and idx < len(counts):
            counts[idx][0] += self.client_count
        try:
            to_await = self._send_input(EngineCoreRequestType.ADD, request,
                                        chosen)
            if not self.engines_running:
                await self.first_req_send_socket.send(
                    msgspec.msgpack.encode(("FIRST_REQ", chosen)))
            await to_await
        except BaseException:
            # The send failed: the counts must not claim a request no engine
            # holds.
            self.reqs_in_flight.pop(request.request_id, None)
            self.engine_inflight[chosen] -= 1
            if counts and idx < len(counts):
                counts[idx][0] -= self.client_count
            raise
        self._ensure_output_queue_task()

    async def _release_all(self):
        lock = self.__dict__.setdefault("_release_lock", asyncio.Lock())
        async with lock:
            buf = _buf(self)
            num = len(self.core_engines)
            assigned = Counter()
            start = getattr(self, "_rr_next", 0) % num
            while buf:
                # The request leaves the buffer before its send: a cancel that
                # lands during the send finds it gone here and, through the
                # send's registration, reaches the engine instead.
                _t0, request = buf.pop(0)
                pinned = getattr(request, "data_parallel_rank", None)
                if pinned is not None:
                    idx = int(pinned)
                else:
                    order = [(i + start) % num for i in range(num)]
                    idx = min(order,
                              key=lambda i: (assigned[i], self.engine_inflight[
                                  self.core_engines[i]]))
                    assigned[idx] += 1
                    start = (idx + 1) % num
                try:
                    await _send_one(self, request, idx)
                except BaseException:
                    # A request whose send failed is reported; the rest stay
                    # held for the next release.
                    logger.exception(
                        "DP placement: request %s could not be "
                        "sent to engine %s", request.request_id,
                        self.core_engines[idx])
                    raise
            self._rr_next = start

    async def _maybe_release(self):
        buf = _buf(self)
        if not buf:
            return
        if getattr(getattr(self, "resources", None), "engine_dead", False):
            buf.clear()
            raise EngineDeadError()
        now = _time.time()
        if (len(buf) >= _thresh(self) or now - buf[0][0] >= deadline_s
                or (idle_trigger and _any_idle(self))
                or (quiet_s > 0 and now - buf[-1][0] >= quiet_s)):
            await _release_all(self)

    async def _poll(self):
        while True:
            await asyncio.sleep(poll_s)
            try:
                await _maybe_release(self)
            except Exception:  # the poller stays alive; the failure is logged above
                logger.exception("DP placement: release poller")

    async def add_request_async(self, request):
        self._ensure_stats_update_task()
        if self.__dict__.get("_release_task") is None:
            self._release_task = asyncio.get_running_loop().create_task(
                _poll(self))
        _buf(self).append((_time.time(), request))
        await _maybe_release(self)

    async def abort_requests_async(self, request_ids):
        ids = set(request_ids)
        buf = _buf(self)
        if buf:
            # Filtered in place: a release in progress holds this same list.
            buf[:] = [(t, r) for t, r in buf if r.request_id not in ids]
        await orig_abort(self, request_ids)

    def shutdown(self, *args, **kwargs):
        task = self.__dict__.get("_release_task")
        if task is not None:
            task.cancel()
        if orig_shutdown is not None:
            return orig_shutdown(self, *args, **kwargs)

    client_cls.add_request_async = add_request_async
    client_cls.abort_requests_async = abort_requests_async
    if orig_shutdown is not None:
        client_cls.shutdown = shutdown
    client_cls._tpu_dp_release_patch = True
    logger.info(
        "DP placement: held in the router and released across the engines "
        "(DP_PLACEMENT=release; threshold %s, deadline %.1f s, idle trigger "
        "%s, quiet release %s)", threshold or "engine count", deadline_s,
        idle_trigger, f"{quiet_s:.1f} s" if quiet_s > 0 else "off")


def install_if_enabled() -> None:
    """Install the release rules on vLLM's DP load-balancing client when
    DP_PLACEMENT is set; any value but 'release' is refused."""
    m = mode()
    if not m:
        return
    if m not in MODES:
        raise ValueError(f"DP_PLACEMENT={m!r}: the only mode is 'release'")
    from vllm.v1.engine.core_client import DPLBAsyncMPClient
    install_release(DPLBAsyncMPClient)

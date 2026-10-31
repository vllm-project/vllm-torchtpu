# SPDX-License-Identifier: Apache-2.0
"""Shortest-uncached-prompt-first waiting queue for a P/D prefill engine.

With ``K3_PREFILL_SJF=1`` a KV producer's scheduler admits waiting requests
in order of their uncached prompt tokens (estimated from the local prefix
cache when the request is queued), minus an aging credit of
``K3_PREFILL_SJF_AGING`` tokens per second waited, so long prompts are not
starved. Running requests keep their chunks; the scheduling policy stays
FCFS, so preemption picks its usual victim and never reorders by size.
Requests put back at the front (``prepend_request``, e.g. after a
preemption) go ahead of everything.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING

from vllm_torchtpu import envs

if TYPE_CHECKING:
    from vllm.v1.core.sched.request_queue import RequestQueue
    from vllm.v1.request import Request

DEFAULT_AGING = 16384.0  # tokens of credit per second waited


def enabled() -> bool:
    return envs.K3_PREFILL_SJF


def aging() -> float:
    return envs.K3_PREFILL_SJF_AGING


class SJFRequestQueue:
    """vLLM's ``RequestQueue`` interface (duck-typed, so importing this module
    does not import vLLM) over a small list ordered at peek/pop time by
    ``uncached - aging * waited`` (ties: arrival time). ``estimate(request)``
    returns the request's uncached prompt tokens; it is called once, when the
    request is added."""

    def __init__(
        self,
        estimate: Callable[[Request], int],
        aging_per_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._items: list[
            tuple[Request, float, float, bool]
        ] = []  # request, size, enqueue time, front
        self._estimate = estimate
        self._aging = aging_per_s
        self._clock = clock
        self._fronts = 0  # prepends so far: the latest goes first
        self._peeked: Request | None = None  # pop returns what peek just showed

    def _key(self, item, now):
        request, size, queued, front = item
        if front:
            return (0, queued, 0.0)
        return (1, size - self._aging * (now - queued), request.arrival_time)

    def _best(self) -> int:
        if not self._items:
            raise IndexError("empty queue")
        now = self._clock()
        return min(
            range(len(self._items)), key=lambda i: self._key(self._items[i], now)
        )

    def add_request(self, request: Request) -> None:
        try:
            size = float(self._estimate(request))
        except Exception:  # noqa: BLE001 - ordering is best effort
            size = float(request.num_tokens)
        self._items.append((request, size, self._clock(), False))

    def pop_request(self) -> Request:
        if self._peeked is not None:
            for i, item in enumerate(self._items):
                if item[0] is self._peeked:
                    self._peeked = None
                    return self._items.pop(i)[0]
        self._peeked = None
        return self._items.pop(self._best())[0]

    def peek_request(self) -> Request:
        self._peeked = self._items[self._best()][0]
        return self._peeked

    def prepend_request(self, request: Request) -> None:
        self._fronts += 1
        self._items.append((request, 0.0, -float(self._fronts), True))

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in reversed(list(requests)):
            self.prepend_request(request)

    def remove_request(self, request: Request) -> None:
        for i, item in enumerate(self._items):
            if item[0] is request:
                del self._items[i]
                return
        raise ValueError("request not in queue")

    def remove_requests(self, requests: Iterable[Request]) -> None:
        drop = {id(r) for r in requests}
        self._items = [item for item in self._items if id(item[0]) not in drop]

    def __bool__(self) -> bool:
        return bool(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Request]:
        now = self._clock()
        return iter(
            [item[0] for item in sorted(self._items, key=lambda it: self._key(it, now))]
        )


def uncached_estimator(scheduler) -> Callable[[Request], int]:
    """Uncached prompt tokens of a request against the scheduler's local
    prefix cache (read-only longest-hit lookup; 0 hits if unavailable)."""

    def estimate(request: Request) -> int:
        manager = scheduler.kv_cache_manager
        hit = 0
        if getattr(manager, "enable_caching", False) and request.block_hashes:
            _, hit, *_ = manager.coordinator.find_longest_cache_hit(
                request.block_hashes, request.num_tokens - 1
            )
        return max(int(request.num_tokens) - int(hit), 0)

    return estimate

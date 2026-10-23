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
"""Decode KV prefetch past a full running set (TPU_SCHED_KV_PREFETCH_WAITING=N).

On a P/D consumer vLLM's waiting loop stops at ``len(running) >=
max_num_running_reqs`` before it asks the connector about the next waiting
request, and only then starts that request's remote KV pull. With every slot
taken, the next request's pull therefore starts only once a slot frees, and
that slot idles for the whole pull (1-2 s for a 150K-token GLM-5.2 context).

With N > 0 on a KV consumer, a step that finds the running set full and fresh
requests waiting lifts the loop's cap and enforces the real one itself, so the
loop can only start async loads:

* a request whose load finished is not promoted while the set is full
  (``_try_promote_blocked_waiting_request``): it waits, KV-ready, for a slot;
* a request that would run without an async load stops the scan
  (``_reserve_prefill_lookahead`` returns 0 for a request not yet running);
* a new load is refused once N requests are held (loading or loaded, not yet
  running), and otherwise must leave TPU_SCHED_KV_PREFETCH_RESERVE_BLOCKS free
  blocks for the running set beyond the in-flight loads' reservation
  (``_inflight_prefill_reserved_blocks``).

When a slot frees, FCFS selection serves ``skipped_waiting`` (the held
requests) first, so a KV-ready request takes the slot on the same step. Ported
from the Qwen3.5 PrefixAwareScheduler's decode prefetch.
"""

import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# A reservation no allocation fits: refuses a new async load.
_NO_ROOM = 1 << 40


class _KvPrefetchMixin:
    """Overrides for vLLM's Scheduler; see the module docstring."""

    _LOG_INTERVAL_S = 60.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        transfer = self.vllm_config.kv_transfer_config
        consumer = (
            self.connector is not None
            and transfer is not None
            and transfer.is_kv_consumer
        )
        n = int(envs.TPU_SCHED_KV_PREFETCH_WAITING)
        if n and not consumer:
            logger.info(
                "TPU_SCHED_KV_PREFETCH_WAITING=%d ignored: not a KV consumer", n
            )
        self._kv_prefetch_n = n if consumer else 0
        self._kv_prefetch_reserve = int(envs.TPU_SCHED_KV_PREFETCH_RESERVE_BLOCKS)
        # The real running cap while a lifted step runs, else None.
        self._kv_prefetch_cap: int | None = None
        self._kv_prefetch_lifted = 0
        self._kv_prefetch_started = 0
        self._kv_prefetch_held_skips = 0
        self._kv_prefetch_full_refusals = 0
        self._kv_prefetch_log_t = time.monotonic()
        if self._kv_prefetch_n:
            logger.info(
                "Decode KV prefetch past a full running set: up to %d held "
                "requests, %d reserve blocks",
                self._kv_prefetch_n,
                self._kv_prefetch_reserve,
            )

    def _kv_prefetch_num_running(self) -> int:
        return len(self.running) + self.num_waiting_for_streaming_input

    def _kv_prefetch_at_cap(self) -> bool:
        cap = self._kv_prefetch_cap
        return cap is not None and self._kv_prefetch_num_running() >= cap

    def _kv_prefetch_num_held(self) -> int:
        """Requests holding blocks for an async load (loading or loaded) that
        have not run yet."""
        return sum(
            1 for r in self._inflight_prefills if r.status != RequestStatus.RUNNING
        )

    def schedule(self, *args, **kwargs):
        cap = self.max_num_running_reqs
        if (
            not self._kv_prefetch_n
            or self._kv_prefetch_num_running() < cap
            or not self.waiting
            or self._kv_prefetch_num_held() >= self._kv_prefetch_n
        ):
            return super().schedule(*args, **kwargs)
        held = self._kv_prefetch_num_held()
        self._kv_prefetch_cap = cap
        # Any value above the cap lets the stock loop reach the waiting queue;
        # the overrides below keep the running set at the cap.
        self.max_num_running_reqs = cap + self._kv_prefetch_n
        try:
            output = super().schedule(*args, **kwargs)
        finally:
            self.max_num_running_reqs = cap
            self._kv_prefetch_cap = None
        self._kv_prefetch_lifted += 1
        self._kv_prefetch_started += max(0, self._kv_prefetch_num_held() - held)
        if len(self.running) > cap:
            logger.error(
                "KV prefetch: running=%d exceeds max_num_seqs=%d; disabling "
                "TPU_SCHED_KV_PREFETCH_WAITING",
                len(self.running),
                cap,
            )
            self._kv_prefetch_n = 0
        self._kv_prefetch_maybe_log()
        return output

    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
        if (
            request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            and self._kv_prefetch_at_cap()
        ):
            # Held, KV-ready or still loading, until a slot frees.
            self._kv_prefetch_held_skips += 1
            return False
        return super()._try_promote_blocked_waiting_request(request)

    def _reserve_prefill_lookahead(
        self, request: Request, num_computed_tokens: int, num_new_tokens: int
    ) -> int:
        if request.status != RequestStatus.RUNNING and self._kv_prefetch_at_cap():
            # Would join the running set past the cap: stop the scan.
            return 0
        return super()._reserve_prefill_lookahead(
            request, num_computed_tokens, num_new_tokens
        )

    def _inflight_prefill_reserved_blocks(self) -> int:
        reserved = super()._inflight_prefill_reserved_blocks()
        if not self._kv_prefetch_at_cap():
            return reserved
        if self._kv_prefetch_num_held() >= self._kv_prefetch_n:
            self._kv_prefetch_full_refusals += 1
            return _NO_ROOM
        return reserved + self._kv_prefetch_reserve

    def _kv_prefetch_maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._kv_prefetch_log_t < self._LOG_INTERVAL_S:
            return
        self._kv_prefetch_log_t = now
        logger.info(
            "KV prefetch: lifted_steps=%d started=%d held_skips=%d "
            "full_refusals=%d held=%d running=%d waiting=%d",
            self._kv_prefetch_lifted,
            self._kv_prefetch_started,
            self._kv_prefetch_held_skips,
            self._kv_prefetch_full_refusals,
            self._kv_prefetch_num_held(),
            len(self.running),
            len(self.waiting) + len(self.skipped_waiting),
        )


class KvPrefetchScheduler(_KvPrefetchMixin, Scheduler):
    """vLLM's Scheduler with decode KV prefetch."""


class KvPrefetchAsyncScheduler(_KvPrefetchMixin, AsyncScheduler):
    """vLLM's AsyncScheduler with decode KV prefetch."""

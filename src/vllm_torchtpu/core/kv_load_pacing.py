# SPDX-License-Identifier: Apache-2.0
"""Caps the remote KV loads one schedule pass starts (engine core).

A request that starts an async KV load leaves vLLM's waiting scan in
WAITING_FOR_REMOTE_KVS without taking a running slot, so the scan moves on
to the next waiting request. Once a slot is free, one pass starts a load for
every waiting request the KV pool can hold. On the P/D decode engine each
start costs the engine core about 3 ms (``allocate_slots``, the local
prefix lookup, the connector's ``update_state_after_alloc``) and every
worker its submit, all inside one step.

With ``VLLM_TPU_KV_LOAD_STARTS_PER_STEP=N`` on a KV-consumer engine, a pass
lets at most N requests that would start a remote load through
``Scheduler._handle_blocked_waiting_request``, the scan's first check, and
skips the rest before any lookup. vLLM's skip path re-queues skipped
requests at the head of their queue in scan order, so the next pass starts
the next N in queue order. A request counts against N once it passes the
check; on the decode every such request starts a load (a local prefix hit
never covers the whole transfer), and a pass that fails to allocate for one
stops scanning anyway.

The cap changes nothing else: loading requests still do not count toward
``max_num_seqs``, loads start only in passes that have a free running slot
(as without the cap), and requests whose load has landed, preempted
requests and requests without remote-KV parameters are scheduled as before.
N = 0 leaves the scheduler unpatched.
"""

import inspect
from functools import wraps

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_PATCH_FLAG = "_tpu_kv_load_pacing_patch"
# Per-Scheduler [cap, requests let through in the current pass], in the
# instance __dict__; created by the first schedule() call.
_STATE = "_tpu_kv_load_pacing"


def _scheduler_cap(scheduler, cap: int) -> int:
    """The cap for this scheduler: 0 unless it consumes remote KV."""
    transfer = scheduler.vllm_config.kv_transfer_config
    if scheduler.connector is None or transfer is None:
        return 0
    return cap if transfer.kv_role == "kv_consumer" else 0


def starts_remote_load(request) -> bool:
    """Whether the waiting scan would query the connector for a remote load.

    Mirrors the scan and ``TPUConnectorScheduler.get_num_new_matched_tokens``:
    only a request with no computed tokens is looked up, and the connector
    loads for parameters it has not marked ``_remote_kv_processed`` (set
    when a load is recorded, so preempted and landed requests do not count).
    """
    if request.num_computed_tokens or request.skip_reading_prefix_cache:
        return False
    params = request.kv_transfer_params
    return (
        isinstance(params, dict)
        and bool(params)
        and not params.get("_remote_kv_processed")
    )


def patch_kv_load_pacing() -> None:
    cap = envs.VLLM_TPU_KV_LOAD_STARTS_PER_STEP
    if not cap:
        return
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler._handle_blocked_waiting_request, _PATCH_FLAG, False):
        return
    scan = inspect.unwrap(Scheduler.schedule)
    if "_handle_blocked_waiting_request" not in scan.__code__.co_names:
        logger.warning(
            "KV load pacing not installed: Scheduler.schedule does not call "
            "_handle_blocked_waiting_request."
        )
        return

    original_schedule = Scheduler.schedule
    original_handle = Scheduler._handle_blocked_waiting_request

    @wraps(original_schedule)
    def schedule(self, *args, **kwargs):
        state = self.__dict__.get(_STATE)
        if state is None:
            state = self.__dict__[_STATE] = [_scheduler_cap(self, cap), 0]
        state[1] = 0
        return original_schedule(self, *args, **kwargs)

    @wraps(original_handle)
    def _handle_blocked_waiting_request(self, request) -> bool:
        if not original_handle(self, request):
            return False
        state = self.__dict__.get(_STATE)
        if not state or not state[0] or not starts_remote_load(request):
            return True
        if state[1] >= state[0]:
            return False
        state[1] += 1
        return True

    for fn in (schedule, _handle_blocked_waiting_request):
        setattr(fn, _PATCH_FLAG, True)
    Scheduler.schedule = schedule
    Scheduler._handle_blocked_waiting_request = _handle_blocked_waiting_request
    logger.info(
        "Applied TPU patch: a schedule pass starts at most %d remote KV loads "
        "(VLLM_TPU_KV_LOAD_STARTS_PER_STEP).",
        cap,
    )

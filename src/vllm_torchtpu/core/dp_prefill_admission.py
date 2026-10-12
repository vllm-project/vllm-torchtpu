# SPDX-License-Identifier: Apache-2.0
"""Coalesce cache-affine requests before a fresh EP/DP wave is scheduled.

Routing and KV ownership stay unchanged. The engine processes requests that
arrive while its peers wake up *before* producing SchedulerOutput, rather than
forcing those requests to trail a dummy by one complete model step. A bounded
window can collect a burst; it never waits for a full DP batch indefinitely.
"""

import math
import queue
import time
from functools import wraps
from typing import Any

import torch
import torch.distributed as dist
from vllm.v1.core.sched.interface import PauseState

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _drain_requests(engine: Any) -> None:
    # Do not call _process_input_queue: it may block again after an abort.
    while True:
        try:
            message = engine.input_queue.get_nowait()
        except queue.Empty:
            return
        engine._handle_client_request(*message)


def coalesce_prefill_wave(engine: Any, window_ms: float) -> None:
    """Reach a common admission decision, including idle DP participants."""
    if not math.isfinite(window_ms) or window_ms < 0:
        raise ValueError("dp_prefill_admission_ms must be finite and nonnegative")
    start = time.monotonic()
    deadline = start + window_ms / 1000
    rounds = 0
    while True:
        _drain_requests(engine)
        scheduler = engine.scheduler
        # Fresh waves normally have no running requests. Resume/pause and
        # in-flight decode must never wait for a prefill burst to fill ranks.
        active = bool(scheduler.running) or bool(engine.batch_queue)
        stopped = (
            engine.pending_pause
            or scheduler.pause_state != PauseState.UNPAUSED
            or not engine.is_running()
        )
        ready = bool(scheduler.waiting) or bool(scheduler.kv_holding_waiting)
        state = torch.tensor(
            [int(ready), int(active or stopped), int(time.monotonic() >= deadline)],
            dtype=torch.int64,
            device="cpu",
        )
        dist.all_reduce(state, op=dist.ReduceOp.SUM, group=engine.dp_group)
        rounds += 1
        ready_ranks, bypass, expired = state.tolist()
        # The first rendezvous itself allows input threads to queue requests.
        # Always drain once more, even with a zero extra-wait budget.
        _drain_requests(engine)
        if bypass or ready_ranks in (0, engine.dp_size) or expired:
            break
        # Release the GIL for input preprocessing, but never extend the budget.
        time.sleep(min(0.0005, max(0.0, deadline - time.monotonic())))
    logger.debug(
        "DP prefill admission: rank=%d wave=%d ready=%d/%d rounds=%d wait_ms=%.3f",
        engine.dp_rank,
        engine.current_wave,
        ready_ranks,
        engine.dp_size,
        rounds,
        (time.monotonic() - start) * 1000,
    )


def patch_dp_prefill_admission() -> None:
    """Install the TPU engine hook; opt in via additional_config milliseconds."""
    from vllm.v1.engine.core import DPEngineCoreProc

    if getattr(DPEngineCoreProc, "_tpu_prefill_admission_patch", False):
        return
    original = DPEngineCoreProc._process_engine_step

    @wraps(original)
    def process_step(self: Any) -> bool:
        config = self.vllm_config
        window = config.additional_config.get("dp_prefill_admission_ms")
        if (
            window is not None
            and config.parallel_config.enable_expert_parallel
            and config.cache_config.enable_prefix_caching
            and self.step_counter == 0
            and self.has_work()
        ):
            coalesce_prefill_wave(self, float(window))
        return original(self)

    DPEngineCoreProc._process_engine_step = process_step
    DPEngineCoreProc._tpu_prefill_admission_patch = True

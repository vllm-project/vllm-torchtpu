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
"""Capacity preemption errors for KV consumers with kv_load_failure_policy='fail'.

Enabled automatically when ``kv_role == 'kv_consumer'`` and
``kv_load_failure_policy == 'fail'`` (upstream default), or overridden explicitly
via ``additional_config['abort_on_preemption']`` (``bool``).

Enabled scheduler instances wrap ``_preempt_request`` and ``update_from_output``
so ``RUNNING`` decode requests evicted under KV cache capacity pressure terminate
with ``finish_reason=FinishReason.ERROR``, ``stop_reason='kv_cache_preempted'``,
and ``kv_transfer_params['kv_transfer_error']`` (``status_code=500``,
``status_name='model.KVCachePreempt'``) instead of re-entering ``waiting`` to
recompute prompts locally, while keeping ``reset_prefix_cache`` administrative
preemption intact.
"""

from functools import wraps
from typing import TYPE_CHECKING

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import (
    EngineCoreEventType,
    EngineCoreOutput,
    EngineCoreOutputs,
    FinishReason,
)
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)


class _CapacityPreemption:
    """Own the terminal snapshots and preserve the scheduler's output chain."""

    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
        self.original_preempt = scheduler._preempt_request
        self.original_reset = scheduler.reset_prefix_cache
        self.original_update = scheduler.update_from_output
        self.pending: dict[str, tuple[int, EngineCoreOutput]] = {}
        self._in_reset = False

    def preempt_request(
        self,
        request: Request,
        timestamp: float,
        drop_stale_output: bool = False,
    ) -> None:
        if self._in_reset:
            self.original_preempt(
                request,
                timestamp,
                drop_stale_output=drop_stale_output,
            )
            return
        self.abort(request, timestamp)

    def reset_prefix_cache(
        self,
        reset_running_requests: bool = False,
        reset_connector: bool = False,
    ) -> bool:
        self._in_reset = True
        try:
            return self.original_reset(
                reset_running_requests=reset_running_requests,
                reset_connector=reset_connector,
            )
        finally:
            self._in_reset = False

    def abort(self, request: Request, timestamp: float) -> None:
        scheduler = self.scheduler
        assert request.status == RequestStatus.RUNNING, (
            "Capacity preemption requires a RUNNING victim: "
            f"request_id={request.request_id}, status={request.status}"
        )
        # Preserve preemption metrics and event timeline so capacity evictions
        # remain observable alongside FinishReason.ERROR.
        request.num_preemptions += 1
        if scheduler.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)
        request.stop_reason = "kv_cache_preempted"
        self.pending[request.request_id] = (
            request.client_index,
            EngineCoreOutput(
                request_id=request.request_id,
                new_token_ids=[],
                finish_reason=FinishReason.ERROR,
                stop_reason=request.stop_reason,
                events=request.take_events(),
                trace_headers=request.trace_headers,
                prefill_stats=request.take_prefill_stats(),
                kv_transfer_params={
                    "kv_transfer_error": {
                        "status_code": 500,
                        "status_name": "model.KVCachePreempt",
                        "status_message": (
                            "D-side request was preempted due to KV cache capacity."
                        ),
                    }
                },
            ),
        )
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ERROR)
        assert request.request_id not in scheduler.requests, (
            "Capacity preemption abort requires immediate block release: "
            f"request_id={request.request_id}"
        )
        scheduler.reset_preempted_req_ids.add(request.request_id)
        logger.warning(
            "Aborted capacity-preempted request_id=%s error=model.KVCachePreempt",
            request.request_id,
        )

    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> dict[int, EngineCoreOutputs]:
        outputs = self.original_update(scheduler_output, model_runner_output)
        # An older async output must not consume a later step's terminal.
        # finished_req_ids is also delivered for a zero-token cleanup step.
        for req_id in scheduler_output.finished_req_ids:
            pending = self.pending.pop(req_id, None)
            if pending is not None:
                client_index, terminal = pending
                batch = outputs.setdefault(client_index, EngineCoreOutputs())
                batch.outputs.append(terminal)
        return outputs


def _should_enable_abort_on_preemption(vllm_config: "VllmConfig") -> bool:
    kv = vllm_config.kv_transfer_config
    is_kv_consumer = kv is not None and kv.kv_role == "kv_consumer"
    config = vllm_config.additional_config
    if isinstance(config, dict) and "abort_on_preemption" in config:
        enabled = config["abort_on_preemption"]
        if not isinstance(enabled, bool):
            raise ValueError("additional_config.abort_on_preemption must be a bool")
        if enabled and not is_kv_consumer:
            raise ValueError("abort_on_preemption requires kv_role='kv_consumer'")
        return enabled
    return is_kv_consumer and kv is not None and kv.kv_load_failure_policy == "fail"


def patch_scheduler_init() -> None:
    if Scheduler.__dict__.get("_tpu_abort_on_preemption_patch", False):
        return
    original_init = Scheduler.__init__

    @wraps(original_init)
    def init(self: Scheduler, vllm_config: "VllmConfig", *args, **kwargs) -> None:
        enabled = _should_enable_abort_on_preemption(vllm_config)
        original_init(self, vllm_config, *args, **kwargs)
        if enabled:
            state = _CapacityPreemption(self)
            self.__dict__["_tpu_capacity_preemption"] = state
            self._preempt_request = state.preempt_request
            self.reset_prefix_cache = state.reset_prefix_cache
            self.update_from_output = state.update_from_output
            logger.info("Enabled capacity preemption abort for KV consumer")

    Scheduler.__init__ = init
    Scheduler._tpu_abort_on_preemption_patch = True

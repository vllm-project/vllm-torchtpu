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
"""Scheduler that runs each chunk of a pipelined prefill alone in its step
(TPU_DP_PIPELINE_PREFILL, ``distributed/dp_pipeline``).

Every engine of a pipeline runs a different range of the layers on the same
chunk, so a step that holds a pipelined chunk can hold nothing else, and every
engine must cut the prompt into the same chunks. While a request carrying a
pipeline plan has prompt tokens left to schedule, each step schedules only
that request, with the step's whole token budget: the other requests sit the
step out, and since the request is the only one the step sees, upstream's
``long_prefill_token_threshold`` does not apply, as for a lone prompt. Its
chunks are then the same on every engine whatever else each engine holds.

The API server only plans a pipeline over engines with nothing in flight
(``core/dp_lb_exact_inflight.py``), so a pipelined request never finds a
running request here when it arrives; requests routed to the engine after it
wait for its prompt to be scheduled. Once its prompt is scheduled the request
is an ordinary one: if it is preempted later, this engine recomputes it alone.
"""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.request import Request

from vllm_torchtpu.distributed.dp_pipeline.plan import is_pipelined

SCHEDULER_CLS = "vllm_torchtpu.core.dp_pipeline_scheduler.DpPipelineScheduler"


class DpPipelineScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Pipelined requests whose prompt is scheduled.
        self._pipelined_done: set[str] = set()

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self._pipelined_done.intersection_update(self.requests)
        request = self._pipelined_prefill()
        if request is None:
            return super().schedule(throttle_prefills)
        return self._schedule_alone(request, throttle_prefills)

    def _pipelined(self, request: Request) -> bool:
        return (
            is_pipelined(request.sampling_params)
            and request.request_id not in self._pipelined_done
        )

    def _pipelined_prefill(self) -> Request | None:
        """The request whose next chunk this step must hold alone, if any."""
        for request in self.running:
            if self._pipelined(request):
                return request
        for request in self.waiting:
            if self._pipelined(request):
                if self.running:
                    raise RuntimeError(
                        f"pipelined request {request.request_id} reached an engine "
                        f"running {len(self.running)} requests; the API server "
                        "plans pipelines over idle engines only"
                    )
                return request
        return None

    def _schedule_alone(
        self, request: Request, throttle_prefills: bool
    ) -> SchedulerOutput:
        """Runs upstream's schedule() over queues that hold only ``request``."""
        running, waiting, holding = (
            self.running,
            self.waiting,
            self.kv_holding_waiting,
        )
        was_running = request in running
        self.running = [request] if was_running else []
        self.waiting = create_request_queue(self.policy)
        self.kv_holding_waiting = create_request_queue(self.policy)
        if not was_running:
            self.waiting.add_request(request)
        try:
            output = super().schedule(throttle_prefills)
            now_running = request in self.running
        finally:
            self.running, self.waiting, self.kv_holding_waiting = (
                running,
                waiting,
                holding,
            )
        if was_running and not now_running:
            # Its other stages hold the chunks it computed; it cannot start over.
            raise RuntimeError(
                f"pipelined request {request.request_id} was preempted: its "
                "prompt does not fit the KV cache"
            )
        if now_running and not was_running:
            waiting.remove_request(request)
            running.append(request)
        if request.num_computed_tokens >= request.num_prompt_tokens:
            self._pipelined_done.add(request.request_id)
        return output

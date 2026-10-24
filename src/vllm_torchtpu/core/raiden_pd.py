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
"""Raiden response lifetimes on the official vLLM schedulers.

The adapter owns only producer transport response state. Scheduling,
asynchronous token accounting, decode load-failure handling, and KV cleanup
remain with Scheduler / AsyncScheduler.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiKVConnectorWorkerMetadata,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu.distributed.kv_transfer.raiden_pd_completion import (
    RaidenSendOutcome,
    RaidenSendResults,
    merge_raiden_send_outcomes,
    raiden_pd_completion_modes,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@dataclass
class _PendingResponse:
    client_index: int
    finish_reason: FinishReason = FinishReason.STOP
    stop_reason: int | str | None = None
    kv_transfer_params: dict[str, Any] = field(default_factory=dict)
    outcome: RaidenSendOutcome | None = None


_ERROR_BY_OUTCOME = {
    RaidenSendOutcome.SEND_FAILURE: (
        "model.KVTPrefillWriteFailure",
        "P-side failed to write KV cache data. Category: internal.",
    ),
    RaidenSendOutcome.UNCLAIMED_TIMEOUT: (
        "model.KVTPrefillTransferTimeout",
        "P-side timed out waiting for KV cache transfer completion.",
    ),
    None: (
        "model.KVTPrefillWriteFailure",
        "P-side KV cache transfer finished without a sender result. "
        "Category: internal.",
    ),
}


class _ProducerCompletion:
    """Keep an opted-in P response open until the existing send barrier."""

    def __init__(self, scheduler: Scheduler):
        self.original_update = scheduler.update_from_output
        self.original_finish = scheduler.finish_requests
        self.pending_responses: dict[str, _PendingResponse] = {}
        scheduler.update_from_output = self.update_from_output
        scheduler.finish_requests = self.finish_requests

    def update_from_output(self, scheduler_output, model_runner_output):
        outputs = self.original_update(scheduler_output, model_runner_output)
        for client_index, batch in outputs.items():
            for output in batch.outputs:
                params = output.kv_transfer_params
                if (
                    output.finish_reason in (FinishReason.STOP, FinishReason.LENGTH)
                    and params
                    and params.get("kv_transfer_pending")
                ):
                    transfer_params = {
                        k: v for k, v in params.items() if k != "kv_transfer_pending"
                    }
                    self.pending_responses[output.request_id] = _PendingResponse(
                        client_index=client_index,
                        finish_reason=output.finish_reason,
                        stop_reason=output.stop_reason,
                        kv_transfer_params=transfer_params,
                    )
                    output.finish_reason = None
                    output.stop_reason = None

        kv_output = model_runner_output.kv_connector_output
        if kv_output is None:
            return outputs
        metadata = kv_output.kv_connector_worker_meta
        children = (
            metadata.metadata
            if isinstance(metadata, MultiKVConnectorWorkerMetadata)
            else (metadata,)
        )
        for child in children:
            if isinstance(child, RaidenSendResults):
                for req_id, outcome in child.outcomes.items():
                    pending = self.pending_responses.get(req_id)
                    if pending is not None:
                        pending.outcome = merge_raiden_send_outcomes(
                            pending.outcome, outcome
                        )

        # Cross-rank counting and KV cleanup belong to vLLM. Preserve outcomes
        # across steps until that existing finished_sending barrier arrives.
        for req_id in kv_output.finished_sending or ():
            pending = self.pending_responses.pop(req_id, None)
            if pending is None:
                continue
            outcome = pending.outcome
            success = outcome is RaidenSendOutcome.SUCCESS
            if success:
                params = {**pending.kv_transfer_params, "kv_transfer_done": True}
                finish_reason = pending.finish_reason
                stop_reason = pending.stop_reason
            else:
                status_name, status_message = _ERROR_BY_OUTCOME[outcome]
                params = {
                    "kv_transfer_error": {
                        "status_code": 500,
                        "status_name": status_name,
                        "status_message": status_message,
                    }
                }
                finish_reason = FinishReason.ERROR
                stop_reason = None
            batch = outputs.setdefault(pending.client_index, EngineCoreOutputs())
            batch.outputs.append(
                EngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=[],
                    finish_reason=finish_reason,
                    stop_reason=stop_reason,
                    kv_transfer_params=params,
                )
            )
            logger.info(
                "Raiden P send terminal: request_id=%s outcome=%s", req_id, outcome
            )
        return outputs

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[Request]:
        if request_ids is None:
            self.pending_responses.clear()
        else:
            request_ids = (
                (request_ids,) if isinstance(request_ids, str) else tuple(request_ids)
            )
            for req_id in request_ids:
                self.pending_responses.pop(req_id, None)
        # Cancelling a model-finished response does not authorize early KV free.
        return self.original_finish(request_ids, finished_status)


def patch_scheduler_init() -> None:
    if Scheduler.__dict__.get("_tpu_raiden_pd_completion_patch", False):
        return
    original_init = Scheduler.__init__

    @wraps(original_init)
    def init(self, vllm_config, *args, **kwargs):
        send_completion = raiden_pd_completion_modes(vllm_config)
        if send_completion and type(self) not in (
            Scheduler,
            AsyncScheduler,
        ):
            raise ValueError(
                "Raiden PD completion requires the official Scheduler or "
                f"AsyncScheduler; got {type(self).__name__}"
            )
        original_init(self, vllm_config, *args, **kwargs)
        if send_completion:
            if self.finished_req_ids_dict is not None:
                raise ValueError(
                    "Raiden PD send completion does not support multi-engine "
                    "finished-request bookkeeping"
                )
            _ProducerCompletion(self)
            logger.info("Enabled Raiden P send completion on %s", type(self).__name__)

    Scheduler.__init__ = init
    Scheduler._tpu_raiden_pd_completion_patch = True

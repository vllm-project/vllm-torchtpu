# SPDX-License-Identifier: Apache-2.0
"""Raiden response lifetimes on the official vLLM schedulers.

The adapters own only transport response state. Scheduling, asynchronous token
accounting, and KV cleanup remain with Scheduler / AsyncScheduler.
"""

import inspect
import linecache
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from functools import wraps

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorWorkerMetadata,
)
from vllm.sampling_params import RequestOutputKind
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu.distributed.kv_transfer.raiden_pd_completion import (
    RaidenLoadFailures,
    RaidenSendOutcome,
    RaidenSendResults,
    merge_raiden_send_outcomes,
    raiden_pd_completion_modes,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def patch_output_processor() -> None:
    """Keep pending KV transfers open after frontend stop-string checks."""
    from vllm.v1.engine.output_processor import OutputProcessor

    original = OutputProcessor.process_outputs
    marker = "_tpu_raiden_pd_completion_patch"
    if original.__dict__.get(marker, False):
        return

    # vLLM 0.29 has no hook between detokenization and request completion.
    # Adapt that boundary, as in core/abort_on_preemption.py, without copying
    # the upstream output loop or changing normal stop handling.
    source = textwrap.dedent(inspect.getsource(original))
    before = "        # 4) Create and handle RequestOutput objects."
    after = (
        "        if (\n"
        "            kv_transfer_params\n"
        '            and kv_transfer_params.get("kv_transfer_pending")\n'
        "            and req_state.output_kind is not RequestOutputKind.FINAL_ONLY\n"
        "        ):\n"
        "            finish_reason = None\n"
        "            stop_reason = None\n\n" + before
    )
    if source.count(before) != 1:
        raise RuntimeError(
            "Raiden KV completion requires the vLLM 0.29 output loop; "
            "expected one RequestOutput creation boundary"
        )
    source = "from __future__ import annotations\n" + source.replace(before, after, 1)
    filename = f"<{__name__}.process_outputs>"
    namespace = original.__globals__.copy()
    exec(compile(source, filename, "exec"), namespace)
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    patched = wraps(original)(namespace[original.__name__])
    patched.__dict__[marker] = True
    OutputProcessor.process_outputs = patched


@dataclass
class _PendingResponse:
    client_index: int
    outcome: RaidenSendOutcome | None = None
    finished_id_pending: bool = False


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
    """Keep a Raiden P response open until the existing send barrier."""

    def __init__(self, scheduler: Scheduler):
        from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
            TPURaidenConnector,
        )

        connector = scheduler.connector
        if isinstance(connector, MultiConnector):
            connector = next(
                child
                for child in connector._connectors
                if isinstance(child, TPURaidenConnector)
            )
        assert isinstance(connector, TPURaidenConnector), (
            "Raiden send completion requires a TPURaidenConnector"
        )
        self.connector = connector
        self.scheduler = scheduler
        self.original_update = scheduler.update_from_output
        self.original_finish = scheduler.finish_requests
        self.pending_responses: dict[str, _PendingResponse] = {}
        scheduler.update_from_output = self.update_from_output
        scheduler.finish_requests = self.finish_requests

    def update_from_output(self, scheduler_output, model_runner_output):
        # Upstream can free the Request on a same-step send completion.
        streaming_req_ids = set()
        for req_id in model_runner_output.req_ids:
            request = self.scheduler.requests.get(req_id)
            if (
                request is not None
                and request.sampling_params is not None
                and request.sampling_params.output_kind
                is not RequestOutputKind.FINAL_ONLY
            ):
                streaming_req_ids.add(req_id)
        outputs = self.original_update(scheduler_output, model_runner_output)
        for client_index, batch in outputs.items():
            for output in batch.outputs:
                params = output.kv_transfer_params
                if (
                    output.request_id in streaming_req_ids
                    and output.finish_reason in (FinishReason.STOP, FinishReason.LENGTH)
                    and params
                    and params.get("kv_transfer_pending")
                ):
                    self.pending_responses[output.request_id] = _PendingResponse(
                        client_index
                    )
                    output.finish_reason = None
                    output.stop_reason = None
            if batch.finished_requests:
                # Keep the DP client's abort route until the response terminates.
                deferred = batch.finished_requests & self.pending_responses.keys()
                for req_id in deferred:
                    self.pending_responses[req_id].finished_id_pending = True
                batch.finished_requests.difference_update(deferred)

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
                params = {"kv_transfer_done": True}
            else:
                status_name, status_message = _ERROR_BY_OUTCOME[outcome]
                params = {
                    "kv_transfer_error": {
                        "status_code": 500,
                        "status_name": status_name,
                        "status_message": status_message,
                    }
                }
            batch = outputs.setdefault(pending.client_index, EngineCoreOutputs())
            if pending.finished_id_pending:
                if batch.finished_requests is None:
                    batch.finished_requests = set()
                batch.finished_requests.add(req_id)
            batch.outputs.append(
                EngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=[],
                    finish_reason=FinishReason.STOP if success else FinishReason.ERROR,
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
        if request_ids is not None:
            request_ids = (
                (request_ids,) if isinstance(request_ids, str) else tuple(request_ids)
            )
        if finished_status == RequestStatus.FINISHED_ABORTED:
            # PP can register sends before model completion.
            pending_request_ids = (
                self.pending_responses
                if request_ids is None
                else (
                    req_id for req_id in request_ids if req_id in self.pending_responses
                )
            )
            self.connector.abort_pending_sends(pending_request_ids)
        pending_ids = (
            tuple(self.pending_responses) if request_ids is None else request_ids
        )
        for req_id in pending_ids:
            pending = self.pending_responses.pop(req_id, None)
            if pending is not None and pending.finished_id_pending:
                finished_req_ids = self.scheduler.finished_req_ids_dict
                assert finished_req_ids is not None, (
                    "Deferred Raiden response completion requires client bookkeeping: "
                    f"request_id={req_id}"
                )
                finished_req_ids[pending.client_index].add(req_id)
        # Expiry still requires an unclaimed registration or a native terminal.
        return self.original_finish(request_ids, finished_status)


def _load_failure_params() -> dict:
    return {
        "kv_transfer_error": {
            "status_code": 500,
            "status_name": "model.KVTDecodeLoadFailure",
            "status_message": (
                "D-side failed to load KV cache data. Category: internal."
            ),
        }
    }


class _DecodeLoadFailures:
    """Finish failed receives before upstream consumes their model outputs."""

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.original_update = scheduler.update_from_output
        self.original_invalid_blocks = scheduler._handle_invalid_blocks
        self.failed_req_ids: set[str] = set()
        scheduler.update_from_output = self.update_from_output
        scheduler._handle_invalid_blocks = self.handle_invalid_blocks

    def handle_invalid_blocks(self, invalid_block_ids, num_scheduled_tokens):
        failed = self.original_invalid_blocks(invalid_block_ids, num_scheduled_tokens)
        self.failed_req_ids.update(failed)
        return failed

    def update_from_output(self, scheduler_output, model_runner_output):
        assert not self.failed_req_ids, (
            "Raiden decode KV load failures leaked across scheduler steps: "
            f"{sorted(self.failed_req_ids)}"
        )
        try:
            kv_output = model_runner_output.kv_connector_output
            failed_req_ids: set[str] = set()
            if kv_output is not None:
                metadata = kv_output.kv_connector_worker_meta
                children = (
                    metadata.metadata
                    if isinstance(metadata, MultiKVConnectorWorkerMetadata)
                    else (metadata,)
                )
                for child in children:
                    if isinstance(child, RaidenLoadFailures):
                        failed_req_ids.update(child.failed_req_ids)
            # WAITING_FOR_REMOTE_KVS buffers remain pinned by upstream until
            # finished_recving. Finishing here only suppresses stale tokens.
            failed_requests = (
                self.scheduler.finish_requests(
                    failed_req_ids, RequestStatus.FINISHED_ERROR
                )
                if failed_req_ids
                else []
            )
            outputs = self.original_update(scheduler_output, model_runner_output)
            for request in failed_requests:
                batch = outputs.setdefault(request.client_index, EngineCoreOutputs())
                batch.outputs.append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=FinishReason.ERROR,
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                        kv_transfer_params=_load_failure_params(),
                    )
                )
            unmatched = set(self.failed_req_ids)
            for batch in outputs.values():
                for output in batch.outputs:
                    if output.request_id not in unmatched:
                        continue
                    assert output.finish_reason is FinishReason.ERROR, (
                        "A failed Raiden KV load must terminate with ERROR: "
                        f"request_id={output.request_id}, "
                        f"finish_reason={output.finish_reason}"
                    )
                    assert output.kv_transfer_params is None, (
                        "The base scheduler unexpectedly attached KV metadata "
                        f"to failed load request {output.request_id}"
                    )
                    output.kv_transfer_params = _load_failure_params()
                    unmatched.remove(output.request_id)
            assert not unmatched, (
                "The base scheduler did not emit terminal outputs for failed "
                f"Raiden KV loads: {sorted(unmatched)}"
            )
            return outputs
        finally:
            self.failed_req_ids.clear()


def patch_scheduler_init() -> None:
    if Scheduler.__dict__.get("_tpu_raiden_pd_completion_patch", False):
        return
    original_init = Scheduler.__init__

    @wraps(original_init)
    def init(self, vllm_config, *args, **kwargs):
        send_completion, decode_fail = raiden_pd_completion_modes(vllm_config)
        if (send_completion or decode_fail) and type(self) not in (
            Scheduler,
            AsyncScheduler,
        ):
            raise ValueError(
                "Raiden PD completion requires the official Scheduler or "
                f"AsyncScheduler; got {type(self).__name__}"
            )
        original_init(self, vllm_config, *args, **kwargs)
        if send_completion:
            _ProducerCompletion(self)
            logger.info("Enabled Raiden P send completion on %s", type(self).__name__)
        elif decode_fail:
            _DecodeLoadFailures(self)
            logger.info("Enabled Raiden decode load errors on %s", type(self).__name__)

    Scheduler.__init__ = init
    Scheduler._tpu_raiden_pd_completion_patch = True

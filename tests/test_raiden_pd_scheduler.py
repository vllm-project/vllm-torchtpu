# SPDX-License-Identifier: Apache-2.0
"""P response lifetime and send-result contracts; no TPU execution required."""

import asyncio
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorWorkerMetadata,
)
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.engine.core_client import DPLBAsyncMPClient
from vllm.v1.engine.detokenizer import BaseIncrementalDetokenizer
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.metrics.stats import PrefillStats
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import _patch_vllm_raiden_pd_completion
from vllm_torchtpu.core.raiden_pd import _ProducerCompletion
from vllm_torchtpu.distributed.kv_transfer.raiden_pd_completion import (
    RaidenSendOutcome,
    RaidenSendResults,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    TPURaidenConnector,
    TPURaidenConnectorScheduler,
)


@pytest.fixture(params=[Scheduler, AsyncScheduler])
def scheduler_cls(request):
    return request.param


@pytest.fixture(params=[RequestOutputKind.DELTA, RequestOutputKind.CUMULATIVE])
def output_kind(request):
    return request.param


def _scheduler(
    scheduler_cls,
    *,
    multi_connector=False,
    include_finished_set=False,
    output_kind=RequestOutputKind.DELTA,
):
    scheduler = scheduler_cls.__new__(scheduler_cls)
    connector = TPURaidenConnector.__new__(TPURaidenConnector)
    connector.connector_scheduler = TPURaidenConnectorScheduler.__new__(
        TPURaidenConnectorScheduler
    )
    connector.connector_scheduler._stage3_finished_sends = {}
    connector.connector_scheduler._stage3_sends_to_expire = {}
    connector.connector_scheduler.reqs_to_release = {}
    scheduler.connector = connector
    if multi_connector:
        scheduler.connector = MultiConnector.__new__(MultiConnector)
        scheduler.connector._connectors = [object(), connector]
    scheduler.finished_req_ids_dict = defaultdict(set) if include_finished_set else None
    scheduler.requests = {
        "p-request": Request(
            "p-request",
            [1, 2, 3],
            SamplingParams(max_tokens=1, output_kind=output_kind),
            None,
            client_index=3,
        )
    }
    scheduler._next_outputs = {}
    scheduler.update_from_output = lambda *_args: scheduler._next_outputs
    scheduler.finish_requests = lambda *_args: []
    return scheduler, _ProducerCompletion(scheduler)


def _step(scheduler, outputs=None, results=None, finished=()):
    kv_output = KVConnectorOutput(
        finished_sending=set(finished),
        kv_connector_worker_meta=MultiKVConnectorWorkerMetadata(
            metadata=(results, None)
        ),
    )
    scheduler._next_outputs = outputs if outputs is not None else {}
    runner_output = SimpleNamespace(
        kv_connector_output=kv_output,
        req_ids=[
            output.request_id
            for batch in scheduler._next_outputs.values()
            for output in batch.outputs
        ],
    )
    return scheduler.update_from_output(None, runner_output)


def _prefill_output():
    stats = PrefillStats(num_cached_tokens=16384)
    return EngineCoreOutput(
        request_id="p-request",
        new_token_ids=[42],
        finish_reason=FinishReason.LENGTH,
        prefill_stats=stats,
        kv_transfer_params={
            "req_id": "p-request",
            "uuid": 7,
            "kv_transfer_pending": True,
        },
    )


def _prefill(scheduler):
    outputs = _step(
        scheduler,
        {
            3: EngineCoreOutputs(
                outputs=[_prefill_output()],
                finished_requests=(
                    {"p-request"}
                    if scheduler.finished_req_ids_dict is not None
                    else None
                ),
            )
        },
    )
    return outputs[3].outputs[0]


class _NewlineDetokenizer(BaseIncrementalDetokenizer):
    def decode_next(self, next_token_id: int) -> str:
        return {271: "\n\n"}[next_token_id]


def _output_processor(output_kind, stop=None):
    _patch_vllm_raiden_pd_completion()
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    request = EngineCoreRequest(
        request_id="p-request",
        external_req_id="client-request",
        prompt_token_ids=[1] * 20000,
        mm_features=None,
        sampling_params=SamplingParams(
            max_tokens=1, stop=stop, output_kind=output_kind
        ),
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    processor.add_request(request, prompt=None)
    if stop:
        processor.request_states["p-request"].detokenizer = _NewlineDetokenizer(request)
    return processor


@pytest.mark.parametrize("stop", [None, "\n\n"])
def test_final_only_prefill_returns_handle_without_waiting_for_decode(
    scheduler_cls, stop
):
    scheduler, state = _scheduler(
        scheduler_cls,
        include_finished_set=True,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    output = _prefill(scheduler)
    assert output.finish_reason is FinishReason.LENGTH
    assert output.kv_transfer_params["uuid"] == 7
    assert scheduler._next_outputs[3].finished_requests == {"p-request"}
    assert state.pending_responses == {}
    processor = _output_processor(RequestOutputKind.FINAL_ONLY, stop)
    output.new_token_ids = [271]
    processed = processor.process_outputs([output], engine_core_timestamp=0)
    assert processed.request_outputs[0].finished
    assert processed.request_outputs[0].kv_transfer_params["uuid"] == 7
    assert processed.reqs_to_abort == []
    assert not processor.has_unfinished_requests()
    assert (
        _step(
            scheduler,
            results=RaidenSendResults({"p-request": RaidenSendOutcome.SUCCESS}),
            finished=["p-request"],
        )
        == {}
    )


@pytest.mark.parametrize("outcome", [*RaidenSendOutcome, None])
def test_dp_keeps_request_route_until_send_terminal(
    scheduler_cls, outcome, output_kind
):
    scheduler, _ = _scheduler(
        scheduler_cls, include_finished_set=True, output_kind=output_kind
    )
    engine = b"engine-0"
    client = SimpleNamespace(
        reqs_in_flight={"p-request": engine, "ordinary": engine},
        engine_inflight={engine: 2},
    )
    first = _step(
        scheduler,
        {
            3: EngineCoreOutputs(
                outputs=[_prefill_output()],
                finished_requests={"p-request", "ordinary"},
            ),
            5: EngineCoreOutputs(finished_requests={"other-client"}),
        },
    )
    asyncio.run(DPLBAsyncMPClient.process_engine_outputs(client, first[3]))
    assert client.reqs_in_flight == {"p-request": engine}
    assert client.engine_inflight == {engine: 1}
    assert first[5].finished_requests == {"other-client"}

    results = RaidenSendResults({"p-request": outcome}) if outcome else None
    if results is not None:
        assert _step(scheduler, results=results) == {}
    final = _step(scheduler, finished=["p-request"])
    assert final[3].finished_requests == {"p-request"}
    asyncio.run(DPLBAsyncMPClient.process_engine_outputs(client, final[3]))
    assert client.reqs_in_flight == {}
    assert client.engine_inflight == {engine: 0}
    assert _step(scheduler, results=results, finished=["p-request"]) == {}


def test_send_terminal_waits_for_aggregated_workers(scheduler_cls):
    scheduler, _ = _scheduler(scheduler_cls)
    _prefill(scheduler)
    scheduler._next_outputs = {}
    aggregator = KVOutputAggregator(2)
    for rank, outcome in enumerate(
        (RaidenSendOutcome.SEND_FAILURE, RaidenSendOutcome.SUCCESS)
    ):
        workers = [
            SimpleNamespace(kv_connector_output=KVConnectorOutput(), req_ids=[])
            for _ in range(2)
        ]
        workers[rank].kv_connector_output = KVConnectorOutput(
            finished_sending={"p-request"},
            kv_connector_worker_meta=RaidenSendResults({"p-request": outcome}),
        )
        outputs = scheduler.update_from_output(None, aggregator.aggregate(workers))
        if rank == 0:
            assert outputs == {}
        else:
            (terminal,) = outputs[3].outputs
            assert terminal.finish_reason is FinishReason.ERROR
            assert "kv_transfer_done" not in terminal.kv_transfer_params


def test_prefill_remains_open_until_success_without_replaying_tokens(scheduler_cls):
    scheduler, state = _scheduler(scheduler_cls)
    first = _prefill(scheduler)
    assert first.finish_reason is None
    assert first.new_token_ids == [42]
    assert first.prefill_stats.num_cached_tokens == 16384
    assert first.kv_transfer_params["uuid"] == 7

    # A subset of ranks finishes first. The existing aggregator, not this
    # scheduler, decides when the whole request reaches finished_sending.
    assert (
        _step(
            scheduler,
            results=RaidenSendResults({"p-request": RaidenSendOutcome.SUCCESS}),
        )
        == {}
    )
    final = _step(
        scheduler,
        results=RaidenSendResults({"p-request": RaidenSendOutcome.SUCCESS}),
        finished=["p-request"],
    )[3].outputs
    assert len(final) == 1
    assert final[0].new_token_ids == []
    assert final[0].finish_reason == FinishReason.STOP
    assert final[0].kv_transfer_params == {"kv_transfer_done": True}
    assert state.pending_responses == {}
    # A repeated/late terminal must not finish the response twice.
    assert _step(scheduler, finished=["p-request"]) == {}


@pytest.mark.parametrize(
    ("outcome", "status_name", "status_message"),
    [
        (
            RaidenSendOutcome.SEND_FAILURE,
            "model.KVTPrefillWriteFailure",
            "P-side failed to write KV cache data. Category: internal.",
        ),
        (
            RaidenSendOutcome.UNCLAIMED_TIMEOUT,
            "model.KVTPrefillTransferTimeout",
            "P-side timed out waiting for KV cache transfer completion.",
        ),
        (
            None,
            "model.KVTPrefillWriteFailure",
            "P-side KV cache transfer finished without a sender result. "
            "Category: internal.",
        ),
    ],
)
def test_failed_or_missing_result_has_structured_error(
    outcome, status_name, status_message, scheduler_cls
):
    scheduler, _ = _scheduler(scheduler_cls)
    _prefill(scheduler)
    if outcome is not None:
        # Failure must survive successful ranks arriving in later steps.
        _step(scheduler, results=RaidenSendResults({"p-request": outcome}))
        _step(
            scheduler,
            results=RaidenSendResults({"p-request": RaidenSendOutcome.SUCCESS}),
        )
    final = _step(scheduler, finished=["p-request"])[3].outputs[0]
    assert final.finish_reason == FinishReason.ERROR
    assert final.kv_transfer_params == {
        "kv_transfer_error": {
            "status_code": 500,
            "status_name": status_name,
            "status_message": status_message,
        }
    }


def test_rank_result_aggregation_preserves_failure_in_both_orders():
    for values in [
        (RaidenSendOutcome.SUCCESS, RaidenSendOutcome.SEND_FAILURE),
        (RaidenSendOutcome.SEND_FAILURE, RaidenSendOutcome.SUCCESS),
    ]:
        result = MultiKVConnectorWorkerMetadata(
            metadata=(RaidenSendResults({"p-request": values[0]}), None)
        )
        result = result.aggregate(
            MultiKVConnectorWorkerMetadata(
                metadata=(RaidenSendResults({"p-request": values[1]}), None)
            )
        )
        assert result.metadata[0].outcomes == {
            "p-request": RaidenSendOutcome.SEND_FAILURE
        }


def test_rank_result_aggregation_prefers_failure_over_timeout():
    result = RaidenSendResults({"p-request": RaidenSendOutcome.UNCLAIMED_TIMEOUT})
    result.aggregate(RaidenSendResults({"p-request": RaidenSendOutcome.SEND_FAILURE}))
    assert result.outcomes == {"p-request": RaidenSendOutcome.SEND_FAILURE}


@pytest.mark.parametrize("include_finished_set", [False, True])
@pytest.mark.parametrize("request_ids", ["p-request", ["p-request"], None])
def test_cancelled_response_does_not_emit_late_pkg2(
    scheduler_cls, include_finished_set, request_ids, output_kind
):
    scheduler, state = _scheduler(
        scheduler_cls,
        include_finished_set=include_finished_set,
        output_kind=output_kind,
    )
    _prefill(scheduler)
    scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)
    if include_finished_set:
        assert dict(scheduler.finished_req_ids_dict) == {3: {"p-request"}}
        # The base scheduler drains this queue into its next output batch.
        finished = scheduler.finished_req_ids_dict.pop(3)
        batch = _step(scheduler, {3: EngineCoreOutputs(finished_requests=finished)})[3]
        client = SimpleNamespace(
            reqs_in_flight={"p-request": b"engine-0"},
            engine_inflight={b"engine-0": 1},
        )
        asyncio.run(DPLBAsyncMPClient.process_engine_outputs(client, batch))
        assert client.engine_inflight == {b"engine-0": 0}
        scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)
        assert not scheduler.finished_req_ids_dict
    assert (
        _step(
            scheduler,
            results=RaidenSendResults({"p-request": RaidenSendOutcome.SUCCESS}),
            finished=["p-request"],
        )
        == {}
    )
    assert state.pending_responses == {}


@pytest.mark.parametrize("request_ids", [None, "p-request", ["p-request", "missing"]])
@pytest.mark.parametrize("multi_connector", [False, True])
@pytest.mark.parametrize("include_finished_set", [False, True])
def test_abort_reaches_model_finished_send(
    scheduler_cls, request_ids, multi_connector, include_finished_set
):
    scheduler, state = _scheduler(
        scheduler_cls,
        multi_connector=multi_connector,
        include_finished_set=include_finished_set,
    )
    _prefill(scheduler)
    scheduler.requests = {"p-request": SimpleNamespace(is_finished=lambda: True)}
    state.original_finish = Scheduler.finish_requests.__get__(scheduler)
    connector = (
        scheduler.connector._connectors[-1] if multi_connector else scheduler.connector
    )
    connector_scheduler = connector.connector_scheduler
    connector_scheduler._stage3_finished_sends["p-request"] = (7, (), 1, {})
    connector_scheduler.is_producer = True
    connector_scheduler.reqs_to_send = {}

    assert scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED) == []
    metadata = connector.build_connector_meta(None)
    assert metadata.reqs_to_expire == {"p-request": 7}
    assert connector.build_connector_meta(None).reqs_to_expire == {}
    assert state.pending_responses == {}
    assert "p-request" in scheduler.requests
    if include_finished_set:
        assert dict(scheduler.finished_req_ids_dict) == {3: {"p-request"}}


def test_abort_keeps_model_running_registration_on_upstream_path(scheduler_cls):
    scheduler, state = _scheduler(scheduler_cls)
    connector_scheduler = scheduler.connector.connector_scheduler
    connector_scheduler._stage3_finished_sends["running"] = (8, (), 1, {})
    original_calls = []
    state.original_finish = (
        lambda ids, status: original_calls.append((set(ids), status)) or []
    )

    scheduler.finish_requests(["running"], RequestStatus.FINISHED_ABORTED)

    assert connector_scheduler._stage3_sends_to_expire == {}
    assert original_calls == [({"running"}, RequestStatus.FINISHED_ABORTED)]


def test_dp_cancel_keeps_kv_until_native_send_completion(scheduler_cls, output_kind):
    scheduler, state = _scheduler(
        scheduler_cls, include_finished_set=True, output_kind=output_kind
    )
    req = Request(
        "p-request",
        [1, 2, 3],
        SamplingParams(max_tokens=1, output_kind=output_kind),
        None,
    )
    req.client_index = 3
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    scheduler.requests = {req.request_id: req}
    scheduler.finished_req_ids = {req.request_id}
    scheduler.connector = None
    scheduler.defer_block_free = False
    scheduler.kv_cache_manager = MagicMock()
    state.original_finish = Scheduler.finish_requests.__get__(scheduler)
    _prefill(scheduler)

    assert scheduler.finish_requests("p-request", RequestStatus.FINISHED_ABORTED) == []
    assert dict(scheduler.finished_req_ids_dict) == {3: {"p-request"}}
    assert scheduler.finished_req_ids == {"p-request"}
    assert scheduler.requests == {"p-request": req}
    assert req.status is RequestStatus.FINISHED_LENGTH_CAPPED
    scheduler.kv_cache_manager.free.assert_not_called()

    # The existing native barrier still owns delayed block reclamation.
    kv_output = KVConnectorOutput(finished_sending={"p-request"})
    scheduler._update_from_kv_xfer_finished(kv_output)
    scheduler.kv_cache_manager.free.assert_called_once_with(req)
    assert scheduler.requests == {}
    assert (
        _step(
            scheduler,
            results=RaidenSendResults({"p-request": RaidenSendOutcome.SUCCESS}),
            finished=["p-request"],
        )
        == {}
    )


def test_cancelled_model_output_can_reference_an_already_removed_request(scheduler_cls):
    scheduler, state = _scheduler(scheduler_cls)
    _prefill(scheduler)
    scheduler.finish_requests("p-request", RequestStatus.FINISHED_ABORTED)
    scheduler.requests.clear()
    scheduler._next_outputs = {}
    result = scheduler.update_from_output(
        None,
        SimpleNamespace(req_ids=["p-request"], kv_connector_output=None),
    )
    assert result == {}
    assert state.pending_responses == {}


@pytest.mark.parametrize("include_finished_set", [False, True])
def test_prefill_and_send_completion_in_same_step(scheduler_cls, include_finished_set):
    scheduler, state = _scheduler(
        scheduler_cls, include_finished_set=include_finished_set
    )
    first = EngineCoreOutput(
        request_id="p-request",
        new_token_ids=[42],
        finish_reason=FinishReason.LENGTH,
        kv_transfer_params={"kv_transfer_pending": True},
    )

    def base_update(*_args):
        # Native completion can reclaim the Request before the adapter runs.
        scheduler.requests.pop("p-request")
        return scheduler._next_outputs

    state.original_update = base_update
    outputs = _step(
        scheduler,
        {
            3: EngineCoreOutputs(
                outputs=[first],
                finished_requests={"p-request"} if include_finished_set else None,
            )
        },
        results=RaidenSendResults({"p-request": RaidenSendOutcome.SUCCESS}),
        finished=["p-request"],
    )[3].outputs
    assert len(outputs) == 2
    assert outputs[0].new_token_ids == [42]
    assert outputs[0].finish_reason is None
    assert outputs[1].new_token_ids == []
    assert outputs[1].finish_reason == FinishReason.STOP
    assert outputs[1].kv_transfer_params == {"kv_transfer_done": True}
    assert scheduler._next_outputs[3].finished_requests == (
        {"p-request"} if include_finished_set else None
    )
    assert state.pending_responses == {}


@pytest.mark.parametrize(
    ("stop", "pending"), [(None, True), ("\n\n", True), ("\n\n", False)]
)
def test_output_processor_keeps_cached_stats_until_pkg2(
    scheduler_cls, output_kind, stop, pending
):
    # Regression: finishing pkg1 deletes OutputProcessor's state, so a later
    # kv_transfer_done is silently dropped. Exercise the real processor.
    processor = _output_processor(output_kind, stop)
    scheduler, _ = _scheduler(scheduler_cls, output_kind=output_kind)
    prefill = _prefill(scheduler)
    prefill.new_token_ids = [271]
    if not pending:
        prefill.kv_transfer_params.pop("kv_transfer_pending")
    processed = processor.process_outputs([prefill], engine_core_timestamp=0)
    first = processed.request_outputs[0]
    if not pending:
        assert first.finished
        assert first.outputs[0].stop_reason == stop
        assert processed.reqs_to_abort == ["p-request"]
        assert not processor.has_unfinished_requests()
        return
    assert not first.finished
    assert first.outputs[0].token_ids == [271]
    assert first.outputs[0].finish_reason is None
    assert first.outputs[0].stop_reason is None
    assert processed.reqs_to_abort == []
    assert first.kv_transfer_params["uuid"] == 7
    assert first.kv_transfer_params["kv_transfer_pending"]
    assert first.num_cached_tokens == 16384
    assert processor.has_unfinished_requests()
    final_outputs = _step(
        scheduler,
        results=RaidenSendResults({"p-request": RaidenSendOutcome.SUCCESS}),
        finished=["p-request"],
    )[3].outputs
    assert len(final_outputs) == 1
    assert final_outputs[0].new_token_ids == []
    final = processor.process_outputs(
        final_outputs, engine_core_timestamp=0
    ).request_outputs[0]
    assert final.request_id == "client-request"
    assert final.finished
    assert final.outputs[0].token_ids == (
        [] if output_kind is RequestOutputKind.DELTA else [271]
    )
    assert final.num_cached_tokens == 16384
    assert final.kv_transfer_params == {"kv_transfer_done": True}
    assert not processor.has_unfinished_requests()


def test_output_processor_keeps_structured_error_after_prefill_output(
    scheduler_cls, output_kind
):
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    processor.add_request(
        EngineCoreRequest(
            request_id="p-request",
            external_req_id="client-request",
            prompt_token_ids=[1] * 20000,
            mm_features=None,
            sampling_params=SamplingParams(
                max_tokens=1, detokenize=False, output_kind=output_kind
            ),
            pooling_params=None,
            arrival_time=0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        ),
        prompt=None,
    )
    scheduler, _ = _scheduler(scheduler_cls, output_kind=output_kind)
    first = processor.process_outputs(
        [_prefill(scheduler)], engine_core_timestamp=0
    ).request_outputs[0]
    assert not first.finished

    terminal_outputs = _step(
        scheduler,
        results=RaidenSendResults({"p-request": RaidenSendOutcome.SEND_FAILURE}),
        finished=["p-request"],
    )[3].outputs
    terminal = processor.process_outputs(
        terminal_outputs, engine_core_timestamp=0
    ).request_outputs[0]

    assert terminal.finished
    assert terminal.outputs[0].token_ids == (
        [] if output_kind is RequestOutputKind.DELTA else [42]
    )
    assert terminal.kv_transfer_params == {
        "kv_transfer_error": {
            "status_code": 500,
            "status_name": "model.KVTPrefillWriteFailure",
            "status_message": (
                "P-side failed to write KV cache data. Category: internal."
            ),
        }
    }
    assert not processor.has_unfinished_requests()

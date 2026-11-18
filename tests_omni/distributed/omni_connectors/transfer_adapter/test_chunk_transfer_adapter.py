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
"""TPU tests for OmniChunkTransferAdapter async streaming and stream windows."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import Request, RequestStatus
from vllm_omni.data_entry_keys import (
    CodesStruct,
    HiddenStatesStruct,
    MetaStruct,
    OmniPayloadStruct,
)
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (  # noqa: E501
    OmniChunkTransferAdapter,
)
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S
from tests_omni.tiny_models import tiny_opt
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = [
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

CHUNK_READY_TIMEOUT_S = 5.0


def _make_chunk_request(
    req_id: str,
    *,
    external_req_id: str | None = None,
    status: RequestStatus = RequestStatus.WAITING,
) -> Request:
    """Create a minimal vLLM Request for chunk transfer adapter tests."""
    req = Request(
        request_id=req_id,
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
    )
    req.status = status
    req.external_req_id = external_req_id or req_id  # type: ignore[attr-defined]
    req.additional_information = None  # type: ignore[attr-defined]
    return req


def _make_adapter(
    *,
    stage_id: int,
    worker_type: str,
    role: str,
    wakeup_scope: str,
    active_stream_window: int = 0,
    max_num_seqs: int = 4,
    custom_process_fn: str | None = None,
) -> OmniChunkTransferAdapter:
    """Construct a real OmniChunkTransferAdapter backed by SharedMemoryConnector."""
    model_config = SimpleNamespace(
        stage_id=stage_id,
        async_chunk=True,
        worker_type=worker_type,
        active_stream_window=active_stream_window,
        max_num_seqs=max_num_seqs,
        custom_process_next_stage_input_func=custom_process_fn,
        stage_connector_config={
            "name": "SharedMemoryConnector",
            "extra": {"role": role, "wakeup_scope": wakeup_scope},
        },
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )
    return OmniChunkTransferAdapter(vllm_config)


def _poll_until_chunk_ready(
    adapter: OmniChunkTransferAdapter,
    waiting_queue: object,
    running_queue: list[Request],
    req: Request,
) -> None:
    """Drive process_pending_chunks until ``req`` transitions back to WAITING."""
    deadline = time.monotonic() + CHUNK_READY_TIMEOUT_S
    while time.monotonic() < deadline:
        adapter.process_pending_chunks(
            waiting_queue,
            running_queue,
            scheduler_requests={req.request_id: req},
        )
        adapter.restore_queues(
            waiting_queue,
            running_queue,
            scheduler_requests={req.request_id: req},
        )
        if (
            req.status == RequestStatus.WAITING
            and req.request_id in adapter.requests_with_ready_chunks
        ):
            return
        time.sleep(0.01)
    raise AssertionError(
        f"No chunk ready for {req.request_id} after {CHUNK_READY_TIMEOUT_S}s "
        f"(status={req.status})"
    )


def _load_window_stage_cfg(tmp_path: Path, *, active_stream_window: int) -> dict:
    """Write a 1-stage deploy YAML with the window and parse its stage back."""
    deploy_yaml = tmp_path / "stage_window.yaml"
    deploy_yaml.write_text(
        yaml.safe_dump(
            {
                "stages": [
                    {
                        "stage_id": 1,
                        "max_num_seqs": 4,
                        "active_stream_window": active_stream_window,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return yaml.safe_load(deploy_yaml.read_text(encoding="utf-8"))["stages"][0]


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="TPUWorker fails validate_worker_omni_connector; tracked by PR #2065",
)
def test_async_chunk_streaming() -> None:
    """Async Chunk Streaming (OmniChunkTransferAdapter).

    - TPU path: Allocate bf16/int64 chunk tensors on ``torch.device("tpu")``,
      validate AR sender and generation receiver ``OmniEngineArgs`` with
      ``async_chunk=True`` on ``OmniTpuPlatform``, and stream 2 chunks across
      ``OmniChunkTransferAdapter`` background threads over
      ``SharedMemoryConnector``.
    - Setup check: Assert ``current_omni_platform`` is ``OmniTpuPlatform``,
      chunk tensors reside on ``device.type == "tpu"``, and both stage engine
      args pass ``validate_worker_omni_connector``.
    - Outcome check: Drive ``process_pending_chunks`` across
      ``WAITING -> WAITING_FOR_CHUNK -> WAITING`` for non-terminal and terminal
      chunks, asserting exact ``torch.equal`` parity, chunk count 2, and
      ``is_done_receiving_chunks``.
    """
    assert isinstance(current_omni_platform, OmniTpuPlatform), (
        f"Expected OmniTpuPlatform, got {type(current_omni_platform).__name__}"
    )

    tpu_device = torch.device("tpu")
    cpu_codes_0 = torch.tensor([[10, 20, 30]], dtype=torch.int64)
    cpu_hidden_0 = torch.tensor([[1.0, -0.5, 2.5]], dtype=torch.bfloat16)
    cpu_codes_1 = torch.tensor([[40, 50, 60]], dtype=torch.int64)
    cpu_hidden_1 = torch.tensor([[-1.5, 3.0, 0.25]], dtype=torch.bfloat16)

    tpu_codes_0 = cpu_codes_0.to(tpu_device)
    tpu_hidden_0 = cpu_hidden_0.to(tpu_device)
    tpu_codes_1 = cpu_codes_1.to(tpu_device)
    tpu_hidden_1 = cpu_hidden_1.to(tpu_device)
    for name, tensor in (
        ("codes_0", tpu_codes_0),
        ("hidden_0", tpu_hidden_0),
        ("codes_1", tpu_codes_1),
        ("hidden_1", tpu_hidden_1),
    ):
        assert tensor.device.type == "tpu", (
            f"{name} expected on tpu, got {tensor.device}"
        )

    opt_dir = tiny_opt()
    OmniEngineArgs(
        model=opt_dir,
        worker_type="ar",
        async_chunk=True,
        stage_connector_spec={
            "name": "SharedMemoryConnector",
            "extra": {"role": "sender"},
        },
    )
    OmniEngineArgs(
        model=opt_dir,
        worker_type="generation",
        async_chunk=True,
        stage_connector_spec={
            "name": "SharedMemoryConnector",
            "extra": {"role": "receiver"},
        },
    )

    scope = f"tpu_chunk_{uuid.uuid4().hex}"
    sender = _make_adapter(
        stage_id=0, worker_type="ar", role="sender", wakeup_scope=scope
    )
    receiver = _make_adapter(
        stage_id=1, worker_type="generation", role="receiver", wakeup_scope=scope
    )
    try:
        req_id = f"req-chunk-{uuid.uuid4().hex[:8]}"
        send_req = _make_chunk_request(
            req_id, external_req_id=req_id, status=RequestStatus.RUNNING
        )
        recv_req = _make_chunk_request(
            req_id, external_req_id=req_id, status=RequestStatus.WAITING
        )

        chunks = [
            OmniPayloadStruct(
                codes=CodesStruct(audio=tpu_codes_0),
                hidden_states=HiddenStatesStruct(output=tpu_hidden_0),
                meta=MetaStruct(finished=torch.tensor(False, dtype=torch.bool)),
            ),
            OmniPayloadStruct(
                codes=CodesStruct(audio=tpu_codes_1),
                hidden_states=HiddenStatesStruct(output=tpu_hidden_1),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
            ),
        ]
        sender.custom_process_next_stage_input_func = (
            lambda *_args, chunk_idx=0, **_kwargs: chunks[chunk_idx]
        )

        waiting_queue = create_request_queue(SchedulingPolicy.FCFS)
        running_queue: list[Request] = []
        waiting_queue.add_request(recv_req)

        sender.custom_process_next_stage_input_func = lambda *_args, **_kwargs: chunks[
            0
        ]
        sender.save_async(multimodal_output={"audio": tpu_codes_0}, request=send_req)

        receiver.process_pending_chunks(
            waiting_queue, running_queue, scheduler_requests={req_id: recv_req}
        )
        assert recv_req.status == RequestStatus.WAITING_FOR_CHUNK, (
            f"Expected WAITING_FOR_CHUNK after first poll, got {recv_req.status}"
        )
        receiver.restore_queues(
            waiting_queue, running_queue, scheduler_requests={req_id: recv_req}
        )
        _poll_until_chunk_ready(receiver, waiting_queue, running_queue, recv_req)
        assert torch.equal(
            recv_req.additional_information["codes"]["audio"],  # type: ignore[index]
            cpu_codes_0,
        ), "Chunk 0 decoded audio codes do not match CPU reference"
        assert torch.equal(
            recv_req.additional_information["hidden_states"]["output"],  # type: ignore[index]
            cpu_hidden_0,
        ), "Chunk 0 hidden_states do not match CPU reference"

        receiver.requests_with_ready_chunks.discard(req_id)
        sender.custom_process_next_stage_input_func = lambda *_args, **_kwargs: chunks[
            1
        ]
        send_req.status = RequestStatus.FINISHED_STOPPED
        sender.save_async(
            multimodal_output={"audio": tpu_codes_1},
            request=send_req,
            is_segment_finished=True,
        )

        _poll_until_chunk_ready(receiver, waiting_queue, running_queue, recv_req)
        assert receiver.get_req_chunk[req_id] == 2, (
            f"Expected 2 chunks received, got {receiver.get_req_chunk[req_id]}"
        )
        assert receiver.is_done_receiving_chunks(req_id) is True, (
            "Receiver did not mark stream done after terminal chunk"
        )
        assert torch.equal(
            recv_req.additional_information["codes"]["audio"],  # type: ignore[index]
            cpu_codes_1,
        ), "Chunk 1 decoded audio codes do not match CPU reference"
        assert torch.equal(
            recv_req.additional_information["hidden_states"]["output"],  # type: ignore[index]
            cpu_hidden_1,
        ), "Chunk 1 hidden_states do not match CPU reference"
    finally:
        sender.shutdown()
        receiver.shutdown()


def test_active_stream_window(tmp_path: Path) -> None:
    """Active Stream Window: K=2 admits the first FIFO streams.

    - TPU path: Parse ``active_stream_window: 2`` from deploy YAML into stage
      ``model_config`` with ``OmniTpuPlatform`` active and run
      ``OmniChunkTransferAdapter.process_pending_chunks`` with 4 waiting
      requests.
    - Setup check: Assert ``current_omni_platform`` is ``OmniTpuPlatform``,
      ``model_config.active_stream_window`` matches the YAML value, and
      ``adapter._active_window`` equals ``min(2, max_num_seqs)``.
    - Outcome check: Assert only the first 2 FIFO requests enter
      ``_active_streams`` and ``cleanup_receiver`` + ``restore_queues``
      promotes the next waiting stream.
    """
    assert isinstance(current_omni_platform, OmniTpuPlatform), (
        f"Expected OmniTpuPlatform, got {type(current_omni_platform).__name__}"
    )

    stage_cfg = _load_window_stage_cfg(tmp_path, active_stream_window=2)
    assert stage_cfg["active_stream_window"] == 2, (
        f"YAML active_stream_window mismatch: {stage_cfg}"
    )

    adapter = _make_adapter(
        stage_id=1,
        worker_type="generation",
        role="receiver",
        wakeup_scope=f"tpu_window_{uuid.uuid4().hex}",
        active_stream_window=stage_cfg["active_stream_window"],
        max_num_seqs=stage_cfg["max_num_seqs"],
    )
    try:
        assert adapter._active_window == 2, (
            f"_active_window={adapter._active_window} != 2"
        )

        reqs = [_make_chunk_request(f"req-{i}") for i in range(4)]
        sched_reqs = {r.request_id: r for r in reqs}
        waiting_queue = create_request_queue(SchedulingPolicy.FCFS)
        running_queue: list[Request] = []
        for req in reqs:
            waiting_queue.add_request(req)
        adapter.process_pending_chunks(
            waiting_queue, running_queue, scheduler_requests=sched_reqs
        )
        assert list(adapter._active_streams.keys()) == ["req-0", "req-1"], (
            f"Expected ['req-0', 'req-1'], got {list(adapter._active_streams)}"
        )
        assert len(waiting_queue) == 0, (
            f"Expected waiting requests parked, got {len(waiting_queue)}"
        )

        del sched_reqs["req-0"]
        adapter.cleanup_receiver("req-0")
        adapter.restore_queues(
            waiting_queue, running_queue, scheduler_requests=sched_reqs
        )
        adapter.process_pending_chunks(
            waiting_queue, running_queue, scheduler_requests=sched_reqs
        )
        assert list(adapter._active_streams.keys()) == ["req-1", "req-2"], (
            f"Expected ['req-1', 'req-2'], got {list(adapter._active_streams)}"
        )
    finally:
        adapter.shutdown()


def test_active_stream_window_disabled(tmp_path: Path) -> None:
    """Active Stream Window: K=0 disables the window.

    - TPU path: Parse ``active_stream_window: 0`` from deploy YAML into stage
      ``model_config`` with ``OmniTpuPlatform`` active and run
      ``OmniChunkTransferAdapter.process_pending_chunks`` with 4 running
      requests.
    - Setup check: Assert ``current_omni_platform`` is ``OmniTpuPlatform``,
      ``model_config.active_stream_window`` matches the YAML value, and
      ``adapter._active_window`` is ``0``.
    - Outcome check: Assert ``_active_streams`` stays empty and all running
      requests preempt.
    """
    assert isinstance(current_omni_platform, OmniTpuPlatform), (
        f"Expected OmniTpuPlatform, got {type(current_omni_platform).__name__}"
    )

    stage_cfg = _load_window_stage_cfg(tmp_path, active_stream_window=0)
    assert stage_cfg["active_stream_window"] == 0, (
        f"YAML active_stream_window mismatch: {stage_cfg}"
    )

    adapter = _make_adapter(
        stage_id=1,
        worker_type="generation",
        role="receiver",
        wakeup_scope=f"tpu_window_{uuid.uuid4().hex}",
        active_stream_window=stage_cfg["active_stream_window"],
        max_num_seqs=stage_cfg["max_num_seqs"],
    )
    try:
        assert adapter._active_window == 0, (
            f"_active_window={adapter._active_window} != 0"
        )

        reqs = [_make_chunk_request(f"req-{i}") for i in range(4)]
        sched_reqs = {r.request_id: r for r in reqs}
        waiting_queue = create_request_queue(SchedulingPolicy.FCFS)
        running_queue: list[Request] = []
        for req in reqs:
            req.status = RequestStatus.RUNNING
            running_queue.append(req)
        adapter.process_pending_chunks(
            waiting_queue, running_queue, scheduler_requests=sched_reqs
        )
        assert adapter._active_streams == {}, (
            f"Expected empty _active_streams for K=0, got {adapter._active_streams}"
        )
        assert len(running_queue) == 0, (
            f"Expected running requests preempted, got {len(running_queue)}"
        )
        assert len(adapter.waiting_for_chunk_running_requests) == 4, (
            "Expected 4 preempted running requests, got "
            f"{len(adapter.waiting_for_chunk_running_requests)}"
        )
    finally:
        adapter.shutdown()

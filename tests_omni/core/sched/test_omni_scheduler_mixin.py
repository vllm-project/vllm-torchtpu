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
"""TPU tests for OmniSchedulerMixin stage-input wait timeout handling."""

from __future__ import annotations

import importlib
import time
from collections import defaultdict, deque
from types import SimpleNamespace

import pytest
from vllm import SamplingParams
from vllm.v1.engine import EngineCoreOutputs, FinishReason
from vllm.v1.request import Request, RequestStatus
from vllm_omni.core.sched import omni_scheduler_mixin
from vllm_omni.core.sched.omni_scheduling_coordinator import (
    OmniSchedulingCoordinator,
)
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (  # noqa: E501
    OmniChunkTransferAdapter,
)
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]


class _TimeoutScheduler(omni_scheduler_mixin.OmniSchedulerMixin):
    """Minimal OmniSchedulerMixin subclass for timeout-hook verification."""

    def __init__(
        self,
        requests: dict[str, Request],
        coordinator: OmniSchedulingCoordinator | None,
        adapter: OmniChunkTransferAdapter | None,
    ) -> None:
        self.requests = requests
        self.input_coordinator = coordinator
        self.chunk_transfer_adapter = adapter
        self.finished_req_ids_dict: dict[int, set[str]] = defaultdict(set)
        self.finish_calls: list[tuple[set[str], RequestStatus]] = []

    def finish_requests(
        self, request_ids: set[str], finished_status: RequestStatus
    ) -> None:
        self.finish_calls.append((set(request_ids), finished_status))
        for req_id in request_ids:
            req = self.requests.pop(req_id, None)
            if req is not None:
                req.status = finished_status
                self.finished_req_ids_dict[req.client_index].add(req_id)


UNSET_ENV_TIMEOUT_S = 600.0
TIMED_OUT_FINISH_CALLS = [
    ({"req-wait-input"}, RequestStatus.FINISHED_ERROR),
    ({"req-wait-chunk"}, RequestStatus.FINISHED_ERROR),
]
TIMED_OUT_STATUSES = (RequestStatus.FINISHED_ERROR, RequestStatus.FINISHED_ERROR)
PARKED_STATUSES = (
    RequestStatus.WAITING_FOR_INPUT,  # type: ignore[attr-defined]
    RequestStatus.WAITING_FOR_CHUNK,  # type: ignore[attr-defined]
)
TIMED_OUT_FINISH_REASONS = {0: FinishReason.ERROR, 1: FinishReason.ERROR}


@pytest.mark.parametrize(
    (
        "env_value",
        "expected_timeout_s",
        "expected_finish_calls",
        "expected_statuses",
        "expected_finish_reasons",
    ),
    [
        pytest.param(
            "600",
            600.0,
            TIMED_OUT_FINISH_CALLS,
            TIMED_OUT_STATUSES,
            TIMED_OUT_FINISH_REASONS,
            id="600",
        ),
        pytest.param(
            "5.0",
            5.0,
            TIMED_OUT_FINISH_CALLS,
            TIMED_OUT_STATUSES,
            TIMED_OUT_FINISH_REASONS,
            id="5.0",
        ),
        pytest.param("0", 0.0, [], PARKED_STATUSES, {}, id="0"),
    ],
)
def test_stage_input_wait_timeout(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
    expected_timeout_s: float,
    expected_finish_calls: list[tuple[set[str], RequestStatus]],
    expected_statuses: tuple[RequestStatus, RequestStatus],
    expected_finish_reasons: dict[int, FinishReason],
) -> None:
    """Stage Input Wait Timeout.

    - TPU path: Reload ``omni_scheduler_mixin`` under parametrized
      ``VLLM_OMNI_INPUT_WAIT_TIMEOUT_S`` values with ``OmniTpuPlatform``
      active and run ``_process_pending_input_timeouts`` and
      ``_process_pending_chunk_timeouts`` on stalled requests.
    - Setup check: Assert ``current_omni_platform`` is ``OmniTpuPlatform``, an
      unset env gives the 600 s default, and ``DEFAULT_INPUT_WAIT_TIMEOUT_S``
      matches ``expected_timeout_s`` (restored on teardown).
    - Outcome check: Assert stalled requests transition to ``FINISHED_ERROR``
      and emit ``FinishReason.ERROR`` when ``timeout > 0`` and remain parked
      when ``timeout == 0``.
    """
    assert isinstance(current_omni_platform, OmniTpuPlatform), (
        f"Expected OmniTpuPlatform, got {type(current_omni_platform).__name__}"
    )

    try:
        monkeypatch.delenv("VLLM_OMNI_INPUT_WAIT_TIMEOUT_S", raising=False)
        unset = importlib.reload(omni_scheduler_mixin)
        assert unset.DEFAULT_INPUT_WAIT_TIMEOUT_S == UNSET_ENV_TIMEOUT_S, (
            f"env {env_value!r}: unset-env DEFAULT_INPUT_WAIT_TIMEOUT_S="
            f"{unset.DEFAULT_INPUT_WAIT_TIMEOUT_S} != {UNSET_ENV_TIMEOUT_S}"
        )

        monkeypatch.setenv("VLLM_OMNI_INPUT_WAIT_TIMEOUT_S", env_value)
        reloaded = importlib.reload(omni_scheduler_mixin)
        assert expected_timeout_s == reloaded.DEFAULT_INPUT_WAIT_TIMEOUT_S, (
            f"DEFAULT_INPUT_WAIT_TIMEOUT_S={reloaded.DEFAULT_INPUT_WAIT_TIMEOUT_S} "
            f"!= {expected_timeout_s}"
        )

        wait_age_s = max(expected_timeout_s + 10.0, 10.0)
        input_req = Request(
            request_id="req-wait-input",
            prompt_token_ids=[1],
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            client_index=0,
        )
        input_req.status = RequestStatus.WAITING_FOR_INPUT  # type: ignore[attr-defined]
        chunk_req = Request(
            request_id="req-wait-chunk",
            prompt_token_ids=[1],
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            client_index=1,
        )
        chunk_req.status = RequestStatus.WAITING_FOR_CHUNK  # type: ignore[attr-defined]

        coord = object.__new__(OmniSchedulingCoordinator)
        coord._stage_id = 1
        coord._waiting_since = {"req-wait-input": time.monotonic() - wait_age_s}
        coord._waiting_for_chunk_waiting = deque()
        coord._waiting_for_chunk_running = deque()
        coord._waiting_for_input = deque()

        adapter = object.__new__(OmniChunkTransferAdapter)
        adapter.receives_chunks = True
        adapter._waiting_since = {"req-wait-chunk": time.monotonic() - wait_age_s}
        adapter.connector = SimpleNamespace(stage_id=1)

        scheduler = _TimeoutScheduler(
            {"req-wait-input": input_req, "req-wait-chunk": chunk_req},
            coord,
            adapter,
        )
        scheduler._process_pending_input_timeouts()
        scheduler._process_pending_chunk_timeouts()

        assert scheduler.finish_calls == expected_finish_calls, (
            f"timeout={expected_timeout_s}: finish_requests calls "
            f"{scheduler.finish_calls} != {expected_finish_calls}"
        )
        assert input_req.status == expected_statuses[0], (
            f"timeout={expected_timeout_s}: input request status "
            f"{input_req.status} != {expected_statuses[0]}"
        )
        assert chunk_req.status == expected_statuses[1], (
            f"timeout={expected_timeout_s}: chunk request status "
            f"{chunk_req.status} != {expected_statuses[1]}"
        )

        outputs: dict[int, EngineCoreOutputs] = {}
        scheduler._attach_finished_request_sets(outputs, synthesize_abort_outputs=False)
        finish_reasons = {
            client: output.outputs[0].finish_reason
            for client, output in outputs.items()
        }
        assert finish_reasons == expected_finish_reasons, (
            f"timeout={expected_timeout_s}: per-client finish_reason "
            f"{finish_reasons} != {expected_finish_reasons}"
        )
    finally:
        monkeypatch.undo()
        importlib.reload(omni_scheduler_mixin)


@pytest.mark.parametrize("invalid", ["-1", "nan", "inf"])
def test_stage_input_wait_timeout_invalid_env(
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    """Stage Input Wait Timeout: invalid env values.

    - TPU path: Reload ``omni_scheduler_mixin`` with ``OmniTpuPlatform`` active
      and a negative or non-finite ``VLLM_OMNI_INPUT_WAIT_TIMEOUT_S``.
    - Setup check: Assert ``current_omni_platform`` is ``OmniTpuPlatform``.
    - Outcome check: the reload raises ``ValueError`` (module restored on
      teardown).
    """
    assert isinstance(current_omni_platform, OmniTpuPlatform), (
        f"Expected OmniTpuPlatform, got {type(current_omni_platform).__name__}"
    )

    try:
        monkeypatch.setenv("VLLM_OMNI_INPUT_WAIT_TIMEOUT_S", invalid)
        with pytest.raises(ValueError):
            importlib.reload(omni_scheduler_mixin)
    finally:
        monkeypatch.undo()
        importlib.reload(omni_scheduler_mixin)

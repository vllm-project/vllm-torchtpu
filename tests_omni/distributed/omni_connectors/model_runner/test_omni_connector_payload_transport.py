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
"""TPU tests for OmniConnector payload transport receive polling interval."""

from __future__ import annotations

import threading

import pytest
import vllm_omni.distributed.omni_connectors.model_runner.omni_connector_payload_transport as transport_mod  # noqa: E501
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import (
    SharedMemoryConnector,
)
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]


class _WaitRecorder:
    """Record ``threading.Event.wait`` timeouts and stop the loop after one call."""

    def __init__(self, stop_event: threading.Event) -> None:
        self.timeouts: list[float | None] = []
        self._stop_event = stop_event

    def wait(self, timeout: float | None = None) -> bool:
        self.timeouts.append(timeout)
        self._stop_event.set()
        return True

    def clear(self) -> None:
        return None


UNSET_ENV_POLL_S = 0.005
RECV_POLL_ENV_CASES = [
    ("20", 0.02),
    ("2.5", 0.0025),
    ("bad", 0.005),
    ("nan", 0.005),
    ("inf", 0.005),
    ("0", 0.005),
    ("-5", 0.005),
]


def _pending_poll_mixin() -> transport_mod._OmniConnectorPayloadTransportMixin:
    """Transport mixin with one pending request that never makes progress."""
    mixin = object.__new__(transport_mod._OmniConnectorPayloadTransportMixin)
    mixin._stop_event = threading.Event()
    mixin._lock = threading.Lock()
    mixin._stage_id = "1"
    mixin._pending_load_reqs = {"req-poll-0": object()}
    mixin._poll_pending_requests_once = lambda _pending_ids: False  # type: ignore[assignment]
    return mixin


@pytest.mark.parametrize(("env_value", "expected_s"), RECV_POLL_ENV_CASES)
def test_connector_receive_polling_interval(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
    expected_s: float,
) -> None:
    """Connector Receive Polling Interval: wakeup-capable connector.

    - TPU path: Resolve ``VLLM_OMNI_CONNECTOR_RECV_POLL_MS`` with
      ``OmniTpuPlatform`` active and run one non-progress iteration of
      ``_OmniConnectorPayloadTransportMixin._recv_loop`` on a
      ``SharedMemoryConnector``.
    - Setup check: Assert ``current_omni_platform`` is ``OmniTpuPlatform``, an
      unset env resolves to 5 ms, and rebind ``_RECV_POLL_S`` on the transport
      module.
    - Outcome check: Assert ``_recv_poll_seconds()`` returns ``expected_s`` and
      ``_recv_loop`` passes ``expected_s`` to ``wait_for_change``.
    """
    assert isinstance(current_omni_platform, OmniTpuPlatform), (
        f"Expected OmniTpuPlatform, got {type(current_omni_platform).__name__}"
    )

    monkeypatch.delenv("VLLM_OMNI_CONNECTOR_RECV_POLL_MS", raising=False)
    unset_s = transport_mod._recv_poll_seconds()
    assert unset_s == pytest.approx(UNSET_ENV_POLL_S), (
        f"env unset: _recv_poll_seconds()={unset_s} != {UNSET_ENV_POLL_S}"
    )
    monkeypatch.setenv("VLLM_OMNI_CONNECTOR_RECV_POLL_MS", env_value)
    resolved_s = transport_mod._recv_poll_seconds()
    assert resolved_s == pytest.approx(expected_s), (
        f"env={env_value!r}: _recv_poll_seconds()={resolved_s} != {expected_s}"
    )

    monkeypatch.setattr(transport_mod, "_RECV_POLL_S", resolved_s)

    mixin = _pending_poll_mixin()
    wait_timeouts: list[float | None] = []
    connector = SharedMemoryConnector(
        {"stage_id": 1, "extra": {"wakeup_scope": "poll-test"}}
    )

    def _record_wait_for_change(_gen: int, timeout: float | None = None) -> int:
        wait_timeouts.append(timeout)
        mixin._stop_event.set()
        return 0

    monkeypatch.setattr(connector, "wait_for_change", _record_wait_for_change)
    mixin._omni_connector = connector
    mixin._work_available = threading.Event()

    try:
        mixin._recv_loop()
    finally:
        connector.close()

    assert len(wait_timeouts) == 1, (
        f"env={env_value!r}: expected 1 wait_for_change call in _recv_loop, "
        f"got {len(wait_timeouts)}"
    )
    assert wait_timeouts[0] == pytest.approx(expected_s), (
        f"env={env_value!r}: _recv_loop passed timeout={wait_timeouts[0]} "
        f"!= expected {expected_s}"
    )


@pytest.mark.parametrize(("env_value", "expected_s"), RECV_POLL_ENV_CASES)
def test_connector_receive_polling_interval_no_wakeup(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
    expected_s: float,
) -> None:
    """Connector Receive Polling Interval: connector without wakeups.

    - TPU path: Resolve ``VLLM_OMNI_CONNECTOR_RECV_POLL_MS`` with
      ``OmniTpuPlatform`` active and run one non-progress iteration of
      ``_OmniConnectorPayloadTransportMixin._recv_loop`` on a connector without
      ``wait_for_change``.
    - Setup check: Assert ``current_omni_platform`` is ``OmniTpuPlatform``, an
      unset env resolves to 5 ms, and rebind ``_RECV_POLL_S`` on the transport
      module.
    - Outcome check: Assert ``_recv_poll_seconds()`` returns ``expected_s`` and
      ``_recv_loop`` passes ``expected_s`` to ``_work_available.wait``.
    """
    assert isinstance(current_omni_platform, OmniTpuPlatform), (
        f"Expected OmniTpuPlatform, got {type(current_omni_platform).__name__}"
    )

    monkeypatch.delenv("VLLM_OMNI_CONNECTOR_RECV_POLL_MS", raising=False)
    unset_s = transport_mod._recv_poll_seconds()
    assert unset_s == pytest.approx(UNSET_ENV_POLL_S), (
        f"env unset: _recv_poll_seconds()={unset_s} != {UNSET_ENV_POLL_S}"
    )
    monkeypatch.setenv("VLLM_OMNI_CONNECTOR_RECV_POLL_MS", env_value)
    resolved_s = transport_mod._recv_poll_seconds()
    assert resolved_s == pytest.approx(expected_s), (
        f"env={env_value!r}: _recv_poll_seconds()={resolved_s} != {expected_s}"
    )

    monkeypatch.setattr(transport_mod, "_RECV_POLL_S", resolved_s)

    mixin = _pending_poll_mixin()
    recorder = _WaitRecorder(mixin._stop_event)
    mixin._omni_connector = object()
    mixin._work_available = recorder  # type: ignore[assignment]

    mixin._recv_loop()

    assert len(recorder.timeouts) == 1, (
        f"env={env_value!r}: expected 1 _work_available.wait call in _recv_loop, "
        f"got {len(recorder.timeouts)}"
    )
    assert recorder.timeouts[0] == pytest.approx(expected_s), (
        f"env={env_value!r}: _recv_loop passed timeout={recorder.timeouts[0]} "
        f"!= expected {expected_s}"
    )

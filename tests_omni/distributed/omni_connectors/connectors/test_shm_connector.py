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
"""TPU tests for SharedMemoryConnector backend."""

import os
import uuid

import pytest
import torch
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import (
    SharedMemoryConnector,
)
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]


@pytest.mark.parametrize(
    "wakeup_env, expected_gen, expected_woke",
    [
        pytest.param("1", 0, True, id="fifo-wakeup"),
        pytest.param("0", None, False, id="polling-fallback"),
    ],
)
def test_shared_memory_connector(
    monkeypatch: pytest.MonkeyPatch,
    wakeup_env: str,
    expected_gen: int | None,
    expected_woke: bool,
) -> None:
    """SharedMemoryConnector backend

    - TPU path: compute bf16/int64 tensors on TPU, put payload dict via
      SharedMemoryConnector, get on stage-1 receiver, and move tensors back to TPU.
    - Setup check: TPU platform active, factory builds SharedMemoryConnector,
      input tensors are on TPU, and wakeup generation matches VLLM_OMNI_SHM_WAKEUP.
    - Outcome check: put/get succeed with puts==1 and gets==1, wait_for_change
      matches wakeup mode, /dev/shm segment is cleaned up, and TPU tensors match
      the CPU reference with zero error.
    """
    assert current_omni_platform.device_type == "tpu", (
        f"Expected TPU omni platform, got {current_omni_platform.device_type!r}"
    )
    monkeypatch.setenv("VLLM_OMNI_SHM_WAKEUP", wakeup_env)

    torch.manual_seed(0)
    base_cpu = torch.randn(4, 8, dtype=torch.float32)
    ref_bf16 = (base_cpu * 1.5 + 0.25).to(torch.bfloat16)
    ref_int64 = torch.arange(16, dtype=torch.int64).reshape(4, 4) * 3 + 1

    tpu_bf16 = (base_cpu.to("tpu") * 1.5 + 0.25).to(torch.bfloat16)
    tpu_int64 = torch.arange(16, dtype=torch.int64, device="tpu").reshape(4, 4) * 3 + 1
    assert tpu_bf16.device.type == "tpu", (
        f"Expected bf16 input on 'tpu', got {tpu_bf16.device.type!r}"
    )
    assert tpu_int64.device.type == "tpu", (
        f"Expected int64 input on 'tpu', got {tpu_int64.device.type!r}"
    )

    scope = f"tpu_shm_test_{uuid.uuid4().hex}"
    key = f"req_{uuid.uuid4().hex}"
    sender = OmniConnectorFactory.create_connector(
        ConnectorSpec(
            name="SharedMemoryConnector",
            extra={"stage_id": 0, "wakeup_scope": scope},
        )
    )
    receiver = OmniConnectorFactory.create_connector(
        ConnectorSpec(
            name="SharedMemoryConnector",
            extra={"stage_id": 1, "wakeup_scope": scope},
        )
    )
    try:
        assert isinstance(sender, SharedMemoryConnector), (
            f"Expected SharedMemoryConnector sender, got {type(sender).__name__}"
        )
        assert isinstance(receiver, SharedMemoryConnector), (
            f"Expected SharedMemoryConnector receiver, got {type(receiver).__name__}"
        )

        generation = receiver.get_wakeup_generation()
        assert generation == expected_gen, (
            f"VLLM_OMNI_SHM_WAKEUP={wakeup_env}: expected generation "
            f"{expected_gen!r}, got {generation!r}"
        )

        ok, put_size, metadata = sender.put(
            "0", "1", key, {"hidden": tpu_bf16, "tokens": tpu_int64}
        )
        assert ok, "SharedMemoryConnector.put returned False"
        assert put_size > 0, f"Expected positive put_size, got {put_size}"
        assert metadata is not None, (
            f"VLLM_OMNI_SHM_WAKEUP={wakeup_env}: expected non-None metadata, got None"
        )
        assert "shm" in metadata, (
            f"VLLM_OMNI_SHM_WAKEUP={wakeup_env}: expected 'shm' handle in metadata, "
            f"got {metadata!r}"
        )

        wait_gen = 0 if generation is None else generation
        woke = receiver.wait_for_change(
            wait_gen, timeout=1.0 if expected_woke else 0.01
        )
        assert woke is expected_woke, (
            f"VLLM_OMNI_SHM_WAKEUP={wakeup_env}: expected wait_for_change="
            f"{expected_woke}, got {woke}"
        )

        result = receiver.get("0", "1", key, metadata=metadata)
        assert result is not None, "SharedMemoryConnector.get returned None"
        received, get_size = result
        assert get_size == put_size, f"Received size {get_size} != put size {put_size}"

        sender_health = sender.health()
        receiver_health = receiver.health()
        assert sender_health["puts"] == 1, (
            f"Expected sender puts==1, got {sender_health['puts']}"
        )
        assert receiver_health["gets"] == 1, (
            f"Expected receiver gets==1, got {receiver_health['gets']}"
        )
        assert not os.path.exists(f"/dev/shm/{key}"), (
            f"/dev/shm/{key} segment was not cleaned up after get"
        )
        assert not os.path.exists(f"/dev/shm/shm_{key}_lockfile.lock"), (
            f"/dev/shm/shm_{key}_lockfile.lock was not cleaned up after get"
        )

        recv_bf16 = received["hidden"].to("tpu")
        recv_int64 = received["tokens"].to("tpu")
        assert recv_bf16.device.type == "tpu", (
            f"Expected received bf16 tensor on 'tpu', got {recv_bf16.device.type!r}"
        )
        assert recv_int64.device.type == "tpu", (
            f"Expected received int64 tensor on 'tpu', got {recv_int64.device.type!r}"
        )
        assert_parity(
            recv_bf16,
            ref_bf16,
            name=f"shm_{wakeup_env}_hidden",
            max_rel_l2=0.0,
            max_abs=0.0,
        )
        assert_parity(
            recv_int64,
            ref_int64,
            name=f"shm_{wakeup_env}_tokens",
            max_abs=0.0,
        )
    finally:
        receiver.close()
        sender.close()

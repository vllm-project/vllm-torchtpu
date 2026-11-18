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
"""TPU tests for YuanrongTransferEngineConnector backend."""

import uuid

import pytest
import torch
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.platforms import current_omni_platform
from vllm_omni.platforms.npu.omni_connectors.yuanrong_transfer_engine_connector import (  # noqa: E501
    YuanrongTransferEngineConnector,
)

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]


@pytest.mark.xfail(
    strict=True,
    raises=ImportError,
    reason="Ascend NPU 'yr.datasystem.TransferEngine' is unavailable on TPU.",
)
def test_yuanrong_transfer_engine_connector() -> None:
    """YuanrongTransferEngineConnector backend

    - TPU path: compute a bf16 tensor on TPU, roundtrip via
      YuanrongTransferEngineConnector put/get, and move output to TPU.
    - Setup check: TPU platform active, 'YuanrongTransferEngineConnector'
      registered in OmniConnectorFactory, and input tensor on TPU.
    - Outcome check: if Yuanrong TransferEngine were available, assert puts==1,
      gets==1, and received TPU tensor matches CPU reference with zero error.
    """
    assert current_omni_platform.device_type == "tpu", (
        f"Expected TPU omni platform, got {current_omni_platform.device_type!r}"
    )
    assert (
        "YuanrongTransferEngineConnector"
        in OmniConnectorFactory.list_registered_connectors()
    ), "YuanrongTransferEngineConnector not registered in OmniConnectorFactory"

    torch.manual_seed(0)
    base_cpu = torch.randn(4, 8, dtype=torch.float32)
    ref_cpu = (base_cpu * 1.5 + 0.25).to(torch.bfloat16)
    tpu_tensor = (base_cpu.to("tpu") * 1.5 + 0.25).to(torch.bfloat16)
    assert tpu_tensor.device.type == "tpu", (
        f"Expected input tensor on 'tpu', got {tpu_tensor.device.type!r}"
    )

    sender = YuanrongTransferEngineConnector(
        {"host": "127.0.0.1", "zmq_port": 0, "role": "sender"}
    )
    receiver = YuanrongTransferEngineConnector(
        {
            "host": "127.0.0.1",
            "sender_host": "127.0.0.1",
            "sender_zmq_port": sender.zmq_port,
            "role": "receiver",
        }
    )
    try:
        key = f"req_{uuid.uuid4().hex}"
        ok, put_size, metadata = sender.put("0", "1", key, {"hidden": tpu_tensor})
        assert ok, "YuanrongTransferEngineConnector.put returned False"
        assert put_size > 0, f"Expected positive put_size, got {put_size}"

        result = receiver.get("0", "1", key, metadata=metadata)
        assert result is not None, "YuanrongTransferEngineConnector.get returned None"
        received, _ = result

        sender_health = sender.health()
        receiver_health = receiver.health()
        assert sender_health["puts"] == 1, (
            f"Expected sender puts==1, got {sender_health['puts']}"
        )
        assert receiver_health["gets"] == 1, (
            f"Expected receiver gets==1, got {receiver_health['gets']}"
        )

        recv_tpu = received["hidden"].to("tpu")
        assert recv_tpu.device.type == "tpu", (
            f"Expected received tensor on 'tpu', got {recv_tpu.device.type!r}"
        )
        assert_parity(
            recv_tpu,
            ref_cpu,
            name="yuanrong_te_hidden",
            max_rel_l2=0.0,
            max_abs=0.0,
        )
    finally:
        receiver.close()
        sender.close()

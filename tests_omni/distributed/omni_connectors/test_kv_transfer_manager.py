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
"""TPU tests for OmniKVTransferManager cross-stage AR-to-DiT KV transfer."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
from vllm_omni.diffusion.sched.interface import KVPrefetchJob
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVTransferManager,
)
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S
from tests_omni.tiny_models import tiny_opt
from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackend
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

PREFETCH_TIMEOUT_S = 5.0


def _receive_kv_sync(
    mgr: OmniKVTransferManager, req: SimpleNamespace, device: torch.device
) -> bool:
    return mgr.receive_kv_cache(req, target_device=device)


def _receive_kv_prefetched(
    mgr: OmniKVTransferManager, req: SimpleNamespace, device: torch.device
) -> bool:
    mgr.start_prefetch(
        KVPrefetchJob(
            request_id=req.request_id,
            kv_sender_info={"host": "127.0.0.1", "zmq_port": 50051},
        ),
        target_device=device,
    )
    assert req.request_id in mgr._prefetch_futures, (
        f"Async KV prefetch was not scheduled for {req.request_id}"
    )
    future = mgr._prefetch_futures[req.request_id]
    assert mgr.wait_prefetch(timeout=PREFETCH_TIMEOUT_S) is True, (
        f"Async KV prefetch for {req.request_id} not ready after {PREFETCH_TIMEOUT_S}s"
    )
    future.result()
    return mgr.consume_and_distribute_kv_cache(req, target_device=device)


@pytest.mark.parametrize(
    ("async_prefetch", "receive_kv"),
    [
        pytest.param(False, _receive_kv_sync, id="False"),
        pytest.param(
            True,
            _receive_kv_prefetched,
            id="True",
            marks=pytest.mark.xfail(
                strict=True,
                raises=ValueError,
                reason=(
                    "OmniKVTransferManager._prefetch_payload calls "
                    "torch.accelerator.set_device_index(target_device.index) "
                    "with index=None from torch.device('tpu') "
                    "(and OmniTpuPlatform lacks Stream)"
                ),
            ),
        ),
    ],
)
def test_cross_stage_kv_transfer_ar_to_dit(
    async_prefetch: bool,
    receive_kv: Callable[[OmniKVTransferManager, SimpleNamespace, torch.device], bool],
    omni_ctx: None,
) -> None:
    """Cross-stage KV transfer, AR to DiT.

    - TPU path: Allocate paged bf16 KV cache blocks on ``torch.device("tpu")``
      using ``PallasAttentionBackend.get_kv_cache_shape``, validate the AR
      sender stage via ``OmniEngineArgs``, and run ``OmniKVTransferManager``
      AR-to-DiT send and sync/async-prefetch receive onto TPU.
    - Setup check: Assert ``current_omni_platform`` is ``OmniTpuPlatform``, KV
      blocks reside on ``device.type == "tpu"``, and the AR sender worker
      passes ``validate_worker_omni_connector``.
    - Outcome check: Assert ``handle_finished_requests_kv_transfer`` publishes
      KV and ``receive_kv_cache`` / ``consume_and_distribute_kv_cache``
      populates ``req.past_key_values`` on TPU with exact CPU ``torch.equal``
      parity.
    """
    assert isinstance(current_omni_platform, OmniTpuPlatform), (
        f"Expected OmniTpuPlatform, got {type(current_omni_platform).__name__}"
    )

    tpu_device = torch.device("tpu")
    num_blocks = 4
    block_size = 16
    num_kv_heads = 2
    head_size = 128
    seq_len = 20

    kv_shape = PallasAttentionBackend.get_kv_cache_shape(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        cache_dtype_str=torch.bfloat16,
    )
    torch.manual_seed(0)
    cpu_kv_blocks = torch.randn(kv_shape, dtype=torch.bfloat16)
    tpu_kv_blocks = cpu_kv_blocks.to(tpu_device)
    assert tpu_kv_blocks.device.type == "tpu", (
        f"Expected KV blocks on tpu, got {tpu_kv_blocks.device}"
    )

    # Pallas v3 layout is [num_blocks, block_size, num_kv_heads * 2 // 2, 2, head_size].
    tpu_k_blocks = tpu_kv_blocks[:, :, :, 0, :]
    tpu_v_blocks = tpu_kv_blocks[:, :, :, 1, :]
    block_ids = [1, 3]
    ref_k = cpu_kv_blocks[:, :, :, 0, :][block_ids].flatten(0, 1)[:seq_len].contiguous()
    ref_v = cpu_kv_blocks[:, :, :, 1, :][block_ids].flatten(0, 1)[:seq_len].contiguous()

    OmniEngineArgs(
        model=tiny_opt(),
        worker_type="ar",
        omni_kv_config={"need_send_cache": True},
        stage_connector_spec={
            "name": "SharedMemoryConnector",
            "extra": {"role": "sender"},
        },
    )

    scope = f"tpu_kv_{uuid.uuid4().hex}"
    sender_mgr = OmniKVTransferManager.from_od_config(
        SimpleNamespace(
            omni_kv_config={
                "connector_config": {
                    "type": "SharedMemoryConnector",
                    "wakeup_scope": scope,
                },
                "omni_from_stage": "0",
                "omni_to_stage": "1",
                "stage_id": 0,
                "need_send_cache": True,
            }
        )
    )
    receiver_mgr = OmniKVTransferManager.from_od_config(
        SimpleNamespace(
            omni_kv_config={
                "connector_config": {
                    "type": "SharedMemoryConnector",
                    "wakeup_scope": scope,
                },
                "omni_from_stage": "0",
                "omni_to_stage": "1",
                "stage_id": 1,
                "need_recv_cache": True,
                "enable_kv_async_prefetch": async_prefetch,
            }
        )
    )
    try:
        req_id = f"req-kv-{uuid.uuid4().hex[:8]}"
        extracted = sender_mgr.handle_finished_requests_kv_transfer(
            finished_reqs={req_id: {"block_ids": block_ids, "seq_len": seq_len}},
            kv_caches=[(tpu_k_blocks, tpu_v_blocks)],
            block_size=block_size,
            cache_dtype="bfloat16",
        )
        assert extracted == [req_id], f"Expected [{req_id!r}], got {extracted}"

        req = SimpleNamespace(
            request_id=req_id,
            sampling_params=SimpleNamespace(),
            past_key_values=None,
        )
        received = receive_kv(receiver_mgr, req, tpu_device)

        assert received is True, f"KV receive failed for {req_id}"
        assert req.past_key_values is not None, "req.past_key_values was not populated"
        recv_k = req.past_key_values.key_cache[0]
        recv_v = req.past_key_values.value_cache[0]
        assert recv_k.device.type == "tpu", (
            f"Received key_cache expected on tpu, got {recv_k.device}"
        )
        assert recv_v.device.type == "tpu", (
            f"Received value_cache expected on tpu, got {recv_v.device}"
        )
        assert torch.equal(recv_k.cpu(), ref_k), (
            "Received key_cache slice does not match source K slice"
        )
        assert torch.equal(recv_v.cpu(), ref_v), (
            "Received value_cache slice does not match source V slice"
        )
    finally:
        receiver_mgr.shutdown_prefetch()
        if sender_mgr.connector is not None:
            sender_mgr.connector.close()
        if receiver_mgr.connector is not None:
            receiver_mgr.connector.close()

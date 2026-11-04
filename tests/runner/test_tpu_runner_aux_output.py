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
"""Unit tests for routed-experts auxiliary output runner integration."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _make_fake_runner(
    max_num_tokens: int = 2048,
    is_first_tp_rank: bool = True,
) -> TPUModelRunner:
    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.speculative_config = None
    runner._pcp_enabled = False
    runner.max_num_tokens = max_num_tokens
    runner.model = MagicMock()
    runner.parallel_config = SimpleNamespace(pipeline_parallel_size=1)
    runner.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_tokens),
        kv_transfer_config=None,
    )
    runner._aux_routing_staging = None
    runner._aux_step_num_rows = 0
    runner._aux_step_staged = False
    runner.aux_output_connector = None
    return runner


@pytest.mark.parametrize(
    ("flag_name", "flag_val"),
    [
        ("USE_MOE_FUSED_EP_KERNEL", True),
        ("FUSED_MOE_DECODE_KERNEL", True),
        ("USE_MOE_TP_DECODE_KERNEL", True),
        ("TPU_MOE_SEQUENCE_PARALLEL", True),
        ("TPU_K3_SP_PREFILL", True),
    ],
)
def test_initialize_aux_output_connector_refuses_incompatible_env_flags(
    flag_name: str, flag_val: bool
):
    runner = _make_fake_runner()
    kv_cache_config = MagicMock()
    with (
        patch.object(envs, flag_name, flag_val),
        pytest.raises(NotImplementedError, match="not supported|bypass"),
    ):
        runner._initialize_aux_output_connector(kv_cache_config)


def test_initialize_aux_output_connector_refuses_spec_decode():
    runner = _make_fake_runner()
    runner.speculative_config = SimpleNamespace(method="eagle3")
    kv_cache_config = MagicMock()
    with pytest.raises(NotImplementedError, match="speculative decoding"):
        runner._initialize_aux_output_connector(kv_cache_config)


def test_initialize_aux_output_connector_refuses_pcp():
    runner = _make_fake_runner()
    runner._pcp_enabled = True
    kv_cache_config = MagicMock()
    with pytest.raises(NotImplementedError, match="prefill context parallelism"):
        runner._initialize_aux_output_connector(kv_cache_config)


def test_initialize_aux_output_connector_refuses_disaggregated_kv_transfer():
    runner = _make_fake_runner()
    runner.vllm_config.kv_transfer_config = SimpleNamespace()
    kv_cache_config = MagicMock()
    with pytest.raises(NotImplementedError, match="disaggregated KV transfer"):
        runner._initialize_aux_output_connector(kv_cache_config)


def test_initialize_aux_output_connector_resizes_device_buffer_when_exceeds():
    runner = _make_fake_runner(max_num_tokens=4096)
    kv_cache_config = MagicMock()

    device_buffer = torch.zeros((2048, 4, 8), dtype=torch.int32)
    fake_capturer = SimpleNamespace(
        device_buffer=device_buffer,
        shape_per_token=(4, 8),
    )
    fake_connector = SimpleNamespace(
        _capturer=fake_capturer,
        _buffer=SimpleNamespace(),
    )

    with patch(
        "vllm.distributed.aux_output_connector.worker.get_aux_output_connector",
        return_value=fake_connector,
    ):
        runner._initialize_aux_output_connector(kv_cache_config)

    assert runner.aux_output_connector is fake_connector
    assert fake_capturer.device_buffer.shape == (4096, 4, 8)
    assert runner._aux_routing_staging is not None
    assert runner._aux_routing_staging.shape == (4096, 4, 8)


def test_staging_bypassed_on_single_chunk():
    runner = _make_fake_runner(max_num_tokens=100)
    device_buffer = torch.zeros((100, 2, 4), dtype=torch.int32)
    device_buffer[:10] = 42
    fake_capturer = SimpleNamespace(device_buffer=device_buffer)
    runner.aux_output_connector = SimpleNamespace(
        _capturer=fake_capturer,
        _buffer=SimpleNamespace(),
    )
    runner._aux_routing_staging = torch.zeros((100, 2, 4), dtype=torch.int32)

    runner._aux_stage_chunk_routing(row_offset=0, num_tokens=10, target_num_chunks=1)
    assert runner._aux_step_num_rows == 10
    assert runner._aux_step_staged is False
    # Staging buffer was NOT copied into (remains 0)
    assert (runner._aux_routing_staging == 0).all()

    # Restore is a no-op
    device_buffer[:10] = 99
    runner._aux_restore_step_routing()
    # device_buffer was NOT overwritten by staging buffer
    assert (device_buffer[:10] == 99).all()


def test_staging_active_on_multi_chunk():
    runner = _make_fake_runner(max_num_tokens=100)
    device_buffer = torch.zeros((100, 2, 4), dtype=torch.int32)
    fake_capturer = SimpleNamespace(device_buffer=device_buffer)
    runner.aux_output_connector = SimpleNamespace(
        _capturer=fake_capturer,
        _buffer=SimpleNamespace(),
    )
    runner._aux_routing_staging = torch.zeros((100, 2, 4), dtype=torch.int32)

    # Chunk 0
    device_buffer[:5] = 11
    runner._aux_stage_chunk_routing(row_offset=0, num_tokens=5, target_num_chunks=2)
    assert runner._aux_step_num_rows == 5
    assert runner._aux_step_staged is True
    assert (runner._aux_routing_staging[:5] == 11).all()

    # Chunk 1 overwrites device_buffer rows 0..4 in forward
    device_buffer[:5] = 22
    runner._aux_stage_chunk_routing(row_offset=5, num_tokens=5, target_num_chunks=2)
    assert runner._aux_step_num_rows == 10
    assert runner._aux_step_staged is True
    assert (runner._aux_routing_staging[5:10] == 22).all()

    # Restore copies back all 10 rows
    runner._aux_restore_step_routing()
    assert runner._aux_step_staged is False
    assert (device_buffer[:5] == 11).all()
    assert (device_buffer[5:10] == 22).all()


def test_aux_output_hbm_reserves_and_shutdown():
    runner = _make_fake_runner(max_num_tokens=100)
    device_buffer = torch.zeros((100, 2, 4), dtype=torch.int32)
    fake_capturer = SimpleNamespace(device_buffer=device_buffer)
    fake_connector = MagicMock()
    fake_connector._capturer = fake_capturer
    runner.aux_output_connector = fake_connector
    runner._aux_routing_staging = torch.zeros((100, 2, 4), dtype=torch.int32)

    expected_bytes = device_buffer.nbytes + runner._aux_routing_staging.nbytes
    assert runner.aux_output_hbm_reserve_bytes() == expected_bytes

    runner.lora_config = None
    runner._pp_topk_buffer = None
    reserves = runner.post_profile_hbm_reserves()
    assert reserves.aux_output == expected_bytes

    # Test shutdown
    runner.shutdown()
    fake_connector.close.assert_called_once()
    assert runner.aux_output_connector is None
    assert runner._aux_routing_staging is None
    assert runner.aux_output_hbm_reserve_bytes() == 0

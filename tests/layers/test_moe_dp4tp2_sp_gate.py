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
"""Unit tests for DP4xTP2 sequence-parallel MoE admission and packed router gather."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm_torchtpu.distributed import tpu_communicator as comm
from vllm_torchtpu.layers.adapter import fused_moe_ep as bridge


def test_dp4tp2_max_node_tokens_scales_by_dp():
    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=512),
        parallel_config=SimpleNamespace(
            data_parallel_size=4,
            tensor_parallel_size=2,
            use_sequence_parallel_moe=True,
        ),
    )
    with patch("vllm.config.get_current_vllm_config", return_value=cfg):
        assert bridge._max_node_tokens(ep=2, pcp=1) == 4096


def test_sp_dispatch_router_logits_packs_same_dtype(monkeypatch):
    def init(instance, *args, **kwargs):
        instance.use_all2all = False

    monkeypatch.setattr(comm.DeviceCommunicatorBase, "__init__", init)
    communicator = comm.TpuDeviceCommunicator()
    ep_group = Mock(world_size=8)
    ep_group.all_gather.side_effect = lambda x, dim: torch.cat([x, x + 1.0], dim=dim)
    monkeypatch.setattr(comm, "get_ep_group", lambda: ep_group)

    hidden = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    logits = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    h_out, l_out = communicator.dispatch_router_logits(
        hidden, logits, is_sequence_parallel=True
    )
    assert ep_group.all_gather.call_count == 1
    torch.testing.assert_close(h_out, torch.cat([hidden, hidden + 1.0], dim=0))
    torch.testing.assert_close(l_out, torch.cat([logits, logits + 1.0], dim=0))

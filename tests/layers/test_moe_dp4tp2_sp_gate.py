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

from dataclasses import dataclass

import pytest
import torch
from vllm.config.parallel import ParallelConfig

import vllm_torchtpu as tpu_init
from vllm_torchtpu import _patch_sequence_parallel_moe_admission
from vllm_torchtpu.distributed import tpu_communicator as comm
from vllm_torchtpu.layers.adapter.fused_moe_ep import _max_node_tokens


@dataclass
class _ParallelConfigStub:
    """Stub parallel configuration."""

    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    enable_expert_parallel: bool = False
    use_sequence_parallel_moe: bool = False


@dataclass
class _SchedulerConfigStub:
    """Stub scheduler configuration."""

    max_num_batched_tokens: int = 512


@dataclass
class _VllmConfigStub:
    """Stub top-level vLLM configuration."""

    scheduler_config: _SchedulerConfigStub
    parallel_config: _ParallelConfigStub


@dataclass
class _EpGroupStub:
    """Stub expert parallel process group."""

    world_size: int = 8
    call_count: int = 0

    def all_gather(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Mock all gather."""
        self.call_count += 1
        return torch.cat([x, x + 1.0], dim=dim)


def test_sp_moe_gate_admits_dp4_tp2_when_flag_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test sequence-parallel MoE admission with DP >= 1."""
    monkeypatch.setattr(tpu_init.envs, "TPU_MOE_SEQUENCE_PARALLEL", True)
    monkeypatch.setattr(ParallelConfig, "_tpu_no_sp_moe_patch", False, raising=False)
    _patch_sequence_parallel_moe_admission()
    dummy = _ParallelConfigStub(
        enable_expert_parallel=True,
        tensor_parallel_size=2,
        data_parallel_size=4,
    )
    assert ParallelConfig.use_sequence_parallel_moe.fget(dummy) is True


def test_dp4tp2_max_node_tokens_uses_ep_over_pcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test max node tokens calculation under DP4xTP2."""
    expected_tokens = 4096
    cfg = _VllmConfigStub(
        scheduler_config=_SchedulerConfigStub(max_num_batched_tokens=512),
        parallel_config=_ParallelConfigStub(
            data_parallel_size=4,
            tensor_parallel_size=2,
            use_sequence_parallel_moe=True,
        ),
    )
    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: cfg)
    assert _max_node_tokens(ep=8, pcp=1) == expected_tokens


def test_sp_dispatch_router_logits_packs_same_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test dispatch router logits packs when dtypes match."""

    def init(
        instance: comm.TpuDeviceCommunicator,
        *_args: object,
        **_kwargs: object,
    ) -> None:
        """Initialize mock communicator."""
        instance.use_all2all = False

    monkeypatch.setattr(comm.DeviceCommunicatorBase, "__init__", init)
    communicator = comm.TpuDeviceCommunicator()
    ep_group = _EpGroupStub(world_size=8)
    monkeypatch.setattr(comm, "get_ep_group", lambda: ep_group)

    hidden = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    logits = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    h_out, l_out = communicator.dispatch_router_logits(
        hidden, logits, is_sequence_parallel=True
    )
    expected_gathers = 1
    assert ep_group.call_count == expected_gathers
    torch.testing.assert_close(h_out, torch.cat([hidden, hidden + 1.0], dim=0))
    torch.testing.assert_close(l_out, torch.cat([logits, logits + 1.0], dim=0))


def test_sp_dispatch_router_logits_separate_all_gather_on_dtype_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test dispatch router logits runs separate all-gathers on dtype mismatch."""

    def init(
        instance: comm.TpuDeviceCommunicator,
        *_args: object,
        **_kwargs: object,
    ) -> None:
        """Initialize mock communicator."""
        instance.use_all2all = False

    monkeypatch.setattr(comm.DeviceCommunicatorBase, "__init__", init)
    communicator = comm.TpuDeviceCommunicator()
    ep_group = _EpGroupStub(world_size=8)
    monkeypatch.setattr(comm, "get_ep_group", lambda: ep_group)

    hidden = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    logits = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    h_out, l_out = communicator.dispatch_router_logits(
        hidden, logits, is_sequence_parallel=True
    )
    expected_gathers = 2
    assert ep_group.call_count == expected_gathers
    torch.testing.assert_close(h_out, torch.cat([hidden, hidden + 1.0], dim=0))
    torch.testing.assert_close(l_out, torch.cat([logits, logits + 1.0], dim=0))

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

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.runner import \
    moe_runner as moe_runner_mod

import vllm_torchtpu.envs as envs


class _LocalExpertQuantMethod:
    """Properties of TorchTPU's local-only monolithic MoE method."""

    is_monolithic = True
    supports_internal_mk = False
    skip_forward_padding = True
    has_unpadded_output = False
    moe_kernel = None


class _RankZeroExperts:
    """One EP shard containing global expert 0 only.

    Expert 0 computes ``2 * x``. Routing weights are derived from the same
    router logits that enter the upstream MoERunner, just as a monolithic MoE
    method does. Routes for expert 1 are deliberately absent from this shard;
    their contribution must arrive through PCP combine.
    """

    def __init__(self):
        self.quant_method = _LocalExpertQuantMethod()

    def _ensure_moe_quant_config_init(self):
        pass

    def forward_monolithic(self, *, x, router_logits, input_ids=None):
        del input_ids
        expert_weights = torch.softmax(router_logits, dim=-1)
        return x * expert_weights[:, 0:1] * 2.0


class _TwoRankPcpGroup:
    """Single-process model of PCP dispatch/combine for EP rank 0.

    Rank 1 owns global expert 1, which computes ``3 * x``. The fake collective
    performs real tensor concatenation and contribution summation so the test
    checks the numerical multi-expert contract rather than merely checking
    whether collective methods were called.
    """

    world_size = 2
    rank_in_group = 0

    def __init__(self, peer_hidden_states, peer_router_logits):
        self._peer_hidden_states = peer_hidden_states
        self._peer_router_logits = peer_router_logits
        self._gathered_hidden_states = None
        self._gathered_router_logits = None

    def all_gather(self, tensor, dim=0):
        assert dim == 0
        if tensor.shape[-1] == self._peer_hidden_states.shape[-1]:
            gathered = torch.cat([tensor, self._peer_hidden_states], dim=dim)
            self._gathered_hidden_states = gathered
            return gathered

        assert tensor.shape[-1] == self._peer_router_logits.shape[-1]
        gathered = torch.cat([tensor, self._peer_router_logits], dim=dim)
        self._gathered_router_logits = gathered
        return gathered

    def reduce_scatter(self, rank_zero_output, dim=0):
        assert dim == 0
        assert self._gathered_hidden_states is not None
        assert self._gathered_router_logits is not None

        expert_weights = torch.softmax(self._gathered_router_logits, dim=-1)
        rank_one_output = (self._gathered_hidden_states *
                           expert_weights[:, 1:2] * 3.0)
        combined = rank_zero_output + rank_one_output
        return combined.chunk(self.world_size, dim=dim)[self.rank_in_group]


def _make_parallel_config(*, pcp_size=2, dp_size=1, sp_size=1):
    return FusedMoEParallelConfig(
        tp_size=1,
        tp_rank=0,
        pcp_size=pcp_size,
        pcp_rank=0,
        dp_size=dp_size,
        dp_rank=0,
        ep_size=pcp_size * dp_size,
        ep_rank=0,
        sp_size=sp_size,
        use_ep=True,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )


def _make_rank_zero_runner():
    parallel_config = _make_parallel_config()
    moe_config = SimpleNamespace(
        hidden_dim=3,
        hidden_dim_unpadded=3,
        tp_size=1,
        pcp_size=2,
        dp_size=1,
        ep_size=2,
        sp_size=1,
        is_sequence_parallel=False,
        skip_final_all_reduce=True,
        moe_parallel_config=parallel_config,
    )

    # Construct the upstream operator without its weight-loading and global
    # registration machinery. All forward-path methods are the real vLLM
    # MoERunner methods; only the local expert kernel and collectives are small
    # deterministic CPU substitutes.
    runner = moe_runner_mod.MoERunner.__new__(moe_runner_mod.MoERunner)
    torch.nn.Module.__init__(runner)
    runner.moe_config = moe_config
    runner.router = object()
    runner.routed_experts = _RankZeroExperts()
    runner._shared_experts = None
    runner.gate = None
    runner.shared_expert_gate = None
    runner._fse_fuse_gate = False
    runner._combined_gate_weight = None
    runner.routed_input_transform = None
    runner.routed_output_transform = None
    runner.routed_scaling_factor = 1.0
    runner.layer_name = "contract_test.moe"
    runner._sequence_parallel_context = contextlib.nullcontext
    runner._forward_entry = moe_runner_mod._moe_forward
    return runner


def test_tpu_pcp_patch_preserves_non_pcp_all2all_triggers():
    from vllm_torchtpu import _patch_moe_explicit_pcp_collectives

    _patch_moe_explicit_pcp_collectives()

    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 0):
        assert not _make_parallel_config().use_all2all_kernels
    assert _make_parallel_config(pcp_size=1, dp_size=2).use_all2all_kernels
    assert _make_parallel_config(pcp_size=1, sp_size=2).use_all2all_kernels


def test_tpu_pcp_chunk_pipeline_owns_dispatch_combine():
    """PCP skips outer collectives when the chunk kernel owns them."""
    from vllm_torchtpu import _patch_moe_explicit_pcp_collectives

    _patch_moe_explicit_pcp_collectives()

    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 16384):
        assert _make_parallel_config().use_all2all_kernels


def test_vllm_moe_pcp_ep_combines_remote_expert_contributions(monkeypatch):
    """The vLLM MoE entry point must combine experts across PCP/EP ranks.

    This is a backend contract test for a local-only monolithic MoE kernel:
    vLLM must either execute explicit PCP collectives around the kernel or use
    an internal dispatch/combine implementation that produces the same result.
    """

    from vllm_torchtpu import _patch_moe_explicit_pcp_collectives

    _patch_moe_explicit_pcp_collectives()

    local_hidden_states = torch.tensor([[1.0, 2.0, 4.0]])
    peer_hidden_states = torch.tensor([[3.0, 5.0, 7.0]])
    local_router_logits = torch.log(torch.tensor([[0.25, 0.75]]))
    peer_router_logits = torch.log(torch.tensor([[0.60, 0.40]]))

    pcp_group = _TwoRankPcpGroup(peer_hidden_states, peer_router_logits)
    monkeypatch.setattr(moe_runner_mod, "get_pcp_group", lambda: pcp_group)

    runner = _make_rank_zero_runner()
    monkeypatch.setattr(moe_runner_mod, "get_layer_from_name",
                        lambda _layer_name: runner)
    actual = runner(local_hidden_states, local_router_logits)

    local_weights = torch.softmax(local_router_logits, dim=-1)
    expected = local_hidden_states * (local_weights[:, 0:1] * 2.0 +
                                      local_weights[:, 1:2] * 3.0)
    torch.testing.assert_close(actual, expected)

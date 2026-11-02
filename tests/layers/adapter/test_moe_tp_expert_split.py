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
"""TPU_MOE_TP_EXPERT_SPLIT: which MoE parallel configs become EP over TP."""

import pytest
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

from vllm_torchtpu.layers.adapter.moe_tp_expert_split import split_experts_over_tp


def _config(**overrides) -> FusedMoEParallelConfig:
    fields = dict(
        tp_size=2,
        tp_rank=1,
        pcp_size=1,
        pcp_rank=0,
        dp_size=1,
        dp_rank=0,
        ep_size=1,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    fields.update(overrides)
    return FusedMoEParallelConfig(**fields)


def test_flat_tp_layer_gets_whole_experts_per_rank():
    split = split_experts_over_tp(_config())
    assert (split.use_ep, split.ep_size, split.ep_rank) == (True, 2, 1)
    assert (split.tp_size, split.tp_rank) == (1, 0)
    # Still no cross-replica traffic: no all2all, no DP dispatch.
    assert not split.use_all2all_kernels
    assert split.dp_size == 1


@pytest.mark.parametrize(
    "overrides",
    [
        dict(use_ep=True, ep_size=2, ep_rank=1, tp_size=1, tp_rank=0),  # already EP
        dict(tp_size=1, tp_rank=0),  # nothing to split
        dict(dp_size=2, dp_rank=1),  # MoE spans data-parallel ranks
        dict(pcp_size=2),
        dict(sp_size=2),
    ],
)
def test_other_layouts_are_left_alone(overrides):
    config = _config(**overrides)
    assert split_experts_over_tp(config) is config

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
"""Deployments the fused EP MoE v3 bridge serves or refuses at weight load."""

from types import SimpleNamespace

import pytest
import vllm.config

from vllm_torchtpu.layers.adapter import fused_moe_ep_v3


def _layer(is_sequence_parallel, pcp=1, use_ep=True):
    parallel = SimpleNamespace(
        is_sequence_parallel=is_sequence_parallel, pcp_size=pcp, use_ep=use_ep
    )
    return SimpleNamespace(moe_config=SimpleNamespace(moe_parallel_config=parallel))


@pytest.fixture
def tp(monkeypatch):
    def set_tp(size):
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=size)
        )
        monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)

    return set_tp


@pytest.mark.parametrize(
    "tp_size,is_sequence_parallel", [(1, False), (8, True)], ids=["dp", "tp-sp"]
)
def test_serves_each_ranks_own_tokens(tp, tp_size, is_sequence_parallel):
    tp(tp_size)
    fused_moe_ep_v3._check_deployment(_layer(is_sequence_parallel))


def test_refuses_tp_with_replicated_tokens(tp):
    """Without sequence parallelism every TP rank holds every token."""
    tp(8)
    with pytest.raises(NotImplementedError, match="TPU_MOE_SEQUENCE_PARALLEL=1"):
        fused_moe_ep_v3._check_deployment(_layer(is_sequence_parallel=False))


@pytest.mark.parametrize(
    "layer,match",
    [
        (_layer(True, pcp=2), "prefill context"),
        (_layer(True, use_ep=False), "enable-expert-parallel"),
    ],
    ids=["pcp", "no-ep"],
)
def test_sequence_parallelism_keeps_the_other_refusals(tp, layer, match):
    tp(8)
    with pytest.raises(NotImplementedError, match=match):
        fused_moe_ep_v3._check_deployment(layer)

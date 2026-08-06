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
"""Unit tests for vllm_torchtpu.layers.vllm.moe_routing helpers."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.layers.vllm import moe_routing


def test_skip_padding_routes_marks_ids_with_minus_one(monkeypatch):
    context = SimpleNamespace(
        is_padding=torch.tensor([False, True, False, True]))
    monkeypatch.setattr(moe_routing.vllm_envs, "VLLM_MOE_SKIP_PADDING", True)
    monkeypatch.setattr(moe_routing, "is_forward_context_available",
                        lambda: True)
    monkeypatch.setattr(moe_routing, "get_forward_context", lambda: context)
    topk_ids = torch.arange(12, dtype=torch.int32).reshape(4, 3)

    result = moe_routing._skip_padding_routes(topk_ids)

    torch.testing.assert_close(result[0], topk_ids[0])
    torch.testing.assert_close(result[1], torch.full((3, ), -1,
                                                     dtype=torch.int32))
    torch.testing.assert_close(result[2], topk_ids[2])
    torch.testing.assert_close(result[3], torch.full((3, ), -1,
                                                     dtype=torch.int32))


def _make_layer(*, use_ep, ep_size=1, ep_rank=0, global_num_experts=0):
    return SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(use_ep=use_ep),
            ep_size=ep_size,
            ep_rank=ep_rank),
        global_num_experts=global_num_experts,
    )


# (ep_size, global_num_experts) -- mix of divisible and non-divisible cases,
# plus the production sizes we actually ship (Qwen3-30B: 128, Qwen3-480B: 160).
@pytest.mark.parametrize("ep_size,global_num_experts", [
    (2, 8),
    (2, 9),
    (4, 18),
    (4, 33),
    (8, 128),
    (8, 160),
])
def test_get_experts_start_matches_determine_expert_map(
        ep_size, global_num_experts):
    """Parity against vLLM's linear-placement contract for every rank."""
    determine_expert_map = pytest.importorskip(
        "vllm.model_executor.layers.fused_moe.expert_map_manager"
    ).determine_expert_map
    for ep_rank in range(ep_size):
        _, expert_map, _ = determine_expert_map(
            ep_size=ep_size,
            ep_rank=ep_rank,
            global_num_experts=global_num_experts,
            expert_placement_strategy="linear",
        )
        owned = (expert_map >= 0).nonzero(as_tuple=True)[0]
        expected = int(owned[0].item()) if owned.numel() else 0
        actual = moe_routing.get_experts_start(
            _make_layer(use_ep=True,
                        ep_size=ep_size,
                        ep_rank=ep_rank,
                        global_num_experts=global_num_experts))
        assert actual == expected, (
            f"ep_size={ep_size} ep_rank={ep_rank} "
            f"global_num_experts={global_num_experts}: "
            f"expected experts_start={expected}, got {actual}")


def test_get_experts_start_returns_none_without_ep():
    assert moe_routing.get_experts_start(_make_layer(use_ep=False)) is None


def test_validate_linear_ep_placement_accepts_linear():
    moe_routing.validate_linear_ep_placement(
        SimpleNamespace(expert_placement_strategy="linear"))


def test_validate_linear_ep_placement_requires_declared_strategy():
    # An object that never declares its placement must fail loudly, not be
    # assumed linear.
    with pytest.raises(AttributeError):
        moe_routing.validate_linear_ep_placement(SimpleNamespace())


def test_validate_linear_ep_placement_rejects_round_robin():
    with pytest.raises(NotImplementedError, match="linear EP placement"):
        moe_routing.validate_linear_ep_placement(
            SimpleNamespace(expert_placement_strategy="round_robin"))

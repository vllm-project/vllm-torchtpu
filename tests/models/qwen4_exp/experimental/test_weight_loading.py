# SPDX-License-Identifier: Apache-2.0
"""Batched per-layer MoE weight loading (CPU, stub loader)."""

from types import SimpleNamespace as NS

import torch
from torch import nn

from vllm_torchtpu.models.qwen4_exp.experimental.model import (
    is_remote_expert,
    load_layer_mlp_weights,
    local_expert_ids,
    moe_weights_mapper,
)


class _RecordingLoader:
    def __init__(self):
        self.calls = []

    def load_weights(self, weights, mapper=None):
        names = [name for name, _ in weights]
        self.calls.append((names, mapper))
        return names


def test_one_loader_call_per_batch_and_shared_local_copy():
    mlp = nn.Module()
    mlp.shared_expert_local = nn.Module()
    loader = _RecordingLoader()
    w = torch.zeros(1)
    items = [
        ("experts.0.gate_proj.weight", w),
        ("shared_expert.up_proj.weight", w),
        ("gate.weight", w),
    ]

    loaded = load_layer_mlp_weights(loader, mlp, iter(items), moe_weights_mapper(), 3)

    assert len(loader.calls) == 2
    assert loader.calls[0][0] == [n for n, _ in items]
    assert loader.calls[1][0] == ["shared_expert_local.up_proj.weight"]
    assert "model.layers.3.mlp.gate.weight" in loaded
    assert "model.layers.3.mlp.shared_expert_local.up_proj.weight" in loaded


def test_no_shared_local_means_one_call():
    loader = _RecordingLoader()
    items = [("shared_expert.up_proj.weight", torch.zeros(1))]
    load_layer_mlp_weights(loader, nn.Module(), iter(items), moe_weights_mapper(), 0)
    assert len(loader.calls) == 1


def test_remote_expert_filter():
    assert not is_remote_expert("experts.5.down_proj.weight", None)
    assert is_remote_expert("experts.5.down_proj.weight", {0, 1})
    assert not is_remote_expert("experts.1.down_proj.weight_scale_inv", {0, 1})
    assert not is_remote_expert("shared_expert.down_proj.weight", {0})
    assert not is_remote_expert("experts.gate.weight", {0})


def test_local_expert_ids_follow_the_expert_map():
    def mlp(use_ep):
        experts = NS(
            use_ep=use_ep,
            global_num_experts=4,
            _map_global_expert_id_to_local_expert_id=lambda e: e - 2,
        )
        return NS(experts=NS(routed_experts=experts))

    assert local_expert_ids(mlp(False)) is None
    assert local_expert_ids(mlp(True)) == {2, 3}

# SPDX-License-Identifier: Apache-2.0
"""CPU tests for TPU_MOE_EXPERT_PLACEMENT: expert map at load, ids after routing."""

import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter import expert_permutation
from vllm_torchtpu.layers.core import expert_permutation as core

pytestmark = pytest.mark.cpu_test

E, EP, NAME = 64, 8, "model.layers.3.mlp.experts"


@pytest.fixture
def slots(monkeypatch):
    counts = np.random.default_rng(0).gamma(2.0, 10.0, size=E)
    slots = core.balance(counts, EP)
    monkeypatch.setattr(expert_permutation, "_PLACEMENT", {NAME: slots})
    monkeypatch.setattr(expert_permutation, "_EP_SIZE", EP)
    return slots


def _layer(ep_rank, name=NAME, strategy="linear", eplb=False):
    em = pytest.importorskip("vllm.model_executor.layers.fused_moe.expert_map_manager")
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    from vllm_torchtpu import _patch_expert_map_host_lookup

    _patch_expert_map_host_lookup()
    parallel = SimpleNamespace(
        use_ep=True,
        ep_size=EP,
        ep_rank=ep_rank,
        tp_size=1,
        tp_rank=0,
        dp_size=EP,
        enable_eplb=eplb,
        use_all2all_kernels=False,
        needs_round_robin_routing_tables=False,
    )
    manager = em.ExpertMapManager(
        max_num_batched_tokens=64,
        top_k=2,
        global_num_experts=E,
        num_redundant_experts=0,
        num_expert_group=None,
        moe_parallel_config=parallel,
        placement_strategy=strategy,
        enable_eplb=False,
    )
    layer = torch.nn.Module()
    layer.layer_name = name
    layer.moe_config = SimpleNamespace(moe_parallel_config=parallel)
    layer.expert_map_manager = manager
    layer.global_num_experts = E
    layer.update_expert_map_info = types.MethodType(
        RoutedExperts.update_expert_map_info, layer
    )
    layer.update_expert_map_info()
    return layer


def test_expert_map_sends_each_expert_to_its_slot(slots):
    local = E // EP
    owner = {}
    for rank in range(EP):
        layer = _layer(rank)
        manager = layer.expert_map_manager
        before = [manager.map_global_to_local(e) for e in range(E)]
        assert before == [
            e - rank * local if e // local == rank else -1 for e in range(E)
        ]
        expert_permutation.place_experts(layer)
        np.testing.assert_array_equal(expert_permutation.host_slots(layer), slots)
        after = [manager.map_global_to_local(e) for e in range(E)]
        assert layer._expert_map.tolist() == after
        for e, index in enumerate(after):
            if index >= 0:
                assert slots[e] == rank * local + index
                owner[e] = rank
    assert sorted(owner) == list(range(E))
    np.testing.assert_array_equal([owner[e] for e in range(E)], slots // local)


def test_layers_outside_the_file_keep_linear_placement(slots):
    layer = _layer(0, name="mtp.layers.0.mlp.experts")
    expert_permutation.place_experts(layer)
    assert expert_permutation.host_slots(layer) is None
    expert_permutation.register_slots_buffer(layer, device=torch.device("cpu"))
    assert layer._expert_slots is None
    ids = torch.tensor([[3, 5]], dtype=torch.int32)
    assert expert_permutation.to_slots(layer, ids) is ids


def test_unplaced_named_layer_and_bad_layouts_raise(slots, monkeypatch):
    layer = _layer(0)
    with pytest.raises(NotImplementedError, match="NVFP4"):
        expert_permutation.register_slots_buffer(layer, device=torch.device("cpu"))
    with pytest.raises(NotImplementedError, match="EPLB"):
        expert_permutation.place_experts(_layer(0, eplb=True))
    monkeypatch.setattr(expert_permutation, "_EP_SIZE", 4)
    with pytest.raises(ValueError, match="EP 4"):
        expert_permutation.place_experts(_layer(0))


def test_to_slots_maps_ids_and_keeps_negative_ones(slots):
    layer = _layer(2)
    expert_permutation.place_experts(layer)
    expert_permutation.register_slots_buffer(layer, device=torch.device("cpu"))
    ids = torch.randint(0, E, (12, 10), dtype=torch.int32)
    ids[0, 3] = -1
    got = expert_permutation.to_slots(layer, ids)
    assert got.dtype == torch.int32
    want = np.where(ids.numpy() >= 0, slots[ids.numpy().clip(0)], -1)
    np.testing.assert_array_equal(got.numpy(), want)

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
    monkeypatch.setattr(expert_permutation, "_PLACEMENT", {NAME: slots[:, None]})
    monkeypatch.setattr(expert_permutation, "_EP_SIZE", EP)
    monkeypatch.setattr(expert_permutation, "_SLOTS_PER_RANK", E // EP)
    return slots


def _hot_counts():
    counts = np.random.default_rng(1).gamma(2.0, 10.0, size=E)
    counts[[3, 40]] = counts.sum() / EP * np.array([2.5, 1.2])
    return counts


@pytest.fixture
def copies(monkeypatch):
    """Installs copies in one extra slot per rank; returns the slots [E, C]."""

    def install(extra_copies=None):
        slots = core.replicate(_hot_counts(), EP, 1, extra_copies)
        monkeypatch.setattr(expert_permutation, "_PLACEMENT", {NAME: slots})
        monkeypatch.setattr(expert_permutation, "_EP_SIZE", EP)
        monkeypatch.setattr(expert_permutation, "_SLOTS_PER_RANK", E // EP + 1)
        return slots

    return install


def _layer(ep_rank, name=NAME, strategy="linear", eplb=False, local=E // EP):
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
    for n in ("w13_weight", "w2_weight_scale"):
        layer.register_parameter(n, torch.nn.Parameter(torch.ones(local, 3)))
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
        np.testing.assert_array_equal(
            expert_permutation.host_slots(layer), slots[:, None]
        )
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


def test_expert_map_holds_each_copy_once(copies):
    slots = copies()
    local = E // EP + 1
    owners = {}
    for rank in range(EP):
        layer = _layer(rank, local=local)
        expert_permutation.place_experts(layer)
        manager = layer.expert_map_manager
        for e in range(E):
            index = manager.map_global_to_local(e)
            here = [s for s in slots[e] if s >= 0 and s // local == rank]
            assert index == (here[0] - rank * local if here else -1)
            if here:
                owners.setdefault(e, []).append(here[0])
        # Every slot of a full copy file holds an expert: nothing is zeroed.
        assert (layer.w13_weight.data == 1).all()
    for e in range(E):
        assert sorted(owners[e]) == sorted(slots[e][slots[e] >= 0].tolist())
    assert expert_permutation.slots_per_rank(layer, E // EP) == local
    assert expert_permutation.slots_per_rank(_layer(0, name="other"), 8) == 8


def test_slot_experts_inverts_the_copies(copies):
    slots = copies()
    layer = _layer(0, local=E // EP + 1)
    expert_permutation.place_experts(layer)
    experts = expert_permutation.slot_experts(layer)
    assert experts.shape == (EP * (E // EP + 1),) and (experts >= 0).all()
    for e, row in enumerate(slots):
        assert (experts[row[row >= 0]] == e).all()


def test_empty_slots_are_zeroed_and_size_is_checked(copies):
    slots = copies(extra_copies=3)  # 5 of the 8 spare slots stay empty
    local = E // EP + 1
    experts = np.full(EP * local, -1)
    e, _ = np.nonzero(slots >= 0)
    experts[slots[slots >= 0]] = e
    for rank in range(EP):
        layer = _layer(rank, local=local)
        expert_permutation.place_experts(layer)
        empty = experts[rank * local : (rank + 1) * local] < 0
        for p in (layer.w13_weight, layer.w2_weight_scale):
            np.testing.assert_array_equal(p.data[:, 0].numpy() == 0, empty)
    with pytest.raises(ValueError, match="slots"):
        expert_permutation.place_experts(_layer(0))  # E // EP slots allocated


def _bf16_bits(weights):
    return (weights.astype(np.float32).view(np.int32) >> 16) & 0xFFFF


def test_to_slots_splits_routes_over_copies_by_weight_hash(copies):
    slots = copies()
    layer = _layer(5, local=E // EP + 1)
    expert_permutation.place_experts(layer)
    expert_permutation.register_slots_buffer(layer, device=torch.device("cpu"))
    assert layer._expert_slots is None and layer._expert_copy_digits is not None
    rng = np.random.default_rng(3)
    ids = np.stack([rng.choice(E, 10, replace=False) for _ in range(400)])
    ids[:, 0] = 3  # the hottest expert in every token
    ids[7, 4] = -1
    weights = rng.dirichlet(np.ones(10), len(ids)).astype(np.float32)
    w = torch.from_numpy(weights).to(torch.bfloat16)
    got = expert_permutation.to_slots(layer, torch.from_numpy(ids).to(torch.int32), w)
    num_copies = (slots >= 0).sum(axis=1)
    safe = ids.clip(0)
    pick = _bf16_bits(w.float().numpy()) % num_copies[safe]
    want = np.where(ids >= 0, slots[safe, pick], -1)
    assert got.dtype == torch.int32
    np.testing.assert_array_equal(got.numpy(), want)
    # A token takes the same copies wherever it sits in the batch.
    perm = rng.permutation(len(ids))
    again = expert_permutation.to_slots(
        layer, torch.from_numpy(ids[perm]).to(torch.int32), w[perm]
    )
    np.testing.assert_array_equal(again.numpy(), want[perm])
    # The hottest expert's routes spread over all of its copies.
    hot = np.bincount(got.numpy()[:, 0], minlength=EP * (E // EP + 1))
    hot = hot[slots[3][slots[3] >= 0]]
    assert len(hot) > 1 and hot.min() > 0.5 * len(ids) / len(hot)
    with pytest.raises(ValueError, match="routing weights"):
        expert_permutation.to_slots(layer, torch.from_numpy(ids).to(torch.int32))

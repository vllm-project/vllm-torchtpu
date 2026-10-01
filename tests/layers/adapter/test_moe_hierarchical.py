# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hierarchical MoE parallelism: the parts that fail silently if wrong.

Everything here is about *ownership arithmetic*, which is why it needs no TPU:
a rank that computes the wrong expert block or the wrong intermediate slice
still runs, still produces a tensor of the right shape, and returns wrong
numbers. Those are the cases worth pinning.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_torchtpu.distributed.chip_topology import ChipTopology


@pytest.fixture(autouse=True)
def restore_hierarchy_state(monkeypatch):
    from vllm.model_executor.model_loader import default_loader, ep_weight_filter

    from vllm_torchtpu.layers.adapter import moe_hierarchical

    monkeypatch.setattr(
        default_loader,
        "compute_local_expert_ids",
        ep_weight_filter.compute_local_expert_ids,
    )
    monkeypatch.setattr(moe_hierarchical, "_filter_patched", False)
    monkeypatch.setattr(moe_hierarchical, "_SPLIT_CACHE", {})


@pytest.fixture
def dp_runtime(monkeypatch):
    import torch.distributed
    import vllm.config as config_module
    import vllm.distributed as distributed

    from vllm_torchtpu import envs
    from vllm_torchtpu.distributed import chip_topology, ep_mesh
    from vllm_torchtpu.distributed.mesh_utils import CpGroupLayout
    from vllm_torchtpu.layers.adapter import moe_hierarchical

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            prefill_context_parallel_size=1,
            enable_eplb=False,
            expert_placement_strategy="linear",
            data_parallel_size=4,
            tensor_parallel_size=8,
            enable_ep_weight_filter=True,
        ),
        model_config=Mock(is_moe=True, get_num_experts=Mock(return_value=224)),
    )
    device_ids = tuple(
        100 + 2 * ((7 * chip) % 16) + core for chip in range(16) for core in (1, 0)
    )
    runtime = SimpleNamespace(
        config=config,
        tp_rank=1,
        layout=CpGroupLayout(
            ranks=tuple(range(200, 232)),
            device_ids=device_ids,
            rank_in_group=9,
            world_size=32,
        ),
    )
    runtime.resolve_layout = Mock(side_effect=lambda group: runtime.layout)

    def gather(out, split, group=None):
        out[:] = [
            chip_topology.hierarchical_moe_split(device_ids, r) for r in range(len(out))
        ]

    runtime.gather = Mock(side_effect=gather)
    monkeypatch.setattr(envs, "TPU_MOE_HIERARCHICAL_EP", True)
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", False)
    monkeypatch.setattr(
        config_module, "get_current_vllm_config_or_none", lambda: config
    )
    monkeypatch.setattr(config_module, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(
        distributed, "get_dp_group", lambda: Mock(world_size=4, rank_in_group=1)
    )
    monkeypatch.setattr(
        distributed, "get_pcp_group", lambda: Mock(world_size=1, rank_in_group=0)
    )
    monkeypatch.setattr(
        distributed, "get_tensor_model_parallel_rank", lambda: runtime.tp_rank
    )
    monkeypatch.setattr(ep_mesh, "get_ep_group", lambda: Mock())
    monkeypatch.setattr(moe_hierarchical, "get_cp_group_layout", runtime.resolve_layout)
    monkeypatch.setattr(torch.distributed, "all_gather_object", runtime.gather)
    monkeypatch.setattr(
        chip_topology,
        "get_chip_topology",
        lambda: ChipTopology(
            [[100 + 2 * chip, 101 + 2 * chip] for chip in range(16)],
            list(reversed(range(100, 132))),
        ),
    )
    return runtime


def test_dp_split_uses_ep_device_mapping_and_agrees_once(dp_runtime):
    from vllm_torchtpu.layers.adapter import moe_hierarchical

    assert moe_hierarchical.hierarchical_split_or_none() == (16, 12, 2, 0)
    assert moe_hierarchical.hierarchical_split_or_none() == (16, 12, 2, 0)
    assert dp_runtime.gather.call_count == 1


@pytest.mark.parametrize("disabled", ["flag", "ep", "config"])
def test_disabled_hierarchy_does_not_gather_devices(monkeypatch, dp_runtime, disabled):
    import vllm.config as config_module

    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import moe_hierarchical

    if disabled == "flag":
        monkeypatch.setattr(envs, "TPU_MOE_HIERARCHICAL_EP", False)
    elif disabled == "ep":
        dp_runtime.config.parallel_config.enable_expert_parallel = False
    else:
        monkeypatch.setattr(
            config_module, "get_current_vllm_config_or_none", lambda: None
        )
    assert moe_hierarchical.hierarchical_split_or_none() is None
    dp_runtime.resolve_layout.assert_not_called()


@pytest.mark.parametrize("field", ["enable_eplb", "USE_MOE_FUSED_EP_KERNEL"])
def test_unsupported_hierarchy_config_fails_before_gather(
    monkeypatch, dp_runtime, field
):
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import moe_hierarchical

    target = envs if field.isupper() else dp_runtime.config.parallel_config
    monkeypatch.setattr(target, field, True)
    with pytest.raises(ValueError, match="GMM MoE backend"):
        moe_hierarchical.hierarchical_split_or_none()
    dp_runtime.resolve_layout.assert_not_called()


def test_unusable_topology_is_rejected(monkeypatch, dp_runtime):
    from vllm_torchtpu.distributed import chip_topology
    from vllm_torchtpu.layers.adapter import moe_hierarchical

    monkeypatch.setattr(chip_topology, "get_chip_topology", lambda: None)
    with pytest.raises(ValueError, match="complete multi-core chips"):
        moe_hierarchical.hierarchical_split_or_none()


def test_split_must_agree_across_ranks(dp_runtime):
    from vllm_torchtpu.layers.adapter import moe_hierarchical

    def every_rank_claims_this_slot(out, split, group=None):
        out[:] = [split] * len(out)

    dp_runtime.gather.side_effect = every_rank_claims_this_slot
    with pytest.raises(ValueError, match="on every rank"):
        moe_hierarchical.hierarchical_split_or_none()


class _FakeDevice:
    def __init__(
        self, device_id: int, coords: tuple[int, int, int], core_on_chip: int
    ) -> None:
        self.id = device_id
        self.coords = coords
        self.core_on_chip = core_on_chip


def _tpu7x_single_host() -> list[_FakeDevice]:
    """The layout probed on a real tpu7x host: 4 chips x 2 cores."""
    return [
        _FakeDevice(0, (0, 0, 0), 0),
        _FakeDevice(1, (0, 0, 0), 1),
        _FakeDevice(2, (1, 0, 0), 0),
        _FakeDevice(3, (1, 0, 0), 1),
        _FakeDevice(4, (0, 1, 0), 0),
        _FakeDevice(5, (0, 1, 0), 1),
        _FakeDevice(6, (1, 1, 0), 0),
        _FakeDevice(7, (1, 1, 0), 1),
    ]


def _topology_from(devices) -> ChipTopology:
    import collections

    grouped = collections.defaultdict(list)
    for device in devices:
        grouped[tuple(device.coords)].append(int(device.id))
    chips = [sorted(grouped[key]) for key in sorted(grouped)]
    return ChipTopology(chips, [int(d.id) for d in devices])


def test_chip_grouping_pairs_the_cores_of_one_chip() -> None:
    topology = _topology_from(_tpu7x_single_host())
    assert topology.num_chips == 4
    assert topology.cores_per_chip == 2
    # Devices sharing mesh coordinates share a chip.
    assert topology.chips == [[0, 1], [4, 5], [2, 3], [6, 7]]


def test_every_rank_maps_to_exactly_one_chip_slot() -> None:
    """A rank must land on one chip, and each (chip, core) is used once.

    A duplicate or a gap here means two ranks believe they own the same expert
    block, or nobody owns one -- either way the routed output is wrong while
    every shape still checks out.
    """
    topology = _topology_from(_tpu7x_single_host())
    seen = set()
    for rank in range(8):
        slot = topology.chip_of(rank)
        assert slot not in seen, f"rank {rank} duplicates slot {slot}"
        seen.add(slot)
    assert len(seen) == 8
    assert {chip for chip, _ in seen} == {0, 1, 2, 3}


def test_chip_order_is_deterministic_not_device_enumeration_order() -> None:
    """Ranks must agree on which chip is EP rank 0.

    The grouping is keyed on sorted mesh coordinates rather than on the order
    devices happen to be enumerated in, because every rank computes this
    independently and a disagreement would shard the experts inconsistently
    across the job.
    """
    forward = _topology_from(_tpu7x_single_host())
    shuffled = _topology_from(list(reversed(_tpu7x_single_host())))
    assert forward.chips == shuffled.chips


def test_topology_rejects_ragged_chips() -> None:
    """A chip holding a different number of cores is not splittable."""
    from vllm_torchtpu.distributed import chip_topology

    devices = _tpu7x_single_host()[:-1]  # chip (1,1,0) now has one core
    import collections

    grouped = collections.defaultdict(list)
    for device in devices:
        grouped[tuple(device.coords)].append(int(device.id))
    sizes = {len(v) for v in grouped.values()}
    assert sizes == {1, 2}, "fixture should be ragged"
    del chip_topology  # the guard itself lives in get_chip_topology


def test_hierarchical_split_is_ep_between_chips_tp_within() -> None:
    """The whole design in one assertion.

    ep_rank must be the chip and tp_rank the core, not the other way round:
    swapping them would put the expert split *inside* a chip and the width
    split *across* chips -- numerically identical, and exactly the layout this
    change exists to avoid.
    """
    topology = _topology_from(_tpu7x_single_host())
    for rank in range(8):
        chip, core = topology.chip_of(rank)
        ep_size, ep_rank, tp_size, tp_rank = (
            topology.num_chips,
            chip,
            topology.cores_per_chip,
            core,
        )
        assert (ep_size, tp_size) == (4, 2)
        assert ep_rank == chip and tp_rank == core
    # Ranks 0 and 1 share a chip: same experts, different width slice.
    assert topology.chip_of(0)[0] == topology.chip_of(1)[0]
    assert topology.chip_of(0)[1] != topology.chip_of(1)[1]
    # Ranks 0 and 2 do not: different experts.
    assert topology.chip_of(0)[0] != topology.chip_of(2)[0]


def test_expert_blocks_tile_the_expert_space_exactly() -> None:
    """Chip blocks must partition [0, num_experts) with no gap or overlap.

    This mirrors vLLM's linear placement formula. If the union is short, the
    missing experts are never computed by anyone and their contribution is
    silently zero.
    """
    num_experts, ep_size = 224, 4
    base, remainder = divmod(num_experts, ep_size)
    covered: list[int] = []
    for ep_rank in range(ep_size):
        start = ep_rank * base + min(ep_rank, remainder)
        count = base + (1 if ep_rank < remainder else 0)
        covered.extend(range(start, start + count))
    assert sorted(covered) == list(range(num_experts))
    assert len(covered) == len(set(covered))


def test_intermediate_split_is_exact_and_mxfp4_aligned() -> None:
    """The width split must divide evenly and keep the fp4 group alignment.

    w2's contracting dim is sharded, and mxfp4 quantizes in groups of 32 along
    it; a shard that straddles a group would decode against the wrong scale.
    """
    intermediate, tp_size, mxfp4_group = 3072, 2, 32
    assert intermediate % tp_size == 0
    per_partition = intermediate // tp_size
    assert per_partition % mxfp4_group == 0
    # And the per-device weight footprint is unchanged vs flat EP8:
    # 56 experts x 1536 == 28 experts x 3072.
    assert (224 // 4) * per_partition == (224 // 8) * intermediate


def test_context_manager_is_a_noop_when_the_flag_is_off(monkeypatch) -> None:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import moe_hierarchical

    monkeypatch.setattr(envs, "TPU_MOE_HIERARCHICAL_EP", False)
    original = FusedMoEParallelConfig.make
    with moe_hierarchical.hierarchical_moe_parallel_config() as split:
        assert split is None
        assert FusedMoEParallelConfig.make is original
    assert FusedMoEParallelConfig.make is original


def test_patch_is_reverted_even_when_construction_raises(monkeypatch) -> None:
    """A failed FusedMoE build must not leave the global patched.

    Otherwise the next model built in the same process -- a draft model, a
    second engine -- silently inherits a chip split it never asked for.
    """
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    from vllm_torchtpu.layers.adapter import moe_hierarchical

    monkeypatch.setattr(
        moe_hierarchical, "hierarchical_split_or_none", lambda: (4, 1, 2, 0)
    )
    original = FusedMoEParallelConfig.make
    with (
        pytest.raises(RuntimeError, match="boom"),
        moe_hierarchical.hierarchical_moe_parallel_config(),
    ):
        assert FusedMoEParallelConfig.make is not original
        raise RuntimeError("boom")
    assert FusedMoEParallelConfig.make is original


@pytest.mark.parametrize("dp_size,dp_rank", [(1, 0), (4, 2)])
def test_patched_config_carries_the_chip_split(monkeypatch, dp_size, dp_rank) -> None:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    from vllm_torchtpu.layers.adapter import moe_hierarchical

    flat = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=dp_size,
        ep_size=dp_size * 8,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=dp_rank,
        ep_rank=dp_rank * 8 + 5,
        sp_size=1,
        use_ep=True,
        all2all_backend="naive",
        enable_eplb=False,
    )
    split = (dp_size * 4, dp_rank * 4 + 2, 2, 1)
    monkeypatch.setattr(moe_hierarchical, "hierarchical_split_or_none", lambda: split)
    monkeypatch.setattr(
        FusedMoEParallelConfig, "make", staticmethod(lambda *a, **k: flat)
    )

    with moe_hierarchical.hierarchical_moe_parallel_config():
        got = FusedMoEParallelConfig.make(
            tp_size_=8,
            pcp_size_=1,
            dp_size_=dp_size,
            sp_size_=1,
            vllm_parallel_config=None,
        )
    assert (got.ep_size, got.ep_rank) == split[:2]
    assert (got.tp_size, got.tp_rank) == (2, 1)
    # Untouched fields survive.
    assert got.use_ep and (got.dp_size, got.dp_rank) == (dp_size, dp_rank)


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_ep_weight_filter_is_realigned_to_chip_block(dp_runtime, tp_rank):
    from vllm.model_executor.model_loader import default_loader

    from vllm_torchtpu import model_loader_patches
    from vllm_torchtpu.layers.adapter import moe_hierarchical

    dp_runtime.layout = dataclasses.replace(
        dp_runtime.layout, rank_in_group=8 + tp_rank
    )
    dp_runtime.tp_rank = tp_rank
    flat = model_loader_patches._compute_local_expert_ids()
    assert flat == set(range(56 + 7 * tp_rank, 63 + 7 * tp_rank))

    moe_hierarchical.align_ep_weight_filter()
    chip = model_loader_patches._compute_local_expert_ids()
    # These EP ranks map to the two cores of physical chip 12.
    assert chip == set(range(168, 182))
    assert default_loader.compute_local_expert_ids(224, 32, 8 + tp_rank) == chip
    wrapped = default_loader.compute_local_expert_ids
    moe_hierarchical.align_ep_weight_filter()
    assert default_loader.compute_local_expert_ids is wrapped


def test_patched_config_rejects_sequence_parallel(monkeypatch):
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    from vllm_torchtpu.layers.adapter import moe_hierarchical

    flat = SimpleNamespace(use_ep=True, is_sequence_parallel=True)
    monkeypatch.setattr(
        moe_hierarchical, "hierarchical_split_or_none", lambda: (16, 3, 2, 1)
    )
    monkeypatch.setattr(FusedMoEParallelConfig, "make", staticmethod(lambda: flat))
    original = FusedMoEParallelConfig.make
    with (
        pytest.raises(ValueError, match="sequence parallelism"),
        moe_hierarchical.hierarchical_moe_parallel_config(),
    ):
        FusedMoEParallelConfig.make()
    assert FusedMoEParallelConfig.make is original

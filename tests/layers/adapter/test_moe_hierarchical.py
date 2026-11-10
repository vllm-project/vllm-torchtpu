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

import pytest

from vllm_torchtpu.distributed.chip_topology import ChipTopology


@pytest.fixture(autouse=True)
def _explicit_ep_tp_env(monkeypatch):
    """Tests here set TPU_MOE_EP_TP_SIZE on the envs module; mark it explicit so
    the default's GLM-only gating (moe_hierarchical.ep_tp_size) stays out of it."""
    from vllm_torchtpu.layers.adapter import moe_hierarchical

    monkeypatch.setenv("TPU_MOE_EP_TP_SIZE", "1")
    monkeypatch.setattr(moe_hierarchical, "_default_ep_tp", None)


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


def test_patched_config_carries_the_chip_split(monkeypatch) -> None:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    from vllm_torchtpu.layers.adapter import moe_hierarchical

    flat = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=8,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=5,
        sp_size=1,
        use_ep=True,
        all2all_backend="naive",
        enable_eplb=False,
    )
    monkeypatch.setattr(
        moe_hierarchical, "hierarchical_split_or_none", lambda: (4, 2, 2, 1)
    )
    monkeypatch.setattr(
        FusedMoEParallelConfig, "make", staticmethod(lambda *a, **k: flat)
    )

    with moe_hierarchical.hierarchical_moe_parallel_config():
        got = FusedMoEParallelConfig.make(
            tp_size_=8, pcp_size_=1, dp_size_=1, sp_size_=1, vllm_parallel_config=None
        )
    assert (got.ep_size, got.ep_rank) == (4, 2)
    assert (got.tp_size, got.tp_rank) == (2, 1)
    # Untouched fields survive.
    assert got.use_ep and got.dp_size == 1


def test_flat_tp_config_is_left_alone(monkeypatch) -> None:
    """With EP off there is no expert split to make hierarchical.

    Forcing tp_size=2 onto a config that shards experts by TP would change what
    the flat layout means and mis-shard the weights.
    """
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    from vllm_torchtpu.layers.adapter import moe_hierarchical

    flat = FusedMoEParallelConfig(
        tp_size=8,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=3,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="naive",
        enable_eplb=False,
    )
    monkeypatch.setattr(
        moe_hierarchical, "hierarchical_split_or_none", lambda: (4, 2, 2, 1)
    )
    monkeypatch.setattr(
        FusedMoEParallelConfig, "make", staticmethod(lambda *a, **k: flat)
    )
    with moe_hierarchical.hierarchical_moe_parallel_config():
        got = FusedMoEParallelConfig.make(
            tp_size_=8, pcp_size_=1, dp_size_=1, sp_size_=1, vllm_parallel_config=None
        )
    assert dataclasses.asdict(got) == dataclasses.asdict(flat)


def test_ep_weight_filter_is_realigned_to_chip_block(monkeypatch) -> None:
    """The loader must read a chip's whole expert block, not a rank's half.

    DefaultModelLoader derives ownership as ep_size = dp*pcp*tp / ep_rank =
    tp_rank, independent of the layer's MoE config. Under the chip split that
    hands rank 1 (core 1 of chip 0) only experts 28..55 when it needs 0..55 --
    and the shortfall never raises, it just leaves those experts at their
    initialized values.
    """
    from vllm.model_executor.model_loader import default_loader

    from vllm_torchtpu.layers.adapter import moe_hierarchical

    monkeypatch.setattr(moe_hierarchical, "_filter_patched", False)
    # Rank 1 under flat EP8 would be ep_rank=1 of 8; under the chip split it is
    # core 1 of chip 0, so ep_rank=0 of 4.
    monkeypatch.setattr(
        moe_hierarchical, "hierarchical_split_or_none", lambda: (4, 0, 2, 1)
    )
    # The pristine function, not whatever a previous test left bound: the
    # context manager patches the loader's namespace permanently (it is a
    # one-time init in production), so `default_loader.compute_local_expert_ids`
    # may already be wrapped by the time this runs.
    from vllm.model_executor.model_loader import ep_weight_filter

    pristine = ep_weight_filter.compute_local_expert_ids
    monkeypatch.setattr(default_loader, "compute_local_expert_ids", pristine)

    flat = pristine(224, 8, 1)  # what the loader would have computed
    moe_hierarchical.align_ep_weight_filter()
    chip = default_loader.compute_local_expert_ids(224, 8, 1)

    assert flat == set(range(28, 56))
    assert chip == set(range(0, 56)), "must cover the whole chip block"
    assert flat < chip, "the flat set is a strict subset -- the missing half"
    # Both chiplets of a chip must request the same experts.
    monkeypatch.setattr(moe_hierarchical, "_filter_patched", False)
    monkeypatch.setattr(
        moe_hierarchical, "hierarchical_split_or_none", lambda: (4, 0, 2, 0)
    )
    moe_hierarchical.align_ep_weight_filter()
    assert default_loader.compute_local_expert_ids(224, 8, 0) == chip


# --- TPU_MOE_EP_TP_SIZE: EP between groups of N consecutive ranks, TP inside ---


def _flat_ep_config(ep_size: int, ep_rank: int):
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    return FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=ep_size,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=ep_rank,
        sp_size=1,
        use_ep=True,
        all2all_backend="naive",
        enable_eplb=False,
    )


def test_ep_tp_split_groups_consecutive_ranks() -> None:
    from vllm_torchtpu.layers.adapter.moe_hierarchical import ep_tp_split

    # GLM-5.2 on 2x2x2: 16 ranks, EP4 x TP4.
    got = [ep_tp_split(16, r, 4) for r in range(16)]
    assert got[0] == (4, 0, 4, 0)
    assert got[6] == (4, 1, 4, 2)
    assert got[15] == (4, 3, 4, 3)
    # Every (ep_rank, tp_rank) cell is owned by exactly one rank.
    assert len({(g[1], g[3]) for g in got}) == 16
    # A TP group shares one ep_rank, so it reads one expert block.
    for ep_rank in range(4):
        assert {g[1] for g in got[4 * ep_rank : 4 * ep_rank + 4]} == {ep_rank}


def test_ep_tp_split_refuses_a_size_that_does_not_divide() -> None:
    from vllm_torchtpu.layers.adapter.moe_hierarchical import ep_tp_split

    assert ep_tp_split(16, 3, 3) is None
    assert ep_tp_split(16, 3, 1) is None


def test_ep_tp_weight_bytes_per_rank_are_unchanged() -> None:
    """EP16 whole experts and EP4 x TP4 quarter experts hold the same bytes."""
    experts, intermediate = 256, 2048
    for tp in (1, 2, 4, 8, 16):
        local_experts = experts // (16 // tp)
        per_partition = intermediate // tp
        assert per_partition % 128 == 0, "block-FP8 scales need 128-aligned slices"
        assert local_experts * per_partition == (experts // 16) * intermediate


def test_split_reads_the_ep_tp_env(monkeypatch) -> None:
    import vllm.distributed as vd

    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import moe_hierarchical

    monkeypatch.setattr(envs, "TPU_MOE_HIERARCHICAL_EP", False)
    monkeypatch.setattr(envs, "TPU_MOE_EP_TP_SIZE", 4)
    monkeypatch.setattr(vd, "get_tensor_model_parallel_world_size", lambda: 16)
    monkeypatch.setattr(vd, "get_tensor_model_parallel_rank", lambda: 6)
    assert moe_hierarchical.hierarchical_split_or_none() == (4, 1, 4, 2)

    monkeypatch.setattr(envs, "TPU_MOE_EP_TP_SIZE", 1)
    assert moe_hierarchical.hierarchical_split_or_none() is None

    monkeypatch.setattr(envs, "TPU_MOE_EP_TP_SIZE", 0)
    assert moe_hierarchical.hierarchical_split_or_none() is None


def test_both_split_envs_together_are_refused(monkeypatch) -> None:
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import moe_hierarchical

    monkeypatch.setattr(envs, "TPU_MOE_HIERARCHICAL_EP", True)
    monkeypatch.setattr(envs, "TPU_MOE_EP_TP_SIZE", 4)
    with pytest.raises(ValueError, match="set one"):
        moe_hierarchical.hierarchical_split_or_none()


def test_ep_tp_patch_is_inert_when_off(monkeypatch) -> None:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import moe_hierarchical

    monkeypatch.setattr(envs, "TPU_MOE_EP_TP_SIZE", 1)
    # Start unpatched: an earlier test may have installed the real patch.
    monkeypatch.setattr(
        FusedMoEParallelConfig, "_tpu_ep_tp_split_patch", False, raising=False
    )
    original = FusedMoEParallelConfig.make
    moe_hierarchical.install_ep_tp_split_patch()
    assert FusedMoEParallelConfig.make is original
    assert not getattr(FusedMoEParallelConfig, "_tpu_ep_tp_split_patch", False)


def test_ep_tp_patch_rewrites_every_make(monkeypatch) -> None:
    """Upstream models (DeepSeek-V2/V3, GLM-5) build FusedMoE with no hook:
    the registry patch must reach every make, the drafter's included."""
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
    from vllm.model_executor.model_loader import default_loader, ep_weight_filter

    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import moe_hierarchical

    flat = _flat_ep_config(16, 6)
    monkeypatch.setattr(envs, "TPU_MOE_EP_TP_SIZE", 4)
    monkeypatch.setattr(
        FusedMoEParallelConfig, "make", staticmethod(lambda *a, **k: flat)
    )
    monkeypatch.setattr(
        FusedMoEParallelConfig, "_tpu_ep_tp_split_patch", False, raising=False
    )
    monkeypatch.setattr(moe_hierarchical, "_filter_patched", False)
    monkeypatch.setattr(
        default_loader,
        "compute_local_expert_ids",
        ep_weight_filter.compute_local_expert_ids,
    )
    monkeypatch.setattr(
        moe_hierarchical, "hierarchical_split_or_none", lambda: (4, 1, 4, 2)
    )

    moe_hierarchical.install_ep_tp_split_patch()
    for _ in range(2):  # target model and drafter
        got = FusedMoEParallelConfig.make(
            tp_size_=16, pcp_size_=1, dp_size_=1, sp_size_=1, vllm_parallel_config=None
        )
        assert (got.ep_size, got.ep_rank, got.tp_size, got.tp_rank) == (4, 1, 4, 2)
        assert got.use_ep
    # Idempotent: a second install does not wrap twice.
    patched = FusedMoEParallelConfig.make
    moe_hierarchical.install_ep_tp_split_patch()
    assert FusedMoEParallelConfig.make is patched

    # The loader's rank-derived EP16 ownership (rank 6 -> experts 96..111)
    # becomes the group's block (ep_rank 1 of 4 -> experts 64..127).
    assert default_loader.compute_local_expert_ids(256, 16, 6) == set(range(64, 128))


def test_ep_tp_patch_leaves_a_flat_world_flat(monkeypatch) -> None:
    """A world the size does not divide keeps vLLM's own config."""
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import moe_hierarchical

    flat = _flat_ep_config(16, 6)
    monkeypatch.setattr(envs, "TPU_MOE_EP_TP_SIZE", 3)
    monkeypatch.setattr(
        FusedMoEParallelConfig, "make", staticmethod(lambda *a, **k: flat)
    )
    monkeypatch.setattr(
        FusedMoEParallelConfig, "_tpu_ep_tp_split_patch", False, raising=False
    )
    monkeypatch.setattr(moe_hierarchical, "_filter_patched", True)
    monkeypatch.setattr(moe_hierarchical, "hierarchical_split_or_none", lambda: None)
    moe_hierarchical.install_ep_tp_split_patch()
    got = FusedMoEParallelConfig.make(
        tp_size_=16, pcp_size_=1, dp_size_=1, sp_size_=1, vllm_parallel_config=None
    )
    assert dataclasses.asdict(got) == dataclasses.asdict(flat)


def _vllm_config(
    arch="GlmMoeDsaForCausalLM", ep=True, dp=1, pcp=1, eplb=False, sp=False, tp=16
):
    from types import SimpleNamespace

    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch])),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=ep,
            data_parallel_size=dp,
            prefill_context_parallel_size=pcp,
            enable_eplb=eplb,
            use_sequence_parallel_moe=sp,
            tensor_parallel_size=tp,
        ),
    )


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, 4),
        ({"arch": "KimiK3ForCausalLM"}, 1),
        ({"ep": False}, 1),
        ({"dp": 2}, 1),
        ({"pcp": 2}, 1),
        ({"eplb": True}, 1),
        ({"sp": True}, 1),
        ({"tp": 6}, 1),
    ],
)
def test_default_ep_tp_applies_only_where_measured(monkeypatch, kwargs, expected):
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import moe_hierarchical

    monkeypatch.setattr(envs, "TPU_MOE_HIERARCHICAL_EP", False)
    monkeypatch.setattr(envs, "TPU_MOE_EP_TP_SIZE", 4)
    assert moe_hierarchical._default_ep_tp_for(_vllm_config(**kwargs)) == expected


def test_shipped_default_ep_tp_splits_glm(monkeypatch):
    """The shipped default splits the experts EP x TP where the default
    applies, and an explicit 1 keeps flat EP."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import moe_hierarchical

    monkeypatch.delenv("TPU_MOE_EP_TP_SIZE", raising=False)
    monkeypatch.setattr(envs, "TPU_MOE_HIERARCHICAL_EP", False)
    assert envs.environment_variables["TPU_MOE_EP_TP_SIZE"]() == 2
    monkeypatch.setattr(
        envs, "TPU_MOE_EP_TP_SIZE", envs.environment_variables["TPU_MOE_EP_TP_SIZE"]()
    )
    assert moe_hierarchical._default_ep_tp_for(_vllm_config()) == 2

    monkeypatch.setenv("TPU_MOE_EP_TP_SIZE", "1")
    monkeypatch.setattr(envs, "TPU_MOE_EP_TP_SIZE", 1)
    assert moe_hierarchical.ep_tp_size() == 1


def test_default_ep_tp_yields_to_hierarchical_ep(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import moe_hierarchical

    monkeypatch.setattr(envs, "TPU_MOE_HIERARCHICAL_EP", True)
    monkeypatch.setattr(envs, "TPU_MOE_EP_TP_SIZE", 4)
    assert moe_hierarchical._default_ep_tp_for(_vllm_config()) == 1


def test_ep_tp_size_explicit_wins_and_default_is_decided_once(monkeypatch):
    import vllm.config as vllm_config

    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import moe_hierarchical

    monkeypatch.setattr(envs, "TPU_MOE_HIERARCHICAL_EP", False)
    monkeypatch.setenv("TPU_MOE_EP_TP_SIZE", "2")
    monkeypatch.setattr(envs, "TPU_MOE_EP_TP_SIZE", 2)
    assert moe_hierarchical.ep_tp_size() == 2

    monkeypatch.delenv("TPU_MOE_EP_TP_SIZE")
    monkeypatch.setattr(envs, "TPU_MOE_EP_TP_SIZE", 4)
    no_model = _vllm_config()
    no_model.model_config = None
    monkeypatch.setattr(
        vllm_config, "get_current_vllm_config_or_none", lambda: no_model
    )
    assert moe_hierarchical.ep_tp_size() == 1  # undecided: no model yet
    monkeypatch.setattr(vllm_config, "get_current_vllm_config_or_none", _vllm_config)
    assert moe_hierarchical.ep_tp_size() == 4
    flat = _vllm_config(ep=False)
    monkeypatch.setattr(vllm_config, "get_current_vllm_config_or_none", lambda: flat)
    assert moe_hierarchical.ep_tp_size() == 4  # decided at the first FusedMoE


def test_experts_start_follows_the_group_block() -> None:
    """The kernel's global->local remap reads ep_rank/ep_size from the layer's
    MoE config, so under EP4 x TP4 all four ranks of a group start at 64*g."""
    from types import SimpleNamespace

    from vllm_torchtpu.layers.adapter.moe_hierarchical import ep_tp_split
    from vllm_torchtpu.layers.adapter.moe_routing import get_experts_start

    for rank in range(16):
        ep_size, ep_rank, _, _ = ep_tp_split(16, rank, 4)
        layer = SimpleNamespace(
            global_num_experts=256,
            moe_config=SimpleNamespace(
                ep_size=ep_size,
                ep_rank=ep_rank,
                moe_parallel_config=SimpleNamespace(use_ep=True),
            ),
        )
        assert get_experts_start(layer) == 64 * (rank // 4)


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("dp_size", 2, "DP=1"),
        ("pcp_size", 2, "PCP=1"),
        ("enable_eplb", True, "EPLB"),
        ("ep_size", 8, "flat EP config"),
    ],
)
def test_ep_tp_patch_refuses_where_partials_would_not_meet(
    monkeypatch, field, value, match
) -> None:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import moe_hierarchical

    bad = dataclasses.replace(_flat_ep_config(16, 6), **{field: value})
    monkeypatch.setattr(envs, "TPU_MOE_EP_TP_SIZE", 4)
    monkeypatch.setattr(
        FusedMoEParallelConfig, "make", staticmethod(lambda *a, **k: bad)
    )
    monkeypatch.setattr(
        FusedMoEParallelConfig, "_tpu_ep_tp_split_patch", False, raising=False
    )
    monkeypatch.setattr(moe_hierarchical, "_filter_patched", True)
    monkeypatch.setattr(
        moe_hierarchical, "hierarchical_split_or_none", lambda: (4, 1, 4, 2)
    )
    moe_hierarchical.install_ep_tp_split_patch()
    with pytest.raises(ValueError, match=match):
        FusedMoEParallelConfig.make(
            tp_size_=16, pcp_size_=1, dp_size_=1, sp_size_=1, vllm_parallel_config=None
        )

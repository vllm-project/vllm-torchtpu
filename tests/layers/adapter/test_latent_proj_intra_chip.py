# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Intra-chip sharding of the latent-MoE projections.

Like the hierarchical-EP tests, everything here is arithmetic that needs no
TPU -- and everything here fails *silently* when wrong. A shard assigned by the
wrong index, or gathered in the wrong order, still produces a tensor of exactly
the right shape holding permuted numbers.
"""

from __future__ import annotations

import collections

import pytest
import torch


class _FakeDevice:
    def __init__(self, device_id: int, coords: tuple[int, int, int]) -> None:
        self.id = device_id
        self.coords = coords


def _tpu7x_two_hosts() -> list[_FakeDevice]:
    """8 chips x 2 cores, device ids interleaved across chips."""
    devices = []
    for chip in range(8):
        coords = (chip % 2, chip // 2, 0)
        devices.append(_FakeDevice(2 * chip, coords))
        devices.append(_FakeDevice(2 * chip + 1, coords))
    return devices


def _topology_from(devices):
    from vllm_torchtpu.distributed.chip_topology import ChipTopology

    grouped = collections.defaultdict(list)
    for device in devices:
        grouped[tuple(device.coords)].append(int(device.id))
    chips = [sorted(grouped[key]) for key in sorted(grouped)]
    return ChipTopology(chips, [int(d.id) for d in devices])


# --------------------------------------------------------------------------
# The sharding arithmetic: what the gather has to undo.
# --------------------------------------------------------------------------


def test_column_shards_gathered_in_rank_order_reconstruct_the_full_matmul():
    """The whole correctness claim of the change, in one assertion.

    ``ColumnParallelLinear`` narrows the weight's output dim at
    ``tp_rank * shard``; the all-gather concatenates in group-rank order. Those
    two orders must be the same one. If they are not, every latent projection
    in the model returns its output halves swapped -- right dtype, right shape,
    wrong model.
    """
    torch.manual_seed(0)
    in_features, out_features, cores = 7168, 3584, 2
    weight = torch.randn(out_features, in_features, dtype=torch.float32)
    x = torch.randn(2, in_features, dtype=torch.float32)

    reference = torch.nn.functional.linear(x, weight)

    shard = out_features // cores
    partials = [
        torch.nn.functional.linear(
            x, weight.narrow(0, rank * shard, shard)
        )  # what rank `rank` loads
        for rank in range(cores)
    ]
    gathered = torch.cat(partials, dim=-1)  # what all_gather(dim=-1) builds

    assert gathered.shape == reference.shape
    torch.testing.assert_close(gathered, reference)


def test_swapped_gather_order_is_not_caught_by_shape():
    """Pins *why* the previous test matters rather than being a tautology."""
    torch.manual_seed(0)
    weight = torch.randn(8, 4)
    x = torch.randn(2, 4)
    reference = torch.nn.functional.linear(x, weight)

    partials = [
        torch.nn.functional.linear(x, weight.narrow(0, r * 4, 4)) for r in range(2)
    ]
    swapped = torch.cat(list(reversed(partials)), dim=-1)

    assert swapped.shape == reference.shape  # shape check would pass
    assert not torch.allclose(swapped, reference)  # numbers would not


def test_column_parallel_is_exact_where_row_parallel_would_resplit_the_sum():
    """Why column-parallel and not row-parallel.

    Column-parallel splits the *output*, so each element is still accumulated
    over the full contraction on one core and the result is bit-identical to
    the replicated layer. Row-parallel would split the contraction and re-sum
    across cores, changing the reduction order. Bit-exactness is what lets this
    ship without re-validating accuracy on every model.
    """
    torch.manual_seed(0)
    weight = torch.randn(64, 128, dtype=torch.bfloat16)
    x = torch.randn(2, 128, dtype=torch.bfloat16)
    reference = torch.nn.functional.linear(x, weight)

    column = torch.cat(
        [torch.nn.functional.linear(x, weight.narrow(0, r * 32, 32)) for r in range(2)],
        dim=-1,
    )
    assert torch.equal(column, reference), "column-parallel must be bit-exact"

    row = sum(
        torch.nn.functional.linear(
            x.narrow(1, r * 64, 64), weight.narrow(1, r * 64, 64)
        )
        for r in range(2)
    )
    # Not an error, just not bit-exact -- which is the point.
    assert row.shape == reference.shape


def test_all_gather_moves_less_than_an_all_reduce_would():
    """The bandwidth argument behind choosing a gather over a reduce.

    Per core, per layer, at the real K3 dimensions.
    """
    hidden, latent, cores = 7168, 3584, 2
    bf16 = 2

    # Column-parallel: each core contributes its own shard once.
    gather_bytes = (hidden // cores + latent // cores) * bf16
    # Row-parallel: each core sends a full-width partial to be summed.
    reduce_bytes = (hidden + latent) * bf16

    assert gather_bytes * cores == reduce_bytes
    assert gather_bytes < reduce_bytes


# --------------------------------------------------------------------------
# Grouping: which ranks share a chip.
# --------------------------------------------------------------------------


def test_chip_groups_partition_the_ranks(monkeypatch):
    """Every rank in exactly one group, every group one chip's worth."""
    from vllm_torchtpu.distributed import intra_chip

    topology = _topology_from(_tpu7x_two_hosts())
    monkeypatch.setattr(intra_chip, "get_chip_topology", lambda: topology)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        intra_chip, "_rank_to_device", lambda: {r: r for r in range(16)}
    )

    groups = intra_chip._chip_rank_groups()

    assert groups is not None
    assert len(groups) == 8
    assert all(len(g) == 2 for g in groups)
    flat = sorted(r for g in groups for r in g)
    assert flat == list(range(16)), "ranks must be partitioned, not sampled"


def test_grouping_follows_each_ranks_device_not_its_rank(monkeypatch):
    """Under DP, global rank does not index the device order.

    Grouping by rank position would pair cores from different chips -- the
    collective would still succeed and would still be the wrong two cores.
    """
    from vllm_torchtpu.distributed import intra_chip

    topology = _topology_from(_tpu7x_two_hosts())
    # Each chip's two cores land on non-adjacent ranks.
    rank_to_device = {r: (r % 8) * 2 + r // 8 for r in range(16)}
    monkeypatch.setattr(intra_chip, "get_chip_topology", lambda: topology)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(intra_chip, "_rank_to_device", lambda: rank_to_device)

    groups = intra_chip._chip_rank_groups()

    assert groups is not None
    chips = {frozenset(chip) for chip in topology.chips}
    for group in groups:
        assert frozenset(rank_to_device[r] for r in group) in chips
    assert sorted(groups) == [[r, r + 8] for r in range(8)]


@pytest.mark.parametrize(
    "rank_to_device",
    [
        {r: (99 if r == 15 else r) for r in range(16)},  # a device on no chip
        {r: r for r in range(4)},  # whole chips, but not the whole topology
        {r: r for r in range(3)},  # half of a chip
        {r: r // 2 * 2 for r in range(16)},  # two ranks on one device
    ],
    ids=["unknown-device", "subset-of-chips", "split-chip", "shared-device"],
)
def test_grouping_refuses_a_world_that_is_not_the_topology(monkeypatch, rank_to_device):
    """The ranks must run on exactly the topology's devices, one each."""
    from vllm_torchtpu.distributed import intra_chip

    topology = _topology_from(_tpu7x_two_hosts())
    monkeypatch.setattr(intra_chip, "get_chip_topology", lambda: topology)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(intra_chip, "_rank_to_device", lambda: rank_to_device)

    assert intra_chip._chip_rank_groups() is None


def test_grouping_declines_single_core_chips(monkeypatch):
    from vllm_torchtpu.distributed import intra_chip
    from vllm_torchtpu.distributed.chip_topology import ChipTopology

    single = ChipTopology([[0], [1]], [0, 1])
    monkeypatch.setattr(intra_chip, "get_chip_topology", lambda: single)

    assert intra_chip._chip_rank_groups() is None


def test_grouping_declines_an_unreadable_topology(monkeypatch):
    from vllm_torchtpu.distributed import intra_chip

    monkeypatch.setattr(intra_chip, "get_chip_topology", lambda: None)
    assert intra_chip._chip_rank_groups() is None


# --------------------------------------------------------------------------
# The fallbacks. Every one of these must leave the model as it was.
# --------------------------------------------------------------------------


class _FakeGroup:
    def __init__(self, world_size: int, rank_in_group: int = 0) -> None:
        self.world_size = world_size
        self.rank_in_group = rank_in_group


@pytest.mark.parametrize(
    "flag,group,output_size,why",
    [
        (False, _FakeGroup(2), 3584, "flag off"),
        (True, None, 3584, "no usable chip group"),
        (True, _FakeGroup(1), 3584, "one core per chip"),
        (True, _FakeGroup(3), 3584, "does not divide evenly"),
    ],
)
def test_declines_to_shard(monkeypatch, flag, group, output_size, why):
    """Anything unclear must leave the projection replicated.

    Decision only: building the layer itself needs an initialized TP group,
    and it is the decision that has the failure modes.
    """
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import latent_proj_intra_chip as mod

    monkeypatch.setattr(envs, "TPU_LATENT_PROJ_INTRA_CHIP_TP", flag)
    monkeypatch.setattr(mod, "get_intra_chip_group", lambda: group)
    monkeypatch.setattr(mod, "_warned", False)

    assert mod.shard_group_or_none(output_size) is None, why


def test_shards_when_everything_lines_up(monkeypatch):
    """The one case that must *not* fall back, so the test above
    can't pass vacuously."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import latent_proj_intra_chip as mod

    group = _FakeGroup(2, rank_in_group=1)
    monkeypatch.setattr(envs, "TPU_LATENT_PROJ_INTRA_CHIP_TP", True)
    monkeypatch.setattr(mod, "get_intra_chip_group", lambda: group)

    assert mod.shard_group_or_none(3584) is group
    assert mod.shard_group_or_none(7168) is group


def _patch_tp_state(
    monkeypatch, mod, *, initialized: bool, group, cores_per_chip: int
) -> None:
    """Replace the module's view of vLLM's parallel state and of the chip."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        mod,
        "parallel_state",
        SimpleNamespace(
            model_parallel_is_initialized=lambda: initialized,
            get_tp_group=lambda: group,
        ),
    )
    monkeypatch.setattr(mod, "get_num_cores_per_chip", lambda: cores_per_chip)


@pytest.mark.parametrize(
    "initialized,group,cores_per_chip,why",
    [
        (False, _FakeGroup(4), 1, "model parallel not initialized"),
        (True, None, 1, "no TP group"),
        (True, _FakeGroup(1), 1, "TP world size 1"),
        (True, _FakeGroup(4), 2, "two cores per chip: TP spans chips"),
    ],
)
def test_tp_shard_declines_silently(
    monkeypatch, initialized, group, cores_per_chip, why
):
    """Expected states of an always-on path: replicate without a warning."""
    from vllm_torchtpu.layers.adapter import latent_proj_intra_chip as mod

    _patch_tp_state(
        monkeypatch,
        mod,
        initialized=initialized,
        group=group,
        cores_per_chip=cores_per_chip,
    )
    monkeypatch.setattr(mod, "_warned", False)

    assert mod.tp_shard_group_or_none(3584) is None, why
    assert mod._warned is False, why


def test_tp_shard_declines_uneven_output_with_one_warning(monkeypatch):
    from vllm_torchtpu.layers.adapter import latent_proj_intra_chip as mod

    _patch_tp_state(
        monkeypatch, mod, initialized=True, group=_FakeGroup(3), cores_per_chip=1
    )
    monkeypatch.setattr(mod, "_warned", False)
    warnings: list[str] = []
    monkeypatch.setattr(
        mod.logger, "warning", lambda message, *args: warnings.append(message)
    )

    assert mod.tp_shard_group_or_none(3584) is None
    assert mod.tp_shard_group_or_none(3584) is None
    assert len(warnings) == 1


def test_tp_shard_uses_the_tp_group_on_one_core_chips(monkeypatch):
    from vllm_torchtpu.layers.adapter import latent_proj_intra_chip as mod

    group = _FakeGroup(16, rank_in_group=3)
    _patch_tp_state(monkeypatch, mod, initialized=True, group=group, cores_per_chip=1)

    assert mod.tp_shard_group_or_none(3584) is group


class _RecordedSharded:
    """Stands in for the sharded layer, whose constructor needs a real TP group."""

    def __init__(self, input_size, output_size, *, bias, quant_config, prefix, group):
        self.shard_group = group


class _RecordedReplicated:
    """Stands in for ReplicatedLinear, whose constructor queries vLLM TP state."""

    def __init__(self, input_size, output_size, *, bias, quant_config, prefix):
        self.input_size = input_size
        self.output_size = output_size


def _make_projection(
    monkeypatch, mod, *, flag: bool, chip_group, output_size: int = 16
):
    """Run ``make_latent_projection`` and return ``(layer, warning messages)``."""
    from vllm_torchtpu import envs

    warnings: list[str] = []
    monkeypatch.setattr(envs, "TPU_LATENT_PROJ_INTRA_CHIP_TP", flag)
    monkeypatch.setattr(mod, "get_intra_chip_group", lambda: chip_group)
    monkeypatch.setattr(mod, "IntraChipColumnParallelLinear", _RecordedSharded)
    monkeypatch.setattr(mod, "ReplicatedLinear", _RecordedReplicated)
    monkeypatch.setattr(mod, "_warned", False)
    monkeypatch.setattr(
        mod.logger, "warning", lambda message, *args: warnings.append(message)
    )
    layer = mod.make_latent_projection(
        32, output_size, bias=False, quant_config=None, prefix="proj"
    )
    return layer, warnings


def test_flag_on_a_one_core_chip_shards_over_tp_without_a_false_warning(monkeypatch):
    """The flag asks for an intra-chip group a one-core chip cannot have.

    The projections are sharded over the TP group anyway, so nothing may claim
    they were left replicated.
    """
    from vllm_torchtpu.layers.adapter import latent_proj_intra_chip as mod

    tp_group = _FakeGroup(16, rank_in_group=3)
    _patch_tp_state(
        monkeypatch, mod, initialized=True, group=tp_group, cores_per_chip=1
    )

    layer, warnings = _make_projection(monkeypatch, mod, flag=True, chip_group=None)

    assert isinstance(layer, _RecordedSharded)
    assert layer.shard_group is tp_group
    assert warnings == []


def test_one_core_chip_indivisible_output_falls_back_to_replicated(monkeypatch):
    from vllm_torchtpu.layers.adapter import latent_proj_intra_chip as mod

    _patch_tp_state(
        monkeypatch, mod, initialized=True, group=_FakeGroup(3), cores_per_chip=1
    )

    layer, warnings = _make_projection(
        monkeypatch, mod, flag=False, chip_group=None, output_size=16
    )

    assert isinstance(layer, _RecordedReplicated)
    assert len(warnings) == 1


def test_multi_core_chip_prefers_the_chip_group_over_the_tp_group(monkeypatch):
    from vllm_torchtpu.layers.adapter import latent_proj_intra_chip as mod

    chip_group = _FakeGroup(2, rank_in_group=1)
    _patch_tp_state(
        monkeypatch, mod, initialized=True, group=_FakeGroup(16), cores_per_chip=2
    )

    layer, warnings = _make_projection(
        monkeypatch, mod, flag=True, chip_group=chip_group
    )

    assert isinstance(layer, _RecordedSharded)
    assert layer.shard_group is chip_group
    assert warnings == []


def test_replicating_across_a_multi_rank_tp_group_warns_once(monkeypatch):
    """A chip not known as one-core keeps the replicated default, and says so."""
    from vllm_torchtpu.layers.adapter import latent_proj_intra_chip as mod

    _patch_tp_state(
        monkeypatch, mod, initialized=True, group=_FakeGroup(16), cores_per_chip=2
    )

    layer, warnings = _make_projection(monkeypatch, mod, flag=False, chip_group=None)
    mod.make_latent_projection(32, 16, bias=False, quant_config=None, prefix="proj")

    assert isinstance(layer, _RecordedReplicated)
    assert len(warnings) == 1
    assert "replicated" in warnings[0]


def test_replicating_without_a_multi_rank_tp_group_is_silent(monkeypatch):
    from vllm_torchtpu.layers.adapter import latent_proj_intra_chip as mod

    _patch_tp_state(
        monkeypatch, mod, initialized=True, group=_FakeGroup(1), cores_per_chip=2
    )

    layer, warnings = _make_projection(monkeypatch, mod, flag=False, chip_group=None)

    assert isinstance(layer, _RecordedReplicated)
    assert warnings == []


def test_intra_chip_linear_exposes_public_shard_group_and_all_gathers():
    from types import SimpleNamespace

    from vllm_torchtpu.layers.adapter.latent_proj_intra_chip import (
        IntraChipColumnParallelLinear,
    )

    gathered_inputs: list[tuple[torch.Tensor, int]] = []
    group = SimpleNamespace(
        rank_in_group=1,
        world_size=2,
        all_gather=lambda t, dim=-1: (
            gathered_inputs.append((t, dim)) or torch.cat((t, t + 1.0), dim=dim)
        ),
    )
    proj = IntraChipColumnParallelLinear.__new__(IntraChipColumnParallelLinear)
    torch.nn.Module.__init__(proj)
    proj.shard_group = group
    proj.input_size = 8
    proj.output_size = 4
    proj.output_size_per_partition = 2
    proj.tp_size = 2
    proj.bias = None
    proj.skip_bias_add = False
    proj.quant_method = SimpleNamespace(apply=lambda _layer, x, _bias: x[:, :2])

    out, out_bias = proj(torch.ones(3, 8))
    assert proj.shard_group is group
    assert out_bias is None
    assert len(gathered_inputs) == 1 and gathered_inputs[0][1] == -1
    assert out.shape == (3, 4)
    assert "shard_group_size=2" in proj.extra_repr()

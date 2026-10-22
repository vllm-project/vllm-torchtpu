# SPDX-License-Identifier: Apache-2.0
"""Checkpoint destinations and route IDs must share a bijective placement."""

import pytest

from vllm_torchtpu.models.vllm.deepseek_v41.expert_placement import (
    parse_expert_permutations,
    permutation_for_module,
    remap_checkpoint_expert,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "value",
    [
        [],
        {8: [1, 0]},
        {"08": [1, 0]},
        {"-1": [1, 0]},
        {"8": []},
        {"8": [0, 0]},
        {"8": [1, 2]},
        {"8": [True, 0]},
        {"8": [0.0, 1]},
    ],
)
def test_refuses_non_bijective_or_ambiguous_mapping(value):
    with pytest.raises((TypeError, ValueError)):
        parse_expert_permutations(value)


@pytest.mark.parametrize("prefix", ["", "model."])
@pytest.mark.parametrize("suffix", ["w1.weight", "w2.scale", "w3.weight_packed"])
def test_checkpoint_and_route_share_physical_expert(prefix, suffix):
    mappings = parse_expert_permutations({"8": [2, 0, 3, 1]})
    assert mappings is not None
    route = permutation_for_module("model.layers.8.ffn.experts", mappings)
    assert route is not None
    for logical, physical in enumerate(route):
        name = f"{prefix}layers.8.ffn.experts.{logical}.{suffix}"
        assert remap_checkpoint_expert(name, mappings) == (
            f"{prefix}layers.8.ffn.experts.{physical}.{suffix}"
        )
    for name in (
        "layers.8.ffn.gate.weight",
        "layers.8.ffn.shared_experts.w1.weight",
        "layers.40.ffn.experts.0.w1.weight",
        "layers.9.ffn.experts.0.w1.weight",
    ):
        assert remap_checkpoint_expert(name, mappings) == name
    assert permutation_for_module("model.layers.40.ffn.experts", mappings) is None


def test_identity_needs_no_extra_runtime_operand():
    assert permutation_for_module("model.layers.0.ffn.experts", {"0": (0, 1)}) is None
    assert parse_expert_permutations({}) is None
    with pytest.raises(ValueError, match="outside"):
        remap_checkpoint_expert("layers.0.ffn.experts.2.w1.weight", {"0": (1, 0)})

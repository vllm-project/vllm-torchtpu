# SPDX-License-Identifier: Apache-2.0
"""Checkpoint destinations and route IDs must share a bijective placement."""

import hashlib
import json

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


def test_default_table_retains_every_calibrated_checkpoint_destination():
    from vllm_torchtpu.models.vllm.deepseek_v41.tuned_expert_placement import (
        ExpertPlacementKey,
        default_expert_permutations,
    )

    key = ExpertPlacementKey("TPU v7", 40, 384, 5120, 2304, 6, 1, 8, 1, 1)
    mapping = default_expert_permutations(key)
    serialized = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(serialized.encode()).hexdigest() == (
        "3a2b821714a4b26187c4d14f033b65aef9ea86cb4402d08fc2844f537370f937"
    )
    for layer, permutation in mapping.items():
        assert sorted(permutation) == list(range(384))
        for logical, physical in enumerate(permutation):
            assert (
                remap_checkpoint_expert(
                    f"model.layers.{layer}.ffn.experts.{logical}.w1.weight", mapping
                )
                == f"model.layers.{layer}.ffn.experts.{physical}.w1.weight"
            )
    mapping.clear()
    assert len(default_expert_permutations(key)) == 30


def test_default_table_refuses_other_topologies():
    from vllm_torchtpu.models.vllm.deepseek_v41.tuned_expert_placement import (
        ExpertPlacementKey,
        default_expert_permutations,
    )

    key = ExpertPlacementKey("TPU v7", 40, 384, 5120, 2304, 6, 1, 8, 1, 1)
    for changed in (
        key._replace(device="TPU v6e"),
        key._replace(layers=60),
        key._replace(experts=256),
        key._replace(hidden=4096),
        key._replace(intermediate=2048),
        key._replace(topk=8),
        key._replace(tp=2),
        key._replace(dp=4),
        key._replace(pcp=2),
        key._replace(pp=2),
    ):
        assert default_expert_permutations(changed) == {}

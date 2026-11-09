# SPDX-License-Identifier: Apache-2.0
"""Loaded bytes and rank/state identity, without device or model snapshots."""

import copy

import pytest
import torch

from vllm_torchtpu.models.vllm.qwen4_exp.pd_identity import (
    LoadedWeightIdentity,
    canonical_rank_identity,
    merged_identity,
)


def digest(weights):
    identity = LoadedWeightIdentity()
    assert all(a is b for (_, a), (_, b) in zip(identity.observe(weights), weights))
    return identity.finish()


@pytest.mark.parametrize(
    "dtype", [torch.float8_e4m3fn, torch.bfloat16, torch.float32, torch.int64]
)
def test_identity_covers_actual_bytes_without_mutation(dtype):
    value = torch.arange(24).reshape(4, 6).to(dtype)
    before = value.view(torch.uint8).clone()
    base = digest([("weight", value)])
    assert base == digest([("weight", value.clone())])
    value.view(torch.uint8).flatten()[0] ^= 1
    assert base != digest([("weight", value)])
    assert torch.equal(before.flatten()[1:], value.view(torch.uint8).flatten()[1:])


def test_identity_order_shape_name_and_duplicate():
    a, b = torch.ones(3, 4), torch.zeros(2)
    assert digest([("a", a), ("b", b)]) == digest([("b", b), ("a", a)])
    assert digest([("a", a)]) != digest([("b", a)])
    assert digest([("a", a)]) != digest([("a", a.reshape(2, 6))])
    with pytest.raises(ValueError, match="Duplicate"):
        digest([("a", a), ("a", a)])
    with pytest.raises(ValueError, match="contiguous CPU"):
        digest([("a", a.T)])
    with pytest.raises(ValueError, match="successfully loaded"):
        LoadedWeightIdentity().finish()


def ep_checkpoint(num_experts=8):
    dense = [
        ("model.embed_tokens.weight", torch.arange(12.0).reshape(3, 4)),
        ("model.layers.0.ple.weight", torch.ones(5, dtype=torch.bfloat16)),
    ]
    experts = [
        (f"model.layers.0.mlp.experts.{e}.w1.weight", torch.full((2, 3), float(e)))
        for e in range(num_experts)
    ]
    return dense, experts


def ep_rank_records(dense, experts, ep_size):
    """What each rank of an EP group records under --enable-ep-weight-filter:
    every dense tensor, plus only the experts it owns."""
    out = []
    for rank in range(ep_size):
        identity = LoadedWeightIdentity()
        owned = [w for i, w in enumerate(experts) if i % ep_size == rank]
        list(identity.observe(dense + owned))
        out.append(identity.records())
    return out


@pytest.mark.parametrize("ep_size", [1, 2, 4, 8])
def test_merged_identity_of_filtered_ranks_matches_an_unfiltered_load(ep_size):
    dense, experts = ep_checkpoint()
    full = LoadedWeightIdentity()
    list(full.observe(dense + experts))
    shards = ep_rank_records(dense, experts, ep_size)
    assert merged_identity(shards) == full.finish()
    assert merged_identity(shards, name_fragment=".ple.") == full.finish(
        name_fragment=".ple."
    )
    if ep_size > 1:
        # Each filtered rank on its own disagrees: the reason for merging.
        assert len({merged_identity([records]) for records in shards}) == ep_size


def test_merged_identity_survives_object_gather():
    # all_gather_object pickles the records; JSON-style lists for shapes
    # must still digest like the original tuples.
    dense, experts = ep_checkpoint()
    shards = ep_rank_records(dense, experts, 2)
    as_lists = [
        {name: [dtype, list(shape), sha] for name, (dtype, shape, sha) in r.items()}
        for r in shards
    ]
    assert merged_identity(as_lists) == merged_identity(shards)


def test_merged_identity_rejects_ranks_that_loaded_different_bytes():
    dense, experts = ep_checkpoint()
    shards = ep_rank_records(dense, experts, 2)
    name = "model.embed_tokens.weight"
    dtype, shape, _ = shards[1][name]
    shards[1][name] = (dtype, shape, "f" * 64)
    with pytest.raises(ValueError, match=f"different bytes for {name}"):
        merged_identity(shards)
    with pytest.raises(ValueError, match="successfully loaded"):
        merged_identity([{}, {}])
    # A tensor outside the fragment does not take part in that digest.
    assert merged_identity(shards, name_fragment=".ple.")


def ranks():
    return [
        dict(
            rank=i,
            tp_rank=i,
            tp_size=2,
            checkpoint=str(i) * 64,
            ple_checkpoint="a" * 64,
            config={"hidden_size": 16},
            kv_scales=[(0, 0.5, 0.25)],
            state_flags={"gdn_qk_pair": True},
        )
        for i in range(2)
    ]


@pytest.mark.parametrize(
    "fault", ["missing", "reorder", "digest", "config", "state", "ple", "tp"]
)
def test_rank_identity_rejects_incomplete_or_inconsistent_peers(fault):
    rows = ranks()
    if fault == "missing":
        rows.pop()
    elif fault == "reorder":
        rows.reverse()
    elif fault == "digest":
        rows[0]["checkpoint"] = None
    elif fault == "config":
        rows[0]["config"] = {}
    elif fault == "state":
        rows[0]["state_flags"] = {}
    elif fault == "ple":
        rows[0]["ple_checkpoint"] = "b" * 64
    else:
        rows[0]["tp_rank"] = 2
    with pytest.raises(ValueError):
        canonical_rank_identity(rows, world_size=2)


def test_rank_identity_keeps_filtered_expert_and_scale_differences_in_signature():
    rows = ranks()
    first = copy.deepcopy(canonical_rank_identity(rows, world_size=2))
    rows[1]["checkpoint"] = "b" * 64
    assert canonical_rank_identity(rows, world_size=2) != first
    rows = ranks()
    rows[1]["kv_scales"] = [(0, 1.0, 0.25)]
    assert canonical_rank_identity(rows, world_size=2) != first


@pytest.mark.parametrize("ssm_dtype", ["bfloat16", "float32"])
def test_rank_identity_requires_matching_ssm_dtype(ssm_dtype):
    rows = ranks()
    for row in rows:
        row["state_flags"]["ssm_dtype"] = ssm_dtype
    identity = canonical_rank_identity(rows, world_size=2)
    assert identity["ranks"][0]["state_flags"]["ssm_dtype"] == ssm_dtype
    rows[1]["state_flags"]["ssm_dtype"] = (
        "float32" if ssm_dtype == "bfloat16" else "bfloat16"
    )
    with pytest.raises(ValueError, match="state configuration"):
        canonical_rank_identity(rows, world_size=2)

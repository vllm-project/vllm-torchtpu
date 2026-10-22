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
"""Loaded bytes and rank/state identity, without device or model snapshots."""

import copy

import pytest
import torch

from vllm_torchtpu.models.vllm.qwen4_exp.pd_identity import (
    LoadedWeightIdentity,
    canonical_rank_identity,
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

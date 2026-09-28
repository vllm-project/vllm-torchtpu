# SPDX-License-Identifier: Apache-2.0
"""Tests for packing the MoE routing tensors into a single all-gather.

The packing casts int32 expert ids through a float dtype so they can ride
along with `topk_weights` in one collective. The whole risk of the change is
that cast being lossy, so these tests pin the exactness boundary rather than
just checking that the plumbing runs. An earlier attempt in this codebase to
smuggle indices through bf16 failed for exactly this reason.
"""

import pytest
import torch

from vllm_torchtpu.layers.adapter import pipelined_fused_moe as pfm


class _FakeGroup:
    """All-gather over a single rank: concatenation is the identity."""

    world_size = 1

    def all_gather(self, t, dim=0):
        return torch.cat([t, t], dim=dim)


@pytest.fixture
def enable_packing(monkeypatch):
    monkeypatch.setattr(pfm.envs, "TPU_MOE_PACK_ROUTING_ALLGATHER", True)


@pytest.mark.parametrize(
    "dtype,limit",
    [
        (torch.float32, 2**24),
        (torch.float16, 2048),
        (torch.bfloat16, 256),
    ],
)
def test_exact_int_limit_is_actually_exact(dtype, limit):
    """The table drives a correctness gate, so verify it against the hardware.

    Every integer strictly below the limit must survive a round trip, and the
    first failure must appear no later than limit + 1.
    """
    below = torch.arange(0, limit, dtype=torch.int32)
    assert torch.equal(below.to(dtype).to(torch.int32), below)

    # limit + 1 is the first odd integer past the mantissa; it must NOT survive,
    # which is what makes the limit meaningful rather than merely conservative.
    over = torch.tensor([limit + 1], dtype=torch.int32)
    assert not torch.equal(over.to(dtype).to(torch.int32), over)


def test_limit_table_matches_module(enable_packing):
    assert pfm._EXACT_INT_LIMIT[torch.float16] == 2048
    assert pfm._EXACT_INT_LIMIT[torch.bfloat16] == 256


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_packed_round_trip_matches_unpacked(enable_packing, dtype):
    """Packed gather must be bitwise identical to two separate gathers."""
    torch.manual_seed(0)
    tokens, topk, num_experts = 37, 6, 256
    weights = torch.rand(tokens, topk, dtype=torch.float32).to(dtype)
    ids = torch.randint(0, num_experts, (tokens, topk), dtype=torch.int32)
    group = _FakeGroup()

    want_w = group.all_gather(weights, dim=0)
    want_i = group.all_gather(ids, dim=0)
    got_w, got_i = pfm._all_gather_routing(group, weights, ids, num_experts)

    assert torch.equal(got_i, want_i), "expert ids corrupted by packing"
    assert torch.equal(got_w, want_w), "routing weights corrupted by packing"
    assert got_i.dtype == ids.dtype
    assert got_w.dtype == weights.dtype


def test_deepseek_v4_flash_bf16_boundary(enable_packing):
    """DSv4-Flash is 256 experts; on bf16 that is exactly at the limit.

    Max id is 255, so packing is permitted and must be lossless. This is the
    tightest real configuration and the one most likely to break silently.
    """
    ids = torch.arange(256, dtype=torch.int32).reshape(-1, 8)
    weights = torch.rand(ids.shape, dtype=torch.float32).to(torch.bfloat16)
    assert pfm._can_pack_routing(weights, ids, 256)

    got_w, got_i = pfm._all_gather_routing(_FakeGroup(), weights, ids, 256)
    assert torch.equal(got_i, torch.cat([ids, ids], dim=0))
    assert torch.equal(got_w, torch.cat([weights, weights], dim=0))


def test_refuses_when_experts_exceed_dtype(enable_packing):
    """257 experts on bf16 is unrepresentable, so packing must decline."""
    weights = torch.rand(4, 6, dtype=torch.float32).to(torch.bfloat16)
    ids = torch.randint(0, 257, (4, 6), dtype=torch.int32)
    assert not pfm._can_pack_routing(weights, ids, 257)


def test_refuses_when_expert_count_unknown(enable_packing):
    weights = torch.rand(4, 6, dtype=torch.float16)
    ids = torch.randint(0, 8, (4, 6), dtype=torch.int32)
    assert not pfm._can_pack_routing(weights, ids, None)


def test_disabled_by_default():
    """Flag is off unless explicitly set, so the default path is untouched."""
    weights = torch.rand(4, 6, dtype=torch.float16)
    ids = torch.randint(0, 8, (4, 6), dtype=torch.int32)
    assert not pfm._can_pack_routing(weights, ids, 8)


def test_falls_back_when_disabled_and_still_correct():
    weights = torch.rand(4, 6, dtype=torch.float16)
    ids = torch.randint(0, 8, (4, 6), dtype=torch.int32)
    group = _FakeGroup()
    got_w, got_i = pfm._all_gather_routing(group, weights, ids, 8)
    assert torch.equal(got_w, group.all_gather(weights, dim=0))
    assert torch.equal(got_i, group.all_gather(ids, dim=0))


def test_handles_none_inputs(enable_packing):
    got_w, got_i = pfm._all_gather_routing(_FakeGroup(), None, None, 256)
    assert got_w is None and got_i is None


def test_refuses_mismatched_shapes(enable_packing):
    weights = torch.rand(4, 6, dtype=torch.float16)
    ids = torch.randint(0, 8, (4, 4), dtype=torch.int32)
    assert not pfm._can_pack_routing(weights, ids, 8)

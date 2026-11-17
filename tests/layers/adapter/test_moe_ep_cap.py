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
"""The decode EP load cap keeps every rank at or under its cap."""

import math

import pytest
import torch

from vllm_torchtpu.layers.adapter.moe_ep_cap import cap_routes, parse_spec

E, EP, K = 128, 8, 4
L = E // EP


def _route(t, seed, skew=True):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(t, E, generator=g)
    if skew:  # tokens of a step share experts, as served routing does
        logits = logits + 2.0 * torch.randn(1, E, generator=g)
    bias = 0.1 * torch.randn(E, generator=g)
    scores = logits.sigmoid()
    ids = torch.topk(scores + bias, K, dim=-1).indices
    w = torch.gather(scores, 1, ids)
    w = w / w.sum(-1, keepdim=True) * 2.5
    return ids.to(torch.int32), w, logits, bias


def _per_rank(ids):
    hit = torch.zeros(E, dtype=torch.bool)
    v = ids[ids >= 0].long()
    hit[v] = True
    return hit.view(EP, L).sum(-1)


@pytest.mark.parametrize("t", [1, 8, 16, 24, 48, 128])
@pytest.mark.parametrize("slack", [0, 1, 2])
@pytest.mark.parametrize("mode", ["drop", "sub"])
def test_cap_bounds_and_invariants(t, slack, mode):
    for seed in range(5):
        ids, w, logits, bias = _route(t, seed)
        new_ids, new_w, stats = cap_routes(
            ids, w, num_experts=E, ep=EP, slack=slack, mode=mode, sub_topn=8,
            router_logits=logits, score_bias=bias,
        )
        distinct = int((torch.bincount(ids.flatten().long(), minlength=E) > 0).sum())
        cap = math.ceil(distinct / EP) + slack
        top1 = ids[torch.arange(t), w.argmax(-1)]
        prot = torch.zeros(E, dtype=torch.bool)
        prot[top1.long()] = True
        per_rank_after = _per_rank(new_ids)
        prot_rank = prot.view(EP, L).sum(-1)
        assert bool((per_rank_after <= torch.maximum(prot_rank, torch.tensor(cap))).all())
        # every token keeps its top-1 route with its expert
        assert bool((new_ids[torch.arange(t), w.argmax(-1)] == top1).all())
        # weight sum preserved per token
        torch.testing.assert_close(new_w.sum(-1), w.sum(-1), rtol=1e-5, atol=1e-5)
        # dropped routes are -1 with zero weight
        assert bool((new_w[new_ids < 0] == 0).all())
        # unchanged tokens are bitwise
        same = (new_ids == ids).all(-1)
        assert torch.equal(new_w[same], w[same])
        # no duplicated expert within a token
        for r in range(t):
            v = new_ids[r][new_ids[r] >= 0].tolist()
            assert len(v) == len(set(v))
        # never more experts per rank than before
        assert bool((per_rank_after <= _per_rank(ids)).all())
        assert stats[7] <= stats[6]


def test_large_slack_is_identity():
    ids, w, logits, bias = _route(16, 3)
    new_ids, new_w, stats = cap_routes(ids, w, num_experts=E, ep=EP, slack=L)
    assert torch.equal(new_ids, ids)
    assert torch.equal(new_w, w)
    assert float(stats[2]) == 0


def test_drop_removes_lightest_expert_on_busiest_rank():
    # Rank 0 holds experts 0..3; only expert 0 is a top-1. Other experts: one per
    # rank 1..7 -> D = 4 + 7 = 11, cap = 2 + 0: rank 0 keeps 0 and the heavier of 1..3.
    ids = torch.tensor(
        [[0, 16, 32, 48], [64, 1, 80, 96], [112, 2, 16, 32], [48, 3, 64, 80]],
        dtype=torch.int32,
    )
    w = torch.tensor(
        [[0.4, 0.3, 0.2, 0.1], [0.5, 0.2, 0.2, 0.1], [0.6, 0.15, 0.15, 0.1],
         [0.7, 0.1, 0.1, 0.1]]
    )
    new_ids, new_w, _ = cap_routes(ids, w, num_experts=E, ep=EP, slack=0)
    assert int(new_ids[0, 0]) == 0
    assert int(new_ids[1, 1]) == 1  # mass 0.2 kept
    assert int(new_ids[2, 1]) == -1 and int(new_ids[3, 1]) == -1
    torch.testing.assert_close(new_w[2].sum(), w[2].sum())
    torch.testing.assert_close(new_w[2, 0] / new_w[2, 2], w[2, 0] / w[2, 2])


def test_padded_rows_are_dropped():
    ids, w, _, _ = _route(8, 1)
    w[6:] = 0
    new_ids, new_w, stats = cap_routes(ids, w, num_experts=E, ep=EP, slack=L)
    assert bool((new_ids[6:] == -1).all())
    assert torch.equal(new_ids[:6], ids[:6])
    assert float(stats[1]) == 6 * K


def test_parse_spec():
    assert parse_spec("") is None
    s = parse_spec("1")
    assert (s.slack, s.mode, s.max_tokens) == (1, "drop", 128)
    s = parse_spec("0:sub6@64")
    assert (s.slack, s.mode, s.sub_topn, s.max_tokens) == (0, "sub", 6, 64)
    with pytest.raises(ValueError):
        parse_spec("1:foo")


def test_max_tokens_mask_keeps_large_steps():
    ids, w, logits, bias = _route(48, 2)
    capped, _, _ = cap_routes(ids, w, num_experts=E, ep=EP, slack=0)
    assert bool((capped < 0).any())
    kept, kw, _ = cap_routes(ids, w, num_experts=E, ep=EP, slack=0, max_tokens=32)
    assert torch.equal(kept, ids) and torch.equal(kw, w)

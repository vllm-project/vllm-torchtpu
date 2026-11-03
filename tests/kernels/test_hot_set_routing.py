# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the hot-set routing simulation (pure torch, no TPU)."""

import math

import pytest
import torch

from vllm_torchtpu.layers.adapter.hot_set_routing import (
    HotSetRouting,
    expected_load_cv,
    hot_set_ids,
)


def _loads(ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    return torch.bincount(ids.flatten().long(), minlength=num_experts).float()


@pytest.mark.parametrize("rate", [0.0, 0.3, 1.0])
def test_hot_tokens_pick_only_hot_experts(rate):
    torch.manual_seed(0)
    num_tokens, num_experts, top_k, hot = 4096, 512, 10, 16
    strategy = HotSetRouting(hot_size=hot, rate=rate)
    hidden = torch.zeros(num_tokens, 8, dtype=torch.bfloat16)
    logits = torch.randn(num_tokens, num_experts)
    weights, ids = strategy.route_tokens(hidden, logits, top_k)
    assert ids.shape == (num_tokens, top_k) and ids.dtype == torch.int32
    assert weights.shape == (num_tokens, top_k) and weights.dtype == hidden.dtype
    # no duplicate expert within a token
    assert all(len(set(row.tolist())) == top_k for row in ids)
    hot_share = (ids < hot).float().mean().item()
    expected = rate * 1.0 + (1 - rate) * hot / num_experts
    assert abs(hot_share - expected) < 0.02, (hot_share, expected)


def test_load_skew_matches_the_analytic_cv():
    torch.manual_seed(1)
    num_tokens, num_experts, top_k, hot, rate = 65536, 512, 10, 16, 0.3
    strategy = HotSetRouting(hot_size=hot, rate=rate)
    _, ids = strategy.route_tokens(
        torch.zeros(num_tokens, 8), torch.randn(num_tokens, num_experts), top_k
    )
    loads = _loads(ids, num_experts)
    cv = (loads.std(unbiased=False) / loads.mean()).item()
    assert abs(cv - expected_load_cv(num_experts, hot, rate)) < 0.05, cv
    assert abs(expected_load_cv(num_experts, hot, rate) - 1.67) < 0.02


def test_spread_puts_one_hot_expert_per_rank():
    ids = hot_set_ids(512, 32, spread=True)
    assert ids.tolist() == list(range(0, 512, 16))
    assert hot_set_ids(512, 16, spread=False).tolist() == list(range(16))


def test_rate_zero_is_uniform():
    torch.manual_seed(2)
    num_tokens, num_experts, top_k = 32768, 512, 10
    _, ids = HotSetRouting(hot_size=16, rate=0.0).route_tokens(
        torch.zeros(num_tokens, 8), torch.randn(num_tokens, num_experts), top_k
    )
    loads = _loads(ids, num_experts)
    assert (loads.std(unbiased=False) / loads.mean()).item() < 0.1
    assert math.isclose(expected_load_cv(num_experts, 16, 0.0), 0.0)

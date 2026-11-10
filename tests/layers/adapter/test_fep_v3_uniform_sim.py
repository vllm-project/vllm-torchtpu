# SPDX-License-Identifier: Apache-2.0
"""fused-EP v3 under VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random.

v3 must arm and route on uniform noise (as v2 does), so v2 and v3 can be
compared under the same fair routing; any other strategy is still refused.
"""

import torch

from vllm_torchtpu.layers.adapter import fused_moe_ep_v3 as v3
from vllm_torchtpu.layers.adapter import moe_routing


def _simulate(monkeypatch, name):
    monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", object())
    monkeypatch.setattr(moe_routing, "_strategy_name", name)


def test_v3_arms_under_uniform_random(monkeypatch):
    _simulate(monkeypatch, "uniform_random")
    assert v3._unsupported_simulation_reason() is None


def test_v3_refuses_other_strategies(monkeypatch):
    _simulate(monkeypatch, "normal_routing")
    assert "uniform_random" in v3._unsupported_simulation_reason()


def test_v3_without_simulation_is_unchanged(monkeypatch):
    monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", None)
    assert v3._unsupported_simulation_reason() is None
    logits = torch.randn(16, 512)
    assert v3._simulated_router_logits(logits) is logits


def _record_op(monkeypatch):
    seen = {}

    def op(*args):
        seen["gating"] = args[5]
        return args[0]

    monkeypatch.setattr(v3, "_OP", op)
    monkeypatch.setattr(v3, "_SKIP_PADDED_TOKENS", False)
    monkeypatch.setattr(v3, "_squeeze_channel_scale_checked", lambda s: s)
    monkeypatch.setattr(v3, "_to_mesh_expert_order", lambda g: g)
    return seen


def test_v3_routes_on_uniform_noise_under_uniform_random(monkeypatch):
    seen = _record_op(monkeypatch)
    _simulate(monkeypatch, "uniform_random")
    logits = torch.full((64, 512), 3.0, dtype=torch.bfloat16)  # a constant router: one expert set
    x = torch.zeros(64, 8)
    v3.fused_moe_ep(x, None, None, None, None, logits)
    g = seen["gating"]
    assert g.dtype == torch.float32 and g.shape == logits.shape
    assert bool(((g >= 3.0) & (g < 4.0)).all())  # noise in [m, m + 1)
    # A uniform draw: the per-token top-10 sets are not all the same.
    tops = torch.topk(g, 10, dim=-1).indices.sort(dim=-1).values
    assert len({tuple(t.tolist()) for t in tops}) > 32


def test_v3_routes_on_real_logits_without_simulation(monkeypatch):
    seen = _record_op(monkeypatch)
    monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", None)
    logits = torch.randn(64, 512)
    v3.fused_moe_ep(torch.zeros(64, 8), None, None, None, None, logits)
    assert seen["gating"] is logits

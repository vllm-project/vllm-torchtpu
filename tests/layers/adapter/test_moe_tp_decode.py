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
"""Arming of moe_tp_decode on real MoE runners with tiny TP16 slices.

The op and the start-up self-test need the 16-device mesh, so a stub op stands
in for the first and the second is stubbed; the topology and the TP-group
gather are stand-ins too. Everything else is the real runner forward: the
gate, SharedExperts, VllmFp8MoEMethodTPU.apply_monolithic and the runner's
scale, add and all-reduce decisions.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.fused_moe.runner import moe_runner as moe_runner_mod

import vllm_torchtpu.envs as envs
from tests.layers.adapter.moe_tp_decode_helpers import (
    FP8,
    ROUTED_SCALE,
    H,
    install_moe_tp_decode_harness,
)
from vllm_torchtpu.layers.adapter import moe_tp_decode, token_padding
from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner


@pytest.fixture
def harness(monkeypatch):
    return install_moe_tp_decode_harness(monkeypatch)


def _x():
    return torch.randn(16, H).to(torch.bfloat16)


def _armed(runner):
    return moe_tp_decode.moe_tp_decode_supported(runner._quant_method)


def test_unarmed_layer_keeps_the_production_forward(harness):
    runner = harness.runner("model.layers.0.mlp.experts")
    shared = runner.shared_experts._layer
    x = _x()

    out = runner(hidden_states=x, router_logits=x)

    assert not _armed(runner) and not runner._fused_output_is_reduced
    assert harness.all_reduces == 1 and shared.calls == 1 and not harness.op_calls
    want = x * 0.5 + (x * 0.25) * ROUTED_SCALE
    assert torch.equal(out, want)


def test_arming_moves_shared_expert_scale_and_all_reduce_into_the_op(harness):
    model = harness.model({}, {}, {"scoring_func": "softmax"})
    draft = harness.model({}, prefix="draft.layers")
    target = list(model.layers)
    sibling = target[2]
    shared0 = target[0].shared_experts._layer
    x = _x()
    sibling_before = sibling(hidden_states=x, router_logits=x)

    assert moe_tp_decode.arm(model, draft) == 3

    assert harness.selftest_calls == 1
    for runner in (target[0], target[1], draft.layers[0]):
        layer = runner.routed_experts
        assert _armed(runner) and runner._fused_output_is_reduced
        assert runner.shared_experts is None
        assert runner._forward_entry is moe_runner_mod._moe_forward
        assert runner.routed_scaling_factor == 1.0
        for name in moe_tp_decode._BUFFERS:
            assert name in layer._non_persistent_buffers_set
        assert (
            layer._moe_tp_decode_bias.data_ptr()
            == layer.e_score_correction_bias.data_ptr()
        )
        assert layer._moe_tp_decode_rank.tolist() == [[0]]
    mlp = target[0].routed_experts
    assert (
        mlp._moe_tp_decode_sw13.dtype == FP8 and not mlp._moe_tp_decode_sw13._is_view()
    )

    out = target[0](hidden_states=x, router_logits=x)
    # One op call; no shared-expert call, routed scale, add or all-reduce (the
    # one all-reduce so far is the sibling's, before arming).
    assert len(harness.op_calls) == 1 and shared0.calls == 0
    assert harness.all_reduces == 1
    assert torch.equal(out, (x.float() * 3).to(x.dtype))
    _, scores, bias, pad_mask, w13, *_, rank = harness.op_calls[0]
    logits = target[0].gate(x)[0]
    assert torch.equal(scores, logits.sigmoid())
    assert torch.equal(bias, target[0].routed_experts.e_score_correction_bias)
    assert pad_mask is token_padding._padding_state.local_padding_mask
    assert 16 - int(pad_mask[:16].sum()) == 11
    assert w13 is target[0].routed_experts.w13_weight
    assert rank.tolist() == [[0]]

    assert not _armed(sibling) and sibling.routed_scaling_factor == ROUTED_SCALE
    sibling_shared = sibling.shared_experts._layer
    calls = sibling_shared.calls
    assert torch.equal(sibling(hidden_states=x, router_logits=x), sibling_before)
    assert sibling_shared.calls == calls + 1 and harness.all_reduces == 2
    assert "armed 2/3 target and 1/1 drafter" in harness.logs[-1]
    assert "kernel token counts [16, 32, 64]" in harness.logs[-1]
    assert "routing is not sigmoid" in harness.logs[-1]


def test_armed_runner_with_skip_final_all_reduce_is_not_reduced(harness):
    model = harness.model({})
    assert moe_tp_decode.arm(model) == 1
    runner = model.layers[0]
    runner.moe_config.skip_final_all_reduce = True
    assert not runner._fused_output_is_reduced


@pytest.mark.parametrize(
    ("armed", "skip_final_all_reduce", "expected"),
    [(False, False, False), (True, False, True), (True, True, False)],
)
def test_fused_output_is_reduced_under_torch_compile(
    armed, skip_final_all_reduce, expected
):
    quant_method = SimpleNamespace(moe_kernel=None)
    if armed:
        setattr(quant_method, moe_tp_decode._OP_ATTR, "op")
    runner = VllmMoERunner.__new__(VllmMoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_experts = SimpleNamespace(quant_method=quant_method)
    runner.moe_config = SimpleNamespace(skip_final_all_reduce=skip_final_all_reduce)

    def reduce_unless_reduced(x):
        return x if runner._fused_output_is_reduced else x * 2

    x = torch.ones(2)
    torch._dynamo.reset()
    compiled = torch.compile(reduce_unless_reduced, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), x if expected else x * 2)


@pytest.mark.parametrize(
    ("spec", "reason"),
    [
        ({"use_ep": True}, "needs TP MoE over 16 ranks"),
        ({"skip_final_all_reduce": True}, "skip_final_all_reduce is set"),
        ({"shared": False}, "no shared expert"),
        ({"scoring_func": "softmax"}, "routing is not sigmoid"),
    ],
)
def test_refused_layer_keeps_its_path(harness, spec, reason):
    model = harness.model(spec)
    assert moe_tp_decode.arm(model) == 0
    runner = model.layers[0]
    assert not _armed(runner) and runner.routed_scaling_factor == ROUTED_SCALE
    assert harness.selftest_calls == 0
    assert "armed 0/1 target" in harness.logs[-1] and reason in harness.logs[-1]


def test_refuses_without_the_padding_mask(harness, monkeypatch):
    monkeypatch.setattr(envs, "TPU_MOE_SKIP_PADDED_TOKENS", False)
    model = harness.model({})
    assert moe_tp_decode.arm(model) == 0
    assert "TPU_MOE_SKIP_PADDED_TOKENS" in harness.logs[-1]


def test_refuses_when_the_moe_path_owns_its_collectives(harness, monkeypatch):
    monkeypatch.setattr(envs, "TPU_MOE_FP8_ALL_GATHER", True)
    model = harness.model({})
    assert moe_tp_decode.arm(model) == 0
    assert "owns its collectives" in harness.logs[-1]


def test_stands_down_when_fused_moe_decode_is_on(harness, monkeypatch):
    monkeypatch.setattr(envs, "FUSED_MOE_DECODE_KERNEL", True)
    model = harness.model({})
    assert moe_tp_decode.arm(model) == 0
    assert not _armed(model.layers[0]) and harness.selftest_calls == 0
    assert "FUSED_MOE_DECODE_KERNEL is set" in harness.logs[-1]


def test_refuses_a_topology_it_cannot_serve(harness, monkeypatch):
    monkeypatch.setattr(moe_tp_decode, "_topology", lambda: "no EP group")
    model = harness.model({})
    assert moe_tp_decode.arm(model) == 0
    assert "no EP group" in harness.logs[-1] and "mesh index -1" in harness.logs[-1]


@pytest.mark.parametrize(
    ("other", "raises"),
    [
        ({"mesh_index": 1}, False),  # mesh index need not equal the TP rank
        ({"mesh_index": 1, "sig": "0" * 16}, True),
        ({"mesh_index": 0}, True),
    ],
)
def test_ranks_must_agree(harness, monkeypatch, other, raises):
    monkeypatch.setattr(moe_tp_decode, "_gather", lambda obj: [obj, {**obj, **other}])
    model = harness.model({})
    if raises:
        with pytest.raises(RuntimeError, match="TP ranks disagree"):
            moe_tp_decode.arm(model)
        assert not _armed(model.layers[0])
    else:
        assert moe_tp_decode.arm(model) == 1


def test_ranks_that_all_refused_agree(harness, monkeypatch):
    monkeypatch.setattr(moe_tp_decode, "_topology", lambda: "no EP group")
    monkeypatch.setattr(moe_tp_decode, "_gather", lambda obj: [obj, dict(obj)])
    assert moe_tp_decode.arm(harness.model({})) == 0


def test_failed_selftest_raises_and_arms_nothing(harness):
    harness.selftest_failures = [(16, H, 5)]
    model = harness.model({})
    with pytest.raises(RuntimeError, match="self-test failed"):
        moe_tp_decode.arm(model)
    runner = model.layers[0]
    assert not _armed(runner) and runner.shared_experts is not None

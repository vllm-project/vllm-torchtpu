# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Arming of moe_tp_decode on real MoE runners with tiny TP16 slices.

The op and the start-up self-test need the 16-device mesh, so a stub op stands
in for the first and the second is stubbed; the topology and the TP-group
gather are stand-ins too. Everything else is the real runner forward: the
gate, SharedExperts, VllmFp8MoEMethodTPU.apply_monolithic and the runner's
scale, add and all-reduce decisions.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import vllm.distributed.parallel_state as parallel_state
from jax.experimental.pallas import tpu as pltpu
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner import moe_runner as moe_runner_mod
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import moe_routing, moe_tp_decode, token_padding
from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.adapter.quantization import fp8 as fp8_mod

# One rank's slice: the smallest hidden size the kernel admits at 16 ranks.
H, E, INTER, FS, TOPK = 2048, 16, 128, 128, 8
ROUTED_SCALE = 2.5
FP8 = torch.float8_e4m3fn


class _Fp8Linear(torch.nn.Module):
    """A K-major per-channel FP8 linear as VllmFp8LinearMethodTPU leaves it."""

    def __init__(self, k, n, *, reduce_results=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            (torch.randn(k, n) / math.sqrt(k)).to(FP8), requires_grad=False
        )
        self.weight_scale = torch.nn.Parameter(torch.rand(n) + 0.5, requires_grad=False)
        self.bias = None
        if reduce_results is not None:
            self.reduce_results = reduce_results
        setattr(self, WEIGHT_FLIPPED_ATTR, True)
        self.quant_method = fp8_mod.VllmFp8LinearMethodTPU.__new__(
            fp8_mod.VllmFp8LinearMethodTPU
        )


class _SharedMLP(torch.nn.Module):
    """DeepseekV2MLP's structure (reduce_results=False); forward counts calls."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = _Fp8Linear(H, 2 * FS)
        self.down_proj = _Fp8Linear(FS, H, reduce_results=False)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x * 0.5


class _Gate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            (torch.randn(E, H) / math.sqrt(H)).to(torch.bfloat16), requires_grad=False
        )

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight).float(), None


class _RoutedExperts(torch.nn.Module):
    forward_monolithic = RoutedExperts.forward_monolithic

    def __init__(self, moe_config, quant_method, scoring_func):
        super().__init__()
        self.moe_config = moe_config
        self.quant_method = quant_method
        self.global_num_experts = E
        self.scoring_func = scoring_func
        self.renormalize = True
        self.use_grouped_topk = True
        self.num_expert_group = 1
        self.topk_group = 1
        self.custom_routing_function = None
        self.routed_scaling_factor = 1.0
        self.expert_placement_strategy = "linear"

        def param(t):
            return torch.nn.Parameter(t, requires_grad=False)

        self.w13_weight = param(torch.randn(E, H, 2 * INTER).to(FP8))
        self.w2_weight = param(torch.randn(E, INTER, H).to(FP8))
        self.w13_weight_scale_inv = param(torch.rand(E, 1, 1, 2 * INTER))
        self.w2_weight_scale_inv = param(torch.rand(E, 1, 1, H))
        self.e_score_correction_bias = param(torch.rand(E) * 0.1)
        moe_routing.register_experts_start_buffer(self, device=torch.device("cpu"))

    def _ensure_moe_quant_config_init(self):
        pass


def _moe_config(*, use_ep=False, skip_final_all_reduce=False):
    par = SimpleNamespace(
        use_ep=use_ep,
        ep_size=16 if use_ep else 1,
        ep_rank=0,
        tp_size=1 if use_ep else 16,
        dp_size=1,
        pcp_size=1,
        enable_eplb=False,
        all2all_backend="allgather_reducescatter",
        use_fi_nvl_two_sided_kernels=False,
        use_all2all_kernels=False,
    )
    return SimpleNamespace(
        moe_parallel_config=par,
        experts_per_token=TOPK,
        is_sequence_parallel=False,
        skip_final_all_reduce=skip_final_all_reduce,
        tp_size=par.tp_size,
        ep_size=par.ep_size,
        ep_rank=0,
        dp_size=1,
        pcp_size=1,
        sp_size=1,
        hidden_dim=H,
        hidden_dim_unpadded=H,
    )


class _Harness:
    def __init__(self):
        self.layers = {}
        self.all_reduces = 0
        self.op_calls = []
        self.selftest_calls = 0
        self.selftest_failures = []
        self.logs = []

    def runner(self, name, *, scoring_func="sigmoid", shared=True, **config):
        mcfg = _moe_config(**config)
        qm = fp8_mod.VllmFp8MoEMethodTPU.__new__(fp8_mod.VllmFp8MoEMethodTPU)
        FusedMoEMethodBase.__init__(qm, mcfg)
        qm._tpu_activation_str = "silu"
        runner = MoERunner(
            layer_name=name,
            moe_config=mcfg,
            router=SimpleNamespace(),
            routed_experts=_RoutedExperts(mcfg, qm, scoring_func),
            gate=_Gate(),
            shared_experts=_SharedMLP() if shared else None,
            routed_scaling_factor=ROUTED_SCALE,
        )
        self.layers[name] = runner
        return runner

    def model(self, *specs, prefix="model.layers"):
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList(
            self.runner(f"{prefix}.{i}.mlp.experts", **spec)
            for i, spec in enumerate(specs)
        )
        return model

    def op(self, x, *operands):
        self.op_calls.append((x, *operands))
        return (x.float() * 3).to(x.dtype)


class _Log:
    def __init__(self, sink):
        self.info = self.warning = lambda fmt, *args: sink.append(fmt % args)


@pytest.fixture
def harness(monkeypatch):
    torch.manual_seed(0)
    h = _Harness()
    # Set the attribute, not the environment: a test that monkeypatched this
    # attribute earlier in the session leaves it set on the module, where it
    # shadows envs.__getattr__ and with it any environment value.
    monkeypatch.setattr(envs, "TPU_MOE_SKIP_PADDED_TOKENS", True)
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    monkeypatch.setattr(moe_runner_mod, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(
        moe_runner_mod, "register_layer_for_moe_forward_op", lambda config, layer: None
    )
    monkeypatch.setattr(moe_runner_mod, "get_layer_from_name", h.layers.__getitem__)
    monkeypatch.setattr(
        moe_runner_mod,
        "get_forward_context",
        lambda: SimpleNamespace(dp_metadata=None, all_moe_layers=None),
    )
    monkeypatch.setattr(moe_runner_mod, "is_forward_context_available", lambda: False)

    def all_reduce(t):
        h.all_reduces += 1
        return t

    monkeypatch.setattr(moe_runner_mod, "tensor_model_parallel_all_reduce", all_reduce)
    monkeypatch.setattr(
        parallel_state, "get_dp_group", lambda: SimpleNamespace(world_size=1)
    )
    # The unarmed path's grouped matmul (a TPU kernel): a stand-in.
    monkeypatch.setattr(
        fp8_mod, "fused_moe_gmm", lambda hidden_states, **kw: hidden_states * 0.25
    )
    padding = token_padding.TokenPaddingState.create(64, torch.device("cpu"))
    padding.update(11, 16)
    monkeypatch.setattr(token_padding, "_padding_state", padding)

    monkeypatch.setattr(
        moe_tp_decode,
        "_topology",
        lambda: moe_tp_decode._Topology(mesh=SimpleNamespace(size=16), mesh_index=0),
    )
    monkeypatch.setattr(moe_tp_decode, "_gather", lambda obj: [obj])
    monkeypatch.setattr(moe_tp_decode, "_get_op", lambda mesh, statics: h.op)

    def selftest(topo, buckets, device):
        h.selftest_calls += 1
        return h.selftest_failures

    monkeypatch.setattr(moe_tp_decode, "_selftest", selftest)
    monkeypatch.setattr(moe_tp_decode, "_RANK_BUFFER", None)
    monkeypatch.setattr(moe_tp_decode, "_SELFTEST_DONE", False)
    monkeypatch.setattr(moe_tp_decode, "logger", _Log(h.logs))
    return h


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

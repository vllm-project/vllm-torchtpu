# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Arming of fused_moe_ep_v3_tp on real MoE runners (EP16 experts, a TP16
shared-expert slice per rank), as tensor-parallel attention over the EP group
serves GLM-5.2.

The op needs the 16-device mesh, so a stub op stands in for it; the topology
and the TP-group gather are stand-ins too. Everything else is the real runner
forward: the gate, SharedExperts, VllmFp8MoEMethodTPU.apply_monolithic and the
runner's scale, add and all-reduce decisions. The op's own arithmetic is
covered by test_fused_moe_ep_v3_tp_op.py.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import vllm.config
import vllm.distributed.parallel_state as parallel_state
from jax.experimental.pallas import tpu as pltpu
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner import moe_runner as moe_runner_mod
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from vllm_torchtpu.layers.adapter import (
    fused_moe_ep_v3,
    fused_moe_ep_v3_tp,
    moe_routing,
    token_padding,
)
from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.adapter.quantization import fp8 as fp8_mod

pytestmark = pytest.mark.cpu_test

# One rank: 16 local experts of a 256-expert layer and its TP16 shared slice.
H, E, INTER, FS, TOPK, RANKS = 2048, 16, 128, 128, 8, 16
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
            (torch.randn(E * RANKS, H) / math.sqrt(H)).to(torch.bfloat16),
            requires_grad=False,
        )

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight).float(), None


class _RoutedExperts(torch.nn.Module):
    forward_monolithic = RoutedExperts.forward_monolithic

    def __init__(self, moe_config, quant_method, scoring_func):
        super().__init__()
        self.moe_config = moe_config
        self.quant_method = quant_method
        self.global_num_experts = E * RANKS
        self.scoring_func = scoring_func
        self.renormalize = True
        self.use_grouped_topk = True
        self.num_expert_group = 1
        self.topk_group = 1
        self.custom_routing_function = None
        self.apply_router_weight_on_input = False
        self.routed_scaling_factor = 1.0
        self.expert_placement_strategy = "linear"

        def param(t):
            return torch.nn.Parameter(t, requires_grad=False)

        self.w13_weight = param(torch.randn(E, H, 2 * INTER).to(FP8))
        self.w2_weight = param(torch.randn(E, INTER, H).to(FP8))
        self.w13_weight_scale_inv = param(torch.rand(E, 1, 1, 2 * INTER))
        self.w2_weight_scale_inv = param(torch.rand(E, 1, 1, H))
        self.e_score_correction_bias = param(torch.rand(E * RANKS) * 0.1)
        moe_routing.register_experts_start_buffer(self, device=torch.device("cpu"))

    def _ensure_moe_quant_config_init(self):
        pass


def _moe_config(*, use_ep=True, skip_final_all_reduce=False):
    par = SimpleNamespace(
        use_ep=use_ep,
        ep_size=RANKS if use_ep else 1,
        ep_rank=0,
        tp_size=1 if use_ep else RANKS,
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
        self.logs = []

    def runner(self, name, *, scoring_func="sigmoid", shared=True, **config):
        mcfg = _moe_config(**config)
        qm = fp8_mod.VllmFp8MoEMethodTPU.__new__(fp8_mod.VllmFp8MoEMethodTPU)
        FusedMoEMethodBase.__init__(qm, mcfg)
        qm._tpu_activation_str = "silu"
        qm._use_adaptive_fused_moe = False
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
        self.info_once = self.info


def _topology():
    return fused_moe_ep_v3_tp._Topology(
        mesh=SimpleNamespace(size=RANKS),
        mesh_index=0,
        rank_order=tuple(range(RANKS)),
        groups=(tuple(range(RANKS)),),
    )


@pytest.fixture
def harness(monkeypatch):
    torch.manual_seed(0)
    h = _Harness()
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "1")
    monkeypatch.setenv("MOE_FUSED_EP_V3_MIN_TOKENS", "512")
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_IMPL", "v3")
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
    monkeypatch.setattr(fused_moe_ep_v3_tp, "_topology", _topology)
    monkeypatch.setattr(fused_moe_ep_v3_tp, "_gather", lambda obj: [obj])
    monkeypatch.setattr(fused_moe_ep_v3_tp, "_get_op", lambda topo, statics: h.op)
    monkeypatch.setattr(fused_moe_ep_v3_tp, "_RANK_BUFFER", None)
    monkeypatch.setattr(fused_moe_ep_v3_tp, "logger", _Log(h.logs))
    return h


def _x():
    return torch.randn(16, H).to(torch.bfloat16)


def _armed(runner):
    return fused_moe_ep_v3_tp.fused_moe_ep_v3_tp_supported(runner._quant_method)


def test_unarmed_layer_keeps_the_production_forward(harness):
    runner = harness.runner("model.layers.0.mlp.experts")
    shared = runner.shared_experts._layer
    x = _x()

    out = runner(hidden_states=x, router_logits=x)

    assert not _armed(runner) and not runner._fused_output_is_reduced
    assert harness.all_reduces == 1 and shared.calls == 1 and not harness.op_calls
    assert torch.equal(out, x * 0.5 + (x * 0.25) * ROUTED_SCALE)


def test_arming_moves_shared_expert_scale_and_all_reduce_into_the_op(harness):
    model = harness.model({}, {}, {"scoring_func": "softmax"})
    draft = harness.model({}, prefix="draft.layers")
    target = list(model.layers)
    sibling = target[2]
    shared0 = target[0].shared_experts._layer
    x = torch.randn(16, H).to(torch.bfloat16)
    logits = torch.randn(16, E * RANKS)
    sibling_before = sibling(hidden_states=x, router_logits=logits)

    assert fused_moe_ep_v3_tp.arm(model, draft) == 3

    for runner in (target[0], target[1], draft.layers[0]):
        layer = runner.routed_experts
        assert _armed(runner) and runner._fused_output_is_reduced
        assert runner.shared_experts is None
        assert runner._forward_entry is moe_runner_mod._moe_forward
        assert runner.routed_scaling_factor == 1.0
        for name in fused_moe_ep_v3_tp._BUFFERS:
            assert name in layer._non_persistent_buffers_set
        assert (
            layer._fused_moe_ep_v3_tp_bias.data_ptr()
            == layer.e_score_correction_bias.data_ptr()
        )
        assert layer._fused_moe_ep_v3_tp_rank.tolist() == [[0]]

    out = target[0](hidden_states=x, router_logits=logits)
    # One op call; no shared-expert call, routed scale, add or all-reduce (the
    # one all-reduce so far is the sibling's, before arming).
    assert len(harness.op_calls) == 1 and shared0.calls == 0
    assert harness.all_reduces == 1
    assert torch.equal(out, (x.float() * 3).to(x.dtype))
    _, scores, bias, pad_mask, w13, w2, s13, s2, sw13, *_, start, rank = (
        harness.op_calls[0]
    )
    layer0 = target[0].routed_experts
    # The runner applies its own gate; the op gets production's sigmoid scores.
    assert torch.equal(scores, target[0].gate(x)[0].sigmoid())
    assert torch.equal(bias, layer0.e_score_correction_bias)
    assert pad_mask is token_padding._padding_state.local_padding_mask
    assert 16 - int(pad_mask[:16].sum()) == 11
    assert w13 is layer0.w13_weight and s13 is layer0.w13_weight_scale_inv
    assert sw13.dtype == FP8 and sw13.shape == (H, 2 * FS)
    assert start.tolist() == [[0]] and rank.tolist() == [[0]]

    assert not _armed(sibling) and sibling.routed_scaling_factor == ROUTED_SCALE
    assert torch.equal(sibling(hidden_states=x, router_logits=logits), sibling_before)
    assert harness.all_reduces == 2
    assert "armed 2/3 target and 1/1 drafter" in harness.logs[-1]
    assert "v3 at >= 512 tokens" in harness.logs[-1]
    assert "routing is not sigmoid" in harness.logs[-1]


def test_armed_runner_with_skip_final_all_reduce_is_not_reduced(harness):
    model = harness.model({})
    assert fused_moe_ep_v3_tp.arm(model) == 1
    runner = model.layers[0]
    runner.moe_config.skip_final_all_reduce = True
    assert not runner._fused_output_is_reduced


@pytest.mark.parametrize(
    ("armed", "skip_final_all_reduce", "expected"),
    [(False, False, False), (True, False, True), (True, True, False)],
)
def test_fused_output_is_reduced_under_torch_compile(
    monkeypatch, armed, skip_final_all_reduce, expected
):
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_IMPL", "v3")
    quant_method = SimpleNamespace(moe_kernel=None)
    if armed:
        setattr(quant_method, fused_moe_ep_v3_tp._OP_ATTR, "op")
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
        ({"use_ep": False}, "needs EP MoE over 16 ranks"),
        ({"skip_final_all_reduce": True}, "skip_final_all_reduce is set"),
        ({"shared": False}, "no shared expert"),
        ({"scoring_func": "softmax"}, "routing is not sigmoid"),
    ],
)
def test_refused_layer_keeps_its_path(harness, spec, reason):
    model = harness.model(spec)
    assert fused_moe_ep_v3_tp.arm(model) == 0
    runner = model.layers[0]
    assert not _armed(runner) and runner.routed_scaling_factor == ROUTED_SCALE
    assert "armed 0/1 target" in harness.logs[-1] and reason in harness.logs[-1]


def test_refuses_a_misplaced_expert_block(harness):
    model = harness.model({})
    # This rank is EP rank 1 while the mesh puts EP rank 0's block here.
    model.layers[0].routed_experts.moe_config.ep_rank = 1
    assert fused_moe_ep_v3_tp.arm(model) == 0
    assert "EP rank 0's first expert is 16, not 0 x 16" in harness.logs[-1]


def test_refuses_without_the_padding_mask(harness, monkeypatch):
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "0")
    assert fused_moe_ep_v3_tp.arm(harness.model({})) == 0
    assert "TPU_MOE_SKIP_PADDED_TOKENS" in harness.logs[-1]


def test_refuses_when_the_moe_path_owns_its_collectives(harness, monkeypatch):
    monkeypatch.setenv("TPU_MOE_FP8_ALL_GATHER", "1")
    assert fused_moe_ep_v3_tp.arm(harness.model({})) == 0
    assert "owns its collectives" in harness.logs[-1]


def test_refuses_a_topology_it_cannot_serve(harness, monkeypatch):
    monkeypatch.setattr(fused_moe_ep_v3_tp, "_topology", lambda: "no EP group")
    assert fused_moe_ep_v3_tp.arm(harness.model({})) == 0
    assert "no EP group" in harness.logs[-1] and "mesh index None" in harness.logs[-1]


@pytest.mark.parametrize(("other", "raises"), [({}, False), ({"sig": "0" * 16}, True)])
def test_ranks_must_agree(harness, monkeypatch, other, raises):
    monkeypatch.setattr(
        fused_moe_ep_v3_tp, "_gather", lambda obj: [obj, {**obj, **other}]
    )
    model = harness.model({})
    if raises:
        with pytest.raises(RuntimeError, match="TP ranks arm different layers"):
            fused_moe_ep_v3_tp.arm(model)
        assert not _armed(model.layers[0])
    else:
        assert fused_moe_ep_v3_tp.arm(model) == 1


def test_glm52_ep16_shape_streams_its_experts(monkeypatch):
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    # GLM-5.2 at EP16: 16 local experts of 6144 x 2048 do not fit whole.
    config, plan = fused_moe_ep_v3_tp._v3_config(6144, 2048, 16, 8)
    assert config.activation_block == fused_moe_ep_v3_tp.STREAM_ACTIVATION_BLOCK
    assert plan.startswith("stream 1024 columns x 256 rows, 2 weight slots")


def _parallel(tp, *, dp=1, pp=1, ep=True):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            pipeline_parallel_size=pp,
            enable_expert_parallel=ep,
        )
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"tp": 16}, True),
        ({"tp": 1}, False),
        ({"tp": 8, "dp": 2}, False),
        ({"tp": 16, "ep": False}, False),
    ],
)
def test_tp_attention_over_ep(monkeypatch, kwargs, expected):
    monkeypatch.setattr(
        vllm.config, "get_current_vllm_config", lambda: _parallel(**kwargs)
    )
    assert fused_moe_ep_v3_tp.tp_attention_over_ep() is expected


def test_v3_deployment_check_defers_tp_attention_and_still_refuses_tp_without_ep(
    monkeypatch,
):
    layer = SimpleNamespace()
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: _parallel(16))
    assert fused_moe_ep_v3._check_deployment(layer) is True
    monkeypatch.setattr(
        vllm.config, "get_current_vllm_config", lambda: _parallel(16, ep=False)
    )
    with pytest.raises(NotImplementedError, match="tensor_parallel_size=16"):
        fused_moe_ep_v3._check_deployment(layer)

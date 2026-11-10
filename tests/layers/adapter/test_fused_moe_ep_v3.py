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
"""MOE_FUSED_EP_KERNEL_IMPL=v3: selection, refusals and the per-shape gate.

None of this needs a TPU. Admission is decided before any op is built, and
the forward tests replace the two device ops (the fused op and the
grouped-matmul path) with dense torch references, so they check the routing
and dispatch plumbing around the ops, not the kernels.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import fused_moe_ep as bridge
from vllm_torchtpu.layers.adapter import fused_moe_ep_v3 as bridge_v3
from vllm_torchtpu.layers.adapter.quantization import fp8 as fp8_module
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

_STUB_EP = 2


def _layer(experts=4, hidden=8, inter=16, **overrides):
    """A layer whose weights and routing the v3 kernel would accept."""
    layer = SimpleNamespace(
        w13_weight=torch.zeros(experts, hidden, 2 * inter, dtype=torch.float8_e4m3fn),
        w2_weight=torch.zeros(experts, inter, hidden, dtype=torch.float8_e4m3fn),
        w13_weight_scale_inv=torch.ones(experts, 1, 1, 2 * inter),
        w2_weight_scale_inv=torch.ones(experts, 1, 1, hidden),
        global_num_experts=experts * _STUB_EP,
        custom_routing_function=None,
        scoring_func="softmax",
        e_score_correction_bias=None,
        use_grouped_topk=False,
        num_expert_group=None,
        topk_group=None,
        routed_scaling_factor=1.0,
        moe_config=SimpleNamespace(
            experts_per_token=2,
            moe_parallel_config=SimpleNamespace(
                use_ep=True, pcp_size=1, is_sequence_parallel=False
            ),
        ),
    )
    for name, value in overrides.items():
        setattr(layer, name, value)
    return layer


@pytest.fixture
def v3(monkeypatch):
    """The operator asked for MOE_FUSED_EP_KERNEL_IMPL=v3."""
    monkeypatch.setattr(envs, "MOE_FUSED_EP_KERNEL_IMPL", "v3")
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", True)
    monkeypatch.setattr(envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS", 1024)
    monkeypatch.setattr(envs, "MOE_FUSED_EP_SORTED_GATHER", False)
    monkeypatch.setattr(envs, "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS", 0)
    monkeypatch.setattr(envs, "TPU_MOE_SKIP_PADDED_TOKENS", False)
    monkeypatch.setattr(bridge_v3, "_tensor_parallel_size", lambda: 1)


def _prebuild(layer, *, smem_bytes=1024 * 1024, **kwargs):
    """Prebuild with the distributed lookups and both op builders stubbed."""
    mesh = SimpleNamespace(shape={bridge.EP_AXIS_NAME: _STUB_EP})
    calls = {"v2": [], "v3": []}
    with (
        patch.object(bridge, "build_ep_mesh", return_value=mesh),
        patch.object(bridge, "ep_mesh_index", return_value=0),
        patch.object(bridge, "ep_rank_order", return_value=None),
        patch.object(bridge, "ep_token_replica_groups", return_value=((0,), (1,))),
        patch.object(bridge, "_max_node_tokens", return_value=4096),
        patch.object(bridge, "_smem_capacity_bytes", return_value=smem_bytes),
        patch.object(
            bridge,
            "_build_op",
            side_effect=lambda *a, **k: calls["v2"].append((a, k)) or "v2 op",
        ),
        patch.object(
            bridge_v3,
            "build_op",
            side_effect=lambda *a, **k: calls["v3"].append((a, k)) or "v3 op",
        ),
    ):
        defaults = dict(topk=2, renormalize=True, activation="silu")
        defaults.update(kwargs)
        return bridge.prebuild_fused_moe_ep(layer, **defaults), calls


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_v3_is_a_kernel_choice_that_enables_fused_ep(monkeypatch):
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_IMPL", "v3")
    monkeypatch.delenv("USE_MOE_FUSED_EP_KERNEL", raising=False)
    assert envs.environment_variables["MOE_FUSED_EP_KERNEL_IMPL"]() == "v3"
    assert envs.environment_variables["USE_MOE_FUSED_EP_KERNEL"]() is True


def test_step_threshold_defaults_off_and_refuses_a_negative(monkeypatch):
    monkeypatch.delenv("TPU_MOE_FUSED_EP_STEP_MIN_TOKENS", raising=False)
    read = envs.environment_variables["TPU_MOE_FUSED_EP_STEP_MIN_TOKENS"]
    assert read() == 0
    monkeypatch.setenv("TPU_MOE_FUSED_EP_STEP_MIN_TOKENS", "48")
    assert read() == 48
    monkeypatch.setenv("TPU_MOE_FUSED_EP_STEP_MIN_TOKENS", "-1")
    with pytest.raises(ValueError):
        read()


def test_v3_arms_with_the_v3_op_and_never_builds_v2(v3):
    op, calls = _prebuild(_layer())
    assert op == "v3 op"
    assert not calls["v2"]
    args, kwargs = calls["v3"][0]
    assert args[1:5] == (2, True, "silu", None)  # topk, renormalize, act, order
    assert kwargs == dict(hidden=8, inter=16, local_experts=4)


def test_v3_is_not_held_to_the_v2_smem_bound(v3):
    op, _ = _prebuild(_layer(), smem_bytes=1)
    assert op == "v3 op"


def test_v2_is_unchanged_without_the_v3_choice(v3, monkeypatch):
    monkeypatch.setattr(envs, "MOE_FUSED_EP_KERNEL_IMPL", None)
    op, calls = _prebuild(_layer())
    assert op == "v2 op"
    assert not calls["v3"]


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (dict(scoring_func="sigmoid"), "softmax"),
        (dict(e_score_correction_bias=torch.zeros(8)), "correction bias"),
        (dict(routed_scaling_factor=2.5), "routed_scaling_factor"),
        (dict(hash_indices_table=torch.zeros(4, 2, dtype=torch.int32)), "hash"),
    ],
)
def test_v3_refuses_routing_its_softmax_top_k_cannot_reproduce(v3, overrides, reason):
    layer = _layer(**overrides)
    assert reason in bridge_v3.unsupported_reason(layer, "silu", "fp8")
    op, calls = _prebuild(layer, supplies_hash_ids=True)
    assert op is None
    assert not calls["v2"] and not calls["v3"]


def test_v3_refuses_other_formats_and_activations(v3):
    layer = _layer()
    assert "FP8" in bridge_v3.unsupported_reason(layer, "silu", "fp4")
    assert "silu" in bridge_v3.unsupported_reason(layer, "gelu", "fp8")


def test_v3_refuses_layouts_that_replicate_rows(v3, monkeypatch):
    pcp_layer = _layer()
    pcp_layer.moe_config.moe_parallel_config.pcp_size = 2
    assert "context parallelism" in bridge_v3.unsupported_reason(
        pcp_layer, "silu", "fp8"
    )
    monkeypatch.setattr(bridge_v3, "_tensor_parallel_size", lambda: 8)
    assert "tensor parallelism" in bridge_v3.unsupported_reason(_layer(), "silu", "fp8")


def test_v3_accepts_the_served_qwen35_routing(v3):
    assert bridge_v3.unsupported_reason(_layer(), "silu", "fp8") is None


# ---------------------------------------------------------------------------
# The per-shape gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("impl", "threshold", "rows", "takes"),
    [
        ("v3", 0, 1, True),
        ("v3", 48, 47, False),
        ("v3", 48, 48, True),
        ("v3", 48, 4096, True),
        ("v2", 48, 1, True),  # the threshold is a v3 option
        (None, 48, 1, True),
    ],
)
def test_fused_kernel_takes(monkeypatch, impl, threshold, rows, takes):
    monkeypatch.setattr(envs, "MOE_FUSED_EP_KERNEL_IMPL", impl)
    monkeypatch.setattr(envs, "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS", threshold)
    assert bridge_v3.fused_kernel_takes(rows) is takes


def test_retained_inputs_do_not_change_the_output(v3, monkeypatch):
    monkeypatch.setattr(envs, "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS", 48)
    out = torch.randn(4, 8)
    kept = bridge_v3.retain_fork_graph_inputs(
        out, torch.tensor(3, dtype=torch.int32), None
    )
    assert torch.equal(kept, out)


# ---------------------------------------------------------------------------
# The FP8 method's forward on both sides of the gate
# ---------------------------------------------------------------------------

_EXPERTS, _HIDDEN, _INTER, _TOPK = 4, 8, 16, 2


class _Experts(torch.nn.Module):
    """The attributes `apply_monolithic` and `moe_routing.route` read."""

    def __init__(self):
        super().__init__()
        gen = torch.Generator().manual_seed(0)
        self.w13_weight = torch.nn.Parameter(
            torch.randn(_EXPERTS, _HIDDEN, 2 * _INTER, generator=gen) / 4,
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            torch.randn(_EXPERTS, _INTER, _HIDDEN, generator=gen) / 4,
            requires_grad=False,
        )
        self.w13_weight_scale_inv = torch.nn.Parameter(
            torch.ones(_EXPERTS, 1, 1, 2 * _INTER), requires_grad=False
        )
        self.w2_weight_scale_inv = torch.nn.Parameter(
            torch.ones(_EXPERTS, 1, 1, _HIDDEN), requires_grad=False
        )
        self.register_buffer("_experts_start", torch.tensor(0, dtype=torch.int32))
        self.moe_config = SimpleNamespace(
            experts_per_token=_TOPK,
            moe_parallel_config=SimpleNamespace(use_ep=True, pcp_size=1),
        )
        self.renormalize = True
        self.scoring_func = "softmax"
        self.custom_routing_function = None
        self.use_grouped_topk = False
        self.num_expert_group = None
        self.topk_group = None
        self.e_score_correction_bias = None
        self.routed_scaling_factor = 1.0


def _expert_mix(x, w1, w2, topk_weights, topk_ids):
    """Sum over the selected experts of weight * silu-gated MLP."""
    out = torch.zeros_like(x, dtype=torch.float32)
    for e in range(w1.shape[0]):
        gate_up = x.float() @ w1[e].float()
        row = (F.silu(gate_up[:, :_INTER]) * gate_up[:, _INTER:]) @ w2[e].float()
        weight = torch.where(topk_ids == e, topk_weights, 0.0).sum(-1, keepdim=True)
        out = out + weight * row
    return out.to(x.dtype)


def _fused_op(x, w1, w2, w1_scale, w2_scale, gating, rank):
    """The fused op's contract on one rank: softmax top-k from the logits,
    renormalized, over every expert."""
    scores = torch.softmax(gating.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(scores, _TOPK, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
    out = _expert_mix(
        x, w1 * w1_scale[:, None, :], w2 * w2_scale[:, None, :], topk_weights, topk_ids
    )
    return out + (rank.reshape(-1)[0] * 0).to(out.dtype)


def _grouped_path(**kwargs):
    """The grouped-matmul path with its collectives, on one rank: the routing
    it was handed applied to the rows it was handed."""
    w1 = kwargs["w1"] * kwargs["w1_scale"].reshape(_EXPERTS, 1, -1)
    w2 = kwargs["w2"] * kwargs["w2_scale"].reshape(_EXPERTS, 1, -1)
    return _expert_mix(
        kwargs["hidden_states"], w1, w2, kwargs["topk_weights"], kwargs["topk_ids"]
    ) + (kwargs["experts_start"] * 0).to(kwargs["hidden_states"].dtype)


@pytest.fixture
def armed_method(v3, monkeypatch):
    """An FP8 MoE method armed with a fused op, with both device ops swapped
    for the references above. Records which path served each call."""
    served = []
    method = object.__new__(VllmFp8MoEMethodTPU)
    method._tpu_activation_str = "silu"

    def fused_op(*args):
        served.append("fused")
        return _fused_op(*args)

    def grouped_path(**kwargs):
        served.append("grouped")
        return _grouped_path(**kwargs)

    setattr(method, bridge.FUSED_MOE_EP_OP_ATTR, fused_op)
    monkeypatch.setattr(fp8_module, "pipelined_fused_moe_gmm", grouped_path)
    monkeypatch.setattr(bridge, "_RANK_BUFFER", torch.zeros(1, 1, dtype=torch.int32))
    return method, served


def _inputs(rows, seed=1):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, _HIDDEN, generator=gen)
    logits = torch.randn(rows, _EXPERTS, generator=gen)
    return x, logits


@pytest.mark.parametrize("rows", [4, 16])
def test_gate_picks_the_path_by_row_count(armed_method, monkeypatch, rows):
    method, served = armed_method
    monkeypatch.setattr(envs, "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS", 8)
    method.apply_monolithic(_Experts(), *_inputs(rows))
    assert served == ["fused" if rows >= 8 else "grouped"]


def test_small_steps_match_the_fused_path(armed_method, monkeypatch):
    """The same rows give the same output whichever side of the gate serves
    them: the threshold only moves where the dispatch happens."""
    method, served = armed_method
    layer = _Experts()
    x, logits = _inputs(4)
    monkeypatch.setattr(envs, "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS", 0)
    fused = method.apply_monolithic(layer, x, logits)
    monkeypatch.setattr(envs, "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS", 8)
    grouped = method.apply_monolithic(layer, x, logits)
    assert served == ["fused", "grouped"]
    torch.testing.assert_close(grouped, fused, rtol=1e-5, atol=1e-5)


def _lifted_inputs(method, layer, rows):
    """The tensor inputs of the graph Dynamo traces for `rows` rows, compiled
    the way vLLM compiles the backbone: fullgraph, static by default, the
    token dimension marked dynamic."""
    graphs = []

    def capture(gm, _example_inputs):
        graphs.append(
            [
                n.name
                for n in gm.graph.nodes
                if n.op == "placeholder"
                and isinstance(n.meta.get("example_value"), torch.Tensor)
            ]
        )
        return gm

    torch._dynamo.reset()
    compiled = torch.compile(
        method.apply_monolithic, backend=capture, fullgraph=True, dynamic=False
    )
    x, logits = _inputs(rows)
    torch._dynamo.mark_dynamic(x, 0)
    torch._dynamo.mark_dynamic(logits, 0)
    compiled(layer, x, logits)
    return graphs[-1]


@pytest.mark.parametrize("skip_padded_tokens", [False, True])
def test_both_sides_of_the_gate_lift_the_same_inputs(
    armed_method, monkeypatch, skip_padded_tokens
):
    """A bucket's executable can only be handed between the two traces when
    they lift the same inputs (compilation/shape_variants)."""
    from vllm_torchtpu.layers.adapter import token_padding

    method, served = armed_method
    monkeypatch.setattr(envs, "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS", 8)
    monkeypatch.setattr(envs, "TPU_MOE_SKIP_PADDED_TOKENS", skip_padded_tokens)
    if skip_padded_tokens:
        state = token_padding.TokenPaddingState.create(16, device="cpu")
        monkeypatch.setattr(token_padding, "_padding_state", state)
    layer = _Experts()
    fused = _lifted_inputs(method, layer, 16)
    grouped = _lifted_inputs(method, layer, 4)
    assert served == ["fused", "grouped"]
    assert sorted(fused) == sorted(grouped), (
        set(fused) - set(grouped),
        set(grouped) - set(fused),
    )
    assert any(name.endswith("fused_moe_ep_rank_buffer") for name in fused)
    assert any("experts_start" in name for name in fused)
    assert any("padding_mask" in name for name in fused) is skip_padded_tokens

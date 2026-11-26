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
"""Gating tests for the fused expert-parallel MoE bridge.

Every refusal in `prebuild_fused_moe_ep` is a claim that a configuration would
be served WRONG rather than merely slowly, so each one needs a test that says
it still refuses. None of this needs a TPU: the refusals are decided before any
op is built, and the one path that does build stops at `_build_op`.
"""

from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner import moe_runner as upstream
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.fused_moe.v2.host import token_gather_smem_bytes
from vllm_torchtpu.layers.adapter import fused_moe_ep as bridge
from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner
from vllm_torchtpu.layers.adapter.quantization import fp8

# The width of the mesh `_prebuild` stubs in. `_layer` sizes the global expert
# count against it, because the kernel requires one equal block per shard.
_STUB_EP = 2

# Positions in `_build_op`'s positional argument list, which is what the tests
# below inspect: (mesh, topk, renormalize, activation, mesh_expert_order,
# sharded_plan, weight_format, rhs_qb, scoring_fn, has_score_bias,
# routed_scaling_factor).
_MESH_ORDER_ARG = 4
_SHARDED_PLAN_ARG = 5
_WEIGHT_FORMAT_ARG = 6


def _layer(experts=4, hidden=8, inter=16, **overrides):
    """A layer whose weights and routing the kernel would accept."""
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


def _prebuild(
    layer,
    *,
    has_mesh=True,
    node_tokens=4096,
    smem_bytes=1024 * 1024,
    ep_order=None,
    replica_groups=((0,), (1,)),
    build_op=None,
    **kwargs,
):
    """Call prebuild with the distributed lookups stubbed out.

    `build_ep_mesh` does an `all_gather_object` over the EP group, so it is
    replaced by a two-way mesh stub. Everything under test decides before the
    op would be built, and `_build_op` is stubbed so the one accepting case
    does not need a device.
    """
    mesh = SimpleNamespace(shape={bridge.EP_AXIS_NAME: _STUB_EP}) if has_mesh else None
    build_op = build_op or (lambda *_args, **_kwargs: "op")
    with (
        patch.object(bridge, "build_ep_mesh", return_value=mesh),
        patch.object(bridge, "ep_mesh_index", return_value=0),
        patch.object(bridge, "ep_rank_order", return_value=ep_order),
        patch.object(bridge, "ep_token_replica_groups", return_value=replica_groups),
        patch.object(bridge, "_max_node_tokens", return_value=node_tokens),
        patch.object(bridge, "_smem_capacity_bytes", return_value=smem_bytes),
        patch.object(bridge, "_build_op", side_effect=build_op),
    ):
        defaults = dict(topk=2, renormalize=True, activation="silu")
        defaults.update(kwargs)
        return bridge.prebuild_fused_moe_ep(layer, **defaults)


@pytest.fixture(autouse=True)
def _default_scale_layouts():
    """Every scale table here gets the default layout unless a test says
    otherwise: torch_tpu's layout query needs a TPU."""
    with patch.object(bridge, "_default_scale_layout", return_value=[2, 1, 0]) as query:
        yield query


@pytest.fixture(autouse=True)
def _armed():
    """Every test here runs as if the operator asked for the kernel."""
    with (
        patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", True),
        patch.object(envs, "MOE_FUSED_EP_ENABLE_W4A8", True),
        patch.object(envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS", 1024),
        patch.object(envs, "MOE_FUSED_EP_V2_SHARDED_PLAN", False),
    ):
        yield


def test_arms_on_a_supported_layer():
    assert _prebuild(_layer()) == "op"


def test_native_replica_groups_reach_prebuilt_op():
    groups = ((0, 1),)
    seen = []
    assert (
        _prebuild(
            _layer(),
            replica_groups=groups,
            build_op=lambda *args, **kw: seen.append(kw) or "op",
        )
        == "op"
    )
    assert seen[0]["token_replica_groups"] == groups


@pytest.mark.parametrize(
    "w1_minor,w2_minor", [(False, False), (True, False), (False, True), (True, True)]
)
def test_fp4_scale_layouts_reach_prebuilt_op(
    _default_scale_layouts, w1_minor, w2_minor
):
    # Each table is read in the layout torch_tpu gives its own [E, blocks, N]
    # shape: w13 has 2 blocks here and w2 has 4.
    def layout(shape):
        minor = w1_minor if shape[1] == 2 else w2_minor
        return [2, 0, 1] if minor else [2, 1, 0]

    _default_scale_layouts.side_effect = layout
    seen = []
    assert (
        _prebuild(
            _fp4_layer(hidden=1024, inter=2048),
            weight_format="fp4",
            rhs_qb=512,
            build_op=lambda *args, **kw: seen.append(kw) or "op",
        )
        == "op"
    )
    queried = {c.args[0] for c in _default_scale_layouts.call_args_list}
    assert queried == {(1, 2, 4096), (1, 4, 1024)}
    assert seen[0]["w1_scales_expert_second_minor"] is w1_minor
    assert seen[0]["w2_scales_expert_second_minor"] is w2_minor


def test_fp8_never_reads_scales_expert_second_minor(_default_scale_layouts):
    _default_scale_layouts.return_value = [2, 0, 1]
    seen = []
    assert (
        _prebuild(_layer(), build_op=lambda *args, **kw: seen.append(kw) or "op")
        == "op"
    )
    _default_scale_layouts.assert_not_called()
    assert seen[0]["w1_scales_expert_second_minor"] is False
    assert seen[0]["w2_scales_expert_second_minor"] is False


@pytest.mark.parametrize(
    "staged,expected,warns",
    [
        ([2, 0, 1], [2, 0, 1], False),
        ([2, 1, 0], [2, 1, 0], False),
        ([2, 1, 0], [2, 0, 1], True),
        ([2, 0, 1], [2, 1, 0], True),
    ],
)
def test_staged_scale_layout_mismatch_warns(
    _default_scale_layouts, staged, expected, warns
):
    _default_scale_layouts.return_value = expected
    with (
        patch.object(bridge, "_staged_scale_layout", return_value=staged),
        patch.object(bridge.logger, "warning_once") as warning,
    ):
        bridge._check_staged_scale_layouts([torch.ones(24, 14, 256)])
    assert warning.called is warns


@pytest.mark.parametrize("sharded_plan", [False, True])
@pytest.mark.parametrize(
    "scoring_fn,has_bias,scale",
    [
        ("softmax", False, 1.0),
        ("sigmoid", True, 2.5),
        ("sqrtsoftplus", True, 0.75),
    ],
)
def test_replica_groups_are_part_of_op_cache_and_kernel_call(
    monkeypatch, sharded_plan, scoring_fn, has_bias, scale
):
    from unittest.mock import Mock

    from vllm_torchtpu.kernels.fused_moe import v2

    monkeypatch.setattr(bridge, "_OPS", {})
    closures = []

    def build(name, fn, **kwargs):
        closures.append(fn)
        return Mock()

    # This branch builds the fused EP op with Torchtpu's native
    # jax_op; the local shim it used to go through is gone.
    monkeypatch.setattr(bridge, "jax_op", build)
    kernel = Mock(side_effect=lambda x, *args, **kwargs: x)
    monkeypatch.setattr(v2, "fused_ep_moe_v2", kernel)
    groups = ((0, 1),)
    args = (None, 2, True, "silu", None, sharded_plan)
    options = dict(
        scoring_fn=scoring_fn,
        has_score_bias=has_bias,
        routed_scaling_factor=scale,
        token_replica_groups=groups,
    )
    replicated = bridge._build_op(*args, **options)
    assert bridge._build_op(*args, **options) is replicated
    # Every independent routing setting and the TP membership separate caches.
    for changed in (
        dict(token_replica_groups=((0,), (1,))),
        dict(scoring_fn="sigmoid" if scoring_fn == "softmax" else "softmax"),
        dict(has_score_bias=not has_bias),
        dict(routed_scaling_factor=scale + 1.0),
        dict(w1_scales_expert_second_minor=True),
        dict(w2_scales_expert_second_minor=True),
        dict(agq_min_local_tokens=2048),
        dict(use_route_kernel=True),
    ):
        assert bridge._build_op(*args, **(options | changed)) is not replicated
    assert len(closures) == 9
    x = jnp.ones((8, 8), dtype=jnp.bfloat16)
    bias = jnp.arange(4, dtype=jnp.bfloat16)
    operands = (
        x,
        None,
        None,
        None,
        None,
        jnp.zeros((8, 4)),
        jnp.zeros((1, 1), dtype=jnp.int32),
    )
    if has_bias:
        operands += (bias,)
    assert closures[0](*operands) is x
    forwarded = kernel.call_args.kwargs
    assert forwarded["token_replica_groups"] == groups
    assert forwarded["scoring_fn"] == scoring_fn
    assert forwarded["routed_scaling_factor"] == scale
    assert forwarded["sharded_plan"] is sharded_plan
    assert forwarded["w1_scales_expert_second_minor"] is False
    assert forwarded["w2_scales_expert_second_minor"] is False
    assert forwarded["use_route_kernel"] is False
    if has_bias:
        np.testing.assert_array_equal(forwarded["score_bias"], bias)
        assert forwarded["score_bias"].dtype == jnp.float32
    else:
        assert "score_bias" not in forwarded


def test_sharded_plan_flag_reaches_the_built_op():
    seen = []
    with patch.object(envs, "MOE_FUSED_EP_V2_SHARDED_PLAN", True):
        assert (
            _prebuild(
                _layer(), build_op=lambda *args, **kwargs: seen.append(args) or "op"
            )
            == "op"
        )
    assert seen and seen[0][_SHARDED_PLAN_ARG] is True


@pytest.mark.parametrize("enabled", [False, True])
def test_route_kernel_flag_reaches_the_built_op(enabled):
    seen = []
    with patch.object(envs, "MOE_FUSED_EP_V2_ROUTE_KERNEL", enabled):
        assert (
            _prebuild(
                _layer(),
                build_op=lambda *args, **kwargs: seen.append(kwargs) or "op",
            )
            == "op"
        )
    assert seen and seen[0]["use_route_kernel"] is enabled


def test_mesh_expert_order_is_closed_into_the_built_op():
    order = (1, 0)
    seen = []
    assert (
        _prebuild(
            _layer(),
            ep_order=order,
            build_op=lambda *args, **kwargs: seen.append(args) or "op",
        )
        == "op"
    )
    assert seen and seen[0][_MESH_ORDER_ARG] == order


def test_mesh_expert_order_is_none_when_ranks_follow_device_ids():
    """The permutation exists only to undo a mesh ordered by device id; where
    EP rank order already is device order the op must be handed nothing."""
    seen = []
    assert (
        _prebuild(
            _layer(),
            ep_order=tuple(range(_STUB_EP)),
            build_op=lambda *args, **kwargs: seen.append(args) or "op",
        )
        == "op"
    )
    assert seen and seen[0][_MESH_ORDER_ARG] is None


def test_ep_rank_order_is_none_without_a_group():
    from vllm_torchtpu.distributed import ep_mesh

    with patch.object(ep_mesh, "get_ep_group", return_value=None):
        assert ep_mesh.ep_rank_order() is None


def test_refused_without_the_env_flag():
    with patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", False):
        assert _prebuild(_layer()) is None


def test_refused_when_experts_are_replicated():
    """vLLM builds an EP group for any MoE model, EP enabled or not.

    Without `use_ep` each rank holds the whole expert set, so the kernel would
    read its shard as one 1/ep block of a global set ep times too large.
    """
    layer = _layer()
    layer.moe_config.moe_parallel_config.use_ep = False
    assert _prebuild(layer) is None


def test_arms_under_prefill_context_parallelism():
    layer = _layer()
    layer.moe_config.moe_parallel_config.pcp_size = 4
    assert _prebuild(layer) == "op"


def test_pcp_threshold_counts_one_logical_scheduler_batch():
    """PCP partitions one batch; DP contributes independent batches."""
    cfg = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=4096))
    with patch("vllm.config.get_current_vllm_config", return_value=cfg):
        assert bridge._max_node_tokens(ep=8, pcp=8) == 4096
        assert bridge._max_node_tokens(ep=8, pcp=1) == 32768


def test_refused_without_an_ep_group():
    assert _prebuild(_layer(), has_mesh=False) is None


def test_refused_below_the_token_threshold():
    with patch.object(envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS", 1 << 20):
        assert _prebuild(_layer()) is None


def test_qwen_4096_bucket_is_not_refused_by_smem():
    """A 4K/rank DP8 bucket no longer scales the SMEM working set."""
    layer = _layer(experts=256)  # 512 global experts over the stub's ep=2
    assert _prebuild(layer, topk=10, node_tokens=32768) == "op"


def test_fixed_smem_working_set_is_still_checked():
    gather_bytes = token_gather_smem_bytes(bridge._TILE_M)
    assert gather_bytes == 2048
    needed = bridge._SMEM_OVERHEAD_BYTES + gather_bytes
    assert _prebuild(_layer(), smem_bytes=needed) == "op"
    assert _prebuild(_layer(), smem_bytes=needed - 1) is None


@pytest.mark.parametrize(
    "weights",
    [
        dict(
            w13_weight=torch.zeros(4, 8, 32, dtype=torch.bfloat16),
            w2_weight=torch.zeros(4, 16, 8, dtype=torch.bfloat16),
        ),
        dict(w2_weight=torch.zeros(4, 999, 8, dtype=torch.float8_e4m3fn)),
        dict(w13_weight_scale_inv=torch.ones(4, 3, 1, 32)),
        dict(w2_weight_scale_inv=torch.ones(4, 3, 1, 8)),
    ],
)
def test_refused_on_an_unservable_weight_layout(weights):
    assert _prebuild(_layer(**weights)) is None


def test_refused_when_the_expert_blocks_are_not_equal():
    """The kernel reads an expert id as `mesh_index * (e_total // ep) + j`, so
    an uneven `determine_expert_map` split -- or EPLB -- would address the
    wrong shard for every id past the first uneven boundary."""
    assert _prebuild(_layer(global_num_experts=9)) is None
    assert _prebuild(_layer(global_num_experts=4 * _STUB_EP)) == "op"


@pytest.mark.parametrize("activation", ["gelu", "silu_bf16"])
def test_refused_on_a_non_silu_activation(activation):
    # "silu_bf16" (additional_config tpu_moe_silu_bf16) needs bfloat16
    # roundings this kernel does not fuse, so such a layer must stay on gmm_v2.
    assert _prebuild(_layer(), activation=activation) is None


def test_refused_when_the_layer_carries_expert_biases():
    """The kernel takes biases; this bridge does not pass them. Dropping one
    is silent -- the output is wrong by the bias term on every routed token."""
    assert _prebuild(_layer(w13_bias=torch.zeros(4, 32))) is None
    assert _prebuild(_layer(w2_bias=torch.zeros(4, 8))) is None


@pytest.mark.parametrize(
    "routing",
    [
        dict(custom_routing_function=lambda **_: None),
        dict(use_grouped_topk=True, num_expert_group=8, topk_group=2),
        dict(hash_indices_table=torch.zeros(8, dtype=torch.int32)),
    ],
)
def test_refused_on_routing_the_kernel_does_not_implement(routing):
    """The fused path never calls `moe_routing.route`: the kernel selects
    top-k itself. Anything it cannot reproduce routes tokens to a different
    set of experts, so it is refused rather than run."""
    assert _prebuild(_layer(**routing)) is None


def test_refused_while_a_non_uniform_routing_simulator_is_on():
    from vllm_torchtpu.layers.adapter import moe_routing

    with (
        patch.object(moe_routing, "_SIMULATION_STRATEGY", object()),
        patch.object(moe_routing, "_strategy_name", "normal_routing"),
    ):
        assert _prebuild(_layer()) is None


def test_arms_under_uniform_random_routing_simulation():
    """uniform_random is honoured in-op (`_simulated_routing_operands`)."""
    from vllm_torchtpu.layers.adapter import moe_routing

    with (
        patch.object(moe_routing, "_SIMULATION_STRATEGY", object()),
        patch.object(moe_routing, "_strategy_name", moe_routing._UNIFORM_RANDOM),
    ):
        assert _prebuild(_layer()) == "op"


def test_simulated_routing_operands_are_untouched_without_a_simulator():
    from vllm_torchtpu.layers.adapter import moe_routing

    logits = torch.randn(4, 8, dtype=torch.bfloat16)
    bias = torch.randn(8)
    ids = torch.zeros(4, 2, dtype=torch.int32)
    with patch.object(moe_routing, "_SIMULATION_STRATEGY", None):
        got = bridge._simulated_routing_operands(logits, bias, ids)
    assert got[0] is logits and got[1] is bias and got[2] is ids


def test_simulated_routing_operands_draw_uniform_experts():
    from vllm_torchtpu.layers.adapter import moe_routing

    tokens, experts, topk = 8192, 64, 4
    logits = torch.randn(tokens, experts, dtype=torch.bfloat16)
    bias = torch.randn(experts)
    ids = torch.zeros(tokens, topk, dtype=torch.int32)
    with (
        patch.object(moe_routing, "_SIMULATION_STRATEGY", object()),
        patch.object(moe_routing, "_strategy_name", moe_routing._UNIFORM_RANDOM),
    ):
        gating, zero_bias, sim_ids = bridge._simulated_routing_operands(
            logits, bias, ids
        )
    # f32 noise: bf16 would tie and bias top-k toward low expert ids.
    assert gating.dtype == torch.float32 and gating.shape == logits.shape
    assert torch.equal(zero_bias, torch.zeros_like(bias))
    assert sim_ids.dtype == torch.int32 and sim_ids.shape == ids.shape
    # Hash ids are the noise's own top-k: distinct per token, in range.
    assert torch.equal(sim_ids, torch.topk(gating, k=topk, dim=-1).indices.int())
    assert int(sim_ids.min()) >= 0 and int(sim_ids.max()) < experts
    assert all(len(set(row)) == topk for row in sim_ids.tolist())
    # Uniform: every expert's load is near tokens * topk / experts (512 here).
    load = torch.bincount(sim_ids.flatten().long(), minlength=experts).float()
    assert float(load.max() / load.mean()) < 1.25


def test_arming_is_per_layer_not_per_process():
    """One armed layer must not answer for a layer that was refused."""
    armed = SimpleNamespace()
    refused = SimpleNamespace()
    setattr(armed, bridge.FUSED_MOE_EP_OP_ATTR, _prebuild(_layer()))
    setattr(
        refused,
        bridge.FUSED_MOE_EP_OP_ATTR,
        _prebuild(_layer(w2_bias=torch.zeros(4, 8))),
    )

    assert bridge.fused_moe_ep_supported(armed)
    assert not bridge.fused_moe_ep_supported(refused)
    assert not bridge.fused_moe_ep_supported(SimpleNamespace())
    assert not bridge.fused_moe_ep_supported(None)


@pytest.mark.timeout(10)
def test_squeeze_channel_scale_terminates_on_a_non_singleton_axis():
    """The hot-path squeeze must not be able to spin: a raise here reaches the
    operator as `Unsupported: Observed exception` with the message discarded,
    because the backbone compiles with fullgraph=True.

    Timed out rather than left to the step timeout: without the guard this
    loops forever, and an assertion alone would never be reached.
    """
    scale = torch.ones(4, 3, 1, 256)
    assert bridge._squeeze_channel_scale_checked(scale).shape == scale.shape


def test_selected_expert_ids_move_each_rank_block_to_its_mesh_index():
    """Relabel only selected IDs, not the full router-logit tensor."""
    from vllm_torchtpu.kernels.fused_moe.v2.layer import (
        _relabel_expert_ids_to_mesh_order,
    )

    ep, per_shard = 4, 2
    order = (0, 2, 3, 1)
    selected = jnp.arange(ep * per_shard, dtype=jnp.int32).reshape(2, 4)
    got = np.asarray(
        _relabel_expert_ids_to_mesh_order(
            selected, g_local=per_shard, mesh_ep_ranks=order
        )
    )
    mesh_index_of_rank = {ep_rank: i for i, ep_rank in enumerate(order)}
    want = np.asarray(
        [
            [
                mesh_index_of_rank[int(e) // per_shard] * per_shard + int(e) % per_shard
                for e in row
            ]
            for row in np.asarray(selected)
        ],
        np.int32,
    )
    np.testing.assert_array_equal(got, want)


def test_forward_passes_router_logits_without_a_full_expert_gather():
    """The Torch graph must hand the original logits straight to the op."""
    seen = []
    owner = SimpleNamespace()

    def op(*args):
        seen.append(args)
        return args[0]

    setattr(owner, bridge.FUSED_MOE_EP_OP_ATTR, op)
    hidden = torch.zeros(1, 8)
    weights = torch.empty(0)
    scales = torch.ones(1, 1)
    logits = torch.zeros(1, 8)
    assert (
        bridge.fused_moe_ep(owner, hidden, weights, weights, scales, scales, logits)
        is hidden
    )
    assert seen[0][5] is logits


def test_forward_masks_padded_router_logits_when_enabled(monkeypatch):
    """Padded rows of router_logits become NaN when TPU_MOE_SKIP_PADDED_TOKENS=1."""
    from vllm_torchtpu.layers.adapter import token_padding

    seen = []
    owner = SimpleNamespace()
    setattr(
        owner, bridge.FUSED_MOE_EP_OP_ATTR, lambda *args: seen.append(args) or args[0]
    )
    hidden = torch.zeros(4, 8)
    weights = torch.empty(0)
    scales = torch.ones(1, 1)
    logits = torch.ones(4, 8, dtype=torch.float32)
    state = token_padding.TokenPaddingState(
        local_padding_mask=torch.tensor([False, True, False, True])
    )
    token_padding.set_padding_state(state)
    try:
        monkeypatch.setattr(envs, "TPU_MOE_SKIP_PADDED_TOKENS", False)
        bridge.fused_moe_ep(owner, hidden, weights, weights, scales, scales, logits)
        assert seen[-1][5] is logits

        monkeypatch.setattr(envs, "TPU_MOE_SKIP_PADDED_TOKENS", True)
        bridge.fused_moe_ep(owner, hidden, weights, weights, scales, scales, logits)
        masked = seen[-1][5]
        assert torch.equal(masked[0], logits[0])
        assert torch.isnan(masked[1]).all()
        assert torch.equal(masked[2], logits[2])
        assert torch.isnan(masked[3]).all()
    finally:
        token_padding.set_padding_state(None)


def test_forward_masks_padding_after_uniform_random_simulation(monkeypatch):
    """Padded rows stay NaN and real rows finite under uniform_random.

    Masking first would put NaN into the simulation's mean anchor and so into
    every row, leaving every token unrouted.
    """
    from vllm_torchtpu.layers.adapter import moe_routing, token_padding

    seen = []
    owner = SimpleNamespace()
    setattr(
        owner, bridge.FUSED_MOE_EP_OP_ATTR, lambda *args: seen.append(args) or args[0]
    )
    hidden = torch.zeros(4, 8)
    weights = torch.empty(0)
    scales = torch.ones(1, 1)
    logits = torch.ones(4, 8, dtype=torch.bfloat16)
    monkeypatch.setattr(envs, "TPU_MOE_SKIP_PADDED_TOKENS", True)
    monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", object())
    monkeypatch.setattr(moe_routing, "_strategy_name", moe_routing._UNIFORM_RANDOM)
    token_padding.set_padding_state(
        token_padding.TokenPaddingState(
            local_padding_mask=torch.tensor([False, True, False, True])
        )
    )
    try:
        bridge.fused_moe_ep(owner, hidden, weights, weights, scales, scales, logits)
    finally:
        token_padding.set_padding_state(None)
    gating = seen[-1][5]
    assert gating.dtype == torch.float32
    assert torch.isfinite(gating[[0, 2]]).all()
    assert torch.isnan(gating[[1, 3]]).all()


@pytest.mark.parametrize(
    "order,expected",
    [
        ((0, 1, 2, 3), None),
        ((0, 2, 3, 1), (0, 2, 3, 1)),
    ],
)
def test_mesh_expert_order(order, expected):
    with patch.object(bridge, "ep_rank_order", return_value=order):
        assert bridge._mesh_expert_order(len(order)) == expected


def test_ep_rank_order_maps_ranks_to_ascending_device_ids():
    """The mesh is in ascending device-id order (the rank-ordered one makes the
    fused MoE's all-gathers slow), so this is what says which rank sits where."""
    from vllm_torchtpu.distributed import ep_mesh

    # EP ranks 0..7 on a tpu7x host's device ids. Sorted ascending, rank r ends
    # up at the position of its device id.
    device_ids = (0, 1, 4, 5, 6, 7, 2, 3)
    group = SimpleNamespace(ranks=list(range(8)), world_size=8)
    with (
        patch.object(ep_mesh, "get_ep_group", return_value=group),
        patch.object(ep_mesh, "ep_device_ids", return_value=device_ids),
    ):
        order = ep_mesh.ep_rank_order()

    assert order == tuple(rank for _, rank in sorted(zip(device_ids, range(8))))
    assert order == (0, 1, 6, 7, 2, 3, 4, 5)


def test_supports_internal_mk_tracks_both_owners():
    """`supports_internal_mk` claims vLLM's dispatch/combine. Either owner --
    chunk pipelining or the fused kernel -- has to claim it alone, and the
    fused half must follow what prebuild ARMED, not what the operator asked
    for: on a refusal vLLM's collectives have to stay."""
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

    prop = VllmFp8MoEMethodTPU.supports_internal_mk
    armed = SimpleNamespace()
    setattr(armed, bridge.FUSED_MOE_EP_OP_ATTR, "op")

    with patch(
        "vllm_torchtpu.layers.adapter.quantization.fp8."
        "enable_pipelined_collective_and_compute",
        return_value=False,
    ):
        assert prop.fget(SimpleNamespace()) is False
        assert prop.fget(armed) is True

    with patch(
        "vllm_torchtpu.layers.adapter.quantization.fp8."
        "enable_pipelined_collective_and_compute",
        return_value=True,
    ):
        assert prop.fget(SimpleNamespace()) is True


def test_env_flag_alone_does_not_claim_the_collectives():
    """The refusals only hold if the request does not claim ownership."""
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

    prop = VllmFp8MoEMethodTPU.supports_internal_mk
    with (
        patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", True),
        patch(
            "vllm_torchtpu.layers.adapter.quantization.fp8."
            "enable_pipelined_collective_and_compute",
            return_value=False,
        ),
    ):
        assert prop.fget(SimpleNamespace()) is False


class TestFusedOutputIsReduced:
    """`VllmMoERunner._fused_output_is_reduced` overrides an upstream answer.

    It is the guard that stops vLLM all-reducing an output the kernel already
    combined, so it has to be right about three things: it stays out of runs
    that do not use the kernel, it defers to upstream under
    `skip_final_all_reduce`, and it answers per layer.
    """

    def _answer(self, *, armed, skip_final_all_reduce=False):
        # `moe_kernel=None` is what the TPU fp8 method actually has, and it
        # is what makes upstream's own getter return False -- so a True here
        # can only have come from the override.
        quant_method = SimpleNamespace(moe_kernel=None)
        if armed:
            setattr(quant_method, bridge.FUSED_MOE_EP_OP_ATTR, "op")
        runner = VllmMoERunner.__new__(VllmMoERunner)
        torch.nn.Module.__init__(runner)
        runner.routed_experts = SimpleNamespace(quant_method=quant_method)
        runner.moe_config = SimpleNamespace(skip_final_all_reduce=skip_final_all_reduce)
        return runner._fused_output_is_reduced

    def test_defers_to_upstream_without_the_env_flag(self):
        with patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", False):
            assert self._answer(armed=True) is False

    def test_reports_reduced_only_for_an_armed_layer(self):
        with patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", True):
            assert self._answer(armed=True) is True
            assert self._answer(armed=False) is False

    def test_defers_to_upstream_under_skip_final_all_reduce(self):
        """The upstream assert requires an un-reduced fused output there."""
        with patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", True):
            assert self._answer(armed=True, skip_final_all_reduce=True) is False


def _fp4_layer(block=512, hidden=512, inter=512):
    layer = _layer(experts=1, hidden=hidden, inter=inter)
    layer.w13_weight = torch.empty((1, hidden, inter), dtype=torch.float4_e2m1fn_x2)
    layer.w2_weight = torch.empty((1, inter, hidden // 2), dtype=torch.float4_e2m1fn_x2)
    layer.w13_weight_scale = torch.ones(1, hidden // block, 1, 2 * inter)
    layer.w2_weight_scale = torch.ones(1, inter // block, 1, hidden)
    del layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
    return layer


@pytest.mark.parametrize("setting,enabled", [(None, False), ("0", False), ("1", True)])
def test_nvfp4_requires_explicit_w4a8_opt_in(monkeypatch, setting, enabled):
    name = "MOE_FUSED_EP_ENABLE_W4A8"
    if setting is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, setting)
    value = envs.environment_variables[name]()
    assert value is enabled
    monkeypatch.setattr(envs, name, value)
    assert _prebuild(_fp4_layer(), weight_format="fp4", rhs_qb=512) == (
        "op" if enabled else None
    )
    # The new opt-in applies only to FP4; FP8 fused EP retains its behavior.
    assert _prebuild(_layer()) == "op"


@pytest.mark.parametrize("block", [64, 128, 256, 512, 1024])
def test_nvfp4_aligned_block_arms_and_passes_format(block):
    seen = []
    assert (
        _prebuild(
            _fp4_layer(block, hidden=1024, inter=1024),
            weight_format="fp4",
            rhs_qb=block,
            build_op=lambda *args, **kwargs: seen.append(args) or "op",
        )
        == "op"
    )
    assert seen[0][_WEIGHT_FORMAT_ARG : _WEIGHT_FORMAT_ARG + 2] == ("fp4", block)


@pytest.mark.parametrize("block", [None, 0, -64, 16, 32, 96, 192, 1024])
def test_nvfp4_refuses_invalid_blocks(block):
    assert _prebuild(_fp4_layer(), weight_format="fp4", rhs_qb=block) is None


@pytest.mark.parametrize("hidden,inter", [(768, 1024), (1024, 768)])
def test_nvfp4_block_must_divide_both_contractions(hidden, inter):
    assert (
        _prebuild(
            _fp4_layer(hidden=hidden, inter=inter), weight_format="fp4", rhs_qb=512
        )
        is None
    )


@pytest.mark.parametrize("block", [64, 256, 512])
@pytest.mark.parametrize("fault", ["scale", "dtype", "routing", "bias"])
def test_nvfp4_refuses_invalid_contract(fault, block):
    layer = _fp4_layer(block)
    if fault == "scale":
        layer.w2_weight_scale = torch.ones(1, 32, 1, 512)
    elif fault == "dtype":
        layer.w2_weight = torch.empty((1, 512, 512), dtype=torch.float8_e4m3fn)
    elif fault == "routing":
        layer.hash_indices_table = torch.zeros(8, dtype=torch.int32)
    else:
        layer.w2_bias = torch.zeros(1, 512)
    assert _prebuild(layer, weight_format="fp4", rhs_qb=block) is None


def test_nvfp4_forward_preserves_single_block_scale_axis():
    layer = _fp4_layer()
    seen = []
    owner = SimpleNamespace()
    setattr(
        owner, bridge.FUSED_MOE_EP_OP_ATTR, lambda *args: seen.append(args) or args[0]
    )
    s1 = layer.w13_weight_scale.squeeze(2)
    s2 = layer.w2_weight_scale.squeeze(2)
    x = torch.ones(2, 512)
    assert (
        bridge.fused_moe_ep(
            owner, x, layer.w13_weight, layer.w2_weight, s1, s2, torch.ones(2, 2)
        )
        is x
    )
    assert seen[0][3] is s1 and seen[0][4] is s2
    assert seen[0][3].shape == (1, 1, 1024)


@pytest.mark.parametrize(
    "armed,pipelined", [(False, False), (True, False), (False, True), (True, True)]
)
def test_nvfp4_communication_ownership(armed, pipelined):
    from vllm_torchtpu.layers.adapter.quantization import nvfp4

    method = object.__new__(nvfp4.VllmNvfp4MoEMethod)
    setattr(method, bridge.FUSED_MOE_EP_OP_ATTR, "op" if armed else None)
    with patch.object(
        nvfp4, "enable_pipelined_collective_and_compute", return_value=pipelined
    ):
        assert method.supports_internal_mk == (armed or pipelined)


def _nvfp4_checkpoint_layer(hidden=512, inter=512, use_ep=True):
    layer = torch.nn.Module()
    for name, shape, dtype in (
        ("w13_weight", (1, 2 * inter, hidden // 2), torch.uint8),
        ("w2_weight", (1, hidden, inter // 2), torch.uint8),
        ("w13_weight_scale", (1, 2 * inter, hidden // 16), torch.float8_e4m3fn),
        ("w2_weight_scale", (1, hidden, inter // 16), torch.float8_e4m3fn),
        ("w13_weight_scale_2", (1, 2), torch.float32),
        ("w2_weight_scale_2", (1,), torch.float32),
    ):
        layer.register_parameter(
            name,
            torch.nn.Parameter(torch.ones(shape, dtype=dtype), requires_grad=False),
        )
    layer.w13_weight_scale_2.data.copy_(torch.tensor([[2.0, 3.0]]))
    layer.w2_weight_scale_2.data.fill_(4.0)
    layer.activation = "silu"
    layer.renormalize = True
    # RoutedExperts.__init__ always sets these, and the test stubs it out.
    layer.custom_routing_function = None
    layer.scoring_func = "softmax"
    layer.e_score_correction_bias = None
    layer.use_grouped_topk = False
    layer.num_expert_group = None
    layer.topk_group = None
    layer.routed_scaling_factor = 1.0
    layer.global_num_experts = _STUB_EP
    layer.moe_config = SimpleNamespace(
        experts_per_token=2,
        moe_parallel_config=SimpleNamespace(
            use_ep=use_ep, pcp_size=1, is_sequence_parallel=False
        ),
    )
    return layer


@pytest.mark.parametrize("block", [None, 64, 128, 256, 512])
@pytest.mark.parametrize("execution", ["fused", "gmm", "pipelined"])
def test_nvfp4_load_prepares_scales_and_dispatches_only_when_armed(
    monkeypatch, execution, block
):
    from vllm_torchtpu.layers.adapter.quantization import nvfp4

    admitted = execution == "fused"
    effective_block = 64 if block is None else block
    layer = _nvfp4_checkpoint_layer()
    method = object.__new__(nvfp4.VllmNvfp4MoEMethod)
    method.moe = SimpleNamespace(has_bias=False, is_act_and_mul=True)
    method.group_size = 16
    seen = []
    gmm_options = []

    def requant(w, scales, block):
        seen.append((scales.clone(), block))
        e, n, half_k = w.shape
        return (
            torch.empty((e, half_k * 2, n // 2), dtype=torch.float4_e2m1fn_x2),
            torch.ones(e, half_k * 2 // block, 1, n),
        )

    def op(*args):
        seen.append(args)
        return args[0]

    monkeypatch.setattr(nvfp4, "RoutedExperts", torch.nn.Module)
    monkeypatch.setattr(nvfp4, "requant_load_kmajor_fp4", requant)
    monkeypatch.setattr(
        nvfp4,
        "load_kmajor_fp4",
        lambda w: torch.empty(
            (w.shape[0], w.shape[2] * 2, w.shape[1] // 2), dtype=torch.float4_e2m1fn_x2
        ),
    )
    monkeypatch.setattr(
        nvfp4, "prebuild_fused_moe_kernel", lambda **kw: gmm_options.append(kw)
    )
    monkeypatch.setattr(
        nvfp4.moe_routing, "register_experts_start_buffer", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        nvfp4.moe_routing, "validate_linear_ep_placement", lambda *a: None
    )

    def prebuild(candidate, **kwargs):
        assert candidate.w13_weight.dtype == torch.uint8
        assert seen == []  # Admission must precede requantization.
        assert all(w.device.type == "meta" for w in kwargs["weights"])
        assert kwargs["weight_format"] == "fp4"
        assert kwargs["rhs_qb"] == effective_block
        return op if admitted else None

    monkeypatch.setattr(bridge, "prebuild_fused_moe_ep", prebuild)
    monkeypatch.setattr(envs, "MOE_REQUANTIZE_BLOCK_SIZE", block)
    method.process_weights_after_loading(layer)
    # Requantization must not enable FP8 activations in the GMM fallback,
    # even when the layer is also admitted to fused EP.
    assert len(gmm_options) == 1
    assert not gmm_options[0].get("quantize_fp4_lhs", False)
    if admitted or block is not None:
        assert seen[0][1] == seen[1][1] == effective_block
        torch.testing.assert_close(seen[0][0][:, :512], torch.full((1, 512, 32), 2.0))
        torch.testing.assert_close(seen[0][0][:, 512:], torch.full((1, 512, 32), 3.0))
        torch.testing.assert_close(seen[1][0], torch.full((1, 512, 32), 4.0))
    else:
        assert seen == []
        assert layer.w13_weight_scale.shape == (1, 32, 1, 1024)
        assert layer.w2_weight_scale.shape == (1, 32, 1, 512)
    assert bridge.fused_moe_ep_supported(method) == admitted
    if admitted:
        x = torch.ones(2, 512)
        assert method.apply_monolithic(layer, x, torch.ones(2, 2)) is x
        assert seen[-1][3].shape == (1, 512 // effective_block, 1024)
        assert seen[-1][4].shape == (1, 512 // effective_block, 512)
    else:
        assert not hasattr(layer, "_tpu_fused_w13_scale")
        assert layer.w13_weight_scale.ndim == 4
        layer._experts_start = torch.zeros((), dtype=torch.int32)
        fallback_calls = []

        def fallback(path, **kwargs):
            fallback_calls.append((path, kwargs))
            return kwargs["hidden_states"]

        monkeypatch.setattr(nvfp4, "fused_moe_gmm", lambda **kw: fallback("gmm", **kw))
        monkeypatch.setattr(
            nvfp4, "pipelined_fused_moe_gmm", lambda **kw: fallback("pipelined", **kw)
        )
        monkeypatch.setattr(
            nvfp4,
            "enable_pipelined_collective_and_compute",
            lambda: execution == "pipelined",
        )
        monkeypatch.setattr(
            nvfp4.moe_routing,
            "route",
            lambda *args, **kwargs: (
                torch.ones(2, 2, dtype=torch.bfloat16),
                torch.zeros(2, 2, dtype=torch.int32),
            ),
        )
        x = torch.ones(2, 512, dtype=torch.bfloat16)
        assert method.apply_monolithic(layer, x, torch.ones(2, 2)) is x
        assert len(fallback_calls) == 1
        path, kwargs = fallback_calls[0]
        assert path == execution
        assert kwargs["hidden_states"].dtype == torch.bfloat16
        assert kwargs["w1"].dtype == torch.float4_e2m1fn_x2
        assert kwargs["w2"].dtype == torch.float4_e2m1fn_x2
        assert not kwargs.get("quantize_fp4_lhs", False)


@pytest.mark.parametrize("w4a8_enabled", [False, True])
@pytest.mark.parametrize(
    "fused_enabled,use_ep,configured_block,hidden,inter,expected_block,armed",
    [
        (True, True, None, 512, 512, 64, True),
        (True, True, None, 512, 528, 64, True),  # Pad intermediate to 576.
        (False, True, None, 512, 512, 16, False),
        (True, False, None, 512, 512, 16, False),
        (True, True, None, 528, 512, 16, False),  # Hidden cannot be padded.
        (True, True, 0, 512, 512, 16, False),  # Explicit values win.
        (True, True, 16, 512, 512, 16, False),
        (True, True, 128, 512, 512, 128, True),
        (False, True, 128, 512, 512, 128, False),
    ],
)
def test_nvfp4_default_block_and_explicit_override(
    monkeypatch,
    fused_enabled,
    use_ep,
    configured_block,
    hidden,
    inter,
    expected_block,
    armed,
    w4a8_enabled,
):
    from vllm_torchtpu.layers.adapter.quantization import nvfp4

    if not w4a8_enabled:
        expected_block = configured_block or 16
        armed = False
    layer = _nvfp4_checkpoint_layer(hidden, inter, use_ep)
    method = object.__new__(nvfp4.VllmNvfp4MoEMethod)
    method.moe = SimpleNamespace(has_bias=False, is_act_and_mul=True)
    method.group_size = 16
    requant_blocks = []
    prebuild_calls = []

    def load(w):
        e, n, half_k = w.shape
        return torch.empty((e, half_k * 2, n // 2), dtype=torch.float4_e2m1fn_x2)

    def requant(w, scales, block):
        requant_blocks.append(block)
        e, n, half_k = w.shape
        return load(w), torch.ones(e, half_k * 2 // block, 1, n)

    # Exercise real bridge admission with only distributed/op building stubbed.
    original_prebuild = bridge.prebuild_fused_moe_ep

    def prebuild(layer, **kwargs):
        assert layer.w13_weight.dtype == torch.uint8
        assert requant_blocks == []
        prebuild_calls.append(kwargs["rhs_qb"])
        with patch.object(bridge, "prebuild_fused_moe_ep", original_prebuild):
            return _prebuild(layer, **kwargs)

    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", fused_enabled)
    monkeypatch.setattr(envs, "MOE_FUSED_EP_ENABLE_W4A8", w4a8_enabled)
    monkeypatch.setattr(envs, "MOE_REQUANTIZE_BLOCK_SIZE", configured_block)
    monkeypatch.setattr(nvfp4, "RoutedExperts", torch.nn.Module)
    monkeypatch.setattr(nvfp4, "load_kmajor_fp4", load)
    monkeypatch.setattr(nvfp4, "requant_load_kmajor_fp4", requant)
    monkeypatch.setattr(nvfp4, "prebuild_fused_moe_kernel", lambda **kw: None)
    monkeypatch.setattr(
        nvfp4.moe_routing, "register_experts_start_buffer", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        nvfp4.moe_routing, "validate_linear_ep_placement", lambda *a: None
    )
    monkeypatch.setattr(bridge, "prebuild_fused_moe_ep", prebuild)
    method.process_weights_after_loading(layer)

    padded_inter = (inter + expected_block - 1) // expected_block * expected_block
    assert layer.w13_weight_scale.shape == (
        1,
        hidden // expected_block,
        1,
        2 * padded_inter,
    )
    assert layer.w2_weight_scale.shape == (1, padded_inter // expected_block, 1, hidden)
    assert bridge.fused_moe_ep_supported(method) == armed
    if (
        w4a8_enabled
        and fused_enabled
        and use_ep
        and (configured_block is None or configured_block > 0)
        and hidden % (configured_block or 64) == 0
    ):
        assert prebuild_calls == [configured_block or 64]
    else:
        assert prebuild_calls == []
    if configured_block or armed:
        if inter % expected_block == 0:
            assert requant_blocks == [expected_block, expected_block]
    else:
        assert requant_blocks == []


@pytest.mark.parametrize("configured_block", [None, 128])
@pytest.mark.parametrize(
    "refusal",
    [
        "mesh",
        "tokens",
        "smem",
        "routing",
        "experts",
        "activation",
        "simulation",
        "prepared_scales",
    ],
)
def test_nvfp4_admission_and_prepared_weight_validation(
    monkeypatch, refusal, configured_block
):
    from vllm_torchtpu.layers.adapter.quantization import nvfp4

    layer = _nvfp4_checkpoint_layer()
    originals = (layer.w13_weight, layer.w2_weight)
    original_values = tuple(w.clone() for w in originals)
    method = object.__new__(nvfp4.VllmNvfp4MoEMethod)
    method.moe = SimpleNamespace(has_bias=False, is_act_and_mul=True)
    method.group_size = 16
    admission = {}
    if refusal == "mesh":
        admission["has_mesh"] = False
    elif refusal == "tokens":
        admission["node_tokens"] = 1
    elif refusal == "smem":
        admission["smem_bytes"] = 1
    elif refusal == "routing":
        layer.hash_indices_table = torch.zeros(8, dtype=torch.int32)
    elif refusal == "experts":
        layer.global_num_experts = 3
    elif refusal == "activation":
        layer.activation = "gelu"
    elif refusal == "simulation":
        monkeypatch.setattr(nvfp4.moe_routing, "_SIMULATION_STRATEGY", object())

    loads = []
    requants = []
    events = []

    def load(w):
        loads.append(w.clone())
        e, n, half_k = w.shape
        return torch.empty((e, 2 * half_k, n // 2), dtype=torch.float4_e2m1fn_x2)

    def requant(w, scales, block):
        events.append("requant")
        requants.append(block)
        # No candidate may replace the checkpoint tensors before validation.
        assert layer.w13_weight is originals[0]
        assert layer.w2_weight is originals[1]
        e, n, half_k = w.shape
        weight = torch.empty((e, 2 * half_k, n // 2), dtype=torch.float4_e2m1fn_x2)
        scale = torch.ones(e, 2 * half_k // block, 1, n)
        if refusal == "prepared_scales":
            scale = scale[..., :-1]  # Force final validation to refuse.
        return weight, scale

    original_prebuild = bridge.prebuild_fused_moe_ep

    def prebuild(candidate, **kwargs):
        events.append("admission")
        assert requants == [] and loads == []
        assert candidate.w13_weight is originals[0]
        assert candidate.w2_weight is originals[1]
        with patch.object(bridge, "prebuild_fused_moe_ep", original_prebuild):
            return _prebuild(candidate, **admission, **kwargs)

    monkeypatch.setattr(envs, "MOE_REQUANTIZE_BLOCK_SIZE", configured_block)
    monkeypatch.setattr(nvfp4, "RoutedExperts", torch.nn.Module)
    monkeypatch.setattr(nvfp4, "load_kmajor_fp4", load)
    monkeypatch.setattr(nvfp4, "requant_load_kmajor_fp4", requant)
    monkeypatch.setattr(nvfp4, "prebuild_fused_moe_kernel", lambda **kw: None)
    monkeypatch.setattr(
        nvfp4.moe_routing, "register_experts_start_buffer", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        nvfp4.moe_routing, "validate_linear_ep_placement", lambda *a: None
    )
    monkeypatch.setattr(bridge, "prebuild_fused_moe_ep", prebuild)
    if refusal == "prepared_scales":
        with pytest.raises(
            AssertionError,
            match="prepared weights violate the admitted layout: w1_scale",
        ):
            method.process_weights_after_loading(layer)
        assert events == ["admission", "requant", "requant"]
        assert requants == [configured_block or 64] * 2
        assert loads == []  # No second preparation or GMM fallback.
        assert layer.w13_weight is originals[0]
        assert layer.w2_weight is originals[1]
        for original, value in zip(originals, original_values):
            torch.testing.assert_close(original, value)
        assert not bridge.fused_moe_ep_supported(method)
        return
    method.process_weights_after_loading(layer)

    assert events[0] == "admission"
    assert not bridge.fused_moe_ep_supported(method)
    assert not hasattr(layer, "_tpu_fused_w13_scale")
    expected_block = configured_block or 16
    assert layer.w13_weight_scale.shape == (1, 512 // expected_block, 1, 1024)
    assert layer.w2_weight_scale.shape == (1, 512 // expected_block, 1, 512)
    expected_requants = [configured_block] * 2 if configured_block else []
    assert requants == expected_requants
    for original, value in zip(originals, original_values):
        torch.testing.assert_close(original, value)
    if configured_block is None:
        assert len(loads) == 2
        for loaded, original in zip(loads, original_values):
            torch.testing.assert_close(loaded, original)
        torch.testing.assert_close(
            layer.w13_weight_scale[..., :512], torch.full((1, 32, 1, 512), 2.0)
        )
        torch.testing.assert_close(
            layer.w13_weight_scale[..., 512:], torch.full((1, 32, 1, 512), 3.0)
        )
        torch.testing.assert_close(
            layer.w2_weight_scale, torch.full((1, 32, 1, 512), 4.0)
        )


@pytest.mark.parametrize(
    "field",
    [
        "custom_routing_function",
        "scoring_func",
        "e_score_correction_bias",
        "use_grouped_topk",
        "routed_scaling_factor",
    ],
)
def test_missing_routing_field_cannot_arm_fused_ep(field):
    layer = _layer()
    delattr(layer, field)
    with pytest.raises(AttributeError, match=field):
        _prebuild(layer)


def test_pcp_size_reads_the_layer_parallel_config():
    layer = _layer()
    assert bridge._pcp_size(layer) == 1
    layer.moe_config.moe_parallel_config.pcp_size = 4
    assert bridge._pcp_size(layer) == 4


def test_fused_scale_buffers_are_fresh_bases_aliased_by_the_parameters():
    # A view graph input costs a materializing program per step on torch_tpu,
    # so the kernel's tables must be base tensors, bit-identical to the
    # squeezed loader tables, and the 4-D parameters must not hold a 2nd copy.
    layer = torch.nn.Module()
    w13 = torch.rand(3, 4, 1, 256)
    w2 = torch.rand(3, 2, 1, 128)
    layer.w13_weight_scale = torch.nn.Parameter(w13.clone(), requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(w2.clone(), requires_grad=False)
    loaded = {
        p: getattr(layer, p).data_ptr() for p in ("w13_weight_scale", "w2_weight_scale")
    }
    bridge.register_fused_scale_buffers(layer)
    for param, buf, ref in (
        ("w13_weight_scale", "_tpu_fused_w13_scale", w13),
        ("w2_weight_scale", "_tpu_fused_w2_scale", w2),
    ):
        table, weight = getattr(layer, buf), getattr(layer, param)
        assert table._base is None and table.is_contiguous()
        assert table.data_ptr() != loaded[param]
        assert torch.equal(table.view(torch.int32), ref.squeeze(2).view(torch.int32))
        assert buf in dict(layer.named_buffers())
        assert isinstance(weight, torch.nn.Parameter) and not weight.requires_grad
        assert weight.shape == ref.shape and weight.data_ptr() == table.data_ptr()


def test_shared_fold_op_is_cached_apart_and_forwards_the_rows(monkeypatch):
    from unittest.mock import Mock

    from vllm_torchtpu.kernels.fused_moe import v2

    monkeypatch.setattr(bridge, "_OPS", {})
    built = []

    def build(name, fn, **kwargs):
        built.append((fn, kwargs))
        return Mock()

    monkeypatch.setattr(bridge, "jax_op", build)
    kernel = Mock(side_effect=lambda x, *args, **kwargs: x)
    monkeypatch.setattr(v2, "fused_ep_moe_v2", kernel)
    args = (None, 2, True, "silu", None, False)
    plain = bridge._build_op(*args)
    folded = bridge._build_op(*args, has_shared_partial=True)
    assert folded is not plain
    assert bridge._build_op(*args, has_shared_partial=True) is folded
    (plain_fn, plain_kw), (fold_fn, fold_kw) = built
    plain_specs = plain_kw["input_partition_specs"]
    assert fold_kw["input_partition_specs"] == plain_specs + plain_specs[:1]
    x = jnp.ones((8, 8), dtype=jnp.bfloat16)
    shared = jnp.full((8, 8), 2.0, dtype=jnp.bfloat16)
    operands = (
        x,
        None,
        None,
        None,
        None,
        jnp.zeros((8, 4)),
        jnp.zeros((1, 1), dtype=jnp.int32),
    )
    assert fold_fn(*operands, shared) is x
    assert kernel.call_args.kwargs["shared_partial"] is shared
    plain_fn(*operands)
    assert "shared_partial" not in kernel.call_args.kwargs


@pytest.mark.parametrize(
    "enabled,overrides,staged,refusal",
    [
        pytest.param(False, {}, None, None, id="env-off"),
        pytest.param(True, {}, "fold-op", None, id="plain"),
        pytest.param(
            True,
            dict(e_score_correction_bias=torch.zeros(4 * _STUB_EP)),
            None,
            "the op takes a score bias",
            id="score-bias",
        ),
    ],
)
def test_prebuild_stages_the_shared_fold_op(enabled, overrides, staged, refusal):
    layer = _layer(**overrides)

    def build(*_args, has_shared_partial=False, **_kwargs):
        return "fold-op" if has_shared_partial else "op"

    with patch.object(envs, "MOE_FUSED_EP_V2_FOLD_SHARED_EXPERT", enabled):
        assert _prebuild(layer, build_op=build) == "op"
    assert getattr(layer, bridge._SHARED_FOLD_OP_ATTR, None) == staged
    assert getattr(layer, bridge._SHARED_FOLD_REFUSAL_ATTR, None) == refusal


@pytest.mark.parametrize(
    "has_score_bias,has_hash_ids,sorted_gather,reason",
    [
        (False, False, None, None),
        (True, False, None, "score bias"),
        (False, True, None, "hash ids"),
        (False, False, ("pg", (0, 1)), "MOE_FUSED_EP_SORTED_GATHER"),
    ],
)
def test_shared_partial_refusal(has_score_bias, has_hash_ids, sorted_gather, reason):
    got = bridge._shared_partial_refusal(has_score_bias, has_hash_ids, sorted_gather)
    if reason is None:
        assert got is None
    else:
        assert reason in got


def test_forward_hands_a_folded_shared_expert_its_rows():
    seen = []
    owner = SimpleNamespace()
    setattr(
        owner, bridge.FUSED_MOE_EP_OP_ATTR, lambda *args: seen.append(args) or args[0]
    )
    setattr(owner, bridge.FUSED_MOE_EP_SHARED_ATTR, lambda x: x + 1)
    hidden = torch.zeros(1, 8)
    weights = torch.empty(0)
    scales = torch.ones(1, 1)
    bridge.fused_moe_ep(owner, hidden, weights, weights, scales, scales, hidden)
    assert len(seen[0]) == 8
    assert torch.equal(seen[0][-1], hidden + 1)


def _fold_runner():
    """A runner `fold_shared_experts` admits, shaped as vLLM builds Qwen's."""
    owner = SimpleNamespace(moe_kernel=None)
    setattr(owner, bridge.FUSED_MOE_EP_OP_ATTR, "op")
    routed = torch.nn.Module()
    routed.quant_method = owner
    setattr(routed, bridge._SHARED_FOLD_OP_ATTR, "fold-op")
    shared = torch.nn.Module()
    shared._layer = torch.nn.Linear(8, 8)
    runner = VllmMoERunner.__new__(VllmMoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_experts = routed
    runner._shared_experts = shared
    runner.shared_expert_gate = None
    runner.routed_input_transform = None
    runner.routed_output_transform = None
    runner.routed_scaling_factor = 1.0
    runner.moe_config = SimpleNamespace(
        is_sequence_parallel=False, skip_final_all_reduce=False
    )
    return runner


def test_fold_hands_the_shared_expert_to_the_op():
    folded, refused = _fold_runner(), _fold_runner()
    refused.routed_scaling_factor = 2.5
    model, draft = torch.nn.Module(), torch.nn.Module()
    model.folded, model.refused = folded, refused
    draft.folded = folded
    owner, mlp = folded._quant_method, folded.shared_experts._layer
    assert bridge.fold_shared_experts(model, draft) == 1
    assert getattr(owner, bridge.FUSED_MOE_EP_OP_ATTR) == "fold-op"
    assert getattr(owner, bridge.FUSED_MOE_EP_SHARED_ATTR) is mlp
    assert folded.shared_experts is None
    assert folded._forward_entry is upstream._moe_forward
    assert any(p is mlp.weight for p in model.parameters())
    assert getattr(refused._quant_method, bridge.FUSED_MOE_EP_OP_ATTR) == "op"
    assert refused.shared_experts is not None


@pytest.mark.parametrize(
    "change,reason",
    [
        (
            lambda r: setattr(r._quant_method, bridge.FUSED_MOE_EP_OP_ATTR, None),
            "not armed",
        ),
        (lambda r: setattr(r, "_shared_experts", None), "no shared expert"),
        (
            lambda r: delattr(r.routed_experts, bridge._SHARED_FOLD_OP_ATTR),
            "staged no fold op",
        ),
        (
            lambda r: (
                setattr(r.routed_experts, bridge._SHARED_FOLD_OP_ATTR, None),
                setattr(
                    r.routed_experts,
                    bridge._SHARED_FOLD_REFUSAL_ATTR,
                    "MOE_FUSED_EP_SORTED_GATHER splits the op",
                ),
            ),
            "MOE_FUSED_EP_SORTED_GATHER",
        ),
        (lambda r: setattr(r, "shared_expert_gate", torch.nn.Linear(8, 1)), "gate"),
        (
            lambda r: setattr(r, "routed_input_transform", torch.nn.Identity()),
            "input transform",
        ),
        (
            lambda r: setattr(r, "routed_output_transform", torch.nn.Identity()),
            "output transform",
        ),
        (lambda r: setattr(r, "routed_scaling_factor", 2.5), "routed_scaling"),
        (
            lambda r: setattr(r.moe_config, "is_sequence_parallel", True),
            "sequence-parallel",
        ),
        (
            lambda r: setattr(r.moe_config, "skip_final_all_reduce", True),
            "skip_final_all_reduce",
        ),
    ],
)
def test_shared_fold_refusals(change, reason):
    runner = _fold_runner()
    assert bridge._shared_fold_refusal(runner) is None
    change(runner)
    assert reason in bridge._shared_fold_refusal(runner)


class _CountingSharedMLP(torch.nn.Module):
    """A TP-partial shared expert that counts its calls."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x * self.scale


class _ArmedRoutedExperts(torch.nn.Module):
    """The RoutedExperts state an armed fp8 layer's forward reads."""

    forward_monolithic = RoutedExperts.forward_monolithic

    def __init__(self, quant_method):
        super().__init__()
        self.quant_method = quant_method
        layer = _layer()
        for name in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale_inv",
            "w2_weight_scale_inv",
        ):
            param = torch.nn.Parameter(getattr(layer, name), requires_grad=False)
            self.register_parameter(name, param)

    def _ensure_moe_quant_config_init(self):
        pass


def test_fold_survives_upstream_moe_runner_forward(monkeypatch):
    """The fold rewrites MoERunner internals, so check it through upstream's
    own __init__ and forward: once folded, the shared expert runs exactly once,
    inside the op, and the runner neither adds nor all-reduces it again."""
    name = "model.layers.0.mlp.experts"
    layers, all_reduces, fold_calls = {}, [], []

    def all_reduce(t):
        all_reduces.append(t)
        return t

    monkeypatch.setattr(envs, "TPU_MOE_SKIP_PADDED_TOKENS", False)
    monkeypatch.setattr(upstream, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(
        upstream, "register_layer_for_moe_forward_op", lambda config, layer: None
    )
    monkeypatch.setattr(upstream, "get_layer_from_name", layers.__getitem__)
    monkeypatch.setattr(upstream, "is_forward_context_available", lambda: False)
    monkeypatch.setattr(
        upstream,
        "get_forward_context",
        lambda: SimpleNamespace(dp_metadata=None, all_moe_layers=None),
    )
    monkeypatch.setattr(upstream, "tensor_model_parallel_all_reduce", all_reduce)

    parallel = SimpleNamespace(
        use_ep=True,
        ep_size=_STUB_EP,
        tp_size=1,
        dp_size=1,
        pcp_size=1,
        enable_eplb=False,
        all2all_backend="allgather_reducescatter",
        use_fi_nvl_two_sided_kernels=False,
        use_all2all_kernels=False,
    )
    moe_config = SimpleNamespace(
        moe_parallel_config=parallel,
        is_sequence_parallel=False,
        skip_final_all_reduce=False,
        tp_size=1,
        ep_size=_STUB_EP,
        dp_size=1,
        pcp_size=1,
        sp_size=1,
        hidden_dim=8,
        hidden_dim_unpadded=8,
    )
    quant_method = fp8.VllmFp8MoEMethodTPU.__new__(fp8.VllmFp8MoEMethodTPU)
    FusedMoEMethodBase.__init__(quant_method, moe_config)
    quant_method._tpu_activation_str = "silu"
    routed = _ArmedRoutedExperts(quant_method)
    # What prebuild stages: the plain op, and the op that also takes the
    # shared expert's TP-partial rows and adds their TP sum.
    setattr(quant_method, bridge.FUSED_MOE_EP_OP_ATTR, lambda x, *_: x * 0.25)

    def fold_op(x, *operands):
        fold_calls.append(operands)
        return x * 0.25 + operands[-1]

    setattr(routed, bridge._SHARED_FOLD_OP_ATTR, fold_op)
    shared = _CountingSharedMLP()
    runner = MoERunner(
        layer_name=name,
        moe_config=moe_config,
        router=SimpleNamespace(prepare_monolithic_logits=lambda logits: logits),
        routed_experts=routed,
        shared_experts=shared,
    )
    layers[name] = runner
    model = torch.nn.Module()
    model.experts = runner
    x = torch.randn(4, 8)
    logits = torch.randn(4, 4 * _STUB_EP)
    want = x * 0.25 + x * 0.5

    # Unfolded, the runner runs the shared expert and all-reduces it.
    assert torch.equal(runner(hidden_states=x, router_logits=logits), want)
    assert shared.calls == 1 and len(all_reduces) == 1

    assert bridge.fold_shared_experts(model) == 1
    out = runner(hidden_states=x, router_logits=logits)

    # Had the runner kept it, the shared expert would run and be added twice.
    assert torch.equal(out, want)
    assert shared.calls == 2 and len(all_reduces) == 1
    assert len(fold_calls) == 1 and torch.equal(fold_calls[0][-1], x * 0.5)

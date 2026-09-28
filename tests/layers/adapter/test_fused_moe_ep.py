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

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.fused_moe.v2.host import token_gather_smem_bytes
from vllm_torchtpu.layers.adapter import fused_moe_ep as bridge

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
    ):
        assert bridge._build_op(*args, **(options | changed)) is not replicated
    assert len(closures) == 5
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


def test_mesh_expert_order_is_none_when_the_mesh_is_rank_ordered():
    """The permutation exists only to undo a mesh ordered by device id.

    Since google-pytorch/torch_tpu#3522 `build_ep_mesh` orders by rank, so mesh
    index i holds EP rank i's experts and the op must be handed nothing.
    """
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


def test_ep_rank_order_is_the_identity():
    """It no longer depends on how the group's ranks are arranged."""
    from vllm_torchtpu.distributed import ep_mesh

    group = SimpleNamespace(ranks=[3, 1, 2, 0])
    with patch.object(ep_mesh, "get_ep_group", return_value=group):
        assert ep_mesh.ep_rank_order() == (0, 1, 2, 3)


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


def test_refused_on_a_non_silu_activation():
    assert _prebuild(_layer(), activation="gelu") is None


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


def test_refused_while_the_routing_simulator_is_on():
    from vllm_torchtpu.layers.adapter import moe_routing

    with patch.object(moe_routing, "_SIMULATION_STRATEGY", object()):
        assert _prebuild(_layer()) is None


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


def test_ep_rank_order_does_not_depend_on_where_the_devices_sit():
    """It used to undo a mesh ordered by device id.

    google-pytorch/torch_tpu#3522 binds partition p to rank p and
    `build_ep_mesh` orders by rank to match, so mesh index i holds EP rank i's
    experts whatever the chips underneath are doing.
    """
    from vllm_torchtpu.distributed import ep_mesh

    group = SimpleNamespace(ranks=list(range(8)), world_size=8)
    orders = []
    for device_ids in ((0, 1, 4, 5, 6, 7, 2, 3), (0, 1, 2, 3, 4, 5, 6, 7)):
        with (
            patch.object(ep_mesh, "get_ep_group", return_value=group),
            patch.object(ep_mesh, "ep_device_ids", return_value=device_ids),
        ):
            orders.append(ep_mesh.ep_rank_order())

    assert orders[0] == orders[1] == tuple(range(8))


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


class TestFusedOutputIsReducedPatch:
    """`_patch_moe_runner_fused_output_is_reduced` rewrites an upstream class.

    It is the guard that stops vLLM all-reducing an output the kernel already
    combined, so it has to be right about three things: it stays out of runs
    that do not use the kernel, it defers to upstream under
    `skip_final_all_reduce`, and it answers per layer.
    """

    @staticmethod
    def _fresh_runner_cls():
        """A MoERunner with the patch un-applied, restored after the test."""
        from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr

        return mr.MoERunner

    @pytest.fixture(autouse=True)
    def _restore(self):
        cls = self._fresh_runner_cls()
        saved = cls.__dict__.get("_fused_output_is_reduced")
        saved_flag = cls.__dict__.get("_tpu_fused_output_reduced_patch")
        if hasattr(cls, "_tpu_fused_output_reduced_patch"):
            del cls._tpu_fused_output_reduced_patch
        yield cls
        if saved is not None:
            cls._fused_output_is_reduced = saved
        if saved_flag is not None:
            cls._tpu_fused_output_reduced_patch = saved_flag
        elif hasattr(cls, "_tpu_fused_output_reduced_patch"):
            del cls._tpu_fused_output_reduced_patch

    def _apply(self):
        import vllm_torchtpu

        vllm_torchtpu._patch_moe_runner_fused_output_is_reduced()

    def _runner(self, cls, *, armed, skip_final_all_reduce=False):
        """A stub standing in for a MoERunner, with `original` forced False."""
        # `moe_kernel=None` is what the TPU fp8 method actually has, and it
        # is what makes upstream's own getter return False -- so a True here
        # can only have come from the patch.
        quant_method = SimpleNamespace(moe_kernel=None)
        if armed:
            setattr(quant_method, bridge.FUSED_MOE_EP_OP_ATTR, "op")
        runner = SimpleNamespace(
            _quant_method=quant_method,
            moe_config=SimpleNamespace(skip_final_all_reduce=skip_final_all_reduce),
        )
        return cls._fused_output_is_reduced.fget(runner)

    def test_not_installed_without_the_env_flag(self, _restore):
        cls = _restore
        before = cls.__dict__.get("_fused_output_is_reduced")
        with patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", False):
            self._apply()
        assert cls.__dict__.get("_fused_output_is_reduced") is before
        assert not getattr(cls, "_tpu_fused_output_reduced_patch", False)

    def test_reports_reduced_only_for_an_armed_layer(self, _restore):
        cls = _restore
        with patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", True):
            self._apply()
        assert cls._tpu_fused_output_reduced_patch is True
        assert self._runner(cls, armed=True) is True
        assert self._runner(cls, armed=False) is False

    def test_defers_to_upstream_under_skip_final_all_reduce(self, _restore):
        """The upstream assert requires an un-reduced fused output there."""
        cls = _restore
        with patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", True):
            self._apply()
        assert self._runner(cls, armed=True, skip_final_all_reduce=True) is False

    def test_is_idempotent(self, _restore):
        cls = _restore
        with patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", True):
            self._apply()
            first = cls.__dict__["_fused_output_is_reduced"]
            self._apply()
        assert cls.__dict__["_fused_output_is_reduced"] is first


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
            lambda *args: (
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

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
from vllm_torchtpu.layers.vllm import fused_moe_ep as bridge

# The width of the mesh `_prebuild` stubs in. `_layer` sizes the global expert
# count against it, because the kernel requires one equal block per shard.
_STUB_EP = 2


def _layer(experts=4, hidden=8, inter=16, **overrides):
    """A layer whose weights and routing the kernel would accept."""
    layer = SimpleNamespace(
        w13_weight=torch.zeros(experts,
                               hidden,
                               2 * inter,
                               dtype=torch.float8_e4m3fn),
        w2_weight=torch.zeros(experts,
                              inter,
                              hidden,
                              dtype=torch.float8_e4m3fn),
        w13_weight_scale_inv=torch.ones(experts, 1, 1, 2 * inter),
        w2_weight_scale_inv=torch.ones(experts, 1, 1, hidden),
        global_num_experts=experts * _STUB_EP,
        moe_config=SimpleNamespace(
            experts_per_token=2,
            moe_parallel_config=SimpleNamespace(use_ep=True, pcp_size=1),
        ),
    )
    for name, value in overrides.items():
        setattr(layer, name, value)
    return layer


def _prebuild(layer,
              *,
              has_mesh=True,
              tp=1,
              node_tokens=4096,
              smem_bytes=1024 * 1024,
              ep_order=None,
              build_op=None,
              **kwargs):
    """Call prebuild with the distributed lookups stubbed out.

    `build_ep_mesh` does an `all_gather_object` over the EP group, so it is
    replaced by a two-way mesh stub. Everything under test decides before the
    op would be built, and `_build_op` is stubbed so the one accepting case
    does not need a device.
    """
    mesh = (SimpleNamespace(
        shape={bridge.EP_AXIS_NAME: _STUB_EP}) if has_mesh else None)
    build_op = build_op or (lambda *_args: "op")
    with patch.object(bridge, "build_ep_mesh", return_value=mesh), \
            patch.object(bridge, "ep_mesh_index", return_value=0), \
            patch.object(bridge, "ep_rank_order", return_value=ep_order), \
            patch.object(bridge, "_max_node_tokens", return_value=node_tokens), \
            patch.object(bridge, "_smem_capacity_bytes",
                         return_value=smem_bytes), \
            patch.object(bridge, "_tensor_parallel_size", return_value=tp), \
            patch.object(bridge, "_build_op", side_effect=build_op):
        defaults = dict(topk=2, renormalize=True, activation="silu")
        defaults.update(kwargs)
        return bridge.prebuild_fused_moe_ep(layer, **defaults)


@pytest.fixture(autouse=True)
def _armed():
    """Every test here runs as if the operator asked for the kernel."""
    with patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", True), \
            patch.object(envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS", 1024), \
            patch.object(envs, "MOE_FUSED_EP_V2_SHARDED_PLAN", False):
        yield


def test_arms_on_a_supported_layer():
    assert _prebuild(_layer()) == "op"


def test_sharded_plan_flag_reaches_the_built_op():
    seen = []
    with patch.object(envs, "MOE_FUSED_EP_V2_SHARDED_PLAN", True):
        assert _prebuild(
            _layer(), build_op=lambda *args: seen.append(args) or "op") == "op"
    assert seen and seen[0][-1] is True


def test_mesh_expert_order_is_closed_into_the_built_op():
    order = (1, 0)
    seen = []
    assert _prebuild(_layer(),
                     ep_order=order,
                     build_op=lambda *args: seen.append(args) or "op") == "op"
    assert seen and seen[0][-2] == order


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


def test_refused_under_prefill_context_parallelism():
    layer = _layer()
    layer.moe_config.moe_parallel_config.pcp_size = 4
    assert _prebuild(layer) is None


def test_refused_under_tensor_parallelism():
    assert _prebuild(_layer(), tp=8) is None


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
    assert gather_bytes == 5120
    needed = bridge._SMEM_OVERHEAD_BYTES + gather_bytes
    assert _prebuild(_layer(), smem_bytes=needed) == "op"
    assert _prebuild(_layer(), smem_bytes=needed - 1) is None


@pytest.mark.parametrize("weights", [
    dict(w13_weight=torch.zeros(4, 8, 32, dtype=torch.bfloat16),
         w2_weight=torch.zeros(4, 16, 8, dtype=torch.bfloat16)),
    dict(w2_weight=torch.zeros(4, 999, 8, dtype=torch.float8_e4m3fn)),
    dict(w13_weight_scale_inv=torch.ones(4, 3, 1, 32)),
    dict(w2_weight_scale_inv=torch.ones(4, 3, 1, 8)),
])
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


@pytest.mark.parametrize("routing", [
    dict(custom_routing_function=lambda **_: None),
    dict(scoring_func="sigmoid"),
    dict(e_score_correction_bias=torch.zeros(8)),
    dict(use_grouped_topk=True, num_expert_group=8, topk_group=2),
    dict(hash_indices_table=torch.zeros(8, dtype=torch.int32)),
    dict(routed_scaling_factor=2.5),
])
def test_refused_on_routing_the_kernel_does_not_implement(routing):
    """The fused path never calls `moe_routing.route`: the kernel scores with
    a plain softmax and selects top-k itself. Anything else routes tokens to a
    different set of experts, so it is refused rather than run."""
    assert _prebuild(_layer(**routing)) is None


def test_refused_while_the_routing_simulator_is_on():
    from vllm_torchtpu.layers.vllm import moe_routing
    with patch.object(moe_routing, "_SIMULATION_STRATEGY", object()):
        assert _prebuild(_layer()) is None


def test_arming_is_per_layer_not_per_process():
    """One armed layer must not answer for a layer that was refused."""
    armed = SimpleNamespace()
    refused = SimpleNamespace()
    setattr(armed, bridge.FUSED_MOE_EP_OP_ATTR, _prebuild(_layer()))
    setattr(refused, bridge.FUSED_MOE_EP_OP_ATTR,
            _prebuild(_layer(w2_bias=torch.zeros(4, 8))))

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
    from vllm_torchtpu.kernels.fused_moe.v2.layer import \
        _relabel_expert_ids_to_mesh_order

    ep, per_shard = 4, 2
    order = (0, 2, 3, 1)
    selected = jnp.arange(ep * per_shard, dtype=jnp.int32).reshape(2, 4)
    got = np.asarray(
        _relabel_expert_ids_to_mesh_order(selected,
                                          g_local=per_shard,
                                          mesh_ep_ranks=order))
    mesh_index_of_rank = {ep_rank: i for i, ep_rank in enumerate(order)}
    want = np.asarray([[
        mesh_index_of_rank[int(e) // per_shard] * per_shard +
        int(e) % per_shard for e in row
    ] for row in np.asarray(selected)], np.int32)
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
    assert bridge.fused_moe_ep(owner, hidden, weights, weights, scales, scales,
                               logits) is hidden
    assert seen[0][5] is logits


@pytest.mark.parametrize("order,expected", [
    ((0, 1, 2, 3), None),
    ((0, 2, 3, 1), (0, 2, 3, 1)),
])
def test_mesh_expert_order(order, expected):
    with patch.object(bridge, "ep_rank_order", return_value=order):
        assert bridge._mesh_expert_order(len(order)) == expected


def test_ep_rank_order_maps_ranks_to_ascending_device_ids():
    """An EP-rank-ordered mesh deadlocks on the first remote DMA; the mesh is
    in ascending device-id order, so this is what says which rank sits where."""
    from vllm_torchtpu.distributed import ep_mesh

    # EP ranks 0..7 sitting on these device ids. Sorted ascending, rank r ends
    # up at the position of its device id.
    device_ids = (0, 1, 4, 5, 6, 7, 2, 3)
    group = SimpleNamespace(ranks=list(range(8)), world_size=8)
    with patch.object(ep_mesh, "get_ep_group", return_value=group), \
            patch.object(ep_mesh, "ep_device_ids", return_value=device_ids):
        order = ep_mesh.ep_rank_order()

    assert order == tuple(rank
                          for _, rank in sorted(zip(device_ids, range(8))))
    assert order == (0, 1, 6, 7, 2, 3, 4, 5)


def test_sharded_jax_op_still_matches_the_line_it_replaces():
    """`sharded_jax_op` finds its one line by string-matching torch_tpu's
    installed source. On a miss it falls back to stock with only a warning,
    and the fused op then sizes its outputs mesh-wide -- a device-time shape
    error, long after load. Fail here instead, on a torch_tpu bump."""
    import inspect

    from torch_tpu._internal.pallas import pallas as pallas_impl

    from vllm_torchtpu.distributed import sharded_jax_op as sjo

    source = inspect.getsource(pallas_impl.JaxCallable.__call__)
    assert sjo._STOCK_PLACEHOLDER_LINE in source
    assert sjo._build_sharded_callable_cls() is not pallas_impl.JaxCallable


def test_supports_internal_mk_tracks_both_owners():
    """`supports_internal_mk` claims vLLM's dispatch/combine. Either owner --
    chunk pipelining or the fused kernel -- has to claim it alone, and the
    fused half must follow what prebuild ARMED, not what the operator asked
    for: on a refusal vLLM's collectives have to stay."""
    from vllm_torchtpu.layers.vllm.quantization.fp8 import VllmFp8MoEMethodTPU

    prop = VllmFp8MoEMethodTPU.supports_internal_mk
    armed = SimpleNamespace()
    setattr(armed, bridge.FUSED_MOE_EP_OP_ATTR, "op")

    with patch(
            "vllm_torchtpu.layers.vllm.quantization.fp8."
            "enable_pipelined_collective_and_compute",
            return_value=False):
        assert prop.fget(SimpleNamespace()) is False
        assert prop.fget(armed) is True

    with patch(
            "vllm_torchtpu.layers.vllm.quantization.fp8."
            "enable_pipelined_collective_and_compute",
            return_value=True):
        assert prop.fget(SimpleNamespace()) is True


def test_env_flag_alone_does_not_claim_the_collectives():
    """The refusals only hold if the request does not claim ownership."""
    from vllm_torchtpu.layers.vllm.quantization.fp8 import VllmFp8MoEMethodTPU

    prop = VllmFp8MoEMethodTPU.supports_internal_mk
    with patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", True), \
            patch("vllm_torchtpu.layers.vllm.quantization.fp8."
                  "enable_pipelined_collective_and_compute",
                  return_value=False):
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
        from vllm.model_executor.layers.fused_moe.runner import \
            moe_runner as mr
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
            moe_config=SimpleNamespace(
                skip_final_all_reduce=skip_final_all_reduce),
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
        assert self._runner(cls, armed=True,
                            skip_final_all_reduce=True) is False

    def test_is_idempotent(self, _restore):
        cls = _restore
        with patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", True):
            self._apply()
            first = cls.__dict__["_fused_output_is_reduced"]
            self._apply()
        assert cls.__dict__["_fused_output_is_reduced"] is first

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
"""Strict admission for the future experimental serving path, without a TPU."""

import dataclasses
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import vllm_adapter as bridge


def layer(fp4=False):
    e, h, i = 4, 512, 256
    dtype = torch.float4_e2m1fn_x2 if fp4 else torch.float8_e4m3fn
    result = SimpleNamespace(
        w13_weight=torch.empty((e, h, i if fp4 else 2 * i), dtype=dtype, device="meta"),
        w2_weight=torch.empty((e, i, h // 2 if fp4 else h), dtype=dtype, device="meta"),
        global_num_experts=8,
        custom_routing_function=None,
        scoring_func="softmax",
        e_score_correction_bias=None,
        use_grouped_topk=False,
        num_expert_group=None,
        topk_group=None,
        routed_scaling_factor=1.0,
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(
                use_ep=True, pcp_size=1, is_sequence_parallel=False
            )
        ),
    )
    if fp4:
        result.w13_weight_scale = torch.empty((e, h // 64, 1, 2 * i), device="meta")
        result.w2_weight_scale = torch.empty((e, i // 64, 1, h), device="meta")
    else:
        result.w13_weight_scale_inv = torch.empty((e, 1, 1, 2 * i), device="meta")
        result.w2_weight_scale_inv = torch.empty((e, 1, 1, h), device="meta")
    return result


@pytest.fixture
def context(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(
        bridge, "build_ep_mesh", lambda: SimpleNamespace(shape={bridge.EP_AXIS_NAME: 2})
    )
    monkeypatch.setattr(bridge, "ep_mesh_index", lambda: 0)
    monkeypatch.setattr(bridge, "ep_rank_order", lambda: None)
    monkeypatch.setattr(bridge, "ep_token_replica_groups", lambda **kw: ((0, 1),))
    monkeypatch.setattr(bridge, "_max_node_tokens", lambda *_: 4096)
    monkeypatch.setattr(bridge, "_device_info", lambda: info)
    monkeypatch.setattr(bridge.envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS", 1024)
    op = Mock(return_value=object())
    monkeypatch.setattr(bridge, "_build_op", op)
    return info, op


def prebuild(value, **kwargs):
    return bridge.prebuild_adaptive_fused_moe(
        value, topk=2, renormalize=True, activation="silu", **kwargs
    )


@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("old_flags", ["0", "1"])
def test_dtype_selects_format_without_legacy_gates(
    context, monkeypatch, fp4, old_flags
):
    for name in ("MOE_FUSED_EP_ENABLE_W4A8", "MOE_FUSED_EP_V2_SHARDED_PLAN"):
        monkeypatch.setenv(name, old_flags)
    _, op = context
    assert prebuild(layer(fp4)) is op.return_value
    assert op.call_args.args[-2:] == (("fp4", 64) if fp4 else ("fp8", None))


@pytest.mark.parametrize(
    "case,match",
    [
        ("no_ep", "expert-parallel"),
        ("no_mesh", "EP group"),
        ("tokens", "token count"),
        ("smem", "SMEM"),
        ("vmem", "VMEM"),
        ("uneven", "equal contiguous"),
        ("dtype", "weight dtype"),
        ("shape", "w2"),
        ("scales", "per-output-channel"),
        ("custom", "custom_routing_function"),
        ("scoring", "scoring_func"),
        ("bias", "biases"),
        ("correction", "correction_bias"),
        ("grouped", "grouped top-k"),
        ("hash", "hash routing"),
        ("scaling", "routed_scaling_factor"),
        ("simulation", "SIMULATION"),
    ],
)
def test_former_fallbacks_raise(context, monkeypatch, case, match):
    info, op = context
    value = layer()
    if case == "no_ep":
        value.moe_config.moe_parallel_config.use_ep = False
    elif case == "no_mesh":
        monkeypatch.setattr(bridge, "build_ep_mesh", lambda: None)
    elif case == "tokens":
        monkeypatch.setattr(bridge, "_max_node_tokens", lambda *_: 512)
    elif case in ("smem", "vmem"):
        field = case + "_capacity_bytes"
        monkeypatch.setattr(
            bridge, "_device_info", lambda: dataclasses.replace(info, **{field: 1})
        )
    elif case == "uneven":
        value.global_num_experts = 9
    elif case == "dtype":
        value.w13_weight = torch.empty(value.w13_weight.shape, device="meta")
    elif case == "shape":
        value.w2_weight = torch.empty(
            (4, 256, 256), dtype=torch.float8_e4m3fn, device="meta"
        )
    elif case == "scales":
        value.w13_weight_scale_inv = torch.empty((4, 2, 1, 512), device="meta")
    elif case == "custom":
        value.custom_routing_function = object()
    elif case == "scoring":
        value.scoring_func = "sigmoid"
    elif case == "bias":
        value.w13_bias = object()
    elif case == "correction":
        value.e_score_correction_bias = object()
    elif case == "grouped":
        value.use_grouped_topk, value.num_expert_group, value.topk_group = True, 4, 2
    elif case == "hash":
        value.hash_indices_table = object()
    elif case == "scaling":
        value.routed_scaling_factor = 2.0
    elif case == "simulation":
        from vllm_torchtpu.layers.adapter import moe_routing

        monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", "random")
    with pytest.raises(ValueError, match=match):
        prebuild(value)
    op.assert_not_called()


def test_fp4_invalid_block_is_an_error(context):
    with pytest.raises(ValueError, match="packed-weight row tile"):
        prebuild(layer(True), rhs_qb=16)


def test_forward_without_prebuilt_op_raises():
    with pytest.raises(RuntimeError, match="not prebuilt"):
        bridge.run_adaptive_fused_moe(SimpleNamespace(), *([None] * 6))


def test_fp4_forward_preserves_single_contraction_block():
    value = layer(True)
    op = Mock()
    owner = SimpleNamespace(**{bridge.FUSED_MOE_EP_OP_ATTR: op})
    s1 = torch.empty((4, 1, 1, 512), device="meta")
    s2 = torch.empty((4, 1, 1, 512), device="meta")
    bridge.run_adaptive_fused_moe(
        owner, None, value.w13_weight, value.w2_weight, s1, s2, None
    )
    assert op.call_args.args[3].shape == (4, 1, 512)
    assert op.call_args.args[4].shape == (4, 1, 512)


def test_built_bridge_calls_experimental_kernel_and_separates_formats(monkeypatch):
    import importlib

    package = importlib.import_module(
        "vllm_torchtpu.kernels.experimental.adaptive_fused_moe"
    )
    kernel = Mock(return_value="result")
    monkeypatch.setattr(package, "adaptive_fused_moe", kernel)
    monkeypatch.setattr(bridge, "_OPS", {})
    built = []

    def create(name, fn, **kwargs):
        op = Mock()
        built.append((name, fn, op))
        return op

    monkeypatch.setattr(bridge, "jax_op", create)
    mesh = object()
    first = bridge._build_op(mesh, 2, True, "silu", None, "fp4", 64)
    second = bridge._build_op(mesh, 2, True, "silu", None, "fp8", None)
    assert first is not second
    assert bridge._build_op(mesh, 2, True, "silu", None, "fp4", 64) is first
    assert len(built) == 2 and built[0][0] != built[1][0]
    gate = SimpleNamespace(astype=lambda _: "gate_f32")
    assert built[0][1](1, 2, 3, 4, 5, gate, 6) == "result"
    assert kernel.call_args.kwargs["weight_format"] == "fp4"
    assert kernel.call_args.kwargs["rhs_qb"] == 64
    assert "sharded_plan" not in kernel.call_args.kwargs


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_prebuild_passes_native_token_groups(context, monkeypatch, sequence_parallel):
    monkeypatch.setenv("TPU_MOE_LOWTOK_WHOLE_EXPERT", "1")
    monkeypatch.setenv("TPU_MOE_LOWTOK_WHOLE_EXPERT_MIN_TOKENS", "192")
    value = layer()
    value.moe_config.moe_parallel_config.is_sequence_parallel = sequence_parallel
    groups = ((0,), (1,)) if sequence_parallel else ((1, 0),)
    discover = Mock(return_value=groups)
    monkeypatch.setattr(bridge, "ep_token_replica_groups", discover)
    prebuild(value)
    discover.assert_called_once_with(is_sequence_parallel=sequence_parallel)
    assert context[1].call_args.kwargs["token_replica_groups"] == groups
    assert context[1].call_args.kwargs["lowtok_whole_expert"] is True
    assert context[1].call_args.kwargs["lowtok_whole_expert_min_tokens"] == 192


def test_op_cache_separates_replica_groups_and_passes_them_to_kernel(monkeypatch):
    import importlib

    package = importlib.import_module(
        "vllm_torchtpu.kernels.experimental.adaptive_fused_moe"
    )
    kernel = Mock(return_value="output")
    monkeypatch.setattr(package, "adaptive_fused_moe", kernel)
    monkeypatch.setattr(bridge, "_OPS", {})
    functions = []

    def create(name, fn, **kwargs):
        functions.append(fn)
        return Mock()

    monkeypatch.setattr(bridge, "jax_op", create)
    mesh = object()
    groups = ((1, 0),)
    original = bridge._build_op(mesh, 2, True, "silu", None)
    tp = bridge._build_op(mesh, 2, True, "silu", None, token_replica_groups=groups)
    assert original is not tp
    assert tp is bridge._build_op(
        mesh, 2, True, "silu", None, token_replica_groups=groups
    )
    gate = SimpleNamespace(astype=lambda _: "gate")
    functions[-1](1, 2, 3, 4, 5, gate, 6)
    assert kernel.call_args.kwargs["token_replica_groups"] == groups


def test_op_cache_separates_lowtok_whole_expert_and_forwards_it(monkeypatch):
    import importlib

    import jax
    import jax.numpy as jnp

    package = importlib.import_module(
        "vllm_torchtpu.kernels.experimental.adaptive_fused_moe"
    )
    monkeypatch.setattr(package, "adaptive_fused_moe", Mock(return_value="adaptive"))
    monkeypatch.setattr(bridge, "_OPS", {})
    lowtok_ep = Mock(return_value="lowtok_out")
    lowtok_scores_ep = Mock(return_value="lowtok_scores_out")
    monkeypatch.setattr(bridge, "moe_lowtok_ep", lowtok_ep)
    monkeypatch.setattr(bridge, "moe_lowtok_scores_ep", lowtok_scores_ep)
    real_resolve = bridge.resolve_whole_expert_ep
    resolve_spy = Mock(wraps=real_resolve)
    monkeypatch.setattr(bridge, "resolve_whole_expert_ep", resolve_spy)
    functions = {}

    def create(name, fn, **kwargs):
        op = Mock(name=name)
        functions[op] = fn
        return op

    monkeypatch.setattr(bridge, "jax_op", create)
    ep = 4
    mesh = Mock(shape={bridge.EP_AXIS_NAME: ep})
    split_op = bridge._build_op(
        mesh, 2, True, "silu", None, lowtok_max_tokens=16, lowtok_whole_expert=False
    )
    # When lowtok_whole_expert=False, a non-zero min_tokens normalizes to 0 and
    # reuses the cached split_op instead of splitting custom-op caches.
    assert (
        bridge._build_op(
            mesh,
            2,
            True,
            "silu",
            None,
            lowtok_max_tokens=16,
            lowtok_whole_expert=False,
            lowtok_whole_expert_min_tokens=12,
        )
        is split_op
    )
    whole_op = bridge._build_op(
        mesh, 2, True, "silu", None, lowtok_max_tokens=16, lowtok_whole_expert=True
    )
    gated_op = bridge._build_op(
        mesh,
        2,
        True,
        "silu",
        None,
        lowtok_max_tokens=16,
        lowtok_whole_expert=True,
        lowtok_whole_expert_min_tokens=12,
    )
    assert split_op is not whole_op
    assert gated_op is not whole_op
    assert (
        bridge._build_op(
            mesh, 2, True, "silu", None, lowtok_max_tokens=16, lowtok_whole_expert=True
        )
        is whole_op
    )

    # Real ShapeDtypeStruct operands exercise real resolve_whole_expert_ep +
    # moe_lowtok_ep_serves without mocking away coverage or VMEM checks.
    w1 = jax.ShapeDtypeStruct((ep * 16, 512, 512), jnp.float8_e4m3fn)
    w2 = jax.ShapeDtypeStruct((ep * 16, 256, 512), jnp.float8_e4m3fn)
    s1 = jax.ShapeDtypeStruct((ep * 16, 512), jnp.float32)
    s2 = jax.ShapeDtypeStruct((ep * 16, 512), jnp.float32)
    x8 = jax.ShapeDtypeStruct((8, 512), jnp.bfloat16)
    x16 = jax.ShapeDtypeStruct((16, 512), jnp.bfloat16)
    x64 = jax.ShapeDtypeStruct((64, 512), jnp.bfloat16)
    gate64 = SimpleNamespace(shape=(64, bridge.LANES), astype=lambda _: "gate_f32")

    assert functions[whole_op](x8, w1, w2, s1, s2, 6, 7) == "lowtok_out"
    assert lowtok_ep.call_args.kwargs["whole_expert"] is True
    assert resolve_spy.call_args.kwargs["scores"] is False

    # Same prebuilt gated_op selects split below 12 rows and whole at >= 12 rows.
    assert functions[gated_op](x8, w1, w2, s1, s2, 6, 7) == "lowtok_out"
    assert lowtok_ep.call_args.kwargs["whole_expert"] is False
    assert functions[gated_op](x16, w1, w2, s1, s2, 6, 7) == "lowtok_out"
    assert lowtok_ep.call_args.kwargs["whole_expert"] is True

    # Prefill buckets above lowtok_max_tokens bypass resolve_whole_expert_ep.
    resolve_spy.reset_mock()
    assert functions[gated_op](x64, w1, w2, s1, s2, gate64, 7) == "adaptive"
    resolve_spy.assert_not_called()

    scores_op = bridge._build_op(
        mesh,
        2,
        True,
        "silu",
        None,
        lowtok_max_tokens=16,
        lowtok_scores=True,
        lowtok_whole_expert=True,
        lowtok_whole_expert_min_tokens=12,
    )
    gating8 = SimpleNamespace(shape=(8, bridge.LANES))
    gating16 = SimpleNamespace(shape=(16, bridge.LANES))
    assert (
        functions[scores_op](x8, w1, w2, s1, s2, gating8, *range(7, 17))
        == "lowtok_scores_out"
    )
    assert lowtok_scores_ep.call_args.kwargs["whole_expert"] is False
    assert resolve_spy.call_args.kwargs["scores"] is True
    assert (
        functions[scores_op](x16, w1, w2, s1, s2, gating16, *range(7, 17))
        == "lowtok_scores_out"
    )
    assert lowtok_scores_ep.call_args.kwargs["whole_expert"] is True

    # Narrow inter=128 (two_inter=256) is only supported by whole_expert=True:
    # even at 8 rows (< min_tokens=12), moe_from_scores falls back to whole=True
    # instead of dropping out to adaptive_fused_moe.
    w1_narrow = jax.ShapeDtypeStruct((ep * 16, 512, 256), jnp.float8_e4m3fn)
    w2_narrow = jax.ShapeDtypeStruct((ep * 16, 128, 512), jnp.float8_e4m3fn)
    s1_narrow = jax.ShapeDtypeStruct((ep * 16, 256), jnp.float32)
    assert (
        functions[scores_op](
            x8, w1_narrow, w2_narrow, s1_narrow, s2, gating8, *range(7, 17)
        )
        == "lowtok_scores_out"
    )
    assert lowtok_scores_ep.call_args.kwargs["whole_expert"] is True

    # Prefill buckets above lowtok_max_tokens fall back to adaptive_fused_moe on
    # the scores entry too instead of calling moe_lowtok_scores_ep.
    scores_copy_op = bridge._build_op(
        mesh,
        2,
        True,
        "silu",
        None,
        lowtok_max_tokens=16,
        lowtok_scores=True,
        lowtok_scores_copies=True,
        lowtok_whole_expert=True,
        lowtok_whole_expert_min_tokens=12,
    )
    resolve_spy.reset_mock()
    lowtok_scores_ep.reset_mock()
    assert (
        functions[scores_copy_op](
            x64,
            w1,
            w2,
            s1,
            s2,
            jnp.zeros((64, bridge.LANES), jnp.bfloat16),
            7,
            jnp.zeros((64,), jnp.int32),
            *range(9, 17),
        )
        == "adaptive"
    )
    resolve_spy.assert_not_called()
    lowtok_scores_ep.assert_not_called()


def test_disabled_lowtok_skips_serves_check(monkeypatch):
    import importlib

    package = importlib.import_module(
        "vllm_torchtpu.kernels.experimental.adaptive_fused_moe"
    )
    monkeypatch.setattr(package, "adaptive_fused_moe", Mock(return_value="adaptive"))
    serves = Mock(side_effect=AssertionError("should not check lowtok when disabled"))
    monkeypatch.setattr(bridge, "moe_lowtok_ep_serves", serves)
    monkeypatch.setattr(bridge, "_OPS", {})
    functions = []
    monkeypatch.setattr(
        bridge, "jax_op", lambda name, fn, **kw: functions.append(fn) or Mock()
    )
    mesh = Mock(shape={bridge.EP_AXIS_NAME: 4})
    bridge._build_op(mesh, 2, True, "silu", None, lowtok_max_tokens=0)
    gate = SimpleNamespace(astype=lambda _: "gate")
    assert functions[-1](1, 2, 3, 4, 5, gate, 6) == "adaptive"
    serves.assert_not_called()

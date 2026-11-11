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
"""Tests for the routing MoE bridge (layers/adapter/routing_moe.py).

None of this needs a TPU: the refusals and arming are decided before any op is
built, `_build_op` is stubbed where prebuild would reach it, and routing and
expert relabelling are plain JAX.
"""

from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax import lax

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import routing_moe as bridge
from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner

_STUB_EP = 2


def _layer(experts=4, hidden=8, inter=16, **overrides):
    """A layer the integration would accept."""
    layer = SimpleNamespace(
        w13_weight=torch.zeros(experts, hidden, 2 * inter, dtype=torch.float8_e4m3fn),
        w2_weight=torch.zeros(experts, inter, hidden, dtype=torch.float8_e4m3fn),
        global_num_experts=experts * _STUB_EP,
        scoring_func="softmax",
        e_score_correction_bias=None,
        routed_scaling_factor=1.0,
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(use_ep=True, is_sequence_parallel=False)
        ),
    )
    for name, value in overrides.items():
        setattr(layer, name, value)
    return layer


def _tp_layer(experts=4, hidden=8, inter=16, **overrides):
    """A tensor-parallel layer: all experts on every rank."""
    layer = _layer(experts, hidden, inter, global_num_experts=experts)
    layer.moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=False, is_sequence_parallel=False)
    )
    for name, value in overrides.items():
        setattr(layer, name, value)
    return layer


def _prebuild(
    layer,
    *,
    has_mesh=True,
    ep_order=(0, 1),
    build_op=None,
    build_tp_op=None,
    tp_reason=None,
):
    """Call prebuild with the distributed lookups and the op build stubbed."""
    mesh = (
        SimpleNamespace(
            shape={bridge.EP_AXIS_NAME: _STUB_EP}, axis_names=(bridge.EP_AXIS_NAME,)
        )
        if has_mesh
        else None
    )
    build_op = build_op or (lambda *_args, **_kwargs: "op")
    build_tp_op = build_tp_op or (lambda *_args, **_kwargs: "tp_op")
    with (
        patch.object(bridge, "build_ep_mesh", return_value=mesh),
        patch.object(bridge, "ep_mesh_index", return_value=0),
        patch.object(bridge, "ep_rank_order", return_value=ep_order),
        patch.object(bridge, "ep_token_replica_groups", return_value=((0, 1),)),
        patch.object(bridge, "_build_op", side_effect=build_op),
        patch.object(bridge, "_build_tp_op", side_effect=build_tp_op),
        patch.object(bridge, "_tp_group_reason", return_value=tp_reason),
    ):
        return bridge.prebuild_routing_moe(
            layer, topk=2, renormalize=True, activation="silu"
        )


@pytest.fixture(autouse=True)
def _armed():
    """Every test here runs as if the operator asked for the kernel."""
    with (
        patch.object(envs, "USE_ROUTING_MOE", True),
        patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", False),
        patch.object(envs, "ROUTING_MOE_REMOTE_SCATTER", True),
    ):
        yield


def test_off_without_the_flag():
    with patch.object(envs, "USE_ROUTING_MOE", False):
        assert _prebuild(_layer()) is None


def test_arms_on_a_supported_layer():
    assert _prebuild(_layer()) == "op"


def test_exclusive_with_the_fused_ep_kernel():
    with (
        patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", True),
        pytest.raises(ValueError, match="exclusive"),
    ):
        _prebuild(_layer())


@pytest.mark.parametrize(
    "overrides, has_mesh, want",
    [
        (dict(scoring_func="noaux_tc"), True, "noaux_tc"),
        (dict(e_score_correction_bias=torch.zeros(8)), True, "correction bias"),
        (dict(routed_scaling_factor=2.5), True, "routed_scaling_factor"),
        (dict(hash_indices_table=torch.zeros(8, 2)), True, "hash routing"),
        (dict(w13_bias=torch.zeros(4, 1, 32)), True, "biases"),
        ({}, False, "no EP group"),
        (dict(global_num_experts=9), True, "split evenly"),
    ],
)
def test_refuses_an_unsupported_layer(overrides, has_mesh, want):
    """A layer the integration can't run raises, rather than running on
    another backend."""
    with pytest.raises(ValueError, match=want):
        _prebuild(_layer(**overrides), has_mesh=has_mesh)


@pytest.mark.parametrize("remote", [True, False])
def test_scatter_mode_follows_the_flag(remote):
    seen = []
    with patch.object(envs, "ROUTING_MOE_REMOTE_SCATTER", remote):
        _prebuild(_layer(), build_op=lambda *args, **kw: seen.append(args) or "op")
    assert seen[0][-1] is remote


def test_identity_ep_order_needs_no_relabelling():
    seen = []
    _prebuild(_layer(), build_op=lambda *args, **kw: seen.append(args) or "op")
    assert seen[0][5] is None


class TestTensorParallel:
    """Without EP the layer runs the kernel's TP mode over the TP group."""

    def test_arms_the_tp_op(self):
        seen = []
        got = _prebuild(
            _tp_layer(),
            build_tp_op=lambda *args, **kw: seen.append(args) or "tp_op",
        )
        assert got == "tp_op"
        # mesh, topk, renormalize, scoring_fn, activation
        assert seen[0][1:] == (2, True, "softmax", "silu")

    def test_does_not_build_the_ep_op(self):
        def no_ep(*_args, **_kwargs):
            raise AssertionError("TP must not build the EP op")

        assert _prebuild(_tp_layer(), build_op=no_ep) == "tp_op"

    @pytest.mark.parametrize(
        "overrides, kwargs, want",
        [
            (
                dict(
                    moe_config=SimpleNamespace(
                        moe_parallel_config=SimpleNamespace(
                            use_ep=False, is_sequence_parallel=True
                        )
                    )
                ),
                {},
                "sequence-parallel",
            ),
            ({}, dict(tp_reason="the TP group has one device"), "one device"),
            ({}, dict(has_mesh=False), "no TP group mesh"),
            (dict(global_num_experts=8), {}, "all 8 experts"),
            (dict(scoring_func="noaux_tc"), {}, "noaux_tc"),
        ],
    )
    def test_refuses(self, overrides, kwargs, want):
        with pytest.raises(ValueError, match=want):
            _prebuild(_tp_layer(**overrides), **kwargs)

    def test_tp_moe_matches_a_dense_reference(self):
        """Each rank's [gate | up] slice and w2 slice, handed over sharded on
        axis 0, reach the kernel split along the intermediate dim, and the
        all-reduced result equals the unsharded MoE."""
        tp = 2
        if jax.device_count() < tp:
            pytest.skip(
                "needs 2 devices (XLA_FLAGS=--xla_force_host_platform_device_count=2)"
            )
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P

        axis = bridge.EP_AXIS_NAME

        def ein(spec, *operands):
            # TPU runs default-precision f32 matmuls through bf16 passes;
            # compare at full precision so the check is on layout, not rounding.
            return jnp.einsum(spec, *operands, precision=lax.Precision.HIGHEST)

        mesh = Mesh(np.array(jax.devices()[:tp]), (axis,))
        tokens, hidden, experts, inter_l, topk = 3, 8, 4, 4, 2
        keys = jax.random.split(jax.random.key(1), 4)
        x = jax.random.normal(keys[0], (tokens, hidden), jnp.float32)
        gating = jax.random.normal(keys[1], (tokens, experts), jnp.float32)
        # Per rank: w13 [E, H, gate_r | up_r], w2 [E, I_r, H].
        w13 = jax.random.normal(keys[2], (tp, experts, hidden, 2 * inter_l))
        w2 = jax.random.normal(keys[3], (tp, experts, inter_l, hidden))
        s13 = jnp.ones((tp, experts, 1, 1, 2 * inter_l))
        s2 = jnp.ones((tp, experts, 1, 1, hidden))

        def stub_kernel(x, weights, ids, w1, w2, w1_scale, w2_scale, b1, b2, **kw):
            assert kw["tp_axis"] == axis and kw["ep_axis"] is None
            assert not kw["use_sp"] and kw["reduce_output"]

            def local(x, weights, ids, w1, w2, s1, s2):
                assert w1.shape == (experts, hidden, 2 * inter_l)
                assert w2.shape == (experts, inter_l, hidden)
                assert s1.shape == (experts, 1, 1, 2 * inter_l)
                assert s2.shape == (experts, 1, 1, hidden)
                h = ein("th,tkhn->tkn", x, w1[ids])
                gate, up = jnp.split(h, 2, axis=-1)
                y = ein("tki,tkih->tkh", jax.nn.silu(gate) * up, w2[ids])
                return lax.psum(ein("tkh,tk->th", y, weights), axis)

            return jax.shard_map(
                local,
                mesh=mesh,
                in_specs=(
                    P(),
                    P(),
                    P(),
                    P(None, None, axis),
                    P(None, axis, None),
                    P(None, None, None, axis),
                    P(None, axis, None, None),
                ),
                out_specs=P(),
                check_vma=False,
            )(x, weights, ids, w1, w2, w1_scale, w2_scale)

        moe = bridge._tp_moe_fn(mesh, topk, True, "softmax", "silu", stub_kernel)
        rows = NamedSharding(mesh, P(axis))

        def stacked(a):
            return jax.device_put(a.reshape(-1, *a.shape[2:]), rows)

        out = moe(
            jax.device_put(jnp.concatenate([x] * tp), rows),
            stacked(w13),
            stacked(w2),
            stacked(s13),
            stacked(s2),
            jax.device_put(jnp.concatenate([gating] * tp), rows),
            jax.device_put(jnp.arange(tp, dtype=jnp.int32).reshape(tp, 1), rows),
        )

        weights, ids = bridge._route(gating, topk, True, "softmax")
        gate_full = jnp.concatenate([w13[r, ..., :inter_l] for r in range(tp)], -1)
        up_full = jnp.concatenate([w13[r, ..., inter_l:] for r in range(tp)], -1)
        w2_full = jnp.concatenate([w2[r] for r in range(tp)], axis=1)
        act = jax.nn.silu(ein("th,tkhi->tki", x, gate_full[ids])) * ein(
            "th,tkhi->tki", x, up_full[ids]
        )
        want = ein("tkh,tk->th", ein("tki,tkih->tkh", act, w2_full[ids]), weights)
        got = np.asarray(out).reshape(tp, tokens, hidden)
        for r in range(tp):
            np.testing.assert_allclose(got[r], want, rtol=2e-5, atol=2e-5)


def test_relabel_table():
    """Mesh index m holds EP rank order[m]'s experts, and the kernel reads
    expert e as mesh index e // local_experts."""
    order = (2, 0, 3, 1)
    table = bridge._relabel_table(order, local_experts=3)
    for g, kernel_id in enumerate(table):
        assert order[kernel_id // 3] == g // 3
        assert kernel_id % 3 == g % 3
    assert sorted(table) == list(range(12))


@pytest.mark.parametrize("topk_impl", ["rowmax", "sort"])
@pytest.mark.parametrize("scoring_fn", ["softmax", "sigmoid"])
@pytest.mark.parametrize("renormalize", [True, False])
def test_route_matches_select_experts(topk_impl, scoring_fn, renormalize):
    gating = jax.random.normal(jax.random.key(0), (5, 16), jnp.bfloat16)
    with patch.object(envs, "TPU_MOE_ROUTER_TOPK", topk_impl):
        weights, ids = bridge._route(gating, 4, renormalize, scoring_fn)
    g = np.asarray(gating, np.float32)
    scores = (
        np.exp(g) / np.exp(g).sum(-1, keepdims=True)
        if scoring_fn == "softmax"
        else 1 / (1 + np.exp(-g))
    )
    want_ids = np.argsort(-scores, axis=-1, kind="stable")[:, :4]
    want = np.take_along_axis(scores, want_ids, axis=-1)
    if renormalize:
        want = want / want.sum(-1, keepdims=True)
    np.testing.assert_array_equal(np.sort(ids, -1), np.sort(want_ids, -1))
    np.testing.assert_allclose(np.sort(weights, -1), np.sort(want, -1), rtol=1e-5)
    assert ids.dtype == jnp.int32


def test_route_dispatches_to_router_topk_on_rowmax():
    gating = jax.random.normal(jax.random.key(7), (16, 512), jnp.bfloat16)
    with (
        patch.object(envs, "TPU_MOE_ROUTER_TOPK", "rowmax"),
        patch.object(bridge, "router_select", wraps=bridge.router_select) as sel_spy,
    ):
        w_rowmax, i_rowmax = bridge._route(gating, 10, True, "softmax")
        assert sel_spy.call_count == 1
    with (
        patch.object(envs, "TPU_MOE_ROUTER_TOPK", "sort"),
        patch.object(bridge, "router_select", wraps=bridge.router_select) as sel_spy,
    ):
        w_sort, i_sort = bridge._route(gating, 10, True, "softmax")
        assert sel_spy.call_count == 0
    np.testing.assert_array_equal(np.asarray(i_rowmax), np.asarray(i_sort))
    np.testing.assert_allclose(
        np.asarray(w_rowmax), np.asarray(w_sort), rtol=1e-6, atol=1e-6
    )


class TestOutputIsReduced:
    """An armed layer returns this rank's tokens combined over all experts,
    so the runner must not all-reduce them again."""

    def _answer(self, *, armed):
        quant_method = SimpleNamespace(moe_kernel=None)
        if armed:
            setattr(quant_method, bridge.ROUTING_MOE_OP_ATTR, "op")
        runner = VllmMoERunner.__new__(VllmMoERunner)
        torch.nn.Module.__init__(runner)
        runner.routed_experts = SimpleNamespace(quant_method=quant_method)
        runner.moe_config = SimpleNamespace(skip_final_all_reduce=False)
        return runner._fused_output_is_reduced

    def test_reports_reduced_only_for_an_armed_layer(self):
        assert self._answer(armed=True) is True
        assert self._answer(armed=False) is False

    def test_defers_to_upstream_without_the_flag(self):
        with patch.object(envs, "USE_ROUTING_MOE", False):
            assert self._answer(armed=True) is False

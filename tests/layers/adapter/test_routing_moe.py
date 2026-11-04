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


def _prebuild(layer, *, has_mesh=True, ep_order=(0, 1), build_op=None):
    """Call prebuild with the distributed lookups and the op build stubbed."""
    mesh = SimpleNamespace(shape={bridge.EP_AXIS_NAME: _STUB_EP}) if has_mesh else None
    build_op = build_op or (lambda *_args, **_kwargs: "op")
    with (
        patch.object(bridge, "build_ep_mesh", return_value=mesh),
        patch.object(bridge, "ep_mesh_index", return_value=0),
        patch.object(bridge, "ep_rank_order", return_value=ep_order),
        patch.object(bridge, "ep_token_replica_groups", return_value=((0, 1),)),
        patch.object(bridge, "_build_op", side_effect=build_op),
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
        (
            dict(
                moe_config=SimpleNamespace(
                    moe_parallel_config=SimpleNamespace(
                        use_ep=False, is_sequence_parallel=False
                    )
                )
            ),
            True,
            "expert-parallel",
        ),
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


def test_relabel_table():
    """Mesh index m holds EP rank order[m]'s experts, and the kernel reads
    expert e as mesh index e // local_experts."""
    order = (2, 0, 3, 1)
    table = bridge._relabel_table(order, local_experts=3)
    for g, kernel_id in enumerate(table):
        assert order[kernel_id // 3] == g // 3
        assert kernel_id % 3 == g % 3
    assert sorted(table) == list(range(12))


@pytest.mark.parametrize("scoring_fn", ["softmax", "sigmoid"])
@pytest.mark.parametrize("renormalize", [True, False])
def test_route_matches_select_experts(scoring_fn, renormalize):
    gating = jax.random.normal(jax.random.key(0), (5, 16), jnp.bfloat16)
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

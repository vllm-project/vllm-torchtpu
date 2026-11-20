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
        patch.object(envs, "ROUTING_MOE_FP8_COMBINE", False),
        patch.object(envs, "ROUTING_MOE_FP8_DISPATCH", False),
        patch.object(envs, "ROUTING_MOE_GMM_PLAN", ""),
        patch.object(envs, "ROUTING_MOE_FP8_DISPATCH_MIN_ROWS", 256),
        patch.object(envs, "ROUTING_MOE_PACK_BEFORE_GATHER", False),
        patch.object(envs, "ROUTING_MOE_FP8_DISPATCH_TO_KERNEL", False),
        patch.object(envs, "ROUTING_MOE_LOCAL_GROUP_SIZES", False),
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


@pytest.mark.parametrize("fp8_combine", [False, True])
def test_fp8_combine_follows_the_flag(fp8_combine):
    seen = []
    with patch.object(envs, "ROUTING_MOE_FP8_COMBINE", fp8_combine):
        _prebuild(
            _layer(hidden=512),
            build_op=lambda *args, **kw: seen.append(kw) or "op",
        )
    assert seen[0]["fp8_combine"] is fp8_combine


def test_fp8_combine_refuses_a_hidden_size_off_512():
    """fp8 combine rows fill whole 512-element chunks; another hidden size
    raises instead of falling back to bf16 rows."""
    with (
        patch.object(envs, "ROUTING_MOE_FP8_COMBINE", True),
        pytest.raises(ValueError, match="divisible by 512"),
    ):
        _prebuild(_layer(hidden=256))


def test_scatter_format():
    """The flag gives tokamax an fp8 lane-strided scatter_format; off, none
    (bf16 rows)."""
    out_format = pytest.importorskip(
        "tokamax._src.ops.experimental.routing_moe.common.out_format"
    )
    assert bridge._scatter_format(False) is None
    fmt = bridge._scatter_format(True)
    assert fmt.dtype == jnp.float8_e4m3fn
    assert isinstance(fmt.quant, out_format.LaneStrided)


@pytest.mark.parametrize("fp8_dispatch", [False, True])
def test_fp8_dispatch_follows_the_flag(fp8_dispatch):
    seen = []
    with patch.object(envs, "ROUTING_MOE_FP8_DISPATCH", fp8_dispatch):
        _prebuild(
            _layer(hidden=512),
            build_op=lambda *args, **kw: seen.append(kw) or "op",
        )
    assert seen[0]["fp8_dispatch"] is fp8_dispatch
    assert seen[0]["fp8_dispatch_min_rows"] == 256


def test_fp8_dispatch_needs_remote_scatter():
    with (
        patch.object(envs, "ROUTING_MOE_FP8_DISPATCH", True),
        patch.object(envs, "ROUTING_MOE_REMOTE_SCATTER", False),
        pytest.raises(ValueError, match="remote scatter"),
    ):
        _prebuild(_layer(hidden=512))


def test_fp8_dispatch_rows_round_trip():
    """One f32 scale per row: each row comes back within fp8's relative
    precision of its largest value, and an all-zero row stays zero."""
    x = jax.random.normal(jax.random.PRNGKey(0), (6, 512), jnp.float32) * 3.0
    x = x.at[2].set(0.0).astype(jnp.bfloat16)
    q, scale = bridge._quantize_rows(x)
    assert q.dtype == jnp.float8_e4m3fn and scale.shape == (6, 1)
    back = bridge._dequantize_rows(q, scale, jnp.bfloat16)
    assert back.dtype == jnp.bfloat16
    err = np.abs(np.asarray(back, np.float32) - np.asarray(x, np.float32))
    amax = np.abs(np.asarray(x, np.float32)).max(-1, keepdims=True)
    assert (err <= amax / 8 + 1e-6).all()
    assert not np.asarray(back[2], np.float32).any()


@pytest.mark.parametrize("pack", [False, True])
def test_pack_before_gather_follows_the_flag(pack):
    seen = []
    with patch.object(envs, "ROUTING_MOE_PACK_BEFORE_GATHER", pack):
        _prebuild(_layer(), build_op=lambda *args, **kw: seen.append(kw) or "op")
    assert seen[0]["pack_before_gather"] is pack


def test_fp8_to_kernel_needs_fp8_dispatch():
    with (
        patch.object(envs, "ROUTING_MOE_FP8_DISPATCH_TO_KERNEL", True),
        pytest.raises(ValueError, match="needs ROUTING_MOE_FP8_DISPATCH"),
    ):
        _prebuild(_layer())


def test_fp8_scale_chunk_words_hold_the_row_scale():
    """Each of the chunk's 128 words, reassembled from its 4 packed bytes
    (byte p at packing position p), is the row's f32 scale."""
    scale = jnp.asarray([[0.5], [3.25e-3], [1.0e6]], jnp.float32)
    chunk = bridge._fp8_scale_chunk(scale)
    assert chunk.shape == (3, 1, 4, 128) and chunk.dtype == jnp.float8_e4m3fn
    octets = np.asarray(jax.lax.bitcast_convert_type(chunk, jnp.uint8), np.uint32)
    words = sum(octets[:, 0, p, :] << (8 * p) for p in range(4)).astype(np.uint32)
    assert (words.view(np.float32) == np.asarray(scale)).all()


def test_local_group_sizes_needs_packed_gather():
    with (
        patch.object(envs, "ROUTING_MOE_LOCAL_GROUP_SIZES", True),
        pytest.raises(ValueError, match="needs ROUTING_MOE_PACK_BEFORE_GATHER"),
    ):
        _prebuild(_layer())


def test_local_group_sizes_follows_the_flag():
    seen = []
    with (
        patch.object(envs, "ROUTING_MOE_LOCAL_GROUP_SIZES", True),
        patch.object(envs, "ROUTING_MOE_PACK_BEFORE_GATHER", True),
    ):
        _prebuild(_layer(), build_op=lambda *args, **kw: seen.append(kw) or "op")
    assert seen[0]["local_group_sizes"] is True


def test_gmm_plan_parsing():
    assert bridge.parse_gmm_plan("") == ()
    flat = bridge.parse_gmm_plan('{"tile_m": 512, "acc_dtype": "float32"}')
    assert bridge.gmm_plan_fields(flat, 47) == {"tile_m": 512, "acc_dtype": "float32"}
    keyed = bridge.parse_gmm_plan(
        '{"by_rows": [[512, {"tile_m": 256}], [64, {"tile_m": 64}]],'
        ' "default": {"tile_m": 512}}'
    )
    assert bridge.gmm_plan_fields(keyed, 47) == {"tile_m": 64}
    assert bridge.gmm_plan_fields(keyed, 512) == {"tile_m": 256}
    assert bridge.gmm_plan_fields(keyed, 1024) == {"tile_m": 512}
    only_rows = bridge.parse_gmm_plan('{"by_rows": [[64, {"tile_m": 64}]]}')
    assert bridge.gmm_plan_fields(only_rows, 512) is None
    hash(keyed)
    with pytest.raises(ValueError, match="unknown keys"):
        bridge.parse_gmm_plan('{"by_rows": [], "tiles": {}}')


def test_gmm_plan_follows_the_env():
    seen = []
    with patch.object(envs, "ROUTING_MOE_GMM_PLAN", '{"x_quant_block_size": 8192}'):
        _prebuild(_layer(), build_op=lambda *args, **kw: seen.append(kw) or "op")
    assert seen[0]["gmm_plan"] == ((None, (("x_quant_block_size", 8192),)),)


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

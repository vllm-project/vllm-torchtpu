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
"""Tests for the GLM 5.3 fused MoE op (`custom_ops/glm53_moe_tpu.py`)."""

from __future__ import annotations

import inspect
import types
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax._src import tpu_info as jax_tpu_info
from jax._src.interpreters import mlir
from jax._src.pallas.mosaic import tpu_info
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import fused_moe_ep, moe_routing
from vllm_torchtpu.layers.adapter.custom_ops import glm53_moe_tpu
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8LinearMethodTPU,
    VllmFp8MoEMethodTPU,
)


def _to_jax(t: torch.Tensor) -> jax.Array:
    return jnp.asarray(t.float().numpy()).astype(
        jnp.bfloat16 if t.dtype == torch.bfloat16 else jnp.float32
    )


def _make_fake_moe(*, use_ep: bool = False):
    experts_start = (
        torch.tensor(36, dtype=torch.int32)
        if use_ep
        else torch.tensor(0, dtype=torch.int32)
    )
    routed_method = mock.MagicMock(spec=VllmFp8MoEMethodTPU)
    routed_method._use_adaptive_fused_moe = False
    routed_method._tpu_activation_str = "silu_and_mul_with_clamp"
    routed = types.SimpleNamespace(
        quant_method=routed_method,
        moe_config=types.SimpleNamespace(
            moe_parallel_config=types.SimpleNamespace(use_ep=use_ep)
        ),
        w13_weight=torch.zeros(36, 512, 64),
        w13_weight_scale_inv=torch.zeros(36, 1, 512),
        w2_weight=torch.zeros(36, 64, 256),
        w2_weight_scale_inv=torch.zeros(36, 1, 64),
        _experts_start=experts_start if use_ep else None,
        renormalize=True,
        custom_routing_function=None,
        w13_bias=None,
        w2_bias=None,
    )
    shared = types.SimpleNamespace(
        gate_up_proj=types.SimpleNamespace(
            quant_method=mock.MagicMock(spec=VllmFp8LinearMethodTPU),
            weight=torch.zeros(1, 512, 64),
            weight_scale=torch.zeros(1, 1, 1, 512),
        ),
        down_proj=types.SimpleNamespace(
            quant_method=mock.MagicMock(spec=VllmFp8LinearMethodTPU),
            weight=torch.zeros(1, 64, 256),
            weight_scale=torch.zeros(1, 1, 1, 64),
        ),
        swiglu_limit=10.0,
    )
    moe = types.SimpleNamespace(
        experts=types.SimpleNamespace(
            routed_experts=routed,
            shared_experts=shared,
            top_k=8,
        ),
        shared_experts=shared,
        gate=types.SimpleNamespace(
            weight=torch.zeros(288, 64),
            e_score_correction_bias=torch.zeros(288),
        ),
        top_k=8,
        routed_scaling_factor=2.5,
        swiglu_limit=10.0,
    )
    return moe, experts_start


@pytest.mark.parametrize(
    ("raw", "want"),
    [("", "fused"), ("Fused", "fused"), ("fused", "fused"), ("vllm", "vllm")],
)
def test_moe_impl_env(monkeypatch, raw, want):
    monkeypatch.setattr(envs, "VLLM_TPU_GLM53_MOE_IMPL", raw, raising=False)
    assert glm53_moe_tpu.glm53_moe_impl() == want


def test_moe_impl_env_rejects_unknown(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_TPU_GLM53_MOE_IMPL", "bogus", raising=False)
    with pytest.raises(ValueError, match="VLLM_TPU_GLM53_MOE_IMPL"):
        glm53_moe_tpu.glm53_moe_impl()


def test_jax_op_bodies_are_fully_annotated():
    for fn in (
        glm53_moe_tpu._glm53_moe_jax,
        glm53_moe_tpu._glm53_moe_stacked_jax,
    ):
        for name, p in inspect.signature(fn).parameters.items():
            assert p.annotation is not inspect.Parameter.empty, f"{fn.__name__}.{name}"


def test_clamped_swiglu_jax_matches_torch():
    torch.manual_seed(0)
    gate_up = (torch.randn(16, 64, dtype=torch.bfloat16) * 8.0).clamp(-20.0, 20.0)
    limit = 7.0
    gate, up = gate_up.chunk(2, dim=-1)
    want = torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)
    got = glm53_moe_tpu.clamped_swiglu_jax(_to_jax(gate_up), limit=limit)
    np.testing.assert_allclose(
        np.asarray(got.astype(jnp.float32)),
        want.float().numpy(),
        rtol=1e-2,
        atol=1e-2,
    )


def test_fused_moe_routing_matches_vllm_select_experts(monkeypatch):
    torch.manual_seed(0)
    tokens, hidden, experts, topk, scale = 16, 64, 32, 8, 2.5
    x = torch.randn(tokens, hidden).to(torch.bfloat16)
    gate_w = (torch.randn(experts, hidden) * 0.2).to(torch.bfloat16)
    bias = torch.randn(experts) * 0.1

    logits = torch.nn.functional.linear(x, gate_w).to(torch.float32)
    layer = types.SimpleNamespace(
        hash_indices_table=None,
        use_grouped_topk=False,
        e_score_correction_bias=bias,
        routed_scaling_factor=scale,
    )
    monkeypatch.setattr(moe_routing, "_ROUTER_TOPK", "sort")
    ref_w, ref_ids = moe_routing.select_experts(
        x,
        logits,
        topk=topk,
        renormalize=True,
        scoring_fn="sigmoid",
        layer=layer,
    )

    jw, jids = glm53_moe_tpu.glm53_route_jax(
        _to_jax(x),
        _to_jax(gate_w),
        _to_jax(bias),
        topk=topk,
        routed_scale=scale,
        renormalize=True,
        topk_fn=glm53_moe_tpu._lax_topk,
    )
    np.testing.assert_array_equal(np.asarray(jids), ref_ids.numpy())
    np.testing.assert_allclose(
        np.asarray(jw.astype(jnp.float32)),
        ref_w.float().numpy(),
        rtol=1e-2,
        atol=1e-2,
    )


def test_router_and_sort_parity():
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    gate_w = jax.random.normal(k2, (288, 128), dtype=jnp.bfloat16) * jnp.bfloat16(0.2)
    bias = jax.random.normal(k3, (288,), dtype=jnp.float32) * 0.1
    for tokens in (1, 8, 16, 64):
        x = jax.random.normal(k1, (tokens, 128), dtype=jnp.bfloat16)
        for renorm in (True, False):
            ref_w, ref_ids = glm53_moe_tpu.glm53_route_jax(
                x,
                gate_w,
                bias,
                topk=8,
                routed_scale=2.5,
                renormalize=renorm,
                topk_fn=glm53_moe_tpu._lax_topk,
            )
            fast_w, fast_ids = glm53_moe_tpu.glm53_route_fast_jax(
                x,
                gate_w,
                bias,
                topk=8,
                routed_scale=2.5,
                renormalize=renorm,
                interpret=True,
            )
            np.testing.assert_array_equal(np.asarray(fast_ids), np.asarray(ref_ids))
            np.testing.assert_array_equal(
                np.asarray(jax.lax.bitcast_convert_type(fast_w, jnp.uint16)),
                np.asarray(jax.lax.bitcast_convert_type(ref_w, jnp.uint16)),
            )

        tok_s, w_s, gs, vc = glm53_moe_tpu._sort_activations_no_revert(
            ref_ids,
            ref_w,
            local_num_experts=36,
            use_ep=True,
            experts_start=jnp.int32(36),
        )
        assert tok_s.shape == (tokens * 8,)
        assert w_s.shape == (tokens * 8,)
        assert gs.shape == (36,)
        assert int(gs.sum()) == int(vc[0])


def test_routed_moe_lowers_to_tpu_stablehlo():
    v7x_info = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7, 1)
    with (
        mock.patch.object(mlir.ModuleContext, "get_backend", return_value=None),
        mock.patch.object(jax, "default_backend", return_value="tpu"),
        mock.patch.object(tpu_info, "is_tpu_device", return_value=True),
        mock.patch.object(pltpu, "get_tpu_info", return_value=v7x_info),
        mock.patch.object(tpu_info, "get_tpu_info", return_value=v7x_info),
        mock.patch.object(jax_tpu_info, "get_tpu_info", return_value=v7x_info),
    ):
        for tokens, want_sc_v3 in ((8, 0), (64, 0), (128, 0), (1024, 1)):
            avals = (
                jax.ShapeDtypeStruct((tokens, 2048), jnp.bfloat16),
                jax.ShapeDtypeStruct((288, 2048), jnp.bfloat16),
                jax.ShapeDtypeStruct((288,), jnp.float32),
                jax.ShapeDtypeStruct((36, 2048, 1024), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((36, 512, 2048), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((36, 16, 1, 1024), jnp.float32),
                jax.ShapeDtypeStruct((36, 4, 1, 2048), jnp.float32),
                jax.ShapeDtypeStruct((), jnp.int32),
            )

            @jax.jit
            def fn(x, gate_w, score_bias, w13, w2, w13_scale, w2_scale, experts_start):
                weights, ids = glm53_moe_tpu.glm53_route_fast_jax(
                    x,
                    gate_w,
                    score_bias,
                    topk=8,
                    routed_scale=2.5,
                    renormalize=True,
                )
                return glm53_moe_tpu._glm53_routed_moe_fast_jax(
                    x,
                    w13,
                    w2,
                    w13_scale,
                    w2_scale,
                    weights,
                    ids,
                    experts_start,
                    topk=8,
                    activation="silu",
                    use_ep=True,
                    use_sparse_core=True,
                    onehot_moe_permute_threshold=511,
                )

            lowered = fn.trace(*avals).lower(lowering_platforms=("tpu",))
            asm = lowered.as_text("stablehlo")
            if want_sc_v3 == 0:
                assert asm.count("stablehlo.sort") == 1
                assert asm.count("stablehlo.gather") == 0
            else:
                assert asm.count("sc_ragged_gather_reduce_v3") == 1


def test_fused_moe_args_ep_and_expert_store():
    moe_ep, experts_start = _make_fake_moe(use_ep=True)
    fake_op = mock.MagicMock(return_value="ok")
    with (
        mock.patch.object(fused_moe_ep, "fused_moe_ep_supported", return_value=False),
        mock.patch.object(
            glm53_moe_tpu, "get_glm53_fused_moe_op", return_value=fake_op
        ) as mock_get_op,
    ):
        args_ep = glm53_moe_tpu.Glm53FusedMoeArgs(moe_ep)
        assert args_ep.unsupported == ""
        assert args_ep.op is fake_op
        assert args_ep.tensors[6] is experts_start
        mock_get_op.assert_called_once()
        assert mock_get_op.call_args.kwargs["use_ep"] is True
        with pytest.raises(ValueError, match="expert store"):
            args_ep.use_expert_store(mock.MagicMock(), 0)

    moe_tp, _ = _make_fake_moe(use_ep=False)
    fake_stacked_op = mock.MagicMock(return_value="ok_stacked")
    store = types.SimpleNamespace(
        w13=torch.zeros(4, 36, 512, 64),
        w13_s=torch.zeros(4, 36, 1, 1, 512),
        w2=torch.zeros(4, 36, 64, 256),
        w2_s=torch.zeros(4, 36, 1, 1, 64),
    )
    with (
        mock.patch.object(fused_moe_ep, "fused_moe_ep_supported", return_value=False),
        mock.patch.object(
            glm53_moe_tpu,
            "get_glm53_fused_moe_op",
            side_effect=[fake_op, fake_stacked_op],
        ) as mock_get_op,
    ):
        args_tp = glm53_moe_tpu.Glm53FusedMoeArgs(moe_tp)
        args_tp.use_expert_store(store, 2)
        assert args_tp.op is fake_stacked_op
        assert args_tp.tensors[2] is store.w13
        assert args_tp.tensors[3] is store.w2
        assert args_tp.tensors[4] is store.w13_s
        assert args_tp.tensors[5] is store.w2_s
        assert int(args_tp.tensors[-1].item()) == 2
        assert mock_get_op.call_args.kwargs["stacked"] is True

    moe_eplb, _ = _make_fake_moe(use_ep=True)
    moe_eplb.experts.routed_experts.moe_config.moe_parallel_config.enable_eplb = True
    with mock.patch.object(fused_moe_ep, "fused_moe_ep_supported", return_value=False):
        args_eplb = glm53_moe_tpu.Glm53FusedMoeArgs(moe_eplb)
        assert "EPLB" in args_eplb.unsupported
        assert args_eplb.op is None

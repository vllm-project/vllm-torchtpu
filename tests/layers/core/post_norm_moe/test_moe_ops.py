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
"""CPU parity tests for post_norm_moe JAX helpers vs upstream vLLM classes."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.models.muse_spark.common.layers import (
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from vllm.models.muse_spark.common.moe import MuseSpark1xRouter

import vllm_torchtpu.envs as tpu_envs
from vllm_torchtpu.layers.core import post_norm_moe
from vllm_torchtpu.layers.core.fused_moe_gmm import quantize_to_native_fp4_kmajor
from vllm_torchtpu.layers.core.post_norm_moe import (
    centered_rms_norm,
    input_scaled_rms_norm,
    post_norm_moe_jax,
    sigmoid_biased_topk_route,
)

from ._ref_gmm import ref_gmm


def _torch_bf16_to_jax(t: torch.Tensor) -> jnp.ndarray:
    return jnp.asarray(t.detach().float().numpy()).astype(jnp.bfloat16)


def _jax_bf16_to_torch(a: jnp.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(a.astype(jnp.float32)).copy()).to(torch.bfloat16)


def test_core_module_has_no_top_level_torch_or_vllm_imports() -> None:
    """layers/core/post_norm_moe.py stays JAX-only at module top level."""
    src = Path(post_norm_moe.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    banned = ("torch", "torch_tpu", "vllm")
    offenders: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(banned):
                    offenders.append(alias.name)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith(banned)
        ):
            offenders.append(node.module)
    assert not offenders, f"Unexpected top-level framework imports: {offenders}"


def test_op_has_no_batch_invariant_surface() -> None:
    """Batch-invariant switches and parameters are absent from post_norm_moe."""
    for fn in (
        centered_rms_norm,
        input_scaled_rms_norm,
        sigmoid_biased_topk_route,
        post_norm_moe_jax,
    ):
        assert "batch_invariant" not in inspect.signature(fn).parameters
    muse_envs = [k for k in dir(tpu_envs) if k.startswith("VLLM_TPU_MUSE_")]
    assert not muse_envs, f"Unexpected model-specific env vars: {muse_envs}"


@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
def test_route_matches_upstream_router(routed_scaling_factor: float) -> None:
    """sigmoid_biased_topk_route matches upstream _compute_routing_native."""
    rng = np.random.default_rng(42)
    tokens, hidden, experts, topk = 12, 512, 256, 8
    output_multiplier = 0.17677669529663687

    h_np = rng.standard_normal((tokens, hidden), dtype=np.float32) * 0.5
    w_np = rng.standard_normal((hidden, experts), dtype=np.float32) * 0.1
    # Space biases apart so the 8th/9th scores are well-separated (> 1e-4).
    bias_np = np.linspace(-0.25, 0.25, experts, dtype=np.float32)
    rng.shuffle(bias_np)

    h_t = torch.from_numpy(h_np).to(torch.bfloat16)
    h_j = _torch_bf16_to_jax(h_t)
    w_j = jnp.asarray(w_np)
    b_j = jnp.asarray(bias_np)

    jax_weights, jax_ids = sigmoid_biased_topk_route(
        h_j,
        w_j,
        b_j,
        topk=topk,
        output_multiplier=output_multiplier,
        routed_scaling_factor=routed_scaling_factor,
        router_eps=1e-15,
        router_topk_impl="lax",
    )

    up_router = MuseSpark1xRouter(
        top_k=topk,
        num_experts=experts,
        expert_bias=torch.from_numpy(bias_np),
        route_scale=routed_scaling_factor,
        router_output_multiplier=output_multiplier,
        router_score_normalization="sigmoid",
        router_score_division_eps=1e-15,
    )
    logits_t = h_t.float() @ torch.from_numpy(w_np)
    ref_weights, ref_ids = up_router._compute_routing_native(logits_t, torch.int32)

    jax_ids_np = np.asarray(jax_ids)
    ref_ids_np = ref_ids.numpy()
    for row in range(tokens):
        assert set(jax_ids_np[row].tolist()) == set(ref_ids_np[row].tolist())
        jax_order = np.argsort(jax_ids_np[row])
        ref_order = np.argsort(ref_ids_np[row])
        np.testing.assert_allclose(
            np.asarray(jax_weights)[row][jax_order],
            ref_weights.numpy()[row][ref_order],
            rtol=1e-6,
            atol=1e-6,
        )


def test_route_rejects_transposed_router_weight() -> None:
    """Explicit K-major [H, E] orientation rejects [E, H] when H != E."""
    h = jnp.zeros((4, 64), dtype=jnp.bfloat16)
    bad_w = jnp.zeros((32, 64), dtype=jnp.float32)
    bias = jnp.zeros((32,), dtype=jnp.float32)
    with pytest.raises(ValueError, match="K-major shape"):
        sigmoid_biased_topk_route(h, bad_w, bias, topk=4)


def test_centered_norm_matches_upstream() -> None:
    """centered_rms_norm matches upstream MuseSpark1xRMSNorm within <= 1 bf16 ULP."""
    torch.manual_seed(7)
    x_t = torch.randn(8, 128, dtype=torch.bfloat16)
    w_t = torch.randn(128, dtype=torch.float32) * 0.2

    up_norm = MuseSpark1xRMSNorm(
        hidden_size=128,
        eps=1e-5,
        has_weight=True,
        is_post_norm=False,
        zero_centered_gamma=True,
        post_norm_gain_center_type="one",
        residual_id=2,
    )
    up_norm.weight.data.copy_(w_t)
    ref = up_norm.forward_native(x_t)

    got = _jax_bf16_to_torch(
        centered_rms_norm(_torch_bf16_to_jax(x_t), jnp.asarray(w_t.numpy()), 1e-5)
    )
    ulp_diff = (got.view(torch.int16).int() - ref.view(torch.int16).int()).abs().max()
    assert int(ulp_diff) <= 1


def test_input_scaled_norm_matches_upstream() -> None:
    """input_scaled_rms_norm matches upstream within <= 1 bf16 ULP.

    Note: ``input_scaled_rms_norm`` intentionally rounds ``x * (1 + weight)`` to
    ``bf16`` before the weightless RMSNorm (matching the TPU megakernel prefill
    path), whereas upstream keeps the product in ``fp32``; outputs agree within
    1 bf16 ULP.
    """
    torch.manual_seed(11)
    x_t = torch.randn(8, 128, dtype=torch.bfloat16) + 0.5
    w_t = torch.randn(128, dtype=torch.float32) * 0.1

    up_norm = MuseSpark1xRMSNormWithInputScale(
        hidden_size=128,
        eps=1e-8,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
        post_norm_gain_center_type="one",
        residual_id=None,
    )
    up_norm.weight.data.copy_(w_t)
    ref = up_norm.forward_native(x_t)

    got = _jax_bf16_to_torch(
        input_scaled_rms_norm(_torch_bf16_to_jax(x_t), jnp.asarray(w_t.numpy()), 1e-8)
    )
    ulp_diff = (got.view(torch.int16).int() - ref.view(torch.int16).int()).abs().max()
    assert int(ulp_diff) <= 1


def test_ref_gmm_fp4_unpack_roundtrip() -> None:
    """Packed uint8 and unpacked float4_e2m1fn give bit-identical ref_gmm outputs."""
    rng = np.random.default_rng(19)
    experts, k_dim, n_dim = 4, 64, 32
    lhs = jnp.asarray(rng.standard_normal((6, k_dim), dtype=np.float32)).astype(
        jnp.bfloat16
    )
    w = jnp.asarray(rng.standard_normal((experts, k_dim, n_dim), dtype=np.float32))
    group_sizes = jnp.array([2, 1, 2, 1], dtype=jnp.int32)

    q_packed, scale_packed = quantize_to_native_fp4_kmajor(w, block=16, pack=True)
    q_unpacked, scale_unpacked = quantize_to_native_fp4_kmajor(w, block=16, pack=False)
    assert q_packed.dtype == jnp.uint8
    assert q_unpacked.dtype == jnp.float4_e2m1fn

    out_packed = ref_gmm(
        lhs, q_packed, scale_packed, None, group_sizes, fuse_act="silu"
    )
    out_unpacked = ref_gmm(
        lhs, q_unpacked, scale_unpacked, None, group_sizes, fuse_act="silu"
    )
    assert jnp.array_equal(out_packed, out_unpacked)

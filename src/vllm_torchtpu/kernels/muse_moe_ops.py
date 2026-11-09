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
"""Pure JAX normalization and router primitives for Muse Spark MoE."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def muse_centered_rms_norm_jax(
    x: jax.Array,
    weight: jax.Array,
    eps: float,
    *,
    batch_invariant: bool = False,
) -> jax.Array:
    """Centered RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + weight) in fp32."""
    if batch_invariant and jax.default_backend() == "tpu":
        from vllm_torchtpu.kernels.batch_invariant.rms_norm import det_centered_rms_norm

        return det_centered_rms_norm(x, weight, eps=eps, out_dtype=x.dtype)
    x_f32 = x.astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(
        jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + jnp.float32(eps)
    )
    out = (x_f32 * inv_rms) * (1.0 + weight.astype(jnp.float32))
    return out.astype(x.dtype)


def muse_input_scaled_rms_norm_jax(
    x: jax.Array,
    weight: jax.Array,
    eps: float,
    *,
    batch_invariant: bool = False,
) -> jax.Array:
    """InputScaledRMSNorm: round(x * (1 + weight), bf16) then weightless RMSNorm."""
    if batch_invariant and jax.default_backend() == "tpu":
        from vllm_torchtpu.kernels.batch_invariant.rms_norm import (
            det_input_scaled_rms_norm,
        )

        return det_input_scaled_rms_norm(x, weight, eps=eps, out_dtype=x.dtype)
    scaled = (x.astype(jnp.float32) * (1.0 + weight.astype(jnp.float32))).astype(
        x.dtype
    )
    scaled_f32 = scaled.astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(
        jnp.mean(scaled_f32 * scaled_f32, axis=-1, keepdims=True) + jnp.float32(eps)
    )
    return (scaled_f32 * inv_rms).astype(x.dtype)


def muse_route_jax(
    h: jax.Array,
    router_w: jax.Array,
    router_bias: jax.Array,
    *,
    topk: int = 8,
    output_multiplier: float = 0.17677669529663687,
    routed_scaling_factor: float = 1.0,
    router_eps: float = 1e-15,
    batch_invariant: bool = False,
    router_topk_impl: str = "rowmax",
) -> tuple[jax.Array, jax.Array]:
    """FP32 sigmoid router with bias for selection only and unbiased renorm."""
    h_f32 = h.astype(jnp.float32)
    w_g = router_w.astype(jnp.float32)
    if w_g.shape[0] != h.shape[-1] and w_g.shape[1] == h.shape[-1]:
        w_g = w_g.T
    if batch_invariant:
        from vllm_torchtpu.kernels.batch_invariant.matmul import det_matmul

        logits = det_matmul(h_f32, w_g, out_dtype=jnp.float32, highest=True)
    else:
        logits = jnp.dot(
            h_f32,
            w_g,
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )
    logits = logits * jnp.float32(output_multiplier)
    scores = jax.nn.sigmoid(logits)
    scores_for_choice = scores + router_bias.astype(jnp.float32)[None, :]

    on_tpu = jax.default_backend() == "tpu"
    if (
        on_tpu
        and scores_for_choice.shape[1] % 128 == 0
        and router_topk_impl == "rowmax"
    ):
        from vllm_torchtpu.kernels.router_topk import router_topk

        _, topk_ids = router_topk(scores_for_choice, topk)
    else:
        _, topk_ids = jax.lax.top_k(scores_for_choice, topk)

    topk_ids = topk_ids.astype(jnp.int32)
    topk_weights = jnp.take_along_axis(scores, topk_ids, axis=-1)
    topk_weights = topk_weights / (
        jnp.sum(topk_weights, axis=-1, keepdims=True) + jnp.float32(router_eps)
    )
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * jnp.float32(routed_scaling_factor)
    return topk_weights, topk_ids


__all__ = [
    "muse_centered_rms_norm_jax",
    "muse_input_scaled_rms_norm_jax",
    "muse_route_jax",
]

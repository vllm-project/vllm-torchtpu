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
"""JAX primitives for MoE layers with per-expert post-projection RMSNorm.

Some MoE architectures (e.g. Muse Spark 1x) insert a per-expert input-scaled
RMSNorm between the expert down projection (GEMM2) and the top-k routing
combine, and route on the full hidden dimension while running experts in a
compressed MoE subspace. Because the per-expert norm is nonlinear, the routing
weights cannot be folded into GEMM2 before normalization.

This module provides pure-JAX implementations of:
- ``centered_rms_norm``: ``(x * rsqrt(mean(x^2) + eps)) * (1 + weight)`` in fp32
- ``input_scaled_rms_norm``: ``round(x * (1 + weight), dtype)`` then RMSNorm
- ``sigmoid_biased_topk_route``: fp32 sigmoid top-k routing with selection-only
  score bias and unbiased renormalization
- ``post_norm_moe_jax``: end-to-end routed MoE up to the EP partial sum (before
  cross-rank all-reduce and unpooler projection)

Only single-device or expert-parallel (EP) execution is supported; TP-sharded
experts would require all-reducing ``[T * topk, moe_hidden]`` GEMM2 partials
before the nonlinear per-expert norm.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp


def centered_rms_norm(
    x: jax.Array,
    weight: jax.Array,
    eps: float,
) -> jax.Array:
    """Zero-centered RMSNorm: ``x * rsqrt(mean(x^2) + eps) * (1 + weight)``."""
    x_f32 = x.astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(
        jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + jnp.float32(eps)
    )
    out = (x_f32 * inv_rms) * (1.0 + weight.astype(jnp.float32))
    return out.astype(x.dtype)


def input_scaled_rms_norm(
    x: jax.Array,
    weight: jax.Array,
    eps: float,
) -> jax.Array:
    """Input-scaled RMSNorm with zero-centered gain.

    Scales ``x * (1 + weight)`` in fp32 and rounds to ``x.dtype`` (matching the
    TPU megakernel prefill convention, within <= 1 bf16 ULP of full-fp32
    pre-scaling), then applies weightless fp32 RMSNorm.
    """
    scaled = (x.astype(jnp.float32) * (1.0 + weight.astype(jnp.float32))).astype(
        x.dtype
    )
    scaled_f32 = scaled.astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(
        jnp.mean(scaled_f32 * scaled_f32, axis=-1, keepdims=True) + jnp.float32(eps)
    )
    return (scaled_f32 * inv_rms).astype(x.dtype)


def sigmoid_biased_topk_route(
    h: jax.Array,
    router_w_kmajor: jax.Array,
    router_bias: jax.Array,
    *,
    topk: int = 8,
    output_multiplier: float = 1.0,
    routed_scaling_factor: float = 1.0,
    router_eps: float = 1e-15,
    router_topk_impl: str = "rowmax",
) -> tuple[jax.Array, jax.Array]:
    """FP32 sigmoid router with selection-only bias and unbiased renorm.

    Args:
        h: Token activations ``[T, H]``.
        router_w_kmajor: Router weights in explicit K-major layout ``[H, E]``.
        router_bias: Selection-only expert score correction bias ``[E]``.
    """
    if router_w_kmajor.ndim != 2 or router_w_kmajor.shape[0] != h.shape[-1]:
        raise ValueError(
            "router_w_kmajor must have K-major shape [H, E] matching h.shape[-1]="
            f"{h.shape[-1]}, got {router_w_kmajor.shape}."
        )
    h_f32 = h.astype(jnp.float32)
    w_g = router_w_kmajor.astype(jnp.float32)
    # 256-expert sigmoid scores have tight gaps; a default (bf16-pass) MXU
    # matmul flips top-8 expert selections relative to upstream's fp32 router
    # (`MuseSpark1xRouter._compute_routing_native`). HIGHEST preserves exact
    # fp32 routing parity.
    logits = jnp.dot(
        h_f32,
        w_g,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )
    if output_multiplier != 1.0:
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


def post_norm_moe_jax(
    h: jax.Array,
    router_w_kmajor: jax.Array,
    router_bias: jax.Array,
    pre_arg: jax.Array,
    pre_norm_w: jax.Array,
    w13: jax.Array,
    w13_scale: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    post_e_norm_w: jax.Array,
    experts_start: jax.Array,
    *,
    topk: int = 8,
    output_multiplier: float = 1.0,
    routed_scaling_factor: float = 1.0,
    rms_norm_eps: float = 1e-5,
    post_norm_eps: float = 1e-8,
    router_eps: float = 1e-15,
    activation: str = "silu",
    use_ep: bool = True,
    use_sparse_core: bool = True,
    onehot_threshold: int = 0,
    pre_projected: bool = False,
    router_topk_impl: str = "rowmax",
    gmm: Callable[..., jax.Array] | None = None,
) -> jax.Array:
    """Post-norm MoE forward up to the local EP partial sum ``[T, moe_hidden]``.

    Args:
        h: Input hidden states ``[T, H]``.
        router_w_kmajor: K-major router weights ``[H, E]``.
        router_bias: Router score correction bias ``[E]``.
        pre_arg: Already-pooled ``[T, moe_hidden]`` activations when
            ``pre_projected=True``, else K-major pooler weights
            ``[H, moe_hidden]``.
        pre_norm_w: Pre-expert centered RMSNorm weight ``[moe_hidden]``.
        w13: Gate/up expert weights ``[E_local, K_packed, 2 * I]``.
        w13_scale: Optional quantization block scales for ``w13``.
        w2: Down expert weights ``[E_local, I_packed, moe_hidden_padded]``.
        w2_scale: Optional quantization block scales for ``w2``.
        post_e_norm_w: Per-expert input-scaled RMSNorm weight ``[moe_hidden]``.
        experts_start: Scalar int32 starting global expert index on this rank.
        gmm: Optional grouped-matmul callable for CPU unit tests. Defaults to
            ``gmm_wrapper`` on TPU and raises ``NotImplementedError`` on CPU
            when omitted.
    """
    from vllm_torchtpu.kernels.megablox.gmm_v2 import get_packing_factor
    from vllm_torchtpu.layers.core.fused_moe_gmm import (
        gmm_wrapper,
        prepare_routed_gmm_inputs,
    )

    on_tpu = jax.default_backend() == "tpu"
    if gmm is None:
        if not on_tpu:
            raise NotImplementedError(
                "post_norm_moe requires a TPU backend; tests inject a reference gmm."
            )
        gmm_fn: Callable[..., Any] = gmm_wrapper
    else:
        gmm_fn = gmm

    if w13_scale is not None and w13_scale.ndim == 3:
        w13_scale = jnp.expand_dims(w13_scale, 2)
    if w2_scale is not None and w2_scale.ndim == 3:
        w2_scale = jnp.expand_dims(w2_scale, 2)

    topk_weights, topk_ids = sigmoid_biased_topk_route(
        h,
        router_w_kmajor,
        router_bias,
        topk=topk,
        output_multiplier=output_multiplier,
        routed_scaling_factor=routed_scaling_factor,
        router_eps=router_eps,
        router_topk_impl=router_topk_impl,
    )

    if pre_projected:
        z = pre_arg.astype(h.dtype)
    else:
        if pre_arg.ndim != 2 or pre_arg.shape[0] != h.shape[-1]:
            raise ValueError(
                "pre_arg must have K-major shape [H, moe_hidden] when "
                f"pre_projected=False, got {pre_arg.shape} for H={h.shape[-1]}."
            )
        w_pre = pre_arg.astype(h.dtype)
        z = jnp.dot(h, w_pre, preferred_element_type=jnp.float32).astype(h.dtype)
    z = centered_rms_norm(z, pre_norm_w, rms_norm_eps)

    num_tokens, moe_hidden_size = z.shape
    local_num_experts, padded_hidden_size, _ = w13.shape
    packing_factor = get_packing_factor(w13.dtype, None)
    padded_hidden_size *= packing_factor

    if use_ep:
        local_ids = topk_ids - experts_start.astype(jnp.int32)
        valid = (local_ids >= 0) & (local_ids < local_num_experts)
        topk_weights_local = jnp.where(
            valid, topk_weights, jnp.zeros_like(topk_weights)
        )
        topk_ids_local = jnp.where(valid, local_ids, jnp.full_like(local_ids, -1))
    else:
        topk_weights_local = topk_weights
        topk_ids_local = topk_ids

    use_sc = use_sparse_core and on_tpu
    if use_ep and use_sc and num_tokens * topk <= onehot_threshold:
        # The one-hot permute/unpermute uses matmuls across all tokens where
        # 0 * NaN == NaN: flush non-finite values so one bad/padded row cannot
        # poison the entire batch (mirrors fused_moe_gmm.py).
        z = jnp.where(jnp.isfinite(z), z, jnp.zeros_like(z))
        topk_weights_local = jnp.where(
            jnp.isfinite(topk_weights_local),
            topk_weights_local,
            jnp.zeros_like(topk_weights_local),
        )
    (
        x,
        group_sizes,
        argsort_revert_indices,
        topk_weights_flat,
        valid_mask_flat,
        _token_indices_sorted,
    ) = prepare_routed_gmm_inputs(
        z,
        topk_ids_local,
        topk_weights_local,
        local_num_experts=local_num_experts,
        topk=topk,
        use_ep=use_ep,
        use_sparse_core=use_sc,
        onehot_moe_permute_threshold=onehot_threshold,
        skip_padded_tokens=False,
    )
    if padded_hidden_size > moe_hidden_size:
        x = jnp.pad(x, ((0, 0), (0, padded_hidden_size - moe_hidden_size)))

    group_offset = jnp.array([0], dtype=jnp.int32)
    gmm1_res = gmm_fn(
        x,
        w13,
        w13_scale,
        None,
        group_sizes,
        group_offset=group_offset,
        zero_initialize=False,
        fuse_act=activation,
        rhs_quant_dtype=None,
    )
    w2_packing_factor = get_packing_factor(w2.dtype, None)
    gmm1_res = gmm1_res[:, : w2.shape[1] * w2_packing_factor]
    gmm2_res = gmm_fn(
        gmm1_res,
        w2,
        w2_scale,
        None,
        group_sizes,
        group_offset=group_offset,
        zero_initialize=False,
        preferred_element_type=x.dtype,
        rhs_quant_dtype=None,
    )
    gmm2_res = gmm2_res[:, :moe_hidden_size]

    # Rows past group_sizes.sum() are not written by gmm_wrapper when
    # zero_initialize=False; zero them before and after the norm so NaNs cannot
    # propagate into the unpermute or weighted combine.
    valid_count = group_sizes.sum(dtype=jnp.int32)
    computed = jnp.arange(gmm2_res.shape[0], dtype=jnp.int32) < valid_count
    gmm2_res = jnp.where(computed[:, None], gmm2_res, jnp.zeros_like(gmm2_res))
    normed_rows = input_scaled_rms_norm(gmm2_res, post_e_norm_w, post_norm_eps)
    normed_rows = jnp.where(computed[:, None], normed_rows, jnp.zeros_like(normed_rows))

    token_hidden = normed_rows[argsort_revert_indices].reshape(
        (num_tokens, topk, moe_hidden_size)
    )
    valid_mask = valid_mask_flat.reshape((num_tokens, topk))
    weights_2d = topk_weights_flat.reshape((num_tokens, topk)).astype(jnp.float32)
    weighted = token_hidden.astype(jnp.float32) * weights_2d[:, :, None]
    weighted = jnp.where(valid_mask[:, :, None], weighted, 0.0)
    return jnp.sum(weighted, axis=1)


__all__ = [
    "centered_rms_norm",
    "input_scaled_rms_norm",
    "post_norm_moe_jax",
    "sigmoid_biased_topk_route",
]

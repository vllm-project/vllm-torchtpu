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
"""One-dispatch TPU MoE custom op for Meta Muse Spark Open.

Muse Spark Open inserts a per-expert InputScaledRMSNorm between the expert down
projection and the routing weight multiply, and routes on the full 8192-wide
hidden state while running experts on a 4096-wide compressed representation.
Because the per-expert norm is nonlinear, the routing weight cannot be folded
into the down projection before the norm. This op fuses the fp32 router,
top-k selection with score correction bias, pre-expert projection and centered
RMSNorm, grouped matmuls via gmm_wrapper, per-expert input-scaled RMSNorm, and
weighted combine into a single jax_op dispatch per layer.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import torch

from vllm_torchtpu.kernels.muse_moe_ops import (
    muse_centered_rms_norm_jax,
    muse_input_scaled_rms_norm_jax,
    muse_route_jax,
)

MOE_IMPL_FUSED = "fused"
MOE_IMPL_EAGER = "eager"


def _fmt_eps(val: float) -> str:
    return f"{val:.6e}".replace(".", "p").replace("-", "m").replace("+", "")


def muse_moe_impl() -> str:
    """Return the MoE implementation selected by VLLM_TPU_MUSE_MOE_IMPL."""
    import vllm_torchtpu.envs as envs

    return str(envs.VLLM_TPU_MUSE_MOE_IMPL).lower()


def _gmm_cpu_fallback(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    fuse_act: str | None = None,
    preferred_element_type: Any = None,
) -> jax.Array:
    """Pure JAX grouped matmul for CPU unit testing where Pallas TPU is unavailable."""
    if rhs.dtype == jnp.uint8:
        rhs_val = jax.lax.bitcast_convert_type(rhs, jnp.float4_e2m1fn).reshape(
            rhs.shape[0], -1, rhs.shape[-1]
        )
    else:
        rhs_val = rhs
    rhs_f32 = rhs_val.astype(jnp.float32)
    if rhs_scale is not None:
        s = rhs_scale.astype(jnp.float32).reshape(
            rhs_f32.shape[0], -1, 1, rhs_f32.shape[-1]
        )
        block_size = rhs_f32.shape[1] // s.shape[1]
        rhs_f32 = (
            rhs_f32.reshape(rhs_f32.shape[0], s.shape[1], block_size, rhs_f32.shape[2])
            * s
        ).reshape(rhs_f32.shape)

    ends = jnp.cumsum(group_sizes, dtype=jnp.int32)
    starts = ends - group_sizes
    row_idx = jnp.arange(lhs.shape[0], dtype=jnp.int32)[:, None]
    in_group = (row_idx >= starts[None, :]) & (row_idx < ends[None, :])
    group_id = jnp.argmax(in_group, axis=1)
    valid_row = jnp.any(in_group, axis=1)

    w_row = rhs_f32[group_id]
    acc = jnp.einsum(
        "mk,mkn->mn",
        lhs.astype(jnp.float32),
        w_row,
        precision=jax.lax.Precision.HIGHEST,
    )
    if rhs_bias is not None:
        b_row = rhs_bias.astype(jnp.float32).reshape(rhs_f32.shape[0], -1)[group_id]
        acc = acc + b_row
    acc = jnp.where(valid_row[:, None], acc, 0.0)

    out_dtype = (
        preferred_element_type if preferred_element_type is not None else lhs.dtype
    )
    if fuse_act is not None:
        gate, up = jnp.split(acc, 2, axis=-1)
        if fuse_act == "silu":
            acc = jax.nn.silu(gate) * up
        elif fuse_act == "gelu":
            acc = jax.nn.gelu(gate) * up
        elif fuse_act == "gelu_tanh":
            acc = jax.nn.gelu(gate, approximate=True) * up
        else:
            raise NotImplementedError(f"Unsupported activation: {fuse_act}")
    return acc.astype(out_dtype)


def _muse_moe_jax(
    h: jax.Array,
    router_w: jax.Array,
    router_bias: jax.Array,
    pre_w: jax.Array,
    pre_norm_w: jax.Array,
    w13: jax.Array,
    w13_scale: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    post_e_norm_w: jax.Array,
    experts_start: jax.Array,
    *,
    topk: int = 8,
    output_multiplier: float = 0.17677669529663687,
    routed_scaling_factor: float = 1.0,
    rms_norm_eps: float = 1e-5,
    post_norm_eps: float = 1e-8,
    router_eps: float = 1e-15,
    activation: str = "silu",
    use_ep: bool = True,
    use_sparse_core: bool = True,
    onehot_threshold: int = 0,
    batch_invariant: bool = False,
    pre_projected: bool = False,
    router_topk_impl: str = "rowmax",
) -> jax.Array:
    """Muse Spark MoE forward up to the EP partial sum (before post_expert_proj).

    batch_invariant: every matmul uses an M-independent K tiling, so a token's
    result does not depend on the other tokens in the step.
    """
    from vllm_torchtpu.kernels.batch_invariant.gmm_tiling import batch_invariant_tiling
    from vllm_torchtpu.kernels.megablox.gmm_v2 import (
        calculate_tiling,
        get_packing_factor,
    )
    from vllm_torchtpu.layers.core.fused_moe_gmm import (
        gmm_wrapper,
        prepare_routed_gmm_inputs,
    )

    on_tpu = jax.default_backend() == "tpu"
    orig_tokens = h.shape[0]
    if batch_invariant and on_tpu and orig_tokens < 8:
        h = jnp.pad(h, ((0, 8 - orig_tokens), (0, 0)))

    if w13_scale is not None and w13_scale.ndim == 3:
        w13_scale = jnp.expand_dims(w13_scale, 2)
    if w2_scale is not None and w2_scale.ndim == 3:
        w2_scale = jnp.expand_dims(w2_scale, 2)

    topk_weights, topk_ids = muse_route_jax(
        h,
        router_w,
        router_bias,
        topk=topk,
        output_multiplier=output_multiplier,
        routed_scaling_factor=routed_scaling_factor,
        router_eps=router_eps,
        batch_invariant=batch_invariant,
        router_topk_impl=router_topk_impl,
    )

    if pre_projected:
        z = pre_w.astype(h.dtype)
        if orig_tokens < h.shape[0]:
            z = jnp.pad(z, ((0, h.shape[0] - orig_tokens), (0, 0)))
    else:
        w_pre = pre_w.astype(h.dtype)
        if w_pre.shape[0] != h.shape[-1] and w_pre.shape[1] == h.shape[-1]:
            w_pre = w_pre.T
        if batch_invariant:
            from vllm_torchtpu.kernels.batch_invariant.matmul import det_matmul

            z = det_matmul(h, w_pre)
        else:
            z = jnp.dot(h, w_pre, preferred_element_type=jnp.float32).astype(h.dtype)
    z = muse_centered_rms_norm_jax(
        z, pre_norm_w, rms_norm_eps, batch_invariant=batch_invariant
    )

    num_tokens, moe_hidden_size = z.shape
    local_num_experts, padded_hidden_size, _ = w13.shape
    packing_factor = get_packing_factor(w13.dtype, None)
    padded_hidden_size *= packing_factor

    if use_ep and experts_start is not None:
        local_ids = topk_ids - experts_start.astype(jnp.int32)
        valid = (local_ids >= 0) & (local_ids < local_num_experts)
        topk_weights_local = jnp.where(
            valid, topk_weights, jnp.zeros_like(topk_weights)
        )
        topk_ids_local = jnp.where(valid, local_ids, jnp.full_like(local_ids, -1))
    else:
        topk_weights_local = topk_weights
        topk_ids_local = topk_ids

    use_sc = (
        use_sparse_core
        and on_tpu
        and (not batch_invariant or num_tokens * topk > onehot_threshold)
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
    if on_tpu:
        tile_fn = batch_invariant_tiling if batch_invariant else calculate_tiling
        gmm1_res = gmm_wrapper(
            x,
            w13,
            w13_scale,
            None,
            group_sizes,
            group_offset=group_offset,
            zero_initialize=False,
            fuse_act=activation,
            rhs_quant_dtype=None,
            tile_info=tile_fn,
        )
        w2_packing_factor = get_packing_factor(w2.dtype, None)
        gmm1_res = gmm1_res[:, : w2.shape[1] * w2_packing_factor]
        gmm2_res = gmm_wrapper(
            gmm1_res,
            w2,
            w2_scale,
            None,
            group_sizes,
            group_offset=group_offset,
            zero_initialize=False,
            preferred_element_type=x.dtype,
            rhs_quant_dtype=None,
            tile_info=tile_fn,
        )
    else:
        gmm1_res = _gmm_cpu_fallback(
            x,
            w13,
            w13_scale,
            None,
            group_sizes,
            fuse_act=activation,
            preferred_element_type=x.dtype,
        )
        w2_packing_factor = get_packing_factor(w2.dtype, None)
        gmm1_res = gmm1_res[:, : w2.shape[1] * w2_packing_factor]
        gmm2_res = _gmm_cpu_fallback(
            gmm1_res,
            w2,
            w2_scale,
            None,
            group_sizes,
            fuse_act=None,
            preferred_element_type=x.dtype,
        )

    gmm2_res = gmm2_res[:, :moe_hidden_size]

    # Rows past group_sizes.sum() are not written by gmm_wrapper when
    # zero_initialize=False; zero them before the norm so NaNs cannot propagate.
    valid_count = group_sizes.sum(dtype=jnp.int32)
    if batch_invariant and on_tpu:
        from vllm_torchtpu.kernels.batch_invariant.rms_norm import (
            det_input_scaled_rms_norm_masked,
        )

        # Under EP most routed rows belong to other ranks; the masked norm
        # skips their tiles. Rows past valid_count are undefined, and the
        # valid_mask select below drops them.
        normed_rows = det_input_scaled_rms_norm_masked(
            gmm2_res,
            post_e_norm_w,
            valid_count,
            eps=post_norm_eps,
            out_dtype=gmm2_res.dtype,
        )
    else:
        computed = jnp.arange(gmm2_res.shape[0], dtype=jnp.int32) < valid_count
        gmm2_res = jnp.where(computed[:, None], gmm2_res, jnp.zeros_like(gmm2_res))
        normed_rows = muse_input_scaled_rms_norm_jax(
            gmm2_res,
            post_e_norm_w,
            post_norm_eps,
            batch_invariant=batch_invariant,
        )
        normed_rows = jnp.where(
            computed[:, None], normed_rows, jnp.zeros_like(normed_rows)
        )

    token_hidden = normed_rows[argsort_revert_indices].reshape(
        (num_tokens, topk, moe_hidden_size)
    )
    valid_mask = valid_mask_flat.reshape((num_tokens, topk))
    weights_2d = topk_weights_flat.reshape((num_tokens, topk)).astype(jnp.float32)
    weighted = token_hidden.astype(jnp.float32) * weights_2d[:, :, None]
    weighted = jnp.where(valid_mask[:, :, None], weighted, 0.0)
    out = jnp.sum(weighted, axis=1)
    return out[:orig_tokens] if orig_tokens < num_tokens else out


_FUSED_MUSE_MOE_OPS: dict[tuple[Any, ...], Callable] = {}


def get_muse_fused_moe_op(
    *,
    topk: int = 8,
    output_multiplier: float = 0.17677669529663687,
    routed_scaling_factor: float = 1.0,
    rms_norm_eps: float = 1e-5,
    post_norm_eps: float = 1e-8,
    router_eps: float = 1e-15,
    activation: str = "silu",
    use_ep: bool = True,
    pre_projected: bool = False,
) -> Callable:
    """Return a cached jax_op for the given static Muse Spark MoE configuration."""
    import vllm_torchtpu.envs as envs
    from vllm_torchtpu.layers.core.fused_moe_gmm import (
        resolve_onehot_permute_threshold,
    )

    use_sparse_core = bool(envs.USE_MOE_SPARSE_CORE)
    onehot = int(resolve_onehot_permute_threshold())
    batch_invariant = bool(envs.VLLM_TPU_MUSE_BATCH_INVARIANT)
    router_topk_impl = str(envs.TPU_MOE_ROUTER_TOPK)
    key = (
        batch_invariant,
        router_topk_impl,
        pre_projected,
        topk,
        output_multiplier,
        routed_scaling_factor,
        rms_norm_eps,
        post_norm_eps,
        router_eps,
        activation,
        use_ep,
        use_sparse_core,
        onehot,
    )
    op = _FUSED_MUSE_MOE_OPS.get(key)
    if op is not None:
        return op

    from torch_tpu._internal import pallas

    name = (
        (
            f"muse::fused_moe_k{topk}_m{output_multiplier:.6g}_s{routed_scaling_factor:g}"
            f"_e1_{_fmt_eps(rms_norm_eps)}_e2_{_fmt_eps(post_norm_eps)}_er_{_fmt_eps(router_eps)}"
            f"_{activation}_ep{int(use_ep)}_sc{int(use_sparse_core)}_oh{onehot}"
            f"_bi{int(batch_invariant)}_pp{int(pre_projected)}_rt{router_topk_impl}"
        )
        .replace(".", "p")
        .replace("-", "m")
    )

    op = pallas.jax_op(
        name,
        functools.partial(
            _muse_moe_jax,
            topk=topk,
            output_multiplier=output_multiplier,
            routed_scaling_factor=routed_scaling_factor,
            rms_norm_eps=rms_norm_eps,
            post_norm_eps=post_norm_eps,
            router_eps=router_eps,
            activation=activation,
            use_ep=use_ep,
            use_sparse_core=use_sparse_core,
            onehot_threshold=onehot,
            batch_invariant=batch_invariant,
            pre_projected=pre_projected,
            router_topk_impl=router_topk_impl,
        ),
    )

    def _fake(
        h: torch.Tensor,
        router_w: torch.Tensor,
        router_bias: torch.Tensor,
        pre_w: torch.Tensor,
        pre_norm_w: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return torch.empty(
            (h.shape[0], pre_norm_w.shape[0]),
            dtype=torch.float32,
            device=h.device,
        )

    op.register_fake(_fake)
    _FUSED_MUSE_MOE_OPS[key] = op
    return op


class MuseFusedMoeArgs:
    """Resolved operands of one MuseSparkMoE layer for the fused op."""

    def __init__(self, moe: Any) -> None:
        from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU
        from vllm_torchtpu.layers.adapter.quantization.nvfp4 import VllmNvfp4MoEMethod
        from vllm_torchtpu.layers.adapter.quantization.unquantized import (
            VllmUnquantizedFusedMoEMethod,
        )

        routed = moe.experts.routed_experts
        if routed.w13_weight.numel() == 0:
            self.unsupported = "routed expert weights are empty"
            self.op: Callable | None = None
            return
        method = routed.quant_method
        reasons: list[str] = []

        w13_scale: torch.Tensor | None = None
        w2_scale: torch.Tensor | None = None
        activation: str | None = None
        if isinstance(method, VllmUnquantizedFusedMoEMethod):
            w13_scale = None
            w2_scale = None
            activation = method._tpu_activation_str
        elif isinstance(method, VllmNvfp4MoEMethod):
            w13_scale = getattr(routed, "w13_weight_scale", None)
            w2_scale = getattr(routed, "w2_weight_scale", None)
            if w13_scale is None or w2_scale is None:
                reasons.append("NVFP4 missing weight_scale")
            activation = method._tpu_activation_str
        elif isinstance(method, VllmFp8MoEMethodTPU):
            w13_scale = getattr(routed, "w13_weight_scale_inv", None)
            w2_scale = getattr(routed, "w2_weight_scale_inv", None)
            if w13_scale is None or w2_scale is None:
                reasons.append("FP8 missing weight_scale_inv")
            activation = method._tpu_activation_str
        else:
            reasons.append(f"unsupported routed method {type(method).__name__}")

        if not reasons and activation is None:
            raise RuntimeError(
                "MuseFusedMoeArgs constructed before process_weights_after_loading "
                "initialized _tpu_activation_str."
            )

        use_ep = bool(routed.moe_config.moe_parallel_config.use_ep)
        experts_start: torch.Tensor | None = (
            getattr(routed, "_experts_start", None) if use_ep else None
        )
        if use_ep and experts_start is None:
            reasons.append("EP missing _experts_start")

        self.unsupported = "; ".join(reasons)
        self.op = None
        if self.unsupported:
            return
        assert activation is not None

        self.pre_projected = int(moe.pre_expert_proj.tp_size) > 1
        self._op_kwargs = dict(
            topk=int(moe.top_k),
            output_multiplier=float(moe.output_multiplier),
            routed_scaling_factor=float(moe.routed_scaling_factor),
            rms_norm_eps=float(moe.rms_norm_eps),
            post_norm_eps=float(moe.post_norm_eps),
            router_eps=float(moe.router_eps),
            activation=activation,
            use_ep=use_ep,
            pre_projected=self.pre_projected,
        )
        self.op = get_muse_fused_moe_op(**self._op_kwargs)
        device = routed.w13_weight.device
        if experts_start is None:
            experts_start = torch.zeros((), dtype=torch.int32, device=device)
        self._head_tensors = (
            moe.router_w_kmajor,
            moe.gate.e_score_correction_bias,
        )
        self._tail_tensors = (
            moe.pre_expert_norm.weight,
            routed.w13_weight,
            w13_scale,
            routed.w2_weight,
            w2_scale,
            moe.post_expert_norm.weight,
            experts_start,
        )
        self._pre_w = moe.pre_expert_proj.weight

    def __call__(
        self, h: torch.Tensor, z_pre: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert self.op is not None, self.unsupported
        pre_arg = z_pre if self.pre_projected else self._pre_w
        assert pre_arg is not None
        return self.op(h, *self._head_tensors, pre_arg, *self._tail_tensors)

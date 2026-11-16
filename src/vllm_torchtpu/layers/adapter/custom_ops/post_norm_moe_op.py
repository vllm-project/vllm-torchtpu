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
"""PyTorch custom-op bridge for MoE layers with per-expert post-norm on TPU.

Wraps :func:`vllm_torchtpu.layers.core.post_norm_moe.post_norm_moe_jax` in a
cached ``pallas.jax_op`` and resolves ``RoutedExperts`` quantization tensors
(BF16, ModelOpt NVFP4, or FP8) after per-layer weight finalization.

Only single-device or expert-parallel (EP) execution is supported; TP-sharded
experts and static TPU expert placement (``TPU_EXPERT_PLACEMENT``) are rejected
at build time.
"""

from __future__ import annotations

import dataclasses
import functools
import re
import struct
from collections.abc import Callable
from typing import Any

import torch

from vllm_torchtpu.layers.core.post_norm_moe import post_norm_moe_jax

_OP_NAME_RE = re.compile(r"^[a-z_]+::[A-Za-z0-9_]+$")


def _f64_tag(value: float) -> str:
    """Lossless, schema-safe 16-hex-char big-endian IEEE-754 float64 tag."""
    return struct.pack(">d", float(value)).hex()


@dataclasses.dataclass(frozen=True)
class PostNormMoeStatic:
    """Static scalar/configuration parameters keying the compiled custom op."""

    topk: int
    output_multiplier: float
    routed_scaling_factor: float
    rms_norm_eps: float
    post_norm_eps: float
    router_eps: float
    activation: str
    use_ep: bool
    pre_projected: bool


_POST_NORM_MOE_OPS: dict[tuple[Any, ...], Callable[..., torch.Tensor]] = {}


def _format_op_name(
    static: PostNormMoeStatic,
    *,
    use_sparse_core: bool,
    onehot_threshold: int,
    router_topk_impl: str,
) -> str:
    rt_clean = re.sub(r"[^A-Za-z0-9_]", "_", router_topk_impl)
    act_clean = re.sub(r"[^A-Za-z0-9_]", "_", static.activation)
    oh_tag = (
        f"m{abs(onehot_threshold)}" if onehot_threshold < 0 else str(onehot_threshold)
    )
    name = (
        f"tpu::post_norm_moe_k{static.topk}"
        f"_m{_f64_tag(static.output_multiplier)}"
        f"_s{_f64_tag(static.routed_scaling_factor)}"
        f"_e1_{_f64_tag(static.rms_norm_eps)}"
        f"_e2_{_f64_tag(static.post_norm_eps)}"
        f"_er_{_f64_tag(static.router_eps)}"
        f"_{act_clean}_ep{int(static.use_ep)}"
        f"_sc{int(use_sparse_core)}_oh{oh_tag}"
        f"_pp{int(static.pre_projected)}_rt_{rt_clean}"
    )
    assert _OP_NAME_RE.fullmatch(name), f"Invalid custom op schema name: {name!r}"
    return name


def get_post_norm_moe_op(static: PostNormMoeStatic) -> Callable[..., torch.Tensor]:
    """Return a cached ``pallas.jax_op`` for ``static``."""
    import vllm_torchtpu.envs as envs
    from vllm_torchtpu.layers.core.fused_moe_gmm import (
        resolve_onehot_permute_threshold,
    )

    use_sparse_core = bool(envs.USE_MOE_SPARSE_CORE)
    onehot = int(resolve_onehot_permute_threshold())
    router_topk_impl = str(envs.TPU_MOE_ROUTER_TOPK)
    key = (static, use_sparse_core, onehot, router_topk_impl)
    cached = _POST_NORM_MOE_OPS.get(key)
    if cached is not None:
        return cached

    from torch_tpu._internal import pallas

    name = _format_op_name(
        static,
        use_sparse_core=use_sparse_core,
        onehot_threshold=onehot,
        router_topk_impl=router_topk_impl,
    )
    op = pallas.jax_op(
        name,
        functools.partial(
            post_norm_moe_jax,
            topk=static.topk,
            output_multiplier=static.output_multiplier,
            routed_scaling_factor=static.routed_scaling_factor,
            rms_norm_eps=static.rms_norm_eps,
            post_norm_eps=static.post_norm_eps,
            router_eps=static.router_eps,
            activation=static.activation,
            use_ep=static.use_ep,
            use_sparse_core=use_sparse_core,
            onehot_threshold=onehot,
            pre_projected=static.pre_projected,
            router_topk_impl=router_topk_impl,
            gmm=None,
        ),
    )

    def _fake(
        h: torch.Tensor,
        router_w_kmajor: torch.Tensor,
        router_bias: torch.Tensor,
        pre_arg: torch.Tensor,
        pre_norm_w: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del router_w_kmajor, router_bias, pre_arg, args, kwargs
        return torch.empty(
            (h.shape[0], pre_norm_w.shape[0]),
            dtype=torch.float32,
            device=h.device,
        )

    op.register_fake(_fake)
    _POST_NORM_MOE_OPS[key] = op
    return op


class PostNormMoe:
    """Bound post-norm MoE callable for one finalized ``RoutedExperts`` layer."""

    def __init__(
        self,
        op: Callable[..., torch.Tensor],
        *,
        pre_projected: bool,
        router_w_kmajor: torch.Tensor,
        router_bias: torch.Tensor,
        pre_w: torch.Tensor | None,
        pre_norm_w: torch.Tensor,
        w13_weight: torch.Tensor,
        w13_scale: torch.Tensor | None,
        w2_weight: torch.Tensor,
        w2_scale: torch.Tensor | None,
        post_norm_w: torch.Tensor,
        experts_start: torch.Tensor,
    ) -> None:
        self.op = op
        self.pre_projected = pre_projected
        self._pre_w = pre_w
        self._head_tensors = (router_w_kmajor, router_bias)
        self._tail_tensors = (
            pre_norm_w,
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
            post_norm_w,
            experts_start,
        )

    def __call__(
        self, h: torch.Tensor, z_pre: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.pre_projected:
            if z_pre is None:
                raise ValueError("PostNormMoe(pre_projected=True) requires z_pre.")
            pre_arg = z_pre
        else:
            if self._pre_w is None:
                raise RuntimeError("PostNormMoe(pre_projected=False) missing pre_w.")
            pre_arg = self._pre_w
        return self.op(h, *self._head_tensors, pre_arg, *self._tail_tensors)


def build_post_norm_moe(
    routed: Any,
    *,
    static: PostNormMoeStatic,
    router_w_kmajor: torch.Tensor,
    router_bias: torch.Tensor,
    pre_w: torch.Tensor | None,
    pre_norm_w: torch.Tensor,
    post_norm_w: torch.Tensor,
    hidden_size: int | None = None,
) -> PostNormMoe:
    """Resolve finalized ``routed`` operands and bind a :class:`PostNormMoe`.

    Raises if ``routed`` is unfinalized, uses an unsupported quantization
    method, is TP-sharded across experts, or has ``TPU_EXPERT_PLACEMENT``
    applied (since ``post_norm_moe_jax`` computes routing from its own router
    operand rather than vLLM's renumbered slot router).
    """
    from vllm_torchtpu.layers.adapter import expert_placement
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU
    from vllm_torchtpu.layers.adapter.quantization.nvfp4 import VllmNvfp4MoEMethod
    from vllm_torchtpu.layers.adapter.quantization.unquantized import (
        VllmUnquantizedFusedMoEMethod,
    )

    if expert_placement.expert_order(routed) is not None:
        raise NotImplementedError(
            "post_norm_moe does not support TPU_EXPERT_PLACEMENT: the fused op "
            "routes directly from router_w_kmajor and cannot honor renumbered "
            "expert slots."
        )

    mpc = routed.moe_config.moe_parallel_config
    if int(mpc.tp_size) > 1:
        raise NotImplementedError(
            "post_norm_moe requires single-device or expert-parallel (EP) "
            f"experts (got moe_parallel_config.tp_size={mpc.tp_size}); "
            "TP-sharded experts cannot apply the nonlinear per-expert post-norm "
            "without an intermediate all-reduce over [T * topk, moe_hidden]."
        )

    expected_experts = int(router_bias.numel())
    if router_w_kmajor.ndim != 2 or router_w_kmajor.shape[1] != expected_experts:
        raise ValueError(
            "router_w_kmajor must have K-major shape [H, E] with E="
            f"{expected_experts}, got {tuple(router_w_kmajor.shape)}."
        )
    if hidden_size is not None and router_w_kmajor.shape[0] != hidden_size:
        raise ValueError(
            f"router_w_kmajor must have shape [{hidden_size}, {expected_experts}], "
            f"got {tuple(router_w_kmajor.shape)}."
        )
    if router_w_kmajor.dtype != torch.float32:
        raise ValueError(
            f"router_w_kmajor must be float32, got {router_w_kmajor.dtype}."
        )

    moe_hidden = int(pre_norm_w.numel())
    if not static.pre_projected:
        if pre_w is None:
            raise ValueError("pre_w is required when static.pre_projected=False.")
        expected_pre = (router_w_kmajor.shape[0], moe_hidden)
        if pre_w.ndim != 2 or tuple(pre_w.shape) != expected_pre:
            raise ValueError(
                f"pre_w must have K-major shape {expected_pre}, "
                f"got {tuple(pre_w.shape)}."
            )

    if routed.w13_weight.numel() == 0:
        raise RuntimeError("Cannot build PostNormMoe: routed.w13_weight is empty.")

    method = routed.quant_method
    w13_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None
    if isinstance(method, VllmUnquantizedFusedMoEMethod):
        # FusedMoE registers w13_bias/w2_bias only when the layer has expert
        # biases; the fused op has no bias operands, so reject them rather
        # than silently dropping them.
        if (
            getattr(routed, "w13_bias", None) is not None
            or getattr(routed, "w2_bias", None) is not None
        ):
            raise NotImplementedError(
                "post_norm_moe does not support routed expert biases "
                "(w13_bias/w2_bias)."
            )
        activation = method._tpu_activation_str
    elif isinstance(method, VllmNvfp4MoEMethod):
        w13_scale = getattr(routed, "w13_weight_scale", None)
        w2_scale = getattr(routed, "w2_weight_scale", None)
        if w13_scale is None or w2_scale is None:
            raise RuntimeError("NVFP4 RoutedExperts missing w13/w2_weight_scale.")
        activation = method._tpu_activation_str
    elif isinstance(method, VllmFp8MoEMethodTPU):
        w13_scale = getattr(routed, "w13_weight_scale_inv", None)
        w2_scale = getattr(routed, "w2_weight_scale_inv", None)
        if w13_scale is None or w2_scale is None:
            raise RuntimeError("FP8 RoutedExperts missing w13/w2_weight_scale_inv.")
        activation = method._tpu_activation_str
    else:
        raise NotImplementedError(
            f"Unsupported routed MoE quant method for post_norm_moe: "
            f"{type(method).__name__}."
        )

    if activation is None:
        raise RuntimeError(
            "build_post_norm_moe called before RoutedExperts "
            "process_weights_after_loading initialized _tpu_activation_str."
        )

    resolved_static = dataclasses.replace(static, activation=str(activation))
    use_ep = bool(mpc.use_ep)
    if static.use_ep != use_ep:
        raise ValueError(
            f"PostNormMoeStatic.use_ep={static.use_ep} does not match "
            f"routed.moe_config.moe_parallel_config.use_ep={use_ep}."
        )
    experts_start: torch.Tensor | None = (
        getattr(routed, "_experts_start", None) if use_ep else None
    )
    if use_ep and experts_start is None:
        raise RuntimeError("EP RoutedExperts missing _experts_start tensor.")
    if experts_start is None:
        experts_start = torch.zeros(
            (), dtype=torch.int32, device=routed.w13_weight.device
        )

    op = get_post_norm_moe_op(resolved_static)
    return PostNormMoe(
        op,
        pre_projected=resolved_static.pre_projected,
        router_w_kmajor=router_w_kmajor,
        router_bias=router_bias,
        pre_w=pre_w,
        pre_norm_w=pre_norm_w,
        w13_weight=routed.w13_weight,
        w13_scale=w13_scale,
        w2_weight=routed.w2_weight,
        w2_scale=w2_scale,
        post_norm_w=post_norm_w,
        experts_start=experts_start,
    )


__all__ = [
    "PostNormMoe",
    "PostNormMoeStatic",
    "_f64_tag",
    "build_post_norm_moe",
    "get_post_norm_moe_op",
]

"""TPU mHC (Manifold-Constrained Hyper-Connections) + hybrid decoder, GLM 5.3 Flash.

In GLM 5.3 Flash (`Glm5NextTextDecoderLayer`, `modeling_glm5_next.py:219-296,
1259-1330`), every one of the 45 decoder layers wraps both its attention block
(34 KDA linear-attention layers + 11 MLA layers) and its MoE/MLP block in a
Manifold-Constrained Hyper-Connection (`Glm5NextTextHyperConnection`, `hc_mult=4`,
`hc_sinkhorn_iters=20`).

## Two TPU hazards in the reference mHC implementation

1. **Precision & MXU tile waste in stream combination (`modeling_glm5_next.py:1316`)**:
   The reference combines the `H=4` residual streams via
   `torch.matmul(comb.transpose(-1, -2), residual)` where `comb` is `[N, 4, 4]`
   and `residual` is `[N, 4, 4096]`. On TPU:
     * `torch.matmul` dispatches to the 128x128 MXU, padding the 4x4 mixer to
       128x128 (99.9% wasted MXU slots).
     * Even with `float32` tensors, default XLA matmul precision downcasts
       operands to `bfloat16` MXU passes (~2e-3 relative error), breaking the
       doubly-stochastic conservation law that 20 iterations of Sinkhorn-Knopp
       just established.
   Replacing `torch.matmul` with an explicit FP32 VPU reduction over the 4-wide
   stream axis keeps the combine on the VPU in true FP32 (~1e-7 relative error).

2. **Op-dispatch storm across 90 mHC sites per token**:
   20 Sinkhorn iterations (40 small reductions + divisions on `[N, 4, 4]`),
   unweighted RMSNorm, stream collapse, and the subsequent `input_layernorm` /
   `post_attention_layernorm` amount to ~50 tiny ops per mHC site -- 4,500 op
   dispatches per forward pass across 45 layers. Fusing `mHC collapse +
   sublayer RMSNorm` and `mHC stream combine` under `torch.compile(backend="tpu")`
   collapses the entire Sinkhorn loop + stream reduction + RMSNorm into a single
   XLA fusion.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F
from torch import nn

try:
    from torch_tpu._internal import pallas as _tpu_pallas
except ImportError:
    _tpu_pallas = None


def _mhc_mix_logits_jax(
    flat_normed: jax.Array,  # [..., hc_mult * hidden_size] float32
    fn_weight: jax.Array,  # [(2 + hc_mult) * hc_mult, hc_mult * hidden_size] float32
) -> jax.Array:
    """3-pass IEEE FP32 MXU contraction `[..., K] x [K, M] -> [..., M]`."""
    return jax.lax.dot_general(
        flat_normed.astype(jnp.float32),
        fn_weight.astype(jnp.float32).T,
        dimension_numbers=(((flat_normed.ndim - 1,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


_mhc_mix_logits_op = (
    _tpu_pallas.jax_op("glm53_mhc::mix_logits", _mhc_mix_logits_jax)
    if _tpu_pallas is not None
    else None
)


def compute_mhc_mix_logits(
    flat_normed: torch.Tensor, fn_weight: torch.Tensor
) -> torch.Tensor:
    """Computes `mix_logits` in full FP32 precision.

    `[..., 16384] x [16384, 24] -> [..., 24]`.

    Uses 3-pass MXU `dot_general` (`precision=jax.lax.Precision.HIGHEST`) to avoid
    1-pass BF16 downcast errors while eliminating the 3.22 GB VPU broadcast.
    """
    if flat_normed.device.type == "tpu" and _mhc_mix_logits_op is not None:
        return _mhc_mix_logits_op(flat_normed.float(), fn_weight.float())
    return F.linear(flat_normed.float(), fn_weight.float())


def _mhc_collapse_and_rmsnorm_fn(
    hidden_streams: torch.Tensor,  # [..., hc_mult, hidden_size]
    fn_weight: torch.Tensor,  # [(2 + hc_mult) * hc_mult, hc_mult * hidden_size]
    base: torch.Tensor,  # [(2 + hc_mult) * hc_mult]
    scale: torch.Tensor,  # [3]
    sublayer_norm_weight: torch.Tensor,  # [hidden_size]
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
    rms_norm_eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused mHC Sinkhorn projection + stream collapse + sublayer RMSNorm."""
    orig_dtype = hidden_streams.dtype
    streams_f32 = hidden_streams.float()
    flat = streams_f32.flatten(start_dim=-2)

    # Unweighted RMSNorm on the flattened streams (`Glm5NextTextUnweightedRMSNorm`).
    flat_normed = flat * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + rms_norm_eps
    )

    # Lower the wide K=16,384 contraction onto the MXU in IEEE FP32 precision
    # (3-pass dot_general) to preserve the Sinkhorn-Knopp doubly-stochastic
    # invariant without materializing a [N, 24, 16384] VPU broadcast tensor.
    mix_logits = compute_mhc_mix_logits(flat_normed, fn_weight)
    pre_w, post_w, comb_w = mix_logits.split(
        [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
    )
    pre_b, post_b, comb_b = base.float().split([hc_mult, hc_mult, hc_mult * hc_mult])
    scale_f32 = scale.float()
    pre_scale, post_scale, comb_scale = (
        scale_f32[0],
        scale_f32[1],
        scale_f32[2],
    )

    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + hc_eps
    post = 2.0 * torch.sigmoid(post_w * post_scale + post_b)

    comb_logits = comb_w.view(
        *comb_w.shape[:-1], hc_mult, hc_mult
    ) * comb_scale + comb_b.view(hc_mult, hc_mult)
    comb = torch.softmax(comb_logits, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)

    # Collapse `hc_mult` streams -> single `[..., hidden_size]` vector in FP32
    # on the VPU, then immediately apply the sublayer RMSNorm before casting back.
    collapsed = (pre.unsqueeze(-1) * streams_f32).sum(dim=-2)
    var = collapsed.square().mean(dim=-1, keepdim=True)
    normed = collapsed * torch.rsqrt(var + rms_norm_eps) * sublayer_norm_weight.float()
    return post, comb, normed.to(orig_dtype)


def _mhc_combine_streams_fn(
    residual_streams: torch.Tensor,  # [..., hc_mult, hidden_size]
    sublayer_out: torch.Tensor,  # [..., hidden_size]
    post: torch.Tensor,  # [..., hc_mult]
    comb: torch.Tensor,  # [..., hc_mult, hc_mult]
) -> torch.Tensor:
    """VPU-native FP32 stream expansion & doubly-stochastic residual combine.

    Avoids `torch.matmul(comb.transpose(-1, -2), residual)` which on TPU routes a
    4x4 matrix to the 128x128 MXU and downcasts FP32 inputs to BF16 passes.
    Computes
    `post[..., h, None] * sublayer_out[..., None, :]
    + sum_i comb[..., i, h] * residual[..., i, :]`
    directly on the VPU in FP32.
    """
    orig_dtype = residual_streams.dtype
    res_f32 = residual_streams.float()
    sub_f32 = sublayer_out.float()
    # comb^T @ residual along the hc_mult=4 axis:
    # comb.transpose(-1, -2) has shape [..., H_out, H_in]
    mixed_res = (
        comb.float().transpose(-1, -2).unsqueeze(-1) * res_f32.unsqueeze(-3)
    ).sum(dim=-2)
    combined = post.float().unsqueeze(-1) * sub_f32.unsqueeze(-2) + mixed_res
    return combined.to(orig_dtype)


_COMPILED_MHC_COLLAPSE_CACHE: dict[tuple[Any, ...], Any] = {}
_COMPILED_MHC_COMBINE_CACHE: dict[tuple[Any, ...], Any] = {}
if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
    torch._dynamo.config.cache_size_limit = max(
        getattr(torch._dynamo.config, "cache_size_limit", 8), 64
    )


def mhc_collapse_and_rmsnorm(
    hidden_streams: torch.Tensor,  # [..., hc_mult, hidden_size]
    fn_weight: torch.Tensor,  # [(2 + hc_mult) * hc_mult, hc_mult * hidden_size]
    base: torch.Tensor,  # [(2 + hc_mult) * hc_mult]
    scale: torch.Tensor,  # [3]
    sublayer_norm_weight: torch.Tensor,  # [hidden_size]
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
    rms_norm_eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused mHC Sinkhorn projection + stream collapse + sublayer RMSNorm.

    Compiled via `torch.compile(backend="tpu")` on TPU so XLA fuses the `bf16 -> f32`
    cast, Sinkhorn loop, and orthogonal stream reductions into a single execution
    graph without HBM layout transpositions (`{0,2,1}` vs `{2,1,0}`).
    """
    if hidden_streams.device.type == "tpu" and not torch.compiler.is_compiling():
        key = (
            tuple(hidden_streams.shape),
            hidden_streams.dtype,
            hc_mult,
            sinkhorn_iters,
            hc_eps,
            rms_norm_eps,
        )
        fn = _COMPILED_MHC_COLLAPSE_CACHE.get(key)
        if fn is None:
            fn = torch.compile(
                _mhc_collapse_and_rmsnorm_fn, backend="tpu", dynamic=False
            )
            _COMPILED_MHC_COLLAPSE_CACHE[key] = fn
        return fn(
            hidden_streams,
            fn_weight,
            base,
            scale,
            sublayer_norm_weight,
            hc_mult,
            sinkhorn_iters,
            hc_eps,
            rms_norm_eps,
        )
    return _mhc_collapse_and_rmsnorm_fn(
        hidden_streams,
        fn_weight,
        base,
        scale,
        sublayer_norm_weight,
        hc_mult,
        sinkhorn_iters,
        hc_eps,
        rms_norm_eps,
    )


def mhc_combine_streams(
    residual_streams: torch.Tensor,  # [..., hc_mult, hidden_size]
    sublayer_out: torch.Tensor,  # [..., hidden_size]
    post: torch.Tensor,  # [..., hc_mult]
    comb: torch.Tensor,  # [..., hc_mult, hc_mult]
) -> torch.Tensor:
    """Fused VPU-native FP32 stream expansion & doubly-stochastic residual combine."""
    if residual_streams.device.type == "tpu" and not torch.compiler.is_compiling():
        key = (
            tuple(residual_streams.shape),
            residual_streams.dtype,
            sublayer_out.dtype,
        )
        fn = _COMPILED_MHC_COMBINE_CACHE.get(key)
        if fn is None:
            fn = torch.compile(_mhc_combine_streams_fn, backend="tpu", dynamic=False)
            _COMPILED_MHC_COMBINE_CACHE[key] = fn
        return fn(residual_streams, sublayer_out, post, comb)
    return _mhc_combine_streams_fn(residual_streams, sublayer_out, post, comb)


class FusedGlm53HyperConnectionTpu(nn.Module):
    """Drop-in TPU module for `Glm5NextTextHyperConnection` + sublayer RMSNorm."""

    def __init__(
        self,
        hidden_size: int = 4096,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        rms_norm_eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.rms_norm_eps = rms_norm_eps

        mix = (2 + hc_mult) * hc_mult
        self.fn = nn.Parameter(
            torch.empty((mix, hc_mult * hidden_size), device=device, dtype=dtype)
        )
        self.base = nn.Parameter(torch.empty(mix, device=device, dtype=dtype))
        self.scale = nn.Parameter(torch.empty(3, device=device, dtype=dtype))
        self.sublayer_norm_weight = nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype)
        )

    def collapse_and_norm(
        self, hidden_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return mhc_collapse_and_rmsnorm(
            hidden_streams,
            self.fn,
            self.base,
            self.scale,
            self.sublayer_norm_weight,
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            hc_eps=self.hc_eps,
            rms_norm_eps=self.rms_norm_eps,
        )

    def combine(
        self,
        residual_streams: torch.Tensor,
        sublayer_out: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        return mhc_combine_streams(residual_streams, sublayer_out, post, comb)

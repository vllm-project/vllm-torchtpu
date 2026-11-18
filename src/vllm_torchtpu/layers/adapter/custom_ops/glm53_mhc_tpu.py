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
"""TPU mHC (Manifold-Constrained Hyper-Connections) ops for GLM 5.3 Flash.

Every GLM 5.3 Flash decoder layer wraps both its attention block and its
MoE/MLP block in a Manifold-Constrained Hyper-Connection (`hc_mult=4` residual
streams, 20 Sinkhorn-Knopp iterations) followed by a weighted sublayer
RMSNorm. Two implementations are provided, selected by
`VLLM_TPU_GLM53_MHC_IMPL`:

* `pallas` (default): `get_glm53_mhc_ops` returns one `jax_op` per mHC site
  running the fused Pallas kernels in `vllm_torchtpu.kernels.glm53_mhc` (mix
  GEMM, Sinkhorn gates, stream collapse and the sublayer RMSNorm in a single
  kernel; the previous sublayer's stream recombine is fused in front of it).
* `compiled`: `FusedGlm53HyperConnectionTpu.collapse_and_norm` / `combine`
  run the reference math under `torch.compile(backend="tpu")`.

Two TPU hazards of the reference implementation are avoided in both:

1. The reference combines the residual streams with
   `torch.matmul(comb.transpose(-1, -2), residual)` for a `[N, 4, 4]` mixer.
   On TPU that pads the 4x4 mixer to a 128x128 MXU tile and, under default
   matmul precision, rounds the float32 operands to bfloat16 (~2e-3 relative
   error), breaking the doubly-stochastic invariant Sinkhorn just
   established. The combine here is an explicit float32 VPU reduction over
   the 4-wide stream axis instead.
2. The Sinkhorn loop, stream collapse and RMSNorm are ~50 tiny ops per mHC
   site; fusing them keeps the per-step op dispatch count low.
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F
from torch import nn
from torch_tpu._internal import pallas

MHC_IMPL_PALLAS = "pallas"
MHC_IMPL_COMPILED = "compiled"

# `post = 2 * sigmoid(...)` in the GLM 5.3 hyper-connection.
GLM53_HC_POST_MULT = 2.0


def glm53_mhc_impl() -> str:
    """The mHC implementation selected by `VLLM_TPU_GLM53_MHC_IMPL`."""
    from vllm_torchtpu import envs

    raw = envs.VLLM_TPU_GLM53_MHC_IMPL
    impl = (raw or MHC_IMPL_PALLAS).strip().lower()
    if impl not in (MHC_IMPL_PALLAS, MHC_IMPL_COMPILED):
        raise ValueError(
            f"VLLM_TPU_GLM53_MHC_IMPL={raw!r}; expected "
            f"{MHC_IMPL_PALLAS!r} or {MHC_IMPL_COMPILED!r}."
        )
    return impl


def split_fn3_torch(
    fn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute the 3 chunk BF16 split of `fn` on CPU in PyTorch.

    Executing the float32 -> bfloat16 -> float32 rounding steps eagerly on CPU
    produces the exact IEEE 754 round to nearest even chunks of
    `jax.lax.reduce_precision(fn, 8, 7)` (see
    `vllm_torchtpu.kernels.glm53_mhc.split_fn3`) without XLA excess precision
    folding.
    """
    target_device = fn.device
    fn_cpu = fn.detach().to(device="cpu", dtype=torch.float32).contiguous()
    fn_hi_bf16 = fn_cpu.to(dtype=torch.bfloat16)
    rem1 = fn_cpu - fn_hi_bf16.to(dtype=torch.float32)
    fn_mid_bf16 = rem1.to(dtype=torch.bfloat16)
    rem2 = rem1 - fn_mid_bf16.to(dtype=torch.float32)
    fn_lo_bf16 = rem2.to(dtype=torch.bfloat16)
    if target_device.type != "cpu":
        return (
            fn_hi_bf16.to(device=target_device),
            fn_mid_bf16.to(device=target_device),
            fn_lo_bf16.to(device=target_device),
        )
    return fn_hi_bf16, fn_mid_bf16, fn_lo_bf16


# ---------------------------------------------------------------------------
# `compiled` implementation.
# ---------------------------------------------------------------------------


def _mhc_mix_logits_jax(
    flat_normed: jax.Array,  # [..., hc_mult * hidden_size] float32
    fn_weight: jax.Array,  # [(2 + hc_mult) * hc_mult, hc_mult * hidden_size]
) -> jax.Array:
    """3-pass IEEE FP32 MXU contraction `[..., K] x [K, M] -> [..., M]`."""
    return jax.lax.dot_general(
        flat_normed.astype(jnp.float32),
        fn_weight.astype(jnp.float32).T,
        dimension_numbers=(((flat_normed.ndim - 1,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


_MIX_LOGITS_OPS: dict[str, Any] = {}


def _fake_mix_logits(
    flat_normed: torch.Tensor, fn_weight: torch.Tensor
) -> torch.Tensor:
    return flat_normed.new_empty(
        (*flat_normed.shape[:-1], fn_weight.shape[0]), dtype=torch.float32
    )


def get_mhc_mix_logits_op() -> Any:
    """Build (or fetch) the FP32 mix-logits op; must run outside a trace."""
    op = _MIX_LOGITS_OPS.get("op")
    if op is None:
        op = pallas.jax_op("glm53_mhc::mix_logits", _mhc_mix_logits_jax)
        # The default jax_op fake rejects symbolic dims, and the compiled
        # backbone traces the token axis symbolically.
        op.register_fake(_fake_mix_logits)
        _MIX_LOGITS_OPS["op"] = op
    return op


def compute_mhc_mix_logits(
    flat_normed: torch.Tensor, fn_weight: torch.Tensor
) -> torch.Tensor:
    """Computes `mix_logits` in full FP32 precision.

    `[..., hc_mult * D] x [hc_mult * D, (2 + hc_mult) * hc_mult] -> [..., 24]`.

    On TPU this is a 3-pass MXU `dot_general`
    (`precision=jax.lax.Precision.HIGHEST`), which avoids the 1-pass BF16
    rounding error without materializing a `[N, 24, K]` VPU broadcast.
    """
    if flat_normed.device.type == "tpu":
        op = _MIX_LOGITS_OPS.get("op")
        if op is None:
            if torch.compiler.is_compiling():
                raise RuntimeError(
                    "glm53 mHC mix-logits op was not built before tracing; "
                    "call get_mhc_mix_logits_op at model construction."
                )
            op = get_mhc_mix_logits_op()
        return op(flat_normed.float(), fn_weight.float())
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

    # Unweighted RMSNorm on the flattened streams.
    flat_normed = flat * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + rms_norm_eps
    )

    # Lower the wide contraction onto the MXU in IEEE FP32 precision (3-pass
    # dot_general) to preserve the Sinkhorn-Knopp doubly-stochastic invariant
    # without materializing a [N, 24, K] VPU broadcast tensor.
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
    post = GLM53_HC_POST_MULT * torch.sigmoid(post_w * post_scale + post_b)

    comb_logits = comb_w.view(
        *comb_w.shape[:-1], hc_mult, hc_mult
    ) * comb_scale + comb_b.view(hc_mult, hc_mult)
    comb = torch.softmax(comb_logits, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)

    # Collapse `hc_mult` streams -> single `[..., hidden_size]` vector in FP32
    # on the VPU, then immediately apply the sublayer RMSNorm before casting.
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
    """VPU-native FP32 stream expansion and doubly-stochastic residual combine.

    Avoids `torch.matmul(comb.transpose(-1, -2), residual)`, which on TPU
    routes a 4x4 matrix to the 128x128 MXU and rounds FP32 inputs to BF16.
    Computes
    `post[..., h, None] * sublayer_out[..., None, :]
    + sum_i comb[..., i, h] * residual[..., i, :]`
    directly on the VPU in FP32.
    """
    orig_dtype = residual_streams.dtype
    res_f32 = residual_streams.float()
    sub_f32 = sublayer_out.float()
    # comb^T @ residual along the hc_mult axis: comb.transpose(-1, -2) has
    # shape [..., H_out, H_in].
    mixed_res = (
        comb.float().transpose(-1, -2).unsqueeze(-1) * res_f32.unsqueeze(-3)
    ).sum(dim=-2)
    combined = post.float().unsqueeze(-1) * sub_f32.unsqueeze(-2) + mixed_res
    return combined.to(orig_dtype)


_COMPILED_MHC_COLLAPSE_CACHE: dict[tuple[Any, ...], Any] = {}
_COMPILED_MHC_COMBINE_CACHE: dict[tuple[Any, ...], Any] = {}


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

    Eager calls on TPU run under `torch.compile(backend="tpu")` so XLA fuses
    the `bf16 -> f32` cast, Sinkhorn loop, and stream reductions into a single
    executable without HBM layout transpositions.
    """
    if hidden_streams.device.type == "tpu" and not torch.compiler.is_compiling():
        get_mhc_mix_logits_op()
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
    """Fused VPU-native FP32 stream expansion and residual combine."""
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


# ---------------------------------------------------------------------------
# Weighted RMSNorm (shared by the mHC sites and the model's other norms).
# ---------------------------------------------------------------------------


def _rms_norm_weighted_fn(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Same op order as vLLM's `ir.ops.rms_norm` (bit-exact with `RMSNorm`)."""
    xf = x.float()
    var = xf.square().mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (xf.to(weight.dtype) * weight).to(x.dtype)


def _rms_norm_weighted_jax(x: jax.Array, weight: jax.Array, *, eps: float) -> jax.Array:
    """JAX twin of `_rms_norm_weighted_fn` (same op order)."""
    xf = x.astype(jnp.float32)
    var = jnp.mean(xf * xf, axis=-1, keepdims=True)
    xf = xf * jax.lax.rsqrt(var + eps)
    return (xf.astype(weight.dtype) * weight).astype(x.dtype)


_RMS_NORM_OPS: dict[float, Any] = {}


def _fake_rms_norm_weighted(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def get_rms_norm_weighted_op(eps: float) -> Any:
    """Build (or fetch) the weighted RMSNorm op for `eps`.

    Call it at model construction. Registering a `jax_op` cannot be traced, so
    a cache miss inside the compiled backbone is an error (see
    `rms_norm_weighted`).
    """
    op = _RMS_NORM_OPS.get(eps)
    if op is None:
        op = pallas.jax_op(
            f"glm53::rms_norm_weighted_e{eps:.0e}".replace("-", "m"),
            functools.partial(_rms_norm_weighted_jax, eps=eps),
        )
        # The default fake rejects the symbolic token axis of the compiled
        # backbone's trace.
        op.register_fake(_fake_rms_norm_weighted)
        _RMS_NORM_OPS[eps] = op
    return op


def rms_norm_weighted(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Weighted RMSNorm; one `jax_op` dispatch on TPU.

    A `jax_op` call costs much less host time than a `torch.compile`d call,
    which matters in host-bound decode.
    """
    if x.device.type == "tpu":
        op = _RMS_NORM_OPS.get(eps)
        if op is None:
            if torch.compiler.is_compiling():
                raise RuntimeError(
                    f"glm53 rms_norm_weighted op for eps={eps} was not built "
                    "before tracing; call get_rms_norm_weighted_op at model "
                    "construction."
                )
            op = get_rms_norm_weighted_op(eps)
        return op(x.contiguous(), weight)
    return _rms_norm_weighted_fn(x, weight, eps)


# ---------------------------------------------------------------------------
# `pallas` implementation.
# ---------------------------------------------------------------------------


class Glm53MhcOps(NamedTuple):
    """GLM 5.3 mHC on the fused Pallas kernels.

    GLM 5.3 mHC is the same math as DeepSeek-V4's (vLLM `mhc_pre_torch` /
    `mhc_post_torch`): RMS-scaled mix GEMM, `sigmoid + eps` read gates,
    `2 * sigmoid` write gates and a softmax-initialized Sinkhorn with one eps
    for both pre and Sinkhorn. GLM additionally applies a weighted RMSNorm to
    the collapsed stream; it is folded into the same kernel, so each mHC site
    is one dispatch. The residual streams are carried flattened to
    `[T, M * D]`.

    ``pre(residual, fn_hi, fn_mid, fn_lo, hc_scale, hc_base, norm_weight)``
        -> (post_mix [T, M, 1] f32, comb_mix [T, M, M] f32,
            normed_layer_input [T, D] bf16).
    ``fused(x, residual, post_mix, comb_mix, fn_hi, fn_mid, fn_lo, hc_scale,
            hc_base, norm_weight)``
        -> (residual_cur, post_mix_cur, comb_mix_cur, normed_layer_input_cur):
        the previous sublayer's post followed by this sublayer's pre.
    ``post(x, residual, post_mix, comb_mix)`` -> residual (shared DeepSeek-V4
        op, on `[T, M, D]`).

    `fn_hi`, `fn_mid` and `fn_lo` are the 3 chunk BF16 split of `fn`
    (`FusedGlm53HyperConnectionTpu.fn_split`).
    """

    pre: Any
    fused: Any
    post: Any


def _glm53_mhc_pre_norm_jax(
    residual: jax.Array,  # [T, M * D]
    fn_hi: jax.Array,
    fn_mid: jax.Array,
    fn_lo: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    norm_weight: jax.Array,
    *,
    rms_eps: float,
    hc_eps: float,
    iters: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    from vllm_torchtpu.kernels.glm53_mhc import mhc_pre_norm_fused

    return mhc_pre_norm_fused(
        residual,
        fn_hi,
        fn_mid,
        fn_lo,
        hc_scale,
        hc_base,
        norm_weight,
        rms_eps,
        hc_eps,
        hc_eps,
        GLM53_HC_POST_MULT,
        iters,
    )


def _glm53_mhc_post_pre_norm_jax(
    x: jax.Array,
    residual: jax.Array,  # [T, M * D]
    post_mix: jax.Array,
    comb_mix: jax.Array,
    fn_hi: jax.Array,
    fn_mid: jax.Array,
    fn_lo: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    norm_weight: jax.Array,
    *,
    rms_eps: float,
    hc_eps: float,
    iters: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    from vllm_torchtpu.kernels.glm53_mhc import mhc_fused_post_pre_norm

    return mhc_fused_post_pre_norm(
        x,
        residual,
        post_mix,
        comb_mix,
        fn_hi,
        fn_mid,
        fn_lo,
        hc_scale,
        hc_base,
        norm_weight,
        rms_eps,
        hc_eps,
        hc_eps,
        GLM53_HC_POST_MULT,
        iters,
    )


def _fake_mhc_pre_norm(
    residual: torch.Tensor,
    fn_hi: torch.Tensor,
    fn_mid: torch.Tensor,
    fn_lo: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t, m = residual.shape[0], residual.shape[1] // norm_weight.shape[0]
    return (
        residual.new_empty((t, m, 1), dtype=torch.float32),
        residual.new_empty((t, m, m), dtype=torch.float32),
        residual.new_empty((t, norm_weight.shape[0])),
    )


def _fake_mhc_post_pre_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    fn_hi: torch.Tensor,
    fn_mid: torch.Tensor,
    fn_lo: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(residual),
        *_fake_mhc_pre_norm(
            residual, fn_hi, fn_mid, fn_lo, hc_scale, hc_base, norm_weight
        ),
    )


_GLM53_MHC_OPS: dict[tuple[float, float, int], Glm53MhcOps] = {}


def get_glm53_mhc_ops(rms_eps: float, hc_eps: float, iters: int) -> Glm53MhcOps:
    """Build (or fetch) the GLM mHC ops; must run outside a compiled region.

    `pre` / `fused` take and return the residual streams flattened to
    `[T, M * D]`: a bf16 `[T, 4, D]` array pads the 4-stream axis to a full
    sublane tile on TPU, so a 3-D carry would cost a relayout copy at every
    kernel boundary. `post` is the shared DeepSeek-V4 op on `[T, M, D]`.
    """
    key = (rms_eps, hc_eps, iters)
    ops = _GLM53_MHC_OPS.get(key)
    if ops is not None:
        return ops
    # Imported here: these modules need the model wrapper context.
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
        deepseek_v4_mhc_op as dsv4,
    )
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_pallas_op_mesh,
    )

    mesh = get_pallas_op_mesh()
    suffix = dsv4._gate_suffix(rms_eps, hc_eps, hc_eps, GLM53_HC_POST_MULT, iters)
    consts = dict(rms_eps=rms_eps, hc_eps=hc_eps, iters=iters)
    tok = dsv4.ATTN_DATA_AXIS
    jax_op_any = cast(Any, pallas.jax_op)
    pre = jax_op_any(
        "pallas::glm53_mhc_pre_norm_v1" + suffix,
        functools.partial(_glm53_mhc_pre_norm_jax, **consts),
        mesh=mesh,
        input_partition_specs=(
            P(tok, None),  # residual [T, M * D]
            P(),  # fn_hi
            P(),  # fn_mid
            P(),  # fn_lo
            P(),  # hc_scale
            P(),  # hc_base
            P(),  # norm_weight
        ),
    )
    pre.register_fake(_fake_mhc_pre_norm)
    fused = jax_op_any(
        "pallas::glm53_mhc_post_pre_norm_v1" + suffix,
        functools.partial(_glm53_mhc_post_pre_norm_jax, **consts),
        mesh=mesh,
        input_partition_specs=(
            P(tok, None),  # x
            P(tok, None),  # residual [T, M * D]
            P(tok, None, None),  # post_mix
            P(tok, None, None),  # comb_mix
            P(),  # fn_hi
            P(),  # fn_mid
            P(),  # fn_lo
            P(),  # hc_scale
            P(),  # hc_base
            P(),  # norm_weight
        ),
    )
    fused.register_fake(_fake_mhc_post_pre_norm)
    ops = Glm53MhcOps(pre=pre, fused=fused, post=dsv4.get_mhc_post_op())
    _GLM53_MHC_OPS[key] = ops
    return ops


class FusedGlm53HyperConnectionTpu(nn.Module):
    """TPU module for the GLM 5.3 hyper-connection + sublayer RMSNorm."""

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
        # The `compiled` path calls this op from inside the traced backbone,
        # where it cannot be registered.
        get_mhc_mix_logits_op()

    def prepare_pallas_weights(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute the 3 chunk BF16 decomposition of `fn` for Pallas mHC."""
        fn_split = split_fn3_torch(self.fn.detach())
        object.__setattr__(self, "_fn_split", fn_split)
        object.__setattr__(
            self,
            "_fn_split_key",
            (id(self.fn), self.fn._version, self.fn.device),
        )
        object.__setattr__(self, "_scale_f32", self.scale.detach().float())
        object.__setattr__(
            self,
            "_scale_f32_key",
            (id(self.scale), self.scale._version, self.scale.device),
        )
        object.__setattr__(self, "_base_f32", self.base.detach().float())
        object.__setattr__(
            self,
            "_base_f32_key",
            (id(self.base), self.base._version, self.base.device),
        )
        return fn_split

    @property
    def fn_split(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cached `(fn_hi, fn_mid, fn_lo)` BF16 chunks of `self.fn`."""
        cached = self.__dict__.get("_fn_split")
        if (
            cached is None
            or cached[0].device != self.fn.device
            or (
                not torch.compiler.is_compiling()
                and self.__dict__.get("_fn_split_key")
                != (id(self.fn), self.fn._version, self.fn.device)
            )
        ):
            return self.prepare_pallas_weights()
        return cached

    @property
    def scale_f32(self) -> torch.Tensor:
        """Float32 view or cached copy of `self.scale`."""
        if self.scale.dtype == torch.float32:
            return self.scale
        cached = self.__dict__.get("_scale_f32")
        if (
            cached is None
            or cached.device != self.scale.device
            or (
                not torch.compiler.is_compiling()
                and self.__dict__.get("_scale_f32_key")
                != (id(self.scale), self.scale._version, self.scale.device)
            )
        ):
            self.prepare_pallas_weights()
            return self.__dict__["_scale_f32"]
        return cached

    @property
    def base_f32(self) -> torch.Tensor:
        """Float32 view or cached copy of `self.base`."""
        if self.base.dtype == torch.float32:
            return self.base
        cached = self.__dict__.get("_base_f32")
        if (
            cached is None
            or cached.device != self.base.device
            or (
                not torch.compiler.is_compiling()
                and self.__dict__.get("_base_f32_key")
                != (id(self.base), self.base._version, self.base.device)
            )
        ):
            self.prepare_pallas_weights()
            return self.__dict__["_base_f32"]
        return cached

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

    def sublayer_norm(self, collapsed: torch.Tensor) -> torch.Tensor:
        """Weighted sublayer RMSNorm (reference for the fused-norm ops)."""
        return rms_norm_weighted(
            collapsed, self.sublayer_norm_weight, self.rms_norm_eps
        )

    @property
    def mhc_ops(self) -> Glm53MhcOps:
        """The Pallas mHC ops for this site's constants.

        Built lazily on the first eager forward, where the model wrapper
        context (and its mesh) is set.
        """
        ops = self.__dict__.get("_mhc_ops_instance")
        if ops is None:
            ops = get_glm53_mhc_ops(
                self.rms_norm_eps, self.hc_eps, self.hc_sinkhorn_iters
            )
            object.__setattr__(self, "_mhc_ops_instance", ops)
        return ops

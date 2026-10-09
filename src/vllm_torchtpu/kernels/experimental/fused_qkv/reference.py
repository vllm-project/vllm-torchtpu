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

"""Unfused XLA attention and GDN prologues built from the shared components.

`gqa_reference()` computes the GQA attention prologue with one of three input
numerics:

  Numerics.KERNEL: what fused_qkv computes; the kernel must match it exactly.
  Numerics.VLLM: what vllm-torchtpu computes, with per-token fp8 activations.
  Numerics.UNQUANTIZED: f32 activations and matmul; the accuracy reference for
    the other two.

Everything after the QKV projection (q/k/v norms, RoPE, KV cast, layouts) is
the same for all three and follows vLLM.

`gdn_reference()` does the same for the GDN prologue: the in_proj_qkvz and
in_proj_ba projections, laid out for the GDN kernel.
"""

from __future__ import annotations

import enum
import functools

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.experimental.batched_rpa import configs as rpa_configs
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as rpa_wrapper
from vllm_torchtpu.kernels.experimental.fused_qkv.kernel import configs
from vllm_torchtpu.kernels.experimental.fused_qkv.kernel.library import (
    norm,
    quant,
    rope,
)

# The input-side fields (input_norm, residual, fp8, act_quant_block, eps) are
# shared by both configs.
_Config = configs.AttentionConfig | configs.GdnConfig


class Numerics(enum.Enum):
    """Input numerics; see the module docstring."""

    KERNEL = "kernel"
    VLLM = "vllm"
    UNQUANTIZED = "unquantized"


def project(
    x: jax.Array,
    w: jax.Array,
    x_scale: jax.Array | None = None,
    w_scale: jax.Array | None = None,
    bias: jax.Array | None = None,
    exact: bool = False,
    dtype: jnp.dtype = jnp.bfloat16,
) -> jax.Array:
    """x @ w in f32, dequantized and biased, rounded to `dtype`.

    exact=True runs the dot on f32 inputs at HIGHEST precision.
    """
    if exact:
        # Keeps XLA from folding the upcast back into a low-precision dot.
        x, w = jax.lax.optimization_barrier(
            (x.astype(jnp.float32), w.astype(jnp.float32))
        )
    y = jnp.dot(
        x,
        w,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST if exact else None,
    )
    if x_scale is not None:
        y = y * x_scale * w_scale
    if bias is not None:
        y = y + bias.astype(jnp.float32)
    return y.astype(dtype)


def matmul_input(
    config: _Config,
    x: jax.Array,
    residual: jax.Array | None,
    weights: configs.QkvWeights,
) -> tuple[
    jax.Array,
    jax.Array | None,
    jax.Array | None,
]:
    """vllm-torchtpu: residual add + input norm (+ per-token fp8 quant).

    Returns (x, x_scale, new residual). Where each step comes from:

    1. norm.add_norm is the decoder layer's input norm, e.g.
       third_party/py/vllm/model_executor/models/qwen3_moe.py
       Qwen3MoeDecoderLayer.forward:
         hidden_states, residual = self.input_layernorm(hidden_states, residual)
       RMSNorm (and GemmaRMSNorm, with weight + 1) forward_native calls
       third_party/py/vllm/ir/ops/layernorm.py fused_add_rms_norm:
         x = x.to(torch.float32)
         x = x + x_residual.to(torch.float32)
         x_residual = x.to(orig_dtype)
         variance = x_var.pow(2).mean(dim=-1, keepdim=True)
         x = x * torch.rsqrt(variance + epsilon)
         x = x.to(weight.dtype) * weight
    2. quant.quantize_fp8_per_token is qkv_proj's activation quant. Paths are in
       experimental/akt/scaffolding/hierevo_v2/workload/vllm-torchtpu/src/
       vllm_torchtpu/. layers/vllm/quantization/fp8.py
       VllmFp8LinearMethodTPU.apply:
         out = quantized_matmul(x, weight, layer.weight_scale)
       layers/vllm/linear_common.py quantized_matmul -> _quantized_matmul_jax
       -> (per-channel w_scale) kernels/quantized_matmul/util.py
       xla_quantized_matmul:
         x_q, x_scale = quantize_tensor(x, w_q.dtype)  # quantize_block(x, -1)
       util.py quantize_block:
         abs_max = jnp.max(jnp.abs(data), axis=axis, keepdims=True)
         scale = abs_max / dtype_max
         scale = jnp.where(scale == 0, 1.0, scale)
         data_q = (data / scale).clip(dtype_min, dtype_max).astype(target_dtype)
    """
    x, residual = norm.add_norm(
        config.input_norm, x, residual, weights.input_norm_w, config.eps
    )
    x_scale = None
    if config.fp8:
        x, x_scale = quant.quantize_fp8_per_token(x)
    return x, x_scale, residual


def fused_matmul_input(
    config: _Config,
    x: jax.Array,
    residual: jax.Array | None,
    weights: configs.QkvWeights,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array | None,
    jax.Array | None,
]:
    """fused_qkv (fp8): residual add + input norm + sub-channel fp8 quant.

    Returns (x_q, x_scale, row_scale, new residual). Differs from matmul_input
    in four ways. All of them let the kernel keep the elementwise work in bf16
    (two values per lane) and skip a normalize pass:
      1. The residual add is bf16, and the rms is taken from that bf16 sum (vLLM
         adds and takes the rms in f32).
      2. x is not divided by its rms before the quant. A per-row scale cancels
         in the quant, so 1/rms is returned as `row_scale` and applied to the
         matmul output instead. This is FlashNorm's deferred
         normalization (https://arxiv.org/abs/2407.09577).
      3. One fp8 scale per act_quant_block columns (sub-channel) instead of one
         per token, which tracks the activations more closely.
      4. The norm weight multiply and the quant are bf16: x * w, then
         x * bf16(1 / scale) (vLLM: f32 x / scale; GemmaRMSNorm's 1 + w in f32).
    """
    x, row_scale, residual = fused_normed_input(config, x, residual, weights)
    x_q, x_scale = fused_quant(config, x)
    return x_q, x_scale, row_scale, residual


def fused_normed_input(
    config: _Config,
    x: jax.Array,
    residual: jax.Array | None,
    weights: configs.QkvWeights,
) -> tuple[
    jax.Array,
    jax.Array | None,
    jax.Array | None,
]:
    """Steps 1, 2 and the x * w of step 4 of fused_matmul_input.

    Returns (x * w, 1/rms, new residual); 1/rms is None without an input norm.
    """
    if residual is not None:
        x = x + residual  # 1.
        residual = x
    row_scale = None
    if config.input_norm != configs.Norm.NONE:
        xf = x.astype(jnp.float32)
        sum_sq = jnp.sum(xf * xf, axis=-1, keepdims=True)
        row_scale = jax.lax.rsqrt(sum_sq / x.shape[-1] + config.eps)  # 2.
        w = weights.input_norm_w.astype(jnp.float32)
        if config.input_norm == configs.Norm.RMS_ONE_PLUS_W:
            w = w + 1.0
        x = x * w.astype(x.dtype)  # 4.
    return x, row_scale, residual


def fused_quant(config: _Config, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Step 3 and the quant of step 4 of fused_matmul_input: (x_q, x_scale)."""
    t, k = x.shape
    x = x.reshape(t, k // config.act_quant_block, config.act_quant_block)  # 3.
    amax = jnp.max(jnp.abs(x), axis=-1, keepdims=True).astype(jnp.float32)
    x_scale = amax / quant.F8_MAX
    x_scale = jnp.where(x_scale == 0, 1.0, x_scale)
    x_q = x * (1.0 / x_scale).astype(x.dtype)  # 4.
    x_q = x_q.clip(-quant.F8_MAX, quant.F8_MAX).astype(quant.F8)
    return x_q, x_scale


def fused_project(
    x_q: jax.Array,
    x_scale: jax.Array,
    row_scale: jax.Array | None,
    weights: configs.QkvWeights,
    dtype: jnp.dtype = jnp.bfloat16,
) -> jax.Array:
    """fused_qkv (fp8): the QKV matmul, dequantized one quant block at a time."""
    num_blocks, block = x_q.shape[1:]
    acc = 0.0
    for b in range(num_blocks):
        w = weights.w[b * block : (b + 1) * block]
        acc = acc + x_scale[:, b] * jnp.dot(
            x_q[:, b], w, preferred_element_type=jnp.float32
        )
    if row_scale is not None:
        acc = acc * row_scale
    acc = acc * weights.w_scale
    if weights.bias is not None:
        acc = acc + weights.bias.astype(jnp.float32)
    return acc.astype(dtype)


def unquantized_input(
    config: _Config,
    x: jax.Array,
    residual: jax.Array | None,
    weights: configs.QkvWeights,
) -> tuple[jax.Array, jax.Array | None]:
    """f32 residual add and input norm: (normed x, new residual).

    All f32: rounding x to bf16 here, as vLLM does, would hide vLLM's rounding
    error and count the kernel's.
    """
    xf = x.astype(jnp.float32)
    if residual is not None:
        xf = xf + residual.astype(jnp.float32)
        residual = xf.astype(x.dtype)
    inv_rms = jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + config.eps)
    if config.input_norm != configs.Norm.NONE:
        w_norm = weights.input_norm_w.astype(jnp.float32)
        xf = norm.scale_norm(config.input_norm, xf, inv_rms, w_norm, jnp.float32)
    return xf, residual


def qkv_projection(
    config: configs.AttentionConfig,
    x: jax.Array,
    residual: jax.Array | None,
    weights: configs.QkvWeights,
    numerics: Numerics,
) -> tuple[jax.Array, jax.Array | None]:
    """The QKV projection with `numerics`: (qkv, new residual)."""
    if numerics == Numerics.KERNEL and config.fp8:
        x_q, x_scale, row_scale, residual = fused_matmul_input(
            config, x, residual, weights
        )
        return fused_project(x_q, x_scale, row_scale, weights), residual
    if numerics == Numerics.UNQUANTIZED:
        xf, residual = unquantized_input(config, x, residual, weights)
        w = weights.w.astype(jnp.float32)
        if weights.w_scale is not None:
            w = w * weights.w_scale
        return project(xf, w, bias=weights.bias, exact=True), residual
    # bf16 configs have no quant, so KERNEL numerics are vLLM's.
    x, x_scale, residual = matmul_input(config, x, residual, weights)
    qkv = project(x, weights.w, x_scale, weights.w_scale, weights.bias)
    return qkv, residual


@functools.partial(
    jax.jit,
    static_argnums=0,
    static_argnames=("kv_layout", "numerics", "page_size"),
)
def gqa_reference(
    config: configs.AttentionConfig,
    x: jax.Array,
    residual: jax.Array | None,
    weights: configs.QkvWeights,
    cos: jax.Array | None,
    sin: jax.Array | None,
    *,
    kv_layout: rpa_configs.KVLayout = rpa_configs.KVLayout.SEQ_ALONG_LANE,
    numerics: Numerics = Numerics.KERNEL,
    page_size: int = 128,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array | None,
    jax.Array | None,
]:
    """The unfused GQA attention prologue: fused_qkv's outputs with `numerics`.

    Args:
      config: the attention config.
      x: decoder hidden states.
      residual: the residual stream, or None.
      weights: as loaded, not prepared.
      cos: RoPE cos table, or None without RoPE.
      sin: RoPE sin table, or None without RoPE.
      kv_layout: batched_rpa KV layout of the returned kv.
      numerics: input numerics; see the module docstring.
      page_size: batched_rpa page size (SEQ_ALONG_LANE KV is padded to it).

    Returns:
      q: batched_rpa's prepared Q; Dp is D aligned to 128.
      kv: batched_rpa's prepared KV, in `kv_layout`.
      gate: per-head output gates, or None without config.gate.
      residual: x + residual, or None.
    """
    t = x.shape[0]
    hq, hkv, d = config.num_q_heads, config.num_kv_heads, config.head_dim
    qkv, residual = qkv_projection(config, x, residual, weights, numerics)
    q = qkv[:, : config.kv_col].reshape(t, hq, -1)
    gate = q[..., d:].reshape(t, -1) if config.gate else None
    q = q[..., :d]
    k = qkv[:, config.k_col(0) : config.k_col(hkv)].reshape(t, hkv, d)
    v = qkv[:, config.v_col(0) : config.v_col(hkv)].reshape(t, hkv, d)
    if cos is not None:
        cos, sin = cos[:, None], sin[:, None]
    q = norm.norm(config.q_norm, q, weights.q_norm_w, config.eps)
    q = rope.rope(config.rope, q, cos, sin)
    k = norm.norm(config.k_norm, k, weights.k_norm_w, config.eps)
    k = rope.rope(config.rope, k, cos, sin)
    v = norm.norm(config.v_norm, v, None, config.eps)
    q, kv = rpa_wrapper.prepare_inputs(
        q,
        quant.quantize_fp8_per_tensor(k, config.k_scale),
        quant.quantize_fp8_per_tensor(v, config.v_scale),
        config.q_dtype,
        config.kv_dtype,
        kv_layout=kv_layout,
        page_size=page_size,
    )
    return q, kv, gate, residual


def gdn_projection(
    config: configs.GdnConfig,
    x: jax.Array,
    residual: jax.Array | None,
    weights: configs.QkvWeights,
    numerics: Numerics,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """in_proj_qkvz and in_proj_ba of the input-normed x: (qkvz, ba, residual).

    qkvz is [mixed_qkv | z] and ba is [b | a], bf16 for VLLM, else f32. KERNEL
    matches fused_gdn exactly.
    """
    if numerics == Numerics.KERNEL:
        xw, row_scale, new_residual = fused_normed_input(config, x, residual, weights)
        # f32 norm_dtype: x * w in f32, rounded to bf16 once.
        if jnp.dtype(config.norm_dtype) == jnp.float32 and row_scale is not None:
            xs = x if residual is None else x + residual
            w = weights.input_norm_w.astype(jnp.float32)
            if config.input_norm == configs.Norm.RMS_ONE_PLUS_W:
                w = w + 1.0
            xw = (xs.astype(jnp.float32) * w).astype(x.dtype)
        x_q, x_scale = fused_quant(config, xw)
        qkvz = fused_project(x_q, x_scale, row_scale, weights, jnp.float32)
        ba = project(xw, weights.side_w, exact=True, dtype=jnp.float32)
        if row_scale is not None:
            ba = ba * row_scale
        return qkvz, ba, new_residual
    if numerics == Numerics.UNQUANTIZED:
        xf, residual = unquantized_input(config, x, residual, weights)
        w = weights.w.astype(jnp.float32) * weights.w_scale
        qkvz = project(xf, w, exact=True, dtype=jnp.float32)
        ba = project(xf, weights.side_w, exact=True, dtype=jnp.float32)
        return qkvz, ba, residual
    x, residual = norm.add_norm(
        config.input_norm, x, residual, weights.input_norm_w, config.eps
    )
    x_q, x_scale = quant.quantize_fp8_per_token(x)
    qkvz = project(x_q, weights.w, x_scale, weights.w_scale)
    ba = project(x, weights.side_w)
    return qkvz, ba, residual


def gdn_layout(
    config: configs.GdnConfig,
    qkvz: jax.Array,
    ba: jax.Array,
    z_dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Splits the projections into the GDN kernel's (mixed_qkv, z, b, a)."""
    t, nv = qkvz.shape[0], config.num_v_heads
    lanes = 128  # The GDN kernel pads b and a to 128 lanes.
    mixed = qkvz[:, : config.qkv_cols].astype(jnp.float32).reshape(t, 1, -1)
    z = qkvz[:, config.z_col :].astype(z_dtype)
    z = z.reshape(t, nv, config.head_dim)

    def pad(v: jax.Array) -> jax.Array:
        v = jnp.pad(v.astype(jnp.float32), ((0, 0), (0, lanes - nv)))
        return v.reshape(t, 1, lanes)

    return mixed, z, pad(ba[:, :nv]), pad(ba[:, nv:])


@functools.partial(jax.jit, static_argnums=0, static_argnames=("numerics",))
def gdn_reference(
    config: configs.GdnConfig,
    x: jax.Array,
    residual: jax.Array | None,
    weights: configs.QkvWeights,
    *,
    numerics: Numerics = Numerics.KERNEL,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array | None,
]:
    """The unfused GDN prologue: fused_gdn's outputs with `numerics`.

    Args:
      config: the GDN config.
      x: decoder hidden states.
      residual: the residual stream, or None.
      weights: as loaded, not prepared.
      numerics: input numerics; see the module docstring.

    Returns:
      mixed_qkv: [q | k | v], f32.
      z: f32 for UNQUANTIZED, else bf16.
      b: f32, Hv lanes zero-padded to 128.
      a: f32, Hv lanes zero-padded to 128.
      residual: x + residual, or None.
    """
    qkvz, ba, residual = gdn_projection(config, x, residual, weights, numerics)
    z_dtype = jnp.float32 if numerics == Numerics.UNQUANTIZED else jnp.bfloat16
    return (*gdn_layout(config, qkvz, ba, z_dtype), residual)

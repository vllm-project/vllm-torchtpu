# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Qwen3.8 prefill equations in plain JAX: the oracle of the stage's kernels.

One request at a time, every layer as written in the model (Hugging Face
``Qwen3_5Moe`` / vLLM ``qwen3_next``) with the checkpoint's precision: static
FP8 projections (``input_scale``, ``weight_scale``), the GDN recurrence token
by token in FP32, exact causal attention, routed experts in the block-64 W4A8
form of the decode megakernel (E2M1 weights, FP32 block scales, dynamic
per-row FP8 activations). Weights are logical (unsharded) arrays, projections ``[out,
in]`` as the checkpoint stores them. Sizes come from :class:`Dims`, so tests
run the same equations at toy sizes.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import jax
import jax.numpy as jnp

F32 = jnp.float32
BF16 = jnp.bfloat16
FP8 = jnp.float8_e4m3fn
FP8_MAX = 448.0
HIGHEST = jax.lax.Precision.HIGHEST
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@dataclasses.dataclass(frozen=True)
class Dims:
    hidden: int = 8192
    k_heads: int = 16  # GDN key heads
    v_heads: int = 128  # GDN value heads
    head_k: int = 128
    head_v: int = 128
    conv: int = 4
    q_heads: int = 64
    kv_heads: int = 4
    head_dim: int = 256
    rotary_dim: int = 64
    rope_theta: float = 1e7
    experts: int = 512
    top_k: int = 10
    expert_inter: int = 2048
    shared_inter: int = 2048
    eps: float = 1e-6

    @property
    def key_dim(self) -> int:
        return self.k_heads * self.head_k

    @property
    def value_dim(self) -> int:
        return self.v_heads * self.head_v

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim


Weights = Mapping[str, jax.Array]


def rms(x, weight, eps):
    """Qwen's RMSNorm, scaled by ``1 + weight``."""
    value = x.astype(F32)
    value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + eps)
    return (value * (1 + weight.astype(F32))).astype(x.dtype)


def rms_gated(x, gate, weight, eps):
    """The GDN output norm (scaled by ``weight``) times ``silu(gate)``, with
    the BF16 rounding points of the model's fallback implementation."""
    value = x.astype(F32)
    value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + eps)
    weighted = (value.astype(x.dtype) * weight).astype(x.dtype)
    return (weighted.astype(F32) * jax.nn.silu(gate.astype(F32))).astype(x.dtype)


def fp8_linear(x, weight, input_scale, weight_scale):
    """Static W8A8: ``x / input_scale`` saturated to FP8, an FP32 contraction
    with the FP8 ``[out, in]`` weight, both scales applied after it; BF16 out."""
    quantized = jnp.clip(x.astype(F32) / input_scale, -FP8_MAX, FP8_MAX).astype(FP8)
    value = jnp.matmul(quantized.astype(F32), weight.astype(F32).T, precision=HIGHEST)
    return (value * (input_scale * weight_scale)).astype(BF16)


def bf16_linear(x, weight):
    """``x`` times the ``[out, in]`` weight, FP32 accumulation, BF16 out."""
    value = jnp.matmul(x.astype(F32), weight.astype(F32).T, precision=HIGHEST)
    return value.astype(BF16)


def row_fp8(x):
    """Dynamic per-row FP8: ``(codes, scale)`` with ``x ~= codes * scale``."""
    value = x.astype(F32)
    scale = (
        jnp.maximum(jnp.max(jnp.abs(value), axis=-1, keepdims=True), 1e-12) / FP8_MAX
    )
    return (value * (1 / scale)).astype(FP8), scale


def dequant_block64(packed, scales):
    """Block-64 W4A8 weight ``[K/8, N]`` uint32 (eight E2M1 codes per word,
    low nibble first, along K) with ``[K/64, N]`` FP32 scales -> ``[K, N]``."""
    shifts = jnp.arange(8, dtype=jnp.uint32) * 4
    codes = (packed[:, None, :] >> shifts[None, :, None]) & 0xF  # [K/8, 8, N]
    codes = codes.reshape(-1, packed.shape[-1])
    magnitude = jnp.asarray(E2M1, F32)[codes & 7]
    value = jnp.where(codes & 8, -magnitude, magnitude)
    return value * jnp.repeat(scales.astype(F32), 64, axis=0)


def w4a8_linear(x, packed, scales):
    """A block-64 W4A8 projection: per-row FP8 activations, FP32 out."""
    codes, scale = row_fp8(x)
    weight = dequant_block64(packed, scales)
    return jnp.matmul(codes.astype(F32), weight, precision=HIGHEST) * scale


def rope(x, positions, rotary_dim, theta):
    """NeoX rotary embedding of the first ``rotary_dim`` dims of each head."""
    inv = theta ** (-2 * jnp.arange(rotary_dim // 2, dtype=F32) / rotary_dim)
    angles = positions[:, None].astype(F32) * inv
    cos = jnp.concatenate((jnp.cos(angles), jnp.cos(angles)), -1).astype(x.dtype)[
        :, None
    ]
    sin = jnp.concatenate((jnp.sin(angles), jnp.sin(angles)), -1).astype(x.dtype)[
        :, None
    ]
    rotating, rest = x[..., :rotary_dim], x[..., rotary_dim:]
    first, second = jnp.split(rotating, 2, axis=-1)
    rotation = jnp.concatenate((-second, first), -1)
    return jnp.concatenate((rotating * cos + rotation * sin, rest), -1)


def gdn(x, w: Weights, conv_state, recurrent, d: Dims):
    """One GDN mixer over a request's new tokens ``x`` ``[T, H]`` (normalized
    input), from its states before them: ``conv_state`` ``[conv - 1,
    conv_dim]`` BF16 and ``recurrent`` ``[v_heads, head_v, head_k]`` FP32.
    Returns the mixer output ``[T, H]`` and both states after the tokens."""
    tokens = x.shape[0]
    qkv = fp8_linear(x, w["qkv"], w["qkv_input_scale"], w["qkv_weight_scale"])
    z = fp8_linear(x, w["z"], w["z_input_scale"], w["z_weight_scale"])
    b = fp8_linear(x, w["b"], w["b_input_scale"], w["b_weight_scale"])
    a = fp8_linear(x, w["a"], w["a_input_scale"], w["a_weight_scale"])
    history = jnp.concatenate((conv_state, qkv), axis=0)  # [conv - 1 + T, C]
    windows = jnp.stack([history[i : i + tokens] for i in range(d.conv)], axis=1)
    convolved = jnp.sum(windows.astype(F32) * w["conv"].T.astype(F32)[None], axis=1)
    convolved = jax.nn.silu(convolved.astype(BF16).astype(F32))
    q, k, v = jnp.split(convolved, (d.key_dim, 2 * d.key_dim), axis=-1)
    q = q.reshape(tokens, d.k_heads, d.head_k)
    k = k.reshape(tokens, d.k_heads, d.head_k)
    q = q * jax.lax.rsqrt(jnp.sum(q * q, -1, keepdims=True) + 1e-6) * d.head_k**-0.5
    k = k * jax.lax.rsqrt(jnp.sum(k * k, -1, keepdims=True) + 1e-6)
    group = d.v_heads // d.k_heads
    q = jnp.repeat(q, group, axis=1)  # value head h reads key head h // group
    k = jnp.repeat(k, group, axis=1)
    v = v.reshape(tokens, d.v_heads, d.head_v)
    beta = jax.nn.sigmoid(b.astype(F32))
    decay = jnp.exp(
        -jnp.exp(w["a_log"].astype(F32))
        * jax.nn.softplus(a.astype(F32) + w["dt_bias"].astype(F32))
    )

    def token(state, inputs):
        q_t, k_t, v_t, beta_t, decay_t = inputs
        state = state * decay_t[:, None, None]
        prediction = jnp.einsum("hk,hvk->hv", k_t, state, precision=HIGHEST)
        state = (
            state + (beta_t[:, None] * (v_t - prediction))[..., None] * k_t[:, None, :]
        )
        return state, jnp.einsum("hk,hvk->hv", q_t, state, precision=HIGHEST)

    recurrent, out = jax.lax.scan(token, recurrent.astype(F32), (q, k, v, beta, decay))
    out = rms_gated(
        out.astype(BF16), z.reshape(tokens, d.v_heads, d.head_v), w["norm"], d.eps
    ).reshape(tokens, d.value_dim)
    out = fp8_linear(out, w["out"], w["out_input_scale"], w["out_weight_scale"])
    return out, history[-(d.conv - 1) :], recurrent


def attention(x, positions, w: Weights, keys, values, d: Dims):
    """One gated full-attention mixer over a request's new tokens ``x``
    ``[T, H]`` at ``positions``, after its cached ``keys`` / ``values``
    ``[P, kv_heads, head_dim]`` (FP8, as stored). Returns the output and the
    new tokens' FP8 keys and values (as the pool stores them)."""
    tokens = x.shape[0]
    query_gate = fp8_linear(x, w["q"], w["q_input_scale"], w["q_weight_scale"])
    query_gate = query_gate.reshape(tokens, d.q_heads, 2 * d.head_dim)
    query, gate = jnp.split(query_gate, 2, axis=-1)
    key = fp8_linear(x, w["k"], w["k_input_scale"], w["k_weight_scale"])
    value = fp8_linear(x, w["v"], w["v_input_scale"], w["v_weight_scale"])
    key = key.reshape(tokens, d.kv_heads, d.head_dim)
    value = value.reshape(tokens, d.kv_heads, d.head_dim)
    query = rope(rms(query, w["q_norm"], d.eps), positions, d.rotary_dim, d.rope_theta)
    key = rope(rms(key, w["k_norm"], d.eps), positions, d.rotary_dim, d.rope_theta)
    new_keys = (key.astype(F32) / w["k_scale"]).astype(FP8)
    new_values = (value.astype(F32) / w["v_scale"]).astype(FP8)
    all_keys = jnp.concatenate((keys, new_keys)).astype(F32) * w["k_scale"]
    all_values = jnp.concatenate((values, new_values)).astype(F32) * w["v_scale"]
    group = d.q_heads // d.kv_heads
    all_keys = jnp.repeat(all_keys, group, axis=1)
    all_values = jnp.repeat(all_values, group, axis=1)
    scores = (
        jnp.einsum("thd,shd->hts", query.astype(F32), all_keys, precision=HIGHEST)
        * d.head_dim**-0.5
    )
    prefix = keys.shape[0]
    visible = (
        jnp.arange(prefix + tokens)[None, :] <= (prefix + jnp.arange(tokens))[:, None]
    )
    scores = jnp.where(visible[None], scores, -jnp.inf)
    probabilities = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("hts,shd->thd", probabilities, all_values, precision=HIGHEST)
    out = out.astype(BF16) * jax.nn.sigmoid(gate.astype(F32)).astype(BF16)
    out = fp8_linear(
        out.reshape(tokens, d.q_heads * d.head_dim),
        w["o"],
        w["o_input_scale"],
        w["o_weight_scale"],
    )
    return out, new_keys, new_values


def moe(x, w: Weights, d: Dims):
    """The sparse MoE block over normalized ``x`` ``[T, H]``: routed experts
    (softmax, top-k, renormalized) plus the sigmoid-gated shared expert."""
    logits = bf16_linear(x, w["router"])
    probabilities = jax.nn.softmax(logits.astype(F32), axis=-1)
    weights, experts = jax.lax.top_k(probabilities, d.top_k)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)

    def expert(e):
        gate_up = w4a8_linear(x, w["w13"][e], w["w13_scale"][e])
        gate, up = jnp.split(gate_up, 2, axis=-1)
        hidden = (jax.nn.silu(gate) * up).astype(BF16)
        return w4a8_linear(hidden, w["w2"][e], w["w2_scale"][e]).astype(BF16)

    def add_expert(e, routed):
        weight = jnp.sum(jnp.where(experts == e, weights, 0), axis=-1, keepdims=True)
        return routed + expert(e).astype(F32) * weight

    routed = jax.lax.fori_loop(0, d.experts, add_expert, jnp.zeros(x.shape, F32))
    shared_gate_up = bf16_linear(x, w["shared_gate_up"])
    gate, up = jnp.split(shared_gate_up, 2, axis=-1)
    shared = bf16_linear(
        (jax.nn.silu(gate.astype(F32)) * up.astype(F32)).astype(BF16), w["shared_down"]
    )
    shared_scale = jax.nn.sigmoid(bf16_linear(x, w["shared_gate"]).astype(F32))
    return (routed + shared.astype(F32) * shared_scale).astype(BF16)


def layer(h, positions, w: Weights, state, d: Dims, full_attention: bool):
    """One decoder layer on a request's residual stream ``h`` ``[T, H]``.
    ``state``: the GDN layer's ``(conv, recurrent)`` or the attention layer's
    cached ``(keys, values)``; returns ``h`` and the layer's new state (GDN:
    the states after the tokens; attention: the new tokens' keys and values)."""
    x = rms(h, w["input_norm"], d.eps)
    first, second = state
    if full_attention:
        out, *new_state = attention(x, positions, w, first, second, d)
    else:
        out, *new_state = gdn(x, w, first, second, d)
    h = (h.astype(F32) + out.astype(F32)).astype(BF16)
    x = rms(h, w["post_norm"], d.eps)
    h = (h.astype(F32) + moe(x, w, d).astype(F32)).astype(BF16)
    return h, tuple(new_state)

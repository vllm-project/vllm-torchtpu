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
"""Pure-JAX reference math for Qwen3.8-2.4T: NVFP4, GDN, DCP8 attention, and MoE.

Used for CPU parity verification against `tests/models/qwen38_2_4t/oracle_fp64.py`
and as the single-rank reference specification for the Qwen3.8-2.4T megakernel
plugin.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from .shapes import Qwen38Config


def rmsnorm(x: jax.Array, weight: jax.Array, eps: float = 1e-6) -> jax.Array:
    """RMSNorm in float32 accumulation, returning bfloat16.

    Unlike `megakernel.core.ops.rmsnorm` (which multiplies `v * scale` in float32
    before rounding to bfloat16 and returns float32), this helper rounds
    `(xf * inv_rms)` to bfloat16 first and then multiplies by the pre-shifted
    bfloat16 `weight` to match `oracle_fp64._rmsnorm_f64`.
    """
    xf = x.astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (xf * inv_rms).astype(jnp.bfloat16) * weight.astype(jnp.bfloat16)


def partial_rope(
    x: jax.Array,
    pos: jax.Array,
    rope_dim: int,
    theta: float = 10_000_000.0,
) -> jax.Array:
    """Apply NeoX rotary embeddings to the first `rope_dim` channels of `x`.

    Unlike `megakernel.core.ops.rope_partial` (which takes precomputed `cos`/`sin`
    tables and rounds each product through bfloat16 before subtracting), this
    helper computes NeoX frequencies directly from `pos` and `theta` in float32
    and casts `(x0 * cos - x1 * sin)` to bfloat16 once at the end.
    """
    half = rope_dim // 2
    idx = jnp.arange(half, dtype=jnp.int32).astype(jnp.float32)
    inv_freq = 1.0 / (theta ** (2.0 * idx / float(rope_dim)))
    angles = pos[:, None, None].astype(jnp.float32) * inv_freq[None, None, :]
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)

    x_rot = x[:, :, :rope_dim].astype(jnp.float32)
    x_pass = x[:, :, rope_dim:]
    x0 = x_rot[:, :, :half]
    x1 = x_rot[:, :, half:]
    y0 = (x0 * cos - x1 * sin).astype(jnp.bfloat16)
    y1 = (x1 * cos + x0 * sin).astype(jnp.bfloat16)
    return jnp.concatenate([y0, y1, x_pass], axis=-1)


def dequant_nvfp4(
    packed_u8: jax.Array,
    scale_bf16: jax.Array,
    group_size: int = 16,
) -> jax.Array:
    """Unpack 2-nibble `uint8` E2M1 weights `(K, N // 2)` and scale by `(K // G, N)`."""
    k_dim, n_half = packed_u8.shape
    u = packed_u8.astype(jnp.uint32)
    lo = u & jnp.uint32(0x0F)
    hi = (u >> jnp.uint32(4)) & jnp.uint32(0x0F)
    nib = jnp.stack([lo, hi], axis=-1).reshape(k_dim, n_half * 2)

    sign = jnp.where((nib & jnp.uint32(0x08)) != 0, -1.0, 1.0)
    code = nib & jnp.uint32(0x07)
    exp = (code >> 1).astype(jnp.float32)
    mant = (code & 1).astype(jnp.float32)
    subnormal = mant * 0.5
    normal = (1.0 + 0.5 * mant) * jnp.power(2.0, exp - 1.0)
    mag = jnp.where(exp == 0.0, subnormal, normal)
    vals = (sign * mag).astype(jnp.bfloat16)
    g_dim = k_dim // group_size
    vals_g = vals.reshape(g_dim, group_size, n_half * 2)
    scaled = vals_g * scale_bf16[:, None, :].astype(jnp.bfloat16)
    return scaled.reshape(k_dim, n_half * 2)


def fp4_matmul(
    x_bf16: jax.Array,
    packed_u8: jax.Array,
    scale_bf16: jax.Array,
    group_size: int = 16,
) -> jax.Array:
    """Dequantize `(K, N//2)` NVFP4 weights and contract with `(B, K)` in fp32."""
    w_bf16 = dequant_nvfp4(packed_u8, scale_bf16, group_size=group_size)
    return jnp.dot(
        x_bf16,
        w_bf16,
        preferred_element_type=jnp.float32,
    )


def dcp_online_softmax_update(
    q_dcp: jax.Array,
    kv_pages: tuple[jax.Array, jax.Array],
    positions: tuple[jax.Array, jax.Array],
    prev_state: tuple[jax.Array, jax.Array, jax.Array],
    cfg: Qwen38Config,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Update online softmax `(m, l, acc)` for one sequence `b` and one KV page."""
    k_page_fp8, v_page_fp8 = kv_pages
    pos_b, tok_positions = positions
    m_prev, l_prev, acc_prev = prev_state

    dcp_q = cfg.DCP * cfg.Q_R
    sm_scale = 1.0 / math.sqrt(float(cfg.D))

    valid = (tok_positions <= pos_b) & (tok_positions < cfg.max_seq_len)
    k_f32 = k_page_fp8.astype(jnp.float32)
    v_f32 = v_page_fp8.astype(jnp.float32)
    v_f32 = jnp.where(valid[None, None, :], v_f32, 0.0)

    k_rep = jnp.repeat(k_f32, cfg.DCP * cfg.gqa_ratio, axis=0)
    v_rep = jnp.repeat(v_f32, cfg.DCP * cfg.gqa_ratio, axis=0)

    q_f32 = q_dcp.astype(jnp.float32)
    scores = jnp.sum(q_f32[:, :, None] * k_rep, axis=1) * sm_scale
    scores = jnp.where(valid[None, :], scores, -1e30)

    m_block = jnp.max(scores, axis=-1)
    m_new = jnp.maximum(m_prev, m_block)
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.exp(scores - m_new[:, None])
    l_new = l_prev * alpha + jnp.sum(p, axis=-1)
    wv = jnp.sum(p[:, None, :] * v_rep, axis=2)
    acc_new = acc_prev * alpha[:, None] + wv
    return m_new.reshape(dcp_q), l_new.reshape(dcp_q), acc_new


def dcp_merge_lse(
    lse_recv: jax.Array,
    q_gate_bf16: jax.Array,
    dcp_slot: int | jax.Array,
    cfg: Qwen38Config,
) -> jax.Array:
    """Merge 8 DCP partial `(O_num, m, l)` tensors and apply output gate."""
    acc_all = lse_recv[:, :, :, : cfg.D]
    m_all = lse_recv[:, :, :, cfg.D]
    l_all = lse_recv[:, :, :, cfg.D + 1]

    m_global = jnp.max(m_all, axis=0)
    w = jnp.exp(m_all - m_global[None, :, :])
    denom = jnp.sum(l_all * w, axis=0)
    numer = jnp.sum(acc_all * w[:, :, :, None], axis=0)
    attn_all = (numer / jnp.maximum(denom[:, :, None], 1e-30)).astype(jnp.bfloat16)

    attn_4d = attn_all.astype(jnp.float32).reshape(cfg.B, cfg.DCP, cfg.Q_R, cfg.D)
    slot_mask = (jnp.arange(cfg.DCP, dtype=jnp.int32) == dcp_slot)[None, :, None, None]
    o_local = jnp.sum(jnp.where(slot_mask, attn_4d, 0.0), axis=1)
    gate = jax.nn.sigmoid(q_gate_bf16.astype(jnp.float32))
    gated = (o_local * gate).astype(jnp.bfloat16)
    return gated.reshape(cfg.B, cfg.Q_R * cfg.D)


def gdn_recurrent_step(
    qz_inputs: tuple[jax.Array, jax.Array, jax.Array],
    states: tuple[jax.Array, jax.Array],
    weights: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    cfg: Qwen38Config,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Execute GDN depthwise conv1d + delta-rule recurrence + gated RMSNorm."""
    qkv_new_bf16, z_bf16, ba_f32 = qz_inputs
    conv_state_bf16, rec_state_f32 = states
    w_conv_bf16, a_log_f32, dt_bias_f32, gdn_norm_w = weights

    b = cfg.B
    kh_r = cfg.KH_R
    vh_r = cfg.VH_R
    kd = cfg.KD
    vd = cfg.VD

    conv_win = jnp.concatenate(
        [
            conv_state_bf16.astype(jnp.bfloat16),
            qkv_new_bf16[:, None, :].astype(jnp.bfloat16),
        ],
        axis=1,
    )
    conv_next = conv_win[:, 1:, :].astype(jnp.bfloat16)

    conv_sum = jnp.sum(
        conv_win.astype(jnp.float32) * w_conv_bf16[None, :, :].astype(jnp.float32),
        axis=1,
    )
    conv_out = jax.nn.silu(conv_sum)

    q_len = kh_r * kd
    k_len = kh_r * kd
    q_raw = conv_out[:, :q_len].reshape(b, kh_r, kd)
    k_raw = conv_out[:, q_len : q_len + k_len].reshape(b, kh_r, kd)
    v = conv_out[:, q_len + k_len :].reshape(b, vh_r, vd)

    q_inv = jax.lax.rsqrt(jnp.sum(q_raw * q_raw, axis=-1, keepdims=True) + 1e-6)
    k_inv = jax.lax.rsqrt(jnp.sum(k_raw * k_raw, axis=-1, keepdims=True) + 1e-6)
    q_normed = (q_raw * q_inv) * (1.0 / math.sqrt(float(kd)))
    k_normed = k_raw * k_inv

    q = jnp.repeat(q_normed, cfg.gdn_v_per_k, axis=1)
    k = jnp.repeat(k_normed, cfg.gdn_v_per_k, axis=1)

    b_raw = ba_f32[:, :vh_r]
    a_raw = ba_f32[:, vh_r:]
    beta = jax.nn.sigmoid(b_raw)
    dt = jax.nn.softplus(a_raw + dt_bias_f32[None, :])
    g = jnp.exp(-jnp.exp(a_log_f32[None, :]) * dt)

    state_decayed = rec_state_f32 * g[:, :, None, None]
    kv_mem = jnp.sum(state_decayed * k[:, :, :, None], axis=2)
    delta = (v - kv_mem) * beta[:, :, None]
    rec_next = state_decayed + (k[:, :, :, None] * delta[:, :, None, :])

    y = jnp.sum(rec_next * q[:, :, :, None], axis=2)
    y_norm = rmsnorm(y.astype(jnp.bfloat16), gdn_norm_w)
    z_3d = z_bf16.astype(jnp.float32).reshape(b, vh_r, vd)
    y_gated = (y_norm.astype(jnp.float32) * jax.nn.silu(z_3d)).astype(jnp.bfloat16)
    return y_gated.reshape(b, cfg.Z_R), conv_next, rec_next


def moe_router_topk(
    x_norm_bf16: jax.Array,
    w_gate_bf16: jax.Array,
    cfg: Qwen38Config,
    forced_ids: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute softmax over `E` experts and select top-`K` `(top_ids, top_weights)`.

    Args:
        x_norm_bf16: `(B, H)` bfloat16 post-attention normalized activations.
        w_gate_bf16: `(H, E)` bfloat16 router projection matrix.
        cfg: Active `Qwen38Config` supplying `E` and `K`.
        forced_ids: Optional `(B, K)` int32 expert indices overriding `top_k`;
            `None` selects dynamic `top_k(probs, cfg.K)`. Negative entries in
            `forced_ids` are masked out (`weight = 0`).

    Returns:
        Tuple `(top_ids, top_weights)` of `(B, K)` int32 indices and normalized
        float32 expert weights.
    """
    logits = jnp.dot(
        x_norm_bf16,
        w_gate_bf16,
        preferred_element_type=jnp.float32,
    )
    probs = jax.nn.softmax(logits, axis=-1)
    idx_range = jnp.arange(cfg.E, dtype=jnp.int32)[None, :]
    if forced_ids is not None:
        top_ids = forced_ids.astype(jnp.int32)
        p_cols: list[jax.Array] = []
        for k_idx in range(cfg.K):
            fid = top_ids[:, k_idx : k_idx + 1]
            p_cols.append(jnp.sum(jnp.where(idx_range == fid, probs, 0.0), axis=-1))
        top_p = jnp.stack(p_cols, axis=-1)
    else:
        cur = probs
        id_cols: list[jax.Array] = []
        p_cols = []
        for _ in range(cfg.K):
            best_id = jnp.argmax(cur, axis=-1).astype(jnp.int32)
            best_p = jnp.max(cur, axis=-1)
            id_cols.append(best_id)
            p_cols.append(best_p)
            cur = jnp.where(idx_range == best_id[:, None], -jnp.inf, cur)
        top_ids = jnp.stack(id_cols, axis=-1)
        top_p = jnp.stack(p_cols, axis=-1)
    denom = jnp.maximum(jnp.sum(top_p, axis=-1, keepdims=True), 1e-12)
    top_w = top_p / denom
    return top_ids, top_w


def swiglu_split(gate_up_f32: jax.Array, inter_dim: int) -> jax.Array:
    """Apply SwiGLU `silu(gate) * up` on concatenated `(..., 2 * inter_dim)`.

    Splits a concatenated `[gate | up]` float32 projection along the last axis
    and multiplies `silu(gate) * up` in float32 before casting to bfloat16
    (whereas `megakernel.core.ops.silu_mul` takes separate `(gate, up)` tensors and
    rounds `silu(gate)` to bfloat16 before multiplying).
    """
    gate = gate_up_f32[..., :inter_dim]
    up = gate_up_f32[..., inter_dim:]
    return (jax.nn.silu(gate) * up).astype(jnp.bfloat16)

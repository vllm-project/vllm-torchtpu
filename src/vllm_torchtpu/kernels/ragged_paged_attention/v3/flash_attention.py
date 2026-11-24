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
"""Shared FlashAttention-2 tile compute steps for ragged paged attention v3."""

import dataclasses
from typing import Any, cast

import jax.numpy as jnp
from jax import lax

from vllm_torchtpu.kernels.pallas_util import broadcast_minor


@dataclasses.dataclass(frozen=True)
class FlashAttentionConfig:
    """Static per-kernel options shared across FlashAttention-2 tile steps."""

    num_q_heads_per_kv_head: int
    sm_scale: float
    out_dtype: Any
    mask_value: float
    span_int_ty: Any = jnp.int32
    use_causal_mask: bool = True
    sliding_window: int | None = None
    soft_cap: float | None = None
    q_scale: float | None = None
    k_scale: float | None = None
    v_scale: float | None = None


def _mask_and(mask, new_mask):
    if mask is None:
        return new_mask
    return jnp.logical_and(mask, new_mask)


def flash_attention_step1_qk_softmax(
    q,
    k,
    v,
    l_ref,
    m_ref,
    *,
    processed_q_len,
    processed_kv_len,
    effective_kv_len,
    cfg: FlashAttentionConfig,
    use_causal_mask: bool | None = None,
    skip_kv_mask: bool = False,
):
    """Compute FlashAttention-2 `Q @ K^T`, masking, online rowmax/softmax, and `l/m`.

    Args:
      q: Query tile of shape `[actual_bq_csz * num_q_heads_per_kv_head, head_dim]`.
      k: Key tile of shape `[bkv_csz, head_dim]`.
      v: Value tile of shape `[bkv_csz, head_dim]`.
      l_ref: Online softmax row-sum ref of shape
        `[actual_bq_csz * num_q_heads_per_kv_head, 128]`.
      m_ref: Online softmax row-max ref of shape
        `[actual_bq_csz * num_q_heads_per_kv_head, 128]`.
      processed_q_len: Starting sequence position of the current query sub-tile.
      processed_kv_len: Starting sequence position of the current KV sub-tile.
      effective_kv_len: Effective sequence KV length for valid-token masking.
      cfg: Frozen `FlashAttentionConfig` holding static kernel options.
      use_causal_mask: Optional per-tile override of `cfg.use_causal_mask`.
      skip_kv_mask: Whether to skip `< effective_kv_len` masking on K and V.

    Returns:
      `(p, v, exp_m_diff)` where `p` has shape `[q_rows, bkv_csz]`, `v` is the
      (optionally tail-zeroed) value tile `[bkv_csz, head_dim]`, and `exp_m_diff`
      has shape `[q_rows, 128]`.
    """
    causal_mask = cfg.use_causal_mask if use_causal_mask is None else use_causal_mask
    assert q.ndim == 2
    assert q.shape[0] % cfg.num_q_heads_per_kv_head == 0
    q_rows, head_dim = q.shape
    assert k.ndim == 2 and k.shape[1] == head_dim
    bkv_csz = k.shape[0]
    assert v.shape == (bkv_csz, head_dim)
    assert k.dtype == v.dtype
    assert l_ref.shape == (q_rows, 128)
    assert m_ref.shape == (q_rows, 128)

    if cfg.q_scale is not None:
        q = q / cfg.q_scale
        if jnp.issubdtype(k.dtype, jnp.floating):
            info = jnp.finfo(k.dtype)
            q = jnp.clip(q, min=float(info.min), max=float(info.max))
        q = q.astype(k.dtype)

    s = jnp.matmul(q, k.T, preferred_element_type=jnp.float32).astype(cfg.out_dtype)
    s *= cfg.sm_scale
    if cfg.k_scale is not None:
        s *= cfg.k_scale
    if cfg.q_scale is not None:
        s *= cfg.q_scale
    if cfg.soft_cap is not None:
        s = cfg.soft_cap * jnp.tanh(s / cfg.soft_cap)

    span_int_ty = cfg.span_int_ty
    processed_q_int = jnp.asarray(processed_q_len, dtype=span_int_ty)
    processed_kv_int = jnp.asarray(processed_kv_len, dtype=span_int_ty)
    effective_kv_int = jnp.asarray(effective_kv_len, dtype=span_int_ty)
    q_span = processed_q_int + (
        lax.broadcasted_iota(jnp.int32, s.shape, 0) // cfg.num_q_heads_per_kv_head
    ).astype(span_int_ty)
    k_span = processed_kv_int + lax.broadcasted_iota(span_int_ty, s.shape, 1)
    v_span = processed_kv_int + lax.broadcasted_iota(span_int_ty, v.shape, 0)

    mask = None
    if causal_mask:
        assert not skip_kv_mask
        mask = _mask_and(mask, q_span >= k_span)
    if not skip_kv_mask:
        mask = _mask_and(mask, k_span < effective_kv_int)
        v = jnp.where(v_span < effective_kv_int, v, 0.0)
    if cfg.sliding_window is not None:
        mask = _mask_and(mask, q_span < k_span + cfg.sliding_window)
    if mask is not None:
        s = jnp.where(mask, s, cast(float, cfg.mask_value))

    s_rowmax = jnp.max(s, axis=1, keepdims=True)
    m_prev = m_ref[...]
    m_curr = jnp.maximum(m_prev, s_rowmax)
    m_ref[...] = m_curr
    p = jnp.exp(s - broadcast_minor(m_curr, s.shape))

    p_rowsum = jnp.sum(p, axis=1, keepdims=True, dtype=cfg.out_dtype)
    exp_m_diff = jnp.exp(m_prev - m_curr)
    l_prev = l_ref[...]
    l_ref[...] = exp_m_diff * l_prev + p_rowsum
    return p, v, exp_m_diff


def flash_attention_step2_pv(p, v, exp_m_diff, o_ref, *, cfg: FlashAttentionConfig):
    """Accumulate `P @ V` into `o_ref` with online FlashAttention-2 rescaling.

    Args:
      p: Unnormalized attention weights of shape `[q_rows, bkv_csz]`.
      v: Value tile of shape `[bkv_csz, head_dim]`.
      exp_m_diff: Row-max rescaling factor of shape `[q_rows, 128]`.
      o_ref: Output accumulator ref of shape `[q_rows, head_dim]`.
      cfg: Frozen `FlashAttentionConfig` holding `out_dtype` and `v_scale`.
    """
    assert p.ndim == 2 and v.ndim == 2
    q_rows, bkv_csz = p.shape
    v_rows, head_dim = v.shape
    assert v_rows == bkv_csz
    assert exp_m_diff.shape == (q_rows, 128)
    assert o_ref.shape == (q_rows, head_dim)

    pv = jnp.matmul(p, v, preferred_element_type=jnp.float32).astype(cfg.out_dtype)
    if cfg.v_scale is not None:
        pv *= cfg.v_scale
    o_prev = o_ref[...]
    o_ref[...] = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv

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

from typing import cast

import jax.numpy as jnp
from jax import lax

from vllm_torchtpu.kernels.ragged_paged_attention.v3.util import broadcast_minor


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
    num_q_heads_per_kv_head: int,
    sm_scale: float,
    out_dtype,
    mask_value,
    span_int_ty=jnp.int32,
    max_kv_len: int | None = None,
    use_causal_mask: bool = True,
    skip_kv_mask: bool = False,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
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
      num_q_heads_per_kv_head: Padded number of query heads per KV head.
      sm_scale: Softmax scale applied to `Q @ K^T`.
      out_dtype: Accumulator/softmax dtype (`float32` or `bfloat16`).
      mask_value: Fill value for masked attention scores.
      span_int_ty: Integer dtype (`int16` or `int32`) used for Q/KV index spans.
      max_kv_len: Optional upper bound (`pages_per_seq * page_size`) on `kv_len`;
        skips the sliding-window comparison when `sliding_window >= max_kv_len`.
      use_causal_mask: Whether to apply causal masking (`q_span >= k_span`).
      skip_kv_mask: Whether to skip `< effective_kv_len` masking on K and V.
      sliding_window: Optional sliding-window length (`(q_span - k_span) < sw`).
      soft_cap: Optional logit soft-capping threshold.
      q_scale: Optional query quantization scale.
      k_scale: Optional key quantization scale.

    Returns:
      `(p, v, exp_m_diff)` where `p` has shape `[q_rows, bkv_csz]`, `v` is the
      (optionally tail-zeroed) value tile `[bkv_csz, head_dim]`, and `exp_m_diff`
      has shape `[q_rows, 128]`.
    """
    assert q.ndim == 2
    assert q.shape[0] % num_q_heads_per_kv_head == 0
    q_rows, head_dim = q.shape
    assert k.ndim == 2 and k.shape[1] == head_dim
    bkv_csz = k.shape[0]
    assert v.shape == (bkv_csz, head_dim)
    assert k.dtype == v.dtype
    assert l_ref.shape == (q_rows, 128)
    assert m_ref.shape == (q_rows, 128)

    if q_scale is not None:
        q = q / q_scale
        if jnp.issubdtype(k.dtype, jnp.floating):
            info = jnp.finfo(k.dtype)
            q = jnp.clip(q, min=float(info.min), max=float(info.max))
        q = q.astype(k.dtype)

    s = (
        jnp.matmul(q, k.T, preferred_element_type=jnp.float32).astype(out_dtype)
        * sm_scale
    )
    if k_scale is not None:
        s *= k_scale
    if q_scale is not None:
        s *= q_scale
    if soft_cap is not None:
        s = soft_cap * jnp.tanh(s / soft_cap)

    processed_kv_int = jnp.asarray(processed_kv_len, dtype=span_int_ty)
    effective_kv_int = jnp.asarray(effective_kv_len, dtype=span_int_ty)
    k_span = processed_kv_int + lax.broadcasted_iota(span_int_ty, s.shape, 1)
    mask = None
    if not skip_kv_mask:
        mask = k_span < effective_kv_int
        v_span = processed_kv_int + lax.broadcasted_iota(span_int_ty, v.shape, 0)
        v = jnp.where(v_span < effective_kv_int, v, 0.0)

    apply_sliding_window = (
        sliding_window is not None
        and (max_kv_len is None or sliding_window < max_kv_len)
        and sliding_window <= jnp.iinfo(span_int_ty).max
    )
    if use_causal_mask or apply_sliding_window:
        q_span = jnp.asarray(processed_q_len, dtype=span_int_ty) + (
            lax.broadcasted_iota(jnp.int32, s.shape, 0) // num_q_heads_per_kv_head
        ).astype(span_int_ty)
        if use_causal_mask:
            assert not skip_kv_mask
            causal = q_span >= k_span
            mask = causal if mask is None else jnp.logical_and(mask, causal)
        if apply_sliding_window:
            assert sliding_window is not None
            sw = (q_span - k_span) < sliding_window
            mask = sw if mask is None else jnp.logical_and(mask, sw)
    if mask is not None:
        s = jnp.where(mask, s, cast(float, mask_value))

    s_rowmax = jnp.max(s, axis=1, keepdims=True)
    m_prev = m_ref[...]
    m_curr = jnp.maximum(m_prev, s_rowmax)
    m_ref[...] = m_curr
    p = jnp.exp(s - broadcast_minor(m_curr, s.shape))

    p_rowsum = jnp.sum(p, axis=1, keepdims=True, dtype=out_dtype)
    exp_m_diff = jnp.exp(m_prev - m_curr)
    l_ref[...] = exp_m_diff * l_ref[...] + p_rowsum
    return p, v, exp_m_diff


def flash_attention_step2_pv(p, v, exp_m_diff, o_ref, *, out_dtype, v_scale=None):
    """Accumulate `P @ V` into `o_ref` with online FlashAttention-2 rescaling.

    Args:
      p: Unnormalized attention weights of shape `[q_rows, bkv_csz]`.
      v: Value tile of shape `[bkv_csz, head_dim]`.
      exp_m_diff: Row-max rescaling factor of shape `[q_rows, 128]`.
      o_ref: Output accumulator ref of shape `[q_rows, head_dim]`.
      out_dtype: Accumulator dtype (`float32` or `bfloat16`).
      v_scale: Optional value quantization scale.
    """
    assert p.ndim == 2 and v.ndim == 2
    q_rows, bkv_csz = p.shape
    v_rows, head_dim = v.shape
    assert v_rows == bkv_csz
    assert exp_m_diff.shape == (q_rows, 128)
    assert o_ref.shape == (q_rows, head_dim)

    pv = jnp.matmul(p, v, preferred_element_type=jnp.float32).astype(out_dtype)
    if v_scale is not None:
        pv *= v_scale
    o_prev = o_ref[...]
    o_ref[...] = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv

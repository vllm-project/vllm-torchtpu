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

"""Pallas FlashAttention inner-loop primitives for Batched Ragged Paged Attention.

Implements tiled QK^T matmul, online softmax rescaling, and PV accumulation
with optional native FP8 KV cache dequantization in VMEM.
"""

import math
import os

import jax
import jax.numpy as jnp
from jax import lax

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, utils


def _use_native_fp8(cfgs: configs.RpaConfigs, which: str) -> bool:
    return cfgs.mode == configs.RpaCase.DECODE and cfgs.serve.native_fp8_enabled(which)


# Weight of the low fp8 half in a split (hi + lo / _SPLIT_LO) quantization.
_SPLIT_LO = 16.0


def _quantize_rows(
    x: jax.Array, dtype: jnp.dtype, split: bool = False
) -> tuple[jax.Array, jax.Array]:
    """Quantizes x to fp8 `dtype` with one dynamic scale per row (last axis).

    Returns the quantized array and the f32 row scales (keepdims), such that
    x ~= x_q * scale. Rows of zeros get scale 1 (and stay zeros).

    With `split`, also quantizes the rounding residual and stacks it under the
    high part along axis -2: x ~= (hi + lo / _SPLIT_LO) * scale, which keeps
    about 7 mantissa bits (bf16-like) at the cost of doubling the matmul's LHS
    rows, which are nearly free in decode (16 of the MXU's rows are used).
    """
    x = x.astype(jnp.float32)
    fp8_max = float(jnp.finfo(dtype).max)
    amax = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    scale = amax * (1.0 / fp8_max)
    scale = jnp.where(scale > 0, scale, 1.0)
    x_s = jnp.clip(x * (1.0 / scale), -fp8_max, fp8_max)
    x_q = x_s.astype(dtype)
    if split:
        resid = (x_s - x_q.astype(jnp.float32)) * _SPLIT_LO
        x_q = jnp.concatenate([x_q, resid.astype(dtype)], axis=-2)
    return x_q, scale


def _merge_split(y: jax.Array, rows: int) -> jax.Array:
    """Combines a matmul over split (hi; lo) LHS rows: [..., 2*rows, N]."""
    return y[..., :rows, :] + y[..., rows:, :] * (1.0 / _SPLIT_LO)


def vpu_diet_enabled() -> bool:
    """A/B switch for the score-tile VPU savings (scale folding, mask skip).

    Read at trace time, like TPU_RPA_SKIP_MIXED. Both paths are numerically
    equivalent: the mask skip is exact, and the scale folding is exact when the
    folded scale is a power of two (sm_scale=1/16, k_scale=1 for head_dim 256).
    """
    return os.environ.get("TPU_RPA_VPU_DIET", "1") == "1"


def q_folded_scale(cfgs: configs.RpaConfigs) -> float | None:
    """Returns the static qk scale if it can be folded into q exactly.

    Scaling a bf16/f32 q by a power of two only changes its exponent, so
    dot(q * s, k) is bit-identical to dot(q, k) * s while touching a
    [TQ, head_dim] tile instead of the [TQ, bkv] f32 score tile.
    """
    serve = cfgs.serve
    if serve.scale_q is not None or serve.per_token_scale:
        return None
    if not jnp.issubdtype(serve.dtype_q, jnp.floating):
        return None
    if jnp.dtype(serve.dtype_q).itemsize < 2:
        return None
    scale = float(cfgs.model.sm_scale)
    if serve.scale_k is not None:
        scale *= float(serve.scale_k)
    if not (scale > 0.0 and math.isfinite(scale)):
        return None
    if math.frexp(scale)[0] != 0.5:
        return None
    return scale


def flash_attention_qk_softmax(
    step: jax.Array,
    q: jax.Array,  # [B, KV, TQ, H]
    k: jax.Array,  # [B, KV, S, H] or [B, KV, H, S]
    m_prev: jax.Array,  # [KV, TQ, 128]
    l_prev: jax.Array,  # [KV, TQ, 128]
    is_last_k: jax.Ref,  # [B]
    *,
    custom_mask: jax.Array | None,
    cfgs: configs.RpaConfigs,
    bq_start: int,
    k_scale: jax.Array | float | None = None,  # [B, KV, S]
):
    """Flash attention kernel.

    `custom_mask=None` means every score in the tile is visible (the caller
    proved the KV block lies entirely below the causal diagonal), so the
    mask select is skipped.
    """
    b, k_heads, tq, h_size = q.shape

    diet = vpu_diet_enabled()
    row_scale = None
    native_qk = _use_native_fp8(cfgs, "qk")
    split_qk = native_qk and cfgs.serve.native_fp8_split("qk")
    folded_scale = q_folded_scale(cfgs) if (diet and not native_qk) else None
    if folded_scale is not None and folded_scale != 1.0:
        q = q * jnp.asarray(folded_scale, dtype=q.dtype)

    if native_qk:
        q, row_scale = _quantize_rows(q, k.dtype, split=split_qk)
    elif cfgs.serve.scale_q is not None:
        q = q / cfgs.serve.scale_q
        if jnp.issubdtype(k.dtype, jnp.floating):
            dtype_info = jnp.finfo(k.dtype)
            minval = float(dtype_info.min)
            maxval = float(dtype_info.max)
            q = jnp.clip(q, min=minval, max=maxval)
        q = q.astype(k.dtype)

    q_rows = 2 * tq if split_qk else tq
    if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        s = k.shape[-1]
        qk = lax.dot(
            q.reshape(-1, q_rows, h_size),
            k.reshape(-1, h_size, s),
            dimension_numbers=(([2], [1]), ([0], [0])),
            preferred_element_type=jnp.float32,
        )
    else:
        s = k.shape[-2]
        qk = lax.dot(
            q.reshape(-1, q_rows, h_size),
            k.reshape(-1, s, h_size),
            dimension_numbers=(([2], [2]), ([0], [0])),
            preferred_element_type=jnp.float32,
        )
    if split_qk:
        qk = _merge_split(qk, tq)
    qk = qk.reshape(b, k_heads, tq, s)

    if folded_scale is not None:
        pass  # Already applied to q.
    elif diet or native_qk:
        # Fold the static scalars together so the [TQ, S] f32 tile is scaled
        # once; a per-token k_scale is only [B, KV, 1, S].
        scale = float(cfgs.model.sm_scale)
        if cfgs.serve.scale_q is not None and not native_qk:
            scale *= float(cfgs.serve.scale_q)
        if k_scale is None:
            pass
        elif isinstance(k_scale, (int, float)):
            scale *= float(k_scale)
        elif row_scale is None:
            k_scale = k_scale * scale
            scale = 1.0
        if row_scale is not None:
            qk *= row_scale.reshape(b, k_heads, tq, 1) * scale
        elif scale != 1.0:
            qk *= scale
        if k_scale is not None and not isinstance(k_scale, (int, float)):
            qk *= k_scale
    else:
        qk *= cfgs.model.sm_scale
        if cfgs.serve.scale_q is not None:
            qk *= cfgs.serve.scale_q
        if k_scale is not None:
            qk *= k_scale

    # We convert to the output dtype after scaling, because especially for very
    # low precision, we want to have 32-bit scale factors and do scaling in fp32.
    qk = qk.astype(cfgs.serve.dtype_out)

    if cfgs.model.soft_cap is not None:
        qk = cfgs.model.soft_cap * jnp.tanh(qk / cfgs.model.soft_cap)

    if custom_mask is not None:
        qk_masked = []
        for b_idx in range(cfgs.block.batch_size):
            mask_b = custom_mask[b_idx]
            qk_masked.append(jnp.where(mask_b, qk[b_idx], cfgs.model.mask_value))
        qk = jnp.stack(qk_masked, axis=0)

    m_curr = jnp.max(qk, axis=-1, keepdims=True)

    alpha_list = []
    m_next_list = []

    for b_idx in range(cfgs.block.batch_size):
        m_curr_b = m_curr[b_idx]
        m_next_b = jnp.maximum(m_prev, m_curr_b)
        alpha_b = jnp.exp(m_prev - m_next_b)
        alpha_list.append(alpha_b)
        m_next_list.append(m_next_b)
        m_prev = jnp.where(is_last_k[step, b_idx], -jnp.inf, m_next_b)

    m_next = jnp.stack(m_next_list, axis=0)
    p = jnp.exp(qk - utils.broadcast_minor(m_next, qk.shape))
    p_rowsum = jnp.sum(p, axis=-1, keepdims=True, dtype=cfgs.serve.dtype_out)

    l_next_list = []
    for b_idx in range(cfgs.block.batch_size):
        l_next_b = alpha_list[b_idx] * l_prev + p_rowsum[b_idx]
        l_next_list.append(l_next_b)
        l_prev = l_next_b

    l_next = jnp.stack(l_next_list, axis=0)

    return p, alpha_list, m_next, l_next, m_prev


def flash_attention_pv(
    p: jax.Array,  # [B, KV, TQ, S]
    v: jax.Array,  # [B, KV, S, H] or [B, KV, H, S]
    alpha_list: list[jax.Array],  # B * [KV, TQ, 128]
    o_prev: jax.Array,  # [KV, TQ, H]
    cfgs: configs.RpaConfigs,
    v_scale: jax.Array | None = None,  # [B, KV, S]
):
    """Flash attention kernel."""
    b, k_heads, tq, s = p.shape

    # Because we sum along the sequence dimension in this matmul, we must do the
    # per-token scaling before the matmul. Therefore, there is separate logic
    # here for per-token and per-tensor scaling, unlike the qk matmul.
    if cfgs.serve.per_token_scale:
        assert v_scale is not None
        v_sc = v_scale[:, :, jnp.newaxis, :]
        # We can scale p or v, and because p will be smaller especially for decode
        # workflows, we scale p to save some time.
        p = p.astype(v_sc.dtype) * v_sc

    p_row_scale = None
    native_pv = _use_native_fp8(cfgs, "pv")
    split_pv = native_pv and cfgs.serve.native_fp8_split("pv")
    if native_pv:
        p, p_row_scale = _quantize_rows(p, v.dtype, split=split_pv)
    p_rows = 2 * tq if split_pv else tq

    if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        h_size = v.shape[-2]
        pv = lax.dot(
            p.reshape(-1, p_rows, s),
            v.reshape(-1, h_size, s),
            dimension_numbers=(([2], [2]), ([0], [0])),
            preferred_element_type=jnp.float32,
        )
    else:
        h_size = v.shape[-1]
        pv = lax.dot(
            p.reshape(-1, p_rows, s),
            v.reshape(-1, s, h_size),
            dimension_numbers=(([2], [1]), ([0], [0])),
            preferred_element_type=jnp.float32,
        )
    if split_pv:
        pv = _merge_split(pv, tq)
    if p_row_scale is not None:
        pv = pv * p_row_scale.reshape(-1, tq, 1)
    pv = pv.astype(cfgs.serve.dtype_out).reshape(b, k_heads, tq, h_size)

    if not cfgs.serve.per_token_scale and cfgs.serve.scale_v is not None:
        pv *= cfgs.serve.scale_v

    o_next_list = []
    for b_idx in range(cfgs.block.batch_size):
        alpha_b = utils.broadcast_minor(alpha_list[b_idx], o_prev.shape)
        o_next_b = alpha_b * o_prev + pv[b_idx]
        o_next_list.append(o_next_b)
        o_prev = o_next_b
    o_next = jnp.stack(o_next_list, axis=0)

    return o_next

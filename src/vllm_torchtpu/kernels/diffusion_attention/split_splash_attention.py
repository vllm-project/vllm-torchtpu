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
"""LSE-merged Split-Splash Pallas kernel for prefix-masked joint diffusion SDPA."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel,
    splash_attention_mask,
)

MIN_SPLASH_BLOCK_SIZE = 128
MIN_SPLASH_SEQ_LEN = 512
MAX_COMPUTE_BLOCK_SIZE = 512
SPLASH_BLOCK_CANDIDATES = (1024, 512, 256, 128)


def select_splash_block_size(img_seq_len: int) -> int | None:
    """Return the largest Splash tile size dividing ``img_seq_len``, or None."""
    if img_seq_len < MIN_SPLASH_SEQ_LEN:
        return None
    for block_size in SPLASH_BLOCK_CANDIDATES:
        if img_seq_len >= block_size and img_seq_len % block_size == 0:
            return block_size
    return None


def is_split_splash_supported(
    q_seq_len: int,
    kv_seq_len: int,
    head_dim: int,
    q_heads: int,
    kv_heads: int,
    causal: bool,
    prefix_len: int,
) -> bool:
    """Return True if prefix-masked Split-Splash supports this geometry."""
    if (
        causal
        or q_seq_len != kv_seq_len
        or prefix_len <= 0
        or prefix_len >= q_seq_len
        or q_heads != kv_heads
        or head_dim <= 0
        or head_dim % MIN_SPLASH_BLOCK_SIZE != 0
    ):
        return False
    return select_splash_block_size(q_seq_len - prefix_len) is not None


@functools.partial(jax.jit, static_argnames=("prefix_len", "softmax_scale"))
def split_splash_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attn_mask: jax.Array,
    *,
    prefix_len: int,
    softmax_scale: float | None = None,
) -> jax.Array:
    """Run LSE-merged Split-Splash on prefix-masked ``(B, S_txt + S_img, H, D)``.

    Splits ``S = prefix_len + S_img`` into:
      1. Unmasked ``S_img x S_img`` image self-attention via Pallas
         ``SplashAttention`` with ``save_residuals=True``.
      2. Key-masked ``S_img x prefix_len`` image-to-text cross-attention in
         FP32, merged with (1) via exact log-sum-exp.
      3. Key-masked ``prefix_len x S`` text query attention via
         ``jax.nn.dot_product_attention``.

    Callers must supply the exact ``prefix_len`` (``S_txt``) and guarantee that
    all ``False`` entries of ``attn_mask`` lie within ``[:, :prefix_len]`` (the
    ``S_img`` suffix keys are unmasked).
    """
    if query.ndim != 4:
        raise ValueError(f"Expected 4D (B, S, H, D) query, got {query.shape}.")
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError(
            f"key {key.shape} and value {value.shape} must match query {query.shape}."
        )
    b, s, h, d = query.shape
    if prefix_len <= 0 or prefix_len >= s:
        raise ValueError(f"prefix_len {prefix_len} must satisfy 0 < prefix_len < {s}.")
    if d <= 0 or d % MIN_SPLASH_BLOCK_SIZE != 0:
        raise ValueError(
            f"Head dim {d} must be a positive multiple of {MIN_SPLASH_BLOCK_SIZE}."
        )
    if attn_mask.shape != (b, s) or attn_mask.dtype != jnp.bool_:
        raise ValueError(
            f"Expected bool attn_mask of shape ({b}, {s}), got "
            f"{attn_mask.shape} {attn_mask.dtype}."
        )

    s_txt = prefix_len
    s_img = s - s_txt
    bs = select_splash_block_size(s_img)
    if bs is None:
        raise ValueError(
            f"Image suffix length {s_img} must be >= {MIN_SPLASH_SEQ_LEN} and "
            f"divisible by {MIN_SPLASH_BLOCK_SIZE}."
        )

    scale = d**-0.5 if softmax_scale is None else softmax_scale

    full_img = splash_attention_mask.FullMask((s_img, s_img))
    multi_img = splash_attention_mask.MultiHeadMask(masks=[full_img] * h)
    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=bs,
        block_kv=bs,
        block_kv_compute=min(bs, MAX_COMPUTE_BLOCK_SIZE),
    )
    kern_img = splash_attention_kernel.make_splash_mha(
        mask=multi_img,
        block_sizes=block_sizes,
        save_residuals=True,
        head_shards=1,
        q_seq_shards=1,
    )

    qh = jnp.swapaxes(query, 1, 2)
    kh = jnp.swapaxes(key, 1, 2)
    vh = jnp.swapaxes(value, 1, 2)

    q_img_f32 = qh[:, :, s_txt:, :].astype(jnp.float32) * scale
    q_img = q_img_f32.astype(query.dtype)
    k_img, v_img = kh[:, :, s_txt:, :], vh[:, :, s_txt:, :]
    k_txt, v_txt = kh[:, :, :s_txt, :], vh[:, :, :s_txt, :]

    out_self, res = jax.vmap(kern_img)(q_img, k_img, v_img)
    lse_self = res[0].astype(jnp.float32)

    logits = jnp.einsum(
        "bhsd,bhtd->bhst",
        q_img_f32,
        k_txt.astype(jnp.float32),
    )
    tmask = attn_mask[:, :s_txt]
    logits = jnp.where(tmask[:, None, None, :], logits, -1e30)
    m_txt = jnp.max(logits, axis=-1, keepdims=True)
    exp_txt = jnp.exp(logits - m_txt)
    sum_exp = jnp.sum(exp_txt, axis=-1, keepdims=True)
    lse_txt = (m_txt[..., 0] + jnp.log(sum_exp[..., 0])).astype(jnp.float32)
    out_txt_part = jnp.einsum(
        "bhst,bhtd->bhsd",
        exp_txt / sum_exp,
        v_txt.astype(jnp.float32),
    )

    lse_max = jnp.maximum(lse_self, lse_txt)
    w_s = jnp.exp(lse_self - lse_max)[..., None]
    w_t = jnp.exp(lse_txt - lse_max)[..., None]
    out_img = (
        (w_s * out_self.astype(jnp.float32) + w_t * out_txt_part) / (w_s + w_t)
    ).astype(query.dtype)

    out_txt = jax.nn.dot_product_attention(
        query[:, :s_txt, :, :],
        key,
        value,
        mask=attn_mask[:, None, None, :],
        scale=scale,
    ).astype(query.dtype)

    return jnp.concatenate([out_txt, jnp.swapaxes(out_img, 1, 2)], axis=1)

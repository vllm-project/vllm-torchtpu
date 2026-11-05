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
"""Reference math of Kimi-K3.1's DSA indexer, as vLLM's GPU model computes it
(BF16 projections, FP32 key LayerNorm rounded to BF16, FP8 queries whose
power-of-two scales are folded into the head weights), and of the MLA decode
attention over its selection as the K3 decode megakernel computes it (in latent
space), in JAX on any backend. Keys arrive dequantized from the index cache
(``dsa_cache.dequantize_index_keys``)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from vllm_torchtpu.megakernel.kimi_k3.dsa import INDEX_DIM, INDEX_HEADS
from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache


def _bf16_dot(a, b):
    return jnp.dot(
        a, b, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)


def key_from_projection(projected, weight, bias, eps):
    """The key LayerNorm of ``x @ wk^T`` (BF16), in FP32, rounded to BF16."""
    key = projected.astype(jnp.float32)
    mean = jnp.mean(key, axis=-1, keepdims=True)
    centered = key - mean
    variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
    normed = centered * jax.lax.rsqrt(variance + eps)
    return (normed * weight.astype(jnp.float32) + bias.astype(jnp.float32)).astype(
        jnp.bfloat16
    )


def index_key(x, wk_t, weight, bias, eps):
    return key_from_projection(_bf16_dot(x, wk_t), weight, bias, eps)


def index_query(q_c, wq_t, x, wv_t):
    """``(q, w)``: the FP8 query heads (as FP32 values) and the head weights
    with ``heads^-0.5``, each head's query scale and ``128^-0.5`` folded in."""
    q = _bf16_dot(q_c, wq_t).astype(jnp.float32).reshape(-1, INDEX_HEADS, INDEX_DIM)
    q, scale = dsa_cache.quantize_index_queries(q)
    v = (_bf16_dot(x, wv_t).astype(jnp.float32) * INDEX_HEADS**-0.5).astype(
        jnp.bfloat16
    )
    return q, v.astype(jnp.float32) * scale * INDEX_DIM**-0.5


def scores(q, w, keys):
    """``[T, L]``: ``sum_h w[t, h] ReLU(q[t, h] . keys[l])`` in FP32."""
    precision = jax.lax.Precision.HIGHEST
    logits = jnp.einsum(
        "thd,ld->thl", q, keys, precision=precision, preferred_element_type=jnp.float32
    )
    return jnp.einsum(
        "th,thl->tl",
        w,
        jax.nn.relu(logits),
        precision=precision,
        preferred_element_type=jnp.float32,
    )


def select_topk(values, lengths, k):
    """The top ``k`` token indices of ``values[t, :lengths[t]]`` per row,
    higher value first, ties to the lower index; ``-1`` past the row's length."""
    index = jnp.arange(values.shape[-1])
    masked = jnp.where(index[None] < lengths[:, None], values, -jnp.inf)
    _, chosen = jax.lax.top_k(masked, min(k, values.shape[-1]))
    return jnp.where(chosen < lengths[:, None], chosen, -1).astype(jnp.int32)


def sparse_mla_attention(q_nope, q_rope, w_uk, latents, selected, new_latent):
    """One query row's latent-space MLA decode attention ``[heads, 512]``
    (FP32), as the decode kernel computes it (``mla_dcp`` steps 1-4), over the
    cached latent rows ``latents[selected]`` (``[512 normalized | 64 rope]``;
    ``-1`` entries ignored) and the new token's row ``new_latent``, always a
    candidate. ``W_UK`` (``w_uk [heads, 512, 128]``: head ``h``'s keys are
    ``C @ w_uk[h]``) is absorbed into the query, ``q_abs = [q_nope @ W_UK^T |
    q_rope] * 192^-0.5``; the softmax is FP32 and ``O = P @ C[:, :512]`` (the
    caller applies ``W_UV``)."""
    precision = jax.lax.Precision.HIGHEST
    absorbed = jnp.einsum(
        "hd,hld->hl",
        q_nope,
        w_uk,
        precision=precision,
        preferred_element_type=jnp.float32,
    )
    query = jnp.concatenate((absorbed, q_rope.astype(jnp.float32)), axis=-1) * (
        (128 + 64) ** -0.5
    )
    valid = jnp.concatenate((selected >= 0, jnp.ones((1,), bool)))
    rows = jnp.concatenate(
        (latents[jnp.maximum(selected, 0), :576], new_latent[None, :576])
    ).astype(jnp.float32)
    rows = jnp.where(valid[:, None], rows, 0.0)
    logits = jnp.einsum(
        "hl,kl->hk",
        query,
        rows,
        precision=precision,
        preferred_element_type=jnp.float32,
    )
    probabilities = jax.nn.softmax(jnp.where(valid[None], logits, -jnp.inf), axis=-1)
    return jnp.einsum(
        "hk,kl->hl",
        probabilities,
        rows[:, :512],
        precision=precision,
        preferred_element_type=jnp.float32,
    )


def finite_bf16_keys():
    """Every finite BF16 value once, as 510 keys of 128 lanes (numpy BF16)."""
    import ml_dtypes
    import numpy as np

    bits = np.arange(1 << 16, dtype=np.uint16)
    return bits[(bits & 0x7F80) != 0x7F80].view(ml_dtypes.bfloat16).reshape(-1, 128)


def every_amax_keys():
    """One key per non-negative finite BF16 amax: lane 0 the amax, lane 1 its
    negation (``-amax * 0.999`` rounds back to ``-amax`` in BF16, a tie for the
    largest magnitude), the rest spread over [0, amax] (numpy BF16)."""
    import ml_dtypes
    import numpy as np

    bf16 = ml_dtypes.bfloat16
    bits = np.arange(1 << 15, dtype=np.uint16)
    amax = bits[(bits & 0x7F80) != 0x7F80].view(bf16).astype(np.float32)
    keys = (amax[:, None] * (np.arange(128, dtype=np.float32) / 127)).astype(bf16)
    keys[:, 0] = amax.astype(bf16)
    keys[:, 1] = (-amax * np.float32(0.999)).astype(bf16)
    return keys


def global_token(local, owner, page_tokens: int, group: int):
    """The global index of local token ``local`` on the DCP member holding the
    global pages ``g % group == owner`` (its local page ``j`` is global page
    ``j group + owner``)."""
    return ((local // page_tokens) * group + owner) * page_tokens + local % page_tokens


def decode_selection(
    local_scores, positions, *, page_tokens: int, group: int, k: int = 2048
):
    """The decode select's reference: each row's candidates are its tokens
    ``0 .. position`` (the new token, index ``position``, included); a row of
    at most ``k`` selects all of them, else the top ``k`` by score with ties
    to the lower global index (:func:`select_topk`). ``local_scores[o]``
    ``[rows, T]``: member ``o``'s scores of its local tokens below its
    ``dsa_decode.local_length``. Returns ``(masks, new_selected)``: per member
    a bool ``[rows, T]`` of its selected local tokens without the new token,
    and per row whether the new token is selected."""
    import numpy as np

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    rows, tokens = np.asarray(local_scores[0]).shape
    masks = [np.zeros((rows, tokens), bool) for _ in range(group)]
    new_selected = np.zeros(rows, bool)
    for r, position in enumerate(positions):
        total = int(position) + 1
        values = np.full(total, np.nan, np.float32)
        for o in range(group):
            length = int(dsa_decode.local_length(position, o, page_tokens, group))
            local = np.arange(length)
            values[global_token(local, o, page_tokens, group)] = np.asarray(
                local_scores[o]
            )[r, :length]
        if total <= k:
            chosen = np.arange(total)
        else:
            assert not np.isnan(values).any(), "every candidate has a score"
            chosen = np.asarray(
                select_topk(jnp.asarray(values[None]), jnp.asarray([total]), k)
            )[0]
        new_selected[r] = bool((chosen == position).any())
        chosen = chosen[chosen != position]
        page = chosen // page_tokens
        owner, local = page % group, page // group * page_tokens + chosen % page_tokens
        for o in range(group):
            masks[o][r, local[owner == o]] = True
    return masks, new_selected


def latent_attention(query, rows, new_row=None):
    """``(o [H, 512], lse [H])`` in FP64 (numpy): the decode attention of
    absorbed queries ``query [H, 640]`` (score scale folded in) over latent
    rows ``rows [n, 640]`` and, when given, the new token's row (lanes
    576:640 of both ignored); ``(0, mla_dcp.NEG)`` with no rows at all."""
    import numpy as np

    from vllm_torchtpu.megakernel.kimi_k3.kernels.mla_dcp import NEG

    every = np.asarray(rows, np.float64).reshape(-1, 640)
    if new_row is not None:
        every = np.concatenate((every, np.asarray(new_row, np.float64)[None]))
    query = np.asarray(query, np.float64)
    if not len(every):
        return np.zeros((len(query), 512)), np.full(len(query), NEG)
    logits = query[:, :576] @ every[:, :576].T
    top = logits.max(axis=1, keepdims=True)
    weights = np.exp(logits - top)
    total = weights.sum(axis=1, keepdims=True)
    return (weights / total) @ every[:, :512], (top + np.log(total))[:, 0]

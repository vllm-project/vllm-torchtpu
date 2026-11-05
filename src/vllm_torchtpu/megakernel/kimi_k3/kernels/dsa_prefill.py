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
"""Kimi-K3.1's DSA on the prefill stages: the window's index keys (this
module's first half), then each query row's scores, top-k selection and its
bitmask (the stage MLA layer).

A key is encoded as the decoder commits it (``dsa_cache``): the LayerNorm of
the row's key projection, in FP32 rounded to BF16, as FP8 E4M3 with a UE8M0
power-of-two scale. Under the rope-probe synthetic indexer the projection is
``[rope, -rope]`` of the row's latent rope lanes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from vllm_torchtpu.megakernel.kimi_k3.dsa import INDEX_DIM, INDEX_HEADS
from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache

_ROPE = slice(512, 576)  # the latent row's rope lanes


def index_keys(x, latent, wk_t, knorm, *, rope_probe: bool, eps: float):
    """``(bytes uint8 [T, 128], exponents uint8 [T])``: the index keys of
    rows ``x`` (``[T, hidden]``, the MLA layer's normalized input) whose
    compressed KV is ``latent`` (``[T, 576]``), with the key projection
    ``wk_t`` (``[hidden, 128]``, ``x -> k``) and the key LayerNorm ``knorm``
    (``[2, 128]``: weight, bias)."""
    if rope_probe:
        rope = latent[:, _ROPE]
        projected = jnp.concatenate((rope, -rope), axis=-1)
    else:
        projected = jax.lax.dot_general(
            x, wk_t, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)
    key = projected.astype(jnp.float32)
    centered = key - jnp.mean(key, axis=-1, keepdims=True)
    variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
    normed = centered * jax.lax.rsqrt(variance + eps)
    weight, bias = (knorm[i].astype(jnp.float32) for i in range(2))
    return dsa_cache.quantize_index_keys((normed * weight + bias).astype(jnp.bfloat16))


def index_pages(key_bytes, exponents, page_tokens: int):
    """``[T / page_tokens, page_tokens + 32, 128]`` uint8: keys ``key_bytes``
    (``[T, 128]``) and ``exponents`` (``[T]``) laid out as the decoder's
    index pages, a page's keys then its exponent rows."""
    pages = key_bytes.shape[0] // page_tokens
    rows = page_tokens // 128
    scales = exponents.reshape(pages, rows, 128)
    scales = jnp.pad(scales, ((0, 0), (0, dsa_cache.SCALE_ROWS - rows), (0, 0)))
    return jnp.concatenate(
        (key_bytes.reshape(pages, page_tokens, 128), scales), axis=1
    )


def commit_index_pages(index, key_bytes, exponents, table, page_tokens: int):
    """``index`` (a stage's index-key cache, ``[blocks, L, split * (P + 32),
    128]``: block ``b`` holds decode pages ``split * b ..``) with each window
    page ``p`` of layer slot ``j``'s keys (``key_bytes [L, T, 128]``,
    ``exponents [L, T]``) written into decode page ``table[p]`` of slot ``j``."""
    rows = page_tokens + dsa_cache.SCALE_ROWS
    split = index.shape[2] // rows
    for j in range(index.shape[1]):
        pages = index_pages(key_bytes[j], exponents[j], page_tokens)

        def write(p, index, pages=pages, j=j):
            page = table[p]
            return jax.lax.dynamic_update_slice(
                index, pages[p][None, None], (page // split, j, page % split * rows, 0)
            )

        index = jax.lax.fori_loop(0, pages.shape[0], write, index)
    return index


def index_query(q_c, wq_t, x, wv_t):
    """``(q f32 [R, 32, 128], w f32 [R, 32])``: rows ``q_c`` (the normalized
    query latent, ``[R, 1536]``) through the index heads ``wq_t`` (``[1536,
    32 * 128]``) as FP8 values with each head's power-of-two scale, and the
    head weights of rows ``x`` (the MLA layer's normalized input) through
    ``wv_t`` (``[hidden, 32]``) rounded to BF16, times ``32^-0.5`` rounded to
    BF16, with the head's scale and ``128^-0.5`` folded in."""
    q = jax.lax.dot_general(q_c, wq_t, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    q = q.astype(jnp.bfloat16).astype(jnp.float32).reshape(-1, INDEX_HEADS, INDEX_DIM)
    q, scale = dsa_cache.quantize_index_queries(q)
    v = jax.lax.dot_general(x, wv_t, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    v = (v.astype(jnp.bfloat16).astype(jnp.float32) * INDEX_HEADS**-0.5).astype(jnp.bfloat16)
    return q, v.astype(jnp.float32) * scale * INDEX_DIM**-0.5


def candidate_valid(segments, length, *, count, tokens, capacity, rows=None):
    """The flash kernel's admission rule as a bool ``[tokens, capacity + tokens]`` over its ``[arena | window]``
    columns, from the ``[5 * count]`` descriptor (offsets, lengths, checkpoints, arena bases, cached lengths): a
    row of sequence ``s`` below ``length`` sees arena columns ``[base_s, base_s + cached_s)`` and window columns
    ``capacity + [offset_s, row]``; other rows see none. ``rows``: only these window rows (``[R]``), in order."""
    rows = (jnp.arange(tokens) if rows is None else rows)[:, None]
    columns = jnp.arange(capacity + tokens)[None]
    window = columns - capacity
    valid = jnp.zeros((rows.shape[0], capacity + tokens), bool)
    for s in range(count):
        offset, end = segments[s], segments[s] + segments[count + s]
        base, cached = segments[3 * count + s], segments[4 * count + s]
        inside = (rows >= offset) & (rows < end) & (rows < length[0])
        seen = jnp.where(columns < capacity, (columns >= base) & (columns < base + cached),
                         (window >= offset) & (window <= rows))
        valid = valid | (inside & seen)
    return valid


def candidate_scores(q, w, keys):
    """``[R, C]`` FP32 index scores ``sum_h w[r, h] ReLU(q[r, h] . keys[c])`` (BF16 keys and queries: exact for
    FP8 values times power-of-two scales; FP32 accumulation), ``R / heads`` rows at a time, so a step's per-head
    logits are no larger than the ``[R, C]`` result."""
    keys = keys.astype(jnp.bfloat16)

    def row(operands):
        q_row, w_row = operands
        logits = jax.lax.dot_general(q_row.astype(jnp.bfloat16), keys, (((1,), (1,)), ((), ())),
                                     preferred_element_type=jnp.float32)
        return jnp.sum(w_row[:, None] * jax.nn.relu(logits), axis=0)

    return jax.lax.map(row, (q, w), batch_size=max(1, q.shape[0] // q.shape[1]))


def order_keys(scores):
    """Order-preserving uint32 keys of FP32 scores (no NaNs; -inf lowest;
    -0 and +0 equal, so zero scores tie by column)."""
    bits = jax.lax.bitcast_convert_type(scores.astype(jnp.float32), jnp.uint32)
    bits = jnp.where(bits == jnp.uint32(0x80000000), jnp.uint32(0), bits)
    return jnp.where(bits >> 31 == 1, ~bits, bits | jnp.uint32(0x80000000))


def topk_threshold(keys, valid, k):
    """Per row, ``(tau, quota)``: the ``k``-th largest valid key and how many
    keys equal to it the top ``k`` takes (rows of at most ``k`` valid keys:
    unused). A bisection over the key bits, from the most significant: each
    pass keeps the next bit when at least ``k`` valid keys reach it."""
    keys = jnp.where(valid, keys, jnp.uint32(0))  # below every valid key

    def bit(i, tau):
        candidate = tau | (jnp.uint32(1) << (31 - i).astype(jnp.uint32))
        count = jnp.sum(keys >= candidate[:, None], axis=1)
        return jnp.where(count >= k, candidate, tau)

    tau = jax.lax.fori_loop(0, 32, bit, jnp.zeros(keys.shape[:1], jnp.uint32))
    return tau, k - jnp.sum(keys > tau[:, None], axis=1)


def select_mask(scores, valid, k):
    """Each row's top ``k`` valid candidates, ties to the lower column; rows
    of at most ``k`` valid candidates select them all."""
    keys = order_keys(scores)
    tau, quota = topk_threshold(keys, valid, k)
    ties = valid & (keys == tau[:, None])
    rank = jnp.cumsum(ties, axis=1) - 1
    chosen = valid & ((keys > tau[:, None]) | (ties & (rank < quota[:, None])))
    dense = jnp.sum(valid, axis=1) <= k
    return jnp.where(dense[:, None], valid, chosen)


def pack_bits(mask):
    """uint32 words ``[R, C / 32]`` of a bool ``[R, C]`` (``C`` a multiple of 32): bit ``c % 32`` of word
    ``c // 32``."""
    bits = mask.reshape(*mask.shape[:-1], -1, 32).astype(jnp.uint32)
    return jnp.sum(bits << jnp.arange(32, dtype=jnp.uint32), axis=-1, dtype=jnp.uint32)


def unpack_bits(words, columns):
    """The bool ``[R, columns]`` that :func:`pack_bits` packed into ``words``."""
    bits = (words[..., None] >> jnp.arange(32, dtype=jnp.uint32)) & 1
    return bits.reshape(*words.shape[:-1], -1)[..., :columns].astype(bool)

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
"""Kimi-K3.1's index-key cache: one block-major tensor ``[blocks, 24,
tokens_per_block + 32, 128]`` uint8 beside vLLM's unified pools, addressed by
the MLA layers' block tables: block ``b``'s page of MLA layer ``m`` is
``[b, m]``.

A page holds ``tokens_per_block`` keys (the MLA page's tokens), token ``t``'s
128 FP8 E4M3 key bytes in row ``t``, then the keys' power-of-two scales as
UE8M0 exponent bytes (scale ``2^(e - 127)``), token ``t``'s at row
``tokens_per_block + t // 128``, lane ``t % 128``, in one 32-row tile. Keys
are encoded as vLLM's DSA cache kernel writes them (scale
``2^ceil(log2(max(amax, 1e-4) / 448))``).

The decode kernel encodes its keys itself (:func:`index_key_rows`), in exact
power-of-two arithmetic equal to :func:`quantize_index_keys` for finite keys.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from vllm_torchtpu.megakernel.kimi_k3.dsa import INDEX_DIM

FP8_MAX = 448.0
KEY_AMAX_FLOOR = 1e-4
QUERY_AMAX_FLOOR = 1e-10
SCALE_ROWS = 32  # a uint8 (32, 128) tile


def index_block_shape(tokens_per_block: int) -> tuple[int, int]:
    if tokens_per_block % 128 or tokens_per_block // 128 > SCALE_ROWS:
        raise ValueError(f"index blocks hold whole 128-token rows, not {tokens_per_block}")
    return (tokens_per_block + SCALE_ROWS, INDEX_DIM)


def pow2_ceil_scale(amax):
    """``(scale, exponent)``: the least power of two at least ``amax / 448``
    (``amax`` positive and at least the floor), and its biased exponent byte."""
    bits = jax.lax.bitcast_convert_type((amax / FP8_MAX).astype(jnp.float32), jnp.uint32)
    exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).astype(jnp.uint32)
    return jax.lax.bitcast_convert_type(exponent << 23, jnp.float32), exponent.astype(jnp.uint8)


def _encode(values, floor):
    x = values.astype(jnp.float32)
    amax = jnp.maximum(jnp.max(jnp.abs(x), axis=-1), floor)
    scale, exponent = pow2_ceil_scale(amax)
    q = jnp.clip(x / scale[..., None], -FP8_MAX, FP8_MAX).astype(jnp.float8_e4m3fn)
    return q, scale, exponent


def quantize_index_keys(keys):
    """``(bytes uint8 [..., 128], exponent uint8 [...])`` of 128-lane keys."""
    q, _, exponent = _encode(keys, KEY_AMAX_FLOOR)
    return jax.lax.bitcast_convert_type(q, jnp.uint8), exponent


def power_of_two(exponent):
    """2^exponent (int32, -126..127) as FP32."""
    return jax.lax.bitcast_convert_type((exponent + 127) << 23, jnp.float32)


def encode_index_keys(keys, floor=KEY_AMAX_FLOOR):
    """``(codes f32 [..., 128], exponent int32 [..., 1])``: :func:`quantize_index_keys`'
    encoding of finite 128-lane keys (with ``floor=QUERY_AMAX_FLOOR``,
    :func:`quantize_index_queries`') in exact power-of-two arithmetic that
    lowers in Mosaic (no division, no FP8 type): ``codes`` are the E4M3 values
    (scaled keys rounded to nearest, ties to even), ``exponent`` the scale's
    biased UE8M0 exponent."""
    x = keys.astype(jnp.float32)
    amax = jnp.maximum(jnp.max(jnp.abs(x), axis=-1, keepdims=True), floor)
    bits = jax.lax.bitcast_convert_type(amax, jnp.int32)
    # 2^ceil(log2(amax / 448)) without a division: 448 is 1.75 * 2^8, so the
    # exponent is amax's minus 8, plus one when amax's significand exceeds 1.75.
    exponent = (bits >> 23) - 8 + ((bits & 0x7FFFFF) > 0x600000).astype(jnp.int32)
    v = x * power_of_two(127 - exponent)
    v_bits = jax.lax.bitcast_convert_type(v, jnp.int32)
    # E4M3 steps are 2^(e - 3) in binade 2^e, and 2^-9 below the least normal 2^-6.
    step = jnp.maximum(((v_bits >> 23) & 0xFF) - 127, -6) - 3
    scaled = jnp.abs(v) * power_of_two(-step)  # [0, 16)
    # Adding 2^23 rounds to an integer, ties to even (FP32 addition's rounding).
    count = jax.lax.bitcast_convert_type(scaled + 2.0**23, jnp.int32) - 0x4B000000
    magnitude = count.astype(jnp.float32) * power_of_two(step)
    # v's sign bit, so keys rounded to zero keep it as the FP8 cast does (Mosaic
    # lowers a negation as 0 - x, which turns -0 into +0).
    code_bits = jax.lax.bitcast_convert_type(magnitude, jnp.int32) | (v_bits & ~0x7FFFFFFF)
    return jax.lax.bitcast_convert_type(code_bits, jnp.float32), exponent


def index_key_rows(keys):
    """``(codes, exponents)`` BF16 ``[..., 128]``: the decode kernel's handoff
    of keys encoded by :func:`encode_index_keys`, the exponent in every lane.
    Both are exact in BF16 (E4M3 values have 4 significant bits; the exponent
    is an integer below 256)."""
    codes, exponent = encode_index_keys(keys)
    return (codes.astype(jnp.bfloat16),
            jnp.broadcast_to(exponent.astype(jnp.float32), codes.shape).astype(jnp.bfloat16))


def index_key_bytes(codes, exponents):
    """``(bytes uint8 [..., 128], exponent uint8 [...])`` of
    :func:`index_key_rows`' handoff, as :func:`quantize_index_keys` returns them."""
    return (jax.lax.bitcast_convert_type(codes.astype(jnp.float8_e4m3fn), jnp.uint8),
            exponents[..., 0].astype(jnp.int32).astype(jnp.uint8))


def dequantize_index_keys(key_bytes, exponent):
    q = jax.lax.bitcast_convert_type(key_bytes, jnp.float8_e4m3fn).astype(jnp.float32)
    scale = jax.lax.bitcast_convert_type(exponent.astype(jnp.uint32) << 23, jnp.float32)
    return q * scale[..., None]


def quantize_index_queries(queries):
    """``(fp8 values as f32 [..., 128], scale f32 [...])``: each 128-lane head
    with its own power-of-two scale (the GPU indexer's query rule)."""
    q, scale, _ = _encode(queries, QUERY_AMAX_FLOOR)
    return q.astype(jnp.float32), scale


def commit_index_rows(index_cache, key_bytes, exponents, block_tables, positions, *, member, group: int,
                      global_pages: bool):
    """Write each row's encoded index keys (``key_bytes [rows, M, 128]`` and
    ``exponents [rows, M]``, MLA layer ``m``'s into page ``[block, m]`` of
    ``index_cache [blocks, M, tokens + 32, 128]``) on the DCP member
    ``member`` of ``group`` that owns the row's position, into the token slot
    its latent row takes (``paged.commit_latent_rows``). Rows of other members
    are dropped."""
    rows, layers = key_bytes.shape[:2]
    tokens = index_cache.shape[2] - SCALE_ROWS
    position = positions[:rows]
    pages = position // tokens
    owned = pages % group == member
    blocks = block_tables[jnp.arange(rows), pages if global_pages else pages // group]
    target = jnp.where(owned, blocks, index_cache.shape[0])[:, None]
    layer = jnp.arange(layers)[None, :]
    slot = (position % tokens)[:, None]
    index_cache = index_cache.at[target, layer, slot].set(key_bytes, mode="drop")
    return index_cache.at[target, layer, tokens + slot // 128, slot % 128].set(exponents, mode="drop")

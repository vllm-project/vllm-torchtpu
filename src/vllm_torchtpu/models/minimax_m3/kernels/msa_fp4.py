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
"""4-bit MSA caches (``--kv-cache-dtype int4_per_token_head``).

Storage. A cache page holds 8 tokens per 32-bit word: the packed MLA layout
of `packed_page` with uint8 storage ``[pages, S / 8, 4, W]``, each byte two
4-bit slots. Slot ``r * 8 + j`` of a page sits in bits ``4j..4j+3`` of word
row ``r``; a width-changing bitcast of the words to int4 on v7x yields the
page's ``S`` rows in slot order. A value is a signed int4 ``q`` in
``[-7, 7]``, read as ``q * scale``.

Scales. One f32 scale per token per head vector, folded into the page as
one extra 32-bit word row (`scales_in_page`): ``[pages, S / 8 + 1, 4, W]``,
the last row read as ``W`` f32 lanes, scale ``j`` of slot ``s`` at lane
``j * S + s``. Index pages (``W = S``) hold one scale per slot; K/V pages of
``G`` KV heads (``W = 2 D G``, ``D = S = 128``) hold ``2G``: ``j = 2h`` the
K and ``j = 2h + 1`` the V scales of head ``h``, so head ``h``'s
``2D``-wide column block carries its own K scales in its first ``S`` lanes
and its V scales in the next ``S`` (`kv_scales`). With one scale per vector
the kernels fold it out of the dot products: a K scale multiplies a score
column, a V scale the probability before ``p @ v``, an index scale the
index score before the block max.

Rotation. Head vectors are rotated by the 128-point Sylvester Hadamard
matrix ``H`` (entries +-1, ``H @ H.T = 128 I``) before quantizing, which
spreads per-channel outliers so one scale per vector suffices. Queries get
``H / 128`` and the attention output ``H.T``, so scores and outputs are
unchanged up to rounding: ``(q H / 128) . (k H) = q . k``. ``H`` and the
1/128 are exact in bf16 and f32; this changes storage precision only, not
the model.
"""

import numpy as np

import jax
import jax.numpy as jnp

TOKENS_PER_WORD = 8
INT4_MAX = 7
SCALE_LANES = 128


def hadamard(n: int = 128) -> np.ndarray:
    """Sylvester Hadamard matrix of order `n` (entries +-1, ``H H^T = n I``)."""
    h = np.ones((1, 1), np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    assert h.shape[0] == n, n
    return h


def _hadamard_dot(x: jax.Array, h: np.ndarray) -> jax.Array:
    """``x @ h`` in f32 for a +-1 matrix ``h``. A bf16 `x` takes one bf16 MXU
    pass with f32 accumulation, which is exact here (every product is a
    signed bf16 value); only wider inputs need the 6-pass HIGHEST f32 dot."""
    if x.dtype == jnp.bfloat16:
        return jnp.dot(
            x, jnp.asarray(h, jnp.bfloat16), preferred_element_type=jnp.float32
        )
    return jnp.dot(
        x.astype(jnp.float32),
        jnp.asarray(h, jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )


def rotate_keys(x: jax.Array) -> jax.Array:
    """``x @ H`` over the last (head) axis, for keys and values."""
    return _hadamard_dot(x, hadamard(x.shape[-1]))


def rotate_queries(x: jax.Array) -> jax.Array:
    """``x @ H / n``: pairs with `rotate_keys` to leave dot products unchanged."""
    n = x.shape[-1]
    return (rotate_keys(x) * (1.0 / n)).astype(x.dtype)


def unrotate_output(x: jax.Array) -> jax.Array:
    """``x @ H^T / n``: undoes `rotate_keys` on an attention output over rotated values."""
    n = x.shape[-1]
    return (_hadamard_dot(x, hadamard(n).T) * (1.0 / n)).astype(x.dtype)


def quantize(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``[..., D]`` -> (int4 values as uint8 nibbles ``[..., D]``, f32 scales ``[...]``).

    Symmetric round-to-nearest-even with ``scale = amax / 7``; an all-zero
    vector gets scale 0 and zero values.
    """
    x = x.astype(jnp.float32)
    amax = jnp.max(jnp.abs(x), axis=-1)
    scale = amax / INT4_MAX
    inv = jnp.where(amax > 0, INT4_MAX / jnp.where(amax > 0, amax, 1.0), 0.0)
    q = jnp.clip(jnp.round(x * inv[..., None]), -INT4_MAX, INT4_MAX).astype(jnp.int32)
    return (q & 0xF).astype(jnp.uint8), scale


def dequantize(nibbles: jax.Array, scale: jax.Array) -> jax.Array:
    """Inverse of `quantize`: f32 ``nibbles`` read as int4, times ``scale[..., None]``."""
    q = nibbles.astype(jnp.int32)
    q = jnp.where(q >= 8, q - 16, q)
    return q.astype(jnp.float32) * scale[..., None]


def is_fp4(cache: jax.Array) -> bool:
    """4-bit caches are the only uint8 MSA caches."""
    return cache.dtype == jnp.uint8


def scales_in_page(rows_per_page: int, dtype) -> bool:
    """Whether a 4-bit page carries its per-slot scales in an extra 32-bit word
    row: ``[pages, S / 8 + 1, 4, W]``, the last row ``W`` f32 lanes, lane ``s``
    the scale of slot ``s`` (index caches: one DMA per page instead of a page
    plus a scale page). S / 8 is even, so an odd row count means folded."""
    return dtype == jnp.uint8 and rows_per_page % 2 == 1


def data_rows(rows_per_page: int) -> int:
    """Word rows of a 4-bit page that hold tokens."""
    return rows_per_page - rows_per_page % 2


def page_tokens(cache: jax.Array) -> int:
    """Tokens per page of a uint8 ``[pages, S / 8 (+ 1), 4, W]`` 4-bit cache."""
    return data_rows(cache.shape[1]) * TOKENS_PER_WORD


def folded_scales(cache: jax.Array) -> jax.Array:
    """``[pages, S / 8 + 1, 4, W]`` folded page -> f32 ``[pages, W]`` slot
    scales of its last word row (XLA reference)."""
    words = jax.lax.bitcast_convert_type(cache[:, -1].transpose(0, 2, 1), jnp.uint32)
    return jax.lax.bitcast_convert_type(words, jnp.float32)


def kv_scales(cache: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Folded 4-bit K/V pages ``[pages, S / 8 + 1, 4, 2 D G]`` -> f32 K and V
    scales ``[pages, G, S]`` of every head and slot (XLA reference)."""
    pages, rows, _, width = cache.shape
    tokens = data_rows(rows) * TOKENS_PER_WORD
    heads = width // (2 * tokens) if width % (2 * tokens) == 0 else 0
    if not scales_in_page(rows, cache.dtype) or heads == 0:
        raise ValueError(f"not a folded 4-bit K/V cache: {cache.shape} {cache.dtype}")
    lanes = folded_scales(cache)[:, : 2 * heads * tokens]
    lanes = lanes.reshape(pages, heads, 2, tokens)
    return lanes[:, :, 0], lanes[:, :, 1]


def unpack_rows(cache: jax.Array) -> jax.Array:
    """``[pages, S / 8, 4, W]`` uint8 -> ``[pages, S, W]`` nibbles in slot order (XLA reference)."""
    cache = cache[:, : data_rows(cache.shape[1])]
    pages, rows, _, width = cache.shape
    words = jax.lax.bitcast_convert_type(
        cache.transpose(0, 1, 3, 2), jnp.uint32
    )  # [pages, S / 8, W]
    shifts = jnp.arange(TOKENS_PER_WORD, dtype=jnp.uint32) * 4
    nib = (words[:, :, None, :] >> shifts[None, None, :, None]) & 0xF
    return nib.reshape(pages, rows * TOKENS_PER_WORD, width).astype(jnp.uint8)


def pack_rows(nibbles: jax.Array, scales: jax.Array | None = None) -> jax.Array:
    """Inverse of `unpack_rows`: ``[pages, S, W]`` nibbles -> uint8 ``[pages, S / 8, 4, W]``."""
    pages, tokens, width = nibbles.shape
    nib = nibbles.astype(jnp.uint32).reshape(
        pages, tokens // TOKENS_PER_WORD, TOKENS_PER_WORD, width
    )
    shifts = jnp.arange(TOKENS_PER_WORD, dtype=jnp.uint32) * 4
    words = jnp.bitwise_or.reduce(nib << shifts[None, None, :, None], axis=2)
    if scales is not None:  # fold [pages, W] f32 slot scales in as the last row
        scale_words = jax.lax.bitcast_convert_type(
            scales.astype(jnp.float32), jnp.uint32
        )
        words = jnp.concatenate([words, scale_words[:, None, :]], axis=1)
    return jax.lax.bitcast_convert_type(words, jnp.uint8).transpose(0, 1, 3, 2)

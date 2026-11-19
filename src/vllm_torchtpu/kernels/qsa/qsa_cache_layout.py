"""Packed fp8 KV-cache layout of QSA decode, and host / XLA-side (un)packing.

Cache layout: `int32[num_pages, page_size * kv_heads, 128]` (512 bytes per
token per KV head), in blocks of `TOKEN_BLOCK = 4` consecutive tokens
(`page_size % 4 == 0`). Row `4 (kv_heads b + h) + p` of a page holds byte
plane `p` of KV head `h` of block `b`'s four tokens; byte j of its word l
belongs to token `4 b + j`:

  p = 0: k[l]   p = 1: k[128 + l]   p = 2: v[l]   p = 3: v[128 + l]

This is the native fp8 sublane packing of `pltpu.bitcast(int32 -> fp8)`, so
the attention kernel gets each plane as a token-ordered fp8 tile with a
stride-`4 kv_heads` row load and a free bitcast -- no per-byte shift / pack.
Every gathered group (4 tokens, page-aligned) is exactly one block, i.e. one
contiguous `kv_heads * 2` KB row holding all KV heads. See `pack_kv_cache`
(`pack_kv_rows` for `kv_heads == 1`).
"""

import jax
from jax import lax
import jax.numpy as jnp

HEAD_DIM = 256
LANES = 128
# int32 words per cache row (one byte plane of one 4-token block).
ROW_WORDS = LANES
CACHE_DTYPE = jnp.float8_e4m3fn
# Tokens per cache block: one int32 word packs this many fp8 bytes.
TOKEN_BLOCK = 4


def pack_kv_rows(k: jax.Array, v: jax.Array) -> jax.Array:
  """fp8 K, V `[..., tokens, 256]` -> int32 cache rows `[..., tokens, 128]`.

  `tokens % TOKEN_BLOCK == 0`; see the module docstring for the block layout.
  """
  assert k.shape == v.shape and k.shape[-1] == HEAD_DIM, (k.shape, v.shape)
  *lead, tokens, _ = k.shape
  assert tokens % TOKEN_BLOCK == 0, tokens
  k = k.astype(CACHE_DTYPE)
  v = v.astype(CACHE_DTYPE)
  planes = (k[..., :LANES], k[..., LANES:], v[..., :LANES], v[..., LANES:])
  words = []
  for plane in planes:
    byte = lax.bitcast_convert_type(plane, jnp.uint8).astype(jnp.uint32)
    byte = byte.reshape(*lead, tokens // TOKEN_BLOCK, TOKEN_BLOCK, LANES)
    word = jnp.zeros(byte.shape[:-2] + (LANES,), jnp.uint32)
    for j in range(TOKEN_BLOCK):
      word = word | (byte[..., j, :] << (8 * j))
    words.append(word)
  rows = jnp.stack(words, axis=-2)  # [..., blocks, plane, 128]
  rows = rows.reshape(*lead, tokens, LANES)
  return lax.bitcast_convert_type(rows, jnp.int32)


def unpack_kv_rows(rows: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Inverse of `pack_kv_rows`: int32 `[..., tokens, 128]` -> fp8 (k, v)."""
  assert rows.shape[-1] == ROW_WORDS, rows.shape
  *lead, tokens, _ = rows.shape
  assert tokens % TOKEN_BLOCK == 0, tokens
  words = lax.bitcast_convert_type(rows, jnp.uint32).reshape(
      *lead, tokens // TOKEN_BLOCK, TOKEN_BLOCK, LANES
  )
  planes = []
  for p in range(TOKEN_BLOCK):
    byte = jnp.stack(
        [(words[..., p, :] >> (8 * j)) & 0xFF for j in range(TOKEN_BLOCK)],
        axis=-2,
    )  # [..., blocks, token-in-block, 128]
    byte = byte.reshape(*lead, tokens, LANES).astype(jnp.uint8)
    planes.append(lax.bitcast_convert_type(byte, CACHE_DTYPE))
  k = jnp.concatenate([planes[0], planes[1]], axis=-1)
  v = jnp.concatenate([planes[2], planes[3]], axis=-1)
  return k, v


def pack_kv_cache(k: jax.Array, v: jax.Array) -> jax.Array:
  """fp8 K, V `[..., tokens, kv_heads, 256]` -> int32 cache rows.

  Returns `[..., tokens * kv_heads, 128]`: per 4-token block, the
  `pack_kv_rows` rows of each KV head in turn (row `4 (kv_heads b + h) + p`);
  for `kv_heads == 1` identical to `pack_kv_rows`.
  """
  assert k.shape == v.shape and k.ndim >= 3, (k.shape, v.shape)
  *lead, tokens, kv_heads, _ = k.shape
  rows = pack_kv_rows(jnp.moveaxis(k, -2, -3), jnp.moveaxis(v, -2, -3))
  # [..., heads, tokens, 128] -> [..., blocks, heads, TOKEN_BLOCK, 128].
  rows = rows.reshape(
      *lead, kv_heads, tokens // TOKEN_BLOCK, TOKEN_BLOCK, LANES
  )
  rows = jnp.moveaxis(rows, -4, -3)
  return rows.reshape(*lead, tokens * kv_heads, LANES)


def unpack_kv_cache(
    rows: jax.Array, kv_heads: int
) -> tuple[jax.Array, jax.Array]:
  """Inverse of `pack_kv_cache` -> fp8 (k, v) `[..., tokens, kv_heads, 256]`."""
  assert rows.shape[-1] == ROW_WORDS, rows.shape
  *lead, num_rows, _ = rows.shape
  assert num_rows % (kv_heads * TOKEN_BLOCK) == 0, (rows.shape, kv_heads)
  tokens = num_rows // kv_heads
  rows = rows.reshape(
      *lead, tokens // TOKEN_BLOCK, kv_heads, TOKEN_BLOCK, LANES
  )
  rows = jnp.moveaxis(rows, -3, -4).reshape(*lead, kv_heads, tokens, LANES)
  k, v = unpack_kv_rows(rows)  # [..., heads, tokens, 256]
  return jnp.moveaxis(k, -3, -2), jnp.moveaxis(v, -3, -2)

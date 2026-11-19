"""Pure-JAX reference and input generators for `sparse_qsa.sparse_qsa`.

Sparse GQA decode attention restricted to `dim == 256` and the packed fp8
cache rows of `qsa_cache_layout.pack_kv_cache`.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.qsa import qsa_cache_layout

# Unused cache slots in tests: the largest finite e4m3fn value, so an unmasked
# out-of-bounds read skews the output (the cache itself is never NaN).
_CACHE_POISON = 448.0


def ref_sparse_qsa_decode(
    q: jax.Array,  # bf16[batch, heads, 256]
    gate: jax.Array,  # bf16[batch, heads, 256]
    blocks: jax.Array,  # i32[batch, budget]
    lengths: jax.Array,  # i32[batch]
    table: jax.Array,  # i32[batch, pages_per_seq]
    cache: jax.Array,  # int32[num_pages, page * kv_heads, 128]
    *,
    ratio: int,
    kv_heads: int = 1,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> jax.Array:
  """Returns bf16[batch, heads, 256] sparse QSA decode attention output."""
  batch, heads, dim = q.shape
  budget = blocks.shape[1]
  page = cache.shape[1] // kv_heads

  complete = lengths // ratio
  count = jnp.minimum(complete, budget)
  tail = lengths % ratio
  cols = jnp.arange(budget + 1)[None, :]
  groups = jnp.concatenate(
      [blocks, jnp.zeros((batch, 1), blocks.dtype)], axis=1
  )
  groups = jnp.where(cols == count[:, None], complete[:, None], groups)
  offs = jnp.arange(ratio)[None, None, :]
  valid = (cols < count[:, None])[..., None] | (
      (cols == count[:, None])[..., None] & (offs < tail[:, None, None])
  )
  tokens = groups[..., None] * ratio + offs
  valid = valid.reshape(batch, -1)  # [batch, T]
  tokens = jnp.where(valid, tokens.reshape(batch, -1), 0)

  logical = jnp.clip(tokens // page, 0, table.shape[1] - 1)
  physical = jnp.take_along_axis(table, logical, axis=1)
  # Cache rows are packed in token blocks: unpack whole pages, then gather.
  # fp8 [P, page, kv_heads, 256]
  k_cache, v_cache = qsa_cache_layout.unpack_kv_cache(cache, kv_heads)
  k = k_cache[physical, tokens % page]  # [batch, T, kv_heads, 256]
  v = v_cache[physical, tokens % page]
  k = jnp.where(valid[..., None, None], k.astype(jnp.float32), 0.0)
  v = jnp.where(valid[..., None, None], v.astype(jnp.float32), 0.0)

  qg = q.astype(jnp.float32).reshape(batch, kv_heads, heads // kv_heads, dim)
  logits = jnp.einsum(
      "bgqd,btgd->bgqt", qg, k, precision=jax.lax.Precision.HIGHEST
  ) * (k_scale / math.sqrt(dim))
  logits = jnp.where(valid[:, None, None, :], logits, -jnp.inf)
  probs = jax.nn.softmax(logits, axis=-1)
  out = jnp.einsum(
      "bgqt,btgd->bgqd", probs, v, precision=jax.lax.Precision.HIGHEST
  )
  out = out.reshape(batch, heads, dim) * v_scale
  out = out * jax.nn.sigmoid(gate.astype(jnp.float32))
  out = jnp.where((lengths > 0)[:, None, None], out, 0.0)
  return out.astype(jnp.bfloat16)


def make_inputs(
    batch: int,
    context: int,
    page: int,
    heads: int = 16,
    kv_heads: int = 1,
    ratio: int = 4,
    groups: int = 512,
    *,
    seed: int = 6317,
    ragged: bool = True,
    poison: bool = True,
    device_cache: bool = False,
):
  """Builds (q, gate, blocks, lengths, table, cache) decode inputs.

  Args:
    batch: number of decode requests (one query token each).
    context: max context length (tokens) per request.
    page: physical KV page size (tokens).
    heads: number of query heads.
    kv_heads: number of KV heads.
    ratio: compression ratio (tokens per selected group).
    groups: top-k budget (number of selectable groups per request).
    seed: RNG seed.
    ragged: vary lengths across requests and make one request empty.
    poison: fill idle requests' q/gate with NaN and unused cache slots (beyond
      each context) with large finite values, so that any out-of-bounds read
      shows up in the output. The cache is never NaN: callers zero-initialize it
      and only write finite values. Cache poisoning requires
      `device_cache=False`.
    device_cache: generate the cache on device (fast, for benchmarks).

  Returns:
    Tuple of jax arrays `(q, gate, blocks, lengths, table, cache)`.
  """
  dim = qsa_cache_layout.HEAD_DIM
  rng = np.random.default_rng(seed + batch + page)
  columns = max(1, math.ceil(max(context, 1) / page))
  table = (
      rng.permutation(batch * columns).reshape(batch, columns).astype(np.int32)
  )
  if ragged:
    lengths = np.maximum(context - np.arange(batch) % 4, 0).astype(np.int32)
    if batch > 1:
      lengths[batch // 2] = 0
  else:
    lengths = np.full((batch,), context, np.int32)
  blocks = np.full((batch, groups), -1, np.int32)
  for row, length in enumerate(lengths):
    n = min(length // ratio, groups)
    blocks[row, :n] = rng.choice(length // ratio, n, replace=False)
  q = jnp.asarray(rng.normal(0, 0.5, (batch, heads, dim)), jnp.bfloat16)
  gate = jnp.asarray(rng.normal(0, 3, q.shape), jnp.bfloat16)
  if poison:
    # Idle requests must not leak their (garbage) query/gate into the output.
    live = jnp.asarray(lengths)[:, None, None] > 0
    q = jnp.where(live, q, jnp.nan)
    gate = jnp.where(live, gate, jnp.nan)
  pages = batch * columns
  if device_cache:
    kv = jax.random.uniform(
        jax.random.key(seed),
        (pages, page, kv_heads, 2, dim),
        minval=-0.5,
        maxval=0.5,
    )
    cache = qsa_cache_layout.pack_kv_cache(kv[..., 0, :], kv[..., 1, :])
  else:
    host = rng.normal(0, 0.4, (pages, page, kv_heads, 2, dim))
    host = host.astype(np.float32)
    if poison:
      for row, length in enumerate(lengths):
        for logical in range(columns):
          start = max(0, int(length) - logical * page)
          if start < page:
            host[int(table[row, logical]), start:] = _CACHE_POISON
    host = jnp.asarray(host)
    cache = qsa_cache_layout.pack_kv_cache(host[..., 0, :], host[..., 1, :])
  return (
      q,
      gate,
      jnp.asarray(blocks),
      jnp.asarray(lengths),
      jnp.asarray(table),
      cache,
  )

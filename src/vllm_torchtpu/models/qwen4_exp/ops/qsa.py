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
"""Pure stateless JAX op for Qwen4Exp QSA (query-selected sparse attention)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import torch
import torch_tpu  # noqa: F401 - attaches torch.tpu.pallas

# Chunk sizes bounding HBM/VMEM peak during tiled sparse attention.
_SPARSE_TOKEN_BATCH = 64
_SPARSE_ATTN_BATCH = 8
_SPARSE_BLOCK_TILE = 4096

_FP8_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)
_RAW_KEY_HEAD = 0
_POOLED_KEY_HEAD = 1


def qsa_attention(
    *,
    q: torch.Tensor,
    index_q: torch.Tensor,
    raw_keys: torch.Tensor,
    dense_out: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    main_pool: torch.Tensor,
    index_pool: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    k_norm_weight: torch.Tensor,
    kv_scales: torch.Tensor,
    num_kv_heads: int,
    compress_ratio: int,
    topk_blocks: int,
    scale: float,
    norm_eps: float,
    token_batch: int = _SPARSE_TOKEN_BATCH,
) -> torch.Tensor:
    """Runs sparse QSA attention and updates ``index_pool`` in place."""
    out, new_index_pool = _qsa_op(
        q,
        index_q,
        raw_keys,
        dense_out,
        positions.to(torch.int32),
        query_start_loc.to(torch.int32),
        seq_lens.to(torch.int32),
        block_tables.to(torch.int32),
        main_pool,
        index_pool,
        cos_sin_cache,
        k_norm_weight,
        kv_scales,
        num_kv_heads,
        compress_ratio,
        topk_blocks,
        scale,
        norm_eps,
        token_batch,
    )
    # Donation plus copy_ is aliased in place by XLA.
    index_pool.copy_(new_index_pool)
    return out


@torch.tpu.pallas.jax_op(  # pyrefly: ignore[missing-attribute]
    "pallas::qwen4_exp_qsa", donate_argnums=(9,)
)
def _qsa_op(
    q: jax.Array,
    index_q: jax.Array,
    raw_keys: jax.Array,
    dense_out: jax.Array,
    positions: jax.Array,
    query_start_loc: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    main_pool: jax.Array,
    index_pool: jax.Array,
    cos_sin: jax.Array,
    k_norm_weight: jax.Array,
    kv_scales: jax.Array,
    num_kv_heads: int,
    compress_ratio: int,
    topk_blocks: int,
    scale: float,
    norm_eps: float,
    token_batch: int = _SPARSE_TOKEN_BATCH,
) -> tuple[jax.Array, jax.Array]:
    return _qsa_attention(
        q,
        index_q,
        raw_keys,
        dense_out,
        positions,
        query_start_loc,
        seq_lens,
        block_tables,
        main_pool,
        index_pool,
        cos_sin,
        k_norm_weight,
        kv_scales,
        num_kv_heads=num_kv_heads,
        compress_ratio=compress_ratio,
        topk_blocks=topk_blocks,
        scale=scale,
        norm_eps=norm_eps,
        token_batch=token_batch,
    )


@_qsa_op.register_fake
def _(
    q: torch.Tensor,
    index_q: torch.Tensor,
    raw_keys: torch.Tensor,
    dense_out: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    main_pool: torch.Tensor,
    index_pool: torch.Tensor,
    cos_sin: torch.Tensor,
    k_norm_weight: torch.Tensor,
    kv_scales: torch.Tensor,
    num_kv_heads: int,
    compress_ratio: int,
    topk_blocks: int,
    scale: float,
    norm_eps: float,
    token_batch: int = _SPARSE_TOKEN_BATCH,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(dense_out), torch.empty_like(index_pool)


def _qsa_attention(
    q: jax.Array,
    index_q: jax.Array,
    raw_keys: jax.Array,
    dense_out: jax.Array,
    positions: jax.Array,
    query_start_loc: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    main_pool: jax.Array,
    index_pool: jax.Array,
    cos_sin: jax.Array,
    k_norm_weight: jax.Array,
    kv_scales: jax.Array,
    *,
    num_kv_heads: int,
    compress_ratio: int,
    topk_blocks: int,
    scale: float,
    norm_eps: float,
    token_batch: int = _SPARSE_TOKEN_BATCH,
    block_tile: int = _SPARSE_BLOCK_TILE,
) -> tuple[jax.Array, jax.Array]:
    """Plain-XLA QSA math; returns (output, updated index_pool)."""
    if q.ndim != 3 or dense_out.shape != q.shape:
        raise ValueError(
            f"q and dense_out must have matching 3-D shapes [T, H, D], got "
            f"{q.shape} and {dense_out.shape}."
        )
    num_tokens, num_heads, head_dim = q.shape
    if num_heads % num_kv_heads:
        raise ValueError(
            f"num_heads ({num_heads}) must be a multiple of "
            f"num_kv_heads ({num_kv_heads})."
        )
    if index_q.ndim != 3 or index_q.shape[0] != num_tokens:
        raise ValueError(
            f"index_q must be [{num_tokens}, heads, index_head_dim], "
            f"got {index_q.shape}."
        )
    index_head_dim = index_q.shape[-1]
    if raw_keys.shape != (num_tokens, index_head_dim):
        raise ValueError(
            f"raw_keys must be [{num_tokens}, {index_head_dim}], got {raw_keys.shape}."
        )
    if kv_scales.shape != (2,):
        raise ValueError(
            f"kv_scales must be [k_scale, v_scale], got {kv_scales.shape}."
        )
    ratio = compress_ratio
    num_reqs = seq_lens.shape[0]
    request, chunk_pos, valid = _token_requests(query_start_loc, num_tokens)
    index_rows = _PagedRows(index_pool, block_tables, num_reqs)

    # Pooled key of the block ending at each position: earlier raw keys come
    # from this chunk or from the cache if the block started in an earlier step.
    cached = index_rows.read(
        index_rows.slots(request[:, None], positions[:, None] - jnp.arange(1, ratio)),
        _RAW_KEY_HEAD,
        index_head_dim,
    ).astype(raw_keys.dtype)
    shifted = jnp.stack(
        [jnp.roll(raw_keys, shift, axis=0) for shift in range(1, ratio)], axis=1
    )
    in_chunk = (chunk_pos[:, None] >= jnp.arange(1, ratio))[..., None]
    block = jnp.concatenate(
        [raw_keys[:, None], jnp.where(in_chunk, shifted, cached)], axis=1
    )
    pooled = block.astype(jnp.float32).mean(axis=1).astype(raw_keys.dtype)
    pooled = _rope(
        _rms_norm(pooled, k_norm_weight, norm_eps),
        jnp.maximum(positions - (ratio - 1), 0),
        cos_sin,
    )
    pooled = jnp.where(((positions + 1) % ratio == 0)[:, None], pooled, 0)
    # Padding tokens write into slot 0, the null block.
    write_slots = jnp.where(valid, index_rows.slots(request, positions), 0)
    index_pool = _write_index_rows(
        index_pool, write_slots, raw_keys, pooled, num_kv_heads
    )

    sparse = valid & (((positions + 1) // ratio) > topk_blocks)

    def sparse_attention() -> jax.Array:
        updated_index = _PagedRows(index_pool, block_tables, num_reqs)
        main_rows = _PagedRows(main_pool, block_tables, num_reqs)
        total_blocks = updated_index.max_tokens // ratio
        tile_size = min(max(block_tile, topk_blocks), total_blocks)
        num_tiles = math.ceil(total_blocks / tile_size)
        tile_offsets = jnp.arange(tile_size, dtype=jnp.int32)
        group = num_heads // num_kv_heads
        k_heads = jnp.arange(num_kv_heads) * 2
        matmul_dtype = q.dtype
        inv_sqrt_d = 1.0 / math.sqrt(index_head_dim)

        def topk_for_req(
            req: jax.Array, visible: jax.Array, iq: jax.Array
        ) -> jax.Array:
            # Shared pooled-key scan over block tiles for M tokens of one request;
            # tiles past max(visible) are skipped without touching HBM.
            iq_m = iq.astype(matmul_dtype)
            max_visible = jnp.max(visible)
            init = (
                jnp.full((iq.shape[0], topk_blocks), -jnp.inf, jnp.float32),
                jnp.zeros((iq.shape[0], topk_blocks), jnp.int32),
            )

            def scan_tile(carry, tile_idx):
                best_scores, best_ids = carry
                tile_start = tile_idx * tile_size

                def score_tile():
                    bids = tile_start + tile_offsets
                    keys = updated_index.read(
                        updated_index.slots(req, bids * ratio + ratio - 1),
                        _POOLED_KEY_HEAD,
                        index_head_dim,
                    ).astype(matmul_dtype)
                    keys = jnp.where((bids < max_visible)[:, None], keys, 0)
                    dots = jnp.einsum(
                        "mhd,bd->mhb",
                        iq_m,
                        keys,
                        preferred_element_type=jnp.float32,
                    )
                    scores = jnp.maximum(dots, 0.0).sum(axis=1) * inv_sqrt_d
                    scores = jnp.where(
                        bids[None, :] < visible[:, None], scores, -jnp.inf
                    )
                    t_scores, t_local = jax.lax.top_k(scores, topk_blocks)
                    all_scores = jnp.concatenate([best_scores, t_scores], axis=-1)
                    all_ids = jnp.concatenate([best_ids, tile_start + t_local], axis=-1)
                    m_scores, m_idx = jax.lax.top_k(all_scores, topk_blocks)
                    return m_scores, jnp.take_along_axis(all_ids, m_idx, axis=-1)

                return (
                    jax.lax.cond(tile_start < max_visible, score_tile, lambda: carry),
                    None,
                )

            (_, top), _ = jax.lax.scan(
                scan_tile, init, jnp.arange(num_tiles, dtype=jnp.int32)
            )
            return top

        def attend_one(args):
            req, pos, top, qh = args
            visible_blocks = (pos + 1) // ratio
            selected = (top[:, None] * ratio + jnp.arange(ratio)).reshape(-1)
            tail = visible_blocks * ratio + jnp.arange(ratio - 1)
            keep = jnp.concatenate([jnp.ones(selected.shape, bool), tail <= pos])
            slots = main_rows.slots(req, jnp.concatenate([selected, tail]))
            k = main_rows.read(slots, k_heads, head_dim)
            v = main_rows.read(slots, k_heads + 1, head_dim)
            # Zero masked tail slots so uninitialized fp8 NaNs cannot poison 0 * NaN.
            k = jnp.where(
                keep[:, None, None],
                (k.astype(jnp.float32) * kv_scales[0]).astype(matmul_dtype),
                0,
            )
            v = jnp.where(
                keep[:, None, None],
                (v.astype(jnp.float32) * kv_scales[1]).astype(matmul_dtype),
                0,
            )
            qg = qh.astype(matmul_dtype).reshape(num_kv_heads, group, head_dim)
            logits = jnp.where(
                keep,
                jnp.einsum("kgd,skd->kgs", qg, k, preferred_element_type=jnp.float32)
                * scale,
                -jnp.inf,
            )
            probs = jax.nn.softmax(logits, axis=-1).astype(matmul_dtype)
            return jnp.einsum(
                "kgs,skd->kgd", probs, v, preferred_element_type=jnp.float32
            ).reshape(num_heads, head_dim)

        chunk_size = min(num_tokens, token_batch)
        pad = (-num_tokens) % chunk_size

        def pad_chunks(arr):
            if pad:
                arr = jnp.pad(arr, ((0, pad),) + ((0, 0),) * (arr.ndim - 1))
            return arr.reshape(-1, chunk_size, *arr.shape[1:])

        def run_chunk(_, chunk):
            c_req, c_pos, c_iq, c_q, c_sp, c_dout = chunk

            def compute_chunk():
                def single_req():
                    top = topk_for_req(c_req[0], (c_pos + 1) // ratio, c_iq)
                    c_out = jax.lax.map(
                        attend_one,
                        (c_req, c_pos, top, c_q),
                        batch_size=min(chunk_size, _SPARSE_ATTN_BATCH),
                    )
                    return jnp.where(
                        c_sp[:, None, None], c_out.astype(dense_out.dtype), c_dout
                    )

                def per_token():
                    def one_tok(_, tok):
                        req, pos, iq, qh, sp, dout = tok

                        def compute():
                            top = topk_for_req(
                                req, ((pos + 1) // ratio)[None], iq[None]
                            )[0]
                            return attend_one((req, pos, top, qh)).astype(
                                dense_out.dtype
                            )

                        return None, jax.lax.cond(sp, compute, lambda: dout)

                    _, out = jax.lax.scan(
                        one_tok, None, (c_req, c_pos, c_iq, c_q, c_sp, c_dout)
                    )
                    return out

                same_req = jnp.all(c_req == c_req[0])
                return jax.lax.cond(same_req, single_req, per_token)

            return None, jax.lax.cond(jnp.any(c_sp), compute_chunk, lambda: c_dout)

        _, chunks_out = jax.lax.scan(
            run_chunk,
            None,
            (
                pad_chunks(request),
                pad_chunks(positions),
                pad_chunks(index_q),
                pad_chunks(q),
                pad_chunks(sparse),
                pad_chunks(dense_out),
            ),
        )
        return chunks_out.reshape(-1, num_heads, head_dim)[:num_tokens]

    out = jax.lax.cond(jnp.any(sparse), sparse_attention, lambda: dense_out)
    return out, index_pool


def _token_requests(query_start_loc: jax.Array, num_tokens: int):
    """Each token's request, its offset in the chunk, and whether it is real."""
    num_reqs = query_start_loc.shape[0] - 1
    tokens = jnp.arange(num_tokens, dtype=jnp.int32)
    request = jnp.clip(
        jnp.searchsorted(query_start_loc[1:], tokens, side="right"), 0, num_reqs - 1
    ).astype(jnp.int32)
    chunk_pos = tokens - query_start_loc[request]
    valid = tokens < query_start_loc[num_reqs]
    return request, chunk_pos, valid


class _PagedRows:
    """Token rows of an attention-shaped pool [pages, page, h2/pack, pack, D]."""

    def __init__(self, pool: jax.Array, block_tables: jax.Array, num_reqs: int):
        if pool.ndim != 5:
            raise NotImplementedError(
                f"Qwen4Exp QSA requires a 5-D NHD KV pool, got {pool.shape}."
            )
        self.pool = pool
        self.page_tokens = pool.shape[1]
        self.tables = block_tables.reshape(num_reqs, -1)
        self.max_tokens = self.tables.shape[1] * self.page_tokens

    def slots(self, request: jax.Array, position: jax.Array) -> jax.Array:
        position = jnp.clip(position, 0, self.max_tokens - 1)
        page = self.tables[request, position // self.page_tokens]
        return page * self.page_tokens + position % self.page_tokens

    def read(self, slots: jax.Array, heads, dim: int) -> jax.Array:
        """Rows [..., (heads,) dim] at ``slots``; K of kv head h is 2h, V is 2h + 1."""
        pool = self.pool
        rows = pool.reshape(-1, *pool.shape[2:])[slots]
        pack = pool.shape[3]
        heads = jnp.asarray(heads)
        return rows[..., heads // pack, heads % pack, :dim]


def _write_index_rows(
    pool: jax.Array,
    slots: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    num_kv_heads: int,
) -> jax.Array:
    """Write K = keys (head 0), V = values (head 1) to every kv head at slots."""
    num_tokens, dim = keys.shape
    groups, pack, width = pool.shape[2:]
    kv = jnp.stack([keys, values], axis=1)
    kv = jnp.tile(kv, (1, num_kv_heads, 1))
    kv = jnp.pad(kv, ((0, 0), (0, groups * pack - 2 * num_kv_heads), (0, width - dim)))
    # TODO(QSA): normalize_kv_cache_specs_for_tpu forces Attention caches to the
    # global kv_cache_dtype, so index_pool inherits unscaled fp8 under fp8 KV.
    # Keep index keys natively bf16 once a dedicated compact index spec lands.
    if pool.dtype == jnp.float8_e4m3fn:
        kv = jnp.clip(kv, -_FP8_MAX, _FP8_MAX)
    kv = kv.astype(pool.dtype).reshape(num_tokens, groups, pack, width)
    flat = pool.reshape(-1, groups, pack, width)
    return flat.at[slots].set(kv).reshape(pool.shape)


def _rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    """GemmaRMSNorm in fp32: x / rms(x) * (1 + weight)."""
    x32 = x.astype(jnp.float32)
    x32 = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return (x32 * (1.0 + weight.astype(jnp.float32))).astype(x.dtype)


def _rope(x: jax.Array, positions: jax.Array, cos_sin: jax.Array) -> jax.Array:
    """NeoX RoPE on the first ``cos_sin.shape[-1]`` dims of [T, dim] rows."""
    rotary_dim = cos_sin.shape[-1]
    half = rotary_dim // 2
    cs = cos_sin[positions].astype(jnp.float32)
    cos, sin = cs[:, :half], cs[:, half:]
    rot = x[:, :rotary_dim].astype(jnp.float32)
    x1, x2 = rot[:, :half], rot[:, half:]
    rot = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return jnp.concatenate([rot.astype(x.dtype), x[:, rotary_dim:]], axis=-1)

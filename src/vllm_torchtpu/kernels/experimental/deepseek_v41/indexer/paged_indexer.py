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
"""TPU7x Pallas paged single-bank index-cache gather, scoring, and top-k selection."""

import contextlib
import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer.sparsecore_topk import (
    BITS_NEG_INF,
    MAX_N,
    MAX_TOPK,
    SC_LANES,
    sparsecore_topk,
)
from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer.word_scores import (
    BF16_BYTES,
    DIM,
    ROW_WORDS,
    SCORE_BYTES,
    SLOTS,
    SUB_BLOCK_ROWS,
    SUBLANES,
    TPU7X_GENERATION,
    VMEM_HEADROOM_DIVISOR,
    VMEM_LIMIT_DIVISOR,
    WEIGHT_BYTES_PER_HEAD,
    score_word_rows,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Avoid GATHER_SUB_BATCH=2: measured 2.25x-3.17x slower end-to-end.
GATHER_SUB_BATCH = 16
MAX_TILE_ROWS = 4096
MAX_SCORING_TOKENS = 288
MIN_SPARSECORE_KEYS = 2048
VALID_COMPRESS_RATIOS = (1, 2)
_COMPILER_PARAMS = pltpu.CompilerParams(dimension_semantics=("parallel",))


def _check_tpu7x(func_name: str) -> None:
    # Avoid calling pltpu.get_tpu_info() on bare CPU: raises ValueError when no
    # TPU backend or AbstractMesh is active.
    if (
        jax.default_backend() != "tpu"
        and jax.sharding.get_abstract_mesh().abstract_device is None
    ):
        raise RuntimeError(
            f"{func_name} requires TPU7x, got backend={jax.default_backend()!r}"
        )
    tpu_info = pltpu.get_tpu_info()
    if (
        tpu_info.generation != TPU7X_GENERATION
        or tpu_info.num_lanes != ROW_WORDS
        or tpu_info.num_sublanes != SUBLANES
    ):
        raise RuntimeError(
            f"{func_name} requires TPU7x ({SUBLANES}x{ROW_WORDS} lanes), got "
            f"generation={tpu_info.generation}, num_lanes={tpu_info.num_lanes}, "
            f"num_sublanes={tpu_info.num_sublanes}"
        )


def _named_scope(name: str, enabled: bool) -> contextlib.AbstractContextManager[Any]:
    return jax.named_scope(name) if enabled else contextlib.nullcontext()


def _max_scoring_tokens(heads: int, padded_rows: int) -> int:
    # Avoid unpadded heads or scalar 4 B/head weights: VMEM tiles 3D q [T, H, 128]
    # and f32[T, H, 1] weights to (8, 128), costing 512 B per padded head for weights.
    padded_heads = ((heads + SUBLANES - 1) // SUBLANES) * SUBLANES
    bytes_per_token = max(
        1,
        padded_heads * (BF16_BYTES * DIM + ROW_WORDS * WEIGHT_BYTES_PER_HEAD)
        + SCORE_BYTES * SLOTS * padded_rows,
    )
    vmem_limit_bytes = (
        int(pltpu.get_tpu_info().vmem_capacity_bytes) // VMEM_LIMIT_DIVISOR
    )
    vmem_headroom_bytes = vmem_limit_bytes // VMEM_HEADROOM_DIVISOR
    usable_vmem_bytes = vmem_limit_bytes - vmem_headroom_bytes
    return max(1, min(MAX_SCORING_TOKENS, usable_vmem_bytes // bytes_per_token))


def _gather_paged_index_rows_tile(
    idx_smem_ref: Any,
    pool_hbm_ref: Any,
    out_vmem_ref: Any,
    sem_ref: Any,
    *,
    total_rows: int,
    dma_rows: int,
    dmas_per_tile: int,
    enable_named_scope: bool,
) -> None:
    out_vmem_ref[...] = jnp.zeros_like(out_vmem_ref)
    with _named_scope("gather_paged_index_rows", enable_named_scope):
        pending_copies: list[tuple[jax.Array, Any]] = []
        for dma_idx in range(dmas_per_tile):
            dest_slot = out_vmem_ref.at[0, pl.ds(dma_idx * dma_rows, dma_rows), :]
            row_idx = idx_smem_ref[0, 0, dma_idx]
            in_pool = (row_idx >= 0) & (row_idx < total_rows)
            row_offset = jnp.where(in_pool, row_idx, 0)
            dma_copy = pltpu.make_async_copy(
                pool_hbm_ref.at[pl.ds(row_offset, dma_rows), :], dest_slot, sem_ref
            )

            @pl.when(in_pool)
            def _start_copy(copy_desc: Any = dma_copy) -> None:
                copy_desc.start()

            pending_copies.append((in_pool, dma_copy))
        for in_pool, dma_copy in pending_copies:

            @pl.when(in_pool)
            def _wait_copy(copy_desc: Any = dma_copy) -> None:
                copy_desc.wait()


def _gather_paged_index_rows(
    k_pool: jax.Array,
    block_tables: jax.Array,
    *,
    padded_rows: int,
    interpret: bool,
    enable_named_scope: bool,
) -> jax.Array:
    num_pages, page_rows, _ = k_pool.shape
    total_rows = num_pages * page_rows
    num_seqs, max_pages = block_tables.shape
    raw_rows = max_pages * page_rows

    dma_rows = math.gcd(page_rows, SUB_BLOCK_ROWS)
    num_chunks = padded_rows // dma_rows
    dmas_per_tile = math.gcd(num_chunks, GATHER_SUB_BATCH)
    tile_rows = dmas_per_tile * dma_rows

    chunk_row_offsets = jnp.arange(num_chunks, dtype=jnp.int32) * dma_rows
    logical_pages = jnp.minimum(chunk_row_offsets // page_rows, max_pages - 1)
    page_row_offsets = chunk_row_offsets % page_rows
    phys_pages = block_tables[:, logical_pages]
    valid_chunk = (
        (chunk_row_offsets[None, :] < raw_rows)
        & (phys_pages >= 0)
        & (phys_pages < num_pages)
    )
    chunk_indices = jnp.where(
        valid_chunk, phys_pages * page_rows + page_row_offsets[None, :], -1
    )

    num_blocks = (num_seqs * num_chunks) // dmas_per_tile
    idx_3d = chunk_indices.reshape(num_blocks, 1, dmas_per_tile)
    pool_2d = k_pool.reshape(total_rows, ROW_WORDS)

    smem_spec = pl.BlockSpec(
        (1, 1, dmas_per_tile),
        lambda block_idx: (block_idx, 0, 0),
        memory_space=pltpu.SMEM,
    )
    hbm_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    vmem_spec = pl.BlockSpec(
        (1, tile_rows, ROW_WORDS),
        lambda block_idx: (block_idx, 0, 0),
        memory_space=pltpu.VMEM,
    )
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=(num_blocks,),
        in_specs=[smem_spec, hbm_spec],
        out_specs=vmem_spec,
        scratch_shapes=[pltpu.SemaphoreType.DMA],
    )
    out_sds = jax.ShapeDtypeStruct((num_blocks, tile_rows, ROW_WORDS), jnp.int32)
    tile_fn = functools.partial(
        _gather_paged_index_rows_tile,
        total_rows=total_rows,
        dma_rows=dma_rows,
        dmas_per_tile=dmas_per_tile,
        enable_named_scope=enable_named_scope,
    )
    gathered = pl.pallas_call(
        tile_fn,
        out_shape=out_sds,
        grid_spec=grid_spec,
        compiler_params=_COMPILER_PARAMS,
        interpret=interpret,
        name="dsv41_gather_paged_index_rows",
    )(idx_3d, pool_2d)
    return gathered.reshape(num_seqs, padded_rows, ROW_WORDS)


@functools.partial(
    jax.jit,
    static_argnames=("topk", "compress_ratio", "interpret", "enable_named_scope"),
)
def paged_indexer_topk(
    q: jax.Array,
    weights: jax.Array,
    k_pool: jax.Array,
    block_tables: jax.Array,
    seq_lens: jax.Array,
    cu_q_lens: jax.Array,
    *,
    topk: int,
    candidate_indices: jax.Array | None = None,
    compress_ratio: int = 1,
    interpret: bool = False,
    enable_named_scope: bool = False,
) -> jax.Array:
    """Select ``int32[T, topk]`` key indices from paged ``int32[P, R, 128]`` pools.

    Scores split ``[even, odd]`` ``q: bf16[T, H, 128]`` with ``weights: f32[T, H, 1]``
    via ``block_tables: int32[S, M]`` (``< 0`` or ``>= P`` unmapped, ``R % 8 == 0``,
    2 keys/row), ``seq_lens``, ``cu_q_lens``, optional ``candidate_indices``; invalid
    slots pad ``-1``.
    """
    q_ok = q.ndim == 3 and q.shape[2] == DIM and q.dtype == jnp.bfloat16
    w_ok = q_ok and weights.shape == (*q.shape[:2], 1) and weights.dtype == jnp.float32
    if not w_ok:
        raise ValueError(
            f"Expected bf16 q [tokens, heads, {DIM}] and fp32 weights "
            f"[tokens, heads, 1]; got q=({q.dtype}, {q.shape}), "
            f"weights=({weights.dtype}, {weights.shape})"
        )
    pool_ok = (
        k_pool.ndim == 3
        and k_pool.shape[0] > 0
        and k_pool.shape[1] > 0
        and k_pool.shape[1] % SUBLANES == 0
        and k_pool.shape[2] == ROW_WORDS
        and k_pool.dtype == jnp.int32
    )
    if not pool_ok:
        raise ValueError(
            f"Expected int32 k_pool [pages, page_rows, {ROW_WORDS}] with "
            f"pages > 0 and page_rows > 0 a multiple of {SUBLANES}; "
            f"got k_pool=({k_pool.dtype}, {k_pool.shape})"
        )
    bt_ok = block_tables.ndim == 2 and block_tables.dtype == jnp.int32
    sl_ok = (
        bt_ok
        and seq_lens.shape == (block_tables.shape[0],)
        and seq_lens.dtype == jnp.int32
    )
    cu_ok = (
        sl_ok
        and cu_q_lens.shape == (block_tables.shape[0] + 1,)
        and cu_q_lens.dtype == jnp.int32
    )
    if not cu_ok:
        raise ValueError(
            "Expected int32 block_tables [num_seqs, max_pages], seq_lens "
            "[num_seqs], and cu_q_lens [num_seqs + 1]; got "
            f"block_tables=({block_tables.dtype}, {block_tables.shape}), "
            f"seq_lens=({seq_lens.dtype}, {seq_lens.shape}), "
            f"cu_q_lens=({cu_q_lens.dtype}, {cu_q_lens.shape})"
        )
    tokens, heads, _ = q.shape
    cand_ok = candidate_indices is None or (
        candidate_indices.ndim == 2
        and candidate_indices.shape[0] == tokens
        and candidate_indices.dtype == jnp.int32
    )
    if topk <= 0 or compress_ratio not in VALID_COMPRESS_RATIOS or not cand_ok:
        cand_desc = (
            None
            if candidate_indices is None
            else (candidate_indices.dtype, candidate_indices.shape)
        )
        raise ValueError(
            f"Expected topk > 0, compress_ratio in (1, 2), and optional int32 "
            f"candidate_indices [{tokens}, C]; got topk={topk}, "
            f"compress_ratio={compress_ratio}, candidate_indices={cand_desc}"
        )
    _check_tpu7x("paged_indexer_topk")
    num_pages, page_rows, _ = k_pool.shape
    num_seqs, max_pages = block_tables.shape
    if (
        tokens == 0
        or num_seqs == 0
        or max_pages == 0
        or (candidate_indices is not None and candidate_indices.shape[1] == 0)
    ):
        return jnp.full((tokens, topk), -1, dtype=jnp.int32)

    raw_rows = max_pages * page_rows
    min_req_rows = max(raw_rows, (topk + SLOTS - 1) // SLOTS)
    lcm_rows = math.lcm(page_rows, SUB_BLOCK_ROWS)
    tile_rows = max(lcm_rows, ((MAX_TILE_ROWS + lcm_rows - 1) // lcm_rows) * lcm_rows)
    step_rows = tile_rows if min_req_rows > tile_rows else SUB_BLOCK_ROWS
    padded_rows = ((min_req_rows + step_rows - 1) // step_rows) * step_rows
    tile_pages = tile_rows // page_rows
    tile_keys = SLOTS * tile_rows
    num_page_tiles = padded_rows // tile_rows

    words_single: jax.Array | None = None
    bt_padded = block_tables
    if padded_rows <= tile_rows:
        words_single = _gather_paged_index_rows(
            k_pool,
            block_tables,
            padded_rows=padded_rows,
            interpret=interpret,
            enable_named_scope=enable_named_scope,
        )
    else:
        padded_pages = num_page_tiles * tile_pages
        bt_padded = jnp.pad(
            block_tables,
            ((0, 0), (0, padded_pages - max_pages)),
            constant_values=-1,
        )

    total_scored_keys, max_keys = SLOTS * padded_rows, SLOTS * raw_rows
    page_keys = SLOTS * page_rows

    all_token_indices = jnp.arange(tokens, dtype=jnp.int32)
    in_seq = (all_token_indices[:, None] >= cu_q_lens[:-1][None, :]) & (
        all_token_indices[:, None] < cu_q_lens[1:][None, :]
    )
    active_token = jnp.any(in_seq, axis=1)
    seq_id = jnp.argmax(in_seq, axis=1).astype(jnp.int32)
    token_positions = seq_lens[seq_id] - (cu_q_lens[seq_id + 1] - all_token_indices)
    valid_count = jnp.where(
        active_token, jnp.maximum(0, (token_positions + 1) // compress_ratio), 0
    )
    eff_row_lengths = jnp.minimum(valid_count, max_keys)

    page_ok = (block_tables >= 0) & (block_tables < num_pages)
    key_col_iota = jnp.arange(total_scored_keys, dtype=jnp.int32)[None, :]
    neg_inf_f32 = jax.lax.bitcast_convert_type(jnp.int32(BITS_NEG_INF), jnp.float32)

    topk_width = (
        total_scored_keys if candidate_indices is None else candidate_indices.shape[1]
    )
    use_sc_topk = (
        not interpret
        and MIN_SPARSECORE_KEYS <= topk_width <= MAX_N
        and topk_width % SC_LANES == 0
        and topk % SC_LANES == 0
        and topk <= MAX_TOPK
        and topk <= topk_width
    )
    if not interpret and not use_sc_topk:
        logger.warning_once(
            "paged_indexer_topk falling back from sparsecore_topk to "
            "jax.lax.top_k for topk_width=%d, topk=%d (requires "
            "%d <= topk_width <= %d, topk_width %% %d == 0, topk %% %d == 0, "
            "and topk <= min(topk_width, %d)).",
            topk_width,
            topk,
            MIN_SPARSECORE_KEYS,
            MAX_N,
            SC_LANES,
            SC_LANES,
            MAX_TOPK,
        )

    def _topk_chunk(
        chunk_q: jax.Array,
        chunk_weights: jax.Array,
        chunk_eff_lens: jax.Array,
        chunk_seq_ids: jax.Array,
        chunk_cu_q_lens: jax.Array,
        chunk_candidates: jax.Array | None,
    ) -> jax.Array:
        chunk_tokens = chunk_q.shape[0]
        if words_single is not None:
            chunk_scores = score_word_rows(
                chunk_q,
                chunk_weights,
                words_single,
                cu_q_lens=chunk_cu_q_lens,
                interpret=interpret,
                enable_named_scope=enable_named_scope,
            )
        else:
            num_trips = jnp.minimum(
                (jnp.maximum(jnp.max(chunk_eff_lens), 0) + tile_keys - 1) // tile_keys,
                num_page_tiles,
            )

            def _tile_step(tile_idx: jax.Array, acc_scores: jax.Array) -> jax.Array:
                bt_tile = jax.lax.dynamic_slice_in_dim(
                    bt_padded, tile_idx * tile_pages, tile_pages, axis=1
                )
                words_tile = _gather_paged_index_rows(
                    k_pool,
                    bt_tile,
                    padded_rows=tile_rows,
                    interpret=interpret,
                    enable_named_scope=enable_named_scope,
                )
                tile_scores = score_word_rows(
                    chunk_q,
                    chunk_weights,
                    words_tile,
                    cu_q_lens=chunk_cu_q_lens,
                    interpret=interpret,
                    enable_named_scope=enable_named_scope,
                )
                return jax.lax.dynamic_update_slice_in_dim(
                    acc_scores, tile_scores, tile_idx * tile_keys, axis=1
                )

            scores_init = jnp.full(
                (chunk_tokens, total_scored_keys), neg_inf_f32, dtype=jnp.float32
            )
            chunk_scores = jax.lax.fori_loop(0, num_trips, _tile_step, scores_init)

        # Avoid jnp.repeat(page_ok, page_keys, 1): lowers to 351 ms gather at 128K.
        key_ok = jnp.broadcast_to(
            page_ok[chunk_seq_ids][:, :, None], (chunk_tokens, max_pages, page_keys)
        ).reshape(chunk_tokens, max_keys)
        key_ok = jnp.pad(
            key_ok,
            ((0, 0), (0, total_scored_keys - max_keys)),
            constant_values=False,
        )
        valid_mask = (key_col_iota < chunk_eff_lens[:, None]) & key_ok
        base_scores = jnp.where(valid_mask, chunk_scores, neg_inf_f32)
        if chunk_candidates is None:
            if use_sc_topk:
                base_bits = jax.lax.bitcast_convert_type(base_scores, jnp.int32)
                return sparsecore_topk(
                    base_bits,
                    topk,
                    row_lengths=chunk_eff_lens,
                    write_empty_rows=True,
                )
            top_vals, top_idxs = jax.lax.top_k(base_scores, topk)
            valid_win = jax.lax.bitcast_convert_type(top_vals, jnp.int32) != jnp.int32(
                BITS_NEG_INF
            )
            return jnp.where(valid_win, top_idxs, -1)

        num_cand = chunk_candidates.shape[1]
        cand_k = min(topk, num_cand)
        chunk_token_iota = jnp.arange(chunk_tokens, dtype=jnp.int32)[:, None]
        valid_cand = (chunk_candidates >= 0) & (chunk_candidates < max_keys)
        cand_sorted = jnp.sort(
            jnp.where(valid_cand, chunk_candidates, max_keys), axis=1
        )
        dedup_valid = (cand_sorted < max_keys) & jnp.concatenate(
            [
                jnp.ones((chunk_tokens, 1), dtype=jnp.bool_),
                cand_sorted[:, 1:] != cand_sorted[:, :-1],
            ],
            axis=1,
        )
        safe_sorted = jnp.where(dedup_valid, cand_sorted, 0)
        cand_vals = jnp.where(
            dedup_valid, base_scores[chunk_token_iota, safe_sorted], neg_inf_f32
        )
        if use_sc_topk:
            cand_bits = jax.lax.bitcast_convert_type(cand_vals, jnp.int32)
            cand_lens = jnp.sum(
                cand_sorted < jnp.maximum(chunk_eff_lens, 0)[:, None],
                axis=1,
                dtype=jnp.int32,
            )
            local_idxs = sparsecore_topk(
                cand_bits,
                topk,
                row_lengths=cand_lens,
                write_empty_rows=True,
            )
            valid_win = local_idxs >= 0
        else:
            top_vals, local_idxs = jax.lax.top_k(cand_vals, cand_k)
            valid_win = jax.lax.bitcast_convert_type(top_vals, jnp.int32) != jnp.int32(
                BITS_NEG_INF
            )
        mapped = jnp.take_along_axis(cand_sorted, jnp.maximum(local_idxs, 0), axis=1)
        selected = jnp.where(valid_win, mapped, -1)
        if cand_k < topk:
            selected = jnp.pad(
                selected,
                ((0, 0), (0, topk - cand_k)),
                constant_values=-1,
            )
        return selected

    # Avoid full padded_rows or unchunked T: splits T=288 or allocates 8.6 GiB HBM.
    max_tokens_per_chunk = _max_scoring_tokens(heads, min(padded_rows, tile_rows))
    if tokens <= max_tokens_per_chunk:
        return _topk_chunk(
            q, weights, eff_row_lengths, seq_id, cu_q_lens, candidate_indices
        )

    num_full_chunks, remainder_tokens = divmod(tokens, max_tokens_per_chunk)

    def _scan_step(_: None, chunk_idx: jax.Array) -> tuple[None, jax.Array]:
        start_token = chunk_idx * max_tokens_per_chunk
        step_q = jax.lax.dynamic_slice_in_dim(q, start_token, max_tokens_per_chunk, 0)
        step_weights = jax.lax.dynamic_slice_in_dim(
            weights, start_token, max_tokens_per_chunk, 0
        )
        step_eff_lens = jax.lax.dynamic_slice_in_dim(
            eff_row_lengths, start_token, max_tokens_per_chunk, 0
        )
        step_seq_ids = jax.lax.dynamic_slice_in_dim(
            seq_id, start_token, max_tokens_per_chunk, 0
        )
        step_cu_q_lens = jnp.clip(cu_q_lens - start_token, 0, max_tokens_per_chunk)
        step_candidates = (
            None
            if candidate_indices is None
            else jax.lax.dynamic_slice_in_dim(
                candidate_indices, start_token, max_tokens_per_chunk, 0
            )
        )
        return None, _topk_chunk(
            step_q,
            step_weights,
            step_eff_lens,
            step_seq_ids,
            step_cu_q_lens,
            step_candidates,
        )

    _, full_out = jax.lax.scan(
        _scan_step, None, jnp.arange(num_full_chunks, dtype=jnp.int32)
    )
    out = full_out.reshape(num_full_chunks * max_tokens_per_chunk, topk)
    if remainder_tokens == 0:
        return out
    start_token = num_full_chunks * max_tokens_per_chunk
    remainder_candidates = (
        None if candidate_indices is None else candidate_indices[start_token:]
    )
    tail = _topk_chunk(
        q[start_token:],
        weights[start_token:],
        eff_row_lengths[start_token:],
        seq_id[start_token:],
        jnp.clip(cu_q_lens - start_token, 0, remainder_tokens),
        remainder_candidates,
    )
    return jnp.concatenate([out, tail], axis=0)

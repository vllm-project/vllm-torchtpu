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
"""Shared attention-metadata, cache-slot, RoPE, and replay helpers for DeepSeek-V4.1."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor
from vllm import forward_context

from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

from .cache import NativeCacheView, PagedCacheRows


def rotate_rope(
    inputs: Tensor,
    positions: Tensor,
    rotary_emb: nn.Module,
    kv_rope_op: Callable[[Tensor, Tensor, Tensor], Tensor],
) -> Tensor:
    """Apply rotary position embeddings to ``inputs`` with clamped int32 positions.

    Args:
        inputs: Query or key activation tensor (Used; contiguous view passed to op).
        positions: Token or compressed-group positions ``[num_tokens]`` (Used).
        rotary_emb: Rotary embedding module owning ``cos_sin_cache`` (Used).
        kv_rope_op: Platform RoPE callable ``(inputs, pos_i32, cos_sin_cache)`` (Used).

    Returns:
        Rotated activation tensor with the same shape and dtype as ``inputs`` (Created).
    """
    cos_sin_cache = cast(Tensor, rotary_emb.cos_sin_cache)
    max_pos = cos_sin_cache.shape[0] - 1
    clamped_positions = positions.clamp(0, max_pos).int()
    return kv_rope_op(inputs.contiguous(), clamped_positions, cos_sin_cache)


def active_cache(layer: NativeCacheView | nn.Module) -> PagedCacheRows | Tensor:
    """Return the bound ``PagedCacheRows`` wrapper when present, else ``kv_cache``.

    Data lifecycle:
        Used: ``layer`` cache-owning module; returns existing reference without copy.
    """
    layer = cast(NativeCacheView, layer)
    paged = layer.paged_cache_rows
    if paged is not None:
        return paged
    return layer.kv_cache


def layer_meta(prefix: str) -> AttentionMetadata | None:
    """Look up per-layer ``AttentionMetadata`` from the active vLLM forward context.

    Args:
        prefix: Layer or cache module prefix key in ``attn_metadata`` dict.

    Returns:
        Matching ``AttentionMetadata`` instance, or ``None`` if absent.
    """
    attn_metadata = forward_context.get_forward_context().attn_metadata
    if isinstance(attn_metadata, dict):
        return cast(AttentionMetadata | None, attn_metadata.get(prefix))
    return cast(AttentionMetadata | None, attn_metadata)


def block_table(meta: AttentionMetadata) -> Tensor:
    """Reshape ``meta.block_tables`` to ``[num_seqs, max_blocks_per_seq]``.

    Args:
        meta: Per-layer ``AttentionMetadata`` (Used).

    Returns:
        2-D ``int32[num_seqs, max_blocks_per_seq]`` block table view (Created/Used).
    """
    num_seqs = cast(Tensor, meta.seq_lens).shape[0]
    return cast(Tensor, meta.block_tables).reshape(num_seqs, -1)


def query_token_requests(query_start_loc: Tensor, positions: Tensor) -> Tensor:
    """Map each token row to its request index from ``query_start_loc``.

    Args:
        query_start_loc: Cumulative query start offsets ``[num_reqs + 1]`` (Used).
        positions: Token positions ``[num_tokens]`` (Used for shape and device).

    Returns:
        ``int32[num_tokens]`` request index per token row (Created).
    """
    rows = torch.arange(positions.shape[0], dtype=torch.int32, device=positions.device)
    query_ends = query_start_loc[1:].to(positions.device)
    return (rows.unsqueeze(1) >= query_ends.unsqueeze(0)).sum(dim=-1, dtype=torch.int32)


def token_requests(meta: AttentionMetadata, positions: Tensor) -> Tensor:
    """Map each token row ``[0, num_tokens)`` to its request index.

    Args:
        meta: Per-layer ``AttentionMetadata`` carrying ``query_start_loc`` (Used).
        positions: Token positions ``[num_tokens]`` (Used).

    Returns:
        ``int32[num_tokens]`` request index per token row (Created).
    """
    return query_token_requests(cast(Tensor, meta.query_start_loc), positions)


def cache_slots(
    cache: PagedCacheRows | Tensor,
    meta: AttentionMetadata,
    positions: Tensor,
    *,
    compress_ratio: int = 1,
    ring: bool = False,
) -> Tensor:
    """Compute physical slot indices ``int32[num_tokens]`` for paged or ring caches.

    Tokens past ``meta.query_start_loc[-1]``, empty batches, negative blocks, and
    ``null_block=0`` blocks (when ``cache`` reserves block 0) are masked to ``-1``.

    Args:
        cache: Target ``PagedCacheRows`` or dense cache tensor (Used).
        meta: Per-layer ``AttentionMetadata`` (Used).
        positions: Token positions ``[num_tokens]`` (Used).
        compress_ratio: KV compression ratio divisor (``1`` or ``2``).
        ring: When ``True``, indexes block column ``0`` modulo page capacity.

    Returns:
        ``int32[num_tokens]`` flat slot indices (``-1`` for masked tokens; Created).
    """
    if cast(Tensor, meta.seq_lens).numel() == 0:
        return torch.full_like(positions, -1, dtype=torch.int32)
    table = block_table(meta)
    rows_per_page = cache.shape[1]
    req_indices = token_requests(meta, positions).clamp(max=table.shape[0] - 1)
    logical_positions = positions.int() // compress_ratio
    num_cols = table.shape[1]
    if ring:
        block_cols = torch.zeros_like(logical_positions)
    else:
        block_cols = (logical_positions // rows_per_page).clamp(max=num_cols - 1)
    flat_indices = (req_indices * num_cols + block_cols).long()
    blocks = table.reshape(-1).index_select(0, flat_indices).int()
    slots = blocks * rows_per_page + logical_positions % rows_per_page
    min_block = 1 if isinstance(cache, PagedCacheRows) and cache.null_block else 0
    token_rows = torch.arange(
        positions.shape[0], dtype=torch.int32, device=positions.device
    )
    query_limit = cast(Tensor, meta.query_start_loc)[-1]
    invalid = (token_rows >= query_limit) | (blocks < min_block)
    return slots.masked_fill(invalid, -1)


def _context_starts(key: str, num_seqs: int, device: torch.device) -> Tensor | None:
    """Fetch per-sequence ``int32[num_seqs]`` lower-bound positions from context.

    Args:
        key: Key in ``forward_context.additional_kwargs``.
        num_seqs: Number of scheduled sequences in the batch.
        device: Target device for the returned tensor.

    Returns:
        Per-sequence start positions ``int32[num_seqs]``, or ``None`` if absent.
    """
    raw = forward_context.get_forward_context().additional_kwargs.get(key)
    if not (isinstance(raw, Tensor) and raw.numel() > 0):
        return None
    seq_indices = torch.arange(num_seqs, device=device).clamp_max(raw.shape[0] - 1)
    return raw.int()[seq_indices]


def replay_starts(
    num_seqs: int,
    device: torch.device,
    *,
    ced_decoder: bool = False,
) -> Tensor | None:
    """Resolve effective per-sequence replay/window lower bounds ``int32[num_seqs]``.

    Args:
        num_seqs: Number of scheduled sequences in the batch.
        device: Target device for the returned tensor.
        ced_decoder: Whether to resolve the lower bound for the CED decoder stage.

    Returns:
        Per-sequence lower bounds ``int32[num_seqs]``, or ``None`` if no replay
        or window lower bounds are active.
    """
    window_starts = _context_starts("deepseek_v41_window_starts", num_seqs, device)
    decoder_starts = _context_starts("deepseek_v41_replay_starts", num_seqs, device)
    if ced_decoder:
        if window_starts is not None and decoder_starts is not None:
            return torch.maximum(window_starts, decoder_starts)
        return decoder_starts if decoder_starts is not None else window_starts
    return window_starts if window_starts is not None else decoder_starts


def not_replayed(
    meta: AttentionMetadata,
    positions: Tensor,
    window_size: int,
) -> Tensor:
    """Return boolean mask ``bool[num_tokens]`` excluding CED prefix-replay tokens.

    For requests with ``replay_start > 0``, tokens in
    ``[replay_start, replay_start + window_size)`` already reside in global
    compressed and indexer caches and must not be rewritten, whereas newly
    scheduled suffix tokens ``>= replay_start + window_size`` remain writable.

    Args:
        meta: Attention metadata containing ``seq_lens``.
        positions: Token positions ``int32[num_tokens]``.
        window_size: Sliding-window size defining the prefix replay span.

    Returns:
        Boolean write-eligibility mask ``bool[num_tokens]``.
    """
    num_seqs = max(cast(Tensor, meta.seq_lens).shape[0], 1)
    starts = replay_starts(num_seqs, positions.device)
    if starts is None:
        return torch.ones_like(positions, dtype=torch.bool)
    req_indices = token_requests(meta, positions).clamp_max(starts.shape[0] - 1)
    seq_starts = starts[req_indices]
    in_replay_window = (
        (seq_starts > 0)
        & (positions >= seq_starts)
        & (positions < seq_starts + window_size)
    )
    return ~in_replay_window

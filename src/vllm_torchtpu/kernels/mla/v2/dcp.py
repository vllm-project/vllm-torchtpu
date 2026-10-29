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
"""Decode context parallelism (DCP) for the dense MLA v2 kernel, per rank.

Under DCP the latent cache is split by position over `dcp_size` ranks, in
`interleave_size` chunks: rank `r` owns the positions `p` with
`(p // interleave_size) % dcp_size == r`, and stores them in order at local
slots `0, 1, ...` of the pages its block-table row names (the vLLM slot-mapping
convention; a block-table entry is the same physical id on every rank). Every
rank attends the whole DCP group's query heads over its own shard and returns
the log-sum-exp, and the partial outputs are merged across ranks.

The MLA v2 kernel needs no DCP mode for decode. A decode query sits at the last
global position, so it attends every key a rank holds and the causal mask is
all-true; the kernel only has to be told the local length. What remains is the
kernel's contract that the last `q_len` keys come from `new_kv` and are written
back to the cache:

- the rank that owns the new token passes it as usual and the kernel stores it
  at its local slot;
- every other rank passes a copy of its own last cached row instead, so the
  kernel attends exactly the local keys and rewrites that row with the bytes it
  already holds;
- a rank that holds nothing of a sequence yet runs it over one copied row and
  its output is then replaced by `(0, -inf)`, the identity of the merge.

Prefill tokens need global positions in their causal masks, so they do not
use this trick; see `mla_prefill_dcp_local`.
"""

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.kv_cache_utils import (
    _row_positions,
    get_dcp_page_and_slot,
)
from vllm_torchtpu.kernels.mla.v2.kernel import mla_ragged_paged_attention


def _check_layout(page_size: int, dcp_size: int, interleave_size: int) -> None:
    if dcp_size < 1 or interleave_size < 1 or page_size % interleave_size:
        raise ValueError(
            f"interleave_size={interleave_size} must divide page_size={page_size} "
            f"(dcp_size={dcp_size}); otherwise a rank's local slots stop being "
            "its owned positions in order."
        )


def local_kv_lens(
    seq_lens: jax.Array,
    dcp_rank: jax.Array | int,
    *,
    dcp_size: int,
    interleave_size: int,
) -> jax.Array:
    """How many of the first `seq_lens` positions rank `dcp_rank` owns."""
    cycle = dcp_size * interleave_size
    rank_start = jnp.asarray(dcp_rank, jnp.int32) * interleave_size
    partial = jnp.clip(seq_lens % cycle - rank_start, 0, interleave_size)
    return (seq_lens // cycle) * interleave_size + partial


def _packed_rows(cache_kv: jax.Array, kv_c: jax.Array, k_pe: jax.Array) -> jax.Array:
    """`[T, lanes]` cache rows `[kv_c | k_pe | 0]`, as the kernel stores them."""
    lanes = cache_kv.shape[-1]
    rows = jnp.concatenate([kv_c, k_pe], axis=-1).astype(cache_kv.dtype)
    return jnp.pad(rows, ((0, 0), (0, lanes - rows.shape[-1])))


def update_kv_cache_dcp(
    cache_kv: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    dcp_rank: jax.Array | int,
    *,
    dcp_size: int,
    interleave_size: int,
) -> jax.Array:
    """Write the new tokens this rank owns into its shard of the MLA v2 cache.

    Args:
      cache_kv: `[num_pages, page_size // packing, packing, lanes]` local shard.
      new_kv_c, new_k_pe: `[T, lkv]`, `[T, r]` this step's rows, request-major,
        identical on every rank.
      seq_lens, page_indices, cu_q_lens: global sequence lengths (including this
        step), block table (flat or 2-D), cumulative query lengths.
      dcp_rank: this rank's index, int or traced scalar.

    Returns:
      The updated shard. Rows of tokens owned elsewhere, and padding, are
      dropped.
    """
    num_pages, rows_per_page, packing, _ = cache_kv.shape
    page_size = rows_per_page * packing
    _check_layout(page_size, dcp_size, interleave_size)
    page, slot = get_dcp_page_and_slot(
        new_kv_c.shape[0],
        seq_lens,
        page_indices,
        cu_q_lens,
        dcp_rank,
        page_size=page_size,
        num_pages=num_pages,
        dcp_size=dcp_size,
        interleave_size=interleave_size,
    )
    rows = _packed_rows(cache_kv, new_kv_c, new_k_pe)
    return cache_kv.at[page, slot // packing, slot % packing].set(rows, mode="drop")


def decode_kernel_inputs(
    cache_kv: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    dcp_rank: jax.Array | int,
    *,
    dcp_size: int,
    interleave_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Kernel inputs that make an unmodified decode call attend only this shard.

    Every scheduled sequence must have exactly one query token.

    Returns:
      `(kv_lens, new_kv_c, new_k_pe, has_kv)`: per-sequence local lengths
      (at least 1), the new rows with non-owned tokens replaced by the rank's
      last cached row, and per-token `False` where the rank holds nothing of
      that token's sequence (padding tokens count as `True`).
    """
    num_seqs = seq_lens.shape[0]
    _, rows_per_page, packing, _ = cache_kv.shape
    page_size = rows_per_page * packing
    _check_layout(page_size, dcp_size, interleave_size)

    local = local_kv_lens(
        seq_lens, dcp_rank, dcp_size=dcp_size, interleave_size=interleave_size
    )
    kv_lens = jnp.maximum(local, 1)

    seq_id, pos, in_batch = _row_positions(new_kv_c.shape[0], seq_lens, cu_q_lens)
    rank = jnp.asarray(dcp_rank, jnp.int32)
    owned = (jnp.maximum(pos, 0) // interleave_size) % dcp_size == rank

    last = kv_lens[seq_id] - 1
    table = page_indices.reshape(num_seqs, -1)
    last_page = table[seq_id, jnp.minimum(last // page_size, table.shape[1] - 1)]
    last_slot = last % page_size
    cached = cache_kv[last_page, last_slot // packing, last_slot % packing]
    lkv, r = new_kv_c.shape[-1], new_k_pe.shape[-1]
    keep = owned[:, None]
    new_kv_c = jnp.where(keep, new_kv_c, cached[:, :lkv].astype(new_kv_c.dtype))
    new_k_pe = jnp.where(
        keep, new_k_pe, cached[:, lkv : lkv + r].astype(new_k_pe.dtype)
    )
    has_kv = jnp.logical_or(local[seq_id] > 0, jnp.logical_not(in_batch))
    return kv_lens, new_kv_c, new_k_pe, has_kv


def mla_decode_dcp_local(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    dcp_rank: jax.Array | int,
    *,
    dcp_size: int,
    interleave_size: int,
    **kernel_kwargs,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One rank's share of a DCP decode step with the MLA v2 kernel.

    Arguments are those of `mla_ragged_paged_attention` with global `seq_lens`
    and the rank's own cache shard; the batch must be decode-only. Stores the
    new tokens this rank owns.

    Returns:
      `(out [H, T, lkv], cache_kv, lse f32[T, H])`, ready for an LSE merge:
      tokens whose sequence has no key on this rank get `out = 0` and
      `lse = -inf`.
    """
    kv_lens, new_kv_c, new_k_pe, has_kv = decode_kernel_inputs(
        cache_kv,
        new_kv_c,
        new_k_pe,
        seq_lens,
        page_indices,
        cu_q_lens,
        dcp_rank,
        dcp_size=dcp_size,
        interleave_size=interleave_size,
    )
    out, cache_kv, lse = mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        return_lse=True,
        **kernel_kwargs,
    )
    out = jnp.where(has_kv[None, :, None], out, jnp.zeros_like(out))
    lse = jnp.where(has_kv[:, None], lse, -jnp.inf)
    return out, cache_kv, lse


# Prefill: every rank attends its own heads over the whole context, gathered.
#
# With `interleave_size == page_size`, global page `k` of a sequence is page
# `block_tables[s, k // d]` of rank `k % d`. Each rank packs the pages it holds
# of the step's prefill contexts, the group all-gathers them into one pool, and
# the unmodified kernel runs on the pool with a page table that lists global
# pages in order. Its causal mask then sees global positions, and the new keys
# come from `new_kv` as usual; what it stores lands in the pool and is dropped.


def prefill_pool_size(max_model_len: int, page_size: int, dcp_size: int) -> int:
    """Pages per rank in the pool: one sequence's whole context, plus a scratch."""
    return -(-max_model_len // (page_size * dcp_size)) + 1


def _prefill_contexts(seq_lens, cu_q_lens, distribution, page_size, dcp_size):
    """Per sequence: context pages, this rank's page count and its pool offset."""
    num_seqs = seq_lens.shape[0]
    seq = jnp.arange(num_seqs, dtype=jnp.int32)
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    prefill = (seq >= distribution[0]) & (seq < distribution[2])
    ctx = jnp.where(prefill, seq_lens - q_lens, 0)
    ctx_pages = -(-ctx // page_size)
    local_pages = -(-ctx // (page_size * dcp_size))
    offsets = jnp.cumsum(local_pages) - local_pages
    return prefill, ctx_pages, local_pages, offsets


def prefill_pool_pages(
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    dcp_size: int,
    max_model_len: int,
) -> jax.Array:
    """This rank's `[pool_size, ...]` share of the prefill pool.

    Sequence `s` holds this rank's pages of its first `cdiv(ctx_s, page * d)`
    blocks at rows `offsets[s]...`; the last row is scratch. The step's prefill
    contexts must fit (`sum_s cdiv(ctx_s, page * d) < pool_size`), which the
    caller checks on the host: rows past the end are silently dropped.
    """
    num_seqs = seq_lens.shape[0]
    page_size = cache_kv.shape[1] * cache_kv.shape[2]
    size = prefill_pool_size(max_model_len, page_size, dcp_size)
    _, _, local_pages, offsets = _prefill_contexts(
        seq_lens, cu_q_lens, distribution, page_size, dcp_size
    )
    row = jnp.arange(size, dtype=jnp.int32)
    seq = jnp.searchsorted(offsets + local_pages, row, side="right").astype(jnp.int32)
    seq = jnp.minimum(seq, num_seqs - 1)
    table = page_indices.reshape(num_seqs, -1)
    block = jnp.clip(row - offsets[seq], 0, table.shape[1] - 1)
    used = row < (offsets + local_pages)[-1]
    return cache_kv[jnp.where(used, table[seq, block], 0)]


def prefill_pool_table(
    seq_lens: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    page_size: int,
    dcp_size: int,
    max_model_len: int,
) -> tuple[jax.Array, jax.Array]:
    """Kernel `(page_indices, kv_lens)` over the all-gathered pool.

    Context pages map into the pool in global order; every other page, and
    every non-prefill sequence (run as one self-attending token), maps to the
    scratch row, where the kernel's writes are discarded.
    """
    size = prefill_pool_size(max_model_len, page_size, dcp_size)
    prefill, ctx_pages, _, offsets = _prefill_contexts(
        seq_lens, cu_q_lens, distribution, page_size, dcp_size
    )
    k = jnp.arange(-(-max_model_len // page_size), dtype=jnp.int32)[None, :]
    pool_row = (k % dcp_size) * size + offsets[:, None] + k // dcp_size
    in_ctx = prefill[:, None] & (k < ctx_pages[:, None])
    table = jnp.where(in_ctx, pool_row, size - 1)
    kv_lens = jnp.where(prefill, seq_lens, 1)
    return table.reshape(-1), kv_lens


def mla_prefill_dcp_local(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    dcp_size: int,
    axis_name: str,
    max_model_len: int,
    **kernel_kwargs,
) -> jax.Array:
    """This rank's own heads for the step's prefill tokens, `[H, T, lkv]`.

    Call inside a `shard_map` over `axis_name`, with
    `interleave_size == page_size`. Rows of non-prefill tokens are garbage.
    The cache is only read; `update_kv_cache_dcp` stores the new tokens.
    """
    page_size = cache_kv.shape[1] * cache_kv.shape[2]
    local = prefill_pool_pages(
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        dcp_size=dcp_size,
        max_model_len=max_model_len,
    )
    pool = jax.lax.all_gather(local, axis_name, tiled=True)
    table, kv_lens = prefill_pool_table(
        seq_lens,
        cu_q_lens,
        distribution,
        page_size=page_size,
        dcp_size=dcp_size,
        max_model_len=max_model_len,
    )
    out, _ = mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        pool,
        kv_lens,
        table,
        cu_q_lens,
        distribution,
        **kernel_kwargs,
    )
    return out

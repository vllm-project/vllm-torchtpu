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
"""Token-parallel prefill attention over a DCP position-sharded KV cache.

Under DCP every rank attends every query token with every head against its own
1/dcp of the positions, so a prefill step moves each token's full-head query to
every rank and each rank's full-head partial output back. For a large bucket
that traffic dwarfs the latent KV the step reads: an MLA latent row is shared by
all heads.

The token-parallel split keeps rows `[0, split_rows)` of the step on the DCP
path and splits rows `[split_rows, T)` over the DCP ranks by token: rank `j`
owns rows `split_rows + j * R + [0, R)`, `R = T // dcp` (rows past `T` are
padding). Each rank

  1. gathers, from every rank, the cache pages of the requests those rows
     belong to (`gather_page_ids`, `gather_latent`, `gather_index_k`): a
     rank-major buffer of `max_pages` virtual pages per rank;
  2. receives its rows' queries for all heads by one all-to-all of the
     head-sharded queries (`heads_to_tokens`);
  3. runs the single-device indexer and sparse MLA kernels on its rows, with
     an exact top-k and an exact softmax: no cross-rank merge of either;
  4. returns its outputs to head sharding by one all-to-all (`tokens_to_heads`).

Everything below runs inside a `shard_map` over the DCP axis with this rank's
index as traced data, so one compiled program serves every rank. The kernels
are passed in, so the layout is checkable on CPU against jnp references.

Coordinates. A virtual page spans `page_size * dcp` positions; position `p` of
a request lives on rank `owner(p)` at slot `slot(p)` of the request's local page
`p // virtual_page_size` (`kv_cache_utils._get_dcp_page_and_slot`). The gather
buffer stacks each rank's `max_pages` compacted pages rank-major, the requests'
pages back to back from `offsets[r]`, so position `p` of request `r` is flat row

    (owner(p) * max_pages + offsets[r] + p // virtual_page_size) * page_size
        + slot(p)

which depends only on the step's sequence lengths, not on the block tables:
the indexer (its own cache group) and every attention layer agree on it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


@dataclasses.dataclass(frozen=True)
class TokenParallelLayout:
    """Static shape of the token-parallel split, one per compiled program."""

    dcp_size: int
    interleave_size: int
    # Tokens per rank-local cache page (latent and index-K caches alike).
    page_size: int
    # Rows [0, split_rows) of a step stay on the DCP path.
    split_rows: int
    # Virtual pages the gather buffer holds per rank, over all requests.
    max_pages: int
    # Page counts a step's gather may be sized to (ascending, ending at
    # max_pages): it moves the smallest that holds the step's pages, so the
    # collective scales with the context rather than the buffer.
    gather_caps: tuple[int, ...] = ()

    def __post_init__(self):
        if self.dcp_size < 2:
            raise ValueError(f"token-parallel split needs dcp_size > 1, got {self}")
        if self.page_size % (self.dcp_size * self.interleave_size):
            raise ValueError(
                "page_size must hold whole interleave cycles for the natural "
                f"index-K order: {self}"
            )
        if self.split_rows < 0 or self.max_pages < 1:
            raise ValueError(f"bad split_rows / max_pages: {self}")
        caps = self.gather_caps or (self.max_pages,)
        if list(caps) != sorted(set(caps)) or caps[-1] != self.max_pages or caps[0] < 1:
            raise ValueError(f"gather_caps must ascend to max_pages: {self}")
        object.__setattr__(self, "gather_caps", tuple(caps))

    @property
    def virtual_page_size(self) -> int:
        return self.page_size * self.dcp_size

    def rows_per_rank(self, num_tokens: int) -> int:
        if num_tokens % self.dcp_size or num_tokens <= self.split_rows:
            raise ValueError(
                f"token-parallel bucket of {num_tokens} tokens must exceed "
                f"split_rows={self.split_rows} and divide by dcp={self.dcp_size}"
            )
        return num_tokens // self.dcp_size


def rank_as_data(axis_name: str, axis_size: int) -> jax.Array:
    """This shard's index along `axis_name` as traced data.

    The same all-to-all of an iota as `streamindex_topk.cp_rank_as_data`
    (`lax.axis_index` lowers to a partition-id the torch_tpu export refuses).
    """
    ranks = jnp.arange(axis_size, dtype=jnp.int32)
    return lax.all_to_all(ranks, axis_name, 0, 0, tiled=True)[0]


def owner_and_slot(
    layout: TokenParallelLayout, positions: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """(rank, slot in its local page) of each global position."""
    cycle = layout.dcp_size * layout.interleave_size
    il = layout.interleave_size
    offset = positions % layout.virtual_page_size
    return (offset % cycle) // il, (offset // cycle) * il + offset % il


def step_pages(
    layout: TokenParallelLayout,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    num_seqs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Virtual pages each request puts in the gather buffer, and its offset.

    A request is gathered iff it has a row at or past `split_rows`; it then
    contributes every page up to its current length, in step order.
    """
    req = jnp.arange(seq_lens.shape[0], dtype=jnp.int32)
    q_start, q_end = query_start_loc[:-1], query_start_loc[1:]
    listed = (req < num_seqs) & (q_end > layout.split_rows) & (q_end > q_start)
    vps = layout.virtual_page_size
    pages = jnp.where(listed, (seq_lens + vps - 1) // vps, 0).astype(jnp.int32)
    return pages, jnp.cumsum(pages) - pages


def total_step_pages(
    layout: TokenParallelLayout,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    num_seqs: jax.Array,
) -> jax.Array:
    """What the gather buffer must hold; at most `max_pages` for a step to run."""
    return jnp.sum(step_pages(layout, seq_lens, query_start_loc, num_seqs)[0])


def gather_page_ids(
    layout: TokenParallelLayout,
    block_tables: jax.Array,
    pages: jax.Array,
    offsets: jax.Array,
    num_local_pages: int,
) -> jax.Array:
    """i32[max_pages]: the local cache page behind each buffer page, 0 past the
    step's pages. `block_tables` holds virtual page ordinals, resolved per rank
    as `% num_local_pages` like every DCP reader."""
    num_reqs = pages.shape[0]
    table = block_tables.reshape(num_reqs, -1)
    k = jnp.arange(layout.max_pages, dtype=jnp.int32)
    ends = offsets + pages
    req = jnp.minimum(jnp.searchsorted(ends, k, side="right"), num_reqs - 1)
    vpage = jnp.clip(k - offsets[req], 0, table.shape[1] - 1)
    page = jnp.mod(table[req, vpage], jnp.int32(num_local_pages))
    return jnp.where(k < ends[-1], page, 0).astype(jnp.int32)


def flat_rows(
    layout: TokenParallelLayout, positions: jax.Array, row_offsets: jax.Array
) -> jax.Array:
    """Global positions `[N, K]` (`-1` = none) of the requests whose buffer
    offsets are `row_offsets [N]` -> flat rows of the gathered latent buffer."""
    owner, slot = owner_and_slot(layout, positions)
    page = (
        owner * layout.max_pages
        + row_offsets[:, None]
        + positions // layout.virtual_page_size
    )
    return jnp.where(positions >= 0, page * layout.page_size + slot, -1)


class RankRows(NamedTuple):
    """This rank's rows as a ragged batch of its own. Local request `k` is
    global request `first_req + k`; requests past `num_seqs` are empty."""

    seq_lens: jax.Array  # i32[R], truncated after this rank's last row
    cu_q_lens: jax.Array  # i32[R + 1], local rows
    num_seqs: jax.Array  # i32
    first_req: jax.Array  # i32
    num_rows: jax.Array  # i32, real rows; the rest of the R rows are padding


def rank_rows(
    layout: TokenParallelLayout,
    rank: jax.Array,
    num_tokens: int,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    num_seqs: jax.Array,
) -> RankRows:
    """Per-rank sequence metadata for rows `split_rows + rank * R + [0, R)`.

    Each row keeps its global position and causal bound: a request's
    `seq_len` is cut by the rows that follow this rank's range, so the last
    local row of the request sits at `seq_len' - 1`.
    """
    num_reqs = seq_lens.shape[0]
    rows = layout.rows_per_rank(num_tokens)
    total = query_start_loc[num_seqs]
    lo = layout.split_rows + rank * rows
    hi = jnp.clip(total, lo, lo + rows)
    q_start, q_end = query_start_loc[:-1], query_start_loc[1:]
    req = jnp.arange(num_reqs, dtype=jnp.int32)
    active = req < num_seqs
    a, b = jnp.clip(q_start, lo, hi), jnp.clip(q_end, lo, hi)
    first = jnp.searchsorted(q_end, lo, side="right").astype(jnp.int32)
    n_local = jnp.sum(active & (req >= first) & (q_start < hi)).astype(jnp.int32)
    k = jnp.arange(num_reqs, dtype=jnp.int32)
    idx = jnp.minimum(first + k, num_reqs - 1)
    keep = k < n_local
    q_len = jnp.where(keep, (b - a)[idx], 0)
    local_seq = jnp.where(keep, (seq_lens - (q_end - b))[idx], 0)
    cu = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(q_len)])
    return RankRows(
        local_seq.astype(jnp.int32),
        cu.astype(jnp.int32),
        n_local,
        first,
        (hi - lo).astype(jnp.int32),
    )


def row_requests(rr: RankRows, num_rows: int) -> jax.Array:
    """Global request of each local row (clamped for padding rows)."""
    i = jnp.arange(num_rows, dtype=jnp.int32)
    local = jnp.searchsorted(rr.cu_q_lens[1:], i, side="right").astype(jnp.int32)
    local = jnp.minimum(local, jnp.maximum(rr.num_seqs - 1, 0))
    return jnp.minimum(rr.first_req + local, rr.seq_lens.shape[0] - 1)


def natural_page_table(
    layout: TokenParallelLayout, rr: RankRows, offsets: jax.Array, vpages_per_seq: int
) -> jax.Array:
    """i32[R * vpages_per_seq * dcp]: each local request's pages of the
    natural-order index-K buffer (`gather_index_k`)."""
    num_reqs = rr.seq_lens.shape[0]
    k = jnp.arange(num_reqs, dtype=jnp.int32)
    req = jnp.minimum(rr.first_req + k, num_reqs - 1)
    width = vpages_per_seq * layout.dcp_size
    pages = offsets[req][:, None] * layout.dcp_size + jnp.arange(width)[None, :]
    pages = jnp.minimum(pages, layout.max_pages * layout.dcp_size - 1)
    return pages.reshape(-1).astype(jnp.int32)


def _sized_gather(
    layout: TokenParallelLayout,
    total_pages: jax.Array,
    gather: Callable[[int], jax.Array],
) -> jax.Array:
    """`gather(cap)` for the smallest cap in `gather_caps` holding
    `total_pages`. Every rank takes the same branch (the metadata is
    replicated); each returns `[dcp, max_pages, ...]`, pages past the cap
    zero (no flat row reaches them)."""

    def branch(cap: int) -> Callable[[], jax.Array]:
        def run() -> jax.Array:
            g = gather(cap)
            pad = [(0, 0)] * g.ndim
            pad[1] = (0, layout.max_pages - cap)
            return jnp.pad(g, pad)

        return run

    caps = layout.gather_caps
    if len(caps) == 1:
        return branch(caps[0])()
    idx = jnp.searchsorted(jnp.asarray(caps, jnp.int32), total_pages, side="left")
    return lax.switch(jnp.minimum(idx, len(caps) - 1), [branch(c) for c in caps])


def gather_latent(
    layout: TokenParallelLayout,
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    page_ids: jax.Array,
    total_pages: jax.Array,
    axis_name: str,
) -> tuple[jax.Array, jax.Array]:
    """Every rank's `page_ids` pages, rank-major: `[dcp * max_pages, ...]`."""

    def gather(cache: jax.Array) -> jax.Array:
        g = _sized_gather(
            layout,
            total_pages,
            lambda cap: lax.all_gather(cache[page_ids[:cap]], axis_name, axis=0),
        )
        return g.reshape(-1, *g.shape[2:])

    return gather(nope_cache), gather(rope_cache)


def gather_index_k(
    layout: TokenParallelLayout,
    cache: jax.Array,
    page_ids: jax.Array,
    total_pages: jax.Array,
    axis_name: str,
) -> jax.Array:
    """Every rank's `page_ids` pages of a token-along-lane index-K cache
    (`[pages, rows, 4, page_size]`, slot on the last axis), reordered into
    natural position order: natural page `offsets[r] * dcp + q` of request `r`
    holds its positions `[q * page_size, (q + 1) * page_size)`.

    Slot `c * il + i` of rank `r` is in-page position `c * dcp * il + r * il +
    i`; with `c = q * cpp + c'` (`cpp = page_size / (dcp * il)` cycles per
    natural page) that is natural page `q`, lane `c' * dcp * il + r * il + i`.
    """
    dcp, il, ps = layout.dcp_size, layout.interleave_size, layout.page_size
    if cache.ndim != 4 or cache.shape[-1] != ps:
        raise ValueError(
            f"index-K cache {cache.shape} is not token-along-lane with page {ps}"
        )
    cpp = ps // (dcp * il)

    def gather(cap: int) -> jax.Array:
        g = lax.all_gather(cache[page_ids[:cap]], axis_name, axis=0)
        rows, sub = g.shape[2:4]
        g = g.reshape(dcp, cap, rows, sub, dcp, cpp, il)
        # -> [cap, dcp (natural page q), ...]; returned as [cap, dcp, ...] so
        # the padding below extends the virtual-page axis.
        return g.transpose(1, 4, 2, 3, 5, 0, 6).reshape(cap, dcp, rows, sub, ps)

    def gather_t(cap: int) -> jax.Array:
        return gather(cap).swapaxes(0, 1)  # pad axis 1 = virtual pages

    g = _sized_gather(layout, total_pages, gather_t).swapaxes(0, 1)
    return g.reshape(layout.max_pages * dcp, *g.shape[2:])


def heads_to_tokens(
    layout: TokenParallelLayout, x: jax.Array, axis_name: str
) -> jax.Array:
    """`[T, h, d]` this rank's heads of every row -> `[R, h * dcp, d]` every
    head of this rank's token-parallel rows (heads in rank order)."""
    s = layout.split_rows
    x = jnp.concatenate([x[s:], jnp.zeros((s, *x.shape[1:]), x.dtype)])
    return lax.all_to_all(x, axis_name, 0, 1, tiled=True)


def tokens_to_heads(y: jax.Array, axis_name: str) -> jax.Array:
    """Inverse of `heads_to_tokens`: `[R, H, d]` -> `[T, H / dcp, d]`, row `i`
    being step row `split_rows + i` (the last `split_rows` rows are padding)."""
    return lax.all_to_all(y, axis_name, 1, 0, tiled=True)


# (q [R, H, D], nope buffer, rope buffer, flat top-k [R, K], real rows) -> [R, H, L]
AttendFn = Callable[[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]
# (nope, rope, kv_c, k_pe, seq_lens, block_tables, query_start_loc, rank) -> caches
WriteKVFn = Callable[..., tuple[jax.Array, jax.Array]]
# (q [R, h, d], weights [R, h], natural cache, RankRows, page table) -> positions
SelectTopkFn = Callable[
    [jax.Array, jax.Array, jax.Array, RankRows, jax.Array], jax.Array
]


def _rank(rank_ids: tuple[jax.Array, ...], axis_name: str, dcp: int) -> jax.Array:
    return rank_ids[0][0] if rank_ids else rank_as_data(axis_name, dcp)


def sparse_mla_attention(
    layout: TokenParallelLayout,
    mesh: Mesh,
    axis_name: str,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    q: jax.Array,
    kv_c_normed: jax.Array,
    k_pe: jax.Array,
    flat_topk: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    *,
    write_kv: WriteKVFn,
    attend: AttendFn,
    rank_ids: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Token-parallel sparse MLA for one layer (global view).

    Args:
      kv_cache_nope, kv_cache_rope: position shards over `axis_name`.
      q: `[dcp * T, h, D]`, each rank's `[T, h, D]` head slice of every row
        (latent nope and rope concatenated).
      kv_c_normed, k_pe: this step's latents, replicated; `write_kv` writes
        every row this rank owns before anything is gathered.
      flat_topk: `[dcp * R, K]`, each rank's rows' top-k as flat buffer rows
        (`flat_rows`), `-1` padded.
      seq_lens, block_tables, query_start_loc, request_distribution: the
        step's metadata, replicated.
      rank_ids: optional i32[dcp] sharded over `axis_name`, entry r = r.

    Returns:
      `(kv_cache_nope, kv_cache_rope, out)`, `out` `[dcp * T, h, L]`: each
      rank's head slice, row `i` being step row `split_rows + i`.
    """
    dcp = layout.dcp_size
    sharded = P(axis_name)
    rep = P()

    def _local(nope, rope, q, kv_c, k_pe, topk, seq_lens, bt, qsl, dist, *rank_ids):
        rank = _rank(rank_ids, axis_name, dcp)
        nope, rope = write_kv(nope, rope, kv_c, k_pe, seq_lens, bt, qsl, rank)
        pages, offsets = step_pages(layout, seq_lens, qsl, dist[2])
        page_ids = gather_page_ids(layout, bt, pages, offsets, nope.shape[0])
        g_nope, g_rope = gather_latent(
            layout, nope, rope, page_ids, jnp.sum(pages), axis_name
        )
        q_tok = heads_to_tokens(layout, q, axis_name)
        rr = rank_rows(layout, rank, q.shape[0], seq_lens, qsl, dist[2])
        out = attend(q_tok, g_nope, g_rope, topk, rr.num_rows)
        return nope, rope, tokens_to_heads(out, axis_name)

    extra = () if rank_ids is None else (rank_ids,)
    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(sharded, sharded, sharded, rep, rep, sharded, rep, rep, rep, rep)
        + (sharded,) * len(extra),
        out_specs=(sharded, sharded, sharded),
        check_vma=False,
    )(
        kv_cache_nope,
        kv_cache_rope,
        q,
        kv_c_normed,
        k_pe,
        flat_topk,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        *extra,
    )


def indexer_topk(
    layout: TokenParallelLayout,
    mesh: Mesh,
    axis_name: str,
    index_cache: jax.Array,
    q: jax.Array,
    weights: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    *,
    select_topk: SelectTopkFn,
    rank_ids: jax.Array | None = None,
) -> jax.Array:
    """Token-parallel indexer top-k for one layer (global view).

    `index_cache` (position shards, this step's K already written), `q`
    `[T, h, d]` and `weights` `[T, h]` (replicated, every row). Returns
    `[dcp * R, K]`: each rank's rows' top-k as flat rows of the latent gather
    buffer (`flat_rows`), `-1` padded, and `-1` on padding rows.
    """
    dcp = layout.dcp_size
    sharded = P(axis_name)
    rep = P()

    def _local(cache, q, weights, seq_lens, bt, qsl, dist, *rank_ids):
        rank = _rank(rank_ids, axis_name, dcp)
        num_tokens = q.shape[0]
        rows = layout.rows_per_rank(num_tokens)
        pages, offsets = step_pages(layout, seq_lens, qsl, dist[2])
        page_ids = gather_page_ids(layout, bt, pages, offsets, cache.shape[0])
        natural = gather_index_k(layout, cache, page_ids, jnp.sum(pages), axis_name)
        rr = rank_rows(layout, rank, num_tokens, seq_lens, qsl, dist[2])
        start = layout.split_rows + rank * rows
        pad = layout.split_rows
        q_r = lax.dynamic_slice_in_dim(
            jnp.pad(q, ((0, pad), (0, 0), (0, 0))), start, rows, axis=0
        )
        w_r = lax.dynamic_slice_in_dim(
            jnp.pad(weights, ((0, pad), (0, 0))), start, rows, axis=0
        )
        vpages_per_seq = bt.shape[0] // seq_lens.shape[0]
        table = natural_page_table(layout, rr, offsets, vpages_per_seq)
        positions = select_topk(q_r, w_r, natural, rr, table)
        flat = flat_rows(layout, positions, offsets[row_requests(rr, rows)])
        real = jnp.arange(rows, dtype=jnp.int32) < rr.num_rows
        return jnp.where(real[:, None], flat, -1)

    extra = () if rank_ids is None else (rank_ids,)
    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(sharded, rep, rep, rep, rep, rep, rep) + (sharded,) * len(extra),
        out_specs=sharded,
        check_vma=False,
    )(
        index_cache,
        q,
        weights,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        *extra,
    )

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
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.mla import shift_quant_split_3d

FP8_MAX = 448.0


def _search(keys: jax.Array, values: jax.Array, side: str) -> jax.Array:
    """`searchsorted` over a few keys (requests, caps): one broadcast compare,
    not the default binary-search while loop."""
    return jnp.searchsorted(keys, values, side=side, method="compare_all")


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
    # Alternating forward/reverse bands per rank across a step (1 = contiguous;
    # even >= 2 enables zero-ppermute dual-zigzag load balancing when band size
    # holds `split_rows`).
    zigzag_bands: int = 1

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
        if self.zigzag_bands < 1 or (
            self.zigzag_bands > 1 and self.zigzag_bands % 2 != 0
        ):
            raise ValueError(
                f"zigzag_bands must be 1 or a positive even int, got {self}"
            )
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

    def effective_zigzag_bands(self, num_tokens: int) -> int:
        nb = self.zigzag_bands
        if (
            nb > 1
            and self.split_rows > 0
            and num_tokens % (self.dcp_size * nb) == 0
            and (num_tokens // (self.dcp_size * nb)) >= self.split_rows
        ):
            return nb
        return 1


def zigzag_band_block_starts(
    rank: jax.Array | int, dcp_size: int, rows_local: int, num_bands: int
) -> jax.Array:
    """Return `i32[num_bands]` starting global token index for each zigzag band."""
    if num_bands < 1 or (num_bands > 1 and num_bands % 2 != 0):
        raise ValueError(f"num_bands must be 1 or even >= 2, got {num_bands}")
    if rows_local % num_bands:
        raise ValueError(
            f"rows_local={rows_local} must divide evenly by num_bands={num_bands}"
        )
    r = jnp.asarray(rank, dtype=jnp.int32).reshape(())
    band_size = rows_local // num_bands
    ks = jnp.arange(num_bands, dtype=jnp.int32)
    even = (ks & 1) == 0
    blk = jnp.where(even, ks * dcp_size + r, (ks + 1) * dcp_size - 1 - r)
    return (blk * band_size).astype(jnp.int32)


def zigzag_rank_global_tokens(
    rank: jax.Array | int, rows_local: int, dcp_size: int, num_bands: int
) -> jax.Array:
    """Return `i32[rows_local]` global step token indices owned by `rank`."""
    r = jnp.asarray(rank, dtype=jnp.int32).reshape(())
    if num_bands <= 1:
        return r * jnp.int32(rows_local) + jnp.arange(rows_local, dtype=jnp.int32)
    starts = zigzag_band_block_starts(r, dcp_size, rows_local, num_bands)
    band_size = rows_local // num_bands
    return (starts[:, None] + jnp.arange(band_size, dtype=jnp.int32)[None, :]).reshape(
        rows_local
    )


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
    req = jnp.minimum(_search(ends, k, "right"), num_reqs - 1)
    vpage = jnp.clip(k - offsets[req], 0, table.shape[1] - 1)
    page = jnp.mod(table[req, vpage], jnp.int32(num_local_pages))
    return jnp.where(k < ends[-1], page, 0).astype(jnp.int32)


def _align_zigzag_r0_select_inputs(
    layout: TokenParallelLayout,
    num_tokens: int,
    rank: jax.Array,
    q_r: jax.Array,
    w_r: jax.Array,
    rr: RankRows,
) -> tuple[jax.Array, jax.Array, RankRows, bool]:
    nb = layout.effective_zigzag_bands(num_tokens)
    s = layout.split_rows
    if nb <= 1 or s <= 0 or q_r.shape[0] <= s:
        return q_r, w_r, rr, False
    is_r0 = rank.reshape(()) == 0
    # Avoid leaving rank 0 zigzag bands at [0, B-s, 2B-s, ...]: dropping
    # split_rows shifts band edges off the 128-row StreamIdxTC chunk grid and
    # makes rank 0 double-scan every cross-band chunk (~+37.5% KV scoring).
    q_s = jnp.concatenate([jnp.zeros_like(q_r[:s]), q_r[:-s]], axis=0)
    w_s = jnp.concatenate([jnp.zeros_like(w_r[:s]), w_r[:-s]], axis=0)
    q_sel = jnp.where(is_r0, q_s, q_r)
    w_sel = jnp.where(is_r0, w_s, w_r)
    cu_sel = jnp.where(is_r0, rr.cu_q_lens + jnp.int32(s), rr.cu_q_lens)
    nr_sel = jnp.where(is_r0, rr.num_rows + jnp.int32(s), rr.num_rows)
    rr_sel = RankRows(
        rr.seq_lens,
        cu_sel,
        rr.num_seqs,
        rr.first_req,
        nr_sel,
        rr.req_ids,
    )
    return q_sel, w_sel, rr_sel, True


def _unalign_zigzag_r0_positions(
    positions: jax.Array,
    rank: jax.Array,
    split_rows: int,
    shifted: bool,
) -> jax.Array:
    if not shifted:
        return positions
    is_r0 = rank.reshape(()) == 0
    pos_r0 = jnp.concatenate(
        [positions[split_rows:], jnp.full_like(positions[:split_rows], -1)], axis=0
    )
    return jnp.where(is_r0, pos_r0, positions)


def flat_rows(
    layout: TokenParallelLayout,
    positions: jax.Array,
    row_offsets: jax.Array,
    *,
    stride_pages: int | None = None,
) -> jax.Array:
    """Global positions `[N, K]` (`-1` = none) of the requests whose buffer
    offsets are `row_offsets [N]` -> flat rows of the gathered latent buffer."""
    stride = layout.max_pages if stride_pages is None else stride_pages
    dcp, il, ps = layout.dcp_size, layout.interleave_size, layout.page_size
    if (
        (dcp & (dcp - 1)) == 0
        and (il & (il - 1)) == 0
        and (ps & (ps - 1)) == 0
        and il <= ps
    ):
        # Avoid 5 signed integer div/mod ops on [1024, 2048] (~45.8 us/L):
        # power-of-2 logical shifts and masks produce identical results in ~30.9 us/L.
        il_sh = jnp.int32(il.bit_length() - 1)
        dcp_sh = jnp.int32(dcp.bit_length() - 1)
        ps_sh = jnp.int32(ps.bit_length() - 1)
        vps_sh = jnp.int32((dcp * ps).bit_length() - 1)
        p_nonneg = jnp.maximum(positions, jnp.int32(0))
        q_il = lax.shift_right_logical(p_nonneg, il_sh)
        owner = q_il & jnp.int32(dcp - 1)
        local_slot_all = (lax.shift_right_logical(q_il, dcp_sh) << il_sh) | (
            p_nonneg & jnp.int32(il - 1)
        )
        slot = local_slot_all & jnp.int32(ps - 1)
        page = (
            owner * jnp.int32(stride)
            + row_offsets[:, None]
            + lax.shift_right_logical(p_nonneg, vps_sh)
        )
        return jnp.where(positions >= 0, (page << ps_sh) | slot, jnp.int32(-1))
    owner, slot = owner_and_slot(layout, positions)
    page = owner * stride + row_offsets[:, None] + positions // layout.virtual_page_size
    return jnp.where(positions >= 0, page * layout.page_size + slot, -1)


class RankRows(NamedTuple):
    """This rank's rows as a ragged batch of its own. Local request `k` is
    global request `first_req + k` (or `req_ids[k]` under dual-zigzag);
    requests past `num_seqs` are empty."""

    seq_lens: jax.Array  # i32[R], truncated after this rank's last row
    cu_q_lens: jax.Array  # i32[R + 1], local rows
    num_seqs: jax.Array  # i32
    first_req: jax.Array  # i32
    num_rows: jax.Array  # i32, real rows; the rest of the R rows are padding
    req_ids: jax.Array | None = None  # i32[R], exact global req per span (zigzag)


def _rank_rows_dual_zigzag(
    layout: TokenParallelLayout,
    rank: jax.Array,
    num_tokens: int,
    num_bands: int,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    num_seqs: jax.Array,
) -> RankRows:
    max_reqs = seq_lens.shape[0]
    rows = layout.rows_per_rank(num_tokens)
    band_size = rows // num_bands
    starts = zigzag_band_block_starts(rank, layout.dcp_size, rows, num_bands)
    total_valid = query_start_loc[num_seqs]
    q_start, q_end = query_start_loc[:-1], query_start_loc[1:]
    req_idx = jnp.arange(max_reqs, dtype=jnp.int32)
    active = req_idx < num_seqs

    r = rank.reshape(())
    is_r0 = r == 0
    band_lo = jnp.where(
        (jnp.arange(num_bands, dtype=jnp.int32) == 0) & is_r0,
        jnp.int32(layout.split_rows),
        starts,
    )
    band_hi = starts + band_size
    lo = band_lo[:, None]
    hi = jnp.clip(total_valid, lo, band_hi[:, None])
    a = jnp.clip(q_start[None, :], lo, hi)
    b = jnp.clip(q_end[None, :], lo, hi)
    q_len = jnp.where(active[None, :] & (b > a), b - a, 0)

    # Proof: at most one request crosses each band gap (last_j <= first_{j+1}),
    # so the total non-empty (band, req) spans <= max_reqs + num_bands - 1.
    out_cap = max_reqs + num_bands
    flat_qlen = q_len.reshape(-1)
    flat_valid = flat_qlen > 0
    flat_req = jnp.broadcast_to(req_idx[None, :], (num_bands, max_reqs)).reshape(-1)
    flat_seq = (seq_lens[None, :] - (q_end[None, :] - b)).reshape(-1)

    pos = jnp.cumsum(flat_valid.astype(jnp.int32)) - 1
    dst = jnp.where(flat_valid, pos, out_cap)
    out_qlen = jnp.zeros((out_cap,), jnp.int32).at[dst].set(flat_qlen, mode="drop")
    out_seq = jnp.zeros((out_cap,), jnp.int32).at[dst].set(flat_seq, mode="drop")
    out_req = jnp.zeros((out_cap,), jnp.int32).at[dst].set(flat_req, mode="drop")
    n_local = jnp.sum(flat_valid.astype(jnp.int32))
    cu = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(out_qlen)])
    return RankRows(
        out_seq.astype(jnp.int32),
        cu.astype(jnp.int32),
        n_local,
        out_req[0],
        cu[n_local].astype(jnp.int32),
        out_req.astype(jnp.int32),
    )


def rank_rows(
    layout: TokenParallelLayout,
    rank: jax.Array,
    num_tokens: int,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    num_seqs: jax.Array,
) -> RankRows:
    """Build per-rank sequence metadata for own rows or zigzag bands.

    Covers rows `split_rows + rank * R + [0, R)` (or `num_bands` alternating
    zigzag bands when configured). Each row keeps its global position and
    causal bound: a request's `seq_len` is cut by the rows that follow this
    rank's range, so the last local row of the request sits at `seq_len' - 1`.
    """
    nb = layout.effective_zigzag_bands(num_tokens)
    if nb > 1:
        return _rank_rows_dual_zigzag(
            layout, rank, num_tokens, nb, seq_lens, query_start_loc, num_seqs
        )
    num_reqs = seq_lens.shape[0]
    rows = layout.rows_per_rank(num_tokens)
    total = query_start_loc[num_seqs]
    lo = layout.split_rows + rank * rows
    hi = jnp.clip(total, lo, lo + rows)
    q_start, q_end = query_start_loc[:-1], query_start_loc[1:]
    req = jnp.arange(num_reqs, dtype=jnp.int32)
    active = req < num_seqs
    a, b = jnp.clip(q_start, lo, hi), jnp.clip(q_end, lo, hi)
    first = _search(q_end, lo, "right").astype(jnp.int32)
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
    local = _search(rr.cu_q_lens[1:], i, "right").astype(jnp.int32)
    local = jnp.minimum(local, jnp.maximum(rr.num_seqs - 1, 0))
    if rr.req_ids is not None:
        return rr.req_ids[local]
    return jnp.minimum(rr.first_req + local, rr.seq_lens.shape[0] - 1)


def natural_page_table(
    layout: TokenParallelLayout,
    rr: RankRows,
    offsets: jax.Array,
    vpages_per_seq: int,
    cap: int | None = None,
) -> jax.Array:
    """Build `i32[R * min(vpages_per_seq, cap) * dcp]` natural-order page indices.

    Clamps padding entries to `cap * dcp - 1` when `cap` is set so every page
    index stays in bounds of a `cap * dcp`-page gathered buffer.
    """
    bound_pages = layout.max_pages if cap is None else cap
    eff_vpages_per_seq = min(vpages_per_seq, bound_pages)
    if rr.req_ids is not None:
        req = rr.req_ids
    else:
        num_reqs = rr.seq_lens.shape[0]
        k = jnp.arange(num_reqs, dtype=jnp.int32)
        req = jnp.minimum(rr.first_req + k, num_reqs - 1)
    width = eff_vpages_per_seq * layout.dcp_size
    pages = offsets[req][:, None] * layout.dcp_size + jnp.arange(width)[None, :]
    pages = jnp.minimum(pages, bound_pages * layout.dcp_size - 1)
    return pages.reshape(-1).astype(jnp.int32)


def _sized_gather(
    layout: TokenParallelLayout,
    total_pages: jax.Array,
    gather: Callable[[int], jax.Array],
    page_axis: int,
    page_rows: int = 1,
) -> jax.Array:
    """`gather(cap)` for the smallest cap in `gather_caps` holding
    `total_pages`. Every rank takes the same branch (the metadata is
    replicated); `page_axis` (`cap * page_rows` long) is zero-padded to
    `max_pages * page_rows` (no flat row reaches the padding)."""

    def branch(cap: int) -> Callable[[], jax.Array]:
        def run() -> jax.Array:
            g = gather(cap)
            pad = [(0, 0)] * g.ndim
            pad[page_axis] = (0, (layout.max_pages - cap) * page_rows)
            return jnp.pad(g, pad)

        return run

    return _switch_on_cap(layout, total_pages, branch)


def _switch_on_cap(
    layout: TokenParallelLayout,
    total_pages: jax.Array,
    branch: Callable[[int], Callable[[], Any]],
) -> Any:
    """`branch(cap)()` for the smallest cap in `gather_caps` holding
    `total_pages`; every rank takes the same branch (replicated metadata)."""
    caps = layout.gather_caps
    if len(caps) == 1:
        return branch(caps[0])()
    idx = _search(jnp.asarray(caps, jnp.int32), total_pages, "left")
    return lax.switch(jnp.minimum(idx, len(caps) - 1), [branch(c) for c in caps])


def take_pages(cache: jax.Array, page_ids: jax.Array) -> jax.Array:
    """`cache[page_ids]`, one flat row per page, by XLA."""
    return cache[page_ids].reshape(page_ids.shape[0], -1)


# (cache [pages, ...], page ids [n]) -> the n pages' bytes in page order
ReadPagesFn = Callable[[jax.Array, jax.Array], jax.Array]


def gather_latent(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    page_ids: jax.Array,
    axis_name: str,
    read_pages: tuple[ReadPagesFn, ReadPagesFn] = (take_pages, take_pages),
) -> tuple[jax.Array, jax.Array]:
    """Every rank's `page_ids` pages, rank-major: `[dcp * len(page_ids), ...]`.
    `read_pages` reads this rank's (nope, rope) pages (`page_gather` on TPU:
    it reads each cache in the layout the SparseCore cache kernels use). The
    pages move flat, so a linear-layout read stays linear up to the kernel."""

    local_nope = read_pages[0](nope_cache, page_ids)
    local_rope = read_pages[1](rope_cache, page_ids)
    if (
        local_nope.dtype == local_rope.dtype
        and local_nope.ndim == 3
        and local_rope.ndim == 3
        and local_nope.shape[0] == local_rope.shape[0]
        and local_nope.shape[2] == local_rope.shape[2]
        and local_nope.shape[1] % 8 == 0
        and local_rope.shape[1] % 8 == 0
    ):
        n_rows = local_nope.shape[1]
        both = jnp.concatenate([local_nope, local_rope], axis=1)
        g_both = lax.all_gather(both, axis_name, axis=0, tiled=True)
        return (
            g_both[:, :n_rows, :].reshape(-1, *nope_cache.shape[1:]),
            g_both[:, n_rows:, :].reshape(-1, *rope_cache.shape[1:]),
        )

    def gather(local: jax.Array, tail_shape: tuple[int, ...]) -> jax.Array:
        return lax.all_gather(local.reshape(-1), axis_name, tiled=True).reshape(
            -1, *tail_shape
        )

    return (
        gather(local_nope, nope_cache.shape[1:]),
        gather(local_rope, rope_cache.shape[1:]),
    )


def restride_flat_rows(
    layout: TokenParallelLayout, flat: jax.Array, cap: int
) -> jax.Array:
    """`flat_rows` of a `max_pages`-per-rank buffer -> the same rows of a
    buffer gathered with `cap` pages per rank (every row a step reaches lies
    in the first `cap` pages of its rank)."""
    rank_rows_max = layout.max_pages * layout.page_size
    owner, rest = flat // rank_rows_max, flat % rank_rows_max
    return jnp.where(flat >= 0, owner * (cap * layout.page_size) + rest, -1)


# Avoid axis=0 all-gather below the 8 MiB SparseCore offload floor (TC
# all-gather on axis=3, tiled=True drops the post-AG transpose for cap<=12),
# and avoid axis=3 above 8 MiB (SparseCore all-gather on axis=0 avoids
# inner-axis scatter overhead for cap>=16).
_PACKED_INDEX_K_AXIS0_AG_MIN_BYTES = 8 << 20


def _gather_index_k_pages(
    cache: jax.Array,
    page_ids: jax.Array,
    cap: int,
    dcp: int,
    il: int,
    ps: int,
    axis_name: str,
    *,
    packed_u32: bool = False,
    axis0_ag_min_bytes: int = _PACKED_INDEX_K_AXIS0_AG_MIN_BYTES,
) -> jax.Array:
    local = cache[page_ids[:cap]]
    rows = local.shape[1]
    if packed_u32:
        u32_local = lax.bitcast_convert_type(jnp.swapaxes(local, 2, 3), jnp.uint32)
        if ps % (dcp * il) == 0:
            blocks = ps // (dcp * il)
            u32_blk = u32_local.reshape(cap, rows, dcp, blocks, il)
            if cap * dcp * rows * ps * 4 >= axis0_ag_min_bytes:
                packed = jnp.transpose(u32_blk, (0, 2, 1, 3, 4)).reshape(
                    cap * dcp, rows, blocks, il
                )
                g = lax.all_gather(packed, axis_name, axis=0, tiled=False)
                return jnp.transpose(g, (1, 2, 3, 0, 4)).reshape(cap * dcp, rows, ps)
            packed = jnp.transpose(u32_blk, (0, 2, 1, 3, 4)).reshape(
                cap * dcp, rows, blocks, 1, il
            )
            g = lax.all_gather(packed, axis_name, axis=3, tiled=True)
            return g.reshape(cap * dcp, rows, ps)
        g = lax.all_gather(
            u32_local.reshape(cap, rows, ps // il, il), axis_name, axis=3
        )
        return g.reshape(cap, rows, dcp, ps).swapaxes(1, 2).reshape(cap * dcp, rows, ps)
    if il == 4 and ps % (dcp * il) == 0:
        # Avoid gathering along inner axis=3 on u32[cap, rows, ps // il, il]:
        # XLA emits two full-buffer post-AG layout copies plus an unpack fusion
        # (~82 us/L at 20 pages); pre-packing q into dim 0 and gathering on
        # axis=0 keeps collective tiles aligned.
        blocks = ps // (dcp * il)
        words = lax.bitcast_convert_type(
            local.reshape(cap, rows, 4, dcp, blocks, il), jnp.uint32
        )
        packed = jnp.transpose(words, (0, 3, 1, 2, 4)).reshape(
            cap * dcp, rows, 4, blocks
        )
        gathered = lax.all_gather(packed, axis_name, axis=0, tiled=False)
        gathered = jnp.transpose(gathered, (1, 2, 3, 4, 0))
        return lax.bitcast_convert_type(
            gathered.reshape(cap * dcp, rows, 4, ps // il), jnp.uint8
        ).reshape(cap * dcp, rows, 4, ps)
    words = lax.bitcast_convert_type(local.swapaxes(2, 3), jnp.uint32)
    g = lax.all_gather(words.reshape(cap, rows, ps // il, il), axis_name, axis=3)
    g = g.reshape(cap, rows, dcp, ps).swapaxes(1, 2)
    return (
        lax.bitcast_convert_type(g, jnp.uint8)
        .swapaxes(3, 4)
        .reshape(cap * dcp, rows, 4, ps)
    )


def gather_index_k(
    layout: TokenParallelLayout,
    cache: jax.Array,
    page_ids: jax.Array,
    total_pages: jax.Array,
    axis_name: str,
) -> jax.Array:
    """Every rank's `page_ids` pages of a token-along-lane index-K cache
    (u8 `[pages, rows, 4, page_size]`, slot on the last axis), in natural
    position order: natural page `offsets[r] * dcp + q` of request `r` holds
    its positions `[q * page_size, (q + 1) * page_size)`.

    Slot `c * il + i` of rank `r` is in-page position `c * dcp * il + r * il +
    i`, so gathering each rank's slots `(c, i)` along a new rank axis between
    `c` and `i` lays the virtual page out in position order. The four bytes of
    a slot's sublane group move as one u32 word.
    """
    dcp, il, ps = layout.dcp_size, layout.interleave_size, layout.page_size
    if cache.ndim != 4 or cache.shape[2:] != (4, ps) or cache.dtype != jnp.uint8:
        raise ValueError(
            f"index-K cache {cache.shape} {cache.dtype} is not token-along-lane "
            f"u8 with page {ps}"
        )

    def gather(cap: int) -> jax.Array:
        return _gather_index_k_pages(
            cache, page_ids, cap, dcp, il, ps, axis_name
        ).reshape(cap, dcp, *cache.shape[1:])

    g = _sized_gather(layout, total_pages, gather, page_axis=0)
    return g.reshape(layout.max_pages * dcp, *g.shape[2:])


def own_rows(layout: TokenParallelLayout, x: jax.Array, rank: jax.Array) -> jax.Array:
    """This rank's token-parallel rows of a replicated `[T, ...]` operand, read
    by row index (no copy of `x`); rows past `T` repeat row `T - 1` and are
    padding."""
    num_tokens = x.shape[0]
    rows = layout.rows_per_rank(num_tokens)
    nb = layout.effective_zigzag_bands(num_tokens)
    if nb > 1:
        r = rank.reshape(())
        s = layout.split_rows
        own_idx = zigzag_rank_global_tokens(r, rows, layout.dcp_size, nb)
        r0_idx = jnp.concatenate(
            [own_idx[s:], jnp.full((s,), num_tokens - 1, dtype=jnp.int32)]
        )
        idx = jnp.where(r == 0, r0_idx, own_idx)
        return jnp.take(x, idx, axis=0)
    idx = layout.split_rows + rank * rows + jnp.arange(rows, dtype=jnp.int32)
    return jnp.take(x, jnp.minimum(idx, num_tokens - 1), axis=0)


def _zigzag_tail_rank_order_indices(
    dcp_size: int, num_tokens: int, split_rows: int, num_bands: int
) -> tuple[list[int], list[int]]:
    """Return forward (rank-major -> global) and inverse tail-row index maps."""
    rows = num_tokens // dcp_size
    band_size = rows // num_bands
    fwd: list[int] = []
    inv = [rows - 1] * num_tokens
    for r in range(dcp_size):
        tokens_r: list[int] = []
        for k in range(num_bands):
            blk = k * dcp_size + r if (k & 1) == 0 else (k + 1) * dcp_size - 1 - r
            start = blk * band_size
            tokens_r.extend(range(start, start + band_size))
        if r == 0:
            shifted = tokens_r[split_rows:] + [num_tokens - 1] * split_rows
            for loc, g in enumerate(tokens_r[split_rows:]):
                inv[g - split_rows] = r * rows + loc
            fwd.extend(shifted)
        else:
            for loc, g in enumerate(tokens_r):
                inv[g - split_rows] = r * rows + loc
            fwd.extend(tokens_r)
    return fwd, inv


def heads_to_tokens(
    layout: TokenParallelLayout, x: jax.Array, axis_name: str
) -> jax.Array:
    """`[T, h, d]` this rank's heads of every row -> `[R, h * dcp, d]` every
    head of this rank's token-parallel rows (heads in rank order)."""
    num_tokens = x.shape[0]
    s = layout.split_rows
    nb = layout.effective_zigzag_bands(num_tokens)
    if nb > 1:
        fwd, _ = _zigzag_tail_rank_order_indices(layout.dcp_size, num_tokens, s, nb)
        x_pad = jnp.take(x, jnp.asarray(fwd, dtype=jnp.int32), axis=0)
        rows = layout.rows_per_rank(num_tokens)
        zero_tail = jnp.zeros((s, *x.shape[1:]), x.dtype)
        x_pad = x_pad.at[rows - s : rows].set(zero_tail)
        return lax.all_to_all(x_pad, axis_name, 0, 1, tiled=True)
    x = jnp.concatenate([x[s:], jnp.zeros((s, *x.shape[1:]), x.dtype)])
    return lax.all_to_all(x, axis_name, 0, 1, tiled=True)


def tokens_to_heads(
    y: jax.Array,
    axis_name: str,
    *,
    layout: TokenParallelLayout | None = None,
) -> jax.Array:
    """Inverse of `heads_to_tokens`: `[R, H, d]` -> `[T, H / dcp, d]`, row `i`
    being step row `split_rows + i` (the last `split_rows` rows are padding)."""
    gathered = lax.all_to_all(y, axis_name, 1, 0, tiled=True)
    if layout is not None:
        num_tokens = gathered.shape[0]
        nb = layout.effective_zigzag_bands(num_tokens)
        if nb > 1:
            s = layout.split_rows
            _, inv = _zigzag_tail_rank_order_indices(layout.dcp_size, num_tokens, s, nb)
            unperm = jnp.take(gathered, jnp.asarray(inv, dtype=jnp.int32), axis=0)
            return unperm.at[num_tokens - s :].set(0)
    return gathered


def dequantize_row_shards(
    w_shards: jax.Array, s_shards: jax.Array, dtype: jnp.dtype
) -> jax.Array:
    """`[G * k, n]` FP8 row shards x `[G, n]` scales -> `[G * k, n]` in `dtype`.

    Shard `g` (rows `[g * k, (g + 1) * k)`) is scaled by its own `s_shards[g]`."""
    groups, n = s_shards.shape
    w = w_shards.reshape(groups, -1, n).astype(dtype) * s_shards[:, None, :].astype(
        dtype
    )
    return w.reshape(-1, n)


def gather_qc_head_rows(
    q_c_local: jax.Array,
    rank: jax.Array,
    *,
    dcp_size: int,
    split_rows: int,
    axis_name: str,
    zigzag_local: bool = False,
) -> jax.Array:
    """Assemble `[split_rows + R, Q_LORA]` head0 broadcast plus per-rank tail rows."""
    r = rank.reshape(())
    head_local = q_c_local[:split_rows]
    # Avoid an all-gather of every rank's head slice: only rank 0's head is
    # broadcast, and a bf16 psum with (dcp_size - 1) exact zeros is bit-exact
    # for finite values while sending 256 KB instead of 4.19 MB at dcp=16.
    head0 = lax.psum(
        jnp.where(r == 0, head_local, jnp.zeros_like(head_local)), axis_name
    )
    if zigzag_local:
        # Dual-zigzag band 0 contains all of [0, split_rows) on rank 0, so
        # nonzero ranks attend all R own rows directly with zero ppermute.
        pad = jnp.broadcast_to(q_c_local[-1:], (split_rows, q_c_local.shape[-1]))
        r0_tail = jnp.concatenate([q_c_local[split_rows:], pad], axis=0)
        tail = jnp.where(r == 0, r0_tail, q_c_local)
        return jnp.concatenate([head0, tail], axis=0)
    perm = [(i, (i - 1) % dcp_size) for i in range(dcp_size)]
    from_next = lax.ppermute(head_local, axis_name, perm)
    tail_s = jnp.where(
        r < dcp_size - 1,
        from_next,
        jnp.broadcast_to(q_c_local[-1:], (split_rows, q_c_local.shape[-1])),
    )
    return jnp.concatenate([head0, q_c_local[split_rows:], tail_s], axis=0)


def token_rows_to_sequence_rows(
    layout: TokenParallelLayout,
    own_rows: jax.Array,
    lead_rows: jax.Array,
    rank: jax.Array,
    axis_name: str,
) -> jax.Array:
    """Token-parallel rows `[R, N]` -> this rank's sequence-parallel rows `[R, N]`.

    Per rank, inside a `shard_map` over `axis_name`: for contiguous layout,
    `own_rows` are step rows `split_rows + rank * R + i`, handed to rank+1 via
    `ppermute`; for dual-zigzag (`zigzag_bands > 1`), only rank 0 swaps in
    `lead_rows` at `[0, split_rows)` with zero cross-rank transfer."""
    s = layout.split_rows
    rows = own_rows.shape[0]
    if lead_rows.shape[0] != s or s > rows:
        raise ValueError(
            f"lead rows {lead_rows.shape[0]} must equal split_rows={s} <= {rows}"
        )
    if s == 0:
        return own_rows
    if layout.effective_zigzag_bands(rows * layout.dcp_size) > 1:
        r0_rows = jnp.concatenate(
            [lead_rows.astype(own_rows.dtype), own_rows[: rows - s]], axis=0
        )
        return jnp.where(rank == 0, r0_rows, own_rows)
    perm = [(i, i + 1) for i in range(layout.dcp_size - 1)]
    carried = lax.ppermute(own_rows[rows - s :], axis_name, perm)
    lead = jnp.where(rank == 0, lead_rows.astype(own_rows.dtype), carried)
    return jnp.concatenate([lead, own_rows[: rows - s]], axis=0)


def project_all_heads(
    rows: jax.Array,
    w_local: jax.Array,
    s_local: jax.Array,
    rank: jax.Array,
    axis_name: str,
    axis_size: int,
) -> jax.Array:
    """Per rank: `rows` `[M, G * k]` (every head) -> `rows @ o_proj` `[M, n]`.

    The rank holds part of o_proj in one of three layouts, told apart by the
    shapes:
      * its own row shard, `w_local` `[K / G, n]`, `s_local` `[n]` (each shard
        quantized per channel on its own): every shard is all-gathered over
        `axis_name`;
      * a pair half of row shards, `w_local` `[K / 2, n]`, `s_local`
        `[G / 2, n]`: ranks `2j` and `2j + 1` (the two cores of one chip) hold
        shards `[0, G / 2)` and `[G / 2, G)` and swap them with one
        `ppermute`;
      * a pair half of a whole-K per-channel FP8 o_proj, `w_local` `[K / 2,
        n]`, `s_local` `[n]` (one scale per channel over all of K): the halves
        swap as above, and both halves contract FP8 x FP8 against the rows'
        per-row FP8 quantization (`quantize_rows`) under that one scale.
    """
    if s_local.ndim == 1 and axis_size > 2 and 2 * w_local.shape[0] == rows.shape[1]:
        return _project_fp8_pair_halves(
            rows, w_local, s_local, rank, axis_name, axis_size
        )
    if s_local.ndim == 1:
        w_all = dequantize_row_shards(
            lax.all_gather(w_local, axis_name, axis=0, tiled=True),
            lax.all_gather(s_local, axis_name, axis=0, tiled=False),
            rows.dtype,
        )
        return jnp.dot(rows, w_all, preferred_element_type=jnp.float32).astype(
            rows.dtype
        )
    if axis_size % 2:
        raise ValueError(f"pair halves need an even axis, got {axis_size}")
    pairs = [(i, i ^ 1) for i in range(axis_size)]
    w_peer = lax.ppermute(w_local, axis_name, pairs)
    s_peer = lax.ppermute(s_local, axis_name, pairs)
    half = w_local.shape[0]
    parity = lax.rem(rank, 2)
    mine = lax.dynamic_slice_in_dim(rows, parity * half, half, axis=1)
    peer = lax.dynamic_slice_in_dim(rows, (1 - parity) * half, half, axis=1)
    y = jnp.dot(
        mine,
        dequantize_row_shards(w_local, s_local, rows.dtype),
        preferred_element_type=jnp.float32,
    ) + jnp.dot(
        peer,
        dequantize_row_shards(w_peer, s_peer, rows.dtype),
        preferred_element_type=jnp.float32,
    )
    return y.astype(rows.dtype)


def quantize_rows(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`[M, K]` -> (`[M, K]` FP8 e4m3, `[M, 1]` f32 scale), one scale per row."""
    x32 = x.astype(jnp.float32)
    scale = jnp.max(jnp.abs(x32), axis=-1, keepdims=True) * (1.0 / FP8_MAX)
    scale = jnp.where(scale == 0.0, 1.0, scale)
    x_q = jnp.clip(x32 * (1.0 / scale), -FP8_MAX, FP8_MAX).astype(jnp.float8_e4m3fn)
    return x_q, scale


def _project_fp8_pair_halves(
    rows: jax.Array,
    w_local: jax.Array,
    s_local: jax.Array,
    rank: jax.Array,
    axis_name: str,
    axis_size: int,
) -> jax.Array:
    if axis_size % 2:
        raise ValueError(f"pair halves need an even axis, got {axis_size}")
    pairs = [(i, i ^ 1) for i in range(axis_size)]
    w_peer = lax.ppermute(w_local, axis_name, pairs)
    half = w_local.shape[0]
    parity = lax.rem(rank, 2)
    x_q, x_scale = quantize_rows(rows)
    mine = lax.dynamic_slice_in_dim(x_q, parity * half, half, axis=1)
    peer = lax.dynamic_slice_in_dim(x_q, (1 - parity) * half, half, axis=1)
    dims = (((1,), (0,)), ((), ()))
    y = lax.dot_general(
        mine, w_local, dims, preferred_element_type=jnp.float32
    ) + lax.dot_general(peer, w_peer, dims, preferred_element_type=jnp.float32)
    return (y * x_scale * s_local.astype(jnp.float32)[None, :]).astype(rows.dtype)


def _merge_lead_partial_latent(
    lead_partial: jax.Array,
    lead_lse: jax.Array,
    axis_name: str,
) -> jax.Array:
    lses = lax.all_gather(lead_lse, axis_name, axis=0, tiled=False)
    m = jax.nn.logsumexp(lses, axis=0)
    safe_m = jnp.where(jnp.isfinite(m), m, 0.0)
    w = jnp.where(jnp.isfinite(lead_lse), jnp.exp(lead_lse - safe_m), 0.0)
    # Avoid casting to bf16 before psum: bf16 cross-rank reduction accumulates
    # cancellation error and varies with XLA all-reduce lowering order.
    weighted = lead_partial.astype(jnp.float32) * w[..., None]
    return lax.psum(weighted, axis_name).astype(lead_partial.dtype)


def _merge_lead_partial_to_v(
    lead_partial: jax.Array,
    lead_lse: jax.Array,
    w_uv_all: jax.Array,
    axis_name: str,
    out_dtype: jnp.dtype,
) -> jax.Array:
    lead_latent = _merge_lead_partial_latent(lead_partial, lead_lse, axis_name)
    return jnp.einsum(
        "rhl,hlv->rhv", lead_latent, w_uv_all.astype(lead_latent.dtype)
    ).astype(out_dtype)


def _merge_lead_partial_to_v_hrv(
    lead_partial: jax.Array,
    lead_lse: jax.Array,
    w_uv_all: jax.Array,
    axis_name: str,
    out_dtype: jnp.dtype,
) -> jax.Array:
    lead_latent = _merge_lead_partial_latent(lead_partial, lead_lse, axis_name)
    return jnp.einsum(
        "hrl,hlv->hrv",
        jnp.swapaxes(lead_latent, 0, 1),
        w_uv_all.astype(lead_latent.dtype),
    ).astype(out_dtype)


def _project_w_uv(out: jax.Array, w_uv_all: jax.Array) -> jax.Array:
    return (
        jnp.einsum(
            "hrl,hlv->hrv",
            jnp.swapaxes(out, 0, 1),
            w_uv_all.astype(out.dtype),
        )
        .swapaxes(0, 1)
        .astype(out.dtype)
    )


def _project_w_uv_hrv(out: jax.Array, w_uv_all: jax.Array) -> jax.Array:
    return jnp.einsum(
        "hrl,hlv->hrv",
        jnp.swapaxes(out, 0, 1),
        w_uv_all.astype(out.dtype),
    ).astype(out.dtype)


def _can_use_fused_wuv_epilogue(
    layout: TokenParallelLayout,
    attend_fn: Callable[..., Any],
    w_uv_all: jax.Array,
    num_rows: int,
    dtype: jnp.dtype,
) -> bool:
    if layout.effective_zigzag_bands(num_rows * layout.dcp_size) <= 1:
        return False
    hm_fn = getattr(attend_fn, "head_major_chunks", None)
    chunk_rows = getattr(attend_fn, "head_major_chunk_rows", None)
    if hm_fn is None or chunk_rows is None:
        return False
    num_heads, lkv_dim, head_dim = (
        int(w_uv_all.shape[0]),
        int(w_uv_all.shape[1]),
        int(w_uv_all.shape[2]),
    )
    return shift_quant_split_3d.supports_wuv_quant_split_3d_local(
        num_heads,
        num_rows,
        int(chunk_rows),
        layout.split_rows,
        lkv_dim,
        head_dim,
        dtype,
    )


def _project_wuv_hm_pair_halves_local(
    layout: TokenParallelLayout,
    own_chunks: tuple[jax.Array, ...],
    lead_hrl: jax.Array,
    w_uv_all: jax.Array,
    rank_vec: jax.Array,
    w_local: jax.Array,
    s_local: jax.Array,
    axis_name: str,
    *,
    num_valid_own_rows: jax.Array,
    use_pallas: bool = True,
) -> jax.Array:
    pairs = [(i, i ^ 1) for i in range(layout.dcp_size)]
    w_peer = lax.ppermute(w_local, axis_name, pairs)
    w_uv_cast = w_uv_all.astype(lead_hrl.dtype)
    if use_pallas:
        mine, peer, x_scale = shift_quant_split_3d.wuv_quant_split_3d_fp8_local(
            lead_hrl,
            own_chunks,
            w_uv_cast,
            rank_vec,
            num_valid_own_rows=num_valid_own_rows,
        )
    else:
        mine, peer, x_scale = shift_quant_split_3d.wuv_quant_split_3d_local_ref(
            lead_hrl,
            own_chunks,
            w_uv_cast,
            rank_vec,
            num_valid_own_rows=num_valid_own_rows,
        )
    dims = (((1,), (0,)), ((), ()))
    y = lax.dot_general(
        mine, w_local, dims, preferred_element_type=jnp.float32
    ) + lax.dot_general(peer, w_peer, dims, preferred_element_type=jnp.float32)
    return (y * x_scale * s_local.astype(jnp.float32)[None, :]).astype(lead_hrl.dtype)


def _can_use_3d_fp8_epilogue(
    layout: TokenParallelLayout,
    num_heads: int,
    num_rows: int,
    head_dim: int,
    dtype: jnp.dtype,
    w_local: jax.Array,
    s_local: jax.Array,
) -> bool:
    return (
        s_local.ndim == 1
        and layout.dcp_size > 2
        and layout.dcp_size % 2 == 0
        and 2 * w_local.shape[0] == num_heads * head_dim
        and shift_quant_split_3d.supports_shift_quant_split_3d(
            num_heads, num_rows, layout.split_rows, head_dim, dtype
        )
    )


def _project_hrv_pair_halves_local(
    layout: TokenParallelLayout,
    own_hrv: jax.Array,
    lead_hrv: jax.Array,
    rank_vec: jax.Array,
    w_local: jax.Array,
    s_local: jax.Array,
    axis_name: str,
    *,
    use_pallas: bool = True,
) -> jax.Array:
    num_rows = own_hrv.shape[1]
    split_rows = layout.split_rows
    pairs = [(i, i ^ 1) for i in range(layout.dcp_size)]
    w_peer = lax.ppermute(w_local, axis_name, pairs)
    if layout.effective_zigzag_bands(num_rows * layout.dcp_size) > 1:
        if use_pallas:
            mine, peer, x_scale = shift_quant_split_3d.shift_quant_split_3d_fp8_local(
                lead_hrv, own_hrv, rank_vec
            )
        else:
            mine, peer, x_scale = shift_quant_split_3d.shift_quant_split_3d_local_ref(
                lead_hrv, own_hrv, rank_vec
            )
    elif use_pallas and shift_quant_split_3d.supports_shift_quant_split_3d_remote(
        own_hrv.shape[0],
        num_rows,
        split_rows,
        own_hrv.shape[2],
        own_hrv.dtype,
        layout.dcp_size,
    ):
        # Avoid a separate bf16[H, S, V] lax.ppermute across all layers: quantize
        # and push the tail slice in FP8 (half the bytes) overlapped with the body.
        mine, peer, x_scale = shift_quant_split_3d.shift_quant_split_3d_fp8_remote(
            lead_hrv,
            own_hrv,
            rank_vec,
            dcp_size=layout.dcp_size,
            axis_name=axis_name,
        )
    else:
        perm_shift = [(i, i + 1) for i in range(layout.dcp_size - 1)]
        carried_hrv = lax.ppermute(
            own_hrv[:, num_rows - split_rows :, :], axis_name, perm_shift
        )
        if use_pallas:
            mine, peer, x_scale = shift_quant_split_3d.shift_quant_split_3d_fp8(
                lead_hrv, carried_hrv, own_hrv, rank_vec
            )
        else:
            mine, peer, x_scale = shift_quant_split_3d.shift_quant_split_3d_ref(
                lead_hrv, carried_hrv, own_hrv, rank_vec
            )
    dims = (((1,), (0,)), ((), ()))
    y = lax.dot_general(
        mine, w_local, dims, preferred_element_type=jnp.float32
    ) + lax.dot_general(peer, w_peer, dims, preferred_element_type=jnp.float32)
    return (y * x_scale * s_local.astype(jnp.float32)[None, :]).astype(own_hrv.dtype)


def _shift_and_project_local(
    layout: TokenParallelLayout,
    own_v: jax.Array,
    lead_v: jax.Array,
    rank: jax.Array,
    w_local: jax.Array,
    s_local: jax.Array,
    axis_name: str,
) -> jax.Array:
    seq_rows = token_rows_to_sequence_rows(
        layout,
        own_v.reshape(own_v.shape[0], -1),
        lead_v.reshape(lead_v.shape[0], -1),
        rank,
        axis_name,
    )
    return project_all_heads(
        seq_rows, w_local, s_local, rank, axis_name, layout.dcp_size
    )


def project_sequence_rows_from_v(
    layout: TokenParallelLayout,
    mesh: Mesh,
    axis_name: str,
    lead_v: jax.Array,
    own_out: jax.Array,
    rank_ids: jax.Array,
    w_o: jax.Array,
    w_o_scale: jax.Array,
) -> jax.Array:
    """Shift pre-projected lead/own rows to sequence rows and apply o_proj."""
    sharded = P(axis_name)

    def _local(lead_rows, own, rank_arr, w_shard, s_shard):
        # Avoid projecting s + R rows (an extra partial tile) before shifting:
        # shifting to each rank's R sequence rows first keeps quantize_rows and
        # o_proj on exact R rows while remaining bit-exact.
        seq_rows = token_rows_to_sequence_rows(
            layout,
            own.reshape(own.shape[0], -1),
            lead_rows.reshape(lead_rows.shape[0], -1),
            rank_arr[0],
            axis_name,
        )
        return project_all_heads(
            seq_rows, w_shard, s_shard, rank_arr[0], axis_name, layout.dcp_size
        )

    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(P(), sharded, sharded, sharded, sharded),
        out_specs=sharded,
        check_vma=False,
    )(lead_v, own_out, rank_ids, w_o, w_o_scale)


def project_sequence_rows(
    layout: TokenParallelLayout,
    mesh: Mesh,
    axis_name: str,
    lead_partial: jax.Array,
    lead_lse: jax.Array,
    own_out: jax.Array,
    rank_ids: jax.Array,
    w_uv: jax.Array,
    w_o: jax.Array,
    w_o_scale: jax.Array,
) -> jax.Array:
    """Split step's attention -> o_proj on sequence-parallel rows `[T, n]`.

    Global view, `[T, n]` sharded over `axis_name`, rank `r` holding step rows
    `r * T / dcp + i` -- what o_proj's reduce-scatter of head-sharded partial
    sums yields, without the all-to-all back to heads or the reduce-scatter:
    each rank gathers o_proj's FP8 row shards (weights only, independent of
    the step) and projects every head of its own token-parallel rows.

    Args:
      lead_partial, lead_lse: `[dcp * s, H, L]` / `[dcp * s, H]`, each rank's
        DCP partial of the `s = split_rows` leading rows over its KV shard.
      own_out: `[dcp * R, H, V]`, every head of each rank's token-parallel rows
        (`sparse_mla_attention(..., token_rows=True)` with `w_uv`).
      rank_ids: i32[dcp] sharded, entry r = r.
      w_uv: every head's `[H, L, V]`, replicated.
      w_o, w_o_scale: o_proj's FP8 row shards, sharded, in either layout of
        `project_all_heads` (rank r's own `(h * V, n)` shard and `[n]` scale,
        as the row-parallel o_proj holds them, or its chip pair's half).

    Notes:
      Assumptions: DCP rank r's heads are o_proj's input rows `[r * h * V,
      (r + 1) * h * V)` and its sequence-parallel rows are the TP rank r's
      (DCP group == TP group, as `moe_prefill_sp` already assumes).
    """
    sharded = P(axis_name)

    def _local(part, lse_part, own, rank_arr, w_uv_all, w_shard, s_shard):
        lead_v = _merge_lead_partial_to_v(
            part, lse_part, w_uv_all, axis_name, own.dtype
        )
        # Avoid projecting s + R rows (an extra partial tile) before shifting:
        # shifting to each rank's R sequence rows first keeps quantize_rows and
        # o_proj on exact R rows while remaining bit-exact.
        seq_rows = token_rows_to_sequence_rows(
            layout,
            own.reshape(own.shape[0], -1),
            lead_v.reshape(lead_v.shape[0], -1),
            rank_arr[0],
            axis_name,
        )
        return project_all_heads(
            seq_rows, w_shard, s_shard, rank_arr[0], axis_name, layout.dcp_size
        )

    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(sharded, sharded, sharded, sharded, P(), sharded, sharded),
        out_specs=sharded,
        check_vma=False,
    )(lead_partial, lead_lse, own_out, rank_ids, w_uv, w_o, w_o_scale)


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


def _sparse_mla_attention_core(
    layout: TokenParallelLayout,
    mesh: Mesh,
    axis_name: str,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    q: jax.Array,
    kv_c_normed: jax.Array | None,
    k_pe: jax.Array | None,
    flat_topk: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    *,
    write_kv: WriteKVFn | None,
    attend: Callable[..., jax.Array],
    rank_ids: jax.Array | None = None,
    q_own_rows: bool = False,
    read_pages: tuple[ReadPagesFn, ReadPagesFn] = (take_pages, take_pages),
    w_uv: jax.Array | None = None,
    q_pe: jax.Array | None = None,
    token_rows: bool = False,
    w_uv_replicated: bool = False,
    num_tokens: int | None = None,
    lead_partial: jax.Array | None = None,
    lead_lse: jax.Array | None = None,
    flat_topk_cap_strided: bool = False,
    w_o: jax.Array | None = None,
    w_o_scale: jax.Array | None = None,
    epilogue_3d: bool = True,
) -> tuple[jax.Array, ...]:
    if (lead_partial is None) != (lead_lse is None):
        raise ValueError("lead_partial and lead_lse must be passed together")
    if (w_o is None) != (w_o_scale is None):
        raise ValueError("w_o and w_o_scale must be passed together")
    has_lead = lead_partial is not None
    has_oproj = w_o is not None
    if has_oproj and not has_lead:
        raise ValueError("w_o/w_o_scale require lead_partial and lead_lse")
    if has_lead and (w_uv is None or not token_rows):
        raise ValueError("lead_partial/lead_lse require w_uv and token_rows=True")

    dcp = layout.dcp_size
    sharded = P(axis_name)
    rep = P()
    has_kv_inputs = write_kv is not None
    if has_kv_inputs and (kv_c_normed is None or k_pe is None):
        raise ValueError("kv_c_normed and k_pe are required when write_kv is set")
    step_tokens = (
        num_tokens
        if num_tokens is not None
        else (kv_c_normed.shape[0] if kv_c_normed is not None else None)
    )
    if step_tokens is None:
        raise ValueError("num_tokens or kv_c_normed must be provided")

    def _local(nope, rope, q, topk, seq_lens, bt, qsl, dist, *extra_args):
        idx = 0
        kv_c = k_pe_local = None
        if has_kv_inputs:
            kv_c, k_pe_local = extra_args[idx], extra_args[idx + 1]
            idx += 2
        q_pe_local = None
        if q_pe is not None:
            q_pe_local = extra_args[idx]
            idx += 1
        n_rank = 1 if rank_ids is not None else 0
        rank_arg = extra_args[idx : idx + n_rank]
        idx += n_rank
        w_uv_local = None
        if w_uv is not None:
            w_uv_local = extra_args[idx]
            idx += 1
        lead_part_local = lead_lse_local = None
        if has_lead:
            lead_part_local, lead_lse_local = extra_args[idx], extra_args[idx + 1]
            idx += 2
        w_o_local = s_o_local = None
        if has_oproj:
            w_o_local, s_o_local = extra_args[idx], extra_args[idx + 1]
        rank = _rank(rank_arg, axis_name, dcp)
        if write_kv is not None:
            nope, rope = write_kv(nope, rope, kv_c, k_pe_local, seq_lens, bt, qsl, rank)
        # Keep cache arrays in their [pages, rows, 128] layout across the cap
        # switch rather than propagating the SparseCore writer's 32-word view.
        nope, rope = lax.optimization_barrier((nope, rope))
        pages, offsets = step_pages(layout, seq_lens, qsl, dist[2])
        page_ids = gather_page_ids(layout, bt, pages, offsets, nope.shape[0])
        q_tok = q if q_own_rows else heads_to_tokens(layout, q, axis_name)
        q_pe_tok = (
            None
            if q_pe_local is None
            else (
                q_pe_local
                if q_own_rows
                else heads_to_tokens(layout, q_pe_local, axis_name)
            )
        )
        rr = rank_rows(layout, rank, step_tokens, seq_lens, qsl, dist[2])
        w_uv_all = (
            None
            if w_uv_local is None
            else (
                w_uv_local
                if w_uv_replicated
                else lax.all_gather(w_uv_local, axis_name, axis=0, tiled=True)
            )
        )
        use_3d_epilogue = False
        use_fused_wuv = False
        if has_oproj and epilogue_3d:
            assert (
                w_uv_all is not None and w_o_local is not None and s_o_local is not None
            )
            use_3d_epilogue = _can_use_3d_fp8_epilogue(
                layout,
                w_uv_all.shape[0],
                q_tok.shape[0],
                w_uv_all.shape[2],
                q_tok.dtype,
                w_o_local,
                s_o_local,
            )
            if use_3d_epilogue and has_lead:
                use_fused_wuv = _can_use_fused_wuv_epilogue(
                    layout, attend, w_uv_all, q_tok.shape[0], q_tok.dtype
                )

        # Gather and attend at the step's page count: the kernel reads a
        # buffer of `cap` pages per rank, with no padding copy to max_pages.
        # Running W_UV inside the branch avoids a relayout copy across the
        # switch boundary and halves switch output bytes.
        def branch(cap: int) -> Callable[[], Any]:
            def run() -> Any:
                g_nope, g_rope = gather_latent(
                    nope, rope, page_ids[:cap], axis_name, read_pages
                )
                flat = (
                    topk
                    if flat_topk_cap_strided
                    else restride_flat_rows(layout, topk, cap)
                )
                lead_v = None
                if has_lead:
                    assert (
                        lead_part_local is not None
                        and lead_lse_local is not None
                        and w_uv_all is not None
                    )
                    w_uv_arr = w_uv_all
                    # Avoid hoisting lead LSE AG / psum ahead of gather_latent:
                    # tying both via optimization_barrier prevents ICI contention
                    # with gather_latent while overlapping lead collectives with
                    # the zero-ICI attention kernel.
                    g_nope, g_rope, lp_t, ll_t = lax.optimization_barrier(
                        (g_nope, g_rope, lead_part_local, lead_lse_local)
                    )
                    if use_fused_wuv:
                        lead_v = jnp.swapaxes(
                            _merge_lead_partial_latent(lp_t, ll_t, axis_name), 0, 1
                        )
                    elif use_3d_epilogue:
                        lead_v = _merge_lead_partial_to_v_hrv(
                            lp_t, ll_t, w_uv_arr, axis_name, q_tok.dtype
                        )
                    else:
                        lead_v = _merge_lead_partial_to_v(
                            lp_t, ll_t, w_uv_arr, axis_name, q_tok.dtype
                        )
                if use_fused_wuv:
                    attend_hm = getattr(attend, "head_major_chunks")  # noqa: B009 -- optional kernel hook
                    if q_pe_tok is not None:
                        out = attend_hm(
                            q_tok, g_nope, g_rope, flat, rr.num_rows, q_pe_tok
                        )
                    else:
                        out = attend_hm(q_tok, g_nope, g_rope, flat, rr.num_rows)
                    assert lead_v is not None
                    return out, lead_v
                if q_pe_tok is not None:
                    out = attend(q_tok, g_nope, g_rope, flat, rr.num_rows, q_pe_tok)
                else:
                    out = attend(q_tok, g_nope, g_rope, flat, rr.num_rows)
                if w_uv_all is not None:
                    w_uv_arr = w_uv_all
                    out = (
                        _project_w_uv_hrv(out, w_uv_arr)
                        if use_3d_epilogue
                        else _project_w_uv(out, w_uv_arr)
                    )
                if has_lead:
                    assert lead_v is not None
                    return out, lead_v
                return out

            return run

        switched = _switch_on_cap(layout, jnp.sum(pages), branch)
        if has_oproj:
            assert w_o_local is not None and s_o_local is not None
            w_o_arr, s_o_arr = w_o_local, s_o_local
            out, lead_v = switched
            if use_fused_wuv:
                assert w_uv_all is not None
                rank_vec = jnp.asarray(
                    rank_arg[0] if rank_arg else jnp.reshape(rank, (1,)),
                    dtype=jnp.int32,
                )
                projected = _project_wuv_hm_pair_halves_local(
                    layout,
                    out,
                    lead_v,
                    w_uv_all,
                    rank_vec,
                    w_o_arr,
                    s_o_arr,
                    axis_name,
                    num_valid_own_rows=rr.num_rows,
                )
            elif use_3d_epilogue:
                rank_vec = jnp.asarray(
                    rank_arg[0] if rank_arg else jnp.reshape(rank, (1,)),
                    dtype=jnp.int32,
                )
                projected = _project_hrv_pair_halves_local(
                    layout,
                    out,
                    lead_v,
                    rank_vec,
                    w_o_arr,
                    s_o_arr,
                    axis_name,
                )
            else:
                projected = _shift_and_project_local(
                    layout,
                    out,
                    lead_v,
                    rank,
                    w_o_arr,
                    s_o_arr,
                    axis_name,
                )
            return nope, rope, projected
        if has_lead:
            out, lead_v = switched
            return nope, rope, out, lead_v
        out = switched
        if token_rows:
            return nope, rope, out
        return nope, rope, tokens_to_heads(out, axis_name, layout=layout)

    extra: tuple[jax.Array, ...] = ()
    extra_specs: tuple[P, ...] = ()
    if has_kv_inputs:
        assert kv_c_normed is not None and k_pe is not None
        extra = (*extra, kv_c_normed, k_pe)
        extra_specs = (*extra_specs, rep, rep)
    if q_pe is not None:
        extra = (*extra, q_pe)
        extra_specs = (*extra_specs, sharded)
    if rank_ids is not None:
        extra = (*extra, rank_ids)
        extra_specs = (*extra_specs, sharded)
    if w_uv is not None:
        extra = (*extra, w_uv)
        extra_specs = (*extra_specs, rep if w_uv_replicated else sharded)
    if has_lead:
        assert lead_partial is not None and lead_lse is not None
        extra = (*extra, lead_partial, lead_lse)
        extra_specs = (*extra_specs, sharded, sharded)
    if has_oproj:
        assert w_o is not None and w_o_scale is not None
        extra = (*extra, w_o, w_o_scale)
        extra_specs = (*extra_specs, sharded, sharded)
    out_specs = (
        (sharded, sharded, sharded, rep)
        if (has_lead and not has_oproj)
        else (sharded, sharded, sharded)
    )
    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(sharded, sharded, sharded, sharded, rep, rep, rep, rep) + extra_specs,
        out_specs=out_specs,
        check_vma=False,
    )(
        kv_cache_nope,
        kv_cache_rope,
        q,
        flat_topk,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        *extra,
    )


def sparse_mla_attention(
    layout: TokenParallelLayout,
    mesh: Mesh,
    axis_name: str,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    q: jax.Array,
    kv_c_normed: jax.Array | None,
    k_pe: jax.Array | None,
    flat_topk: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    *,
    write_kv: WriteKVFn | None,
    attend: Callable[..., jax.Array],
    rank_ids: jax.Array | None = None,
    q_own_rows: bool = False,
    read_pages: tuple[ReadPagesFn, ReadPagesFn] = (take_pages, take_pages),
    w_uv: jax.Array | None = None,
    q_pe: jax.Array | None = None,
    token_rows: bool = False,
    w_uv_replicated: bool = False,
    num_tokens: int | None = None,
    flat_topk_cap_strided: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Token-parallel sparse MLA for one layer (global view).

    Args:
      kv_cache_nope, kv_cache_rope: position shards over `axis_name`.
      q: `[dcp * T, h, D]`, each rank's `[T, h, D]` head slice of every row
        (latent nope and rope concatenated); with `q_own_rows`, `[dcp * R, H,
        D]`, every head of each rank's own rows (no all-to-all needed).
      kv_c_normed, k_pe: this step's latents, replicated; `write_kv` writes
        every row this rank owns before anything is gathered.
      flat_topk: `[dcp * R, K]`, each rank's rows' top-k as flat buffer rows
        (`flat_rows`), `-1` padded.
      seq_lens, block_tables, query_start_loc, request_distribution: the
        step's metadata, replicated.
      rank_ids: optional i32[dcp] sharded over `axis_name`, entry r = r.

    Returns:
      `(kv_cache_nope, kv_cache_rope, out)`, `out` `[dcp * T, h, L]` (or
      `[dcp * T, h, V]` when `w_uv` is given): each rank's head slice, row `i`
      being step row `split_rows + i`. With `token_rows`, `out` is instead
      `[dcp * R, H, L or V]`: every head of each rank's own token-parallel
      rows, with no all-to-all back to head sharding.
      With `w_uv_replicated`, `w_uv` is every head's `[H, L, V]` on every
      rank (no per-step all-gather of it).
    """
    nope, rope, out = _sparse_mla_attention_core(
        layout,
        mesh,
        axis_name,
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
        write_kv=write_kv,
        attend=attend,
        rank_ids=rank_ids,
        q_own_rows=q_own_rows,
        read_pages=read_pages,
        w_uv=w_uv,
        q_pe=q_pe,
        token_rows=token_rows,
        w_uv_replicated=w_uv_replicated,
        num_tokens=num_tokens,
        flat_topk_cap_strided=flat_topk_cap_strided,
    )
    return nope, rope, out


def sparse_mla_attention_with_lead(
    layout: TokenParallelLayout,
    mesh: Mesh,
    axis_name: str,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    q: jax.Array,
    kv_c_normed: jax.Array | None,
    k_pe: jax.Array | None,
    flat_topk: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    *,
    lead_partial: jax.Array,
    lead_lse: jax.Array,
    write_kv: WriteKVFn | None,
    attend: Callable[..., jax.Array],
    w_uv: jax.Array,
    rank_ids: jax.Array | None = None,
    q_own_rows: bool = False,
    read_pages: tuple[ReadPagesFn, ReadPagesFn] = (take_pages, take_pages),
    q_pe: jax.Array | None = None,
    w_uv_replicated: bool = False,
    num_tokens: int | None = None,
    flat_topk_cap_strided: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run token-parallel sparse MLA with overlapped lead merge + W_UV inside branch."""
    nope, rope, own_v, lead_v = _sparse_mla_attention_core(
        layout,
        mesh,
        axis_name,
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
        write_kv=write_kv,
        attend=attend,
        rank_ids=rank_ids,
        q_own_rows=q_own_rows,
        read_pages=read_pages,
        w_uv=w_uv,
        q_pe=q_pe,
        token_rows=True,
        w_uv_replicated=w_uv_replicated,
        num_tokens=num_tokens,
        lead_partial=lead_partial,
        lead_lse=lead_lse,
        flat_topk_cap_strided=flat_topk_cap_strided,
    )
    return nope, rope, own_v, lead_v


def sparse_mla_attention_with_lead_oproj(
    layout: TokenParallelLayout,
    mesh: Mesh,
    axis_name: str,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    q: jax.Array,
    kv_c_normed: jax.Array | None,
    k_pe: jax.Array | None,
    flat_topk: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    *,
    lead_partial: jax.Array,
    lead_lse: jax.Array,
    write_kv: WriteKVFn | None,
    attend: Callable[..., jax.Array],
    w_uv: jax.Array,
    w_o: jax.Array,
    w_o_scale: jax.Array,
    rank_ids: jax.Array | None = None,
    q_own_rows: bool = False,
    read_pages: tuple[ReadPagesFn, ReadPagesFn] = (take_pages, take_pages),
    q_pe: jax.Array | None = None,
    w_uv_replicated: bool = False,
    num_tokens: int | None = None,
    flat_topk_cap_strided: bool = False,
    epilogue_3d: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run token-parallel sparse MLA, lead merge, W_UV, and o_proj in one shard_map."""
    nope, rope, projected = _sparse_mla_attention_core(
        layout,
        mesh,
        axis_name,
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
        write_kv=write_kv,
        attend=attend,
        rank_ids=rank_ids,
        q_own_rows=q_own_rows,
        read_pages=read_pages,
        w_uv=w_uv,
        q_pe=q_pe,
        token_rows=True,
        w_uv_replicated=w_uv_replicated,
        num_tokens=num_tokens,
        lead_partial=lead_partial,
        lead_lse=lead_lse,
        flat_topk_cap_strided=flat_topk_cap_strided,
        w_o=w_o,
        w_o_scale=w_o_scale,
        epilogue_3d=epilogue_3d,
    )
    return nope, rope, projected


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
    q_own_rows: bool = False,
    flat_topk_cap_strided: bool = False,
    index_k_packed_u32: bool = False,
) -> jax.Array:
    """Token-parallel indexer top-k for one layer (global view).

    `index_cache` (position shards, this step's K already written), `q`
    `[T, h, d]` and `weights` `[T, h]` (replicated, every row; or, with
    `q_own_rows`, `[dcp * R, h, d]` and `[dcp * R, h]`, each rank's own rows
    sharded on `axis_name`). Returns `[dcp * R, K]`: each rank's rows' top-k
    as flat rows of the latent gather buffer (`flat_rows`), `-1` padded, and
    `-1` on padding rows.
    """
    dcp = layout.dcp_size
    sharded = P(axis_name)
    rep = P()

    def _local(cache, q, weights, seq_lens, bt, qsl, dist, *rank_ids):
        rank = _rank(rank_ids, axis_name, dcp)
        num_tokens = q.shape[0] * dcp if q_own_rows else q.shape[0]
        rows = layout.rows_per_rank(num_tokens)
        pages, offsets = step_pages(layout, seq_lens, qsl, dist[2])
        page_ids = gather_page_ids(layout, bt, pages, offsets, cache.shape[0])
        rr = rank_rows(layout, rank, num_tokens, seq_lens, qsl, dist[2])
        q_r, w_r = (
            (q, weights)
            if q_own_rows
            else (own_rows(layout, q, rank), own_rows(layout, weights, rank))
        )
        vpages_per_seq = bt.shape[0] // seq_lens.shape[0]
        il, ps = layout.interleave_size, layout.page_size
        row_offsets = offsets[row_requests(rr, rows)]
        real = jnp.arange(rows, dtype=jnp.int32) < rr.num_rows
        q_sel, w_sel, rr_sel, r0_shifted = _align_zigzag_r0_select_inputs(
            layout, num_tokens, rank, q_r, w_r, rr
        )

        def branch(cap: int) -> Callable[[], jax.Array]:
            def run() -> jax.Array:
                nat = _gather_index_k_pages(
                    cache,
                    page_ids,
                    cap,
                    dcp,
                    il,
                    ps,
                    axis_name,
                    packed_u32=index_k_packed_u32,
                )
                # Avoid hoisting natural_page_table above branch(cap): full
                # vpages_per_seq width inflates score buffers to max_pages * dcp
                # and forces multi-partition top-k barriers on <=64K-token caps.
                table = natural_page_table(layout, rr, offsets, vpages_per_seq, cap=cap)
                positions = _unalign_zigzag_r0_positions(
                    select_topk(q_sel, w_sel, nat, rr_sel, table),
                    rank,
                    layout.split_rows,
                    r0_shifted,
                )
                if flat_topk_cap_strided:
                    # Avoid striding at max_pages and restriding in every MLA
                    # layer: all layers of the step select the same cap from
                    # step_pages, so striding at cap once removes redundant
                    # per-layer int32 div/mod.
                    flat = flat_rows(layout, positions, row_offsets, stride_pages=cap)
                    return jnp.where(real[:, None], flat, -1)
                return positions

            return run

        out = _switch_on_cap(layout, jnp.sum(pages), branch)
        if flat_topk_cap_strided:
            return out
        flat = flat_rows(layout, out, row_offsets)
        return jnp.where(real[:, None], flat, -1)

    extra = () if rank_ids is None else (rank_ids,)
    q_spec = sharded if q_own_rows else rep
    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(sharded, q_spec, q_spec, rep, rep, rep, rep)
        + (sharded,) * len(extra),
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


def _unified_indexer_mla_core(
    layout: TokenParallelLayout,
    mesh: Mesh,
    axis_name: str,
    index_cache: jax.Array,
    q_idx: jax.Array,
    w_idx: jax.Array,
    idx_block_tables: jax.Array,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    q: jax.Array,
    q_pe: jax.Array,
    mla_block_tables: jax.Array,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    *,
    select_topk: SelectTopkFn,
    attend: Callable[..., jax.Array],
    lead_partial: jax.Array,
    lead_lse: jax.Array,
    w_uv: jax.Array,
    num_tokens: int,
    rank_ids: jax.Array | None = None,
    read_pages: tuple[ReadPagesFn, ReadPagesFn] = (take_pages, take_pages),
    w_uv_replicated: bool = False,
    tie_after_idx_max_cap: int = 0,
    w_o: jax.Array | None = None,
    w_o_scale: jax.Array | None = None,
    epilogue_3d: bool = True,
    index_k_packed_u32: bool = False,
) -> tuple[jax.Array, ...]:
    if (w_o is None) != (w_o_scale is None):
        raise ValueError("w_o and w_o_scale must be passed together")
    has_oproj = w_o is not None
    dcp = layout.dcp_size
    sharded = P(axis_name)
    rep = P()

    def _local(
        idx_cache,
        q_idx_r,
        w_idx_r,
        idx_bt,
        nope,
        rope,
        q_tok,
        q_pe_tok,
        mla_bt,
        sl,
        qsl,
        dist,
        lead_part_local,
        lead_lse_local,
        w_uv_local,
        *extra_args,
    ):
        idx = 0
        n_rank = 1 if rank_ids is not None else 0
        rank_arg = extra_args[idx : idx + n_rank]
        idx += n_rank
        w_o_local = s_o_local = None
        if has_oproj:
            w_o_local, s_o_local = extra_args[idx], extra_args[idx + 1]
        rank = _rank(rank_arg, axis_name, dcp)
        # Avoid propagating the SparseCore writer's 32-word view across the cap
        # switch: keep latent cache arrays in their [pages, rows, 128] layout.
        nope, rope = lax.optimization_barrier((nope, rope))
        pages, offsets = step_pages(layout, sl, qsl, dist[2])
        idx_page_ids = gather_page_ids(
            layout, idx_bt, pages, offsets, idx_cache.shape[0]
        )
        mla_page_ids = gather_page_ids(layout, mla_bt, pages, offsets, nope.shape[0])
        rr = rank_rows(layout, rank, num_tokens, sl, qsl, dist[2])
        rows = layout.rows_per_rank(num_tokens)
        idx_vpages_per_seq = idx_bt.shape[0] // sl.shape[0]
        il, ps = layout.interleave_size, layout.page_size
        row_offsets = offsets[row_requests(rr, rows)]
        real = jnp.arange(rows, dtype=jnp.int32) < rr.num_rows
        w_uv_all = (
            w_uv_local
            if w_uv_replicated
            else lax.all_gather(w_uv_local, axis_name, axis=0, tiled=True)
        )
        use_3d_epilogue = False
        use_fused_wuv = False
        if has_oproj and epilogue_3d:
            assert w_o_local is not None and s_o_local is not None
            use_3d_epilogue = _can_use_3d_fp8_epilogue(
                layout,
                w_uv_all.shape[0],
                rows,
                w_uv_all.shape[2],
                q_tok.dtype,
                w_o_local,
                s_o_local,
            )
            if use_3d_epilogue:
                use_fused_wuv = _can_use_fused_wuv_epilogue(
                    layout, attend, w_uv_all, rows, q_tok.dtype
                )

        q_idx_sel, w_idx_sel, rr_sel, r0_shifted = _align_zigzag_r0_select_inputs(
            layout, num_tokens, rank, q_idx_r, w_idx_r, rr
        )

        def branch(cap: int) -> Callable[[], tuple[Any, jax.Array, jax.Array]]:
            def run() -> tuple[Any, jax.Array, jax.Array]:
                nat = _gather_index_k_pages(
                    idx_cache,
                    idx_page_ids,
                    cap,
                    dcp,
                    il,
                    ps,
                    axis_name,
                    packed_u32=index_k_packed_u32,
                )
                table = natural_page_table(
                    layout, rr, offsets, idx_vpages_per_seq, cap=cap
                )
                if tie_after_idx_max_cap > 0 and cap <= tie_after_idx_max_cap:
                    positions = select_topk(q_idx_sel, w_idx_sel, nat, rr_sel, table)
                    # Avoid small-cap page_gather/StreamIdxTC contention: wait
                    # for StreamIdxTC before launching gather_latent on tiny caps.
                    positions, nope_t, rope_t = lax.optimization_barrier(
                        (positions, nope, rope)
                    )
                    g_nope, g_rope = gather_latent(
                        nope_t, rope_t, mla_page_ids[:cap], axis_name, read_pages
                    )
                else:
                    g_nope, g_rope = gather_latent(
                        nope, rope, mla_page_ids[:cap], axis_name, read_pages
                    )
                    positions = select_topk(q_idx_sel, w_idx_sel, nat, rr_sel, table)
                positions = _unalign_zigzag_r0_positions(
                    positions, rank, layout.split_rows, r0_shifted
                )
                flat = jnp.where(
                    real[:, None],
                    flat_rows(layout, positions, row_offsets, stride_pages=cap),
                    -1,
                )
                # Avoid hoisting lead LSE AG / psum ahead of gather_latent:
                # tying both via optimization_barrier prevents ICI contention
                # with gather_latent while overlapping lead collectives with
                # the zero-ICI attention kernel.
                g_nope, g_rope, lp_t, ll_t = lax.optimization_barrier(
                    (g_nope, g_rope, lead_part_local, lead_lse_local)
                )
                if use_fused_wuv:
                    lead_v = jnp.swapaxes(
                        _merge_lead_partial_latent(lp_t, ll_t, axis_name), 0, 1
                    )
                    attend_hm = getattr(attend, "head_major_chunks")  # noqa: B009 -- optional kernel hook
                    out = attend_hm(q_tok, g_nope, g_rope, flat, rr.num_rows, q_pe_tok)
                    return out, lead_v, flat
                if use_3d_epilogue:
                    lead_v = _merge_lead_partial_to_v_hrv(
                        lp_t, ll_t, w_uv_all, axis_name, q_tok.dtype
                    )
                else:
                    lead_v = _merge_lead_partial_to_v(
                        lp_t, ll_t, w_uv_all, axis_name, q_tok.dtype
                    )
                out = attend(q_tok, g_nope, g_rope, flat, rr.num_rows, q_pe_tok)
                out = (
                    _project_w_uv_hrv(out, w_uv_all)
                    if use_3d_epilogue
                    else _project_w_uv(out, w_uv_all)
                )
                return out, lead_v, flat

            return run

        out, lead_v, flat = _switch_on_cap(layout, jnp.sum(pages), branch)
        if has_oproj:
            assert w_o_local is not None and s_o_local is not None
            w_o_arr, s_o_arr = w_o_local, s_o_local
            if use_fused_wuv:
                rank_vec = jnp.asarray(
                    rank_arg[0] if rank_arg else jnp.reshape(rank, (1,)),
                    dtype=jnp.int32,
                )
                projected = _project_wuv_hm_pair_halves_local(
                    layout,
                    out,
                    lead_v,
                    w_uv_all,
                    rank_vec,
                    w_o_arr,
                    s_o_arr,
                    axis_name,
                    num_valid_own_rows=rr.num_rows,
                )
            elif use_3d_epilogue:
                rank_vec = jnp.asarray(
                    rank_arg[0] if rank_arg else jnp.reshape(rank, (1,)),
                    dtype=jnp.int32,
                )
                projected = _project_hrv_pair_halves_local(
                    layout, out, lead_v, rank_vec, w_o_arr, s_o_arr, axis_name
                )
            else:
                projected = _shift_and_project_local(
                    layout, out, lead_v, rank, w_o_arr, s_o_arr, axis_name
                )
            return projected, flat
        return out, lead_v, flat

    extra: tuple[jax.Array, ...] = () if rank_ids is None else (rank_ids,)
    if has_oproj:
        assert w_o is not None and w_o_scale is not None
        extra = (*extra, w_o, w_o_scale)
    w_uv_spec = rep if w_uv_replicated else sharded
    out_specs = (sharded, sharded) if has_oproj else (sharded, rep, sharded)
    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            sharded,
            sharded,
            sharded,
            rep,
            sharded,
            sharded,
            sharded,
            sharded,
            rep,
            rep,
            rep,
            rep,
            sharded,
            sharded,
            w_uv_spec,
        )
        + (sharded,) * len(extra),
        out_specs=out_specs,
        check_vma=False,
    )(
        index_cache,
        q_idx,
        w_idx,
        idx_block_tables,
        kv_cache_nope,
        kv_cache_rope,
        q,
        q_pe,
        mla_block_tables,
        seq_lens,
        query_start_loc,
        request_distribution,
        lead_partial,
        lead_lse,
        w_uv,
        *extra,
    )


def unified_indexer_mla_with_lead(
    layout: TokenParallelLayout,
    mesh: Mesh,
    axis_name: str,
    index_cache: jax.Array,
    q_idx: jax.Array,
    w_idx: jax.Array,
    idx_block_tables: jax.Array,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    q: jax.Array,
    q_pe: jax.Array,
    mla_block_tables: jax.Array,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    *,
    select_topk: SelectTopkFn,
    attend: Callable[..., jax.Array],
    lead_partial: jax.Array,
    lead_lse: jax.Array,
    w_uv: jax.Array,
    num_tokens: int,
    rank_ids: jax.Array | None = None,
    read_pages: tuple[ReadPagesFn, ReadPagesFn] = (take_pages, take_pages),
    w_uv_replicated: bool = False,
    tie_after_idx_max_cap: int = 0,
    index_k_packed_u32: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run token-parallel indexer top-k, latent gather, lead merge, and MLA together.

    Returns `(own_v [dcp*R, H, V], lead_v [S, H, V], tail_flat_topk [dcp*R, K])`
    bit-identical to separate `indexer_topk(..., flat_topk_cap_strided=True)` and
    `sparse_mla_attention_with_lead(..., flat_topk_cap_strided=True)`.
    """
    own_v, lead_v, flat = _unified_indexer_mla_core(
        layout,
        mesh,
        axis_name,
        index_cache,
        q_idx,
        w_idx,
        idx_block_tables,
        kv_cache_nope,
        kv_cache_rope,
        q,
        q_pe,
        mla_block_tables,
        seq_lens,
        query_start_loc,
        request_distribution,
        select_topk=select_topk,
        attend=attend,
        lead_partial=lead_partial,
        lead_lse=lead_lse,
        w_uv=w_uv,
        num_tokens=num_tokens,
        rank_ids=rank_ids,
        read_pages=read_pages,
        w_uv_replicated=w_uv_replicated,
        tie_after_idx_max_cap=tie_after_idx_max_cap,
        index_k_packed_u32=index_k_packed_u32,
    )
    return own_v, lead_v, flat


def unified_indexer_mla_with_lead_oproj(
    layout: TokenParallelLayout,
    mesh: Mesh,
    axis_name: str,
    index_cache: jax.Array,
    q_idx: jax.Array,
    w_idx: jax.Array,
    idx_block_tables: jax.Array,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    q: jax.Array,
    q_pe: jax.Array,
    mla_block_tables: jax.Array,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    *,
    select_topk: SelectTopkFn,
    attend: Callable[..., jax.Array],
    lead_partial: jax.Array,
    lead_lse: jax.Array,
    w_uv: jax.Array,
    w_o: jax.Array,
    w_o_scale: jax.Array,
    num_tokens: int,
    rank_ids: jax.Array | None = None,
    read_pages: tuple[ReadPagesFn, ReadPagesFn] = (take_pages, take_pages),
    w_uv_replicated: bool = False,
    tie_after_idx_max_cap: int = 0,
    epilogue_3d: bool = True,
    index_k_packed_u32: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Run unified indexer + sparse MLA + sequence-row o_proj in one shard_map."""
    projected, flat = _unified_indexer_mla_core(
        layout,
        mesh,
        axis_name,
        index_cache,
        q_idx,
        w_idx,
        idx_block_tables,
        kv_cache_nope,
        kv_cache_rope,
        q,
        q_pe,
        mla_block_tables,
        seq_lens,
        query_start_loc,
        request_distribution,
        select_topk=select_topk,
        attend=attend,
        lead_partial=lead_partial,
        lead_lse=lead_lse,
        w_uv=w_uv,
        num_tokens=num_tokens,
        rank_ids=rank_ids,
        read_pages=read_pages,
        w_uv_replicated=w_uv_replicated,
        tie_after_idx_max_cap=tie_after_idx_max_cap,
        w_o=w_o,
        w_o_scale=w_o_scale,
        epilogue_3d=epilogue_3d,
        index_k_packed_u32=index_k_packed_u32,
    )
    return projected, flat


def _sparse_mla_in_branch_qb_core(
    layout: TokenParallelLayout,
    mesh: Mesh,
    axis_name: str,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    q_c_own: jax.Array,
    w_qb: jax.Array,
    s_qb: jax.Array,
    w_uk_t: jax.Array,
    cos_sin: jax.Array,
    head_topk: jax.Array,
    flat_topk: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    *,
    fused_qb: Callable[..., tuple[jax.Array, jax.Array, jax.Array, jax.Array]],
    lead_attend: Callable[..., tuple[jax.Array, jax.Array]],
    tail_attend: Callable[..., jax.Array],
    w_uv: jax.Array,
    num_tokens: int,
    rank_ids: jax.Array | None = None,
    read_pages: tuple[ReadPagesFn, ReadPagesFn] = (take_pages, take_pages),
    w_uv_replicated: bool = False,
    flat_topk_cap_strided: bool = False,
    free_nobar_max_cap: int = 0,
    w_o: jax.Array | None = None,
    w_o_scale: jax.Array | None = None,
    epilogue_3d: bool = True,
) -> Any:
    if (w_o is None) != (w_o_scale is None):
        raise ValueError("w_o and w_o_scale must be passed together")
    has_oproj = w_o is not None
    dcp = layout.dcp_size
    s = layout.split_rows
    sharded = P(axis_name)
    rep = P()

    def _local(
        nope,
        rope,
        qc_r,
        wqb_r,
        sqb_r,
        wuk_r,
        cs_r,
        h_topk_r,
        f_topk_r,
        sl,
        bt,
        qsl,
        dist,
        w_uv_local,
        *extra_args,
    ):
        idx = 0
        n_rank = 1 if rank_ids is not None else 0
        rank_arg = extra_args[idx : idx + n_rank]
        idx += n_rank
        w_o_local = s_o_local = None
        if has_oproj:
            w_o_local, s_o_local = extra_args[idx], extra_args[idx + 1]
        rank = _rank(rank_arg, axis_name, dcp)
        # Avoid propagating the SparseCore writer's 32-word view across the cap
        # switch: keep latent cache arrays in their [pages, rows, 128] layout.
        nope, rope = lax.optimization_barrier((nope, rope))
        pages, offsets = step_pages(layout, sl, qsl, dist[2])
        page_ids = gather_page_ids(layout, bt, pages, offsets, nope.shape[0])
        rr = rank_rows(layout, rank, num_tokens, sl, qsl, dist[2])
        rows = layout.rows_per_rank(num_tokens)
        w_uv_all = (
            w_uv_local
            if w_uv_replicated
            else lax.all_gather(w_uv_local, axis_name, axis=0, tiled=True)
        )
        use_3d_epilogue = False
        use_fused_wuv = False
        if has_oproj and epilogue_3d:
            assert w_o_local is not None and s_o_local is not None
            use_3d_epilogue = _can_use_3d_fp8_epilogue(
                layout,
                w_uv_all.shape[0],
                rows,
                w_uv_all.shape[2],
                qc_r.dtype,
                w_o_local,
                s_o_local,
            )
            if use_3d_epilogue:
                use_fused_wuv = _can_use_fused_wuv_epilogue(
                    layout, tail_attend, w_uv_all, rows, qc_r.dtype
                )

        def branch(cap: int) -> Callable[[], tuple[jax.Array, jax.Array]]:
            def run() -> tuple[jax.Array, jax.Array]:
                g_nope, g_rope = gather_latent(
                    nope, rope, page_ids[:cap], axis_name, read_pages
                )
                flat = (
                    f_topk_r
                    if flat_topk_cap_strided
                    else restride_flat_rows(layout, f_topk_r, cap)
                )
                ql_head, ql_tail, qpe_head, qpe_tail = fused_qb(
                    qc_r, wqb_r, sqb_r, wuk_r, cs_r, s
                )
                if cap > free_nobar_max_cap:
                    # Avoid SparseCore contention between lead_mla and SC-offloaded
                    # gather_latent on caps >= 12: hold lead_mla behind a barrier
                    # until gather_latent completes while pure-TC fused_qb overlaps
                    # gather_latent; smaller caps leave fused_qb free to overlap
                    # Chunk-0 SparseCore KV gather.
                    g_nope, g_rope, ql_head, qpe_head = lax.optimization_barrier(
                        (g_nope, g_rope, ql_head, qpe_head)
                    )
                lead_partial, lead_lse = lead_attend(
                    ql_head, qpe_head, nope, rope, h_topk_r, bt, sl, qsl, dist
                )
                if use_fused_wuv:
                    lead_v = jnp.swapaxes(
                        _merge_lead_partial_latent(lead_partial, lead_lse, axis_name),
                        0,
                        1,
                    )
                    attend_hm = getattr(tail_attend, "head_major_chunks")  # noqa: B009 -- optional kernel hook
                    out = attend_hm(
                        ql_tail, g_nope, g_rope, flat, rr.num_rows, qpe_tail
                    )
                    return out, lead_v
                if use_3d_epilogue:
                    lead_v = _merge_lead_partial_to_v_hrv(
                        lead_partial, lead_lse, w_uv_all, axis_name, ql_tail.dtype
                    )
                else:
                    lead_v = _merge_lead_partial_to_v(
                        lead_partial, lead_lse, w_uv_all, axis_name, ql_tail.dtype
                    )
                out = tail_attend(ql_tail, g_nope, g_rope, flat, rr.num_rows, qpe_tail)
                out = (
                    _project_w_uv_hrv(out, w_uv_all)
                    if use_3d_epilogue
                    else _project_w_uv(out, w_uv_all)
                )
                return out, lead_v

            return run

        out, lead_v = _switch_on_cap(layout, jnp.sum(pages), branch)
        if has_oproj:
            assert w_o_local is not None and s_o_local is not None
            w_o_arr, s_o_arr = w_o_local, s_o_local
            if use_fused_wuv:
                rank_vec = jnp.asarray(
                    rank_arg[0] if rank_arg else jnp.reshape(rank, (1,)),
                    dtype=jnp.int32,
                )
                return _project_wuv_hm_pair_halves_local(
                    layout,
                    out,
                    lead_v,
                    w_uv_all,
                    rank_vec,
                    w_o_arr,
                    s_o_arr,
                    axis_name,
                    num_valid_own_rows=rr.num_rows,
                )
            if use_3d_epilogue:
                rank_vec = jnp.asarray(
                    rank_arg[0] if rank_arg else jnp.reshape(rank, (1,)),
                    dtype=jnp.int32,
                )
                return _project_hrv_pair_halves_local(
                    layout, out, lead_v, rank_vec, w_o_arr, s_o_arr, axis_name
                )
            return _shift_and_project_local(
                layout, out, lead_v, rank, w_o_arr, s_o_arr, axis_name
            )
        return out, lead_v

    extra: tuple[jax.Array, ...] = () if rank_ids is None else (rank_ids,)
    if has_oproj:
        assert w_o is not None and w_o_scale is not None
        extra = (*extra, w_o, w_o_scale)
    w_uv_spec = rep if w_uv_replicated else sharded
    out_specs = sharded if has_oproj else (sharded, rep)
    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            sharded,
            sharded,
            sharded,
            rep,
            rep,
            rep,
            sharded,
            sharded,
            sharded,
            rep,
            rep,
            rep,
            rep,
            w_uv_spec,
        )
        + (sharded,) * len(extra),
        out_specs=out_specs,
        check_vma=False,
    )(
        kv_cache_nope,
        kv_cache_rope,
        q_c_own,
        w_qb,
        s_qb,
        w_uk_t,
        cos_sin,
        head_topk,
        flat_topk,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        w_uv,
        *extra,
    )


def sparse_mla_attention_with_in_branch_qb(
    layout: TokenParallelLayout,
    mesh: Mesh,
    axis_name: str,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    q_c_own: jax.Array,
    w_qb: jax.Array,
    s_qb: jax.Array,
    w_uk_t: jax.Array,
    cos_sin: jax.Array,
    head_topk: jax.Array,
    flat_topk: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    *,
    fused_qb: Callable[..., tuple[jax.Array, jax.Array, jax.Array, jax.Array]],
    lead_attend: Callable[..., tuple[jax.Array, jax.Array]],
    tail_attend: Callable[..., jax.Array],
    w_uv: jax.Array,
    num_tokens: int,
    rank_ids: jax.Array | None = None,
    read_pages: tuple[ReadPagesFn, ReadPagesFn] = (take_pages, take_pages),
    w_uv_replicated: bool = False,
    flat_topk_cap_strided: bool = False,
    free_nobar_max_cap: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Run fused `q_b`, lead MLA, and token-parallel MLA in one cap switch.

    Returns `(own_v [dcp*R, H, V], lead_v [S, H, V])` matching separate
    `fused_qb_rope_absorb_split_qpe` + `sparse_mla_attention_dcp` +
    `sparse_mla_attention_with_lead` within 1 bf16 ULP on `lead_v` and bit-exact
    on `own_v`.
    """
    return _sparse_mla_in_branch_qb_core(
        layout,
        mesh,
        axis_name,
        kv_cache_nope,
        kv_cache_rope,
        q_c_own,
        w_qb,
        s_qb,
        w_uk_t,
        cos_sin,
        head_topk,
        flat_topk,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        fused_qb=fused_qb,
        lead_attend=lead_attend,
        tail_attend=tail_attend,
        w_uv=w_uv,
        num_tokens=num_tokens,
        rank_ids=rank_ids,
        read_pages=read_pages,
        w_uv_replicated=w_uv_replicated,
        flat_topk_cap_strided=flat_topk_cap_strided,
        free_nobar_max_cap=free_nobar_max_cap,
    )


def sparse_mla_attention_with_in_branch_qb_oproj(
    layout: TokenParallelLayout,
    mesh: Mesh,
    axis_name: str,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    q_c_own: jax.Array,
    w_qb: jax.Array,
    s_qb: jax.Array,
    w_uk_t: jax.Array,
    cos_sin: jax.Array,
    head_topk: jax.Array,
    flat_topk: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    *,
    fused_qb: Callable[..., tuple[jax.Array, jax.Array, jax.Array, jax.Array]],
    lead_attend: Callable[..., tuple[jax.Array, jax.Array]],
    tail_attend: Callable[..., jax.Array],
    w_uv: jax.Array,
    w_o: jax.Array,
    w_o_scale: jax.Array,
    num_tokens: int,
    rank_ids: jax.Array | None = None,
    read_pages: tuple[ReadPagesFn, ReadPagesFn] = (take_pages, take_pages),
    w_uv_replicated: bool = False,
    flat_topk_cap_strided: bool = False,
    free_nobar_max_cap: int = 0,
    epilogue_3d: bool = True,
) -> jax.Array:
    """Run in-branch `q_b`, lead MLA, sparse MLA, and o_proj in one shard_map."""
    return _sparse_mla_in_branch_qb_core(
        layout,
        mesh,
        axis_name,
        kv_cache_nope,
        kv_cache_rope,
        q_c_own,
        w_qb,
        s_qb,
        w_uk_t,
        cos_sin,
        head_topk,
        flat_topk,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        fused_qb=fused_qb,
        lead_attend=lead_attend,
        tail_attend=tail_attend,
        w_uv=w_uv,
        num_tokens=num_tokens,
        rank_ids=rank_ids,
        read_pages=read_pages,
        w_uv_replicated=w_uv_replicated,
        flat_topk_cap_strided=flat_topk_cap_strided,
        free_nobar_max_cap=free_nobar_max_cap,
        w_o=w_o,
        w_o_scale=w_o_scale,
        epilogue_3d=epilogue_3d,
    )


def _unified_indexer_in_branch_qb_core(
    layout: TokenParallelLayout,
    mesh: Mesh,
    axis_name: str,
    index_cache: jax.Array,
    q_idx: jax.Array,
    w_idx: jax.Array,
    idx_block_tables: jax.Array,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    q_c_own: jax.Array,
    w_qb: jax.Array,
    s_qb: jax.Array,
    w_uk_t: jax.Array,
    cos_sin: jax.Array,
    head_topk: jax.Array,
    mla_block_tables: jax.Array,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    *,
    select_topk: SelectTopkFn,
    fused_qb: Callable[..., tuple[jax.Array, jax.Array, jax.Array, jax.Array]],
    lead_attend: Callable[..., tuple[jax.Array, jax.Array]],
    tail_attend: Callable[..., jax.Array],
    w_uv: jax.Array,
    num_tokens: int,
    rank_ids: jax.Array | None = None,
    read_pages: tuple[ReadPagesFn, ReadPagesFn] = (take_pages, take_pages),
    w_uv_replicated: bool = False,
    tie_after_idx_max_cap: int = 0,
    tie_lead_min_cap: int = 0,
    w_o: jax.Array | None = None,
    w_o_scale: jax.Array | None = None,
    epilogue_3d: bool = True,
    index_k_packed_u32: bool = False,
) -> tuple[jax.Array, ...]:
    if (w_o is None) != (w_o_scale is None):
        raise ValueError("w_o and w_o_scale must be passed together")
    has_oproj = w_o is not None
    dcp = layout.dcp_size
    s = layout.split_rows
    sharded = P(axis_name)
    rep = P()

    def _local(
        idx_cache,
        q_idx_r,
        w_idx_r,
        idx_bt,
        nope,
        rope,
        qc_r,
        wqb_r,
        sqb_r,
        wuk_r,
        cs_r,
        h_topk_r,
        mla_bt,
        sl,
        qsl,
        dist,
        w_uv_local,
        *extra_args,
    ):
        idx = 0
        n_rank = 1 if rank_ids is not None else 0
        rank_arg = extra_args[idx : idx + n_rank]
        idx += n_rank
        w_o_local = s_o_local = None
        if has_oproj:
            w_o_local, s_o_local = extra_args[idx], extra_args[idx + 1]
        rank = _rank(rank_arg, axis_name, dcp)
        # Avoid propagating the SparseCore writer's 32-word view across the cap
        # switch: keep latent cache arrays in their [pages, rows, 128] layout.
        nope, rope = lax.optimization_barrier((nope, rope))
        pages, offsets = step_pages(layout, sl, qsl, dist[2])
        idx_page_ids = gather_page_ids(
            layout, idx_bt, pages, offsets, idx_cache.shape[0]
        )
        mla_page_ids = gather_page_ids(layout, mla_bt, pages, offsets, nope.shape[0])
        rr = rank_rows(layout, rank, num_tokens, sl, qsl, dist[2])
        rows = layout.rows_per_rank(num_tokens)
        idx_vpages_per_seq = idx_bt.shape[0] // sl.shape[0]
        il, ps = layout.interleave_size, layout.page_size
        row_offsets = offsets[row_requests(rr, rows)]
        real = jnp.arange(rows, dtype=jnp.int32) < rr.num_rows
        w_uv_all = (
            w_uv_local
            if w_uv_replicated
            else lax.all_gather(w_uv_local, axis_name, axis=0, tiled=True)
        )
        use_3d_epilogue = False
        use_fused_wuv = False
        if has_oproj and epilogue_3d:
            assert w_o_local is not None and s_o_local is not None
            use_3d_epilogue = _can_use_3d_fp8_epilogue(
                layout,
                w_uv_all.shape[0],
                rows,
                w_uv_all.shape[2],
                qc_r.dtype,
                w_o_local,
                s_o_local,
            )
            if use_3d_epilogue:
                use_fused_wuv = _can_use_fused_wuv_epilogue(
                    layout, tail_attend, w_uv_all, rows, qc_r.dtype
                )

        q_idx_sel, w_idx_sel, rr_sel, r0_shifted = _align_zigzag_r0_select_inputs(
            layout, num_tokens, rank, q_idx_r, w_idx_r, rr
        )

        def branch(cap: int) -> Callable[[], tuple[jax.Array, jax.Array, jax.Array]]:
            def run() -> tuple[jax.Array, jax.Array, jax.Array]:
                nat = _gather_index_k_pages(
                    idx_cache,
                    idx_page_ids,
                    cap,
                    dcp,
                    il,
                    ps,
                    axis_name,
                    packed_u32=index_k_packed_u32,
                )
                table = natural_page_table(
                    layout, rr, offsets, idx_vpages_per_seq, cap=cap
                )
                if tie_after_idx_max_cap > 0 and cap <= tie_after_idx_max_cap:
                    positions = select_topk(q_idx_sel, w_idx_sel, nat, rr_sel, table)
                    # Avoid small-cap page_gather/StreamIdxTC contention: wait
                    # for StreamIdxTC before launching gather_latent on tiny caps.
                    positions, nope_t, rope_t = lax.optimization_barrier(
                        (positions, nope, rope)
                    )
                    g_nope, g_rope = gather_latent(
                        nope_t, rope_t, mla_page_ids[:cap], axis_name, read_pages
                    )
                else:
                    g_nope, g_rope = gather_latent(
                        nope, rope, mla_page_ids[:cap], axis_name, read_pages
                    )
                    positions = select_topk(q_idx_sel, w_idx_sel, nat, rr_sel, table)
                positions = _unalign_zigzag_r0_positions(
                    positions, rank, layout.split_rows, r0_shifted
                )
                flat = jnp.where(
                    real[:, None],
                    flat_rows(layout, positions, row_offsets, stride_pages=cap),
                    -1,
                )
                ql_head, ql_tail, qpe_head, qpe_tail = fused_qb(
                    qc_r, wqb_r, sqb_r, wuk_r, cs_r, s
                )
                if tie_lead_min_cap > 0 and cap >= tie_lead_min_cap:
                    g_nope, g_rope, ql_head, qpe_head, flat = lax.optimization_barrier(
                        (g_nope, g_rope, ql_head, qpe_head, flat)
                    )
                lead_partial, lead_lse = lead_attend(
                    ql_head, qpe_head, nope, rope, h_topk_r, mla_bt, sl, qsl, dist
                )
                if use_fused_wuv:
                    lead_v = jnp.swapaxes(
                        _merge_lead_partial_latent(lead_partial, lead_lse, axis_name),
                        0,
                        1,
                    )
                    attend_hm = getattr(tail_attend, "head_major_chunks")  # noqa: B009 -- optional kernel hook
                    out = attend_hm(
                        ql_tail, g_nope, g_rope, flat, rr.num_rows, qpe_tail
                    )
                    return out, lead_v, flat
                if use_3d_epilogue:
                    lead_v = _merge_lead_partial_to_v_hrv(
                        lead_partial, lead_lse, w_uv_all, axis_name, ql_tail.dtype
                    )
                else:
                    lead_v = _merge_lead_partial_to_v(
                        lead_partial, lead_lse, w_uv_all, axis_name, ql_tail.dtype
                    )
                out = tail_attend(ql_tail, g_nope, g_rope, flat, rr.num_rows, qpe_tail)
                out = (
                    _project_w_uv_hrv(out, w_uv_all)
                    if use_3d_epilogue
                    else _project_w_uv(out, w_uv_all)
                )
                return out, lead_v, flat

            return run

        out, lead_v, flat = _switch_on_cap(layout, jnp.sum(pages), branch)
        if has_oproj:
            assert w_o_local is not None and s_o_local is not None
            w_o_arr, s_o_arr = w_o_local, s_o_local
            if use_fused_wuv:
                rank_vec = jnp.asarray(
                    rank_arg[0] if rank_arg else jnp.reshape(rank, (1,)),
                    dtype=jnp.int32,
                )
                projected = _project_wuv_hm_pair_halves_local(
                    layout,
                    out,
                    lead_v,
                    w_uv_all,
                    rank_vec,
                    w_o_arr,
                    s_o_arr,
                    axis_name,
                    num_valid_own_rows=rr.num_rows,
                )
            elif use_3d_epilogue:
                rank_vec = jnp.asarray(
                    rank_arg[0] if rank_arg else jnp.reshape(rank, (1,)),
                    dtype=jnp.int32,
                )
                projected = _project_hrv_pair_halves_local(
                    layout, out, lead_v, rank_vec, w_o_arr, s_o_arr, axis_name
                )
            else:
                projected = _shift_and_project_local(
                    layout, out, lead_v, rank, w_o_arr, s_o_arr, axis_name
                )
            return projected, flat
        return out, lead_v, flat

    extra: tuple[jax.Array, ...] = () if rank_ids is None else (rank_ids,)
    if has_oproj:
        assert w_o is not None and w_o_scale is not None
        extra = (*extra, w_o, w_o_scale)
    w_uv_spec = rep if w_uv_replicated else sharded
    out_specs = (sharded, sharded) if has_oproj else (sharded, rep, sharded)
    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            sharded,
            sharded,
            sharded,
            rep,
            sharded,
            sharded,
            sharded,
            rep,
            rep,
            rep,
            sharded,
            sharded,
            rep,
            rep,
            rep,
            rep,
            w_uv_spec,
        )
        + (sharded,) * len(extra),
        out_specs=out_specs,
        check_vma=False,
    )(
        index_cache,
        q_idx,
        w_idx,
        idx_block_tables,
        kv_cache_nope,
        kv_cache_rope,
        q_c_own,
        w_qb,
        s_qb,
        w_uk_t,
        cos_sin,
        head_topk,
        mla_block_tables,
        seq_lens,
        query_start_loc,
        request_distribution,
        w_uv,
        *extra,
    )


def unified_indexer_in_branch_qb_mla(
    layout: TokenParallelLayout,
    mesh: Mesh,
    axis_name: str,
    index_cache: jax.Array,
    q_idx: jax.Array,
    w_idx: jax.Array,
    idx_block_tables: jax.Array,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    q_c_own: jax.Array,
    w_qb: jax.Array,
    s_qb: jax.Array,
    w_uk_t: jax.Array,
    cos_sin: jax.Array,
    head_topk: jax.Array,
    mla_block_tables: jax.Array,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    *,
    select_topk: SelectTopkFn,
    fused_qb: Callable[..., tuple[jax.Array, jax.Array, jax.Array, jax.Array]],
    lead_attend: Callable[..., tuple[jax.Array, jax.Array]],
    tail_attend: Callable[..., jax.Array],
    w_uv: jax.Array,
    num_tokens: int,
    rank_ids: jax.Array | None = None,
    read_pages: tuple[ReadPagesFn, ReadPagesFn] = (take_pages, take_pages),
    w_uv_replicated: bool = False,
    tie_after_idx_max_cap: int = 0,
    tie_lead_min_cap: int = 0,
    index_k_packed_u32: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run indexer top-k, in-branch `q_b`, lead MLA, and tail MLA in one cap switch.

    Returns `(own_v [dcp*R, H, V], lead_v [S, H, V], tail_flat_topk [dcp*R, K])`.
    """
    own_v, lead_v, flat = _unified_indexer_in_branch_qb_core(
        layout,
        mesh,
        axis_name,
        index_cache,
        q_idx,
        w_idx,
        idx_block_tables,
        kv_cache_nope,
        kv_cache_rope,
        q_c_own,
        w_qb,
        s_qb,
        w_uk_t,
        cos_sin,
        head_topk,
        mla_block_tables,
        seq_lens,
        query_start_loc,
        request_distribution,
        select_topk=select_topk,
        fused_qb=fused_qb,
        lead_attend=lead_attend,
        tail_attend=tail_attend,
        w_uv=w_uv,
        num_tokens=num_tokens,
        rank_ids=rank_ids,
        read_pages=read_pages,
        w_uv_replicated=w_uv_replicated,
        tie_after_idx_max_cap=tie_after_idx_max_cap,
        tie_lead_min_cap=tie_lead_min_cap,
        index_k_packed_u32=index_k_packed_u32,
    )
    return own_v, lead_v, flat


def unified_indexer_in_branch_qb_mla_oproj(
    layout: TokenParallelLayout,
    mesh: Mesh,
    axis_name: str,
    index_cache: jax.Array,
    q_idx: jax.Array,
    w_idx: jax.Array,
    idx_block_tables: jax.Array,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    q_c_own: jax.Array,
    w_qb: jax.Array,
    s_qb: jax.Array,
    w_uk_t: jax.Array,
    cos_sin: jax.Array,
    head_topk: jax.Array,
    mla_block_tables: jax.Array,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    *,
    select_topk: SelectTopkFn,
    fused_qb: Callable[..., tuple[jax.Array, jax.Array, jax.Array, jax.Array]],
    lead_attend: Callable[..., tuple[jax.Array, jax.Array]],
    tail_attend: Callable[..., jax.Array],
    w_uv: jax.Array,
    w_o: jax.Array,
    w_o_scale: jax.Array,
    num_tokens: int,
    rank_ids: jax.Array | None = None,
    read_pages: tuple[ReadPagesFn, ReadPagesFn] = (take_pages, take_pages),
    w_uv_replicated: bool = False,
    tie_after_idx_max_cap: int = 0,
    tie_lead_min_cap: int = 0,
    epilogue_3d: bool = True,
    index_k_packed_u32: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Run indexer top-k, in-branch `q_b`, MLA, and `o_proj` in one `shard_map`."""
    attn_out, flat = _unified_indexer_in_branch_qb_core(
        layout,
        mesh,
        axis_name,
        index_cache,
        q_idx,
        w_idx,
        idx_block_tables,
        kv_cache_nope,
        kv_cache_rope,
        q_c_own,
        w_qb,
        s_qb,
        w_uk_t,
        cos_sin,
        head_topk,
        mla_block_tables,
        seq_lens,
        query_start_loc,
        request_distribution,
        select_topk=select_topk,
        fused_qb=fused_qb,
        lead_attend=lead_attend,
        tail_attend=tail_attend,
        w_uv=w_uv,
        num_tokens=num_tokens,
        rank_ids=rank_ids,
        read_pages=read_pages,
        w_uv_replicated=w_uv_replicated,
        tie_after_idx_max_cap=tie_after_idx_max_cap,
        tie_lead_min_cap=tie_lead_min_cap,
        w_o=w_o,
        w_o_scale=w_o_scale,
        epilogue_3d=epilogue_3d,
        index_k_packed_u32=index_k_packed_u32,
    )
    return attn_out, flat

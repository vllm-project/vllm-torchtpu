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
"""Context-parallel prefill attention on batched RPA, one prompt per step.

The prompt's T tokens are cut into 2 * world page-aligned pieces, and rank r
owns pieces r and 2 * world - 1 - r ("zigzag"), so every rank attends to the
same number of causal (query, key) pairs. Each rank runs batched RPA for its two
pieces as two sub-requests whose queries sit at their absolute positions:

* full attention: the K/V pages of all ranks, all-gathered in rank order, are
  the cache, and a page table that inverts the zigzag maps each position's page
  to where it landed, so the gathered buffer is never reordered;
* sliding window: each piece's cache is the piece's own pages after a halo of
  the pages that precede it, which the rank holding the preceding piece sends.

All K/V, the pieces' own tokens included, is read from that cache
(`skip_kv_update`). The serving cache is written separately: each rank keeps
the KV heads it owns under tensor parallelism (`tp_kv_heads`), every position
of a full-attention layer from the gathered buffer and, of a sliding-window
layer, the pages the next decode's window reaches, which the ranks holding
them send.

Positions past the prompt's length are padding. Causality keeps them out of
every real query's keys, and the cache writes skip their pages.

A rank is a Python int or a traced scalar (`jax.lax.axis_index` under
`shard_map`). This module holds the page arithmetic; moving pages between ranks
is the caller's.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, utils, wrapper

Rank = int | jax.Array


@dataclasses.dataclass(frozen=True)
class ZigzagPlan:
    """Static zigzag split of a T-token prompt over `world` ranks.

    Attributes:
        num_tokens: T, the step's (bucket) token count; positions 0..T-1.
        world: Number of ranks.
        page_size: KV page size; every piece is a whole number of pages.
    """

    num_tokens: int
    world: int
    page_size: int

    def __post_init__(self):
        unit = 2 * self.world * self.page_size
        if self.num_tokens % unit:
            raise ValueError(
                f"Zigzag CP needs the token count ({self.num_tokens}) to be a "
                f"multiple of 2 * world * page_size ({unit})."
            )

    @property
    def num_pieces(self) -> int:
        return 2 * self.world

    @property
    def piece_tokens(self) -> int:
        return self.num_tokens // self.num_pieces

    @property
    def piece_pages(self) -> int:
        return self.piece_tokens // self.page_size

    @property
    def num_pages(self) -> int:
        return self.num_tokens // self.page_size

    def rank_pieces(self, rank: Rank) -> tuple[Rank, Rank]:
        """The two pieces rank `rank` owns, in the order its rows hold them."""
        if isinstance(rank, int) and not 0 <= rank < self.world:
            raise ValueError(f"rank {rank} outside world {self.world}")
        return rank, self.num_pieces - 1 - rank

    def piece_owner(self, piece: int) -> tuple[int, int]:
        """(rank, slot) holding `piece`: slot 0 is a rank's first piece."""
        if piece < self.world:
            return piece, 0
        return self.num_pieces - 1 - piece, 1

    def rank_slices(self, rank: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """Token ranges [start, end) of the rank's two pieces, in row order."""
        p = self.piece_tokens
        return tuple((i * p, (i + 1) * p) for i in self.rank_pieces(rank))

    def gathered_page_table(self) -> np.ndarray:
        """Position page -> page of the rank-major all-gather of every rank's
        [first piece | second piece] pages."""
        pp = self.piece_pages
        table = np.empty(self.num_pages, np.int32)
        for page in range(self.num_pages):
            rank, slot = self.piece_owner(page // pp)
            table[page] = rank * 2 * pp + slot * pp + page % pp
        return table

    def gathered_position_pages(self) -> np.ndarray:
        """Page of the all-gather -> the position page it holds."""
        return np.argsort(self.gathered_page_table()).astype(np.int32)

    def sliding_page_tables(self, rank: Rank, halo_pages: int) -> jax.Array:
        """Per sub-request position page -> page of the rank's local buffer
        [halo | first piece | halo | second piece]; [2, num_pages].

        Pages a piece's window never reaches map to its first halo page, whose
        contents are finite and fully masked wherever a KV block reaches them.
        """
        if halo_pages > self.piece_pages:
            raise ValueError(
                f"A halo of {halo_pages} pages spans more than one piece of "
                f"{self.piece_pages} pages."
            )
        pp = self.piece_pages
        pieces = jnp.stack(self.rank_pieces(rank)).astype(jnp.int32)
        local = jnp.arange(self.num_pages, dtype=jnp.int32) - (
            pieces[:, None] * pp - halo_pages
        )
        base = jnp.arange(2, dtype=jnp.int32)[:, None] * (halo_pages + pp)
        return base + jnp.clip(local, 0, halo_pages + pp - 1)

    def sub_request_metadata(self, rank: Rank) -> tuple[jax.Array, ...]:
        """(cu_q_lens, kv_lens, distribution) of the rank's two sub-requests."""
        p = self.piece_tokens
        cu_q_lens = jnp.array([0, p, 2 * p], jnp.int32)
        kv_lens = (jnp.stack(self.rank_pieces(rank)).astype(jnp.int32) + 1) * p
        distribution = jnp.array([0, 0, 2], jnp.int32)
        return cu_q_lens, kv_lens, distribution


def make_zigzag_plan(num_tokens: int, world: int, page_size: int) -> ZigzagPlan:
    return ZigzagPlan(num_tokens=num_tokens, world=world, page_size=page_size)


def pack_kv_rows(k: jax.Array, v: jax.Array) -> jax.Array:
    """[n, kv_heads, d] K and V -> [n, 2 * kv_heads * d] rows K_0 V_0 K_1 V_1 ..."""
    n, kv_heads, head_dim = k.shape
    return jnp.concatenate([k, v], axis=-1).reshape(n, 2 * kv_heads * head_dim)


def kv_pages_from_rows(rows: jax.Array, head_dim: int, page_size: int) -> jax.Array:
    """`pack_kv_rows` rows -> HEAD_ALONG_SUBLANE pages
    [n / page_size, page_size, ceil(2 kv_heads / p), p, head_dim], p the dtype
    packing: the layout `wrapper.prepare_inputs` gives new K/V, cut into pages.
    """
    n = rows.shape[0]
    if n % page_size:
        raise ValueError(f"{n} rows are not whole pages of {page_size}")
    heads_x2 = rows.shape[1] // head_dim
    packing = utils.get_dtype_packing(rows.dtype)
    slots = utils.align_to(heads_x2, packing)
    kv = rows.reshape(n, heads_x2, head_dim)
    if slots != heads_x2:
        kv = jnp.pad(kv, ((0, 0), (0, slots - heads_x2), (0, 0)))
    return kv.reshape(n // page_size, page_size, slots // packing, packing, head_dim)


def pack_kv_pages(k: jax.Array, v: jax.Array, page_size: int) -> jax.Array:
    """[n, kv_heads, d] K and V -> `kv_pages_from_rows` pages."""
    return kv_pages_from_rows(pack_kv_rows(k, v), k.shape[-1], page_size)


def num_halo_pages(sliding_window: int, page_size: int) -> int:
    """Pages before a piece that the window of its first query reaches."""
    return utils.align_to(sliding_window - 1, page_size) // page_size


def halo_permutations(world: int) -> tuple[list[tuple[int, int]], ...]:
    """(source, target) rank pairs that send each rank's piece tails on to the
    rank holding the next piece: first-piece tails go up (r -> r + 1),
    second-piece tails down (r -> r - 1). Both wrap around; `sliding_local_pages`
    ignores what the wrap delivers."""
    up = [(r, (r + 1) % world) for r in range(world)]
    down = [(r, (r - 1) % world) for r in range(world)]
    return up, down


def piece_tails(
    own_pages: jax.Array, plan: ZigzagPlan, halo_pages: int
) -> tuple[jax.Array, jax.Array]:
    """The last `halo_pages` pages of the rank's first and second piece."""
    pp = plan.piece_pages
    return own_pages[pp - halo_pages : pp], own_pages[2 * pp - halo_pages :]


def sliding_local_pages(
    own_pages: jax.Array,
    halo_first: jax.Array,
    halo_second: jax.Array,
    plan: ZigzagPlan,
    rank: Rank,
) -> jax.Array:
    """The rank's [halo | first piece | halo | second piece] page buffer.

    Args:
        own_pages: [2 * piece_pages, ...] pages of the rank's two pieces.
        halo_first: Tail of the piece before the first piece, as the
            `halo_permutations` up-shift delivers it (unused on rank 0, whose
            first piece starts the prompt).
        halo_second: Tail of the piece before the second piece, as the
            down-shift delivers it. The last rank's second piece follows its own
            first piece, whose tail is used instead.
        plan: The step's zigzag plan.
        rank: This rank.
    """
    pp, halo_pages = plan.piece_pages, halo_first.shape[0]
    first, second = own_pages[:pp], own_pages[pp:]
    own_tail = first[pp - halo_pages :]
    last = plan.world - 1
    if isinstance(rank, int):
        halo_second = own_tail if rank == last else halo_second
    else:
        halo_second = jnp.where(rank == last, own_tail, halo_second)
    return jnp.concatenate([halo_first, first, halo_second, second])


def cp_prefill_attention(
    q: jax.Array,
    kv_pages: jax.Array,
    page_tables: jax.Array,
    plan: ZigzagPlan,
    rank: Rank,
    *,
    num_kv_heads: int,
    sliding_window: int | None = None,
    fp8_compute: bool = False,
    sm_scale: float = 1.0,
    k_scale: float | None = None,
    v_scale: float | None = None,
    **rpa_kwargs,
) -> jax.Array:
    """Attention of the rank's two pieces against `kv_pages`.

    Args:
        q: [2 * piece_tokens, num_q_heads, head_dim], the rank's rows (first piece,
            then second).
        kv_pages: [pages, page_size, slots, packing, head_dim] packed K/V
            (`kv_pages_from_rows`): the gathered buffer for full attention, the
            `sliding_local_pages` buffer for a sliding window.
        page_tables: [num_pages] (shared by both pieces) or [2, num_pages].
        plan: The step's zigzag plan.
        rank: This rank.
        num_kv_heads: KV heads in `kv_pages`.
        sliding_window: The layer's window, None for full attention.
        fp8_compute: Run QK^T and PV as fp8 x fp8 (`ragged_paged_attention`).
        sm_scale, k_scale, v_scale: As for `ragged_paged_attention`.
        **rpa_kwargs: Further static `ragged_paged_attention` arguments.

    Returns:
        [2 * piece_tokens, num_q_heads, head_dim] attention output.
    """
    cu_q_lens, kv_lens, distribution = plan.sub_request_metadata(rank)
    tables = jnp.broadcast_to(jnp.asarray(page_tables), (2, plan.num_pages))
    head_dim = kv_pages.shape[-1]
    # With skip_kv_update the kernel reads every key from the cache; new K/V only
    # fill the operand slots.
    new_kv = jnp.zeros((q.shape[0], num_kv_heads, head_dim), kv_pages.dtype)
    out, _ = wrapper.ragged_paged_attention(
        q,
        new_kv,
        new_kv,
        kv_pages,
        kv_lens,
        tables.reshape(-1),
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        k_scale=k_scale,
        v_scale=v_scale,
        skip_kv_update=True,
        # The decode segment is always empty.
        skip_empty_launch=True,
        fp8_compute=fp8_compute,
        **rpa_kwargs,
    )
    return out


def tp_kv_heads(num_kv_heads: int, tp_size: int, tp_rank: Rank) -> tuple[Rank, int]:
    """(first KV head, KV head count) that tensor-parallel rank `tp_rank` caches.

    With more ranks than KV heads, each head is replicated on consecutive ranks.
    """
    if tp_size >= num_kv_heads:
        if tp_size % num_kv_heads:
            raise ValueError(f"{tp_size} ranks do not replicate {num_kv_heads} heads")
        return tp_rank // (tp_size // num_kv_heads), 1
    if num_kv_heads % tp_size:
        raise ValueError(f"{num_kv_heads} KV heads do not split over {tp_size} ranks")
    per_rank = num_kv_heads // tp_size
    return tp_rank * per_rank, per_rank


def kv_head_slots(pages: jax.Array, first_head: Rank, num_heads: int) -> jax.Array:
    """[..., page_size, slot_rows, p, d] pages of all KV heads -> the
    [..., page_size, 2 * num_heads, d] K/V slots of heads
    [first_head, first_head + num_heads)."""
    *lead, slot_rows, packing, head_dim = pages.shape
    flat = pages.reshape(*lead, slot_rows * packing, head_dim)
    return jax.lax.dynamic_slice_in_dim(
        flat, 2 * first_head, 2 * num_heads, axis=len(lead)
    )


def cache_pages_from_slots(slots: jax.Array, kv_layout: configs.KVLayout) -> jax.Array:
    """[pages, page_size, 2 * kv_heads, d] K/V slots -> pages of a paged KV cache
    in `kv_layout`, as `wrapper.get_kv_cache_shape` lays them out."""
    num_pages, page_size, heads_x2, head_dim = slots.shape
    packing = utils.get_dtype_packing(slots.dtype)
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        # [pages, 2 * kv_heads, d / p, p, page_size]
        return slots.reshape(
            num_pages, page_size, heads_x2, head_dim // packing, packing
        ).transpose(0, 2, 3, 4, 1)
    aligned = utils.align_to(heads_x2, packing)
    if aligned != heads_x2:
        slots = jnp.pad(slots, ((0, 0), (0, 0), (0, aligned - heads_x2), (0, 0)))
    return slots.reshape(num_pages, page_size, aligned // packing, packing, head_dim)


def write_kv_pages(
    kv_cache: jax.Array, pages: jax.Array, dst_pages: jax.Array
) -> jax.Array:
    """kv_cache[dst_pages[i]] = pages[i]; an index past the cache drops page i."""
    return kv_cache.at[dst_pages].set(pages, mode="drop")


def full_attention_dst_pages(
    plan: ZigzagPlan, block_table: jax.Array, seq_len: jax.Array, num_cache_pages: int
) -> jax.Array:
    """Cache page of each all-gather page under the request's `block_table`;
    `num_cache_pages` (dropped by `write_kv_pages`) for pages past `seq_len`."""
    position_pages = plan.gathered_position_pages()
    dst = jnp.take(block_table, position_pages, mode="clip")
    return jnp.where(position_pages * plan.page_size < seq_len, dst, num_cache_pages)


def window_position_pages(
    seq_len: jax.Array, sliding_window: int, page_size: int
) -> tuple[jax.Array, jax.Array]:
    """(position pages, valid) that the window of the token after a
    `seq_len`-token prompt reaches; a fixed count of
    `num_halo_pages(sliding_window, page_size) + 1` pages."""
    count = num_halo_pages(sliding_window, page_size) + 1
    first = jnp.maximum(seq_len - (sliding_window - 1), 0) // page_size
    pages = first + jnp.arange(count, dtype=jnp.int32)
    return pages, pages * page_size < seq_len


def window_slots_for_ranks(
    own_pages: jax.Array,
    plan: ZigzagPlan,
    rank: Rank,
    position_pages: jax.Array,
    num_kv_heads: int,
) -> jax.Array:
    """What this rank sends each rank in an all-to-all of the window: per
    destination, the K/V slots of the destination's KV heads (`tp_kv_heads`,
    no replication) in the window pages this rank holds; the other pages carry
    nothing (`window_slots_from_ranks` takes each page from its holder).
    [world, len(position_pages), page_size, 2 * heads per rank, d]."""
    if num_kv_heads % plan.world:
        raise ValueError(
            f"{num_kv_heads} KV heads do not split over {plan.world} ranks"
        )
    pp = plan.piece_pages
    first, second = plan.rank_pieces(rank)
    piece = position_pages // pp
    local = jnp.where(
        piece == first, position_pages - first * pp, position_pages - second * pp + pp
    )
    pages = jnp.take(own_pages, jnp.clip(local, 0, 2 * pp - 1), axis=0)
    per_rank = num_kv_heads // plan.world
    return jnp.stack(
        [kv_head_slots(pages, r * per_rank, per_rank) for r in range(plan.world)]
    )


def window_slots_from_ranks(
    received: jax.Array, plan: ZigzagPlan, position_pages: jax.Array
) -> jax.Array:
    """The window from what every rank sent this one (`window_slots_for_ranks`
    after an all-to-all, sources along axis 0): each page from the rank holding
    it."""
    piece = position_pages // plan.piece_pages
    owner = jnp.where(piece < plan.world, piece, plan.num_pieces - 1 - piece)
    return received[owner, jnp.arange(position_pages.shape[0])]

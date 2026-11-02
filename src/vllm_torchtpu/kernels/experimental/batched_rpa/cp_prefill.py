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

Full attention reads all K/V, the pieces' own tokens included, from the
gathered buffer (`skip_kv_update`); a sliding-window layer reads only the halos
from its buffer and takes the pieces' own K/V as the kernel's new tokens. The
serving cache is written separately: each rank keeps
the KV heads it owns under tensor parallelism (`tp_kv_heads`), every position
of a full-attention layer from the gathered buffer and, of a sliding-window
layer, the pages the next decode's window reaches, which the ranks holding
them send.

Positions past the prompt's length are padding. Causality keeps them out of
every real query's keys, and the cache writes skip their pages. A sliding-window
layer's earlier pages are left unwritten, so the cache blocks of such a prompt
must not be reused through prefix caching.

A rank is a Python int or a traced scalar (`jax.lax.axis_index` under
`shard_map`). This module holds the page arithmetic and what one layer runs
between the collectives; of those, only the halo permutes (`exchange_halos`,
under the caller's `shard_map`) are here.
"""

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

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
        [first piece's halo | second piece's halo | scratch]; [2, num_pages].

        The pieces' own pages map to the scratch page: the kernel reads their
        keys from its new-token operand and only writes them there. Pages
        before the halo map to its first page, finite and fully masked
        wherever a KV block reaches them.
        """
        if not 1 <= halo_pages <= self.piece_pages:
            raise ValueError(
                f"A halo of {halo_pages} pages is not 1 to one piece of "
                f"{self.piece_pages} pages."
            )
        pieces = jnp.stack(self.rank_pieces(rank)).astype(jnp.int32)
        local = jnp.arange(self.num_pages, dtype=jnp.int32) - (
            pieces[:, None] * self.piece_pages - halo_pages
        )
        base = jnp.arange(2, dtype=jnp.int32)[:, None] * halo_pages
        halo = base + jnp.clip(local, 0, halo_pages - 1)
        return jnp.where(local < halo_pages, halo, 2 * halo_pages)

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


def pack_piece_tails(
    keys: jax.Array, values: jax.Array, halo_pages: int, page_size: int
) -> jax.Array:
    """The last `halo_pages` pages of the rank's first and second piece, packed
    (`pack_kv_pages`); `keys` and `values` are the rank's rows."""
    piece, halo = keys.shape[0] // 2, halo_pages * page_size
    return jnp.concatenate(
        [
            pack_kv_pages(keys[end - halo : end], values[end - halo : end], page_size)
            for end in (piece, 2 * piece)
        ]
    )


def exchange_halos(tails: jax.Array, axis_name: str, world: int) -> jax.Array:
    """[first piece's halo | second piece's halo] of this rank, inside a
    `shard_map` over `axis_name`: the `halo_permutations` shifts of every rank's
    `pack_piece_tails`."""
    halo_pages = tails.shape[0] // 2
    up, down = halo_permutations(world)
    return jnp.concatenate(
        [
            jax.lax.ppermute(tails[:halo_pages], axis_name, up),
            jax.lax.ppermute(tails[halo_pages:], axis_name, down),
        ]
    )


def sliding_local_pages(
    halos: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    plan: ZigzagPlan,
    rank: Rank,
) -> jax.Array:
    """The rank's [first piece's halo | second piece's halo | scratch] page
    buffer for `cp_prefill_attention` with the pieces' K/V as new tokens
    (`ZigzagPlan.sliding_page_tables`).

    `halos` is what `exchange_halos` delivers (fp8). Rank 0's first piece starts
    the prompt, so its first halo goes unread; the last rank's second piece
    follows its own first piece, whose tail (from `keys`, `values`, the rank's
    rows) is used instead of what the permute delivered.
    """
    halo_pages, page_size = halos.shape[0] // 2, halos.shape[1]
    halo_first, halo_second = halos[:halo_pages], halos[halo_pages:]
    piece, halo = plan.piece_tokens, halo_pages * page_size
    own_tail = pack_kv_pages(
        keys[piece - halo : piece], values[piece - halo : piece], page_size
    )
    last = plan.world - 1
    if isinstance(rank, int):
        halo_second = own_tail if rank == last else halo_second
    else:
        halo_second = jnp.where(rank == last, own_tail, halo_second)
    scratch = jnp.zeros((1, *halos.shape[1:]), halos.dtype)
    return jnp.concatenate([halo_first, halo_second, scratch])


def cp_prefill_attention(
    q: jax.Array,
    kv_pages: jax.Array,
    page_tables: jax.Array,
    plan: ZigzagPlan,
    rank: Rank,
    *,
    num_kv_heads: int,
    keys: jax.Array | None = None,
    values: jax.Array | None = None,
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
        keys, values: [2 * piece_tokens, num_kv_heads, head_dim] K/V of the
            rank's rows, or None. Given, the pieces' own keys come from them and
            the kernel writes them into their pages of `kv_pages`; else every
            key, the pieces' own included, is read from `kv_pages`.
        sliding_window: The layer's window, None for full attention.
        fp8_compute: Run QK^T and PV as fp8 x fp8 (`ragged_paged_attention`).
        sm_scale, k_scale, v_scale: As for `ragged_paged_attention`.
        **rpa_kwargs: Further static `ragged_paged_attention` arguments.

    Returns:
        [2 * piece_tokens, num_q_heads, head_dim] attention output.

    `ragged_paged_attention` donates its cache argument, so outside a jit this
    consumes `kv_pages`.
    """
    cu_q_lens, kv_lens, distribution = plan.sub_request_metadata(rank)
    tables = jnp.broadcast_to(jnp.asarray(page_tables), (2, plan.num_pages))
    skip_kv_update = keys is None
    if skip_kv_update:
        # The kernel reads every key from the cache; new K/V only fill the
        # operand slots.
        shape = (q.shape[0], num_kv_heads, kv_pages.shape[-1])
        keys = values = jnp.zeros(shape, kv_pages.dtype)
    out, _ = wrapper.ragged_paged_attention(
        q,
        keys,
        values,
        kv_pages,
        kv_lens,
        tables.reshape(-1),
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        k_scale=k_scale,
        v_scale=v_scale,
        skip_kv_update=skip_kv_update,
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


def _copy_page_kernel(_count_ref, _src_ref, _dst_ref, page_ref, _cache_ref, out_ref):
    out_ref[...] = page_ref[...]


def write_kv_pages(
    kv_cache: jax.Array,
    pages: jax.Array,
    src_pages: jax.Array,
    dst_pages: jax.Array,
    count: jax.Array,
) -> jax.Array:
    """kv_cache[dst_pages[i]] = pages[src_pages[i]] for i < max(count, 1).

    A Pallas copy into the aliased cache, so it costs the pages written; an XLA
    scatter would rewrite the whole cache. Steps past `count` repeat the last
    page, which re-fetches and writes back nothing. A count of 0 (a dummy step
    with no tokens) still copies page 0, into the null block a dummy step's
    zero-filled block table names.
    """
    block = (1, *kv_cache.shape[1:])
    rest = (0,) * (kv_cache.ndim - 1)

    def step(i, count_ref):
        return jnp.maximum(jnp.minimum(i, count_ref[0] - 1), 0)

    def page_index(i, count_ref, src_ref, dst_ref):
        return (src_ref[step(i, count_ref)], *rest)

    def cache_index(i, count_ref, src_ref, dst_ref):
        return (dst_ref[step(i, count_ref)], *rest)

    return pl.pallas_call(
        _copy_page_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(dst_pages.shape[0],),
            in_specs=[
                pl.BlockSpec(block, page_index),
                # Aliased with the output and never read.
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(block, cache_index),
        ),
        out_shape=jax.ShapeDtypeStruct(kv_cache.shape, kv_cache.dtype),
        input_output_aliases={4: 0},
    )(
        jnp.reshape(count, (1,)).astype(jnp.int32),
        src_pages.astype(jnp.int32),
        dst_pages.astype(jnp.int32),
        pages,
        kv_cache,
    )


def full_attention_page_writes(
    plan: ZigzagPlan, block_table: jax.Array, seq_len: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """`write_kv_pages` (src, dst, count) that cache every all-gather page of a
    position below `seq_len` under the request's `block_table`."""
    src = jnp.asarray(plan.gathered_page_table())
    dst = jnp.take(block_table, jnp.arange(plan.num_pages), mode="clip")
    count = jnp.minimum(
        utils.align_to(seq_len, plan.page_size) // plan.page_size, plan.num_pages
    )
    return src, dst, count


def window_position_pages(
    seq_len: jax.Array, sliding_window: int, page_size: int
) -> tuple[jax.Array, jax.Array]:
    """(position pages, count): the first `count` of the
    `num_halo_pages(sliding_window, page_size) + 1` position pages are those the
    window of the token after a `seq_len`-token prompt reaches."""
    num = num_halo_pages(sliding_window, page_size) + 1
    first = jnp.maximum(seq_len - (sliding_window - 1), 0) // page_size
    pages = first + jnp.arange(num, dtype=jnp.int32)
    last = utils.align_to(seq_len, page_size) // page_size
    return pages, jnp.minimum(last - first, num)


def window_page_writes(
    block_table: jax.Array, position_pages: jax.Array, count: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """`write_kv_pages` (src, dst, count) that cache the window pages
    (`window_position_pages`, in that order) under the request's `block_table`."""
    src = jnp.arange(position_pages.shape[0], dtype=jnp.int32)
    return src, jnp.take(block_table, position_pages, mode="clip"), count


def window_slots_for_ranks(
    keys: jax.Array,
    values: jax.Array,
    plan: ZigzagPlan,
    rank: Rank,
    position_pages: jax.Array,
    num_kv_heads: int,
) -> jax.Array:
    """What this rank sends each rank in an all-to-all of the window: per
    destination, the K/V slots of the destination's `tp_kv_heads` in the window
    pages this rank holds; the other pages carry nothing
    (`window_slots_from_ranks` takes each page from its holder). `keys` and
    `values` are the rank's rows.
    [world, len(position_pages), page_size, 2 * heads per rank, d]."""
    pp, ps = plan.piece_pages, plan.page_size
    first, second = plan.rank_pieces(rank)
    piece = position_pages // pp
    local = jnp.where(
        piece == first, position_pages - first * pp, position_pages - second * pp + pp
    )
    rows = jnp.clip(local, 0, 2 * pp - 1)[:, None] * ps + jnp.arange(ps)
    pages = pack_kv_pages(
        jnp.take(keys, rows.reshape(-1), axis=0),
        jnp.take(values, rows.reshape(-1), axis=0),
        ps,
    )
    return jnp.stack(
        [
            kv_head_slots(pages, *tp_kv_heads(num_kv_heads, plan.world, r))
            for r in range(plan.world)
        ]
    )


def window_page_owners(plan: ZigzagPlan, position_pages: jax.Array) -> jax.Array:
    """The rank holding each position page."""
    piece = position_pages // plan.piece_pages
    return jnp.where(piece < plan.world, piece, plan.num_pieces - 1 - piece)


def window_slots_from_ranks(received: jax.Array, owners: jax.Array) -> jax.Array:
    """The window from what every rank sent this one (`window_slots_for_ranks`
    after an all-to-all, sources along axis 0): each page from its
    `window_page_owners` rank."""
    return received[owners, jnp.arange(owners.shape[0])]


CacheView = Callable[
    [jax.Array], tuple[jax.Array, int, Callable[[jax.Array], jax.Array]]
]


@dataclasses.dataclass(frozen=True)
class LayerConfig:
    """One attention layer of a context-parallel prefill on `world` ranks.

    `num_heads` and `num_kv_heads` count every rank's heads; the serving cache
    holds `tp_kv_heads` of them.
    """

    rank: int
    world: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    sliding_window: int | None
    sm_scale: float
    soft_cap: float | None
    k_scale: float
    v_scale: float
    fp8_compute: bool
    kv_layout: configs.KVLayout
    sliding_prefill_max_bkv: int | None = None
    fence_output: bool = False

    @property
    def tp_kv_heads(self) -> tuple[int, int]:
        return tp_kv_heads(self.num_kv_heads, self.world, self.rank)

    def plan(self, num_rows: int, page_size: int) -> ZigzagPlan:
        """The plan of a step in which every rank holds `num_rows` rows."""
        return make_zigzag_plan(num_rows * self.world, self.world, page_size)

    def rpa_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = dict(
            num_kv_heads=self.num_kv_heads,
            sm_scale=self.sm_scale,
            soft_cap=self.soft_cap,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
            fp8_compute=self.fp8_compute,
        )
        if self.sliding_window is not None:
            kwargs["sliding_window"] = self.sliding_window
            if self.sliding_prefill_max_bkv:
                kwargs["sliding_prefill_max_bkv"] = self.sliding_prefill_max_bkv
        if self.fence_output:
            kwargs["fence_output"] = True
        return kwargs


def _cache_pages(kv_cache: jax.Array, cfg: LayerConfig, cache_view: CacheView):
    """(native pages, native pages per block, inverse view, page size)."""
    native, page_stride, restore = cache_view(kv_cache)
    page_axis = 4 if cfg.kv_layout == configs.KVLayout.SEQ_ALONG_LANE else 1
    return native, page_stride, restore, native.shape[page_axis]


def _check_page_size(plan_page_size: int, cache_page_size: int) -> None:
    if plan_page_size != cache_page_size:
        raise ValueError(
            f"The zigzag plan's pages ({plan_page_size} tokens) are not the KV "
            f"cache's ({cache_page_size})."
        )


def _request_block_table(block_tables: jax.Array, seq_lens: jax.Array) -> jax.Array:
    """Block table row of the step's one request (batch index 0)."""
    return block_tables[: block_tables.shape[0] // seq_lens.shape[0]]


def full_attention_layer(
    kv_cache: jax.Array,
    query: jax.Array,
    kv_pages: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    *,
    cfg: LayerConfig,
    cache_view: CacheView,
) -> tuple[jax.Array, jax.Array]:
    """A full-attention layer: attention of the rank's rows over the
    all-gathered pages, and the cache with this rank's heads of every prompt
    position.

    Args:
        kv_cache: The layer's serving cache; `cache_view` gives its native pages.
        query: [rows, num_heads, head_dim].
        kv_pages: Every rank's `pack_kv_pages` pages in rank order, as int8.
        seq_lens, block_tables: The step's attention metadata (one request).
        cfg: The layer.
        cache_view: `kv_cache` -> (native pages, pages per block, inverse view).

    Returns:
        (the new kv_cache, [rows, num_heads, head_dim] attention output).
    """
    native, page_stride, restore, page_size = _cache_pages(kv_cache, cfg, cache_view)
    _check_page_size(kv_pages.shape[1], page_size)
    plan = cfg.plan(query.shape[0], page_size)
    pages = jax.lax.bitcast_convert_type(kv_pages, native.dtype)
    out = cp_prefill_attention(
        query, pages, plan.gathered_page_table(), plan, cfg.rank, **cfg.rpa_kwargs()
    )
    first, count = cfg.tp_kv_heads
    mine = cache_pages_from_slots(kv_head_slots(pages, first, count), cfg.kv_layout)
    table = _request_block_table(block_tables, seq_lens) * page_stride
    writes = full_attention_page_writes(plan, table, seq_lens[0])
    return restore(write_kv_pages(native, mine, *writes)), out


def sliding_attention_layer(
    query: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    halos: jax.Array,
    seq_lens: jax.Array,
    *,
    cfg: LayerConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """A sliding-window layer's attention, within [halo | piece].

    Args:
        query: [rows, num_heads, head_dim].
        keys, values: [rows, num_kv_heads, head_dim] fp8 K/V of the rank's rows.
        halos: `exchange_halos` pages, as int8.
        seq_lens: The step's sequence lengths (one request).
        cfg: The layer.

    Returns:
        (attention output, the int8 `window_slots_for_ranks` blocks for an
        all-to-all, the `window_page_owners` of the window pages).
    """
    if cfg.sliding_window is None:
        raise ValueError("sliding_attention_layer needs a sliding window")
    page_size = halos.shape[1]
    plan = cfg.plan(query.shape[0], page_size)
    halo_pages = num_halo_pages(cfg.sliding_window, page_size)
    halos = jax.lax.bitcast_convert_type(halos, keys.dtype)
    local = sliding_local_pages(halos, keys, values, plan, cfg.rank)
    tables = plan.sliding_page_tables(cfg.rank, halo_pages)
    out = cp_prefill_attention(
        query,
        local,
        tables,
        plan,
        cfg.rank,
        keys=keys,
        values=values,
        **cfg.rpa_kwargs(),
    )
    position_pages, _ = window_position_pages(
        seq_lens[0], cfg.sliding_window, page_size
    )
    send = window_slots_for_ranks(
        keys, values, plan, cfg.rank, position_pages, cfg.num_kv_heads
    )
    owners = window_page_owners(plan, position_pages)
    return out, jax.lax.bitcast_convert_type(send, jnp.int8), owners


def window_write_layer(
    kv_cache: jax.Array,
    received: jax.Array,
    owners: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    *,
    cfg: LayerConfig,
    cache_view: CacheView,
) -> jax.Array:
    """The cache with this rank's heads of the decode window, from the
    all-to-all of `sliding_attention_layer`'s blocks."""
    if cfg.sliding_window is None:
        raise ValueError("window_write_layer needs a sliding window")
    native, page_stride, restore, page_size = _cache_pages(kv_cache, cfg, cache_view)
    _check_page_size(received.shape[2], page_size)
    slots = window_slots_from_ranks(
        jax.lax.bitcast_convert_type(received, native.dtype), owners
    )
    position_pages, count = window_position_pages(
        seq_lens[0], cfg.sliding_window, page_size
    )
    table = _request_block_table(block_tables, seq_lens) * page_stride
    writes = window_page_writes(table, position_pages, count)
    mine = cache_pages_from_slots(slots, cfg.kv_layout)
    return restore(write_kv_pages(native, mine, *writes))

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

The page helpers take a rank as a Python int or a traced scalar
(`jax.lax.axis_index` under `shard_map`); the layer functions take it as a [1]
int32 array, which a torch caller passes as a device tensor so that every rank
runs one program. This module holds the page arithmetic and what one layer runs
around its one collective (an all-gather of `pack_kv_rows` rows, or of
`sliding_exchange_rows`), which is the caller's.
"""

import dataclasses
import functools
import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs,
    prologue,
    utils,
    wrapper,
)

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

    def check_halo_pages(self, halo_pages: int) -> None:
        """Refuses a halo that is not within the one piece before each piece."""
        if not 1 <= halo_pages <= self.piece_pages:
            raise ValueError(
                f"A halo of {halo_pages} pages must be between 1 and the piece's "
                f"{self.piece_pages} pages."
            )

    def sliding_page_tables(self, rank: Rank, halo_pages: int) -> jax.Array:
        """Per sub-request position page -> page of the rank's local buffer
        [first piece's halo | second piece's halo | scratch]; [2, num_pages].

        The pieces' own pages map to the scratch page: the kernel reads their
        keys from its new-token operand and only writes them there. Pages
        before the halo map to its first page, finite and fully masked
        wherever a KV block reaches them.
        """
        self.check_halo_pages(halo_pages)
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


def halo_neighbours(rank: Rank, world: int) -> tuple[Rank, Rank]:
    """(rank holding the piece before this rank's first piece, rank holding the
    piece before its second piece): this rank's halos are the first-piece tail
    of the one and the second-piece tail of the other. Both wrap around;
    `sliding_local_pages` ignores the halos the wrap gives."""
    return (rank - 1) % world, (rank + 1) % world


def pack_piece_tails(
    keys: jax.Array, values: jax.Array, halo_pages: int, page_size: int
) -> jax.Array:
    """The `pack_kv_rows` rows of the last `halo_pages` pages of the rank's
    first and second piece; `keys` and `values` are the rank's rows."""
    piece, halo = keys.shape[0] // 2, halo_pages * page_size
    return jnp.concatenate(
        [
            pack_kv_rows(keys[end - halo : end], values[end - halo : end])
            for end in (piece, 2 * piece)
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

    `halos` holds the fp8 [first piece's halo | second piece's halo] pages from
    the exchange (`sliding_exchange_rows`). Rank 0's first piece starts the
    prompt, so its first halo goes unread; the last rank's second piece follows
    its own first piece, whose tail (from `keys`, `values`, the rank's rows) is
    used instead of what the exchange delivered.
    """
    halo_pages, page_size = halos.shape[0] // 2, halos.shape[1]
    piece, halo = plan.piece_tokens, halo_pages * page_size
    own_tail = pack_kv_pages(
        keys[piece - halo : piece], values[piece - halo : piece], page_size
    )
    return _halo_buffer(halos, own_tail, plan, rank)


def _halo_buffer(
    halos: jax.Array, own_tail: jax.Array, plan: ZigzagPlan, rank: Rank
) -> jax.Array:
    """`sliding_local_pages` given the pages of the rank's first-piece tail."""
    halo_pages = halos.shape[0] // 2
    halo_first, halo_second = halos[:halo_pages], halos[halo_pages:]
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
    return_kv_pages: bool = False,
    prepared_inputs: tuple[jax.Array, jax.Array] | None = None,
    **rpa_kwargs,
) -> Any:
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
        return_kv_pages: Also return `kv_pages` as the kernel leaves them.
        prepared_inputs: (q_hbm, new_kv_hbm) as `wrapper.prepare_inputs` lays
            out `q` and the new K/V (`prologue.qkv_to_rpa_inputs`); then only
            the shapes of `q`, `keys` and `values` are read.
        **rpa_kwargs: Further static `ragged_paged_attention` arguments.

    Returns:
        [2 * piece_tokens, num_q_heads, head_dim] attention output (with
        `output_quant_dtype` in `rpa_kwargs`, its per-token quantized rows and
        scales), and with `return_kv_pages` the pages (the kernel's aliased
        cache output).

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
    out, kv_pages = wrapper.ragged_paged_attention(
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
        prepared_inputs=prepared_inputs,
        **rpa_kwargs,
    )
    return (out, kv_pages) if return_kv_pages else out


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


def _copy_head_page_kernel(
    _count_ref, _src_ref, _dst_ref, head_ref, page_ref, _cache_ref, out_ref, *, half
):
    if not half:
        out_ref[...] = page_ref[...]
        return
    # A 32-bit word packs one position's slots K_h V_h K_h+1 V_h+1, lowest
    # byte first; the cache keeps K_h V_h and zero padding slots.
    words = pltpu.bitcast(page_ref[...], jnp.uint32)
    odd = head_ref[0] % 2

    @pl.when(odd == 0)
    def _():
        out_ref[...] = pltpu.bitcast(words & jnp.uint32(0xFFFF), out_ref.dtype)

    @pl.when(odd == 1)
    def _():
        out_ref[...] = pltpu.bitcast(words >> 16, out_ref.dtype)


def _seq_along_lane_head_page_kernel(
    _count_ref,
    _src_ref,
    _dst_ref,
    head_ref,
    page_ref,
    _cache_ref,
    out_ref,
    scratch_ref,
    *,
    shared_kv_rotary_dims=0,
):
    """One head's K and V of a gathered page as a SEQ_ALONG_LANE cache page:
    from (token, d) words of four slots to (d group, token) words of four
    consecutive d, each slot's bytes transposed through `scratch_ref`. With
    `shared_kv_rotary_dims` the page is a shared-KV page: V in slot 0 and K's
    rotated d groups leading slot 1 (`wrapper.prepare_inputs`), zero after."""
    _, page_size, _, _, head_dim = page_ref.shape
    lanes = scratch_ref.shape[-1]
    chunks, groups = page_size // lanes, head_dim // 4
    words = pltpu.bitcast(page_ref[...], jnp.uint32).reshape(page_size, head_dim)

    def every_4th_row(i):
        """Rows i, i + 4, ... of the transposed slot in scratch: [groups, page_size]."""
        rows = [
            scratch_ref[pl.ds(chunks * i + c, groups, stride=4 * chunks), :]
            for c in range(chunks)
        ]
        return jnp.concatenate(rows, axis=1)

    def store(shift):
        for slot in (0, 1):  # K, then V
            byte = (words >> (shift + 8 * slot)) & jnp.uint32(0xFF)
            transposed = byte.T  # [head_dim, page_size]
            for c in range(chunks):
                scratch_ref[pl.ds(c, head_dim, stride=chunks), :] = transposed[
                    :, c * lanes : (c + 1) * lanes
                ]
            w = every_4th_row(0)
            for i in range(1, 4):
                w = w | (every_4th_row(i) << (8 * i))
            out_slot = slot
            if shared_kv_rotary_dims:  # V, then K's rotated groups of each half
                rot, half = shared_kv_rotary_dims // 4, groups // 2
                out_slot = 1 - slot
                if slot == 0:
                    zeros = jnp.zeros((groups - 2 * rot, page_size), w.dtype)
                    w = jnp.concatenate([w[:rot], w[half : half + rot], zeros], 0)
            out_ref[0, out_slot] = pltpu.bitcast(w, out_ref.dtype).reshape(
                groups, 4, page_size
            )

    odd = head_ref[0] % 2

    @pl.when(odd == 0)
    def _():
        store(0)

    @pl.when(odd == 1)
    def _():
        store(16)


def write_kv_head_pages(
    kv_cache: jax.Array,
    pages: jax.Array,
    first_head: Rank,
    src_pages: jax.Array,
    dst_pages: jax.Array,
    count: jax.Array,
    *,
    num_heads: int,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    shared_kv_rotary_dims: int = 0,
) -> jax.Array:
    """`write_kv_pages` of the KV heads [first_head, first_head + num_heads)
    of `pages`, which hold every KV head (`kv_pages_from_rows`). With
    `shared_kv_rotary_dims` (a SEQ_ALONG_LANE cache) each written page is a
    shared-KV page of the head (`wrapper.ragged_paged_attention`).

    The kernel takes the heads' slots from each page in VMEM, so it costs the
    pages it reads and writes; `kv_head_slots` before `write_kv_pages` is an
    XLA slice of the packed slot axis, whose relayout can cost far more.
    `kv_cache` holds fp8 pages of `num_heads` heads in `kv_layout`: one or an
    even number of heads (as `tp_kv_heads` deals them out) in a
    HEAD_ALONG_SUBLANE cache, one head in a SEQ_ALONG_LANE cache.
    """
    *_, page_size, rows, packing, head_dim = pages.shape
    if packing != 4 or kv_cache.dtype != pages.dtype:
        raise NotImplementedError(
            f"Head pages need packing-4 (fp8) pages, got {pages.dtype} pages "
            f"and a {kv_cache.dtype} cache."
        )
    half = num_heads == 1
    if not half and num_heads % 2:
        raise NotImplementedError(f"{num_heads} heads are neither one nor even.")
    # Word rows of a page the heads occupy; two heads' K and V fill one.
    head_rows = 1 if half else num_heads // 2
    if rows % head_rows:
        raise ValueError(f"{rows} word rows do not split into {head_rows}-row heads")
    in_block = (1, page_size, head_rows, packing, head_dim)
    scratch_shapes = []
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        if not half or page_size % 128:
            raise NotImplementedError(
                f"SEQ_ALONG_LANE head pages take one head of whole 128-token lane "
                f"tiles, got {num_heads} heads of {page_size}-token pages."
            )
        out_block = (1, 2, head_dim // packing, packing, page_size)
        kernel = functools.partial(
            _seq_along_lane_head_page_kernel,
            shared_kv_rotary_dims=shared_kv_rotary_dims,
        )
        scratch_shapes.append(
            pltpu.VMEM((head_dim * page_size // 128, 128), jnp.uint32)
        )
    else:
        if shared_kv_rotary_dims:
            raise NotImplementedError("Shared-KV pages are SEQ_ALONG_LANE pages.")
        out_block = in_block
        kernel = functools.partial(_copy_head_page_kernel, half=half)
    if kv_cache.shape[1:] != out_block[1:]:
        raise ValueError(
            f"A {kv_layout} cache of {kv_cache.shape[1:]} pages does not hold "
            f"{num_heads} head(s) of {pages.shape[1:]} pages."
        )

    def step(i, count_ref):
        return jnp.maximum(jnp.minimum(i, count_ref[0] - 1), 0)

    def page_index(i, count_ref, src_ref, dst_ref, head_ref):
        # The block of word rows holding the first head, in head_rows units.
        return (src_ref[step(i, count_ref)], 0, head_ref[0] // 2 // head_rows, 0, 0)

    def cache_index(i, count_ref, src_ref, dst_ref, head_ref):
        return (dst_ref[step(i, count_ref)], 0, 0, 0, 0)

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(dst_pages.shape[0],),
            in_specs=[
                pl.BlockSpec(in_block, page_index),
                # Aliased with the output and never read.
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(out_block, cache_index),
            scratch_shapes=scratch_shapes,
        ),
        out_shape=jax.ShapeDtypeStruct(kv_cache.shape, kv_cache.dtype),
        input_output_aliases={5: 0},
    )(
        jnp.reshape(count, (1,)).astype(jnp.int32),
        src_pages.astype(jnp.int32),
        dst_pages.astype(jnp.int32),
        jnp.reshape(first_head, (1,)).astype(jnp.int32),
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


def window_rows_held(
    keys: jax.Array,
    values: jax.Array,
    plan: ZigzagPlan,
    rank: Rank,
    position_pages: jax.Array,
) -> jax.Array:
    """The `pack_kv_rows` rows (every KV head) of the window pages, page after
    page, where this rank holds them; the other pages carry whatever the
    clipped rows hold (`window_slots_from_rows` takes each page from its
    holder). `keys` and `values` are the rank's rows.
    [len(position_pages) * page_size, 2 * kv_heads * d]."""
    rows = _window_rows(plan, rank, position_pages)
    return pack_kv_rows(jnp.take(keys, rows, axis=0), jnp.take(values, rows, axis=0))


def _window_local_pages(
    plan: ZigzagPlan, rank: Rank, position_pages: jax.Array
) -> jax.Array:
    """The rank's pages (indices into its rows' pages) of the window pages,
    clipped to its pages."""
    pp = plan.piece_pages
    first, second = plan.rank_pieces(rank)
    piece = position_pages // pp
    local = jnp.where(
        piece == first, position_pages - first * pp, position_pages - second * pp + pp
    )
    return jnp.clip(local, 0, 2 * pp - 1)


def _window_rows(plan: ZigzagPlan, rank: Rank, position_pages: jax.Array) -> jax.Array:
    """The rank's rows (indices) of the window pages, clipped to its rows."""
    ps = plan.page_size
    local = _window_local_pages(plan, rank, position_pages)
    return (local[:, None] * ps + jnp.arange(ps)).reshape(-1)


def _exchange_rows_of(
    x: jax.Array,
    plan: ZigzagPlan,
    rank: Rank,
    position_pages: jax.Array,
    halo_pages: int,
) -> jax.Array:
    """x's rows (the rank's rows, first axis) in `sliding_exchange_rows`'
    order: the tails of the rank's two pieces, then the window pages
    (`_window_rows`), each page one dynamic slice."""
    piece, ps = plan.piece_tokens, plan.page_size
    halo = halo_pages * ps
    local = _window_local_pages(plan, rank, position_pages)
    window = [
        jax.lax.dynamic_slice_in_dim(x, local[j] * ps, ps)
        for j in range(position_pages.shape[0])
    ]
    return jnp.concatenate(
        [x[piece - halo : piece], x[2 * piece - halo : 2 * piece], *window]
    )


def window_page_owners(plan: ZigzagPlan, position_pages: jax.Array) -> jax.Array:
    """The rank holding each position page."""
    piece = position_pages // plan.piece_pages
    return jnp.where(piece < plan.world, piece, plan.num_pieces - 1 - piece)


def window_slots_from_rows(
    window_rows: jax.Array,
    owners: jax.Array,
    first_head: Rank,
    num_heads: int,
    head_dim: int,
) -> jax.Array:
    """[world, window pages, page_size, 2 * kv_heads * d] `window_rows_held` of
    every rank (or [..., 2 * kv_heads, d]) -> the [window pages, page_size,
    2 * num_heads, d] K/V slots of heads [first_head, first_head + num_heads),
    each page from its `window_page_owners` rank."""
    pages = window_rows[owners, jnp.arange(owners.shape[0])]
    if pages.ndim == 4:
        return jax.lax.dynamic_slice_in_dim(
            pages, 2 * first_head, 2 * num_heads, axis=2
        )
    slots = jax.lax.dynamic_slice_in_dim(
        pages, 2 * first_head * head_dim, 2 * num_heads * head_dim, axis=2
    )
    return slots.reshape(*slots.shape[:2], 2 * num_heads, head_dim)


CacheView = Callable[
    [jax.Array], tuple[jax.Array, int, Callable[[jax.Array], jax.Array]]
]


@dataclasses.dataclass(frozen=True)
class LayerConfig:
    """One attention layer of a context-parallel prefill on `world` ranks.

    `num_heads` and `num_kv_heads` count every rank's heads; each rank's serving
    cache holds `local_kv_heads` of them (`tp_kv_heads`). The rank itself is
    not part of it: the layer functions take it as an array, so every rank
    traces the same program.
    """

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
    # A full-attention layer caches its heads with `write_kv_head_pages`.
    head_page_writes: bool = False
    # The q/k/v norm epsilon of a layer whose attention starts from its qkv
    # projection rows (`prologue.qkv_to_rpa_inputs`); None: from q, k and v.
    prologue_eps: float | None = None
    # The layer returns its output as the o projection's per-token fp8 rows and
    # scales (`ragged_paged_attention(output_quant_dtype=...)`) instead of bf16
    # rows.
    fp8_out_rows: bool = False
    # A sliding-window layer's exchange rows come from their own qkv rows
    # (`sliding_exchange_rows_from_qkv`), and the layer bitcasts the gathered
    # int8 rows to fp8 after taking the ones it reads.
    exchange_from_qkv: bool = False
    # The attention kernel skips the mask on steps that mask nothing
    # (`ragged_paged_attention(unmasked_interior=True)`).
    unmasked_interior: bool = False
    # The serving cache holds shared-KV pages of this full-attention layer
    # (`write_kv_head_pages(shared_kv_rotary_dims=...)`); the layer's own
    # attention runs on its gathered K/V pages as usual.
    shared_kv_rotary_dims: int = 0
    # The from-qkv layers all-gather their fp8 K/V as int8 rows in the new-K/V
    # layout ([rows, slots, packing, d], `exchange_rows_int8`), which a page
    # buffer splits without a copy, instead of as [rows, slots * packing * d].
    paged_gather: bool = False
    # The qkv prologue writes its packed outputs with strided DMAs
    # (`prologue.qkv_to_rpa_inputs(dma_stores=True)`, the same bits).
    prologue_dma_stores: bool = False
    # With prologue_dma_stores, on a full-attention fp8_compute layer: the
    # prologue also writes the FP8 queries the attention kernel reads, which it
    # would otherwise cast from the bf16 ones (the same bits).
    prologue_fp8_queries: bool = False

    def __post_init__(self):
        if self.prologue_dma_stores and self.prologue_eps is None:
            raise ValueError("prologue_dma_stores needs a layer with prologue_eps")
        if self.prologue_fp8_queries and not (
            self.prologue_dma_stores
            and self.fp8_compute
            and self.sliding_window is None
        ):
            raise ValueError(
                "prologue_fp8_queries needs prologue_dma_stores on a full-attention "
                "fp8_compute layer"
            )
        if self.shared_kv_rotary_dims and not self.head_page_writes:
            raise NotImplementedError(
                "Shared-KV pages are cached by the head-page kernel: set "
                "TPU_CP_KV_WRITE_KERNEL=1."
            )
        if not self.head_page_writes:
            return
        layout, heads = self.kv_layout, self.local_kv_heads
        if layout == configs.KVLayout.SEQ_ALONG_LANE and heads != 1:
            raise NotImplementedError(
                f"Head page writes cache one KV head per rank in a SEQ_ALONG_LANE "
                f"cache, not {heads}."
            )
        if heads != 1 and heads % 2:
            raise NotImplementedError(
                f"Head page writes cache one or an even number of KV heads per "
                f"rank, not {heads}."
            )

    @property
    def local_kv_heads(self) -> int:
        # The count is the same on every rank; only the first head depends on it.
        return tp_kv_heads(self.num_kv_heads, self.world, 0)[1]

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
        if self.unmasked_interior:
            kwargs["unmasked_interior"] = True
        if self.fp8_out_rows:
            kwargs["output_quant_dtype"] = jnp.float8_e4m3fn
        return kwargs


def _cache_pages(kv_cache: jax.Array, cfg: LayerConfig, cache_view: CacheView):
    """(native pages, native pages per block, inverse view, page size)."""
    native, page_stride, restore = cache_view(kv_cache)
    page_axis = 4 if cfg.kv_layout == configs.KVLayout.SEQ_ALONG_LANE else 1
    return native, page_stride, restore, native.shape[page_axis]


def _request_block_table(block_tables: jax.Array, seq_lens: jax.Array) -> jax.Array:
    """Block table row of the step's one request (batch index 0)."""
    return block_tables[: block_tables.shape[0] // seq_lens.shape[0]]


def qkv_prologue(
    qkv: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    q_norm_weight: jax.Array,
    k_norm_weight: jax.Array,
    *,
    cfg: LayerConfig,
) -> tuple[jax.Array, ...]:
    """(q_hbm, new_kv): `prologue.qkv_to_rpa_inputs` of a layer like `cfg`
    (every rank's heads), whose norms take `cfg.prologue_eps`; with
    `cfg.prologue_fp8_queries` also the FP8 queries, (q_hbm, new_kv, q_fp8)."""
    if cfg.prologue_eps is None:
        raise ValueError("qkv_prologue needs a layer with prologue_eps")
    return prologue.qkv_to_rpa_inputs(
        qkv,
        cos,
        sin,
        q_norm_weight,
        k_norm_weight,
        num_q_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        eps=cfg.prologue_eps,
        k_scale=cfg.k_scale,
        v_scale=cfg.v_scale,
        dma_stores=cfg.prologue_dma_stores,
        fp8_queries=cfg.prologue_fp8_queries,
    )


def full_attention_layer(
    kv_cache: jax.Array,
    query: jax.Array,
    kv_rows: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    rank: jax.Array,
    *,
    cfg: LayerConfig,
    cache_view: CacheView,
) -> tuple[jax.Array, jax.Array]:
    """A full-attention layer: attention of the rank's rows over every rank's
    K/V, and the cache with this rank's heads of every prompt position.

    Args:
        kv_cache: The layer's serving cache; `cache_view` gives its native pages.
        query: [rows, num_heads, head_dim].
        kv_rows: Every rank's `pack_kv_rows` rows in rank order, as int8.
        seq_lens, block_tables: The step's attention metadata (one request).
        rank: [1] int32, this rank.
        cfg: The layer.
        cache_view: `kv_cache` -> (native pages, pages per block, inverse view).

    Returns:
        (the new kv_cache, [rows, num_heads, head_dim] attention output).
    """
    return _full_attention(
        kv_cache, query, kv_rows, seq_lens, block_tables, rank[0], cfg, cache_view
    )


def full_attention_layer_from_q(
    kv_cache: jax.Array,
    q_hbm: jax.Array,
    kv_rows: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    rank: jax.Array,
    *,
    cfg: LayerConfig,
    cache_view: CacheView,
) -> tuple[jax.Array, jax.Array]:
    """`full_attention_layer` with the queries as `wrapper.prepare_inputs` lays
    them out ([kv_heads, rows, q_per_kv // p, p, head_dim], from
    `prologue.qkv_to_rpa_inputs`)."""
    num_rows = q_hbm.shape[1]
    query = jnp.zeros((num_rows, cfg.num_heads, cfg.head_dim), q_hbm.dtype)
    return _full_attention(
        kv_cache,
        query,
        kv_rows,
        seq_lens,
        block_tables,
        rank[0],
        cfg,
        cache_view,
        q_hbm=q_hbm,
    )


def full_attention_layer_from_q8(
    kv_cache: jax.Array,
    q_hbm: jax.Array,
    q_fp8: jax.Array,
    kv_rows: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    rank: jax.Array,
    *,
    cfg: LayerConfig,
    cache_view: CacheView,
) -> tuple[jax.Array, jax.Array]:
    """`full_attention_layer_from_q` with the prologue's FP8 queries too
    (`LayerConfig.prologue_fp8_queries`), which the fp8_compute kernel reads
    instead of casting `q_hbm`."""
    if not cfg.prologue_fp8_queries:
        raise ValueError("full_attention_layer_from_q8 needs prologue_fp8_queries")
    num_rows = q_hbm.shape[1]
    query = jnp.zeros((num_rows, cfg.num_heads, cfg.head_dim), q_hbm.dtype)
    return _full_attention(
        kv_cache,
        query,
        kv_rows,
        seq_lens,
        block_tables,
        rank[0],
        cfg,
        cache_view,
        q_hbm=q_hbm,
        q_fp8=q_fp8,
    )


def _layer_output(out: Any, cfg: LayerConfig) -> tuple[jax.Array, ...]:
    """The layer output after the attention kernel: its bf16 rows, or with
    `cfg.fp8_out_rows` the fp8 rows and per-token scales the o projection
    takes (`LayerConfig.rpa_kwargs`)."""
    return tuple(out) if cfg.fp8_out_rows else (out,)


def _full_attention(
    kv_cache: jax.Array,
    query: jax.Array,
    kv_rows: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    rank: jax.Array,
    cfg: LayerConfig,
    cache_view: CacheView,
    q_hbm: jax.Array | None = None,
    q_fp8: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    native, page_stride, restore, page_size = _cache_pages(kv_cache, cfg, cache_view)
    plan = cfg.plan(query.shape[0], page_size)
    rows = jax.lax.bitcast_convert_type(kv_rows, native.dtype)
    pages = _gathered_pages(rows, cfg.head_dim, page_size)
    first, count = tp_kv_heads(cfg.num_kv_heads, cfg.world, rank)
    table = _request_block_table(block_tables, seq_lens) * page_stride
    writes = full_attention_page_writes(plan, table, seq_lens[0])
    kwargs = cfg.rpa_kwargs()
    if q_hbm is not None:
        # Every key comes from `pages`: the new K/V operand is never read.
        new_kv = jnp.zeros((query.shape[0], *pages.shape[2:]), pages.dtype)
        prepared = (q_hbm, new_kv) if q_fp8 is None else (q_hbm, new_kv, q_fp8)
        kwargs["prepared_inputs"] = prepared
    tables = plan.gathered_page_table()
    if cfg.head_page_writes:
        # The kernel reads every key from `pages` and leaves them as they are.
        out, pages = cp_prefill_attention(
            query, pages, tables, plan, rank, return_kv_pages=True, **kwargs
        )
        new_native = write_kv_head_pages(
            native,
            pages,
            first,
            *writes,
            num_heads=count,
            kv_layout=cfg.kv_layout,
            shared_kv_rotary_dims=cfg.shared_kv_rotary_dims,
        )
        return restore(new_native), *_layer_output(out, cfg)
    out = cp_prefill_attention(query, pages, tables, plan, rank, **kwargs)
    mine = cache_pages_from_slots(kv_head_slots(pages, first, count), cfg.kv_layout)
    return restore(write_kv_pages(native, mine, *writes)), *_layer_output(out, cfg)


def _sliding_exchange_pages(cfg: LayerConfig, page_size: int) -> tuple[int, int]:
    """(halo pages, window pages) of a sliding-window layer's
    `sliding_exchange_rows`: two piece tails of halo pages, then the window."""
    assert cfg.sliding_window is not None
    halo_pages = num_halo_pages(cfg.sliding_window, page_size)
    return halo_pages, halo_pages + 1


def sliding_exchange_rows(
    keys: jax.Array,
    values: jax.Array,
    seq_lens: jax.Array,
    rank: jax.Array,
    page_size: int,
    *,
    cfg: LayerConfig,
) -> jax.Array:
    """This rank's part of a sliding-window layer's one all-gather:
    [(3 * halo pages + 1) * page_size, 2 * kv_heads * d] int8 `pack_kv_rows`
    rows. They hold the tails of the rank's first and second piece
    (`pack_piece_tails`: the halos of the ranks holding the next pieces,
    `halo_neighbours`), then the decode window's pages where this rank holds
    them (`window_rows_held`), every KV head.

    `keys` and `values` are the rank's fp8 rows, `rank` [1] int32 this rank and
    `page_size` the KV cache's. The all-gather delivers every tail to every
    rank, world / 2 times the halo bytes they read, and its result holds world
    times these rows. Token-major rows keep the leading dimension large: the
    same bytes as one [1, n] row took 3.8 ms to gather on 8 ranks.
    """
    rank = rank[0]
    plan = cfg.plan(keys.shape[0], page_size)
    halo_pages, _ = _sliding_exchange_pages(cfg, page_size)
    plan.check_halo_pages(halo_pages)
    position_pages, _ = window_position_pages(
        seq_lens[0], cfg.sliding_window, page_size
    )
    rows = jnp.concatenate(
        [
            pack_piece_tails(keys, values, halo_pages, page_size),
            window_rows_held(keys, values, plan, rank, position_pages),
        ]
    )
    return jax.lax.bitcast_convert_type(rows, jnp.int8)


def exchange_rows_int8(new_kv: jax.Array, cfg: LayerConfig) -> jax.Array:
    """The all-gather operand of fp8 K/V rows in the new-K/V layout ([rows,
    slots, packing, d], `wrapper.prepare_inputs`): int8 rows of that layout with
    `cfg.paged_gather`, else 2-D [rows, slots * packing * d]."""
    rows = new_kv if cfg.paged_gather else new_kv.reshape(new_kv.shape[0], -1)
    return jax.lax.bitcast_convert_type(rows, jnp.int8)


def _gathered_pages(rows: jax.Array, head_dim: int, page_size: int) -> jax.Array:
    """Exchanged fp8 rows -> HEAD_ALONG_SUBLANE pages: 2-D `pack_kv_rows`
    rows through `kv_pages_from_rows`, new-K/V-layout rows by a split of their
    leading axis."""
    if rows.ndim == 2:
        return kv_pages_from_rows(rows, head_dim, page_size)
    return rows.reshape(rows.shape[0] // page_size, page_size, *rows.shape[1:])


def sliding_exchange_rows_from_kv(
    new_kv: jax.Array,
    seq_lens: jax.Array,
    rank: jax.Array,
    page_size: int,
    *,
    cfg: LayerConfig,
) -> jax.Array:
    """`sliding_exchange_rows` of the rank's K/V as `wrapper.prepare_inputs`
    lays new K/V out ([rows, slots, packing, head_dim], every slot a K or V
    head: the logical row-major order of `pack_kv_rows`)."""
    rank = rank[0]
    plan = cfg.plan(new_kv.shape[0], page_size)
    halo_pages, _ = _sliding_exchange_pages(cfg, page_size)
    plan.check_halo_pages(halo_pages)
    position_pages, _ = window_position_pages(
        seq_lens[0], cfg.sliding_window, page_size
    )
    piece, halo = plan.piece_tokens, halo_pages * page_size
    rows = jnp.concatenate(
        [
            new_kv[piece - halo : piece],
            new_kv[2 * piece - halo : 2 * piece],
            jnp.take(new_kv, _window_rows(plan, rank, position_pages), axis=0),
        ]
    )
    return exchange_rows_int8(rows, cfg)


def sliding_exchange_rows_from_qkv(
    qkv: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    q_norm_weight: jax.Array,
    k_norm_weight: jax.Array,
    seq_lens: jax.Array,
    rank: jax.Array,
    page_size: int,
    *,
    cfg: LayerConfig,
) -> jax.Array:
    """`sliding_exchange_rows_from_kv` of `qkv_prologue(qkv, cos, sin, ...)`'s
    K/V, from the prologue of only the exchanged rows: the all-gather can start
    while the prologue of the others runs. The prologue is row-wise, so the
    rows are the same."""
    rank = rank[0]
    plan = cfg.plan(qkv.shape[0], page_size)
    halo_pages, _ = _sliding_exchange_pages(cfg, page_size)
    plan.check_halo_pages(halo_pages)
    position_pages, _ = window_position_pages(
        seq_lens[0], cfg.sliding_window, page_size
    )
    pick = functools.partial(
        _exchange_rows_of,
        plan=plan,
        rank=rank,
        position_pages=position_pages,
        halo_pages=halo_pages,
    )
    new_kv = qkv_prologue(
        pick(qkv), pick(cos), pick(sin), q_norm_weight, k_norm_weight, cfg=cfg
    )[1]
    return exchange_rows_int8(new_kv, cfg)


def sliding_attention_layer(
    kv_cache: jax.Array,
    query: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    gathered: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    rank: jax.Array,
    *,
    cfg: LayerConfig,
    cache_view: CacheView,
) -> tuple[jax.Array, jax.Array]:
    """A sliding-window layer: attention of the rank's rows within [halo |
    piece], and the cache with this rank's heads of the decode window.

    Args:
        kv_cache: The layer's serving cache; `cache_view` gives its native pages.
        query: [rows, num_heads, head_dim].
        keys, values: [rows, num_kv_heads, head_dim] fp8 K/V of the rank's rows.
        gathered: Every rank's `sliding_exchange_rows`, all-gathered in rank
            order: [world * rows, 2 * kv_heads * d] int8.
        seq_lens, block_tables: The step's attention metadata (one request).
        rank: [1] int32, this rank.
        cfg: The layer.
        cache_view: `kv_cache` -> (native pages, pages per block, inverse view).

    Returns:
        (the new kv_cache, [rows, num_heads, head_dim] attention output).
    """
    piece = query.shape[0] // 2

    def own_tail(halo: int, page_size: int) -> jax.Array:
        span = slice(piece - halo, piece)
        return pack_kv_pages(keys[span], values[span], page_size)

    return _sliding_attention(
        kv_cache,
        query,
        keys,
        values,
        gathered,
        seq_lens,
        block_tables,
        rank[0],
        cfg,
        cache_view,
        own_tail,
    )


def sliding_attention_layer_from_q(
    kv_cache: jax.Array,
    q_hbm: jax.Array,
    new_kv: jax.Array,
    gathered: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    rank: jax.Array,
    *,
    cfg: LayerConfig,
    cache_view: CacheView,
) -> tuple[jax.Array, jax.Array]:
    """`sliding_attention_layer` with the queries and the rank's new K/V as
    `wrapper.prepare_inputs` lays them out (`prologue.qkv_to_rpa_inputs`), and
    the all-gather of `sliding_exchange_rows_from_kv`."""
    num_rows, piece = q_hbm.shape[1], q_hbm.shape[1] // 2
    query = jnp.zeros((num_rows, cfg.num_heads, cfg.head_dim), q_hbm.dtype)
    keys = jnp.zeros((num_rows, cfg.num_kv_heads, cfg.head_dim), new_kv.dtype)

    def own_tail(halo: int, page_size: int) -> jax.Array:
        tail = new_kv[piece - halo : piece]
        return tail.reshape(halo // page_size, page_size, *new_kv.shape[1:])

    return _sliding_attention(
        kv_cache,
        query,
        keys,
        keys,
        gathered,
        seq_lens,
        block_tables,
        rank[0],
        cfg,
        cache_view,
        own_tail,
        prepared_inputs=(q_hbm, new_kv),
    )


def _sliding_attention(
    kv_cache: jax.Array,
    query: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    gathered: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    rank: jax.Array,
    cfg: LayerConfig,
    cache_view: CacheView,
    own_tail: Callable[[int, int], jax.Array],
    prepared_inputs: tuple[jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, ...]:
    """`sliding_attention_layer`'s body; `own_tail(halo tokens, page size)`
    gives the pages of the rank's first-piece tail."""
    if cfg.sliding_window is None:
        raise ValueError("sliding_attention_layer needs a sliding window")
    native, page_stride, restore, page_size = _cache_pages(kv_cache, cfg, cache_view)
    plan = cfg.plan(query.shape[0], page_size)
    halo_pages, window_pages = _sliding_exchange_pages(cfg, page_size)
    halo, width = halo_pages * page_size, 2 * cfg.num_kv_heads * cfg.head_dim
    num_rows = 2 * halo + window_pages * page_size
    row_shape = gathered.shape[1:]
    if gathered.shape[0] != cfg.world * num_rows or math.prod(row_shape) != width:
        raise ValueError(
            f"Exchange rows {gathered.shape} are not {cfg.world} x {num_rows} "
            f"rows of {width} bytes ({page_size}-token pages)."
        )
    if cfg.exchange_from_qkv:
        # Bitcast only the rows read: the whole buffer's bitcast is a copy.
        rows = gathered.reshape(cfg.world, num_rows, *row_shape)

        def fp8(x):
            return jax.lax.bitcast_convert_type(x, keys.dtype)
    else:
        rows = jax.lax.bitcast_convert_type(gathered, keys.dtype)
        rows = rows.reshape(cfg.world, num_rows, *row_shape)

        def fp8(x):
            return x

    first_src, second_src = halo_neighbours(rank, cfg.world)
    halos = _gathered_pages(
        fp8(
            jnp.concatenate([rows[first_src, :halo], rows[second_src, halo : 2 * halo]])
        ),
        cfg.head_dim,
        page_size,
    )
    local = _halo_buffer(halos, own_tail(halo, page_size), plan, rank)
    tables = plan.sliding_page_tables(rank, halo_pages)
    out = cp_prefill_attention(
        query,
        local,
        tables,
        plan,
        rank,
        keys=keys,
        values=values,
        prepared_inputs=prepared_inputs,
        **cfg.rpa_kwargs(),
    )
    window = rows[:, 2 * halo :].reshape(
        cfg.world,
        window_pages,
        page_size,
        # New-K/V-layout rows keep d minor: merging (slots, packing), packing
        # being the sublane tile, keeps their tiling.
        *((width,) if len(row_shape) == 1 else (-1, cfg.head_dim)),
    )
    position_pages, count = window_position_pages(
        seq_lens[0], cfg.sliding_window, page_size
    )
    slots = window_slots_from_rows(
        window,
        window_page_owners(plan, position_pages),
        *tp_kv_heads(cfg.num_kv_heads, cfg.world, rank),
        cfg.head_dim,
    )
    slots = fp8(slots)
    table = _request_block_table(block_tables, seq_lens) * page_stride
    writes = window_page_writes(table, position_pages, count)
    mine = cache_pages_from_slots(slots, cfg.kv_layout)
    return restore(write_kv_pages(native, mine, *writes)), *_layer_output(out, cfg)


# The layer ops of a `LayerConfig` with `fp8_out_rows`: (the new kv_cache,
# [rows, num_heads * head_dim] fp8 rows, [rows, 1] f32 scales).


def full_attention_layer_fp8_rows(
    kv_cache: jax.Array,
    query: jax.Array,
    kv_rows: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    rank: jax.Array,
    *,
    cfg: LayerConfig,
    cache_view: CacheView,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """`full_attention_layer` returning the o projection's fp8 input."""
    return _fp8_rows_layer(
        full_attention_layer,
        cfg,
        kv_cache,
        query,
        kv_rows,
        seq_lens,
        block_tables,
        rank,
        cache_view=cache_view,
    )


def full_attention_layer_from_q_fp8_rows(
    kv_cache: jax.Array,
    q_hbm: jax.Array,
    kv_rows: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    rank: jax.Array,
    *,
    cfg: LayerConfig,
    cache_view: CacheView,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """`full_attention_layer_from_q` returning the o projection's fp8 input."""
    return _fp8_rows_layer(
        full_attention_layer_from_q,
        cfg,
        kv_cache,
        q_hbm,
        kv_rows,
        seq_lens,
        block_tables,
        rank,
        cache_view=cache_view,
    )


def full_attention_layer_from_q8_fp8_rows(
    kv_cache: jax.Array,
    q_hbm: jax.Array,
    q_fp8: jax.Array,
    kv_rows: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    rank: jax.Array,
    *,
    cfg: LayerConfig,
    cache_view: CacheView,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """`full_attention_layer_from_q8` returning the o projection's fp8 input."""
    return _fp8_rows_layer(
        full_attention_layer_from_q8,
        cfg,
        kv_cache,
        q_hbm,
        q_fp8,
        kv_rows,
        seq_lens,
        block_tables,
        rank,
        cache_view=cache_view,
    )


def sliding_attention_layer_fp8_rows(
    kv_cache: jax.Array,
    query: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    gathered: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    rank: jax.Array,
    *,
    cfg: LayerConfig,
    cache_view: CacheView,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """`sliding_attention_layer` returning the o projection's fp8 input."""
    return _fp8_rows_layer(
        sliding_attention_layer,
        cfg,
        kv_cache,
        query,
        keys,
        values,
        gathered,
        seq_lens,
        block_tables,
        rank,
        cache_view=cache_view,
    )


def sliding_attention_layer_from_q_fp8_rows(
    kv_cache: jax.Array,
    q_hbm: jax.Array,
    new_kv: jax.Array,
    gathered: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    rank: jax.Array,
    *,
    cfg: LayerConfig,
    cache_view: CacheView,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """`sliding_attention_layer_from_q` returning the o projection's fp8 input."""
    return _fp8_rows_layer(
        sliding_attention_layer_from_q,
        cfg,
        kv_cache,
        q_hbm,
        new_kv,
        gathered,
        seq_lens,
        block_tables,
        rank,
        cache_view=cache_view,
    )


def _fp8_rows_layer(layer: Callable[..., Any], cfg: LayerConfig, *args, **kwargs):
    if not cfg.fp8_out_rows:
        raise ValueError(f"{layer.__name__}: the layer config has no fp8_out_rows")
    kv_cache, rows, scales = layer(*args, cfg=cfg, **kwargs)
    return kv_cache, rows, scales

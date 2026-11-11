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
"""Zigzag context-parallel prefill attention on batched RPA (one prompt per step)."""

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
    """Static zigzag split of a T-token prompt over ``world`` ranks."""

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

    def gathered_page_table(self) -> np.ndarray:
        """Position page -> page of the rank-major all-gather of every rank's
        [first piece | second piece] pages."""
        pp = self.piece_pages
        table = np.empty(self.num_pages, np.int32)
        for page in range(self.num_pages):
            rank, slot = self.piece_owner(page // pp)
            table[page] = rank * 2 * pp + slot * pp + page % pp
        return table

    def check_halo_pages(self, halo_pages: int) -> None:
        """Refuses a halo that is not within the one piece before each piece."""
        if not 1 <= halo_pages <= self.piece_pages:
            raise ValueError(
                f"A halo of {halo_pages} pages must be between 1 and the piece's "
                f"{self.piece_pages} pages."
            )

    def sliding_page_tables(self, rank: Rank, halo_pages: int) -> jax.Array:
        """Map sub-request pages to ``[halo_first | halo_second | scratch]``."""
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
    return (rank - 1) % world, (rank + 1) % world


def pack_piece_tails(
    keys: jax.Array, values: jax.Array, halo_pages: int, page_size: int
) -> jax.Array:
    piece, halo = keys.shape[0] // 2, halo_pages * page_size
    return jnp.concatenate(
        [
            pack_kv_rows(keys[end - halo : end], values[end - halo : end])
            for end in (piece, 2 * piece)
        ]
    )


def _halo_buffer(
    halos: jax.Array, own_tail: jax.Array, plan: ZigzagPlan, rank: Rank
) -> jax.Array:
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
    page_tables: jax.Array | np.ndarray,
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
    """Attention of the rank's two pieces against ``kv_pages``."""
    cu_q_lens, kv_lens, distribution = plan.sub_request_metadata(rank)
    tables = jnp.broadcast_to(jnp.asarray(page_tables), (2, plan.num_pages))
    skip_kv_update = keys is None
    if skip_kv_update:
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
        skip_empty_launch=True,
        fp8_compute=fp8_compute,
        prepared_inputs=prepared_inputs,
        **rpa_kwargs,
    )
    return (out, kv_pages) if return_kv_pages else out


def tp_kv_heads(num_kv_heads: int, tp_size: int, tp_rank: Rank) -> tuple[Rank, int]:
    """(first KV head, KV head count) that tensor-parallel rank `tp_rank` caches."""
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
        return slots.reshape(
            num_pages, page_size, heads_x2, head_dim // packing, packing
        ).transpose(0, 2, 3, 4, 1)
    aligned = utils.align_to(heads_x2, packing)
    if aligned != heads_x2:
        slots = jnp.pad(slots, ((0, 0), (0, 0), (0, aligned - heads_x2), (0, 0)))
    return slots.reshape(num_pages, page_size, aligned // packing, packing, head_dim)


def _copy_page_kernel(
    count_ref,
    src_ref,
    dst_ref,
    pages_hbm_ref,
    _cache_hbm_ref,
    out_hbm_ref,
    page_vmem_ref,
    dma_sem,
):
    i = pl.program_id(0)

    @pl.when(i < count_ref[0])
    def _():
        src = jnp.clip(src_ref[i], 0, pages_hbm_ref.shape[0] - 1)
        dst = jnp.clip(dst_ref[i], 0, out_hbm_ref.shape[0] - 1)
        rd = pltpu.make_async_copy(pages_hbm_ref.at[src], page_vmem_ref, dma_sem.at[0])
        rd.start()
        rd.wait()
        wr = pltpu.make_async_copy(page_vmem_ref, out_hbm_ref.at[dst], dma_sem.at[0])
        wr.start()
        wr.wait()


def write_kv_pages(
    kv_cache: jax.Array,
    pages: jax.Array,
    src_pages: jax.Array,
    dst_pages: jax.Array,
    count: jax.Array,
) -> jax.Array:
    """Scatter ``pages[src_pages[:count]]`` into ``kv_cache[dst_pages[:count]]``."""
    return pl.pallas_call(
        _copy_page_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(dst_pages.shape[0],),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM(kv_cache.shape[1:], kv_cache.dtype),
                pltpu.SemaphoreType.DMA((1,)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(kv_cache.shape, kv_cache.dtype),
        input_output_aliases={4: 0},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name="cp_write_kv_pages",
    )(
        jnp.reshape(count, (1,)).astype(jnp.int32),
        src_pages.astype(jnp.int32),
        dst_pages.astype(jnp.int32),
        pages,
        kv_cache,
    )


def _copy_head_page_kernel(
    count_ref,
    _src_ref,
    dst_ref,
    head_ref,
    page_ref,
    _cache_hbm_ref,
    out_hbm_ref,
    out_vmem_ref,
    dma_sem,
    *,
    half,
):
    i = pl.program_id(0)

    @pl.when(i < count_ref[0])
    def _():
        if not half:
            out_vmem_ref[...] = page_ref[0]
        else:
            words = pltpu.bitcast(page_ref[0], jnp.uint32)
            odd = head_ref[0] % 2

            @pl.when(odd == 0)
            def _even():
                out_vmem_ref[...] = pltpu.bitcast(
                    words & jnp.uint32(0xFFFF), out_vmem_ref.dtype
                )

            @pl.when(odd == 1)
            def _odd():
                out_vmem_ref[...] = pltpu.bitcast(words >> 16, out_vmem_ref.dtype)

        dst = jnp.clip(dst_ref[i], 0, out_hbm_ref.shape[0] - 1)
        wr = pltpu.make_async_copy(out_vmem_ref, out_hbm_ref.at[dst], dma_sem.at[0])
        wr.start()
        wr.wait()


def _seq_along_lane_head_page_kernel(
    count_ref,
    _src_ref,
    dst_ref,
    head_ref,
    page_ref,
    _cache_hbm_ref,
    out_hbm_ref,
    out_vmem_ref,
    dma_sem,
    scratch_ref,
    *,
    shared_kv_rotary_dims=0,
):
    _, page_size, _, _, head_dim = page_ref.shape
    lanes = scratch_ref.shape[-1]
    chunks, groups = page_size // lanes, head_dim // 4

    def every_4th_row(k):
        """Rows k, k + 4, ... of the transposed slot in scratch: [groups, page_size]."""
        rows = [
            scratch_ref[pl.ds(chunks * k + c, groups, stride=4 * chunks), :]
            for c in range(chunks)
        ]
        return jnp.concatenate(rows, axis=1)

    def store(words, shift):
        for slot in (0, 1):  # K, then V
            byte = (words >> (shift + 8 * slot)) & jnp.uint32(0xFF)
            transposed = byte.T  # [head_dim, page_size]
            for c in range(chunks):
                scratch_ref[pl.ds(c, head_dim, stride=chunks), :] = transposed[
                    :, c * lanes : (c + 1) * lanes
                ]
            w = every_4th_row(0)
            for k in range(1, 4):
                w = w | (every_4th_row(k) << (8 * k))
            out_slot = slot
            if shared_kv_rotary_dims:  # V, then K's rotated groups of each half
                rot, half = shared_kv_rotary_dims // 4, groups // 2
                out_slot = 1 - slot
                if slot == 0:
                    zeros = jnp.zeros((groups - 2 * rot, page_size), w.dtype)
                    w = jnp.concatenate([w[:rot], w[half : half + rot], zeros], 0)
            out_vmem_ref[out_slot] = pltpu.bitcast(w, out_vmem_ref.dtype).reshape(
                groups, 4, page_size
            )

    step_idx = pl.program_id(0)

    @pl.when(step_idx < count_ref[0])
    def _():
        words = pltpu.bitcast(page_ref[...], jnp.uint32).reshape(page_size, head_dim)
        odd = head_ref[0] % 2

        @pl.when(odd == 0)
        def _even():
            store(words, 0)

        @pl.when(odd == 1)
        def _odd():
            store(words, 16)

        dst = jnp.clip(dst_ref[step_idx], 0, out_hbm_ref.shape[0] - 1)
        wr = pltpu.make_async_copy(out_vmem_ref, out_hbm_ref.at[dst], dma_sem.at[0])
        wr.start()
        wr.wait()


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
    """Scatter TP KV heads `[first_head, first_head + num_heads)` into `kv_cache`."""
    *_, page_size, rows, packing, head_dim = pages.shape
    if packing != 4 or kv_cache.dtype != pages.dtype:
        raise NotImplementedError(
            f"Head pages need packing-4 (fp8) pages, got {pages.dtype} pages "
            f"and a {kv_cache.dtype} cache."
        )
    half = num_heads == 1
    if not half and num_heads % 2:
        raise NotImplementedError(f"{num_heads} heads are neither one nor even.")
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

    def page_index(i, count_ref, src_ref, _dst_ref, head_ref):
        src = jnp.clip(src_ref[step(i, count_ref)], 0, pages.shape[0] - 1)
        return (src, 0, head_ref[0] // 2 // head_rows, 0, 0)

    all_scratch = [
        pltpu.VMEM(out_block[1:], kv_cache.dtype),
        pltpu.SemaphoreType.DMA((1,)),
        *scratch_shapes,
    ]
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(dst_pages.shape[0],),
            in_specs=[
                pl.BlockSpec(in_block, page_index),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=all_scratch,
        ),
        out_shape=jax.ShapeDtypeStruct(kv_cache.shape, kv_cache.dtype),
        input_output_aliases={5: 0},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name="cp_write_kv_head_pages",
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


def window_page_align(sliding_window: int, page_size: int) -> int:
    """Align sliding-window start pages to the capped decode block stride."""
    return max(1, (sliding_window // 2) // page_size)


def window_position_pages(
    seq_len: jax.Array, sliding_window: int, page_size: int
) -> tuple[jax.Array, jax.Array]:
    """Return ``(position_pages, count)`` covering the decode window at ``seq_len``."""
    align = window_page_align(sliding_window, page_size)
    num = num_halo_pages(sliding_window, page_size) + align
    first = (
        (jnp.maximum(seq_len - (sliding_window - 1), 0) // page_size) // align
    ) * align
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
    rows = _window_rows(plan, rank, position_pages)
    return pack_kv_rows(jnp.take(keys, rows, axis=0), jnp.take(values, rows, axis=0))


def _window_local_pages(
    plan: ZigzagPlan, rank: Rank, position_pages: jax.Array
) -> jax.Array:
    pp = plan.piece_pages
    first, second = plan.rank_pieces(rank)
    piece = position_pages // pp
    local = jnp.where(
        piece == first, position_pages - first * pp, position_pages - second * pp + pp
    )
    return jnp.clip(local, 0, 2 * pp - 1)


def _window_rows(plan: ZigzagPlan, rank: Rank, position_pages: jax.Array) -> jax.Array:
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
    piece = position_pages // plan.piece_pages
    return jnp.where(piece < plan.world, piece, plan.num_pieces - 1 - piece)


def window_slots_from_rows(
    window_rows: jax.Array,
    owners: jax.Array,
    first_head: Rank,
    num_heads: int,
    head_dim: int,
) -> jax.Array:
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
    """Static attention configuration for CP prefill over `world` ranks."""

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
    fence_output: bool = False
    head_page_writes: bool = False
    prologue_eps: float | None = None
    fp8_out_rows: bool = False
    exchange_from_qkv: bool = False
    unmasked_interior: bool = False
    shared_kv_rotary_dims: int = 0
    paged_gather: bool = False
    prologue_dma_stores: bool = False
    prologue_fp8_queries: bool = False
    cache_all_pages: bool = False

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
                "Shared-KV pages are cached by the head-page kernel "
                "(requires head_page_writes=True)."
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
        if self.fence_output:
            kwargs["fence_output"] = True
        if self.unmasked_interior:
            kwargs["unmasked_interior"] = True
        if self.fp8_out_rows:
            kwargs["output_quant_dtype"] = jnp.float8_e4m3fn
        return kwargs


def _cache_pages(kv_cache: jax.Array, cfg: LayerConfig, cache_view: CacheView):
    native, page_stride, restore = cache_view(kv_cache)
    page_axis = 4 if cfg.kv_layout == configs.KVLayout.SEQ_ALONG_LANE else 1
    return native, page_stride, restore, native.shape[page_axis]


def _request_block_table(block_tables: jax.Array, seq_lens: jax.Array) -> jax.Array:
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


def _layer_output(out: Any, cfg: LayerConfig) -> tuple[jax.Array, ...]:
    return tuple(out) if cfg.fp8_out_rows else (out,)


def _gathered_pages(rows: jax.Array, head_dim: int, page_size: int) -> jax.Array:
    if rows.ndim == 2:
        return kv_pages_from_rows(rows, head_dim, page_size)
    return rows.reshape(rows.shape[0] // page_size, page_size, *rows.shape[1:])


def exchange_rows_int8(new_kv: jax.Array, cfg: LayerConfig) -> jax.Array:
    rows = new_kv if cfg.paged_gather else new_kv.reshape(new_kv.shape[0], -1)
    return jax.lax.bitcast_convert_type(rows, jnp.int8)


def cp_full_attention(
    kv_cache: jax.Array,
    *args: jax.Array,
    cfg: LayerConfig,
    cache_view: CacheView,
    from_q: bool = False,
    from_q8: bool = False,
) -> tuple[jax.Array, ...]:
    """Full-attention CP prefill entry (`q/k/v`, `q_hbm`, or `q_hbm+q_fp8`)."""
    if from_q8:
        if not cfg.prologue_fp8_queries:
            raise ValueError(
                "cp_full_attention(from_q8=True) needs prologue_fp8_queries"
            )
        q_hbm, q_fp8, kv_rows, seq_lens, block_tables, rank_arr = args
        query = jnp.zeros((q_hbm.shape[1], cfg.num_heads, cfg.head_dim), q_hbm.dtype)
    elif from_q:
        q_hbm, kv_rows, seq_lens, block_tables, rank_arr = args
        q_fp8 = None
        query = jnp.zeros((q_hbm.shape[1], cfg.num_heads, cfg.head_dim), q_hbm.dtype)
    else:
        query, kv_rows, seq_lens, block_tables, rank_arr = args
        q_hbm = q_fp8 = None
    rank = rank_arr[0]
    native, page_stride, restore, page_size = _cache_pages(kv_cache, cfg, cache_view)
    plan = cfg.plan(query.shape[0], page_size)
    rows = jax.lax.bitcast_convert_type(kv_rows, native.dtype)
    pages = _gathered_pages(rows, cfg.head_dim, page_size)
    first, count = tp_kv_heads(cfg.num_kv_heads, cfg.world, rank)
    table = _request_block_table(block_tables, seq_lens) * page_stride
    writes = full_attention_page_writes(plan, table, seq_lens[0])
    kwargs = cfg.rpa_kwargs()
    if q_hbm is not None:
        new_kv = jnp.zeros((query.shape[0], *pages.shape[2:]), pages.dtype)
        kwargs["prepared_inputs"] = (
            (q_hbm, new_kv) if q_fp8 is None else (q_hbm, new_kv, q_fp8)
        )
    tables = plan.gathered_page_table()
    if cfg.head_page_writes:
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
    assert cfg.sliding_window is not None
    halo_pages = num_halo_pages(cfg.sliding_window, page_size)
    return halo_pages, halo_pages + window_page_align(cfg.sliding_window, page_size)


def cp_sliding_exchange_rows(
    *args: Any, page_size: int | None = None, cfg: LayerConfig, source: str = "kv"
) -> jax.Array:
    """Pack this rank's sliding-window halo tails and decode-window pages as int8."""
    if page_size is None:
        *tensors, raw_page_size = args
        page_size = int(raw_page_size)
    else:
        tensors = list(args)
    seq_lens, rank = tensors[-2], tensors[-1][0]
    plan = cfg.plan(tensors[0].shape[0], page_size)
    halo_pages, _ = _sliding_exchange_pages(cfg, page_size)
    plan.check_halo_pages(halo_pages)
    assert cfg.sliding_window is not None
    position_pages, _ = window_position_pages(
        seq_lens[0], cfg.sliding_window, page_size
    )
    if source == "kv":
        keys, values = tensors[:2]
        rows = jnp.concatenate(
            [
                pack_piece_tails(keys, values, halo_pages, page_size),
                window_rows_held(keys, values, plan, rank, position_pages),
            ]
        )
        return jax.lax.bitcast_convert_type(rows, jnp.int8)
    if source == "prepared":
        new_kv = tensors[0]
        piece, halo = plan.piece_tokens, halo_pages * page_size
        rows = jnp.concatenate(
            [
                new_kv[piece - halo : piece],
                new_kv[2 * piece - halo : 2 * piece],
                jnp.take(new_kv, _window_rows(plan, rank, position_pages), axis=0),
            ]
        )
        return exchange_rows_int8(rows, cfg)
    if source == "qkv":
        qkv, cos, sin, q_norm_weight, k_norm_weight = tensors[:5]
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
    raise ValueError(f"Unsupported sliding exchange source: {source!r}")


def cp_sliding_attention(
    kv_cache: jax.Array,
    *args: jax.Array,
    cfg: LayerConfig,
    cache_view: CacheView,
    from_q: bool = False,
) -> tuple[jax.Array, ...]:
    """Sliding-window CP prefill layer entry point (`q/k/v` or `q_hbm + new_kv`)."""
    if cfg.sliding_window is None:
        raise ValueError("cp_sliding_attention needs a sliding window")
    if from_q:
        q_hbm, new_kv, gathered, seq_lens, block_tables, rank_arr = args
        num_rows, piece = q_hbm.shape[1], q_hbm.shape[1] // 2
        query = jnp.zeros((num_rows, cfg.num_heads, cfg.head_dim), q_hbm.dtype)
        keys = values = jnp.zeros(
            (num_rows, cfg.num_kv_heads, cfg.head_dim), new_kv.dtype
        )
        prepared_inputs: tuple[jax.Array, jax.Array] | None = (q_hbm, new_kv)

        def own_tail(halo: int, page_size: int) -> jax.Array:
            tail = new_kv[piece - halo : piece]
            return tail.reshape(halo // page_size, page_size, *new_kv.shape[1:])
    else:
        query, keys, values, gathered, seq_lens, block_tables, rank_arr = args
        piece = query.shape[0] // 2
        prepared_inputs = None

        def own_tail(halo: int, page_size: int) -> jax.Array:
            span = slice(piece - halo, piece)
            return pack_kv_pages(keys[span], values[span], page_size)

    rank = rank_arr[0]
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
        rows = gathered.reshape(cfg.world, num_rows, *row_shape)

        def fp8(x: jax.Array) -> jax.Array:
            return jax.lax.bitcast_convert_type(x, keys.dtype)
    else:
        rows = jax.lax.bitcast_convert_type(gathered, keys.dtype).reshape(
            cfg.world, num_rows, *row_shape
        )

        def fp8(x: jax.Array) -> jax.Array:
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
        *((width,) if len(row_shape) == 1 else (-1, cfg.head_dim)),
    )
    position_pages, count = window_position_pages(
        seq_lens[0], cfg.sliding_window, page_size
    )
    slots = fp8(
        window_slots_from_rows(
            window,
            window_page_owners(plan, position_pages),
            *tp_kv_heads(cfg.num_kv_heads, cfg.world, rank),
            cfg.head_dim,
        )
    )
    table = _request_block_table(block_tables, seq_lens) * page_stride
    writes = window_page_writes(table, position_pages, count)
    mine = cache_pages_from_slots(slots, cfg.kv_layout)
    return restore(write_kv_pages(native, mine, *writes)), *_layer_output(out, cfg)

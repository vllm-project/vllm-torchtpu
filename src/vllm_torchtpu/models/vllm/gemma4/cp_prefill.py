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
"""Zigzag context-parallel prefill attention kernels and page layout helpers.

Splits a single T-token prompt into ``2 * world`` page-aligned pieces and
assigns pieces ``r`` and ``2 * world - 1 - r`` to rank ``r`` so every rank
attends to the same number of causal ``(query, key)`` pairs. Each rank runs
batched RPA on its two pieces as two sub-requests at their absolute prompt
positions:

* Full attention: all-gathered FP8 K/V pages stay in rank order while an
  inverted zigzag page table maps each logical position page to its gathered
  slot, followed by ``write_kv_head_pages`` writing this rank's TP KV heads
  into the serving cache (both ``HEAD_ALONG_SUBLANE`` and ``SEQ_ALONG_LANE``).
* Sliding window: ranks exchange only their piece tails (halos) plus the decode
  window pages from sliced QKV rows via ``sliding_exchange_rows_from_qkv``,
  attend within ``[halo | piece]``, and cache this rank's TP KV heads of the
  final decode window.
"""

from __future__ import annotations

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
    utils,
    wrapper,
)
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    prologue as rpa_prologue,
)

Rank = int | jax.Array

_CPU_FALLBACK_VMEM_BYTES = 32 * 1024 * 1024
_HEAD_PAGE_VMEM_FRACTION = 0.8


def _get_vmem_capacity_bytes() -> int:
    """Query active TPU core VMEM capacity in bytes with a 32 MiB CPU fallback."""
    try:
        return int(pltpu.get_tpu_info().vmem_capacity_bytes)
    except ValueError:
        return _CPU_FALLBACK_VMEM_BYTES


@dataclasses.dataclass(frozen=True)
class ZigzagPlan:
    """Static zigzag split of a ``num_tokens`` prompt over ``world`` ranks."""

    num_tokens: int
    world: int
    page_size: int

    def __post_init__(self) -> None:
        unit = 2 * self.world * self.page_size
        if self.num_tokens <= 0 or self.num_tokens % unit:
            raise ValueError(
                f"Zigzag CP requires token count ({self.num_tokens}) to be a "
                f"positive multiple of 2 * world * page_size ({unit})."
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
        """Return the two piece indices owned by ``rank`` in local row order."""
        if isinstance(rank, int) and not 0 <= rank < self.world:
            raise ValueError(f"rank {rank} outside world {self.world}")
        return rank, self.num_pieces - 1 - rank

    def piece_owner(self, piece: int) -> tuple[int, int]:
        """Return ``(rank, slot)`` owning ``piece`` (``slot`` in ``{0, 1}``)."""
        if piece < self.world:
            return piece, 0
        return self.num_pieces - 1 - piece, 1

    def rank_slices(self, rank: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return token ``[start, end)`` ranges for ``rank``'s two pieces."""
        p = self.piece_tokens
        first, second = self.rank_pieces(rank)
        return (int(first) * p, (int(first) + 1) * p), (
            int(second) * p,
            (int(second) + 1) * p,
        )

    def gathered_page_table(self) -> np.ndarray:
        """Map logical position page -> page index in rank-major all-gather."""
        pp = self.piece_pages
        table = np.empty(self.num_pages, np.int32)
        for page in range(self.num_pages):
            rank, slot = self.piece_owner(page // pp)
            table[page] = rank * 2 * pp + slot * pp + page % pp
        return table

    def gathered_position_pages(self) -> np.ndarray:
        """Map all-gathered page index -> logical position page."""
        return np.argsort(self.gathered_page_table()).astype(np.int32)

    def check_halo_pages(self, halo_pages: int) -> None:
        """Validate that ``halo_pages`` fits within a single preceding piece."""
        if not 1 <= halo_pages <= self.piece_pages:
            raise ValueError(
                f"Halo of {halo_pages} pages must be between 1 and "
                f"piece_pages ({self.piece_pages})."
            )

    def sliding_page_tables(self, rank: Rank, halo_pages: int) -> jax.Array:
        """Map ``[2, num_pages]`` sub-request pages to ``[halo0 | halo1 | scratch]``."""
        self.check_halo_pages(halo_pages)
        pieces = jnp.stack(self.rank_pieces(rank)).astype(jnp.int32)
        local = jnp.arange(self.num_pages, dtype=jnp.int32) - (
            pieces[:, None] * self.piece_pages - halo_pages
        )
        base = jnp.arange(2, dtype=jnp.int32)[:, None] * halo_pages
        halo = base + jnp.clip(local, 0, halo_pages - 1)
        return jnp.where(local < halo_pages, halo, 2 * halo_pages)

    def sub_request_metadata(self, rank: Rank) -> tuple[jax.Array, ...]:
        """Return ``(cu_q_lens, kv_lens, distribution)`` for ``rank``'s pieces."""
        p = self.piece_tokens
        cu_q_lens = jnp.array([0, p, 2 * p], jnp.int32)
        kv_lens = (jnp.stack(self.rank_pieces(rank)).astype(jnp.int32) + 1) * p
        distribution = jnp.array([0, 0, 2], jnp.int32)
        return cu_q_lens, kv_lens, distribution


def make_zigzag_plan(num_tokens: int, world: int, page_size: int) -> ZigzagPlan:
    return ZigzagPlan(num_tokens=num_tokens, world=world, page_size=page_size)


def pack_kv_rows(k: jax.Array, v: jax.Array) -> jax.Array:
    """Pack ``[n, kv_heads, d]`` K and V into ``[n, 2 * kv_heads * d]`` rows."""
    n, kv_heads, head_dim = k.shape
    return jnp.concatenate([k, v], axis=-1).reshape(n, 2 * kv_heads * head_dim)


def kv_pages_from_rows(rows: jax.Array, head_dim: int, page_size: int) -> jax.Array:
    """Reshape ``pack_kv_rows`` rows into ``HEAD_ALONG_SUBLANE`` KV pages."""
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


def num_halo_pages(sliding_window: int, page_size: int) -> int:
    """Return preceding pages a sliding window of ``sliding_window`` reaches."""
    if sliding_window < 1:
        raise ValueError(f"sliding_window must be >= 1, got {sliding_window}")
    return pl.cdiv(sliding_window - 1, page_size)


def halo_neighbours(rank: Rank, world: int) -> tuple[Rank, Rank]:
    """Return ``(first_src, second_src)`` ranks holding the preceding pieces."""
    return (rank - 1) % world, (rank + 1) % world


def _halo_buffer(
    halos: jax.Array, own_first_tail: jax.Array, plan: ZigzagPlan, rank: Rank
) -> jax.Array:
    halo_pages = own_first_tail.shape[0]
    halo_first = jnp.where(
        rank == 0, jnp.zeros_like(own_first_tail), halos[:halo_pages]
    )
    halo_second = jnp.where(rank == plan.world - 1, own_first_tail, halos[halo_pages:])
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
    prepared_inputs: tuple[jax.Array, ...] | None = None,
    **rpa_kwargs: Any,
) -> Any:
    """Run batched RPA on ``rank``'s two zigzag pieces against ``kv_pages``."""
    cu_q_lens, kv_lens, distribution = plan.sub_request_metadata(rank)
    tables = jnp.broadcast_to(jnp.asarray(page_tables), (2, plan.num_pages))
    skip_kv_update = keys is None
    if skip_kv_update:
        shape = (q.shape[0], num_kv_heads, kv_pages.shape[-1])
        keys = values = jnp.zeros(shape, kv_pages.dtype)
    assert keys is not None and values is not None
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
    """Return ``(first_head, count)`` of KV heads owned by ``tp_rank``."""
    if num_kv_heads >= tp_size:
        if num_kv_heads % tp_size:
            raise ValueError(
                f"tp_size ({tp_size}) does not divide num_kv_heads ({num_kv_heads})"
            )
        count = num_kv_heads // tp_size
        return tp_rank * count, count
    if tp_size % num_kv_heads:
        raise ValueError(
            f"num_kv_heads ({num_kv_heads}) does not divide tp_size ({tp_size})"
        )
    return tp_rank * num_kv_heads // tp_size, 1


def cache_pages_from_slots(slots: jax.Array, kv_layout: configs.KVLayout) -> jax.Array:
    """Format ``[pages, page_size, 2 * local_kv_heads, head_dim]`` K/V slots."""
    num_pages, page_size, num_slots, head_dim = slots.shape
    packing = utils.get_dtype_packing(slots.dtype)
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        if num_slots != 2:
            raise NotImplementedError(
                "SEQ_ALONG_LANE expects 1 local KV head (2 slots), "
                f"got {num_slots // 2}"
            )
        pages = slots.transpose(0, 2, 3, 1)
        return pages.reshape(num_pages, 2, head_dim // packing, packing, page_size)
    padded = utils.align_to(num_slots, packing)
    if padded != num_slots:
        slots = jnp.pad(slots, ((0, 0), (0, 0), (0, padded - num_slots), (0, 0)))
    return slots.reshape(num_pages, page_size, padded // packing, packing, head_dim)


def _copy_page_kernel(_count_ref, _src_ref, _dst_ref, page_ref, _cache_ref, out_ref):
    out_ref[...] = page_ref[...]


def write_kv_pages(
    kv_cache: jax.Array,
    pages: jax.Array,
    src_pages: jax.Array,
    dst_pages: jax.Array,
    count: jax.Array,
    *,
    interpret: Any = False,
) -> jax.Array:
    """Copy ``pages[src_pages[:count]]`` into ``kv_cache[dst_pages[:count]]``."""
    if pages.shape[1:] != kv_cache.shape[1:] or pages.dtype != kv_cache.dtype:
        raise ValueError(
            f"pages {pages.shape} {pages.dtype} incompatible with "
            f"kv_cache {kv_cache.shape} {kv_cache.dtype}"
        )
    if src_pages.shape != dst_pages.shape or src_pages.ndim != 1:
        raise ValueError(
            f"src_pages {src_pages.shape} and dst_pages {dst_pages.shape} must "
            "be matching 1-D index arrays"
        )
    page_block = (1, *kv_cache.shape[1:])
    vmem_capacity = _get_vmem_capacity_bytes()

    def step(i, count_ref):
        return jnp.maximum(jnp.minimum(i, count_ref[0] - 1), 0)

    return pl.pallas_call(
        _copy_page_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(dst_pages.shape[0],),
            in_specs=[
                pl.BlockSpec(
                    page_block,
                    lambda i, c, s, _d: (s[step(i, c)], *(0,) * (len(page_block) - 1)),
                ),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(
                page_block,
                lambda i, c, _s, d: (d[step(i, c)], *(0,) * (len(page_block) - 1)),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct(kv_cache.shape, kv_cache.dtype),
        input_output_aliases={4: 0},
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_capacity),
        interpret=interpret,
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
    words = pltpu.bitcast(page_ref[...], jnp.uint32)
    odd = head_ref[0] % 2

    @pl.when(odd == 0)
    def _() -> None:
        out_ref[...] = pltpu.bitcast(words & jnp.uint32(0xFFFF), out_ref.dtype)

    @pl.when(odd == 1)
    def _() -> None:
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
):
    """Transpose one head's packed FP8 K/V page into ``SEQ_ALONG_LANE`` layout."""
    _, page_size, _, _, head_dim = page_ref.shape
    lanes = scratch_ref.shape[-1]
    chunks, groups = page_size // lanes, head_dim // 4
    words = pltpu.bitcast(page_ref[...], jnp.uint32).reshape(page_size, head_dim)

    def every_4th_row(i: int) -> jax.Array:
        rows = [
            scratch_ref[pl.ds(chunks * i + c, groups, stride=4 * chunks), :]
            for c in range(chunks)
        ]
        return jnp.concatenate(rows, axis=1)

    def store(shift: int) -> None:
        for slot in (0, 1):
            byte = (words >> (shift + 8 * slot)) & jnp.uint32(0xFF)
            transposed = byte.T
            for c in range(chunks):
                scratch_ref[pl.ds(c, head_dim, stride=chunks), :] = transposed[
                    :, c * lanes : (c + 1) * lanes
                ]
            w = every_4th_row(0)
            for i in range(1, 4):
                w = w | (every_4th_row(i) << (8 * i))
            out_ref[0, slot] = pltpu.bitcast(w, out_ref.dtype).reshape(
                groups, 4, page_size
            )

    odd = head_ref[0] % 2

    @pl.when(odd == 0)
    def _() -> None:
        store(0)

    @pl.when(odd == 1)
    def _() -> None:
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
    interpret: Any = False,
) -> jax.Array:
    """Write ``[first_head, first_head + num_heads)`` from ``pages`` to ``kv_cache``."""
    *_, page_size, rows, packing, head_dim = pages.shape
    if packing != 4 or kv_cache.dtype != pages.dtype:
        raise NotImplementedError(
            f"Head pages require packing-4 FP8 pages, got {pages.dtype} pages "
            f"and {kv_cache.dtype} cache."
        )
    half = num_heads == 1
    if not half and num_heads % 2:
        raise NotImplementedError(f"{num_heads} heads must be 1 or even.")
    head_rows = 1 if half else num_heads // 2
    if rows % head_rows:
        raise ValueError(f"{rows} word rows do not split into {head_rows}-row heads")
    in_block = (1, page_size, head_rows, packing, head_dim)
    scratch_shapes: list[Any] = []
    scratch_bytes = 0
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        if not half or page_size % 128:
            raise NotImplementedError(
                "SEQ_ALONG_LANE head pages require 1 head and 128-multiple page_size, "
                f"got {num_heads} heads of {page_size}-token pages."
            )
        out_block = (1, 2, head_dim // packing, packing, page_size)
        kernel: Callable[..., Any] = _seq_along_lane_head_page_kernel
        scratch_words = (head_dim * page_size // 128) * 128
        scratch_bytes = scratch_words * 4
        scratch_shapes.append(
            pltpu.VMEM((head_dim * page_size // 128, 128), jnp.uint32)
        )
    else:
        out_block = in_block
        kernel = functools.partial(_copy_head_page_kernel, half=half)
    if kv_cache.shape[1:] != out_block[1:]:
        raise ValueError(
            f"{kv_layout} cache shape {kv_cache.shape[1:]} does not match "
            f"{num_heads} head(s) of {pages.shape[1:]} pages."
        )

    vmem_capacity = _get_vmem_capacity_bytes()
    step_vmem_bytes = 2 * (math.prod(in_block) + math.prod(out_block)) + scratch_bytes
    if step_vmem_bytes > _HEAD_PAGE_VMEM_FRACTION * vmem_capacity:
        raise ValueError(
            f"write_kv_head_pages step needs {step_vmem_bytes} bytes of VMEM, "
            f"exceeding {int(_HEAD_PAGE_VMEM_FRACTION * vmem_capacity)} byte budget."
        )

    def step(i, count_ref):
        return jnp.maximum(jnp.minimum(i, count_ref[0] - 1), 0)

    def page_index(i, count_ref, src_ref, dst_ref, head_ref):
        del dst_ref
        return (src_ref[step(i, count_ref)], 0, head_ref[0] // 2 // head_rows, 0, 0)

    def cache_index(i, count_ref, src_ref, dst_ref, head_ref):
        del src_ref, head_ref
        return (dst_ref[step(i, count_ref)], 0, 0, 0, 0)

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(dst_pages.shape[0],),
            in_specs=[
                pl.BlockSpec(in_block, page_index),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(out_block, cache_index),
            scratch_shapes=scratch_shapes,
        ),
        out_shape=jax.ShapeDtypeStruct(kv_cache.shape, kv_cache.dtype),
        input_output_aliases={5: 0},
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_capacity),
        interpret=interpret,
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
    """Return ``(src, dst, count)`` caching every prompt page below ``seq_len``."""
    src = jnp.asarray(plan.gathered_page_table())
    dst = jnp.take(block_table, jnp.arange(plan.num_pages), mode="clip")
    count = jnp.minimum(
        utils.align_to(seq_len, plan.page_size) // plan.page_size, plan.num_pages
    )
    return src, dst, count


def window_position_pages(
    seq_len: jax.Array, sliding_window: int, page_size: int
) -> tuple[jax.Array, jax.Array]:
    """Return ``(position_pages, count)`` reached by the next decode token."""
    num = num_halo_pages(sliding_window, page_size) + 1
    first = jnp.maximum(seq_len - (sliding_window - 1), 0) // page_size
    pages = first + jnp.arange(num, dtype=jnp.int32)
    last = utils.align_to(seq_len, page_size) // page_size
    return pages, jnp.minimum(last - first, num)


def window_page_writes(
    block_table: jax.Array, position_pages: jax.Array, count: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return ``(src, dst, count)`` caching ``position_pages`` into ``block_table``."""
    src = jnp.arange(position_pages.shape[0], dtype=jnp.int32)
    return src, jnp.take(block_table, position_pages, mode="clip"), count


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
    """Return the rank owning each logical page in ``position_pages``."""
    piece = position_pages // plan.piece_pages
    return jnp.where(piece < plan.world, piece, plan.num_pieces - 1 - piece)


def window_slots_from_rows(
    window_rows: jax.Array,
    owners: jax.Array,
    first_head: Rank,
    num_heads: int,
    head_dim: int,
) -> jax.Array:
    """Select ``[window_pages, page_size, 2 * num_heads, head_dim]`` K/V slots."""
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
    """Configuration for one Gemma 4 context-parallel prefill attention layer."""

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
    prologue_eps: float
    fp8_out_rows: bool = False
    unmasked_interior: bool = True

    def __post_init__(self) -> None:
        if self.sliding_window is not None:
            return
        heads = self.local_kv_heads
        if self.kv_layout == configs.KVLayout.SEQ_ALONG_LANE and heads != 1:
            raise NotImplementedError(
                "Full-attention head page writes support 1 local KV head in "
                f"SEQ_ALONG_LANE layout, got {heads}."
            )
        if heads != 1 and heads % 2:
            raise NotImplementedError(
                "Full-attention head page writes support 1 or an even number of "
                f"local KV heads, got {heads}."
            )

    @property
    def local_kv_heads(self) -> int:
        return tp_kv_heads(self.num_kv_heads, self.world, 0)[1]

    def plan(self, num_rows: int, page_size: int) -> ZigzagPlan:
        return make_zigzag_plan(num_rows * self.world, self.world, page_size)

    def rpa_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = dict(
            num_kv_heads=self.num_kv_heads,
            sm_scale=self.sm_scale,
            soft_cap=self.soft_cap,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
            fp8_compute=self.fp8_compute,
            unmasked_interior=self.unmasked_interior,
        )
        if self.sliding_window is not None:
            kwargs["sliding_window"] = self.sliding_window
        if self.fp8_out_rows:
            kwargs["output_quant_dtype"] = jnp.float8_e4m3fn
        return kwargs


def _cache_pages(kv_cache: jax.Array, cfg: LayerConfig, cache_view: CacheView):
    native, page_stride, restore = cache_view(kv_cache)
    page_axis = 4 if cfg.kv_layout == configs.KVLayout.SEQ_ALONG_LANE else 1
    return native, page_stride, restore, native.shape[page_axis]


def _request_block_table(block_tables: jax.Array, seq_lens: jax.Array) -> jax.Array:
    return block_tables[: block_tables.shape[0] // seq_lens.shape[0]]


def _layer_output(out: Any, cfg: LayerConfig) -> tuple[jax.Array, ...]:
    return tuple(out) if cfg.fp8_out_rows else (out,)


def qkv_prologue(
    qkv: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    q_norm_weight: jax.Array,
    k_norm_weight: jax.Array,
    *,
    cfg: LayerConfig,
) -> tuple[jax.Array, ...]:
    """Run fused QKV RMSNorm + RoPE + FP8 KV prologue for a CP layer."""
    return rpa_prologue.qkv_to_rpa_inputs(
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
        v_rms_norm=True,
        emit_fp8_queries=cfg.fp8_compute,
    )


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
    """Compute halo + decode-window FP8 K/V rows (as int8) from sliced QKV rows."""
    assert cfg.sliding_window is not None
    rank_scalar = rank[0]
    plan = cfg.plan(qkv.shape[0], page_size)
    halo_pages = num_halo_pages(cfg.sliding_window, page_size)
    plan.check_halo_pages(halo_pages)
    position_pages, _ = window_position_pages(
        seq_lens[0], cfg.sliding_window, page_size
    )
    pick = functools.partial(
        _exchange_rows_of,
        plan=plan,
        rank=rank_scalar,
        position_pages=position_pages,
        halo_pages=halo_pages,
    )
    _, new_kv = rpa_prologue.qkv_to_rpa_inputs(
        pick(qkv),
        pick(cos),
        pick(sin),
        q_norm_weight,
        k_norm_weight,
        num_q_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        eps=cfg.prologue_eps,
        k_scale=cfg.k_scale,
        v_scale=cfg.v_scale,
        v_rms_norm=True,
        emit_fp8_queries=False,
    )
    raw = jax.lax.bitcast_convert_type(new_kv, jnp.int8)
    return raw.reshape(raw.shape[0], -1)


def _run_full_attention(
    kv_cache: jax.Array,
    q_hbm: jax.Array,
    q_fp8: jax.Array | None,
    kv_rows: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    rank: jax.Array,
    cfg: LayerConfig,
    cache_view: CacheView,
) -> tuple[jax.Array, ...]:
    native, page_stride, restore, page_size = _cache_pages(kv_cache, cfg, cache_view)
    num_rows = q_hbm.shape[1]
    plan = cfg.plan(num_rows, page_size)
    rows = jax.lax.bitcast_convert_type(kv_rows, native.dtype)
    pages = kv_pages_from_rows(rows, cfg.head_dim, page_size)
    first, count = tp_kv_heads(cfg.num_kv_heads, cfg.world, rank[0])
    table = _request_block_table(block_tables, seq_lens) * page_stride
    writes = full_attention_page_writes(plan, table, seq_lens[0])

    query = jnp.zeros((num_rows, cfg.num_heads, cfg.head_dim), q_hbm.dtype)
    new_kv = jnp.zeros((num_rows, *pages.shape[2:]), pages.dtype)
    prepared: tuple[jax.Array, ...] = (
        (q_hbm, new_kv) if q_fp8 is None else (q_hbm, new_kv, q_fp8)
    )
    out, pages = cp_prefill_attention(
        query,
        pages,
        plan.gathered_page_table(),
        plan,
        rank[0],
        return_kv_pages=True,
        prepared_inputs=prepared,
        **cfg.rpa_kwargs(),
    )
    new_native = write_kv_head_pages(
        native, pages, first, *writes, num_heads=count, kv_layout=cfg.kv_layout
    )
    return restore(new_native), *_layer_output(out, cfg)


def full_attention_layer(
    kv_cache: jax.Array,
    q_hbm: jax.Array,
    kv_rows: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    rank: jax.Array,
    *,
    cfg: LayerConfig,
    cache_view: CacheView,
) -> tuple[jax.Array, ...]:
    """Run full CP prefill attention from prologue ``q_hbm`` and gathered K/V."""
    return _run_full_attention(
        kv_cache, q_hbm, None, kv_rows, seq_lens, block_tables, rank, cfg, cache_view
    )


def full_attention_layer_fp8_q(
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
) -> tuple[jax.Array, ...]:
    """Run full CP prefill attention with pre-quantized FP8 queries."""
    return _run_full_attention(
        kv_cache, q_hbm, q_fp8, kv_rows, seq_lens, block_tables, rank, cfg, cache_view
    )


def _run_sliding_attention(
    kv_cache: jax.Array,
    q_hbm: jax.Array,
    new_kv: jax.Array,
    q_fp8: jax.Array | None,
    gathered: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    rank: jax.Array,
    cfg: LayerConfig,
    cache_view: CacheView,
) -> tuple[jax.Array, ...]:
    if cfg.sliding_window is None:
        raise ValueError("sliding_attention_layer requires a sliding window")
    native, page_stride, restore, page_size = _cache_pages(kv_cache, cfg, cache_view)
    num_rows, piece = q_hbm.shape[1], q_hbm.shape[1] // 2
    plan = cfg.plan(num_rows, page_size)
    halo_pages = num_halo_pages(cfg.sliding_window, page_size)
    window_pages = halo_pages + 1
    halo = halo_pages * page_size
    width = 2 * cfg.num_kv_heads * cfg.head_dim
    exchange_rows = 2 * halo + window_pages * page_size
    if gathered.shape != (cfg.world * exchange_rows, width):
        raise ValueError(
            f"Exchange rows {gathered.shape} do not match "
            f"({cfg.world * exchange_rows}, {width})."
        )

    rows = gathered.reshape(cfg.world, exchange_rows, width)
    rank_scalar = rank[0]
    first_src, second_src = halo_neighbours(rank_scalar, cfg.world)
    halo_bytes = jnp.concatenate(
        [rows[first_src, :halo], rows[second_src, halo : 2 * halo]]
    )
    halos = kv_pages_from_rows(
        jax.lax.bitcast_convert_type(halo_bytes, new_kv.dtype),
        cfg.head_dim,
        page_size,
    )
    own_tail = new_kv[piece - halo : piece].reshape(
        halo_pages, page_size, *new_kv.shape[1:]
    )
    local = _halo_buffer(halos, own_tail, plan, rank_scalar)
    tables = plan.sliding_page_tables(rank_scalar, halo_pages)

    query = jnp.zeros((num_rows, cfg.num_heads, cfg.head_dim), q_hbm.dtype)
    keys = jnp.zeros((num_rows, cfg.num_kv_heads, cfg.head_dim), new_kv.dtype)
    prepared: tuple[jax.Array, ...] = (
        (q_hbm, new_kv) if q_fp8 is None else (q_hbm, new_kv, q_fp8)
    )
    out = cp_prefill_attention(
        query,
        local,
        tables,
        plan,
        rank_scalar,
        keys=keys,
        values=keys,
        prepared_inputs=prepared,
        **cfg.rpa_kwargs(),
    )

    window = rows[:, 2 * halo :].reshape(cfg.world, window_pages, page_size, width)
    position_pages, count = window_position_pages(
        seq_lens[0], cfg.sliding_window, page_size
    )
    slots = window_slots_from_rows(
        window,
        window_page_owners(plan, position_pages),
        *tp_kv_heads(cfg.num_kv_heads, cfg.world, rank_scalar),
        cfg.head_dim,
    )
    slots_fp8 = jax.lax.bitcast_convert_type(slots, new_kv.dtype)
    table = _request_block_table(block_tables, seq_lens) * page_stride
    writes = window_page_writes(table, position_pages, count)
    mine = cache_pages_from_slots(slots_fp8, cfg.kv_layout)
    return restore(write_kv_pages(native, mine, *writes)), *_layer_output(out, cfg)


def sliding_attention_layer(
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
) -> tuple[jax.Array, ...]:
    """Run sliding-window CP prefill attention from prologue outputs."""
    return _run_sliding_attention(
        kv_cache,
        q_hbm,
        new_kv,
        None,
        gathered,
        seq_lens,
        block_tables,
        rank,
        cfg,
        cache_view,
    )


def sliding_attention_layer_fp8_q(
    kv_cache: jax.Array,
    q_hbm: jax.Array,
    new_kv: jax.Array,
    q_fp8: jax.Array,
    gathered: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    rank: jax.Array,
    *,
    cfg: LayerConfig,
    cache_view: CacheView,
) -> tuple[jax.Array, ...]:
    """Run sliding-window CP prefill attention with pre-quantized FP8 queries."""
    return _run_sliding_attention(
        kv_cache,
        q_hbm,
        new_kv,
        q_fp8,
        gathered,
        seq_lens,
        block_tables,
        rank,
        cfg,
        cache_view,
    )

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
"""Custom BufferedRef overrides for StreamIndex Top-K kernel."""

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import metadata


def _step_offset(meta_ref: Any, chunk_idx: int):
    """Flat schedule row where chunk `chunk_idx`'s pipeline steps begin."""
    return sum(meta_ref.num_steps[m] for m in range(chunk_idx))


def _token_window(
    refs: tuple[Any, ...],
    p_id: int | jax.Array,
    chunk_tokens: int,
    *,
    bq_sz: int,
    seq_batch_size: int,
    chunk_idx: int = 0,
):
    """Flat-token rows one pipeline step covers, as (token_start, size)."""
    _, cu_q_lens_ref, meta_ref, start_end_seq_idx_ref = refs
    chunk_start = start_end_seq_idx_ref[2]
    p_id = p_id + _step_offset(meta_ref, chunk_idx)
    seq_idx = meta_ref.batch_tile_idx[p_id]
    lo = jnp.maximum(cu_q_lens_ref[seq_idx], chunk_start)

    if seq_batch_size > 1:
        return lo, seq_batch_size
    token_start = lo + meta_ref.bq_idx[p_id] * bq_sz
    if bq_sz == 1:
        return token_start, 1
    hi = jnp.minimum(cu_q_lens_ref[seq_idx + 1], chunk_start + chunk_tokens)
    return token_start, jnp.maximum(0, jnp.minimum(bq_sz, hi - token_start))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StreamIndexKVBufferedRef(pltpu.BufferedRef):
    """Handles fetching paged KV cache into VMEM buffers across sequence sublanes."""

    bkv_p: int = dataclasses.field(default=1, metadata=dict(static=True))
    page_size_per_kv_packing: int = dataclasses.field(
        default=64, metadata=dict(static=True)
    )
    kv_packing: int = dataclasses.field(default=4, metadata=dict(static=True))
    pages_per_seq: int = dataclasses.field(default=1, metadata=dict(static=True))
    seq_batch_size: int = dataclasses.field(default=1, metadata=dict(static=True))
    chunk_idx: int = dataclasses.field(default=0, metadata=dict(static=True))
    table_expansion: tuple[int, int, int] = dataclasses.field(
        default=(1, 1, 0), metadata=dict(static=True)
    )
    coalesce_entries: bool = dataclasses.field(
        default=False, metadata=dict(static=True)
    )
    # > 0: every copy moves only lanes [0, lanes) of each storage row (the
    # VMEM window is `lanes` wide); 0 moves whole rows.
    lanes: int = dataclasses.field(default=0, metadata=dict(static=True))
    # q38 PORT-BL IDX (ported from Benliang Wang, 2ce02bb1a;
    # VLLM_TPU_QSA_INDEXER_DENSE_STEP): page ids from the block's first table
    # row with the static offset, no per-page carry or end clamp (`copy_in`).
    lean_addresses: bool = dataclasses.field(
        default=False, metadata=dict(static=True)
    )
    # q38 PORT-BL IDX (VLLM_TPU_QSA_INDEXER_TAIL_SKIP): (group pages, keys per
    # page, block keys, compression shift, cp_size, interleave_c): copy only
    # the groups of `group pages` kernel pages that hold keys of the step's
    # sequence; the source tuple then also carries seq_lens_ref and
    # cp_rank_ref. None copies whole blocks.
    tail_skip: tuple[int, int, int, int, int, int] | None = dataclasses.field(
        default=None, metadata=dict(static=True)
    )

    @classmethod
    def create(  # pytype: disable=signature-mismatch
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        bkv_p: int,
        page_size_per_kv_packing: int,
        kv_packing: int,
        pages_per_seq: int,
        seq_batch_size: int,
        grid_rank: int = 1,
        chunk_idx: int = 0,
        table_expansion: tuple[int, int, int] = (1, 1, 0),
        coalesce_entries: bool = False,
        lanes: int = 0,
        lean_addresses: bool = False,
        tail_skip: tuple[int, int, int, int, int, int] | None = None,
    ):
        assert buffer_type == pltpu.BufferType.INPUT
        standard_ref = pltpu.BufferedRef.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            grid_rank=grid_rank,
            use_lookahead=use_lookahead,
        )
        return cls(
            bkv_p=bkv_p,
            page_size_per_kv_packing=page_size_per_kv_packing,
            kv_packing=kv_packing,
            pages_per_seq=pages_per_seq,
            seq_batch_size=seq_batch_size,
            chunk_idx=chunk_idx,
            table_expansion=table_expansion,
            coalesce_entries=coalesce_entries,
            lanes=lanes,
            lean_addresses=lean_addresses,
            tail_skip=tail_skip,
            **{
                f.name: getattr(standard_ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    @classmethod
    def input(  # pytype: disable=signature-mismatch
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_count: int,
        use_lookahead: bool,
        bkv_p: int,
        page_size_per_kv_packing: int,
        kv_packing: int,
        pages_per_seq: int,
        seq_batch_size: int,
        grid_rank: int = 1,
        chunk_idx: int = 0,
        table_expansion: tuple[int, int, int] = (1, 1, 0),
        coalesce_entries: bool = False,
        lanes: int = 0,
        lean_addresses: bool = False,
        tail_skip: tuple[int, int, int, int, int, int] | None = None,
        **kwargs: Any,
    ):
        return cls.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=pltpu.BufferType.INPUT,
            buffer_count=buffer_count,
            use_lookahead=use_lookahead,
            bkv_p=bkv_p,
            page_size_per_kv_packing=page_size_per_kv_packing,
            kv_packing=kv_packing,
            pages_per_seq=pages_per_seq,
            seq_batch_size=seq_batch_size,
            grid_rank=grid_rank,
            chunk_idx=chunk_idx,
            table_expansion=table_expansion,
            coalesce_entries=coalesce_entries,
            lanes=lanes,
            lean_addresses=lean_addresses,
            tail_skip=tail_skip,
        )

    def row_source(self, cache_ref: Any, index: Any) -> Any:
        """The copy source ``cache_ref.at[index]``: with ``lanes`` set, only
        lanes ``[0, lanes)`` of its last (storage lane) axis, as one indexer."""
        if not self.lanes:
            return cache_ref.at[index]
        return cache_ref.at[index, ..., pl.ds(0, self.lanes)]

    def page_id(self, page_indices_ref: jax.Ref, seq: Any, start: Any, i: int) -> Any:
        """Physical page of kernel page ``start + i`` of sequence ``seq``.

        With ``table_expansion == (expansion, stride, offset)`` each table
        entry stands for ``expansion`` consecutive kernel pages: kernel page k
        is ``entry[k // expansion] * stride + offset + k % expansion``
        (``pages_per_seq`` counts kernel pages, a whole number of entries per
        sequence). Out-of-range kernel pages clamp to the last kernel page,
        exactly as on the pre-expanded table.

        ``i`` is static: the KV block's copy loop divides its first page once
        and carries the static offset, keeping integer division off the
        scalar unit's per-page path.
        """
        expansion, stride, offset = self.table_expansion
        last = page_indices_ref.shape[0] * expansion - 1
        base = seq * self.pages_per_seq + start
        if expansion == 1:
            return page_indices_ref[jnp.minimum(base + i, last)]
        entry0 = base // expansion
        run = base - entry0 * expansion + i % expansion
        carry = run >= expansion
        entry = entry0 + i // expansion + carry.astype(entry0.dtype)
        run = run - jnp.where(carry, expansion, 0)
        over = base + i > last
        entry = jnp.where(over, page_indices_ref.shape[0] - 1, entry)
        run = jnp.where(over, expansion - 1, run)
        return page_indices_ref[entry] * stride + offset + run

    # ---- q38 PORT-BL IDX (ported from Benliang Wang <benliangw@google.com>,
    # commit 2ce02bb1a on benliangw/q38fh-integ-pin-5b44702aea): tail skip and
    # lean addresses. Taken: his `_num_groups` / `_groups_with_keys` /
    # `_for_groups` and his copy loops; changed for v8: every copy reads
    # through `row_source` (VLLM_TPU_QSA_INDEXER_KEY_LANES narrow copies), and
    # with both off `copy_in` runs today's loops verbatim (same programs).

    def _num_groups(self):
        return -(-self.bkv_p // self.tail_skip[0])

    def _groups_with_keys(self, src_ref: Any, p_id: Any) -> Any:
        """Page groups of step `p_id`'s block that hold keys of its sequence,
        at least one."""
        _, _, meta_ref, seq_lens_ref, cp_rank_ref = src_ref
        group, keys_per_page, bkv_sz, shift, cp_size, interleave_c = self.tail_skip
        kv = seq_lens_ref[meta_ref.batch_tile_idx[p_id]] >> shift
        local = metadata.cp_local_length(kv, cp_rank_ref[0], cp_size, interleave_c)
        rem = local - meta_ref.bkv_idx[p_id] * bkv_sz
        group_keys = group * keys_per_page
        return jnp.clip((rem + group_keys - 1) // group_keys, 1, self._num_groups())

    def _for_groups(self, src_ref: Any, p_id: Any, body: Any) -> None:
        """body(first page, pages) for each page group with keys of the step's
        sequence; one call over the whole block without tail_skip."""
        if self.tail_skip is None:
            body(0, self.bkv_p)
            return
        group = self.tail_skip[0]
        count = self._groups_with_keys(src_ref, p_id)
        for g in range(self._num_groups()):
            run = functools.partial(body, g * group, min(group, self.bkv_p - g * group))
            if g == 0:
                run()
            else:
                pl.when(g < count)(run)

    def _copy_in_ported(  # pytype: disable=attribute-error
        self,
        src_ref: tuple[Any, ...],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        """`copy_in` with tail skip and / or lean addresses (his loops)."""
        cache_kv_hbm, page_indices_ref, meta_ref = src_ref[:3]
        p_id = grid_indices[0] + _step_offset(meta_ref, self.chunk_idx)
        batch_tile_idx = meta_ref.batch_tile_idx[p_id]
        bkv_idx = meta_ref.bkv_idx[p_id]

        assert self.sem_recvs is not None
        assert self.window_ref is not None
        assert isinstance(cache_kv_hbm, jax.Ref)
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst: Any = self.window_ref.at[slot]  # pytype: disable=attribute-error

        reshaped_cache_hbm_ref: Any = (
            cache_kv_hbm.reshape(  # pytype: disable=attribute-error
                cache_kv_hbm.shape[0] * self.page_size_per_kv_packing,
                self.kv_packing,
                cache_kv_hbm.shape[-1],
            )
        )
        max_hbm_pages = reshaped_cache_hbm_ref.shape[0]
        num_page_indices = page_indices_ref.shape[0]
        effective_seq_idx = batch_tile_idx

        expansion = self.table_expansion[0]
        if self.coalesce_entries and expansion > 1 and self.bkv_p % expansion == 0:
            run = expansion * self.page_size_per_kv_packing
            _, stride, page_offset = self.table_expansion
            for batch_idx in range(self.seq_batch_size):
                kv_p_start = bkv_idx * self.bkv_p
                # Sequences and blocks start on whole entries (pages_per_seq
                # and bkv_p are multiples of the expansion), so entry e of the
                # block is its first entry plus e. An entry past the table
                # clamps to the last one: its keys lie past the sequence and
                # are masked, whatever page they come from.
                first = (effective_seq_idx + batch_idx) * (
                    self.pages_per_seq // expansion
                ) + bkv_idx * (self.bkv_p // expansion)

                def entries(first_page, pages, batch_idx=batch_idx,
                            kv_p_start=kv_p_start, first=first):
                    for e in range(first_page // expansion,
                                   (first_page + pages) // expansion):
                        if self.lean_addresses:
                            entry = jnp.minimum(first + e, num_page_indices - 1)
                            page = page_indices_ref[entry] * stride + page_offset
                        else:
                            page = self.page_id(
                                page_indices_ref,
                                effective_seq_idx + batch_idx,
                                kv_p_start,
                                e * expansion,
                            )
                        offset = jnp.minimum(
                            page * self.page_size_per_kv_packing,
                            jnp.maximum(0, max_hbm_pages - run),
                        )
                        pltpu.make_async_copy(
                            self.row_source(reshaped_cache_hbm_ref, pl.ds(offset, run)),
                            vmem_dst.at[  # pytype: disable=attribute-error
                                batch_idx, pl.ds(e * run, run)
                            ],
                            sem,
                        ).start()

                self._for_groups(src_ref, p_id, entries)
            return

        # With whole blocks per sequence row (pages_per_seq a multiple of
        # bkv_p) a block's pages never leave its row, so the end clamp of
        # `page_id` is dropped.
        lean = (
            self.lean_addresses
            and expansion == 1
            and self.pages_per_seq % self.bkv_p == 0
        )
        for batch_idx in range(self.seq_batch_size):
            kv_p_start = bkv_idx * self.bkv_p
            row = (effective_seq_idx + batch_idx) * self.pages_per_seq + kv_p_start

            def pages_of(first_page, pages, batch_idx=batch_idx,
                         kv_p_start=kv_p_start, row=row):
                for i in range(first_page, first_page + pages):
                    if lean:
                        page = page_indices_ref[row + i]
                    else:
                        page = self.page_id(
                            page_indices_ref, effective_seq_idx + batch_idx,
                            kv_p_start, i,
                        )
                    safe_page_offset = jnp.minimum(
                        page * self.page_size_per_kv_packing,
                        jnp.maximum(0, max_hbm_pages - self.page_size_per_kv_packing),
                    )
                    pltpu.make_async_copy(
                        self.row_source(
                            reshaped_cache_hbm_ref,
                            pl.ds(safe_page_offset, self.page_size_per_kv_packing),
                        ),
                        vmem_dst.at[  # pytype: disable=attribute-error
                            batch_idx,
                            pl.ds(
                                i * self.page_size_per_kv_packing,
                                self.page_size_per_kv_packing,
                            ),
                        ],
                        sem,
                    ).start()

            self._for_groups(src_ref, p_id, pages_of)

    # ---- end q38 PORT-BL IDX

    def copy_in(  # pytype: disable=attribute-error
        self,
        src_ref: tuple[jax.Ref, jax.Ref, Any],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        if self.tail_skip is not None or self.lean_addresses:
            # q38 PORT-BL IDX: tail skip / lean addresses (above).
            return self._copy_in_ported(src_ref, grid_indices)
        cache_kv_hbm, page_indices_ref, meta_ref = src_ref
        p_id = grid_indices[0] + _step_offset(meta_ref, self.chunk_idx)
        batch_tile_idx = meta_ref.batch_tile_idx[p_id]
        bkv_idx = meta_ref.bkv_idx[p_id]

        assert self.sem_recvs is not None
        assert self.window_ref is not None
        assert isinstance(cache_kv_hbm, jax.Ref)
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst: Any = self.window_ref.at[slot]  # pytype: disable=attribute-error

        reshaped_cache_hbm_ref: Any = (
            cache_kv_hbm.reshape(  # pytype: disable=attribute-error
                cache_kv_hbm.shape[0] * self.page_size_per_kv_packing,
                self.kv_packing,
                cache_kv_hbm.shape[-1],
            )
        )
        max_hbm_pages = reshaped_cache_hbm_ref.shape[0]
        num_page_indices = page_indices_ref.shape[0]
        effective_seq_idx = batch_tile_idx

        expansion = self.table_expansion[0]
        if self.coalesce_entries and expansion > 1 and self.bkv_p % expansion == 0:
            # Blocks start on whole table entries and each entry's kernel
            # pages are consecutive in HBM: one DMA per entry instead of one
            # per kernel page (same bytes, same order, `expansion`x fewer
            # descriptors). Clamped tails read in-bounds pages past the
            # sequence, which the scores mask like any other.
            run = expansion * self.page_size_per_kv_packing
            for batch_idx in range(self.seq_batch_size):
                kv_p_start = bkv_idx * self.bkv_p
                for e in range(self.bkv_p // expansion):
                    page = self.page_id(
                        page_indices_ref,
                        effective_seq_idx + batch_idx,
                        kv_p_start,
                        e * expansion,
                    )
                    offset = jnp.minimum(
                        page * self.page_size_per_kv_packing,
                        jnp.maximum(0, max_hbm_pages - run),
                    )
                    pltpu.make_async_copy(
                        self.row_source(reshaped_cache_hbm_ref, pl.ds(offset, run)),
                        vmem_dst.at[  # pytype: disable=attribute-error
                            batch_idx, pl.ds(e * run, run)
                        ],
                        sem,
                    ).start()
            return

        for batch_idx in range(self.seq_batch_size):
            kv_p_start = bkv_idx * self.bkv_p
            for i in range(self.bkv_p):
                page = self.page_id(
                    page_indices_ref, effective_seq_idx + batch_idx, kv_p_start, i
                )
                safe_page_offset = jnp.minimum(
                    page * self.page_size_per_kv_packing,
                    jnp.maximum(0, max_hbm_pages - self.page_size_per_kv_packing),
                )
                pltpu.make_async_copy(
                    self.row_source(
                        reshaped_cache_hbm_ref,
                        pl.ds(safe_page_offset, self.page_size_per_kv_packing),
                    ),
                    vmem_dst.at[  # pytype: disable=attribute-error
                        batch_idx,
                        pl.ds(
                            i * self.page_size_per_kv_packing,
                            self.page_size_per_kv_packing,
                        ),
                    ],
                    sem,
                ).start()

    def wait_in(
        self,
        src_ref: tuple[jax.Ref, jax.Ref, Any],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        assert self.is_input
        if not self.is_buffered:
            return
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst: Any = self.window_ref.at[slot]  # pytype: disable=attribute-error
        if self.tail_skip is None:
            pltpu.make_async_copy(
                vmem_dst,
                vmem_dst,
                sem,
            ).wait()
            return
        # q38 PORT-BL IDX (tail skip, his wait): the bytes of the groups
        # copy_in issued for this step, group by group.
        p_id = grid_indices[0] + _step_offset(src_ref[2], self.chunk_idx)
        ps = self.page_size_per_kv_packing

        def wait_group(first_page, pages):
            rows = vmem_dst.at[:, pl.ds(first_page * ps, pages * ps)]
            pltpu.make_async_copy(rows, rows, sem).wait()

        self._for_groups(src_ref, p_id, wait_group)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StreamIndexKVSeqAlongLaneBufferedRef(pltpu.BufferedRef):
    """Fetches paged KV cache into VMEM for the SEQ_ALONG_LANE layout.

    The HBM cache is `[total_pages, kv_sublane_groups, kv_packing, page_size]`,
    i.e. one page is a single contiguous block whose minor-most dimension is the
    token (lane) dimension. Gathering `bkv_p` pages into one block therefore
    means writing each page into a *lane* window of the VMEM buffer, instead of
    the sublane window used by the HEAD_ALONG_SUBLANE layout.
    """

    bkv_p: int = dataclasses.field(default=1, metadata=dict(static=True))
    page_size: int = dataclasses.field(default=128, metadata=dict(static=True))
    pages_per_seq: int = dataclasses.field(default=1, metadata=dict(static=True))
    seq_batch_size: int = dataclasses.field(default=1, metadata=dict(static=True))
    chunk_idx: int = dataclasses.field(default=0, metadata=dict(static=True))

    @classmethod
    def create(  # pytype: disable=signature-mismatch
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        bkv_p: int,
        page_size: int,
        pages_per_seq: int,
        seq_batch_size: int,
        grid_rank: int = 1,
        chunk_idx: int = 0,
    ):
        assert buffer_type == pltpu.BufferType.INPUT
        standard_ref = pltpu.BufferedRef.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            grid_rank=grid_rank,
            use_lookahead=use_lookahead,
        )
        return cls(
            bkv_p=bkv_p,
            page_size=page_size,
            pages_per_seq=pages_per_seq,
            seq_batch_size=seq_batch_size,
            chunk_idx=chunk_idx,
            **{
                f.name: getattr(standard_ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    @classmethod
    def input(  # pytype: disable=signature-mismatch
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_count: int,
        use_lookahead: bool,
        bkv_p: int,
        page_size: int,
        pages_per_seq: int,
        seq_batch_size: int,
        grid_rank: int = 1,
        chunk_idx: int = 0,
        **kwargs: Any,
    ):
        return cls.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=pltpu.BufferType.INPUT,
            buffer_count=buffer_count,
            use_lookahead=use_lookahead,
            bkv_p=bkv_p,
            page_size=page_size,
            pages_per_seq=pages_per_seq,
            seq_batch_size=seq_batch_size,
            grid_rank=grid_rank,
            chunk_idx=chunk_idx,
        )

    def copy_in(  # pytype: disable=attribute-error
        self,
        src_ref: tuple[jax.Ref, jax.Ref, Any],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        cache_kv_hbm, page_indices_ref, meta_ref = src_ref
        p_id = grid_indices[0] + _step_offset(meta_ref, self.chunk_idx)
        batch_tile_idx = meta_ref.batch_tile_idx[p_id]
        bkv_idx = meta_ref.bkv_idx[p_id]

        assert self.sem_recvs is not None
        assert self.window_ref is not None
        assert isinstance(cache_kv_hbm, jax.Ref)
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst: Any = self.window_ref.at[slot]  # pytype: disable=attribute-error

        max_hbm_pages = cache_kv_hbm.shape[0]
        num_page_indices = page_indices_ref.shape[0]
        effective_seq_idx = batch_tile_idx

        for batch_idx in range(self.seq_batch_size):
            kv_p_start = bkv_idx * self.bkv_p
            page_indices_offset = (
                effective_seq_idx + batch_idx
            ) * self.pages_per_seq + kv_p_start
            for i in range(self.bkv_p):
                page_idx = jnp.minimum(page_indices_offset + i, num_page_indices - 1)
                safe_page = jnp.minimum(page_indices_ref[page_idx], max_hbm_pages - 1)
                # One page is contiguous in HBM; it lands in lanes
                # [i * page_size, (i + 1) * page_size) of the block. The offset is a
                # compile-time constant and page_size is a multiple of the lane count,
                # so the destination slice is always tile aligned.
                dst_lane_start = i * self.page_size
                pltpu.make_async_copy(
                    cache_kv_hbm.at[safe_page],
                    vmem_dst.at[  # pytype: disable=attribute-error
                        batch_idx,
                        :,
                        :,
                        pl.ds(dst_lane_start, self.page_size),
                    ],
                    sem,
                ).start()

    def wait_in(
        self,
        src_ref: tuple[jax.Ref, jax.Ref, Any],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        assert self.is_input
        if not self.is_buffered:
            return
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst: Any = self.window_ref.at[slot]  # pytype: disable=attribute-error
        pltpu.make_async_copy(
            vmem_dst,
            vmem_dst,
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StreamIndexQBufferedRef(pltpu.BufferedRef):
    """Handles fetching query slices into VMEM buffers."""

    bq_sz: int = dataclasses.field(default=16, metadata=dict(static=True))
    seq_batch_size: int = dataclasses.field(default=1, metadata=dict(static=True))
    chunk_tokens: int = dataclasses.field(default=0, metadata=dict(static=True))
    chunk_idx: int = dataclasses.field(default=0, metadata=dict(static=True))

    @classmethod
    def create(  # pytype: disable=signature-mismatch
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        bq_sz: int,
        chunk_tokens: int,
        seq_batch_size: int = 1,
        grid_rank: int = 1,
        chunk_idx: int = 0,
    ):
        assert buffer_type == pltpu.BufferType.INPUT
        standard_ref = pltpu.BufferedRef.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            grid_rank=grid_rank,
            use_lookahead=use_lookahead,
        )
        return cls(
            bq_sz=bq_sz,
            seq_batch_size=seq_batch_size,
            chunk_tokens=chunk_tokens,
            chunk_idx=chunk_idx,
            **{
                f.name: getattr(standard_ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    @classmethod
    def input(  # pytype: disable=signature-mismatch
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_count: int,
        use_lookahead: bool,
        bq_sz: int,
        chunk_tokens: int,
        seq_batch_size: int = 1,
        grid_rank: int = 1,
        chunk_idx: int = 0,
        **kwargs: Any,
    ):
        return cls.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=pltpu.BufferType.INPUT,
            buffer_count=buffer_count,
            use_lookahead=use_lookahead,
            bq_sz=bq_sz,
            seq_batch_size=seq_batch_size,
            chunk_tokens=chunk_tokens,
            grid_rank=grid_rank,
            chunk_idx=chunk_idx,
        )

    def copy_in(  # pytype: disable=attribute-error
        self,
        src_ref: tuple[Any, ...],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        q_hbm = src_ref[0]
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst: Any = self.window_ref.at[slot]  # pytype: disable=attribute-error

        q_len_start, total_sz = _token_window(
            src_ref,
            grid_indices[0],
            self.chunk_tokens,
            bq_sz=self.bq_sz,
            seq_batch_size=self.seq_batch_size,
            chunk_idx=self.chunk_idx,
        )

        pltpu.make_async_copy(
            q_hbm.at[pl.ds(q_len_start, total_sz)],
            vmem_dst.at[pl.ds(0, total_sz)],  # pytype: disable=attribute-error
            sem,
        ).start()

    def wait_in(
        self,
        src_ref: tuple[Any, ...],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        assert self.is_input
        if not self.is_buffered:
            return
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst: Any = self.window_ref.at[slot]  # pytype: disable=attribute-error
        # Must match copy_in's size exactly or the semaphore never clears.
        _, total_sz = _token_window(
            src_ref,
            grid_indices[0],
            self.chunk_tokens,
            bq_sz=self.bq_sz,
            seq_batch_size=self.seq_batch_size,
            chunk_idx=self.chunk_idx,
        )

        pltpu.make_async_copy(
            vmem_dst.at[pl.ds(0, total_sz)],
            vmem_dst.at[pl.ds(0, total_sz)],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StreamIndexOBufferedRef(pltpu.BufferedRef):
    """Handles scattering computed Top-K scores back to HBM."""

    bq_sz: int = dataclasses.field(default=16, metadata=dict(static=True))
    num_sublanes: int = dataclasses.field(default=1, metadata=dict(static=True))
    seq_batch_size: int = dataclasses.field(default=1, metadata=dict(static=True))
    chunk_idx: int = dataclasses.field(default=0, metadata=dict(static=True))

    @classmethod
    def create(  # pytype: disable=signature-mismatch
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        bq_sz: int,
        num_sublanes: int,
        seq_batch_size: int = 1,
        grid_rank: int = 1,
        chunk_idx: int = 0,
    ):
        assert buffer_type == pltpu.BufferType.OUTPUT
        standard_ref = pltpu.BufferedRef.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            grid_rank=grid_rank,
            use_lookahead=use_lookahead,
        )
        return cls(
            bq_sz=bq_sz,
            num_sublanes=num_sublanes,
            seq_batch_size=seq_batch_size,
            chunk_idx=chunk_idx,
            **{
                f.name: getattr(standard_ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    @classmethod
    def output(  # pytype: disable=signature-mismatch
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_count: int,
        use_lookahead: bool,
        bq_sz: int,
        num_sublanes: int,
        seq_batch_size: int = 1,
        grid_rank: int = 1,
        chunk_idx: int = 0,
        **kwargs: Any,
    ):
        return cls.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=pltpu.BufferType.OUTPUT,
            buffer_count=buffer_count,
            use_lookahead=use_lookahead,
            bq_sz=bq_sz,
            num_sublanes=num_sublanes,
            seq_batch_size=seq_batch_size,
            grid_rank=grid_rank,
            chunk_idx=chunk_idx,
        )

    def copy_out(  # pytype: disable=attribute-error
        self,
        dst_ref: tuple[Any, ...],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        scores_hbm, _, meta_ref, start_end_seq_idx_ref = dst_ref
        bkv_idx = meta_ref.bkv_idx[
            grid_indices[0] + _step_offset(meta_ref, self.chunk_idx)
        ]

        assert self.sem_sends is not None
        assert self.window_ref is not None
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]
        vmem_src: Any = self.window_ref.at[slot]  # pytype: disable=attribute-error

        token_start, total_sz = _token_window(
            dst_ref,
            grid_indices[0],
            scores_hbm.shape[0],
            bq_sz=self.bq_sz,
            seq_batch_size=self.seq_batch_size,
            chunk_idx=self.chunk_idx,
        )
        # HBM scores hold one chunk, so rebase the write onto the chunk's start.
        dst_start = token_start - start_end_seq_idx_ref[2]
        sublane_start = bkv_idx * self.num_sublanes

        pltpu.make_async_copy(
            self._unpadded_slice(vmem_src, total_sz),
            scores_hbm.at[
                pl.ds(dst_start, total_sz),
                pl.ds(sublane_start, self.num_sublanes),
            ],
            sem,
        ).start()

    def wait_out(
        self,
        dst_ref: tuple[Any, ...],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        assert self.is_output
        if not self.is_buffered:
            return
        assert self.sem_sends is not None
        assert self.window_ref is not None
        slot = self.current_wait_out_slot
        sem = self.sem_sends.at[slot]
        vmem_src: Any = self.window_ref.at[slot]  # pytype: disable=attribute-error

        # Must match copy_out's size exactly or the semaphore never clears.
        _, total_sz = _token_window(
            dst_ref,
            grid_indices[0],
            dst_ref[0].shape[0],
            bq_sz=self.bq_sz,
            seq_batch_size=self.seq_batch_size,
            chunk_idx=self.chunk_idx,
        )

        unpadded_src = self._unpadded_slice(vmem_src, total_sz)
        pltpu.make_async_copy(unpadded_src, unpadded_src, sem).wait()

    def _unpadded_slice(self, vmem_src: Any, total_sz: int | jax.Array):
        """Slices out valid tokens and unpadded sublanes (`num_sublanes`).

        The buffer may give each token more rows (`score_pitch`) than it copies out,
        so that the kernel's sublane-strided stores into it stay free of bank
        conflicts.

        Args:
          vmem_src: one [tokens, rows per token, 128] buffer slot.
          total_sz: number of tokens copied out.
        """
        return vmem_src.at[pl.ds(0, total_sz), pl.ds(0, self.num_sublanes)]

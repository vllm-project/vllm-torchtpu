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
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.metadata import (
    cp_local_length,
)


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
        )

    def copy_in(  # pytype: disable=attribute-error
        self,
        src_ref: tuple[Any, ...],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        cache_kv_hbm, page_indices_ref, meta_ref, *_ = src_ref
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

        for batch_idx in range(self.seq_batch_size):
            kv_p_start = bkv_idx * self.bkv_p
            page_indices_offset = (
                effective_seq_idx + batch_idx
            ) * self.pages_per_seq + kv_p_start
            for i in range(self.bkv_p):
                page_idx = jnp.minimum(page_indices_offset + i, num_page_indices - 1)
                safe_page_offset = jnp.minimum(
                    page_indices_ref[page_idx] * self.page_size_per_kv_packing,
                    jnp.maximum(0, max_hbm_pages - self.page_size_per_kv_packing),
                )
                pltpu.make_async_copy(
                    reshaped_cache_hbm_ref.at[
                        pl.ds(safe_page_offset, self.page_size_per_kv_packing)
                    ],
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
        pltpu.make_async_copy(
            vmem_dst,
            vmem_dst,
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StreamIndexKVSeqAlongLaneBufferedRef(pltpu.BufferedRef):
    """Fetches paged KV cache into VMEM for the SEQ_ALONG_LANE layout.

    The HBM cache is `[total_pages, kv_sublane_groups, kv_packing, page_size]`,
    i.e. one page is a single contiguous block whose minor-most dimension is the
    token (lane) dimension. The VMEM block keeps one such slab per page,
    `[seq_batch, bkv_p, kv_sublane_groups, kv_packing, page_size]`, so each page
    moves as one contiguous copy. Placing pages side by side along lanes instead
    would split every page into `kv_sublane_groups * kv_packing` short rows.

    Only the pages that hold the sequence's keys on this rank are fetched. A KV
    block can be far longer than a short sequence or one rank's share of a long
    one, and the top-k never reads the scores of the slabs left stale.
    """

    bkv_p: int = dataclasses.field(default=1, metadata=dict(static=True))
    page_size: int = dataclasses.field(default=128, metadata=dict(static=True))
    pages_per_seq: int = dataclasses.field(default=1, metadata=dict(static=True))
    seq_batch_size: int = dataclasses.field(default=1, metadata=dict(static=True))
    chunk_idx: int = dataclasses.field(default=0, metadata=dict(static=True))
    compression_ratio: int = dataclasses.field(default=1, metadata=dict(static=True))
    cp_size: int = dataclasses.field(default=1, metadata=dict(static=True))
    interleave_c: int = dataclasses.field(default=1, metadata=dict(static=True))

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
        compression_ratio: int = 1,
        cp_size: int = 1,
        interleave_c: int = 1,
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
            compression_ratio=compression_ratio,
            cp_size=cp_size,
            interleave_c=interleave_c,
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
        compression_ratio: int = 1,
        cp_size: int = 1,
        interleave_c: int = 1,
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
            compression_ratio=compression_ratio,
            cp_size=cp_size,
            interleave_c=interleave_c,
        )

    def num_pages(self, src_ref: tuple[Any, ...], p_id, batch_idx: int):
        """Pages of step `p_id`'s KV block holding sequence `batch_idx`'s keys."""
        _, _, meta_ref, seq_lens_ref, cp_rank_ref = src_ref
        seq_idx = jnp.minimum(
            meta_ref.batch_tile_idx[p_id] + batch_idx, seq_lens_ref.shape[0] - 1
        )
        # This rank's share of the sequence's compressed keys.
        kv_len = cp_local_length(
            seq_lens_ref[seq_idx] // self.compression_ratio,
            cp_rank_ref[0],
            self.cp_size,
            self.interleave_c,
        )
        first_page = meta_ref.bkv_idx[p_id] * self.bkv_p
        return jnp.clip(pl.cdiv(kv_len, self.page_size) - first_page, 0, self.bkv_p)

    def copy_in(  # pytype: disable=attribute-error
        self,
        src_ref: tuple[Any, ...],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        cache_kv_hbm, page_indices_ref, meta_ref, *_ = src_ref
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
            num_pages = self.num_pages(src_ref, p_id, batch_idx)
            for i in range(self.bkv_p):

                @pl.when(i < num_pages)
                def _(*, i=i, batch_idx=batch_idx, offset=page_indices_offset):
                    page_idx = jnp.minimum(offset + i, num_page_indices - 1)
                    safe_page = jnp.minimum(
                        page_indices_ref[page_idx], max_hbm_pages - 1
                    )
                    # One page is contiguous in HBM and keeps that layout in its
                    # own slab of the block, so it moves as a single contiguous
                    # copy.
                    pltpu.make_async_copy(
                        cache_kv_hbm.at[safe_page],
                        vmem_dst.at[batch_idx, i],  # pytype: disable=attribute-error
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
        meta_ref = src_ref[2]
        p_id = grid_indices[0] + _step_offset(meta_ref, self.chunk_idx)
        # Must cover exactly the pages copy_in started or the semaphore never
        # clears.
        for batch_idx in range(self.seq_batch_size):
            num_pages = self.num_pages(src_ref, p_id, batch_idx)

            @pl.when(num_pages > 0)
            def _(*, batch_idx=batch_idx, num_pages=num_pages):
                pages = vmem_dst.at[batch_idx, pl.ds(0, num_pages)]
                pltpu.make_async_copy(pages, pages, sem).wait()


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

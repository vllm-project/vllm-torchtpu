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

import abc
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs


class _BaseBufferedRef(pltpu.BufferedRef):
    def __post_init__(self):
        # pallas doesn't allow you to set buffer_count > 2 for output refs, so
        # we override to bypass this check.
        pass


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class _BypassRef(_BaseBufferedRef):
    """`pltpu.BufferedRef` bound to explicit `(step, slot, total_steps)` DMAs."""

    src_ref: Any = dataclasses.field(metadata=dict(static=True))
    dst_ref: Any = dataclasses.field(metadata=dict(static=True))
    # HBM zeros shaped like one window slot; source for `zero_fill_in`.
    zero_src_ref: Any = dataclasses.field(default=None, metadata=dict(static=True))
    window_ref: Any = dataclasses.field(default=None, metadata=dict(static=True))
    sem_recvs: Any = dataclasses.field(default=None, metadata=dict(static=True))
    sem_sends: Any = dataclasses.field(default=None, metadata=dict(static=True))

    def bind(self, src_ref=None, dst_ref=None, zero_src_ref=None):
        return dataclasses.replace(
            self, src_ref=src_ref, dst_ref=dst_ref, zero_src_ref=zero_src_ref
        )

    def zero_fill_in(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
    ) -> None:
        """DMAs zeros into the part of `slot` that `copy_in(step)` doesn't write."""
        raise NotImplementedError(type(self).__name__)

    def init_copy_in(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
        zero_fill: bool | jax.Array,
    ) -> None:
        """`copy_in`, plus zero DMAs over the rest of the slot when `zero_fill`.

        Used for a slot's first use, when the rest of the slot holds uninitialized
        VMEM. The zero and data regions are disjoint (DMAs are unordered) and
        together cover the whole slot, so `wait_in(..., init=True)` waits for the
        full slot.

        Args:
          step: The pipeline step to fetch.
          slot: The input slot to fetch into.
          total_steps: The number of in-range steps.
          zero_fill: Whether to also zero-fill the rest of the slot.
        """
        self.copy_in(step, slot, total_steps)

        @pl.when(zero_fill)
        def _():
            self.zero_fill_in(step, slot, total_steps)

    def _wait_recv(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
        schedule_rows: jax.Array,
        init: bool | jax.Array,
    ) -> None:
        """Waits on `sem_recvs[slot]` for `schedule_rows` 512-byte rows.

        When `init`, waits for the full slot instead (data + zero fill).

        Args:
          step: The pipeline step being waited on.
          slot: The input slot being waited on.
          total_steps: The number of in-range steps.
          schedule_rows: Rows to wait for, from the schedule.
          init: Whether the slot was fetched with `init_copy_in(zero_fill=True)`.
        """
        is_no_op, _ = _step_guard(step, total_steps)
        flat = self.window_ref[slot].bitcast(jnp.uint32).reshape((-1, 128))
        rows = jnp.where(init, flat.shape[0], schedule_rows)
        rows = jnp.where(is_no_op, 0, rows)
        pltpu.make_async_copy(
            flat[pl.ds(0, rows), :],
            flat[pl.ds(0, rows), :],
            self.sem_recvs[slot],
        ).wait()

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool = False,
        **kwargs,
    ):
        standard_ref = _BaseBufferedRef.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            grid_rank=1,
            use_lookahead=use_lookahead,
        )
        return cls(
            src_ref=None,
            dst_ref=None,
            **kwargs,
            **{
                f.name: getattr(standard_ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    @abc.abstractmethod
    def copy_in(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
    ) -> None:
        pass

    @abc.abstractmethod
    def wait_in(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
        init: bool | jax.Array = False,
    ) -> None:
        pass

    @abc.abstractmethod
    def copy_out(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
    ) -> None:
        pass

    @abc.abstractmethod
    def wait_out(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
    ) -> None:
        pass


def _step_guard(
    step: int | jax.Array, total_steps: int | jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Returns `(is_no_op, block_idx)` for a possibly out-of-range pipeline step.

    Out-of-range steps occur in the pipeline prologue/epilogue. They are clamped
    to block 0 (so indexing stays in bounds) and flagged so that all DMA sizes
    can be forced to zero.

    Args:
      step: The pipeline step, which may fall outside `[0, total_steps)`.
      total_steps: The number of in-range steps.

    Returns:
      A tuple of `is_no_op`, true when `step` is out of range, and `block_idx`,
      the step clamped into `[0, total_steps)`.
    """
    is_no_op = jnp.logical_or(step < 0, step >= total_steps)
    block_idx = jnp.where(is_no_op, 0, jnp.maximum(step, 0))
    return is_no_op, block_idx


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class KVBufferedRefSeqAlongLane(_BypassRef):
    """Handles fetching/updating KV cache using SEQ_ALONG_LANE memory layout."""

    cfgs: configs.RpaConfigs = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool = False,
        cfgs: configs.RpaConfigs | None = None,
        **kwargs,
    ):
        assert cfgs is not None
        assert buffer_type == pltpu.BufferType.INPUT_OUTPUT
        return super().create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            use_lookahead=use_lookahead,
            cfgs=cfgs,
            **kwargs,
        )

    def copy_in(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
    ):
        # src_ref: (kv_cache_hbm, new_kv_hbm, schedule_ref, page_indices_ref)
        is_no_op, block_idx = _step_guard(step, total_steps)
        kv_cache_hbm, new_kv_hbm, schedule_ref, page_indices_ref = self.src_ref
        sem = self.sem_recvs[slot]
        vmem_dst_lane = self.window_ref[slot]
        num_lanes = pltpu.get_tpu_info().num_lanes
        for b in range(self.cfgs.batch_size):
            for i in range(self.cfgs.bkv_p_cache):
                p_idx, dst_off, sz = schedule_ref.get_dma_kv_cache(block_idx, b, i)
                hbm_p_idx = page_indices_ref[p_idx]
                sz = jnp.where(is_no_op, 0, sz)
                dst_off = pl.multiple_of(dst_off, num_lanes)
                sz = pl.multiple_of(sz, num_lanes)
                # kv_cache_hbm: (num_pages, num_kv_heads * 2,
                # kv_head_dim // packing, packing, page_size)
                # vmem_dst_lane: (batch_size, num_kv_heads * 2,
                # kv_head_dim // packing, packing, page_size)
                pltpu.make_async_copy(
                    kv_cache_hbm[hbm_p_idx, :, :, :, pl.ds(0, sz)],
                    vmem_dst_lane[b, :, :, :, pl.ds(dst_off, sz)],
                    sem,
                ).start()

            # The new tokens are contiguous in both new_kv_hbm and VMEM, so entry 0
            # carries the single coalesced fetch for all of them.
            dma_entry = schedule_ref.dma_kv_new[block_idx, b, 0]
            src_new_off = pl.multiple_of(dma_entry.fetch_hbm[...], num_lanes)
            dst_vmem_off = pl.multiple_of(dma_entry.fetch_vmem[...], num_lanes)
            sz = pl.multiple_of(jnp.where(is_no_op, 0, dma_entry.fetch_val), num_lanes)
            # new_kv_hbm:
            # (num_kv_heads * 2, kv_head_dim // packing, packing, total_new_tokens)
            pltpu.make_async_copy(
                new_kv_hbm[:, :, :, pl.ds(src_new_off, sz)],
                vmem_dst_lane[b, :, :, :, pl.ds(dst_vmem_off, sz)],
                sem,
            ).start()

    def copy_out(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
    ):
        is_no_op, block_idx = _step_guard(step, total_steps)
        kv_out_ref, _, schedule_ref, page_indices_ref = self.dst_ref
        sem = self.sem_sends[slot]
        vmem_src_lane = self.window_ref[slot]
        num_lanes = pltpu.get_tpu_info().num_lanes
        for b in range(self.cfgs.batch_size):
            do_writeback = schedule_ref.do_writeback[block_idx, b] == 1
            for i in range(self.cfgs.bkv_p_new):
                dma_entry = schedule_ref.dma_kv_new[block_idx, b, i]
                encoded_dst_hbm_off = dma_entry.wb_hbm[...]
                hbm_p_idx = page_indices_ref[
                    encoded_dst_hbm_off >> self.cfgs.serve.page_size_log2
                ]
                dst_off = pl.multiple_of(
                    encoded_dst_hbm_off & self.cfgs.serve.page_size_mask, num_lanes
                )
                src_vmem_off = pl.multiple_of(dma_entry.wb_vmem[...], num_lanes)
                skip = jnp.logical_or(is_no_op, jnp.logical_not(do_writeback))
                sz = pl.multiple_of(jnp.where(skip, 0, dma_entry.wb_val), num_lanes)
                pltpu.make_async_copy(
                    vmem_src_lane[b, :, :, :, pl.ds(src_vmem_off, sz)],
                    kv_out_ref[hbm_p_idx, :, :, :, pl.ds(dst_off, sz)],
                    sem,
                ).start()

    def zero_fill_in(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
    ):
        # Data DMAs fill the lane prefix [0, end); zero [end, kv_vmem_lanes).
        is_no_op, block_idx = _step_guard(step, total_steps)
        _, _, schedule_ref, _ = self.src_ref
        sem = self.sem_recvs[slot]
        vmem_dst_lane = self.window_ref[slot]
        num_lanes = pltpu.get_tpu_info().num_lanes
        slot_lanes = vmem_dst_lane.shape[-1]
        for b in range(self.cfgs.batch_size):
            end = jnp.int32(0)
            for i in range(self.cfgs.bkv_p_cache):
                _, dst_off, sz = schedule_ref.get_dma_kv_cache(block_idx, b, i)
                end = jnp.maximum(end, jnp.where(sz > 0, dst_off + sz, 0))
            dma_entry = schedule_ref.dma_kv_new[block_idx, b, 0]
            new_sz = dma_entry.fetch_val
            end = jnp.maximum(
                end, jnp.where(new_sz > 0, dma_entry.fetch_vmem[...] + new_sz, 0)
            )
            start = pl.multiple_of(end, num_lanes)
            sz = pl.multiple_of(jnp.where(is_no_op, 0, slot_lanes - end), num_lanes)
            pltpu.make_async_copy(
                self.zero_src_ref[b, :, :, :, pl.ds(start, sz)],
                vmem_dst_lane[b, :, :, :, pl.ds(start, sz)],
                sem,
            ).start()

    def wait_in(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
        init: bool | jax.Array = False,
    ):
        _, block_idx = _step_guard(step, total_steps)
        _, _, schedule_ref, _ = self.src_ref
        self._wait_recv(
            step,
            slot,
            total_steps,
            schedule_ref.total_wait_kv_in[block_idx],
            init,
        )

    def wait_out(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
    ):
        is_no_op, block_idx = _step_guard(step, total_steps)
        _, _, schedule_ref, _ = self.dst_ref
        sem = self.sem_sends[slot]
        wait_lanes = schedule_ref.total_wait_kv_out[block_idx]
        wait_lanes = jnp.where(is_no_op, 0, wait_lanes)

        vmem_src = self.window_ref[slot]
        vmem_u32 = vmem_src.bitcast(jnp.uint32)
        flat_src = vmem_u32.reshape((-1, 128))
        pltpu.make_async_copy(
            flat_src[pl.ds(0, wait_lanes), :],
            flat_src[pl.ds(0, wait_lanes), :],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class KVBufferedRefHeadAlongSublane(_BypassRef):
    """Handles fetching and updating KV cache using HEAD_ALONG_SUBLANE memory layout."""

    cfgs: configs.RpaConfigs = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool = False,
        cfgs: configs.RpaConfigs | None = None,
        **kwargs,
    ):
        assert cfgs is not None
        assert buffer_type == pltpu.BufferType.INPUT_OUTPUT
        return super().create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            use_lookahead=use_lookahead,
            cfgs=cfgs,
            **kwargs,
        )

    def copy_in(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
    ):
        # src_ref: (kv_cache_hbm, new_kv_hbm, schedule_ref, page_indices_ref)
        is_no_op, block_idx = _step_guard(step, total_steps)
        kv_cache_hbm, new_kv_hbm, schedule_ref, page_indices_ref = self.src_ref
        sem = self.sem_recvs[slot]
        vmem_dst = self.window_ref[slot, :, :, : self.cfgs.kv_hbm_stride]
        # kv_cache_hbm: (num_pages, num_kv_heads * 2,
        # kv_head_dim // packing, packing, page_size)
        # kv_cache_hbm_flat: (num_pages * num_kv_heads * 2,
        # kv_head_dim // packing, packing, page_size)
        kv_cache_hbm_flat = kv_cache_hbm.reshape(-1, *kv_cache_hbm.shape[2:])

        for b in range(self.cfgs.batch_size):
            for i in range(self.cfgs.bkv_p_cache):
                p_idx, dst_off, sz = schedule_ref.get_dma_kv_cache(block_idx, b, i)
                sz = jnp.where(is_no_op, 0, sz)
                src_off = page_indices_ref[p_idx] * self.cfgs.serve.page_size
                pltpu.make_async_copy(
                    kv_cache_hbm_flat[pl.ds(src_off, sz)],
                    vmem_dst[b, pl.ds(dst_off, sz)],
                    sem,
                ).start()

            # Contiguous fetch for new KV
            dma_entry_0 = schedule_ref.dma_kv_new[block_idx, b, 0]
            src_new_off = dma_entry_0.fetch_hbm[...]
            dst_vmem_off = dma_entry_0.fetch_vmem[...]
            total_new_sz = 0
            for i in range(self.cfgs.bkv_p_new):
                dma_entry = schedule_ref.dma_kv_new[block_idx, b, i]
                total_new_sz += dma_entry.fetch_val
            total_new_sz = jnp.where(is_no_op, 0, total_new_sz)
            pltpu.make_async_copy(
                new_kv_hbm[pl.ds(src_new_off, total_new_sz)],
                vmem_dst[b, pl.ds(dst_vmem_off, total_new_sz)],
                sem,
            ).start()

    def copy_out(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
    ):
        is_no_op, block_idx = _step_guard(step, total_steps)
        kv_out_ref, _, schedule_ref, page_indices_ref = self.dst_ref
        sem = self.sem_sends[slot]
        kv_out_ref_flat = kv_out_ref.reshape(-1, *kv_out_ref.shape[2:])
        vmem_src = self.window_ref[slot, :, :, : self.cfgs.kv_hbm_stride]

        for b in range(self.cfgs.batch_size):
            do_writeback = schedule_ref.do_writeback[block_idx, b] == 1
            for i in range(self.cfgs.bkv_p_new):
                dma_entry = schedule_ref.dma_kv_new[block_idx, b, i]
                encoded_dst_hbm_off = dma_entry.wb_hbm[...]
                src_vmem_off = dma_entry.wb_vmem[...]
                new_sz = dma_entry.wb_val
                global_p_idx = encoded_dst_hbm_off >> self.cfgs.serve.page_size_log2
                p_off = encoded_dst_hbm_off & self.cfgs.serve.page_size_mask
                dst_hbm_off = (
                    page_indices_ref[global_p_idx] << self.cfgs.serve.page_size_log2
                ) | p_off
                sz = jnp.where(
                    jnp.logical_or(is_no_op, jnp.logical_not(do_writeback)), 0, new_sz
                )
                pltpu.make_async_copy(
                    vmem_src[b, pl.ds(src_vmem_off, sz)],
                    kv_out_ref_flat[pl.ds(dst_hbm_off, sz)],
                    sem,
                ).start()

    def zero_fill_in(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
    ):
        # Data DMAs fill tokens [0, end) x stride [0, kv_hbm_stride). Zero the
        # token tail [end, bkv_sz) over the full stride, and the pad column
        # [kv_hbm_stride, bkv_stride) of tokens [0, end).
        is_no_op, block_idx = _step_guard(step, total_steps)
        _, _, schedule_ref, _ = self.src_ref
        sem = self.sem_recvs[slot]
        vmem_dst = self.window_ref[slot]
        bkv_sz = vmem_dst.shape[1]
        hbm_stride = self.cfgs.kv_hbm_stride
        pad = vmem_dst.shape[2] - hbm_stride
        for b in range(self.cfgs.batch_size):
            end = jnp.int32(0)
            for i in range(self.cfgs.bkv_p_cache):
                _, dst_off, sz = schedule_ref.get_dma_kv_cache(block_idx, b, i)
                end = jnp.maximum(end, jnp.where(sz > 0, dst_off + sz, 0))
            total_new_sz = 0
            for i in range(self.cfgs.bkv_p_new):
                total_new_sz += schedule_ref.dma_kv_new[block_idx, b, i].fetch_val
            new_off = schedule_ref.dma_kv_new[block_idx, b, 0].fetch_vmem[...]
            end = jnp.maximum(
                end, jnp.where(total_new_sz > 0, new_off + total_new_sz, 0)
            )
            tail_sz = jnp.where(is_no_op, 0, bkv_sz - end)
            pltpu.make_async_copy(
                self.zero_src_ref[b, pl.ds(end, tail_sz)],
                vmem_dst[b, pl.ds(end, tail_sz)],
                sem,
            ).start()
            if pad > 0:
                pad_tokens = jnp.where(is_no_op, 0, end)
                pltpu.make_async_copy(
                    self.zero_src_ref[b, pl.ds(0, pad_tokens), hbm_stride:],
                    vmem_dst[b, pl.ds(0, pad_tokens), hbm_stride:],
                    sem,
                ).start()

    def wait_in(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
        init: bool | jax.Array = False,
    ):
        _, block_idx = _step_guard(step, total_steps)
        _, _, schedule_ref, _ = self.src_ref
        self._wait_recv(
            step,
            slot,
            total_steps,
            schedule_ref.total_wait_kv_in[block_idx],
            init,
        )

    def wait_out(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
    ):
        is_no_op, block_idx = _step_guard(step, total_steps)
        _, _, schedule_ref, _ = self.dst_ref
        sem = self.sem_sends[slot]
        wait_lanes = schedule_ref.total_wait_kv_out[block_idx]
        wait_lanes = jnp.where(is_no_op, 0, wait_lanes)

        vmem_src = self.window_ref[slot]
        vmem_u32 = vmem_src.bitcast(jnp.uint32)
        flat_src = vmem_u32.reshape((-1, 128))
        pltpu.make_async_copy(
            flat_src[pl.ds(0, wait_lanes), :],
            flat_src[pl.ds(0, wait_lanes), :],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class BatchingORef(_BypassRef):
    """Handles normalizing and storing the final attention output."""

    cfgs: configs.RpaConfigs = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool = False,
        cfgs: configs.RpaConfigs | None = None,
        **kwargs,
    ):
        assert cfgs is not None
        assert buffer_type == pltpu.BufferType.OUTPUT
        return super().create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            use_lookahead=use_lookahead,
            cfgs=cfgs,
            **kwargs,
        )

    def copy_out(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
    ):
        # dst_ref: (o_hbm, schedule_ref)
        is_no_op, block_idx = _step_guard(step, total_steps)
        o_hbm, schedule_ref = self.dst_ref
        sem = self.sem_sends[slot]
        vmem_src = self.window_ref[slot]

        # is_last_k stride: batch size
        for b in range(self.cfgs.batch_size):
            is_last_k = schedule_ref.is_last_k[block_idx, b] == 1
            q_src, q_sz = schedule_ref.get_dma_q(block_idx, b)
            sz = jnp.where(
                jnp.logical_or(is_no_op, jnp.logical_not(is_last_k)), 0, q_sz
            )
            pltpu.make_async_copy(
                vmem_src[b, :, pl.ds(0, sz)],
                o_hbm[:, pl.ds(q_src, sz)],
                sem,
            ).start()

    def wait_out(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
    ):
        # dst_ref: (o_hbm, schedule_ref)
        is_no_op, block_idx = _step_guard(step, total_steps)
        o_hbm, schedule_ref = self.dst_ref
        sem = self.sem_sends[slot]
        wait_lanes = schedule_ref.total_wait_o_out[block_idx]
        wait_lanes = jnp.where(is_no_op, 0, wait_lanes)

        ref_u32 = o_hbm.bitcast(jnp.uint32)
        flat_ref = ref_u32.reshape((-1, 128))
        pltpu.make_async_copy(
            flat_ref[pl.ds(0, wait_lanes), :],
            flat_ref[pl.ds(0, wait_lanes), :],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class BatchingLSERef(_BypassRef):
    """Handles writing LSE values to HBM, overlapped with compute via
    multiple buffering."""

    cfgs: configs.RpaConfigs = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool = False,
        cfgs: configs.RpaConfigs | None = None,
        **kwargs,
    ):
        assert cfgs is not None
        assert buffer_type == pltpu.BufferType.OUTPUT
        return super().create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            use_lookahead=use_lookahead,
            cfgs=cfgs,
            **kwargs,
        )

    def copy_out(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
    ):
        # dst_ref: (lse_hbm, schedule_ref)
        is_no_op, block_idx = _step_guard(step, total_steps)
        lse_hbm, schedule_ref = self.dst_ref
        sem = self.sem_sends[slot]
        vmem_src = self.window_ref[slot]

        for b in range(self.cfgs.batch_size):
            is_last_k = schedule_ref.is_last_k[block_idx, b] == 1
            q_src, q_sz = schedule_ref.get_dma_q(block_idx, b)
            q_sz = jnp.where(
                jnp.logical_or(is_no_op, jnp.logical_not(is_last_k)), 0, q_sz
            )
            pltpu.make_async_copy(
                vmem_src[b, :, pl.ds(0, q_sz)],
                lse_hbm[:, pl.ds(q_src, q_sz)],
                sem,
            ).start()

    def wait_out(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
    ):
        is_no_op, block_idx = _step_guard(step, total_steps)
        lse_hbm, schedule_ref = self.dst_ref
        sem = self.sem_sends[slot]
        wait_lanes = schedule_ref.total_wait_lse_out[block_idx]
        wait_lanes = jnp.where(is_no_op, 0, wait_lanes)
        # wait_lanes = pl.multiple_of(wait_lanes, 8)

        ref_u32 = lse_hbm.bitcast(jnp.uint32)
        flat_ref = ref_u32.reshape((-1, 128))
        pltpu.make_async_copy(
            flat_ref[pl.ds(0, wait_lanes), :],
            flat_ref[pl.ds(0, wait_lanes), :],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class BatchingQRef(_BypassRef):
    """Handles fetching Q blocks using precomputed metadata."""

    cfgs: configs.RpaConfigs = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool = False,
        cfgs: configs.RpaConfigs | None = None,
        **kwargs,
    ):
        assert cfgs is not None
        assert buffer_type == pltpu.BufferType.INPUT
        return super().create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            use_lookahead=use_lookahead,
            cfgs=cfgs,
            **kwargs,
        )

    def copy_in(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
    ):
        # src_ref: (q_hbm, schedule_ref)
        is_no_op, block_idx = _step_guard(step, total_steps)
        q_hbm, schedule_ref = self.src_ref
        sem = self.sem_recvs[slot]
        vmem_dst = self.window_ref[slot]

        for b in range(self.cfgs.batch_size):
            q_src, q_sz = schedule_ref.get_dma_q(block_idx, b)
            sz = jnp.where(is_no_op, 0, q_sz)
            pltpu.make_async_copy(
                q_hbm[:, pl.ds(q_src, sz)],
                vmem_dst[b, :, pl.ds(0, sz)],
                sem,
            ).start()

    def zero_fill_in(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
    ):
        # Data DMAs fill query rows [0, q_sz); zero [q_sz, bq_sz).
        is_no_op, block_idx = _step_guard(step, total_steps)
        _, schedule_ref = self.src_ref
        sem = self.sem_recvs[slot]
        vmem_dst = self.window_ref[slot]
        bq_sz = vmem_dst.shape[2]
        for b in range(self.cfgs.batch_size):
            _, q_sz = schedule_ref.get_dma_q(block_idx, b)
            tail_sz = jnp.where(is_no_op, 0, bq_sz - q_sz)
            pltpu.make_async_copy(
                self.zero_src_ref[b, :, pl.ds(q_sz, tail_sz)],
                vmem_dst[b, :, pl.ds(q_sz, tail_sz)],
                sem,
            ).start()

    def wait_in(
        self,
        step: int | jax.Array,
        slot: int | jax.Array,
        total_steps: int | jax.Array,
        init: bool | jax.Array = False,
    ):
        _, block_idx = _step_guard(step, total_steps)
        _, schedule_ref = self.src_ref
        self._wait_recv(
            step,
            slot,
            total_steps,
            schedule_ref.total_wait_q_in[block_idx],
            init,
        )

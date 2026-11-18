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
"""Context Parallel (CP) schedulers for RPA."""

import dataclasses
import functools
from collections.abc import Sequence
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs,
    schedule,
    scheduler,
    utils,
)


@dataclasses.dataclass(frozen=True)
class CPScheduler(scheduler.BaseScheduler):
    """Base class for page-striped Context Parallel (CP) schedulers.

    Expects `extra_scalars=(cp_rank,)` where `cp_rank` is an `int32[1]` array
    giving the rank of the current device in the CP group.

    Attributes:
        cp_group_size: Number of devices in the CP group.
    """

    cp_group_size: int

    def get_cp_group_size(self) -> int:
        return self.cp_group_size

    def _get_cp_rank(self, extra: Sequence[jax.Array] | Sequence[jax.Ref]) -> jax.Array:
        if not extra:
            raise ValueError(
                f"{type(self).__name__} requires `extra_scalars=(cp_rank,)`."
            )
        return extra[0][0]


@dataclasses.dataclass(frozen=True)
class StripedFullScheduler(CPScheduler):
    """Single-phase striped CP FULL scheduler attending local cached and new KV."""

    scope: ClassVar[configs.AttentionScope] = configs.AttentionScope.FULL

    def get_kv_lens(
        self,
        kv_lens: jax.Array,
        q_lens: jax.Array,
        page_size: int = 1,
        extra_scalars: Sequence[jax.Array] = (),
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        cp_rank = self._get_cp_rank(extra_scalars)
        global_cache_lens = kv_lens - q_lens
        local_cache_lens = utils.cp_local_cache_len(
            global_cache_lens, self.cp_group_size, cp_rank, page_size
        )
        local_kv_lens = utils.cp_local_cache_len(
            kv_lens, self.cp_group_size, cp_rank, page_size
        )
        return (
            local_cache_lens,
            local_kv_lens - local_cache_lens,
            global_cache_lens,
        )

    def get_kv_start(
        self,
        k_idx: jax.Array,
        cfgs: configs.RpaConfigs,
        extra_refs: Sequence[jax.Ref] = (),
    ) -> jax.Array:
        cp_rank = self._get_cp_rank(extra_refs)
        interleaved_size = self.get_kv_interleaved_size(cfgs)
        return k_idx * cfgs.bkv_sz * interleaved_size + (
            cp_rank << cfgs.serve.page_size_log2
        )

    def get_kv_interleaved_size(
        self, cfgs: configs.RpaConfigs | configs.ServingConfigs | None = None
    ) -> int:
        del cfgs
        return self.cp_group_size

    @jax.named_scope("q_loop_striped_full")
    def q_loop(
        self,
        q_idx,
        carry: schedule.LoopCarry,
        *,
        s_idx,
        q_start,
        q_end,
        q_offset,
        kv_cache_len,
        kv_new_len,
        num_k,
        p_offset,
        page_indices_ref: jax.Ref,
        schedule_ref: schedule.RpaSchedule,
        schedule_hbm_ref: schedule.RpaSchedule,
        dma_sem: jax.Ref,
        cfgs: configs.RpaConfigs,
        extra_refs: Sequence[jax.Ref] = (),
    ) -> schedule.LoopCarry:
        cp_rank = self._get_cp_rank(extra_refs)
        cp_group_size = self.cp_group_size
        num_q = pl.cdiv(q_end - q_start, cfgs.bq_sz)
        q_src = q_start + q_idx * cfgs.bq_sz
        q_sz_task = jnp.clip(q_end - q_src, 0, cfgs.bq_sz)

        q_end_global = q_offset + q_idx * cfgs.bq_sz + q_sz_task
        local_k_causal_len = utils.cp_local_cache_len(
            q_end_global, cp_group_size, cp_rank, cfgs.serve.page_size
        )
        end_k_idx_causal = pl.cdiv(local_k_causal_len, cfgs.bkv_sz)
        end_k_idx = jnp.minimum(num_k, end_k_idx_causal)
        if not cfgs.serve.skip_kv_update:
            end_k_idx = jnp.where(q_idx == num_q - 1, num_k, end_k_idx)

        k_loop_fn = functools.partial(
            self.k_loop,
            s_idx=s_idx,
            q_idx=q_idx,
            q_end=q_end,
            q_src=q_src,
            q_sz_task=q_sz_task,
            q_offset=q_offset,
            kv_cache_len=kv_cache_len,
            kv_new_len=kv_new_len,
            kv_new_start=q_start,
            end_k_idx=end_k_idx,
            num_q=num_q,
            p_offset=p_offset,
            page_indices_ref=page_indices_ref,
            schedule_ref=schedule_ref,
            schedule_hbm_ref=schedule_hbm_ref,
            dma_sem=dma_sem,
            cfgs=cfgs,
            extra_refs=extra_refs,
        )
        return jax.lax.fori_loop(0, end_k_idx, k_loop_fn, carry)

    @jax.named_scope("k_loop_striped_full")
    def k_loop(
        self,
        k_idx,
        carry: schedule.LoopCarry,
        *,
        s_idx,
        q_idx,
        q_end,
        q_src,
        q_sz_task,
        q_offset=0,
        kv_cache_len,
        kv_new_len,
        end_k_idx,
        kv_new_start=0,
        num_q=0,
        p_offset,
        page_indices_ref: jax.Ref,
        schedule_ref: schedule.RpaSchedule,
        schedule_hbm_ref: schedule.RpaSchedule,
        dma_sem: jax.Ref,
        cfgs: configs.RpaConfigs,
        extra_refs: Sequence[jax.Ref] = (),
    ) -> schedule.LoopCarry:
        del q_end
        count = carry.count
        step, target_lane = divmod(count, cfgs.batch_size)

        schedule_ref.s_idx[step, target_lane] = s_idx
        schedule_ref.q_idx[step, target_lane] = q_idx
        schedule_ref.k_idx[step, target_lane] = k_idx

        is_last_k = jnp.where(k_idx == end_k_idx - 1, 1, 0)
        schedule_ref.is_last_k[step, target_lane] = is_last_k

        schedule_ref.dma_q[step, target_lane, 0] = q_src
        schedule_ref.dma_q[step, target_lane, 1] = q_sz_task

        kv_len_start = k_idx * cfgs.bkv_sz
        kv_p_start = k_idx * cfgs.bkv_p
        k_len = kv_cache_len + kv_new_len
        kv_left = k_len - kv_len_start
        if cfgs.serve.skip_kv_update:
            kv_left_frm_cache = kv_left
        else:
            kv_left_frm_cache = jnp.maximum(kv_cache_len - kv_len_start, 0)

        # The schedule carries physical page ids: `page_indices_ref` holds this
        # rank's local page table window, and `p_offset` is the sequence's row
        # in it.
        for i in range(cfgs.bkv_p_cache):
            dst_vmem = i << cfgs.serve.page_size_log2
            dma_sz = kv_left_frm_cache - dst_vmem
            dma_sz = jnp.clip(dma_sz, 0, cfgs.serve.page_size)
            p_idx = jnp.minimum(kv_p_start + i, cfgs.serve.pages_per_seq - 1)
            src_hbm = page_indices_ref[p_offset + p_idx]
            schedule_ref.dma_kv_cache[step, target_lane, i, 0] = src_hbm
            schedule_ref.dma_kv_cache[step, target_lane, i, 1] = dst_vmem
            schedule_ref.dma_kv_cache[step, target_lane, i, 2] = utils.align_to(
                dma_sz, cfgs.kv_token_align
            )

        kv_left_frm_new = kv_left - kv_left_frm_cache
        bkv_sz_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
        new_sz = jnp.minimum(cfgs.bkv_sz - bkv_sz_cache, kv_left_frm_new)

        q_wb = num_q - 1
        do_writeback = jnp.where((new_sz > 0) & (q_idx == q_wb), 1, 0)
        schedule_ref.do_writeback[step, target_lane] = do_writeback

        interleaved_size = self.get_kv_interleaved_size(cfgs)
        k_base = self.get_kv_start(k_idx, cfgs, extra_refs)
        first_global_tok_idx = (
            k_base
            + (
                ((bkv_sz_cache >> cfgs.serve.page_size_log2) * interleaved_size)
                << cfgs.serve.page_size_log2
            )
            + (bkv_sz_cache & cfgs.serve.page_size_mask)
        )
        first_src_hbm = kv_new_start + jnp.maximum(0, first_global_tok_idx - q_offset)

        def fill_dma_kv_new(i, dst_vmem, dma_sz):
            dma_entry = schedule_ref.dma_kv_new[step, target_lane, i]
            local_tok_idx = kv_len_start + dst_vmem
            local_p_idx = jnp.minimum(
                local_tok_idx >> cfgs.serve.page_size_log2,
                cfgs.serve.pages_per_seq - 1,
            )
            p_off = local_tok_idx & cfgs.serve.page_size_mask
            global_tok_idx = (
                k_base
                + (
                    ((dst_vmem >> cfgs.serve.page_size_log2) * interleaved_size)
                    << cfgs.serve.page_size_log2
                )
                + p_off
            )
            rel_off = jnp.maximum(0, global_tok_idx - q_offset)
            src_hbm = kv_new_start + rel_off

            if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
                align = cfgs.kv_token_align
                src, _, fetch_val = utils.align_span(src_hbm, dma_sz, align)
                if cfgs.uses_multi_page_lane_staging:
                    chunks_per_page = cfgs.serve.page_size // align
                    delta = (first_src_hbm - bkv_sz_cache) & (align - 1)
                    page_slot = dst_vmem >> cfgs.serve.page_size_log2
                    slot_base = (page_slot * (chunks_per_page + 1) + 1) * align
                    chunk_in_slot = ((p_off + delta) // align) * align
                    fetch_vmem = slot_base + chunk_in_slot
                else:
                    cache_end = utils.align_to(bkv_sz_cache, align)
                    fetch_vmem = cache_end + utils.align_to(
                        dst_vmem - bkv_sz_cache, align
                    )
                dma_entry.fetch_hbm[...] = jnp.where(dma_sz > 0, src, 0)
                dma_entry.fetch_vmem[...] = jnp.where(dma_sz > 0, fetch_vmem, 0)
                fetch_val = jnp.where(dma_sz > 0, fetch_val, 0)
                dst, lead, wb_val = utils.align_span(p_off, dma_sz, align)
                dma_entry.wb_hbm[...] = (
                    page_indices_ref[p_offset + local_p_idx]
                    << cfgs.serve.page_size_log2
                ) | dst
                dma_entry.wb_vmem[...] = jnp.where(dma_sz > 0, dst_vmem - lead, 0)
                dma_entry.set_flags(fetch_val, jnp.where(dma_sz > 0, wb_val, 0))
            else:
                dst_hbm = (
                    page_indices_ref[p_offset + local_p_idx]
                    << cfgs.serve.page_size_log2
                ) | p_off
                dma_entry.fetch_hbm[...] = src_hbm
                dma_entry.fetch_vmem[...] = dst_vmem
                dma_entry.wb_hbm[...] = dst_hbm
                dma_entry.set_flags(dma_sz, dma_sz)

        if cfgs.bkv_p_new < cfgs.bkv_p:
            curr_vmem = bkv_sz_cache
            curr_rem = new_sz
            for i in range(cfgs.bkv_p_new):
                slot_start = (curr_vmem // cfgs.serve.page_size) * cfgs.serve.page_size
                slot_end = slot_start + cfgs.serve.page_size
                dma_sz = jnp.minimum(curr_rem, slot_end - curr_vmem)
                dma_sz = jnp.where(curr_rem > 0, dma_sz, 0)
                fill_dma_kv_new(i, curr_vmem, dma_sz)
                curr_vmem += dma_sz
                curr_rem = jnp.maximum(0, curr_rem - dma_sz)
        else:
            iters = max(cfgs.bkv_p, cfgs.bkv_p_new)
            for i in range(iters):
                slot_start = i * cfgs.serve.page_size
                slot_end = slot_start + cfgs.serve.page_size
                dst_vmem = jnp.maximum(slot_start, bkv_sz_cache)
                end_in_slot = jnp.minimum(slot_end, bkv_sz_cache + new_sz)
                dma_sz = jnp.maximum(0, end_in_slot - dst_vmem)
                fill_dma_kv_new(i, dst_vmem, dma_sz)

        return schedule.advance_carry(
            carry,
            schedule_ref,
            schedule_hbm_ref,
            dma_sem,
            cfgs=cfgs,
        )

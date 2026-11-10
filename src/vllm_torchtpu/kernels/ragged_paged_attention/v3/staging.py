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
"""Shared Pallas Q/O DMA staging, sequence pipeline, and launch helpers for RPA v3."""

from collections.abc import Callable
from enum import Enum
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.ragged_paged_attention.v3.util import (
    align_to,
    async_copy,
    broadcast_minor,
    cdiv,
    has_bank_conflicts,
    strided_load,
    strided_store,
)

# Row indices into the `(4, 2)` double-buffered DMA semaphore array `sems`.
SEM_BKV_FETCH = 0
SEM_BQ_FETCH = 1
SEM_BO_SEND = 2
SEM_BKV_UPDATE = 3


class RpaCase(Enum):
    """Represents the different execution passes for Ragged Paged Attention.

    - DECODE: Sequences are in decode-only mode (`q_len == 1`).
    - PREFILL: Sequences are in chunked-prefill-only mode (`q_len` static).
    - MIXED: Sequences can be a mix of prefill and decode (`q_len` dynamic).
    """

    DECODE = 0
    PREFILL = 1
    MIXED = 2

    @property
    def symbol(self) -> str:
        return {
            RpaCase.DECODE: "d",
            RpaCase.PREFILL: "p",
            RpaCase.MIXED: "m",
        }[self]

    def get_range(self, distribution, *, has_prefill_pass: bool = True):
        assert distribution.shape == (3,)
        if self == RpaCase.DECODE:
            return 0, distribution[0]
        if self == RpaCase.PREFILL:
            return distribution[0], distribution[1]
        if self == RpaCase.MIXED:
            start = distribution[1] if has_prefill_pass else distribution[0]
            return start, distribution[2]
        raise ValueError(f"Unsupported RPA case: {self}")


class StagingRefs(NamedTuple):
    """Double-buffered Q/O HBM, VMEM, and ring-tracking references."""

    q_hbm_ref: Any
    o_hbm_ref: Any
    bq_x2_ref: Any
    bo_x2_ref: Any
    cu_q_lens_ref: Any
    bo_ids_ref: Any
    sems: Any


class KvUpdateRing(NamedTuple):
    """Double-buffered KV-cache writeback ring tracker."""

    bkv_update_ids_ref: Any
    update_fn: Callable[..., Any]

    def start(self, seq_idx, bkv_sem_idx, offset, update_sz) -> None:
        self.bkv_update_ids_ref[bkv_sem_idx] = seq_idx
        self.bkv_update_ids_ref[bkv_sem_idx + 2] = offset
        self.bkv_update_ids_ref[bkv_sem_idx + 4] = update_sz
        self.update_fn(seq_idx, bkv_sem_idx, offset, update_sz)

    def wait(self, bkv_sem_idx) -> None:
        update_sz = self.bkv_update_ids_ref[bkv_sem_idx + 4]

        @pl.when(update_sz > 0)
        def _():
            seq_idx = self.bkv_update_ids_ref[bkv_sem_idx]
            offset = self.bkv_update_ids_ref[bkv_sem_idx + 2]
            self.bkv_update_ids_ref[bkv_sem_idx + 4] = 0
            self.update_fn(seq_idx, bkv_sem_idx, offset, update_sz, wait=True)


class BkvStepContext(NamedTuple):
    """Per-`(bq, bkv)` step indices and lengths passed to `compute_bkv_fn`."""

    bq_sem_idx: Any
    bkv_sem_idx: Any
    bkv_idx: Any
    offset: Any
    kv_q_gap: Any
    processed_q_len: Any
    processed_kv_len: Any
    effective_kv_len: Any
    effective_bkv_sz: Any
    actual_bq_sz: int
    actual_bq_csz: int


def _transfer_qo(
    refs: StagingRefs,
    seq_idx,
    blk_idx,
    sem_idx,
    *,
    bq_sz: int,
    is_send: bool,
    wait: bool = False,
    debug_mode: bool = False,
) -> None:
    sem = refs.sems.at[SEM_BO_SEND if is_send else SEM_BQ_FETCH, sem_idx]
    vmem_ref = (refs.bo_x2_ref if is_send else refs.bq_x2_ref).at[sem_idx]
    q_len_start = refs.cu_q_lens_ref[seq_idx] + blk_idx * bq_sz
    q_end = refs.cu_q_lens_ref[seq_idx + 1]
    sz = jnp.minimum(bq_sz, q_end - q_len_start)
    if debug_mode:
        op = "send_bo" if is_send else "fetch_bq"
        idx_name = "bo_idx" if is_send else "bq_idx"
        sem_name = "bo_sem_idx" if is_send else "bq_sem_idx"
        pl.debug_print(
            f"[RPA debug] -----------{'wait' if wait else 'start'}_{op}-----------"
        )
        pl.debug_print("[RPA debug] seq_idx={}", seq_idx)
        pl.debug_print(f"[RPA debug] {idx_name}={{}}", blk_idx)
        pl.debug_print(f"[RPA debug] {sem_name}={{}}", sem_idx)
        pl.debug_print("[RPA debug] q_len_start={}", q_len_start)
        pl.debug_print("[RPA debug] q_end={}", q_end)
        pl.debug_print("[RPA debug] sz={}", sz)
    vmem_slice = vmem_ref.at[:, pl.ds(0, sz)]
    hbm_slice = (refs.o_hbm_ref if is_send else refs.q_hbm_ref).at[
        :, pl.ds(q_len_start, sz)
    ]
    src, dst = (vmem_slice, hbm_slice) if is_send else (hbm_slice, vmem_slice)
    async_copy(src, dst, sem, wait, debug_mode=debug_mode)


def fetch_bq(
    refs: StagingRefs,
    seq_idx,
    bq_idx,
    bq_sem_idx,
    *,
    bq_sz: int,
    wait: bool = False,
    debug_mode: bool = False,
) -> None:
    _transfer_qo(
        refs,
        seq_idx,
        bq_idx,
        bq_sem_idx,
        bq_sz=bq_sz,
        is_send=False,
        wait=wait,
        debug_mode=debug_mode,
    )


def send_bo(
    refs: StagingRefs,
    seq_idx,
    bo_idx,
    bo_sem_idx,
    *,
    bq_sz: int,
    wait: bool = False,
    debug_mode: bool = False,
) -> None:
    _transfer_qo(
        refs,
        seq_idx,
        bo_idx,
        bo_sem_idx,
        bq_sz=bq_sz,
        is_send=True,
        wait=wait,
        debug_mode=debug_mode,
    )


def start_send_bo(
    refs: StagingRefs,
    seq_idx,
    bo_idx,
    bo_sem_idx,
    *,
    bq_sz: int,
    debug_mode: bool = False,
) -> None:
    refs.bo_ids_ref[bo_sem_idx] = seq_idx
    refs.bo_ids_ref[bo_sem_idx + 2] = bo_idx
    send_bo(refs, seq_idx, bo_idx, bo_sem_idx, bq_sz=bq_sz, debug_mode=debug_mode)


def wait_send_bo(
    refs: StagingRefs,
    start_seq_idx,
    seq_idx,
    bo_sem_idx,
    *,
    bq_sz: int,
    debug_mode: bool = False,
) -> None:
    old_seq_idx = refs.bo_ids_ref[bo_sem_idx]
    old_bo_idx = refs.bo_ids_ref[bo_sem_idx + 2]

    @pl.when(jnp.logical_and(start_seq_idx <= old_seq_idx, old_seq_idx <= seq_idx))
    def _():
        send_bo(
            refs,
            old_seq_idx,
            old_bo_idx,
            bo_sem_idx,
            bq_sz=bq_sz,
            wait=True,
            debug_mode=debug_mode,
        )


def load_bq(bq_x2_ref, bq_sem_idx, kv_head_idx, start, sz):
    """Load `sz` query tokens for `kv_head_idx` from double-buffered `bq_x2_ref`."""
    _, _, bq_sz, nq_per_kv_packed, _, head_dim = bq_x2_ref.shape
    q_ref = (
        bq_x2_ref.bitcast(jnp.uint32)
        .at[bq_sem_idx, kv_head_idx]
        .reshape(bq_sz * nq_per_kv_packed, head_dim)
    )
    return strided_load(
        q_ref,
        start * nq_per_kv_packed,
        sz * nq_per_kv_packed,
        1,
        dtype=bq_x2_ref.dtype,
    )


def load_bkv(bkv_x2_ref, bkv_sem_idx, kv_head_idx, start, sz):
    """Load `(k, v)` tiles of `sz` tokens for `kv_head_idx` from `bkv_x2_ref`."""
    _, bkv_sz, bkv_stride, kv_packing, head_dim = bkv_x2_ref.shape
    kv_dtype = bkv_x2_ref.dtype
    start *= bkv_stride
    sz *= bkv_stride
    kv_ref = (
        bkv_x2_ref.bitcast(jnp.uint32)
        .at[bkv_sem_idx]
        .reshape(bkv_sz * bkv_stride, head_dim)
    )
    if kv_packing == 1:
        start += kv_head_idx * 2
        return (
            strided_load(kv_ref, start, sz, bkv_stride, dtype=kv_dtype),
            strided_load(kv_ref, start + 1, sz, bkv_stride, dtype=kv_dtype),
        )

    num_kv_per_load = kv_packing // 2
    kv = strided_load(kv_ref, start + kv_head_idx // num_kv_per_load, sz, bkv_stride)
    bitwidth = 32 // kv_packing
    repack_ty = jnp.dtype(f"uint{bitwidth}")
    k = kv >> ((kv_head_idx % num_kv_per_load) * 2 * bitwidth)
    v = k >> bitwidth
    return (
        pltpu.bitcast(k.astype(repack_ty), kv_dtype),
        pltpu.bitcast(v.astype(repack_ty), kv_dtype),
    )


def finalize_and_store_bo(acc_ref, l_ref, bo_x2_ref, bo_sem_idx) -> None:
    """Normalize `acc_ref` by `l_ref` and store into `bo_x2_ref[bo_sem_idx]`."""
    _, actual_num_kv_heads, bq_sz, nq_per_kv_packed, _, head_dim = bo_x2_ref.shape
    store_dtype = bo_x2_ref.dtype
    acc = acc_ref[...]
    denom = broadcast_minor(l_ref[...], acc.shape)
    out = (
        acc * pltpu.reciprocal(denom, approx=True)
        if (denom.dtype == jnp.float32 and store_dtype != jnp.float32)
        else lax.div(acc, denom)
    ).astype(store_dtype)
    out_ref = (
        bo_x2_ref.at[bo_sem_idx]
        .bitcast(jnp.int32)
        .reshape(actual_num_kv_heads * bq_sz * nq_per_kv_packed, head_dim)
    )
    out_i32 = pltpu.bitcast(out, out_ref.dtype).reshape(out_ref.shape)
    strided_store(out_ref, 0, out_ref.shape[0], 1, out_i32)


def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx, *, num_bq):
    next_bq_idx = bq_idx + 1
    is_last_bq = next_bq_idx == num_bq
    return (
        lax.select(is_last_bq, seq_idx + 1, seq_idx),
        lax.select(is_last_bq, 0, next_bq_idx),
        lax.select(bq_sem_idx == 0, 1, 0),
    )


def get_next_bkv_ids(
    seq_idx,
    bq_idx,
    bkv_idx,
    bkv_sem_idx,
    *,
    num_bq,
    num_bkv,
    kv_q_gap,
    actual_bq_sz,
    bkv_sz: int,
    sliding_window: int | None,
    next_seq_start_bkv_idx,
):
    next_bkv_idx = bkv_idx + 1
    is_last_bkv = next_bkv_idx == num_bkv
    next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
    is_last_bq = next_bq_idx == num_bq
    next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
    next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
    next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)
    next_bq_start_bkv_idx = (
        0
        if sliding_window is None
        else jnp.maximum(kv_q_gap + (bq_idx + 1) * actual_bq_sz - sliding_window, 0)
        // bkv_sz
    )
    next_bkv_idx = lax.select(is_last_bkv, next_bq_start_bkv_idx, next_bkv_idx)
    next_bkv_idx = lax.select(is_last_bq, next_seq_start_bkv_idx, next_bkv_idx)
    return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx


def run_rpa_sequence_pipeline(
    seq_idx,
    *,
    start_seq_idx,
    end_seq_idx,
    staging_refs: StagingRefs,
    kv_lens_ref,
    sem_ids_ref,
    kv_update_ring: KvUpdateRing,
    l_ref,
    m_ref,
    acc_ref,
    fetch_bkv_fn: Callable[..., tuple[Any, Any]],
    compute_bkv_fn: Callable[[BkvStepContext], None],
    bq_sz: int,
    bkv_sz: int,
    bq_csz: int,
    use_causal_mask: bool = True,
    sliding_window: int | None = None,
    static_q_len: int | None = None,
    debug_mode: bool = False,
) -> None:
    """Execute one sequence step of the double-buffered RPA v3 DMA/compute pipeline.

    Args:
      seq_idx: Current sequence index in `[start_seq_idx, end_seq_idx)`.
      start_seq_idx: Inclusive start sequence index for this RPA pass.
      end_seq_idx: Exclusive end sequence index for this RPA pass.
      staging_refs: Double-buffered Q/O HBM and VMEM refs; `bq_x2_ref` and
        `bo_x2_ref` have shape `[2, actual_num_kv_heads, bq_sz, nq_packed, q_pack, D]`.
      kv_lens_ref: SMEM ref of shape `[max_num_seqs]` holding KV lengths.
      sem_ids_ref: SMEM ref of shape `[3]` holding
        `(bq_sem_idx, bkv_sem_idx, bo_sem_idx)`.
      kv_update_ring: `KvUpdateRing` tracking in-flight KV-cache writebacks.
      l_ref: VMEM scratch ref of shape `[actual_num_kv_heads, bq_sz * nq, 128]`.
      m_ref: VMEM scratch ref of shape `[actual_num_kv_heads, bq_sz * nq, 128]`.
      acc_ref: VMEM scratch ref of shape `[actual_num_kv_heads, bq_sz * nq, head_dim]`.
      fetch_bkv_fn: Callback `(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False)`
        returning `(offset, update_sz)`.
      compute_bkv_fn: Callback consuming a `BkvStepContext` for one `(bq, bkv)` tile.
      bq_sz: Query DMA fetch block size.
      bkv_sz: KV DMA fetch block size.
      bq_csz: Query compute sub-tile size.
      use_causal_mask: Whether causal masking bounds `effective_kv_len`.
      sliding_window: Optional sliding-window size; also flushes KV updates on early
        blocks falling below `next_bq_start_bkv_idx`.
      static_q_len: Optional compile-time static query length (`1` for decode,
        `chunk_prefill_size` for prefill).
      debug_mode: If true, emits `[RPA debug]` trace prints and skips DMAs/compute.
    """
    assert staging_refs.bq_x2_ref.shape == staging_refs.bo_x2_ref.shape
    assert staging_refs.bq_x2_ref.dtype == staging_refs.bo_x2_ref.dtype
    _, actual_num_kv_heads, ref_bq_sz, nq_per_kv_packed, q_packing, head_dim = (
        staging_refs.bq_x2_ref.shape
    )
    assert ref_bq_sz == bq_sz
    q_rows = bq_sz * nq_per_kv_packed * q_packing
    assert l_ref.shape == (actual_num_kv_heads, q_rows, 128)
    assert m_ref.shape == (actual_num_kv_heads, q_rows, 128)
    assert acc_ref.shape == (actual_num_kv_heads, q_rows, head_dim)

    cu_q_lens_ref = staging_refs.cu_q_lens_ref
    q_start = cu_q_lens_ref[seq_idx]
    q_end = cu_q_lens_ref[seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[seq_idx]
    kv_q_gap = kv_len - q_len
    cur_seq_start_bkv_idx = 0
    next_seq_start_bkv_idx = 0
    if sliding_window is not None:
        cur_seq_start_bkv_idx = jnp.maximum(kv_q_gap - sliding_window, 0) // bkv_sz
        next_seq_idx = jnp.minimum(seq_idx + 1, end_seq_idx - 1)
        next_q_len = cu_q_lens_ref[next_seq_idx + 1] - cu_q_lens_ref[next_seq_idx]
        next_kv_q_gap = kv_lens_ref[next_seq_idx] - next_q_len
        next_seq_start_bkv_idx = (
            jnp.maximum(next_kv_q_gap - sliding_window, 0) // bkv_sz
        )

    if debug_mode:
        pl.debug_print("[RPA debug] ======= In loop seq_idx={}", seq_idx)
        pl.debug_print("[RPA debug] start_seq_idx={}", start_seq_idx)
        pl.debug_print("[RPA debug] end_seq_idx={}", end_seq_idx)
        pl.debug_print("[RPA debug] bkv_sz={}", bkv_sz)
        pl.debug_print("[RPA debug] bq_sz={}", bq_sz)
        pl.debug_print(f"[RPA debug] static_q_len={static_q_len}")
        pl.debug_print("[RPA debug] q_start={}", q_start)
        pl.debug_print("[RPA debug] q_end={}", q_end)
        pl.debug_print("[RPA debug] q_len={}", q_len)
        pl.debug_print("[RPA debug] kv_len={}", kv_len)
        pl.debug_print("[RPA debug] kv_q_gap={}", kv_q_gap)
        pl.debug_print(f"[RPA debug] sliding_window={sliding_window}")
        pl.debug_print("[RPA debug] cur_seq_start_bkv_idx={}", cur_seq_start_bkv_idx)
        pl.debug_print("[RPA debug] next_seq_start_bkv_idx={}", next_seq_start_bkv_idx)

    def process() -> None:
        actual_bq_sz = bq_sz if static_q_len is None else min(bq_sz, static_q_len)
        num_bq = cdiv(q_len if static_q_len is None else static_q_len, actual_bq_sz)
        actual_bq_csz = min(bq_csz, actual_bq_sz)

        @pl.loop(0, num_bq, unroll=False)
        def compute_with_bq(bq_idx):
            l_ref[...] = jnp.full_like(l_ref, 0.0)
            m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
            acc_ref[...] = jnp.full_like(acc_ref, 0.0)

            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                seq_idx, bq_idx, bq_sem_idx, num_bq=num_bq
            )
            processed_q_len = kv_q_gap + bq_idx * actual_bq_sz
            start_bkv_idx = (
                0
                if sliding_window is None
                else jnp.maximum(processed_q_len - sliding_window, 0) // bkv_sz
            )
            next_bq_start_bkv_idx = (
                0
                if sliding_window is None
                else jnp.maximum(processed_q_len + actual_bq_sz - sliding_window, 0)
                // bkv_sz
            )
            effective_kv_len = (
                jnp.minimum(kv_len, processed_q_len + actual_bq_sz)
                if use_causal_mask
                else kv_len
            )
            end_bkv_idx = cdiv(effective_kv_len, bkv_sz)

            @pl.when(next_seq_idx < end_seq_idx)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                fetch_bq(
                    staging_refs,
                    next_seq_idx,
                    next_bq_idx,
                    next_bq_sem_idx,
                    bq_sz=bq_sz,
                    debug_mode=debug_mode,
                )

            @pl.loop(start_bkv_idx, end_bkv_idx, unroll=False)
            def compute_with_bkv(bkv_idx):
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    seq_idx,
                    bq_idx,
                    bkv_idx,
                    bkv_sem_idx,
                    num_bq=num_bq,
                    num_bkv=end_bkv_idx,
                    kv_q_gap=kv_q_gap,
                    actual_bq_sz=actual_bq_sz,
                    bkv_sz=bkv_sz,
                    sliding_window=sliding_window,
                    next_seq_start_bkv_idx=next_seq_start_bkv_idx,
                )
                processed_kv_len = bkv_idx * bkv_sz

                @pl.when(next_seq_idx < end_seq_idx)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    fetch_bkv_fn(next_seq_idx, next_bkv_idx, next_bkv_sem_idx)

                @pl.when(bkv_idx == start_bkv_idx)
                def wait_cur_bq():
                    fetch_bq(
                        staging_refs,
                        seq_idx,
                        bq_idx,
                        bq_sem_idx,
                        bq_sz=bq_sz,
                        wait=True,
                        debug_mode=debug_mode,
                    )

                offset, update_sz = fetch_bkv_fn(
                    seq_idx, bkv_idx, bkv_sem_idx, wait=True
                )
                is_last_bkv_visit = (
                    jnp.logical_or(
                        bq_idx == num_bq - 1, bkv_idx < next_bq_start_bkv_idx
                    )
                    if sliding_window is not None
                    else (bq_idx == num_bq - 1)
                )

                @pl.when(jnp.logical_and(update_sz > 0, is_last_bkv_visit))
                def update_cur_bkv_to_cache():
                    kv_update_ring.start(seq_idx, bkv_sem_idx, offset, update_sz)

                if debug_mode:
                    pl.debug_print("[RPA debug] -----------flash attention-----------")
                    pl.debug_print("[RPA debug] seq_idx={}", seq_idx)
                    pl.debug_print("[RPA debug] bq_idx={}", bq_idx)
                    pl.debug_print("[RPA debug] bkv_idx={}", bkv_idx)
                    return

                effective_bkv_sz = jnp.maximum(
                    jnp.minimum(effective_kv_len - processed_kv_len, bkv_sz), 0
                )
                compute_bkv_fn(
                    BkvStepContext(
                        bq_sem_idx=bq_sem_idx,
                        bkv_sem_idx=bkv_sem_idx,
                        bkv_idx=bkv_idx,
                        offset=offset,
                        kv_q_gap=kv_q_gap,
                        processed_q_len=processed_q_len,
                        processed_kv_len=processed_kv_len,
                        effective_kv_len=effective_kv_len,
                        effective_bkv_sz=effective_bkv_sz,
                        actual_bq_sz=actual_bq_sz,
                        actual_bq_csz=actual_bq_csz,
                    )
                )

            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            wait_send_bo(
                staging_refs,
                start_seq_idx,
                seq_idx,
                bo_sem_idx,
                bq_sz=bq_sz,
                debug_mode=debug_mode,
            )
            finalize_and_store_bo(
                acc_ref,
                l_ref,
                staging_refs.bo_x2_ref,
                bo_sem_idx,
            )
            start_send_bo(
                staging_refs,
                seq_idx,
                bq_idx,
                bo_sem_idx,
                bq_sz=bq_sz,
                debug_mode=debug_mode,
            )

    @pl.when(seq_idx == start_seq_idx)
    def prologue():
        fetch_bq(staging_refs, start_seq_idx, 0, 0, bq_sz=bq_sz, debug_mode=debug_mode)
        fetch_bkv_fn(start_seq_idx, cur_seq_start_bkv_idx, 0)

    @pl.when(jnp.logical_and(start_seq_idx <= seq_idx, seq_idx < end_seq_idx))
    def pipeline():
        process()

    @pl.when(seq_idx == end_seq_idx - 1)
    def epilogue():
        for i in range(2):
            wait_send_bo(
                staging_refs,
                start_seq_idx,
                seq_idx,
                i,
                bq_sz=bq_sz,
                debug_mode=debug_mode,
            )
            kv_update_ring.wait(i)


def get_smem_estimate_bytes(max_num_seqs: int, pages_per_seq: int) -> int:
    """Estimate SMEM bytes consumed by the 7 `make_scalar_prefetches` buffers."""
    total_bits = (
        # [0] kv_lens_ref: i32[max_num_seqs]
        align_to(max_num_seqs, 128) * 32
        +
        # [1] page_indices_ref: i32[max_num_seqs * pages_per_seq]
        align_to(max_num_seqs * pages_per_seq, 128) * 32
        +
        # [2] cu_q_lens_ref: i32[max_num_seqs + 1]
        align_to(max_num_seqs + 1, 128) * 32
        +
        # [3] distribution_ref: i32[3]
        128 * 32
        +
        # [4] sem_ids_ref: i32[3]
        128 * 32
        +
        # [5] bo_ids_ref: i32[4]
        128 * 32
        +
        # [6] bkv_update_ids_ref: i32[6]
        128 * 32
    )
    return cdiv(total_bits, 8)


def make_scalar_prefetches(
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
):
    """Build the 7-tuple scalar-prefetch metadata shared by RPA v3 kernels.

    Slot layout:
      [0] `kv_lens`: `i32[max_num_seqs]`
      [1] `page_indices`: `i32[max_num_seqs * pages_per_seq]`
      [2] `cu_q_lens`: `i32[max_num_seqs + 1]`
      [3] `distribution`: `i32[3]` `(decode_end, prefill_end, mixed_end)`
      [4] `sem_ids`: `i32[3]` `(bq_sem_idx, bkv_sem_idx, bo_sem_idx)`
      [5] `bo_ids`: `i32[4]` `(bo_sem_0_seq_idx, bo_sem_1_seq_idx,
          bo_sem_0_bo_idx, bo_sem_1_bo_idx)`
      [6] `bkv_update_ids`: `i32[6]` `(bkv_sem_0_seq_idx, bkv_sem_1_seq_idx,
          bkv_sem_0_offset, bkv_sem_1_offset, bkv_sem_0_sz, bkv_sem_1_sz)`
    """
    return (
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        jnp.zeros((3,), jnp.int32),
        jnp.full((4,), -1, jnp.int32),
        jnp.full((6,), -1, jnp.int32),
    )


def make_base_rpa_scratch_shapes(
    *,
    q: jax.Array,
    kv_tail_shape: tuple[int, ...],
    kv_dtype,
    out_dtype,
    actual_num_kv_heads: int,
    num_q_heads_per_kv_head: int,
    head_dim: int,
    bq_sz: int,
    bkv_sz: int,
    num_kv_heads_x2_per_kv_packing: int,
) -> list[Any]:
    """Build the 7 shared double-buffered VMEM and DMA scratch specs for RPA v3.

    Scratch slot layout:
      [0] `bkv_x2_ref`: double-buffered KV VMEM tile (padded by +1 on bank conflict)
      [1] `bq_x2_ref`: double-buffered Q VMEM tile
      [2] `bo_x2_ref`: double-buffered O VMEM tile
      [3] `sems`: `DMA((4, 2))` rows `(SEM_BKV_FETCH, SEM_BQ_FETCH, SEM_BO_SEND,
          SEM_BKV_UPDATE)`
      [4] `l_ref`: online softmax row-sum VMEM scratch
      [5] `m_ref`: online softmax row-max VMEM scratch
      [6] `acc_ref`: output accumulator VMEM scratch
    """
    bkv_stride = num_kv_heads_x2_per_kv_packing + int(
        has_bank_conflicts(num_kv_heads_x2_per_kv_packing)
    )
    bkv_double_buf = pltpu.VMEM((2, bkv_sz, bkv_stride, *kv_tail_shape), kv_dtype)
    bq_double_buf = pltpu.VMEM((2, actual_num_kv_heads, bq_sz, *q.shape[2:]), q.dtype)
    l_scratch = pltpu.VMEM(
        (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128), out_dtype
    )
    acc_scratch = pltpu.VMEM(
        (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim), out_dtype
    )
    return [
        bkv_double_buf,
        bq_double_buf,
        bq_double_buf,
        pltpu.SemaphoreType.DMA((4, 2)),
        l_scratch,
        l_scratch,
        acc_scratch,
    ]


def invoke_rpa_pallas_kernel(
    *,
    kernel_fn: Callable[..., Any],
    scalar_prefetches: tuple[jax.Array, ...],
    hbm_inputs: tuple[jax.Array, ...],
    scratch_shapes: list[Any],
    out_arrays: tuple[jax.Array, jax.Array],
    input_output_aliases: dict[int, int],
    vmem_limit_bytes: int,
    scope_name: str,
    tpu_version: int,
    disable_bounds_checks: bool = True,
    disable_semaphore_checks: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Compile and invoke a single-core pipelined RPA v3 `pallas_call`."""
    hbm_cls = pltpu.HBM if tpu_version >= 7 else jax.ShapeDtypeStruct
    kernel = pl.pallas_call(
        kernel_fn,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * len(hbm_inputs),
            out_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * len(out_arrays),
            grid=(1,),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=disable_bounds_checks,
            disable_semaphore_checks=disable_semaphore_checks,
        ),
        out_shape=[hbm_cls(shape=a.shape, dtype=a.dtype) for a in out_arrays],
        input_output_aliases=input_output_aliases,
        name=scope_name,
    )
    if tpu_version >= 7:

        @jax.jit
        def run(scalars, *args):
            return kernel(
                *scalars,
                *(pltpu.with_memory_space_constraint(a, pltpu.HBM) for a in args),
            )

        return run(scalar_prefetches, *hbm_inputs)
    return kernel(*scalar_prefetches, *hbm_inputs)


def run_bundled_rpa(
    rpa_fn: Callable[..., tuple[jax.Array, jax.Array]],
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache_bundle: jax.Array,
    layer_idx: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    **rpa_kwargs,
) -> tuple[jax.Array, jax.Array]:
    """Flatten a block-major `[num_pages, num_layers, ...]` bundle and run `rpa_fn`."""
    num_blocks, num_layers = kv_cache_bundle.shape[:2]
    flat_bundle = kv_cache_bundle.reshape(
        num_blocks * num_layers, *kv_cache_bundle.shape[2:]
    )
    out, new_flat_bundle = rpa_fn(
        queries,
        keys,
        values,
        flat_bundle,
        kv_lens,
        page_indices * num_layers + layer_idx,
        cu_q_lens,
        distribution,
        **rpa_kwargs,
    )
    return out, new_flat_bundle.reshape(
        num_blocks, num_layers, *new_flat_bundle.shape[1:]
    )

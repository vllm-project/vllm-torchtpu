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
"""Double-buffered sequence/tile execution pipeline for ragged paged attention v3."""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.pallas_util import cdiv
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import staging


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
    staging_refs: staging.StagingRefs,
    kv_lens_ref,
    sem_ids_ref,
    kv_update_ring: staging.KvUpdateRing,
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
    flush_sliding_window_kv_updates: bool = False,
    static_q_len: int | None = None,
    debug_mode: bool = False,
) -> None:
    """Execute one sequence step of the double-buffered RPA v3 DMA/compute pipeline.

    Before starting each KV fetch (`prologue` and `prefetch_next_bkv`), this
    pipeline calls `kv_update_ring.wait(bkv_sem_idx)` so `fetch_bkv_fn` never
    overwrites a double-buffer slot while an earlier KV writeback is in flight.

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
      sliding_window: Optional sliding-window size.
      static_q_len: Optional compile-time static query length (`1` for decode,
        `chunk_prefill_size` for prefill).
      debug_mode: If true, emits `[RPA debug]` trace prints and skips DMAs/compute.
    """
    assert staging_refs.bq_x2_ref.shape == staging_refs.bo_x2_ref.shape
    assert staging_refs.bq_x2_ref.dtype == staging_refs.bo_x2_ref.dtype
    assert acc_ref.dtype == staging_refs.bo_x2_ref.dtype
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
            # Re-initialize l, m, acc to 0 before bkv loop.
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

            # Prefetch next bq
            @pl.when(next_seq_idx < end_seq_idx)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                staging.fetch_bq(
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

                # Prefetch next bkv once any prior writeback from that slot finishes.
                @pl.when(next_seq_idx < end_seq_idx)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    kv_update_ring.wait(next_bkv_sem_idx)
                    fetch_bkv_fn(next_seq_idx, next_bkv_idx, next_bkv_sem_idx)

                # Wait for cur bq if not ready yet.
                @pl.when(bkv_idx == start_bkv_idx)
                def wait_cur_bq():
                    staging.fetch_bq(
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

                # Start updating bkv to kv cache if applicable.
                is_last_bkv_visit = (
                    jnp.logical_or(
                        bq_idx == num_bq - 1, bkv_idx < next_bq_start_bkv_idx
                    )
                    if (flush_sliding_window_kv_updates and sliding_window is not None)
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

            # Load acc and calculate final output before waiting on previous bo DMA.
            out = staging.normalize_bo(acc_ref, l_ref)

            # Wait for previous bo to be fully sent before storing new bo.
            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            staging.wait_send_bo(
                staging_refs,
                start_seq_idx,
                seq_idx,
                bo_sem_idx,
                bq_sz=bq_sz,
                debug_mode=debug_mode,
            )

            # Store output from acc to bo and send cur bo.
            staging.store_bo(staging_refs.bo_x2_ref, bo_sem_idx, out)
            staging.start_send_bo(
                staging_refs,
                seq_idx,
                bq_idx,
                bo_sem_idx,
                bq_sz=bq_sz,
                debug_mode=debug_mode,
            )

    @pl.when(seq_idx == start_seq_idx)
    def prologue():
        staging.fetch_bq(
            staging_refs, start_seq_idx, 0, 0, bq_sz=bq_sz, debug_mode=debug_mode
        )
        kv_update_ring.wait(0)
        fetch_bkv_fn(start_seq_idx, cur_seq_start_bkv_idx, 0)

    @pl.when(jnp.logical_and(start_seq_idx <= seq_idx, seq_idx < end_seq_idx))
    def pipeline():
        process()

    @pl.when(seq_idx == end_seq_idx - 1)
    def epilogue():
        for i in range(2):
            staging.wait_send_bo(
                staging_refs,
                start_seq_idx,
                seq_idx,
                i,
                bq_sz=bq_sz,
                debug_mode=debug_mode,
            )
            kv_update_ring.wait(i)

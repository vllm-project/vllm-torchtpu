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
"""Stateless Pallas TPU DMA and batch-width helpers for SparseMLA."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla import kv_cache_utils
from vllm_torchtpu.kernels.mla.sparse import native_sc_gather
from vllm_torchtpu.kernels.mla.sparse.config import (
    Q_ROPE_HEAD_DIM,
    SEND_PAD_ROWS,
    AttentionKernelConfig,
)
from vllm_torchtpu.kernels.mla.sparse.kernel_context import KernelGeometry
from vllm_torchtpu.kernels.mla.sparse.ref_registry import AttentionRefs
from vllm_torchtpu.kernels.mla.sparse.testing_faults import FaultHooks


def async_copy(src: Any, dst: Any, sem: Any, *, wait: bool = False) -> None:
    """Starts or waits a Pallas TPU asynchronous DMA transfer."""
    cp = pltpu.make_async_copy(src, dst, sem)
    if wait:
        cp.wait()
    else:
        cp.start()


def get_batch_max_kv_len(
    kv_lens_ref: Any,
    seq_idx_start: Any,
    batch_size: int,
) -> Any:
    """Returns the maximum ``kv_len`` across a batch of ``batch_size`` tokens."""
    max_kv = jnp.int32(0)
    for b in range(batch_size):
        max_kv = jnp.maximum(max_kv, kv_lens_ref[seq_idx_start + b])
    return max_kv


def with_row_width(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    hooks: FaultHooks,
    seq_idx_start: Any,
    fn: Callable[[int | None], None],
) -> None:
    """Dispatches ``fn(rows)`` for the smallest ``row_widths`` bucket covering a batch.

    Passes ``None`` (all ``bkv_sz`` rows) when no configured width covers every
    token's ``kv_len`` or when ``cfg.row_widths`` is empty.
    """
    row_widths = cfg.row_widths
    if not row_widths:
        fn(None)
        return
    kv_lens_ref = refs.kv_lens
    need = kv_lens_ref[seq_idx_start]
    for b in range(1, cfg.batch_size):
        need = jnp.maximum(need, kv_lens_ref[seq_idx_start + b])
    widths = hooks.effective_row_widths(row_widths)
    lower: int | None = None
    for upper, width in zip((*row_widths, None), widths):
        if lower is None:
            assert upper is not None  # row_widths is non-empty here
            fits = need <= upper
        elif upper is None:
            fits = need > lower
        else:
            fits = (need > lower) & (need <= upper)
        pl.when(fits)(functools.partial(fn, width))
        lower = upper


def fetch_live_rows(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    seq_idx_start: Any,
    bkv_sem_idx: Any,
    rows: int,
    *,
    wait: bool = False,
) -> None:
    """Fetches or waits per-token live KV rows rounded up to 32 (<= ``rows``)."""
    sem_nope = refs.sems.kv_nope(bkv_sem_idx)
    sem_rope = refs.sems.kv_rope(bkv_sem_idx)
    bkv_nope_vmem_ref = refs.bkv_nope_x2.at[bkv_sem_idx, :]
    bkv_rope_vmem_ref = refs.bkv_rope_x2.at[bkv_sem_idx, :]
    # Token b moves rows [0, n_b) where n_b is kv_len rounded up to 32
    # (>= 32, <= rows); one DMA per (token, cache) on the cache semaphore.
    page_idx_start = seq_idx_start - geom.start_seq_idx
    rope_div = geom.bkv_sz // refs.cache_kv_rope.shape[1]
    for b in range(cfg.batch_size):
        kv_b = refs.kv_lens[seq_idx_start + b]
        n_b = jnp.clip((kv_b + 31) // 32 * 32, 32, rows)
        n_b = pl.multiple_of(n_b, 32)
        r_b = pl.multiple_of(n_b // rope_div, 8)
        nope_dst = bkv_nope_vmem_ref.at[pl.ds(b, 1), pl.ds(0, n_b)]
        rope_dst = bkv_rope_vmem_ref.at[pl.ds(b, 1), pl.ds(0, r_b)]
        if wait:
            async_copy(nope_dst, nope_dst, sem_nope, wait=True)
            async_copy(rope_dst, rope_dst, sem_rope, wait=True)
        else:
            async_copy(
                refs.cache_kv_nope.at[pl.ds(page_idx_start + b, 1), pl.ds(0, n_b)],
                nope_dst,
                sem_nope,
                wait=False,
            )
            async_copy(
                refs.cache_kv_rope.at[pl.ds(page_idx_start + b, 1), pl.ds(0, r_b)],
                rope_dst,
                sem_rope,
                wait=False,
            )


def fetch_bkv_slice(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    seq_idx_start: Any,
    bkv_sem_idx: Any,
    slice_sz: int,
    *,
    wait: bool = False,
) -> None:
    """Fetches or waits the first ``slice_sz`` KV rows for one batch."""
    sem_nope = refs.sems.kv_nope(bkv_sem_idx)
    sem_rope = refs.sems.kv_rope(bkv_sem_idx)
    nope_rows = (
        slice_sz if cfg.sparsecore_layout else slice_sz * kv_cache_utils.WORD_BYTES
    )
    rope_rows = (
        slice_sz // native_sc_gather.ROPE_PACKING if cfg.sparsecore_layout else slice_sz
    )
    bkv_nope_vmem_ref = refs.bkv_nope_x2.at[bkv_sem_idx, :, pl.ds(0, nope_rows), :]
    bkv_rope_vmem_ref = refs.bkv_rope_x2.at[bkv_sem_idx, :, pl.ds(0, rope_rows), :]
    if not wait:
        page_idx_start = seq_idx_start - geom.start_seq_idx
        async_copy(
            refs.cache_kv_nope.at[
                pl.ds(page_idx_start, cfg.batch_size), pl.ds(0, nope_rows), :
            ],
            bkv_nope_vmem_ref,
            sem_nope,
            wait=False,
        )
        async_copy(
            refs.cache_kv_rope.at[
                pl.ds(page_idx_start, cfg.batch_size), pl.ds(0, rope_rows), :
            ],
            bkv_rope_vmem_ref,
            sem_rope,
            wait=False,
        )
    else:
        async_copy(bkv_nope_vmem_ref, bkv_nope_vmem_ref, sem_nope, wait=True)
        async_copy(bkv_rope_vmem_ref, bkv_rope_vmem_ref, sem_rope, wait=True)


def fetch_live_blocks(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    seq_idx_start: Any,
    bkv_sem_idx: Any,
    *,
    wait: bool = False,
) -> None:
    """Fetches or waits live-group atom blocks for one mixed-bucket batch."""
    group_tab_ref = refs.group_tab
    live_nope_hbm_ref = refs.live_nope
    live_rope_hbm_ref = refs.live_rope
    assert group_tab_ref is not None
    assert live_nope_hbm_ref is not None and live_rope_hbm_ref is not None

    nope_dst = refs.bkv_nope_x2.at[bkv_sem_idx, :]
    rope_dst = refs.bkv_rope_x2.at[bkv_sem_idx, :]
    sem_nope = refs.sems.kv_nope(bkv_sem_idx)
    sem_rope = refs.sems.kv_rope(bkv_sem_idx)

    batch_size = cfg.batch_size
    bkv_sz = geom.bkv_sz
    num_groups = group_tab_ref.shape[0] // 2
    atom_rows = live_nope_hbm_ref.shape[2]
    rope_atom_rows = live_rope_hbm_ref.shape[2]
    g = seq_idx_start // batch_size
    live = group_tab_ref[g]
    off = group_tab_ref[num_groups + g]
    num_atoms = bkv_sz // atom_rows
    # A whole group (every atom live) sits token-major at block `off`:
    # one DMA per cache from its first token row.
    whole_row = off * (batch_size * atom_rows // bkv_sz)

    @pl.when(live == num_atoms)
    def _copy_whole() -> None:
        if wait:
            async_copy(nope_dst, nope_dst, sem_nope, wait=True)
            async_copy(rope_dst, rope_dst, sem_rope, wait=True)
        else:
            async_copy(
                refs.cache_kv_nope.at[pl.ds(whole_row, batch_size)],
                nope_dst,
                sem_nope,
                wait=False,
            )
            async_copy(
                refs.cache_kv_rope.at[pl.ds(whole_row, batch_size)],
                rope_dst,
                sem_rope,
                wait=False,
            )

    for a in range(num_atoms):

        @pl.when((a < live) & (live < num_atoms))
        def _copy_block(*, a: int = a) -> None:
            nope_slot = nope_dst.at[:, pl.ds(a * atom_rows, atom_rows)]
            rope_slot = rope_dst.at[:, pl.ds(a * rope_atom_rows, rope_atom_rows)]
            if wait:
                async_copy(nope_slot, nope_slot, sem_nope, wait=True)
                async_copy(rope_slot, rope_slot, sem_rope, wait=True)
            else:
                async_copy(
                    live_nope_hbm_ref.at[off + a],
                    nope_slot,
                    sem_nope,
                    wait=False,
                )
                async_copy(
                    live_rope_hbm_ref.at[off + a],
                    rope_slot,
                    sem_rope,
                    wait=False,
                )


def fetch_bkv_batch(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    seq_idx_start: Any,
    bkv_sem_idx: Any,
    *,
    wait: bool = False,
    rows: int | None = None,
) -> None:
    """Fetches or waits one batch of gathered KV tiles across all kernel modes."""
    if refs.group_tab is not None:
        fetch_live_blocks(refs, cfg, geom, seq_idx_start, bkv_sem_idx, wait=wait)
        return
    if rows is not None and cfg.body.fetch_live_rows_only:
        fetch_live_rows(refs, cfg, geom, seq_idx_start, bkv_sem_idx, rows, wait=wait)
        return
    if rows is not None or (cfg.is_dcp_mode and cfg.row_widths):
        slice_sz = geom.bkv_sz if rows is None else rows
        fetch_bkv_slice(
            refs, cfg, geom, seq_idx_start, bkv_sem_idx, slice_sz, wait=wait
        )
        return
    if cfg.is_dcp_mode:
        batch_max_kv = get_batch_max_kv_len(refs.kv_lens, seq_idx_start, cfg.batch_size)
        lo = 0
        rungs = geom.kv_rungs
        for i, rung in enumerate(rungs):
            hi_ok = True if i == len(rungs) - 1 else batch_max_kv <= rung
            pl.when((batch_max_kv > lo) & hi_ok)(
                functools.partial(
                    fetch_bkv_slice,
                    refs,
                    cfg,
                    geom,
                    seq_idx_start,
                    bkv_sem_idx,
                    rung,
                    wait=wait,
                )
            )
            lo = rung
        return

    sem_nope = refs.sems.kv_nope(bkv_sem_idx)
    sem_rope = refs.sems.kv_rope(bkv_sem_idx)
    bkv_nope_vmem_ref = refs.bkv_nope_x2.at[bkv_sem_idx, :]
    bkv_rope_vmem_ref = refs.bkv_rope_x2.at[bkv_sem_idx, :]
    if not wait:
        # Index into cache_kv_hbm_ref is relative to the current chunk.
        page_idx_start = seq_idx_start - geom.start_seq_idx
        async_copy(
            refs.cache_kv_nope.at[pl.ds(page_idx_start, cfg.batch_size)],
            bkv_nope_vmem_ref,
            sem_nope,
            wait=False,
        )
        async_copy(
            refs.cache_kv_rope.at[pl.ds(page_idx_start, cfg.batch_size)],
            bkv_rope_vmem_ref,
            sem_rope,
            wait=False,
        )
    else:
        async_copy(bkv_nope_vmem_ref, bkv_nope_vmem_ref, sem_nope, wait=True)
        async_copy(bkv_rope_vmem_ref, bkv_rope_vmem_ref, sem_rope, wait=True)


def fetch_bq_batch(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    seq_idx_start: Any,
    bq_sem_idx: Any,
    *,
    wait: bool = False,
) -> None:
    """Fetches or waits one batch of query (and split-Q / Q-RoPE) tiles."""
    sem = refs.sems.q(bq_sem_idx)
    bq_vmem_ref = refs.bq_x2.at[bq_sem_idx, :]
    batch_size = cfg.batch_size

    if not wait:
        async_copy(
            refs.q.at[pl.ds(seq_idx_start, batch_size)],
            bq_vmem_ref,
            sem,
            wait=False,
        )
        if cfg.split_q and cfg.body.narrow_q_rope_dma:
            # narrow_q_rope_dma: only lanes 128:256 of the raw head, the lanes
            # `load_bq` reads, into the 128-lane q_pe buffer.
            assert refs.q_pe is not None and refs.bq_pe_x2 is not None
            async_copy(
                refs.q_pe.at[
                    pl.ds(seq_idx_start, batch_size),
                    :,
                    pl.ds(Q_ROPE_HEAD_DIM - 128, 128),
                ],
                refs.bq_pe_x2.at[bq_sem_idx, :],
                sem,
                wait=False,
            )
        elif cfg.split_q:
            # Same semaphore; the waits below mirror both starts.
            assert refs.q_pe is not None and refs.bq_pe_x2 is not None
            async_copy(
                refs.q_pe.at[pl.ds(seq_idx_start, batch_size)],
                refs.bq_pe_x2.at[bq_sem_idx, :],
                sem,
                wait=False,
            )
        if cfg.q_rope:
            assert refs.q_rope_table is not None and refs.bq_tab_x2 is not None
            async_copy(
                refs.q_rope_table.at[pl.ds(seq_idx_start, batch_size)],
                refs.bq_tab_x2.at[bq_sem_idx, :],
                sem,
                wait=False,
            )
    else:
        async_copy(bq_vmem_ref, bq_vmem_ref, sem, wait=True)
        if cfg.split_q:
            assert refs.bq_pe_x2 is not None
            bq_pe_vmem_ref = refs.bq_pe_x2.at[bq_sem_idx, :]
            async_copy(bq_pe_vmem_ref, bq_pe_vmem_ref, sem, wait=True)
        if cfg.q_rope:
            assert refs.bq_tab_x2 is not None
            bq_tab_vmem_ref = refs.bq_tab_x2.at[bq_sem_idx, :]
            async_copy(bq_tab_vmem_ref, bq_tab_vmem_ref, sem, wait=True)


def send_bo_batch(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    seq_idx_start: Any,
    bo_sem_idx: Any,
    *,
    wait: bool = False,
) -> None:
    """Sends or waits one batch of output tiles from VMEM to HBM."""
    if cfg.xchg_enabled:
        return  # The slabs leave by the exchange's remote pushes.
    sem = refs.sems.o(bo_sem_idx)
    vmem_ref = refs.bo_x2.at[bo_sem_idx, :]
    batch_size = cfg.batch_size

    if cfg.preserve_suffix:
        # Split routing owns only the prefix, including in the last rounded
        # batch. Leave suffix queries untouched in HBM.
        assert geom.prefix_end is not None
        rows = jnp.minimum(batch_size, geom.prefix_end - seq_idx_start)
        async_copy(
            vmem_ref.at[pl.ds(0, rows)],
            refs.out.at[pl.ds(seq_idx_start, rows)],
            sem,
            wait=wait,
        )
        return

    if not wait:
        if cfg.send_enabled:
            # This step's partial rows of every destination block.
            assert refs.lse is not None
            row = pl.multiple_of(
                SEND_PAD_ROWS + seq_idx_start * geom.send_per, SEND_PAD_ROWS
            )
            dst = refs.lse.at[:, pl.ds(row, batch_size * geom.send_per), :]
        else:
            dst = refs.out.at[pl.ds(seq_idx_start - cfg.out_row_offset, batch_size)]
        async_copy(vmem_ref, dst, sem, wait=False)
    else:
        async_copy(vmem_ref, vmem_ref, sem, wait=True)


def send_blse_batch(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    seq_idx_start: Any,
    bo_sem_idx: Any,
    *,
    wait: bool = False,
) -> None:
    """Sends or waits one batch of LSE tiles from VMEM to HBM or fill VMEM."""
    del geom
    sem = refs.sems.lse(bo_sem_idx)
    assert refs.blse_x2 is not None
    vmem_ref = refs.blse_x2.at[bo_sem_idx, :]

    if cfg.send_enabled:
        # No lse output: keep the step's rows for the word build.
        if not wait:
            assert refs.lse_fill is not None
            refs.lse_fill.fill[lax.div(seq_idx_start, cfg.batch_size)] = vmem_ref[...]
        return
    if not wait:
        assert refs.lse is not None
        async_copy(
            vmem_ref,
            refs.lse.at[pl.ds(seq_idx_start - cfg.out_row_offset, cfg.batch_size)],
            sem,
            wait=False,
        )
    else:
        async_copy(vmem_ref, vmem_ref, sem, wait=True)


def make_lse_fill_copy(refs: AttentionRefs) -> Any:
    """Builds the async DMA descriptor for the kernel-owned LSE sentinel fill."""
    assert refs.lse_fill is not None and refs.lse is not None
    return pltpu.make_async_copy(refs.lse_fill.fill, refs.lse, refs.lse_fill.sem)


def run_lse_fill(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    hooks: FaultHooks,
    has_work: Any,
) -> None:
    """Initializes the kernel-owned LSE sentinel buffer on program 0."""
    if not cfg.lse_fill:
        return

    @pl.when(pl.program_id(0) == 0)
    def _fill_lse() -> None:
        default_sentinel = (
            -jnp.inf if cfg.lse_mask_empty else float(jnp.finfo(jnp.float32).min)
        )
        sentinel = hooks.lse_fill_sentinel(default_sentinel)
        assert refs.lse_fill is not None
        fill_ref = refs.lse_fill.fill
        fill_ref[...] = jnp.full(fill_ref.shape, sentinel, jnp.float32)
        if cfg.send_enabled:
            return  # VMEM only: the word build reads it.
        if geom.nofill:
            # The only step stores every row; the fill is needed only when
            # there is no work (no step stores anything).
            @pl.when(jnp.logical_not(has_work))
            def _() -> None:
                make_lse_fill_copy(refs).start()
                make_lse_fill_copy(refs).wait()

            return
        make_lse_fill_copy(refs).start()

        @pl.when(jnp.logical_not(has_work))
        def _() -> None:
            make_lse_fill_copy(refs).wait()

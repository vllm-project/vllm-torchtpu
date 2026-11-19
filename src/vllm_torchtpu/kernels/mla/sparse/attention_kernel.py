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
"""Mode-specialized Pallas TPU SparseMLA attention kernels."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla.sparse import dcp_exchange, dma
from vllm_torchtpu.kernels.mla.sparse.attention_body import (
    _DEFAULT_BODY,
    dispatch_attention_body,
    maybe_fixup_capped_overflow,
)
from vllm_torchtpu.kernels.mla.sparse.config import AttentionKernelConfig
from vllm_torchtpu.kernels.mla.sparse.kernel_context import (
    DoubleBufferState,
    KernelGeometry,
)
from vllm_torchtpu.kernels.mla.sparse.ref_registry import (
    AttentionRefs,
    BoundsLayout,
    RefLayout,
)
from vllm_torchtpu.kernels.mla.sparse.testing_faults import NO_FAULT_HOOKS, FaultHooks


def is_decode_kernel(cfg: AttentionKernelConfig) -> bool:
    """Returns whether ``cfg`` uses the decode/send-exchange kernel path."""
    return (
        not cfg.has_group_tab
        and not cfg.preserve_suffix
        and bool(cfg.lse_fill or cfg.send_enabled or cfg.xchg_enabled or cfg.row_widths)
    )


def _prefetch_next_and_wait_current(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    hooks: FaultHooks,
    db: DoubleBufferState,
    *,
    wait_bkv_now: bool,
) -> tuple[Any, Any]:
    """Prefetches the next batch, waits current inputs and 2-steps-ago sends."""
    bi_sem_idx = db.current_in_slot()
    next_seq_idx = geom.batch_start_seq_idx + cfg.batch_size
    next_bi_sem_idx = db.next_slot(bi_sem_idx)

    @pl.when(next_seq_idx < geom.end_seq_idx)
    def _prefetch_next_seq() -> None:
        db.flip_in_slot(next_bi_sem_idx)
        dma.fetch_bq_batch(refs, cfg, next_seq_idx, next_bi_sem_idx)
        dma.with_row_width(
            refs,
            cfg,
            hooks,
            next_seq_idx,
            lambda rows: dma.fetch_bkv_batch(
                refs, cfg, geom, next_seq_idx, next_bi_sem_idx, rows=rows
            ),
        )

    bo_sem_idx = db.acquire_out_slot()

    dma.fetch_bq_batch(refs, cfg, geom.batch_start_seq_idx, bi_sem_idx, wait=True)
    if wait_bkv_now:
        dma.fetch_bkv_batch(
            refs, cfg, geom, geom.batch_start_seq_idx, bi_sem_idx, wait=True
        )

    @pl.when(pl.program_id(0) >= 2)
    def _wait_send() -> None:
        # This slot holds the transfer from two batches earlier.
        sent_start = (
            geom.batch_start_seq_idx - 2 * cfg.batch_size
            if cfg.preserve_suffix
            else geom.batch_start_seq_idx
        )
        dma.send_bo_batch(refs, cfg, geom, sent_start, bo_sem_idx, wait=True)
        if cfg.return_lse:
            dma.send_blse_batch(
                refs, cfg, geom, geom.batch_start_seq_idx, bo_sem_idx, wait=True
            )

    return bi_sem_idx, bo_sem_idx


def _run_zero_batch(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    hooks: FaultHooks,
    bo_sem_idx: Any,
) -> None:
    """Stores zero outputs and ``-inf`` LSE for an all-empty DCP batch."""
    zero_out = jnp.zeros((geom.num_q_heads, geom.o_dim), dtype=geom.q_dtype)
    neg_inf_lse = jnp.full((geom.num_q_heads, 1), -jnp.inf, dtype=jnp.float32)
    for batch_idx in range(cfg.batch_size):
        dcp_exchange.store_out(refs, cfg, geom, hooks, bo_sem_idx, batch_idx, zero_out)
        if cfg.return_lse:
            dcp_exchange.store_blse(refs, cfg, geom, bo_sem_idx, batch_idx, neg_inf_lse)


def _run_dcp_rungs_body(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    hooks: FaultHooks,
    bi_sem_idx: Any,
    bo_sem_idx: Any,
    batch_max_kv_len: Any,
) -> None:
    """Dispatches DCP KV slice rungs matching the batch's maximum ``kv_len``."""
    lo = 0
    rungs = geom.kv_rungs
    for i, rung in enumerate(rungs):
        hi_ok = True if i == len(rungs) - 1 else batch_max_kv_len <= rung
        pl.when((batch_max_kv_len > lo) & hi_ok)(
            functools.partial(
                dispatch_attention_body,
                refs,
                cfg,
                geom,
                hooks,
                bi_sem_idx,
                bo_sem_idx,
                rung,
                _DEFAULT_BODY,
            )
        )
        lo = rung


def _run_live_group_body(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    hooks: FaultHooks,
    bi_sem_idx: Any,
    bo_sem_idx: Any,
) -> None:
    """Runs the live-group attention body sized by ``cfg.live_widths``."""
    if cfg.live_widths:
        assert refs.group_tab is not None and refs.live_nope is not None
        atom_rows = refs.live_nope.shape[2]
        need = refs.group_tab[geom.batch_start_seq_idx // cfg.batch_size] * atom_rows
        lower: int | None = None
        for width in (*cfg.live_widths, geom.bkv_sz):
            fits = need <= width if lower is None else (need > lower) & (need <= width)
            pl.when(fits)(
                functools.partial(
                    dispatch_attention_body,
                    refs,
                    cfg,
                    geom,
                    hooks,
                    bi_sem_idx,
                    bo_sem_idx,
                    width,
                    cfg.body,
                )
            )
            lower = width
    else:
        dispatch_attention_body(
            refs, cfg, geom, hooks, bi_sem_idx, bo_sem_idx, geom.bkv_sz, cfg.body
        )


def _run_row_widths_body(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    hooks: FaultHooks,
    bi_sem_idx: Any,
    bo_sem_idx: Any,
    batch_max_kv_len: Any = None,
) -> None:
    """Waits the selected row-width KV DMA and runs the attention body."""

    def _wait_and_run(rows: int | None) -> None:
        # Every batch started a fetch of one width; wait on it even when empty.
        dma.fetch_bkv_batch(
            refs,
            cfg,
            geom,
            geom.batch_start_seq_idx,
            bi_sem_idx,
            wait=True,
            rows=rows,
        )
        eff_rows = geom.bkv_sz if rows is None else rows
        if batch_max_kv_len is not None:
            pl.when(batch_max_kv_len > 0)(
                functools.partial(
                    dispatch_attention_body,
                    refs,
                    cfg,
                    geom,
                    hooks,
                    bi_sem_idx,
                    bo_sem_idx,
                    eff_rows,
                    cfg.body,
                )
            )
        else:
            dispatch_attention_body(
                refs, cfg, geom, hooks, bi_sem_idx, bo_sem_idx, eff_rows, cfg.body
            )

    dma.with_row_width(refs, cfg, hooks, geom.batch_start_seq_idx, _wait_and_run)


def _finish_step_and_send(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    hooks: FaultHooks,
    bi_sem_idx: Any,
    bo_sem_idx: Any,
) -> None:
    """Runs optional overflow fix-up and starts output/LSE async sends."""
    maybe_fixup_capped_overflow(refs, cfg, geom, hooks, bi_sem_idx, bo_sem_idx)
    dma.send_bo_batch(refs, cfg, geom, geom.batch_start_seq_idx, bo_sem_idx)
    if cfg.lse_fill and not cfg.send_enabled and not geom.nofill:
        # The sentinel fill must land before this step's rows overwrite it.
        @pl.when(pl.program_id(0) == 0)
        def _wait_lse_fill() -> None:
            dma.make_lse_fill_copy(refs).wait()

    if cfg.return_lse:
        dma.send_blse_batch(refs, cfg, geom, geom.batch_start_seq_idx, bo_sem_idx)


def _run_epilogue(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    has_work: Any,
) -> None:
    """Waits on the final one or two in-flight output/LSE DMA slots."""

    @pl.when(jnp.logical_and(geom.batch_end_seq_idx == geom.end_seq_idx - 1, has_work))
    def _epilogue() -> None:
        # Partial stores must wait for the same byte count as their send.
        last_even = (pl.num_programs(0) - 1) // 2 * 2
        sent_start = (
            geom.start_seq_idx + last_even * cfg.batch_size
            if cfg.preserve_suffix
            else 0
        )
        dma.send_bo_batch(refs, cfg, geom, sent_start, 0, wait=True)
        if cfg.return_lse:
            dma.send_blse_batch(refs, cfg, geom, 0, 0, wait=True)

        @pl.when(pl.num_programs(0) >= 2)
        def _wait_1() -> None:
            last_odd = (pl.num_programs(0) - 2) // 2 * 2 + 1
            sent_start_odd = (
                geom.start_seq_idx + last_odd * cfg.batch_size
                if cfg.preserve_suffix
                else 0
            )
            dma.send_bo_batch(refs, cfg, geom, sent_start_odd, 1, wait=True)
            if cfg.return_lse:
                dma.send_blse_batch(refs, cfg, geom, 0, 1, wait=True)


def _decode_attention_kernel(
    *raw_refs: Any,
    cfg: AttentionKernelConfig,
    ref_layout: RefLayout,
    bounds_layout: BoundsLayout,
    hooks: FaultHooks = NO_FAULT_HOOKS,
) -> None:
    """Pallas TPU attention kernel specialized for decode and fused DCP exchange."""
    assert not cfg.has_group_tab, "decode kernel does not support live-group tables"
    assert not cfg.preserve_suffix, "decode kernel does not support preserve_suffix"
    refs = AttentionRefs.from_bound(
        ref_layout.bind(*raw_refs), bounds_layout, return_lse=cfg.return_lse
    )
    geom = KernelGeometry.from_refs(refs, cfg)
    db = DoubleBufferState(refs.sem_ids)
    has_work = geom.end_seq_idx > geom.start_seq_idx if cfg.lse_fill else True

    dma.run_lse_fill(refs, cfg, geom, hooks, has_work)

    @pl.when(jnp.logical_and(geom.batch_start_seq_idx == geom.start_seq_idx, has_work))
    def _prologue() -> None:
        dma.fetch_bq_batch(refs, cfg, geom.batch_start_seq_idx, 0)
        dma.with_row_width(
            refs,
            cfg,
            hooks,
            geom.batch_start_seq_idx,
            lambda rows: dma.fetch_bkv_batch(
                refs, cfg, geom, geom.batch_start_seq_idx, 0, rows=rows
            ),
        )

    if cfg.xchg_enabled:
        if cfg.xchg_merge:
            hooks.poison_merge_recv_scratch(refs.recv_v)
        # One grid step: every rank signals and participates unconditionally.
        dcp_exchange.xchg_signal(refs, cfg)
        if hooks.wait_exchange_barrier_at_entry():
            pl.semaphore_wait(pltpu.get_barrier_semaphore(), cfg.send_world)

    def _step() -> None:
        bi, bo = _prefetch_next_and_wait_current(
            refs, cfg, geom, hooks, db, wait_bkv_now=not cfg.row_widths
        )
        if cfg.is_dcp_mode:
            max_kv = dma.get_batch_max_kv_len(
                refs.kv_lens, geom.batch_start_seq_idx, cfg.batch_size
            )
            pl.when(max_kv == 0)(
                functools.partial(_run_zero_batch, refs, cfg, geom, hooks, bo)
            )
            if cfg.row_widths:
                _run_row_widths_body(refs, cfg, geom, hooks, bi, bo, max_kv)
            else:
                _run_dcp_rungs_body(refs, cfg, geom, hooks, bi, bo, max_kv)
        elif cfg.row_widths:
            _run_row_widths_body(refs, cfg, geom, hooks, bi, bo)
        else:
            dispatch_attention_body(refs, cfg, geom, hooks, bi, bo, None, cfg.body)
        _finish_step_and_send(refs, cfg, geom, hooks, bi, bo)

    if cfg.lse_fill:
        pl.when(has_work)(_step)
    else:
        _step()

    is_last = (
        pl.program_id(0) == pl.num_programs(0) - 1
        if cfg.send_enabled and not cfg.xchg_enabled
        else None
    )
    dcp_exchange.finalize_send_exchange(refs, cfg, geom, hooks, is_last=is_last)
    _run_epilogue(refs, cfg, geom, has_work)
    if is_last is not None:
        pl.when(is_last)(lambda: dcp_exchange.make_send_words_copy(refs).wait())


def _mixed_prefill_attention_kernel(
    *raw_refs: Any,
    cfg: AttentionKernelConfig,
    ref_layout: RefLayout,
    bounds_layout: BoundsLayout,
    hooks: FaultHooks = NO_FAULT_HOOKS,
) -> None:
    """Pallas TPU attention kernel specialized for mixed and prefill buckets."""
    assert not cfg.row_widths, "mixed/prefill kernel does not support row_widths"
    assert not cfg.send_enabled and not cfg.xchg_enabled, (
        "mixed/prefill kernel does not support send layout or fused exchange"
    )
    refs = AttentionRefs.from_bound(
        ref_layout.bind(*raw_refs), bounds_layout, return_lse=cfg.return_lse
    )
    geom = KernelGeometry.from_refs(refs, cfg)
    db = DoubleBufferState(refs.sem_ids)
    has_work = geom.end_seq_idx > geom.start_seq_idx if cfg.lse_fill else True

    if cfg.lse_fill:
        dma.run_lse_fill(refs, cfg, geom, hooks, has_work)

    @pl.when(jnp.logical_and(geom.batch_start_seq_idx == geom.start_seq_idx, has_work))
    def _prologue() -> None:
        dma.fetch_bq_batch(refs, cfg, geom.batch_start_seq_idx, 0)
        dma.fetch_bkv_batch(refs, cfg, geom, geom.batch_start_seq_idx, 0)

    def _step() -> None:
        bi_sem_idx, bo_sem_idx = _prefetch_next_and_wait_current(
            refs, cfg, geom, hooks, db, wait_bkv_now=True
        )
        if cfg.has_group_tab:
            _run_live_group_body(refs, cfg, geom, hooks, bi_sem_idx, bo_sem_idx)
        elif cfg.is_dcp_mode:
            batch_max_kv_len = dma.get_batch_max_kv_len(
                refs.kv_lens, geom.batch_start_seq_idx, cfg.batch_size
            )
            pl.when(batch_max_kv_len == 0)(
                functools.partial(_run_zero_batch, refs, cfg, geom, hooks, bo_sem_idx)
            )
            _run_dcp_rungs_body(
                refs, cfg, geom, hooks, bi_sem_idx, bo_sem_idx, batch_max_kv_len
            )
        else:
            dispatch_attention_body(
                refs, cfg, geom, hooks, bi_sem_idx, bo_sem_idx, None, cfg.body
            )
        _finish_step_and_send(refs, cfg, geom, hooks, bi_sem_idx, bo_sem_idx)

    if cfg.lse_fill:
        pl.when(has_work)(_step)
    else:
        _step()

    _run_epilogue(refs, cfg, geom, has_work)


def select_attention_kernel(cfg: AttentionKernelConfig) -> Callable[..., None]:
    """Selects the mode-specialized Pallas attention kernel for ``cfg``."""
    if is_decode_kernel(cfg):
        return _decode_attention_kernel
    return _mixed_prefill_attention_kernel


def _attention_kernel(
    *raw_refs: Any,
    cfg: AttentionKernelConfig,
    ref_layout: RefLayout,
    bounds_layout: BoundsLayout,
    hooks: FaultHooks = NO_FAULT_HOOKS,
) -> None:
    """Dispatches a Pallas attention call to the mode-specialized kernel."""
    select_attention_kernel(cfg)(
        *raw_refs,
        cfg=cfg,
        ref_layout=ref_layout,
        bounds_layout=bounds_layout,
        hooks=hooks,
    )

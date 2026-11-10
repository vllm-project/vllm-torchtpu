# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Pallas kernel that builds the PCP current-pass page pool from token runs.

The all-RPA current pass attends a temporary page pool: pool page ``p`` holds
the global-order tokens of one request's (boundary page + current chunk),
``page`` tokens per page, token axis last (``[NTP, A, B, C, page]``). Each
pool page is a handful of contiguous runs of two token-last sources: the
gathered fresh KV of all ranks and the gathered boundary pages. The default
path (``history_rpa._element_pool``) gathers every slot's token element by
element and relayouts the result; this kernel copies each run instead.

Per pool page and run ``k`` (``runs[p, k] = (dst, src_sel, col, n)``): DMA the
two-page window of source ``src_sel`` that holds columns ``[col, col + n)``,
rotate it so ``col`` lands on lane ``dst`` and keep lanes ``[dst, dst + n)``.
Slots no run covers stay zero, like the element-wise build's invalid slots.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

RUN_FIELDS = 4  # dst lane, source (0 fresh, 1 boundary), source column, length


def max_runs_per_page(page_size: int, interleave_size: int) -> int:
    """A page's boundary tokens alternate rank every interleave chunk, and so do
    its fresh tokens (on the batch-flat coordinate); each part spans at most
    page // interleave + 1 chunks."""
    return 2 * (page_size // interleave_size + 1)


def _kernel(runs_ref, fresh_hbm, bnd_hbm, out_ref, win, sem, *, k_max, page):
    """Sources are row-major stacks [R, A, B, C, N]: source column ``col`` is
    row ``col // N``, lane ``col % N`` (the raw all-gathered fresh KV is
    [P, A, B, C, LT_pad]; the gathered boundary pages are [P * S, A, B, C, page]).
    A run never crosses a row. Rows at least two pages wide are read through a
    two-page window; one-page rows are read whole."""
    p = pl.program_id(0)
    shape = out_ref.shape[1:]
    nd = len(shape)
    lane = jax.lax.broadcasted_iota(jnp.int32, shape, nd - 1)
    out_ref[0] = jnp.zeros(shape, out_ref.dtype)
    for k in range(k_max):
        dst = runs_ref[p, k, 0]
        sel = runs_ref[p, k, 1]
        col = runs_ref[p, k, 2]
        n = runs_ref[p, k, 3]

        def _copy(src_hbm):
            width = src_hbm.shape[-1]
            span = 2 * page if width >= 2 * page else page
            row = col // width
            c = col - row * width
            w0 = jnp.minimum((c // page) * page, width - span)
            if span == 2 * page:
                cp = pltpu.make_async_copy(src_hbm.at[row, ..., pl.ds(w0, span)], win, sem)
            else:
                cp = pltpu.make_async_copy(src_hbm.at[row], win.at[..., pl.ds(0, page)], sem)
            cp.start()
            cp.wait()
            shift = jnp.mod(dst - (c - w0), 2 * page)
            rolled = pltpu.roll(win[...], shift, nd - 1)[..., :page]
            keep = jnp.logical_and(lane >= dst, lane < dst + n)
            out_ref[0] = jnp.where(keep, rolled, out_ref[0])

        @pl.when(jnp.logical_and(n > 0, sel == 0))
        def _():
            _copy(fresh_hbm)

        @pl.when(jnp.logical_and(n > 0, sel == 1))
        def _():
            _copy(bnd_hbm)


def build_pool(runs, fresh, boundary, *, page_size: int, interpret: bool | None = None):
    """runs [NTP, K, 4] int32. ``fresh`` / ``boundary`` are token-last sources,
    either flat [A, B, C, N] or row-stacked [R, A, B, C, N] (column = row * N +
    lane; flat is R = 1). Row-stacked inputs avoid relayouting the all-gather
    outputs. Returns the pool [NTP, A, B, C, page]."""
    ntp, k_max, _ = runs.shape
    srcs = []
    for src in (fresh, boundary):
        if src.ndim == 4:
            src = src[None]
        if src.shape[-1] % page_size or src.shape[-1] < page_size:
            raise ValueError(
                f"pool kernel source rows must be whole pages, got {src.shape[-1]} for page {page_size}"
            )
        srcs.append(src)
    fresh, boundary = srcs
    a, b, c = fresh.shape[1:4]
    if interpret is None:
        interpret = jax.default_backend() == "cpu"
    return pl.pallas_call(
        functools.partial(_kernel, k_max=k_max, page=page_size),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(ntp,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=pl.BlockSpec((1, a, b, c, page_size), lambda p, runs: (p, 0, 0, 0, 0)),
            scratch_shapes=[
                pltpu.VMEM((a, b, c, 2 * page_size), fresh.dtype),
                pltpu.SemaphoreType.DMA(()),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct((ntp, a, b, c, page_size), fresh.dtype),
        name="pcp_cur_pool_build",
        interpret=interpret,
    )(runs, fresh, boundary)


def pool_runs(is_b, valid, b_take, c_take, *, ntp, page_size, k_max, onehot=False):
    """Per-page runs from the element-wise build's slot selectors.

    Slot ``s`` of page ``p`` (``s = p * page + r``) takes boundary column
    ``b_take`` (``is_b``) or fresh column ``c_take``; invalid slots take
    nothing. A run is a maximal stretch of valid slots whose source and column
    advance by one. Returns (runs [NTP, K, 4] int32, overflow bool): overflow is
    True if some page needs more than K runs (callers must fall back).
    ``onehot`` (TPU_PCP_META_ONEHOT): compact each page's runs into its K slots with a
    one-hot select-reduce instead of scatters (run k has at most one start slot on its
    page, so the selected sum is that slot's value; absent runs stay 0). Same result.
    """
    sel = is_b.astype(jnp.int32).reshape(ntp, page_size)
    col = jnp.where(is_b, b_take, c_take).astype(jnp.int32).reshape(ntp, page_size)
    ok = valid.reshape(ntp, page_size)
    r = jnp.arange(page_size, dtype=jnp.int32)[None, :]
    prev_ok = jnp.concatenate((jnp.zeros((ntp, 1), bool), ok[:, :-1]), axis=1)
    prev_sel = jnp.concatenate((jnp.full((ntp, 1), -1, jnp.int32), sel[:, :-1]), axis=1)
    prev_col = jnp.concatenate((jnp.full((ntp, 1), -2, jnp.int32), col[:, :-1]), axis=1)
    start = jnp.logical_and(
        ok,
        jnp.logical_or(
            jnp.logical_not(prev_ok),
            jnp.logical_or(sel != prev_sel, col != prev_col + 1),
        ),
    )
    run_id = jnp.cumsum(start.astype(jnp.int32), axis=1) - 1  # [NTP, page]
    overflow = jnp.any(jnp.logical_and(ok, run_id >= k_max))
    if onehot:
        at_k = run_id[..., None] == jnp.arange(k_max, dtype=jnp.int32)  # [NTP, page, K]
        st = jnp.logical_and(start[..., None], at_k)
        pick = lambda v: jnp.sum(jnp.where(st, v[..., None], 0), axis=1, dtype=jnp.int32)
        dst = pick(jnp.broadcast_to(r, (ntp, page_size)))
        length = jnp.sum(jnp.logical_and(ok[..., None], at_k).astype(jnp.int32), axis=1, dtype=jnp.int32)
        return jnp.stack((dst, pick(sel), pick(col), length), axis=-1), overflow
    flat = jnp.where(jnp.logical_and(ok, run_id < k_max),
                     jnp.arange(ntp, dtype=jnp.int32)[:, None] * k_max + run_id, ntp * k_max)
    is_start = jnp.logical_and(start, run_id < k_max)
    tgt = jnp.where(is_start, flat, ntp * k_max).reshape(-1)
    n_slots = ntp * k_max
    dst = jnp.zeros((n_slots,), jnp.int32).at[tgt].set(jnp.broadcast_to(r, (ntp, page_size)).reshape(-1), mode="drop")
    src_sel = jnp.zeros((n_slots,), jnp.int32).at[tgt].set(sel.reshape(-1), mode="drop")
    src_col = jnp.zeros((n_slots,), jnp.int32).at[tgt].set(col.reshape(-1), mode="drop")
    length = jnp.zeros((n_slots,), jnp.int32).at[flat.reshape(-1)].add(1, mode="drop")
    runs = jnp.stack((dst, src_sel, src_col, length), axis=-1).reshape(ntp, k_max, RUN_FIELDS)
    return runs, overflow

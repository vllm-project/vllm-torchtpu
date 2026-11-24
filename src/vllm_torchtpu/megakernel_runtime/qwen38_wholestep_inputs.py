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
"""Whole-step kernel QSA inputs from the runner's metadata, with the stock addressing.

Builds, inside the whole-step JAX callable and before the kernel, the stage-C extra
inputs
(`wholestep.inputs.EXTRA_NAMES_C`) of every QSA layer from what the stock `_pooled_qsa`
receives: the unified pool, per-layer `(ring_ids, main_table, compressed_table)` flat
block
tables, `(main_slot, compressed_slot)`, the rope cache, positions, `query_start_loc` and
the
context lengths. Every address is the stock formula: main KV page = `main_table[req,
col]`
(one pool block of 128 tokens per column), compressed-key carrier page =
`compressed_table[req, col] * physical_pages + slot` (`pooled_qsa.select_blocks`), ring
and
tags read with `qsa_state.read_ring_and_tags`.

Rows: request `q` owns kernel rows `q*T .. q*T+T-1` (uniform verify windows, as the
eligibility check guarantees); per-layer tables are flattened `row = layer * nreq + q`.
"""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import compressed_kernel_pages
from vllm_torchtpu.kernels.qwen4_exp.qsa_state import read_ring_and_tags

from .qwen38_io_lean import io_lean

RING_ROWS = 8  # newest positions kept per request (lane = position % 8), >= ratio
IDX_DIM = 128
GROUP = 4  # compress ratio


def _fit(x: Any, n: int) -> Any:
    """Rows (requests / positions) padded or cropped to the kernel's static count.
    Cropping only
    drops padding rows: the driver calls the kernel for steps whose requests fit its
    bucket."""
    x = x[:n]
    if x.shape[0] < n:
        x = jnp.pad(x, ((0, n - x.shape[0]),) + ((0, 0),) * (x.ndim - 1))
    return x


def carrier_pages_per_block(pool: Any) -> int:
    """Compressed-key carrier pages (32 groups of 2 x 128 BF16 lanes) per pool block."""
    rows, width = 128 // GROUP, pool.shape[-1]
    return math.prod(pool.shape[1:]) * pool.dtype.itemsize // (rows * width * 2)


def manager_page_table(cktab: Any, mp: int, lp: int, physical: int) -> Any:
    """The per-kernel-page key table `[..., n_ck]` as a table of runs of `mp` consecutive
    kernel pages: entry m = `cktab[..., m * mp] // mp`, and the kernel copies pages
    `[entry * mp, entry * mp + mp)`. Exact when `mp` divides the `lp` kernel pages of a block
    column (`compressed_kernel_pages`: page `block * physical + slot * lp + j`, j < lp, so a
    column's pages are consecutive) and the block stride `physical`; refused otherwise (two
    block columns are not adjacent in general)."""
    assert mp > 0 and lp % mp == 0 and physical % mp == 0, (mp, lp, physical)
    assert cktab.shape[-1] % mp == 0, (cktab.shape, mp)
    return cktab[..., ::mp] // mp


def manager_page_table_direct(comp: Any, cslots: Any, mp: int, lp: int, physical: int,
                              n_ck: int) -> Any:
    """`manager_page_table(cktab, mp, lp, physical)` of the per-kernel-page table
    `cktab[..., k] = comp[..., k // lp] * physical + cslot * lp + k % lp` (k < n_ck), built at
    its final width from the block-column table `comp [layers, nreq, cols]` without the
    `[..., cols * lp]` table (VLLM_TPU_MK_IO_LEAN=cktab).

    Entry m reads page k = m * mp: column `m // q`, j = `(m % q) * mp` with q = lp // mp. Since
    mp divides `physical`, `lp` and j, `(comp * physical + cslot * lp + j) // mp` =
    `comp * (physical // mp) + cslot * q + m % q` exactly, for any int32 values for which the
    left side does not overflow (served: block ids < 2**31 / physical).
    """
    assert mp > 0 and lp % mp == 0 and physical % mp == 0 and n_ck % mp == 0, (
        mp, lp, physical, n_ck)
    q = lp // mp
    m = jnp.arange(n_ck // mp, dtype=jnp.int32)
    col = comp[..., m // q] if q > 1 else comp[..., : n_ck // mp]
    first = (jnp.asarray(cslots, jnp.int32) * q)[:, None, None] + (m % q)[None, None, :]
    return (col * (physical // mp) + first).astype(jnp.int32)


def _layer_batched_tables(pool, tables, slots, live, ctx, n_req, nreq, pages_per_seq, n_ck, lp,
                          manager_pages=0):
    """`qsa_step_inputs`' per-layer table and ring prep for all layers at once
    (VLLM_TPU_MK_WHOLE_BATCHED_INPUTS): `[layers, nreq, cols]` page tables and one ring
    read for every layer's requests. Same values as the per-layer loop; 11 ring-read calls
    and ~200 small ops per step become a few (xprof j-30c70bee: ~216 us of the head).
    """
    def stacked(i, cols):
        t = jnp.stack([_fit(tab[i].reshape(n_req, -1).astype(jnp.int32), nreq) for tab in tables])
        return jnp.pad(t, ((0, 0), (0, 0), (0, max(0, cols - t.shape[2]))))[:, :, :cols]

    mtab = stacked(1, pages_per_seq)
    cols = -(-n_ck // lp)
    comp = stacked(2, cols)
    physical = carrier_pages_per_block(pool)
    if manager_pages:  # VLLM_TPU_MK_IO_LEAN=cktab: the manager-page table directly
        cktab = manager_page_table_direct(
            comp, [cslot for _, cslot in slots], manager_pages, lp, physical, n_ck
        )
    else:
        # `compressed_kernel_pages` with each layer's slot offset.
        first = jnp.asarray([[cslot * lp + j for j in range(lp)] for _, cslot in slots],
                            jnp.int32)
        cktab = comp[..., None] * physical + first[:, None, None, :]
        cktab = cktab.reshape(len(tables), nreq, cols * lp)
        cktab = cktab[:, :, :n_ck].astype(jnp.int32)
    ids = jnp.stack(
        [_fit(tab[0].reshape(n_req, -1)[:, 0].astype(jnp.int32), nreq) for tab in tables]
    )
    ids = jnp.where(live[None], ids, 0).reshape(-1)
    ring, _ = read_ring_and_tags(pool, ids, RING_ROWS * IDX_DIM, RING_ROWS * 2)
    ring = ring.reshape(len(tables), nreq, RING_ROWS, IDX_DIM)
    ring = jnp.where((ctx > 0)[None, :, None, None], ring, 0).astype(jnp.bfloat16)
    return mtab, cktab, ring.reshape(-1, RING_ROWS, IDX_DIM)


def qsa_step_inputs(
    pool: Any,
    tables: list[tuple[Any, Any, Any]],
    slots: list[tuple[int, int]],
    trig: Any,
    positions: Any,
    starts: Any,
    context: Any,
    *,
    M: int,
    T: int,
    pages_per_seq: int,
    max_groups: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    layer_major_tables: bool = False,
    manager_pages: int = 0,
) -> dict[str, Any]:
    """`{name: array}` of the stage-C QSA extras for `len(tables)` QSA layers.

    tables[o]: layer o's flat `(ring_ids, main_table, compressed_table)` as the runner
    hands
    them to the stock op; slots[o]: its `(main_slot, compressed_slot)` (-1 main slot =
    whole
    rows: head-pair offset 0). `layer_major_tables`: page tables `[layers, requests,
    cols]`
    (the indexer's `hbm_tables` form) instead of `[layers * requests, cols]`.
    `manager_pages` = N > 0: `qsa_cktab` holds one entry per run of N consecutive kernel pages,
    `first page // N` (`QSAFullIndexedBlock.manager_pages`); N divides a block column's pages.
    """
    n_req = len(context)
    nreq = M // T
    ctx = _fit(context.astype(jnp.int32), nreq)
    q_len = _fit((starts[1:] - starts[:-1]).astype(jnp.int32), nreq)
    live = q_len > 0
    # Rows: lengths seen by verify row t of request q = context + t + 1 (0 for idle
    # requests).
    t = jnp.arange(T, dtype=jnp.int32)
    lens = jnp.where(live[:, None], ctx[:, None] + t[None] + 1, 0).reshape(M)
    n_ck = max_groups // (128 // GROUP)
    # Carrier (kernel) pages per block-table column: a pool block of `pool.shape[1]`
    # tokens
    # holds `pool.shape[1] // 128` pages of 32 groups for this layer (served 512-token
    # blocks:
    # 4). Addressed by `pooled_qsa.compressed_kernel_pages`, the stock selection's own
    # mapping.
    lp = pool.shape[1] // 128
    if envs.VLLM_TPU_MK_WHOLE_BATCHED_INPUTS:
        # VLLM_TPU_MK_IO_LEAN=cktab: the manager-page key table built at its final width.
        direct = manager_pages if io_lean("cktab", rows=M) else 0
        mtab, cktab, ring = _layer_batched_tables(
            pool, tables, slots, live, ctx, n_req, nreq, pages_per_seq, n_ck, lp,
            manager_pages=direct,
        )
        if direct:
            manager_pages = 0  # already applied
        if not layer_major_tables:
            mtab = mtab.reshape(-1, mtab.shape[-1])
            cktab = cktab.reshape(-1, cktab.shape[-1])
    else:
        mtab, cktab, rings = [], [], []
        for (ring_ids, main, comp), (_, cslot) in zip(tables, slots, strict=True):
            main = _fit(main.reshape(n_req, -1).astype(jnp.int32), nreq)
            comp = _fit(comp.reshape(n_req, -1).astype(jnp.int32), nreq)
            ids = _fit(ring_ids.reshape(n_req, -1)[:, 0].astype(jnp.int32), nreq)
            ids = jnp.where(live, ids, 0)
            # Columns past `pages_per_seq` / the compressed width only address contexts past
            # `wholecap`, which the driver routes to stock (`_whole_eligible`); padding
            # covers
            # tables narrower than the cap.
            mtab.append(
                jnp.pad(main, ((0, 0), (0, max(0, pages_per_seq - main.shape[1]))))[
                    :, :pages_per_seq
                ]
            )
            cols = -(-n_ck // lp)
            comp = jnp.pad(comp, ((0, 0), (0, max(0, cols - comp.shape[1]))))[:, :cols]
            pages = compressed_kernel_pages(comp, pool, cslot, GROUP).reshape(
                nreq, cols * lp
            )
            cktab.append(pages[:, :n_ck].astype(jnp.int32))
            ring, _ = read_ring_and_tags(pool, ids, RING_ROWS * IDX_DIM, RING_ROWS * 2)
            ring = ring.reshape(nreq, RING_ROWS, IDX_DIM)
            rings.append(jnp.where((ctx > 0)[:, None, None], ring, 0).astype(jnp.bfloat16))
        stack = jnp.stack if layer_major_tables else jnp.concatenate
        mtab, cktab, ring = stack(mtab), stack(cktab), jnp.concatenate(rings)

    if manager_pages:
        cktab = manager_page_table(cktab, manager_pages, lp, carrier_pages_per_block(pool))
    # Gather the rows, then widen: converting the whole [max_model_len, 128] table every
    # step cost
    # 64 us (xprof j-1e002dca, convert f32[262144,128]); same values.
    def trig32(rows):
        return trig[rows].astype(jnp.float32)

    pos = _fit(positions.astype(jnp.int32), M)
    # First position of the group a verify window of T = 4 completes (stock: the group's
    # first member's position, from the step or from the ring tags).
    j0 = ctx + (GROUP - 1 - ctx % GROUP) % GROUP - (GROUP - 1)
    lean_rope = io_lean("rope")
    if lean_rope:
        # VLLM_TPU_MK_IO_LEAN=rope: one gather + widen for both row sets (same rows, same
        # values) instead of two gathers and two converts.
        both = trig32(jnp.concatenate([pos, jnp.maximum(j0, 0)]))
    return {
        "qsa_pool": pool,
        "qsa_ck": pool.reshape(-1, 128 // GROUP // 2, pool.shape[-2], pool.shape[-1]),
        "qsa_cktab": cktab,
        "qsa_nvis": lens // GROUP,
        "qsa_tail": lens % GROUP,
        "qsa_mtab": mtab,
        "qsa_off": jnp.asarray([max(m, 0) for m, _ in slots], jnp.int32),
        "qsa_scales": jnp.asarray([k_scale, v_scale], jnp.float32),
        "qsa_rope": both[:M] if lean_rope else trig32(pos),
        "qsa_ctx": ctx,
        "qsa_ring": ring,
        "qsa_rope_ck": (both[M:] if lean_rope else trig32(jnp.maximum(j0, 0)))[:, None, :],
    }


def selection_groups(
    ids: Any, context: Any, starts: Any, *, T: int, K: int
) -> tuple[Any, Any]:
    """`(groups [M, K] int32, lens [M])`: each verify row's full selection of one QSA layer
    from the whole-step kernel's exported pool-path ids (`QSAFullIndexedBlock.export_ids`),
    in the stock selection's terms (`qsa_rpa.selected_tokens` blocks, -1 = none).

    `ids [M, 1, K]` (or `[M, K]`): per row the selected groups below `g_a = context // 4`,
    ascending, slots past the count 0. A row that sees at most K complete groups selects
    every one of them (the stock indexer's rule), whatever its ids. A row that sees more
    selects exactly K; the kernel moves `g_a` (the group holding the step's first tokens)
    out of the pool path when selected, so its count is then K - 1 and slot K - 1 holds
    0 (with K >= 2 the K-th ascending id of K is never 0): `g_a` goes there. `lens` are
    the rows' lengths (`qsa_step_inputs`: context + t + 1, 0 for idle requests)."""
    M = ids.shape[0]
    raw = ids.reshape(M, K).astype(jnp.int32)
    nreq = M // T
    ctx = _fit(context.astype(jnp.int32), nreq)
    q_len = _fit((starts[1:] - starts[:-1]).astype(jnp.int32), nreq)
    t = jnp.arange(T, dtype=jnp.int32)
    lens = jnp.where((q_len > 0)[:, None], ctx[:, None] + t[None] + 1, 0).reshape(M)
    nv = lens // GROUP
    ga = jnp.repeat(ctx // GROUP, T)
    j = jnp.arange(K, dtype=jnp.int32)[None]
    visible = jnp.where(j < nv[:, None], j, -1)
    sga = raw[:, K - 1 :] == 0
    over = jnp.where(j < K - sga.astype(jnp.int32), raw, jnp.where(sga, ga[:, None], -1))
    return jnp.where((nv > K)[:, None], over, visible), lens


def export_selection(
    ids: Any, context: Any, starts: Any, *, T: int, K: int, rows: int, tails: bool = False
) -> Any:
    """The draft's shared-selection sink rows `[rows, 4 K + 3]` int32 of the kernel's
    exported ids (`selection_groups`), in stock `pooled_qsa`'s `selected` layout: group j
    as tokens `4 g + 0..3` at columns `[4 j, 4 j + 4)`, -1 for none. The only reader,
    `pooled_qsa.shared_blocks`, takes the groups from columns `4 j` alone, so by default
    the partial group's tokens (`selected_tokens`' per-row scatters, ~+20 us at 32 rows)
    are left -1: one elementwise fusion. `tails`: those tokens too (the full stock row)."""
    groups, lens = selection_groups(ids, context, starts, T=T, K=K)
    groups, lens = groups[:rows], lens[:rows]
    if tails:
        from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import selected_tokens

        return selected_tokens(groups, lens, GROUP * K, GROUP)
    tok = groups[:, :, None] * GROUP + jnp.arange(GROUP, dtype=jnp.int32)
    tok = jnp.where(groups[:, :, None] >= 0, tok, -1).reshape(rows, GROUP * K)
    return jnp.pad(tok, ((0, 0), (0, GROUP - 1)), constant_values=-1)

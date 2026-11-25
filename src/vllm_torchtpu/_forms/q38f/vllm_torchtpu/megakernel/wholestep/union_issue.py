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
"""q38 L2-UAISSUE (X-UA-ISSUE): cheap, exact union-walk DMA descriptors for the whole step's union attention.

`VLLM_TPU_MK_UNION_ISSUE` (default 0 = off; read by `qwen38_wholestep_call`, registered, so it keys the compile cache)
selects the indexed block option `union_issue`, which `qsa_full_lean.lean_step_block` serves with the lean class (every
D08 S1 stage keeps its own option):

* level 1, stage (a): the walk reads physical micro-page ids, MM's `wholephys` recipe (`union_attn.physical_ids`: each
  copy issues from one SMEM id load, no table load, no clamp). Instead of `wholephys`'s per-request table-row DMA
  (start + wait on the request's critical path) and its lane compare over all `pages_per_seq` columns, every QSA layer
  copies the columns a union id can address (`slab_cols`: `ceil(max_groups / (per_page * dcp))`, 128 at D14's 262K cap
  and DCP 4, against 2048) of every request's main-table row into a VMEM slab `[nreq, 1, cols]` once (issued with the
  layer's SMEM table windows, waited before the first request), and `physical_ids` compares against that bound row.
  Under D08 S1 B3 the physical ids of request q + 1 are built into a second buffer before its double-buffered SMEM
  copy starts (MM's `wholephys` refuses B3).
* level 2, stages (a) + (b): MM D12 F0a's issue recipe on the union walk (`fast_copies` / `fast_wait`): flat
  micro-page source offsets (the pool viewed as `[pages * page_tokens, ...]`, source row `v * MICRO`), the slot's ring
  base and semaphore taken once per tile outside the 64-copy unroll, the tile's ids read at static offsets of one row
  slice, no copies for the last tile's padded slots (each copy runs only below the union count), and one wait sized
  to the bytes the tile's live copies move.

Exactness: every scored ring row receives the same 4 KiB as on MM's path (same physical page and offset); level 2
leaves the last tile's padded rows unwritten, and those rows are never scored (`tok < length` masks the scores and
zeroes the values before any arithmetic reaches the output). Outputs are bitwise MM's
(tests/megakernel/test_l2_uaissue.py).

Credit: stage (b) ports the idea of MM DSTOCK's D12 fast fetch, F0a (`VLLM_TPU_QSA_DECODE_RPA_FETCH=fast`;
q38mm-dstock-rpadc 238881491 / 92c1f3fbb, batched_rpa/bref_override.py `_full_tile` / `_partial_tile` and `wait_in`;
mm/reports/lane_dstock.md §23.4-23.8: 8.05 -> 3.94 bundles per page, -0.40...-0.47 ms per 128-row step). Stage (a) is
MM's `wholephys` (union_attn.physical_ids, qsa_full_c.py's per-request recipe) with the table row read from a slab.
Nothing of MM's trees is imported.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..blocks.qsa_attn.block import MICRO, TILE

LEVELS = (0, 1, 2)  # off, stage (a), stages (a) + (b)
GROUPS = TILE // MICRO  # copies per ring tile (`union_attn.GROUPS_PER_TILE`)


def slab_cols(max_groups: int, per_page: int, dcp: int, pps: int) -> int:
    """Main-table columns a union id `v < max_groups` can address (`v >> cshift`, a column covering `per_page * dcp`
    groups), 128-lane aligned, at most `pps`. Exact: `(max_groups - 1) >> cshift < ceil(max_groups / (per_page * dcp))`
    (both powers of two), so every live id's column sits inside the slab."""
    assert per_page & (per_page - 1) == 0 and dcp & (dcp - 1) == 0, (per_page, dcp)
    need = -(-max_groups // (per_page * dcp))
    assert need <= pps, ("the main table cannot address max_groups", max_groups, per_page, dcp, pps)
    return min(pps, -(-need // 128) * 128)


def slab_width(pps: int, dcp: int) -> int:
    """Slab columns allocated per request (`pps / dcp`, 128-lane aligned, at most `pps`). The block plans its scratch
    before it sees the pool, so this is the page-size-free bound (`pps = max_groups * 4 / 128` counts 128-token
    columns: for pool pages of >= 128 tokens `slab_cols <= slab_width`, asserted when the layer is traced)."""
    need = -(-pps // max(dcp, 1))
    return min(pps, -(-need // 128) * 128)


def slab_copy(mtab: Any, o: Any, cols: int, slab: Any, sem: Any) -> Any:
    """Layer `o`'s main-table rows of every request, columns `[0, cols)`, in one DMA into `slab [nreq, width]` VMEM
    (`mtab` `[layers, nreq, pps]` HBM). Started with the layer's SMEM table windows, waited before `prep`."""
    return pltpu.make_async_copy(mtab.at[o, :, pl.ds(0, cols)], slab.at[:, pl.ds(0, cols)], sem)


def prep(*, nreq: int, union_ids: Any, n_union: Any, slab: Any, cols: int, per_page: int, dcp: int, pv: Any) -> None:
    """Every request's union ids (`union_ids(q)` `[1, KU]` ref) -> physical micro-page ids in `pv[q]` (`[nreq, KU]`
    VMEM), against its slab row bounded to `cols` lanes; requests unrolled, so slab and id rows are static sublane
    slices (a dynamic one-row window of a 2-D tile is not provably aligned)."""
    for q in range(nreq):
        phys_ids(union_ids(q), slab[q : q + 1, 0:cols], n_union(q), per_page, dcp, pv.at[pl.ds(q, 1)])


def phys_ids(src: Any, tab: jax.Array, n_u: Any, per_page: int, dcp: int, dst: Any) -> None:
    """Union ids `src [1, KU]` (ref) -> physical micro-page ids in `dst [1, KU]` (ref) for every slot a tile reads
    (`< ceil(n_u / 64) * 64 <= ceil(n_u / 128) * 128`), against the request's slab row `tab [1, cols]`: MM's `wholephys`
    chunk loop (qsa_full_c.py), with the lane compare bounded to the slab's columns."""
    from . import union_attn as ua

    def chunk(c: Any, carry: Any) -> Any:
        lo = pl.multiple_of(c * 128, 128)
        dst[:, pl.ds(lo, 128)] = ua.physical_ids(src[:, pl.ds(lo, 128)], tab, per_page, dcp)
        return carry

    jax.lax.fori_loop(0, (n_u + 127) // 128, chunk, 0)


class When:
    """A copy that starts (and, waited per copy, waits) only when the traced `pred` holds; it keeps the descriptor
    interface (`start` / `wait`), so a caller may interleave starts as it does with plain copies."""

    def __init__(self, cp: Any, pred: Any) -> None:
        self.cp, self.pred = cp, pred

    def start(self) -> None:
        pl.when(self.pred)(self.cp.start)

    def wait(self) -> None:
        pl.when(self.pred)(self.cp.wait)


def fast_copies(*, pool: Any, ring: Any, sems: Any, ids: Any, t: Any, s: Any, n_union: Any, interpret: bool) -> list[Any]:
    """Stage (b): tile `t`'s copies into ring slot `s` from physical micro-page ids (`ids` SMEM `[1, KU + 128]`).

    F0a's recipe: one row slice of the ids (static offsets), the slot's ring base and semaphore once, flat micro-page
    source rows `v * MICRO` of the pool viewed as `[pages * page_tokens, 1, 4, 256]`, and copy `j` only while
    `t * 64 + j < n_union` (the last tile's padded slots move nothing). TPU interpret mode has no ref reshape, so there
    the same rows are addressed as `(v >> shift, (v & (per_page - 1)) * MICRO)`; the device-only flat form is covered
    by the bench's per-cell bitwise check."""
    per_page = pool.shape[1] // MICRO
    assert per_page & (per_page - 1) == 0, per_page
    shift = per_page.bit_length() - 1
    base = t * GROUPS
    row = ids.at[0, pl.ds(base, GROUPS)]
    dst, sem = ring.at[s], sems.at[s]
    live = n_union - base
    flat = None if interpret else pool.reshape(pool.shape[0] * pool.shape[1], *pool.shape[2:])
    out = []
    for j in range(GROUPS):
        v = row[j]
        if flat is None:
            src = pool.at[v >> shift, pl.ds((v & (per_page - 1)) * MICRO, MICRO)]
        else:
            src = flat.at[pl.ds(pl.multiple_of(v * MICRO, MICRO), MICRO)]
        out.append(When(pltpu.make_async_copy(src, dst.at[pl.ds(j * MICRO, MICRO)], sem), j < live))
    return out


def fast_wait(*, ring: Any, sems: Any, t: Any, s: Any, n_union: Any) -> None:
    """Stage (b): one wait on slot `s` sized to the bytes tile `t`'s live copies move (`min(n_union - 64 t, 64)` micro
    pages; MM D12's `wait_in`). Same form in TPU interpret mode, whose DMA semaphores count bytes too."""
    rows = jnp.clip(n_union - t * GROUPS, 0, GROUPS) * MICRO
    view = ring.at[s, pl.ds(0, rows)]
    pltpu.make_async_copy(view, view, sems.at[s]).wait()

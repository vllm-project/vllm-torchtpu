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
"""`QSASparseAttnBlock` row body vs the pure-JAX oracle (CPU interpret).

The block's per-row attention (`attend_row`: micro-page DMAs from the paged FP8
pool through a 2-slot ring, online softmax, stock gate rounding) runs in a
small Pallas harness. Rows cover empty, partial-group, exact-tile and multi-tile
lengths. Positive controls: the other head pair of the packed row, and a
dropped last tile, must both be caught.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.blocks.qsa_attn.block import (
    MICRO,
    TILE,
    attend_row,
    reference_rows,
)

NH, DIM, BUDGET, PAGES = 3, 256, 512, 24
SM = DIM**-0.5
SK, SV = 0.5, 0.25
# empty, 1 token, partial micro-page, one exact tile, 2 tiles + partial, long,
# full budget (9 tiles: crosses into the next row even with 8 slots), short, full
LENGTHS = [0, 1, 7, TILE, 2 * TILE + 3, 1500, 2051, 300, 2051]


def _inputs(seed: int):
    rng = np.random.default_rng(seed)
    pool = jnp.asarray(
        rng.standard_normal((PAGES, 128, 1, 4, DIM)) * 2, jnp.float32
    ).astype(jnp.float8_e4m3fn)
    n = len(LENGTHS)
    pages = rng.integers(0, PAGES * 128 // MICRO, size=(n, BUDGET + 1)).astype(np.int32)
    q = jnp.asarray(rng.standard_normal((n, NH, DIM)), jnp.bfloat16)
    gate = jnp.asarray(rng.standard_normal((n, NH, DIM)), jnp.bfloat16)
    return pool, jnp.asarray(pages), jnp.asarray(LENGTHS, jnp.int32), q, gate


def _kernel_rows(
    pool, pages, kv_len, q, gate, *, off, tile_cut=0, fold=False, slots=2,
    table=None, rows_per_req=1, table_row_base=0, recent=None,
):
    """Rows through `attend_row`; `table` (group-id pages) and `recent` (per row
    `(k [n, NREC, DIM] f32, v, ok [n, NREC] int32)`) exercise the optional hooks."""
    n = q.shape[0]
    has_t, has_r = table is not None, recent is not None

    def kernel(*refs):
        it = iter(refs)
        pages_ref, len_ref = next(it), next(it)
        tab_ref = next(it) if has_t else None
        q_ref, g_ref = next(it), next(it)
        if has_r:
            rk_ref, rv_ref, okr_ref, okc_ref = next(it), next(it), next(it), next(it)
        pool_ref, out_ref, ring, sems = next(it), next(it), next(it), next(it)

        def row(r, carry):
            g0, pre = carry
            rec = None
            if has_r:
                rec = (rk_ref[r], rv_ref[r], okr_ref[r] > 0, okc_ref[r] > 0)
            out, g0, pre = attend_row(
                r,
                g0,
                pre,
                nrows=n,
                pool=pool_ref,
                ring=ring,
                sems=sems,
                pages=pages_ref,
                kv_len=len_ref,
                q=q_ref[r],
                gate=g_ref[r],
                off=off,
                scale_k=SK,
                scale_v=SV,
                sm_scale=SM,
                interpret=True,
                fold=fold,
                table=tab_ref,
                rows_per_req=rows_per_req,
                table_row_base=table_row_base,
                recent=rec,
            )
            out_ref[r] = out
            return g0, pre

        jax.lax.fori_loop(0, n, row, (jnp.int32(0), jnp.int32(0)))

    lens = jnp.maximum(kv_len - tile_cut, jnp.minimum(kv_len, 1)) if tile_cut else kv_len
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    prefetch = [pages, lens] + ([table] if has_t else [])
    ins = [q, gate]
    if has_r:
        rk, rv, ok = recent
        ins += [rk, rv, ok[:, None, :], ok[:, :, None]]
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(prefetch),
            grid=(),
            in_specs=[vmem] * len(ins) + [pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=vmem,
            scratch_shapes=[
                pltpu.VMEM((slots, TILE, 1, 4, DIM), jnp.float8_e4m3fn),
                pltpu.SemaphoreType.DMA((slots,)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct((n, NH, DIM), jnp.float32),
        interpret=True,
    )(*prefetch, *ins, pool)


@pytest.mark.parametrize("off,fold,slots", [(0, False, 2), (1, False, 4), (1, True, 3), (0, False, 8)])
def test_attend_row_matches_reference(off, fold, slots):
    pool, pages, kv_len, q, gate = _inputs(3 + off)
    got = np.asarray(_kernel_rows(pool, pages, kv_len, q, gate, off=off, fold=fold, slots=slots))
    want = np.asarray(
        reference_rows(q, gate, pool, pages, kv_len, off=off, scale_k=SK, scale_v=SV, sm_scale=SM)
    )
    assert np.all(got[0] == 0)  # empty row
    assert float(np.abs(want).mean()) > 1e-2
    np.testing.assert_allclose(got, want, atol=2e-2, rtol=2e-2)
    assert float(np.abs(got - want).mean()) < 2e-3


def test_attend_row_catches_wrong_pair_and_dropped_tile():
    pool, pages, kv_len, q, gate = _inputs(11)
    want = np.asarray(
        reference_rows(q, gate, pool, pages, kv_len, off=0, scale_k=SK, scale_v=SV, sm_scale=SM)
    )
    wrong_pair = np.asarray(_kernel_rows(pool, pages, kv_len, q, gate, off=1))
    dropped = np.asarray(_kernel_rows(pool, pages, kv_len, q, gate, off=0, tile_cut=TILE))
    for bad in (wrong_pair, dropped):
        assert not np.allclose(bad[1:], want[1:], atol=2e-2, rtol=2e-2)


def test_table_group_ids_match_direct_micro_pages():
    """`table` + group ids give bit-identical rows to the equivalent micro-page ids."""
    pool, _, kv_len, q, gate = _inputs(21)
    n, per_page, pps, rpr, base = len(LENGTHS), 128 // MICRO, 8, 2, 1
    rng = np.random.default_rng(22)
    table = rng.integers(0, PAGES, size=(base + (n + rpr - 1) // rpr, pps)).astype(np.int32)
    groups = rng.integers(0, pps * per_page, size=(n, BUDGET + 1)).astype(np.int32)
    req = base + np.arange(n)[:, None] // rpr
    direct = table[req, groups >> 5] * per_page + (groups & (per_page - 1))
    want = np.asarray(_kernel_rows(pool, jnp.asarray(direct), kv_len, q, gate, off=1))
    got = np.asarray(
        _kernel_rows(
            pool, jnp.asarray(groups), kv_len, q, gate, off=1,
            table=jnp.asarray(table), rows_per_req=rpr, table_row_base=base,
        )
    )
    np.testing.assert_array_equal(got, want)
    assert float(np.abs(want).mean()) > 1e-2


@pytest.mark.parametrize("fold", [False, True])
def test_recent_tokens_merge_like_pool_tokens(fold):
    """VMEM `recent` tokens equal the same tokens appended to the row's pool pages."""
    off, nrec = 1, 8
    lengths = [0, 0, 4, 64, 256, 300, 512]
    nrecs = [8, 0, 4, 8, 4, 8, 4]  # row 0: recent only (attended), row 1: nothing (zero)
    n = len(lengths)
    rng = np.random.default_rng(31)
    pool = jnp.asarray(rng.standard_normal((PAGES + 1, 128, 1, 4, DIM)) * 2, jnp.float32).astype(
        jnp.float8_e4m3fn
    )
    per_page = 128 // MICRO
    q = jnp.asarray(rng.standard_normal((n, NH, DIM)), jnp.bfloat16)
    gate = jnp.asarray(rng.standard_normal((n, NH, DIM)), jnp.bfloat16)
    pages = rng.integers(0, PAGES * per_page, size=(n, BUDGET + 1)).astype(np.int32)
    ext = pages.copy()
    rk = np.zeros((n, nrec, DIM), np.float32)
    rv = np.zeros((n, nrec, DIM), np.float32)
    ok = np.zeros((n, nrec), np.int32)
    pool_np = np.asarray(pool.astype(jnp.float32))
    for r in range(n):
        # Row r's recent tokens live in page PAGES, micro-pages 2r and 2r + 1.
        for j in range(nrecs[r] // MICRO):
            mp = PAGES * per_page + 2 * r + j
            ext[r, lengths[r] // MICRO + j] = mp
            toks = pool_np[PAGES, (mp % per_page) * MICRO : (mp % per_page + 1) * MICRO, 0]
            rk[r, j * MICRO : (j + 1) * MICRO] = toks[:, 2 * off]
            rv[r, j * MICRO : (j + 1) * MICRO] = toks[:, 2 * off + 1]
        ok[r, : nrecs[r]] = 1
    lens = jnp.asarray(lengths, jnp.int32)
    want = np.asarray(
        reference_rows(
            q, gate, pool, jnp.asarray(ext), lens + jnp.asarray(nrecs, jnp.int32),
            off=off, scale_k=SK, scale_v=SV, sm_scale=SM,
        )
    )
    got = np.asarray(
        _kernel_rows(
            pool, jnp.asarray(pages), lens, q, gate, off=off, fold=fold,
            recent=(jnp.asarray(rk), jnp.asarray(rv), jnp.asarray(ok)),
        )
    )
    assert np.all(got[1] == 0)  # no pool and no recent tokens
    assert float(np.abs(got[0]).mean()) > 1e-2  # recent only: attended
    np.testing.assert_allclose(got, want, atol=2e-2, rtol=2e-2)
    assert float(np.abs(got - want).mean()) < 2e-3

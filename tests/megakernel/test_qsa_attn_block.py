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
import ml_dtypes
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.blocks.qsa_attn.block import (
    MICRO,
    TILE,
    _unpack_kv,
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
    pool,
    pages,
    kv_len,
    q,
    gate,
    *,
    off,
    tile_cut=0,
    fold=False,
    slots=2,
    table=None,
    rows_per_req=1,
    table_row_base=0,
    recent=None,
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

    lens = (
        jnp.maximum(kv_len - tile_cut, jnp.minimum(kv_len, 1)) if tile_cut else kv_len
    )
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


@pytest.mark.parametrize(
    "off,fold,slots", [(0, False, 2), (1, False, 4), (1, True, 3), (0, False, 8)]
)
def test_attend_row_matches_reference(off, fold, slots):
    pool, pages, kv_len, q, gate = _inputs(3 + off)
    got = np.asarray(
        _kernel_rows(pool, pages, kv_len, q, gate, off=off, fold=fold, slots=slots)
    )
    want = np.asarray(
        reference_rows(
            q, gate, pool, pages, kv_len, off=off, scale_k=SK, scale_v=SV, sm_scale=SM
        )
    )
    assert np.all(got[0] == 0)  # empty row
    assert float(np.abs(want).mean()) > 1e-2
    np.testing.assert_allclose(got, want, atol=2e-2, rtol=2e-2)
    assert float(np.abs(got - want).mean()) < 2e-3


def test_attend_row_catches_wrong_pair_and_dropped_tile():
    pool, pages, kv_len, q, gate = _inputs(11)
    want = np.asarray(
        reference_rows(
            q, gate, pool, pages, kv_len, off=0, scale_k=SK, scale_v=SV, sm_scale=SM
        )
    )
    wrong_pair = np.asarray(_kernel_rows(pool, pages, kv_len, q, gate, off=1))
    dropped = np.asarray(
        _kernel_rows(pool, pages, kv_len, q, gate, off=0, tile_cut=TILE)
    )
    for bad in (wrong_pair, dropped):
        assert not np.allclose(bad[1:], want[1:], atol=2e-2, rtol=2e-2)


def test_table_group_ids_match_direct_micro_pages():
    """`table` + group ids give bit-identical rows to the equivalent micro-page ids."""
    pool, _, kv_len, q, gate = _inputs(21)
    n, per_page, pps, rpr, base = len(LENGTHS), 128 // MICRO, 8, 2, 1
    rng = np.random.default_rng(22)
    table = rng.integers(0, PAGES, size=(base + (n + rpr - 1) // rpr, pps)).astype(
        np.int32
    )
    groups = rng.integers(0, pps * per_page, size=(n, BUDGET + 1)).astype(np.int32)
    req = base + np.arange(n)[:, None] // rpr
    direct = table[req, groups >> 5] * per_page + (groups & (per_page - 1))
    want = np.asarray(_kernel_rows(pool, jnp.asarray(direct), kv_len, q, gate, off=1))
    got = np.asarray(
        _kernel_rows(
            pool,
            jnp.asarray(groups),
            kv_len,
            q,
            gate,
            off=1,
            table=jnp.asarray(table),
            rows_per_req=rpr,
            table_row_base=base,
        )
    )
    np.testing.assert_array_equal(got, want)
    assert float(np.abs(want).mean()) > 1e-2


@pytest.mark.parametrize("fold", [False, True])
def test_recent_tokens_merge_like_pool_tokens(fold):
    """VMEM `recent` tokens equal the same tokens appended to the row's pool pages."""
    off, nrec = 1, 8
    lengths = [0, 0, 4, 64, 256, 300, 512]
    nrecs = [
        8,
        0,
        4,
        8,
        4,
        8,
        4,
    ]  # row 0: recent only (attended), row 1: nothing (zero)
    n = len(lengths)
    rng = np.random.default_rng(31)
    pool = jnp.asarray(
        rng.standard_normal((PAGES + 1, 128, 1, 4, DIM)) * 2, jnp.float32
    ).astype(jnp.float8_e4m3fn)
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
            toks = pool_np[
                PAGES, (mp % per_page) * MICRO : (mp % per_page + 1) * MICRO, 0
            ]
            rk[r, j * MICRO : (j + 1) * MICRO] = toks[:, 2 * off]
            rv[r, j * MICRO : (j + 1) * MICRO] = toks[:, 2 * off + 1]
        ok[r, : nrecs[r]] = 1
    lens = jnp.asarray(lengths, jnp.int32)
    want = np.asarray(
        reference_rows(
            q,
            gate,
            pool,
            jnp.asarray(ext),
            lens + jnp.asarray(nrecs, jnp.int32),
            off=off,
            scale_k=SK,
            scale_v=SV,
            sm_scale=SM,
        )
    )
    got = np.asarray(
        _kernel_rows(
            pool,
            jnp.asarray(pages),
            lens,
            q,
            gate,
            off=off,
            fold=fold,
            recent=(jnp.asarray(rk), jnp.asarray(rv), jnp.asarray(ok)),
        )
    )
    assert np.all(got[1] == 0)  # no pool and no recent tokens
    assert float(np.abs(got[0]).mean()) > 1e-2  # recent only: attended
    np.testing.assert_allclose(got, want, atol=2e-2, rtol=2e-2)
    assert float(np.abs(got - want).mean()) < 2e-3


# ----------------------------------------- VLLM_TPU_MK_FP8_DENSE_READ (`_unpack_kv`)
def _slot_with_all_pairs(off: int, rows: int = TILE) -> np.ndarray:
    """Raw `[rows, 1, 4, DIM]` bytes: pair `off`'s K / V take every byte pair."""
    raw = np.random.default_rng(31 + off).integers(0, 256, (rows, 1, 4, DIM), np.uint8)
    pairs = np.arange(rows * DIM) % 65536
    raw[:, 0, 2 * off] = (pairs & 0xFF).reshape(rows, DIM)
    raw[:, 0, 2 * off + 1] = (pairs >> 8).reshape(rows, DIM)
    return raw


def _unpacked(
    raw: np.ndarray, off: int, unpack, interpret: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    def kernel(off_ref, slot_ref, k_ref, v_ref):
        k8, v8 = unpack(slot_ref, off_ref[0])
        k_ref[...] = k8.astype(jnp.float32)
        v_ref[...] = v8.astype(jnp.float32)

    out = jax.ShapeDtypeStruct((raw.shape[0], DIM), jnp.float32)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    k, v = pl.pallas_call(
        kernel,
        out_shape=(out, out),
        in_specs=[smem, vmem],
        out_specs=(vmem, vmem),
        interpret=interpret,
    )(jnp.asarray([off], jnp.int32), jnp.asarray(raw).view(jnp.float8_e4m3fn))
    return np.asarray(k), np.asarray(v)


@pytest.mark.parametrize("off", [0, 1])
def test_unpack_kv_dense_bit_identical_over_all_byte_pairs(off):
    """Every (K, V) FP8 byte pair, NaN included: the dense split is bit-identical to the
    bytewise unpack and equal to the FP8 values (a traced `off`, as the kernels pass it)."""
    raw = _slot_with_all_pairs(off)
    got = _unpacked(raw, off, lambda ref, o: _unpack_kv(ref, o, True, dense=True))
    old = _unpacked(raw, off, lambda ref, o: _unpack_kv(ref, o, True, dense=False))
    for c, g, o in zip((2 * off, 2 * off + 1), got, old):
        np.testing.assert_array_equal(g.view(np.uint32), o.view(np.uint32))
        want = raw[:, 0, c].view(ml_dtypes.float8_e4m3fn).astype(np.float32)
        np.testing.assert_array_equal(g, want)  # NaN == NaN here


def test_unpack_kv_dense_flag(monkeypatch):
    """`dense=None` follows VLLM_TPU_MK_FP8_DENSE_READ (default off)."""
    from vllm_torchtpu.megakernel.blocks.qsa_attn import block

    monkeypatch.delenv("VLLM_TPU_MK_FP8_DENSE_READ", raising=False)
    assert not block.fp8_dense_read()
    monkeypatch.setenv("VLLM_TPU_MK_FP8_DENSE_READ", "1")
    assert block.fp8_dense_read()
    raw = _slot_with_all_pairs(1)
    got = _unpacked(raw, 1, lambda ref, o: _unpack_kv(ref, o, True))
    old = _unpacked(raw, 1, lambda ref, o: _unpack_kv(ref, o, True, dense=False))
    for g, o in zip(got, old):
        np.testing.assert_array_equal(g.view(np.uint32), o.view(np.uint32))


@pytest.mark.parametrize("off,rows", [(0, TILE), (1, TILE), (1, MICRO)])
def test_unpack_kv_dense_tpu_path_matches(off, rows):
    """On device, the strided dense-row read gives the stock read's bits."""
    if jax.devices()[0].platform != "tpu":
        pytest.skip("needs a TPU")
    raw = _slot_with_all_pairs(off, rows)
    got = _unpacked(raw, off, lambda ref, o: _unpack_kv(ref, o, False, dense=True), False)
    old = _unpacked(raw, off, lambda ref, o: _unpack_kv(ref, o, False, dense=False), False)
    want = _unpacked(raw, off, lambda ref, o: _unpack_kv(ref, o, True, dense=False))
    for g, o, w in zip(got, old, want):
        np.testing.assert_array_equal(g.view(np.uint32), o.view(np.uint32))
        np.testing.assert_array_equal(g.view(np.uint32), w.view(np.uint32))


@pytest.mark.parametrize("W", [128, 256])
def test_fp8_carrier_dense_key_read_tpu(W):
    """On device, the key stream's dense read (`QSAFullIndexedBlock._consume` under
    VLLM_TPU_MK_FP8_DENSE_READ) gives the stock ref-bitcast read's BF16 keys bit for bit."""
    if jax.devices()[0].platform != "tpu":
        pytest.skip("needs a TPU")
    CH = 4
    CG = CH * 32
    raw = np.random.default_rng(7).integers(0, 256, (CH, 16, 4, W), np.uint8)

    def kernel(slot_ref, old_ref, new_ref):
        old = slot_ref.bitcast(jnp.bfloat16)[...].reshape(CG, W)[:, :128]
        wrows = slot_ref.bitcast(jnp.int32).reshape(CG // 2 * (W // 128), 128)
        words = wrows[pl.ds(0, CG // 2, stride=W // 128)] if W > 128 else wrows[...]
        old_ref[...] = old
        new_ref[...] = pltpu.bitcast(words, jnp.bfloat16)

    out = jax.ShapeDtypeStruct((CG, 128), jnp.bfloat16)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    old, new = pl.pallas_call(
        kernel, out_shape=(out, out), in_specs=[vmem], out_specs=(vmem, vmem)
    )(jnp.asarray(raw).view(jnp.float8_e4m3fn))
    old, new = np.asarray(old).view(np.uint16), np.asarray(new).view(np.uint16)
    np.testing.assert_array_equal(old, new)
    assert len(np.unique(old)) > 1000  # random keys: a wrong row order would show

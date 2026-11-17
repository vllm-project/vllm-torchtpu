# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""glm52_decode_mk (stage A), CPU interpret: the barrier-free cache writer against a numpy model of its ownership
and placement rules, and the interpret-mode shared-storage views (interp_alias) the oracle runs the alias layout on.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.glm52_decode_mk import config as C
from vllm_torchtpu.kernels.glm52_decode_mk import interp_alias, pool_alias
from vllm_torchtpu.kernels.glm52_decode_mk import layout as LY
from vllm_torchtpu.kernels.glm52_decode_mk.writer import build_writer


def test_interp_alias_views_share_storage():
    """Views of one u8 pool alias in interpret mode exactly as their byte windows say: an overlapping later write
    shows through, an adjacent one does not, a DMA lands in the view's own bytes."""
    interp_alias.install()

    def k(h_ref, o_ov, o_adj, o_dma, pool, sem):
        n = (
            pool_alias.storage_size((2, 8, 128), jnp.dtype(jnp.float32)) // 128
        )  # 64 rows
        x = pool_alias.view(pool.at[pl.ds(0, n)], (2, 8, 128), jnp.float32)
        y = pool_alias.view(
            pool.at[pl.ds(32, n)], (2, 8, 128), jnp.float32
        )  # overlaps x's 2nd half
        z = pool_alias.view(
            pool.at[pl.ds(n, n)], (2, 8, 128), jnp.float32
        )  # adjacent to x
        x[...] = jnp.full((2, 8, 128), 1.0, jnp.float32)
        z[...] = jnp.full((2, 8, 128), 3.0, jnp.float32)
        o_adj[...] = x[...]
        y[...] = jnp.full((2, 8, 128), 2.0, jnp.float32)
        o_ov[...] = x[...]
        u = pool_alias.view(pool.at[pl.ds(128, 32)], (32, 128), jnp.uint8)
        c = pltpu.make_async_copy(h_ref, u, sem.at[0])
        c.start()
        c.wait()
        o_dma[...] = pool[128:160, :].astype(jnp.int32)

    f = pl.pallas_call(
        k,
        out_shape=[
            jax.ShapeDtypeStruct((2, 8, 128), jnp.float32),
            jax.ShapeDtypeStruct((2, 8, 128), jnp.float32),
            jax.ShapeDtypeStruct((32, 128), jnp.int32),
        ],
        in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
        scratch_shapes=[
            pltpu.VMEM((256, 128), jnp.uint8),
            pltpu.SemaphoreType.DMA((1,)),
        ],
        interpret=pltpu.InterpretParams(),
    )
    h = (np.arange(32 * 128) % 251).astype(np.uint8).reshape(32, 128)
    ov, adj, dma = (np.asarray(a) for a in f(jnp.asarray(h)))
    assert (adj == 1.0).all()
    assert (ov.reshape(-1)[: ov.size // 2] == 1.0).all() and (
        ov.reshape(-1)[ov.size // 2 :] == 2.0
    ).all()
    assert (dma == h.astype(np.int32)).all()


def _writer_case(seed, R=4, TMAX=16, L=4, nb=8, max_pages=4):
    cfg = dataclasses.replace(C.FULL, R=R, L=L, TMAX=TMAX)
    rng = np.random.default_rng(seed)
    pg, G_ = cfg.page, cfg.ILV * R
    nbi = nb + 1
    T = 0
    rmeta = np.zeros((TMAX, 8), np.int32)
    req_starts = []
    b = 0
    while (
        T < TMAX - 4 and b < 4
    ):  # requests of 2..4 consecutive positions, some straddling a 4-block
        n = int(rng.integers(2, 5))
        p0 = int(rng.integers(0, max_pages * pg * R - 8))
        if b == 0:
            p0 = 4 * G_ - 2  # straddles an ILV block and a rank boundary
        for i in range(n):
            rmeta[T] = (1, p0 + i, b, p0 + n, 0, 0, 0, 0)
            T += 1
        req_starts.append(p0)
        b += 1
    bt = np.zeros((cfg.BMAX, max_pages), np.int32)
    bti = np.zeros((cfg.BMAX, max_pages), np.int32)
    for r_ in range(b):
        bt[r_] = rng.choice(nb, size=max_pages, replace=False)
        bti[r_] = rng.choice(nbi, size=max_pages, replace=False)
    nrow = rng.integers(0, 2**32, size=(L, TMAX, 128), dtype=np.uint32)
    rrow = rng.integers(0, 2**32, size=(L, TMAX, 128), dtype=np.uint32)
    irec = rng.integers(0, 256, size=(L, TMAX, 256)).astype(np.int32)
    ck = LY.cache_kinds(cfg, nb, nbi)
    caches0 = [
        rng.integers(
            0,
            2**32 if ck[k][2] == jnp.uint32 else 256,
            size=(len(ck[k][0]),) + ck[k][1],
            dtype=np.uint32 if ck[k][2] == jnp.uint32 else np.uint8,
        )
        for k in ("nope", "rope", "idx")
    ]
    return cfg, T, rmeta, bt, bti, nrow, rrow, irec, caches0, nb, nbi, max_pages


def _writer_numpy(cfg, T, rmeta, bt, bti, nrow, rrow, irec, caches0, me, max_pages):
    pg, ILV, G_ = cfg.page, cfg.ILV, cfg.ILV * cfg.R
    nope, rope, idx = (c.copy() for c in caches0)
    full = [l for l in range(cfg.L) if cfg.full_indexer(l)]
    IR4 = cfg.IREC // 4
    for t in range(T):
        valid, pos, req = rmeta[t, 0], rmeta[t, 1], rmeta[t, 2]
        if not valid or (pos % G_) // ILV != me:
            continue
        lo = (pos // G_) * ILV + pos % ILV
        page, sl = bt[req, min(lo // pg, max_pages - 1)], lo % pg
        pagei = bti[req, min(lo // pg, max_pages - 1)]
        q = pos & 3
        for l in range(cfg.L):
            nope[l, page, sl] = nrow[l, t]
            rope[l, page, sl >> 2, 32 * q : 32 * q + 32] = rrow[
                l, t, 32 * q : 32 * q + 32
            ]
            if l in full:
                for w in range(IR4):
                    for qq in range(4):
                        idx[full.index(l), pagei, w, qq, sl] = np.uint8(
                            irec[l, t, 4 * w + qq] & 0xFF
                        )
    return nope, rope, idx


@pytest.mark.parametrize("seed", [0, 1])
def test_writer_matches_numpy_model(seed):
    """Every rank writes exactly its owned rows (nope row, merged rope quarter, idx record columns) and nothing
    else: the hoisted per-step ownership flags (one rope write per 4-token row) against a direct numpy model."""
    cfg, T, rmeta, bt, bti, nrow, rrow, irec, caches0, nb, nbi, max_pages = (
        _writer_case(seed)
    )
    writer, _ = build_writer(cfg, max_pages=max_pages, nb=nb, nbi=nbi, interpret=True)
    for me in range(cfg.R):
        scal = np.zeros(16, np.int32)
        scal[0], scal[2] = T, me
        got = writer(
            jnp.asarray(scal),
            jnp.asarray(rmeta),
            jnp.asarray(bt),
            jnp.asarray(bti),
            jnp.asarray(nrow),
            jnp.asarray(rrow),
            jnp.asarray(irec),
            *[jnp.asarray(c) for c in caches0],
        )
        want = _writer_numpy(
            cfg, T, rmeta, bt, bti, nrow, rrow, irec, caches0, me, max_pages
        )
        for kind, g, w in zip(("nope", "rope", "idx"), got, want):
            np.testing.assert_array_equal(np.asarray(g), w, err_msg=f"rank {me} {kind}")

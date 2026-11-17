# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""glm52_decode_mk (stage A): admission, the static operand / stream / alias plans, and (opt-in) the teacher-forced
per-layer oracle of the campaign's instruments on the kernel in TPU interpret mode.

The oracle check needs the instruments of campaign glm52-decode-mk (glm_ref.py, check.py, inputs.py, spec.py,
tolerances.json), which live outside this repository: set GLM52_MK_INSTRUMENTS=<dir> to run it (a subprocess with
16 virtual CPU devices; minutes per cell). Mosaic lowering, VMEM and bundles are checked by the local AOT script of
the campaign (stageA/scripts/aot.py), not here.
"""

import dataclasses
import math
import os
import subprocess
import sys
import textwrap

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.glm52_decode_mk import config as C
from vllm_torchtpu.kernels.glm52_decode_mk import dsa as D
from vllm_torchtpu.kernels.glm52_decode_mk import kernel as K
from vllm_torchtpu.kernels.glm52_decode_mk import layout as LY
from vllm_torchtpu.kernels.glm52_decode_mk import pool_alias


def test_admission():
    C.FULL.check()
    C.FULL_B16.check()
    for bad in (
        dict(ILV=2),
        dict(page=128),
        dict(QMAX=5),
        dict(QMAX=2),
        dict(TMAX=24),
        dict(R=3),
    ):
        with pytest.raises(AssertionError):
            dataclasses.replace(C.FULL, **bad).check()


def test_full_layer_kinds():
    cfg = C.FULL
    fresh = [l for l in range(cfg.L) if cfg.full_indexer(l)]
    assert fresh[:5] == [0, 1, 2, 6, 10] and len(fresh) == 21
    assert sum(cfg.is_moe(l) for l in range(cfg.L)) == 75


def test_stream_tables_decode():
    """dtab decodes back to each layer's static tile order; ltab g0 points at each layer's first tile."""
    cfg = C.FULL
    plan = LY.tile_plan(cfg, LY.bank_bytes(cfg, 3 << 20))
    ltab, dtab = LY.stream_tables(cfg, plan)
    for l in range(cfg.L):
        seq = LY.layer_tiles(cfg, l, plan)
        g0 = int(ltab[l, 0])
        for i, (k, t) in enumerate(seq):
            d = int(dtab[g0 + i])
            assert LY.DKINDS[d >> 20] == k and (d & 255) == t
            assert ((d >> 8) & 4095) == LY.layers_of(cfg, k).index(l)
        assert bool(ltab[l, 1]) == cfg.full_indexer(l) and bool(
            ltab[l, 2]
        ) == cfg.is_moe(l)
    assert len(dtab) == int(ltab[-1, 0]) + len(LY.layer_tiles(cfg, cfg.L - 1, plan))
    # rule 5: padded minor dims (no ENTRY layout copy of a 2624- / 160-wide operand)
    assert (
        LY.dense_shapes(cfg)["qkva"][0][1] == 2688
        and LY.dense_shapes(cfg)["iwk"][0][1] == 256
    )


@pytest.mark.parametrize("tmax", [16, 32])
def test_epool_alias_plan(tmax):
    """WHITEBOARD.v1 §7.1: the expert ring is one u8 pool of 6 x 2.25 MiB slots; every expert-slot and aliased DSA
    view has pool_alias storage == raw bytes at 4096-B aligned, non-overlapping offsets, and the DSA views fit in it."""
    cfg = dataclasses.replace(C.FULL, TMAX=tmax)
    tiles = K.Tiles()
    rows, views, ring_b, dsa_b = K.epool_layout(cfg, tiles, 256)
    assert ring_b == 6 * 2_359_296 == rows * 128
    assert dsa_b <= ring_b
    for cls in ("slot", "dsa"):
        spans = []
        for nm, (r0, shp, dt) in views.items():
            if (cls == "slot") != (nm.startswith("w13_") or nm.startswith("w2_")):
                continue
            sz = pool_alias.storage_size(shp, jnp.dtype(dt))
            assert sz == math.prod(shp) * jnp.dtype(dt).itemsize, nm
            assert (r0 * 128) % 4096 == 0, nm
            spans.append((r0 * 128, r0 * 128 + sz))
        spans.sort()
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), cls
    assert set(D.ALIASED) <= set(views)


@pytest.mark.parametrize("tmax", [16, 32])
@pytest.mark.parametrize(
    "fault", ["", "f_alias_poison", "f_stale_is_alias", "f_alias_poison_red"]
)
def test_uv_alias_fault_rows_are_tile_aligned(tmax, fault):
    """F-STALE-IS-ALIAS / F-ALIAS-POISON-RED place own_uv in the alias window as an (8, 128)-tiled pool_alias view:
    own_uv and own_u get ceil8(UR2) rows there (so udma's 8-row-aligned granules stay in bounds: libtpu 0.0.47
    rejects unaligned DMA windows into the view). Every other build keeps UR2 rows and own_uv as native scratch."""
    cfg = dataclasses.replace(C.FULL, TMAX=tmax)
    tiles = K.Tiles(fault=fault)
    g = D.geometry(cfg, tiles, 256, cfg.BMAX)
    scr = D.scratch(cfg, tiles, 256, cfg.BMAX)
    in_alias = fault in D.UV_ALIAS_FAULTS
    rows = -(-g["UR2"] // 8) * 8 if in_alias else g["UR2"]
    assert scr["own_u"].shape == scr["own_uv"].shape == (rows, 128)
    assert ("own_uv" in K.aliased_names(tiles)) == in_alias
    _, views, _, _ = K.epool_layout(cfg, tiles, 256)
    if in_alias:
        r0, shp, _ = views["own_uv"]
        assert shp == (rows, 128) and (r0 * 128) % 4096 == 0
        # udma's granules: ceil8(U) <= rows for every U the extraction can produce (U <= UR2 - XKC / 128)
        assert -(-(g["UR2"] - g["XKC"] // 128) // 8) * 8 <= rows
    else:
        assert "own_uv" not in views


def test_wrapper_metadata_roundtrip():
    """Request / row metadata and the two block tables (each cache's own page modulus)."""
    import jax

    from vllm_torchtpu.kernels.glm52_decode_mk import wrapper as W

    cfg = dataclasses.replace(C.FULL, TMAX=16, R=4, NH=16)
    qsl = jnp.asarray([0, 4, 8, 11, 11, 11], jnp.int32)
    seq = jnp.asarray([100, 7, 300, 5, 9], jnp.int32)
    pos = jnp.asarray([96, 97, 98, 99, 3, 4, 5, 6, 297, 298, 299] + [0] * 5, jnp.int32)
    bt = jnp.arange(5 * 3, dtype=jnp.int32).reshape(5, 3) + 40
    with jax.default_device(jax.devices("cpu")[0]):
        scal, rmeta, qmeta, btl, bti, rvec = W.metadata(
            cfg, 16, pos, seq, qsl, bt, 13, 14, 3, 2
        )
    assert int(scal[0]) == 11 and int(scal[1]) == 3 and int(scal[2]) == 2
    assert np.array_equal(
        np.asarray(qmeta[:4]),
        [[0, 4, 100, 1], [4, 4, 7, 1], [8, 3, 300, 1], [11, 0, 5, 0]],
    )
    assert np.array_equal(
        np.asarray(rmeta[8:12, :4]),
        [[1, 297, 2, 300], [1, 298, 2, 300], [1, 299, 2, 300], [0, 0, 0, 0]],
    )
    assert np.array_equal(np.asarray(btl[:2]), np.asarray(bt[:2]) % 13)
    assert np.array_equal(np.asarray(bti[:2]), np.asarray(bt[:2]) % 14)


ORACLE = textwrap.dedent("""
    import os, sys, dataclasses
    sys.path.insert(0, os.environ["GLM52_MK_INSTRUMENTS"])
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16 --xla_allow_excess_precision=false"
    import inputs as IN, spec as S, check as C
    from vllm_torchtpu.kernels.glm52_decode_mk import kernel as K, wrapper as W
    R = int(sys.argv[1])
    dims = dataclasses.replace(S.TINY, name="TINY2L", n_layers=2)
    si = IN.make_step_inputs(dims, R, S.cell_by_name(dims, R, "decode_first_b16"))
    si.meta = dataclasses.replace(si.meta, mixed_bf16_reduce=False)
    fn = W.candidate_factory(tiles=K.Tiles(dot="mixed", recip="exact"))
    out, caches, aux = C.run(si, "native", step_fn=fn)
    v = C.check_results(si, out, caches, aux, C.tol_for(C.load_tolerances(), "TINY"))
    print("VERDICT", v.passed, v.reasons[:4])
""")


@pytest.mark.skipif(
    not os.environ.get("GLM52_MK_INSTRUMENTS"),
    reason="set GLM52_MK_INSTRUMENTS=<instruments dir>",
)
@pytest.mark.parametrize("R", [2])
def test_instruments_oracle_tiny(R):
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    r = subprocess.run(
        [sys.executable, "-c", ORACLE, str(R)],
        capture_output=True,
        text=True,
        env=env,
        timeout=3600,
    )
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("VERDICT")]
    assert line and line[0].startswith("VERDICT True"), (
        r.stdout[-2000:],
        r.stderr[-4000:],
    )

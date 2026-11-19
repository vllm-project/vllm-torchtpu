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
"""q38 L2-UAPIPE: X-UA-PIPE (`union_attn.attend_request_pipe` + `pipe_request_loop`) bitwise against MM's
`attend_request` driven the way D08 S1 B3 drives it (double-buffered union-id rows), in CPU interpret mode.

One kernel runs `L = 2` layers back to back on one ring and one set of ring semaphores (as the whole step does), each
layer a request loop over `nreq` requests (b16 / b32 / b48 / b64 at T = 4 = 4 / 8 / 12 / 16 requests) at the D14
geometry (DCP 4: 12 heads per row, rank-owned groups of 512-token pool blocks, a 3-slot ring = 2 tiles in flight), or at
DCP 1. Union sizes: real D14 sizes (5-6 tiles per request-layer: 257-384 groups), and 0-, 1- and 2-tile requests at the
start, middle and end of the loop, so the clamped starts, the cross-request prefetch across short and empty requests
and the last request (nothing prefetched past it) all run. Every request's table points at its own pool pages, so a tile
fetched with another request's ids or table reads other data; a DMA left in flight by one layer is consumed by the
next layer's wait on that slot, so it corrupts the second layer's outputs.

Every mode's outputs (both layers, every row: the DCP partial and its LSE, or the gated row at DCP 1) must equal the
base bit for bit, and the base must match a float64 NumPy reference within f32 tolerance (the harness computes what it
claims, and is not vacuous). Lines `L2-UAPIPE COVER` / `L2-UAPIPE IDENTITY` are printed for the job log.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.wholestep import union_attn as ua

T, NHR, HD, TILE, MICRO, BT = 4, 3, 256, 256, 4, 512
PER_PAGE = BT // MICRO
KU = 512
NSLOTS = 3
L = 2
F8 = jnp.float8_e4m3fn
MODES = [m for m in ua.PIPE_MODES]
INTERP = os.environ.get("Q38_UAPIPE_RACES", "0") == "1"

# Union sizes in groups (64 groups = one tile) per layer per request. Real D14 unions are 5.2-5.9 tiles per
# request-layer at DCP 4 (333-378 groups); 0 = an empty request (a padded slot), 1-64 = one tile, 65-128 = two.
SCEN = {
    # b16 (4 requests): real sizes, then a 1-tile request in the middle; layer 1 starts with an empty request.
    "b16_d4": dict(nreq=4, dcp=4, n=[[350, 371, 64, 333], [0, 384, 340, 1]]),
    # b32 (8 requests): real sizes with a 0-tile and a 1-tile request mid-loop, last request 2 tiles; layer 1 ends empty.
    "b32_d4": dict(nreq=8, dcp=4, n=[[345, 0, 378, 1, 362, 333, 351, 100], [357, 365, 340, 383, 64, 65, 344, 0]]),
    # b48 (12 requests): consecutive short / long alternation, two empty requests in a row, 1-tile last request.
    "b48_d4": dict(nreq=12, dcp=4, n=[[360, 1, 370, 0, 0, 341, 128, 129, 376, 352, 335, 3],
                                      [64, 384, 33, 349, 368, 0, 371, 338, 2, 359, 343, 361]]),
    # b64 (16 requests): real sizes throughout with 0 / 1 / 2-tile requests at both ends of each layer.
    "b64_d4": dict(nreq=16, dcp=4, n=[[0, 1, 339, 347, 372, 355, 366, 334, 380, 341, 0, 358, 363, 349, 337, 70],
                                      [128, 336, 361, 344, 377, 1, 1, 352, 359, 340, 369, 333, 348, 374, 357, 0]]),
    # DCP 1 (gated rows, `has` flags), 8 requests, a mix including the long tail of a 6-tile union.
    "b32_d1": dict(nreq=8, dcp=1, n=[[384, 2, 0, 300, 64, 350, 1, 129], [0, 0, 371, 340, 365, 1, 384, 63]]),
}


def _tiles(n: int) -> int:
    return -(-n * MICRO // TILE)


def _inputs(scen: str, seed: int = 7) -> dict:
    """Random inputs of scenario `scen` (NumPy), per layer and request: owned group ids (DCP rank 1 of 4 at DCP 4),
    codes, counts, page tables onto disjoint pages, queries, gates, recent tiles and their masks."""
    sp = SCEN[scen]
    nreq, dcp = sp["nreq"], sp["dcp"]
    rk_ = 1 if dcp == 4 else 0
    NH = NHR * dcp
    rng = np.random.default_rng(seed + 1000 * nreq + dcp)
    ncol = 4  # table columns per request (each covers dcp * 128 groups, 128 of them this rank's)
    P = 1 + nreq * ncol
    # Page 0 is never mapped. Pool rows: finite FP8 everywhere (K and V bytes of both head pairs).
    pool = (rng.standard_normal((P, BT, 1, 4, HD)) * 0.6).astype(np.float32)
    pool[0] = np.nan
    uids = np.zeros((L, nreq, 1, KU), np.int32)
    ucodes = np.zeros((L, nreq, 1, KU), np.int32)
    un = np.zeros((L, nreq), np.int32)
    tab = np.zeros((L, nreq, 8), np.int32)
    for lyr in range(L):
        perm = rng.permutation(np.arange(1, P))  # each layer maps every request onto its own pages
        for q in range(nreq):
            n = sp["n"][lyr][q]
            tab[lyr, q, :ncol] = perm[q * ncol : (q + 1) * ncol]
            cand = np.array([c * dcp * PER_PAGE + rk_ * PER_PAGE + i for c in range(ncol) for i in range(PER_PAGE)])
            g = np.sort(rng.choice(cand, n, replace=False)) if n else np.zeros(0, np.int64)
            uids[lyr, q, 0, :n] = g
            code = rng.integers(1, 16, n)
            if q % 3 == 1 and n:  # one row with no pool token in some requests
                code = np.where(code == 8, 1, code) & 7
                code = np.where(code == 0, 3, code)
            ucodes[lyr, q, 0, :n] = code
            un[lyr, q] = n
    M = nreq * T
    qv = (rng.standard_normal((M, NH, HD)) * 0.5).astype(np.float32)
    gv = rng.standard_normal((M, NH, HD)).astype(np.float32)
    kvl = rng.integers(0, 3, M).astype(np.int32)
    rk = (rng.standard_normal((nreq, 8, HD)) * 0.6).astype(np.float32)
    rv = (rng.standard_normal((nreq, 8, HD)) * 0.6).astype(np.float32)
    okr = (rng.random((M, 1, 8)) < 0.4).astype(np.int32)
    okr[1::7] = 0  # some rows with no recent token
    oka = np.zeros((nreq, 8, 1), np.int32)
    for q in range(nreq):
        oka[q, :, 0] = okr[q * T : (q + 1) * T, 0, :].max(axis=0)
    return dict(scen=scen, nreq=nreq, dcp=dcp, NH=NH, pool=pool.astype(F8), uids=uids, ucodes=ucodes, un=un, tab=tab,
                qv=qv, gv=gv, kvl=kvl, rk=rk, rv=rv, okr=okr, oka=oka, off=np.array([1], np.int32),
                scales=np.array([0.75, 1.25], np.float32))


def _run(d: dict, mode: int, interpret=None):
    """One kernel: `L` layers of `nreq` requests. mode 0 = base (`attend_request`, B3's double-buffered ids);
    else `attend_request_pipe` with `pipe=mode` (B3's ids without PIPE_XREQ, `pipe_request_loop` with it)."""
    nreq, dcp, NH = d["nreq"], d["dcp"], d["NH"]
    M = nreq * T
    xreq = bool(mode & ua.PIPE_XREQ)

    def kernel(off_s, sc_s, un_s, tab_s, kvl_s, pool, uids, ucodes, qv, gv, rk, rv, okr, oka, qiz,
               out, lse, ring, sems, rowb, bsem, rows3, xsem):
        for lyr in range(L):

            def body(q, ctl, carry, lyr=lyr, ids=None, row=0):
                rs = [q * T + t for t in range(T)]
                common = dict(
                    T=T, nh=NH, pool=pool, ring=ring, sems=sems, codes=ucodes.at[lyr, q], n_union=un_s[lyr, q],
                    table=tab_s.at[lyr], table_row=q, qs=[qv[r] for r in rs], gates=[gv[r] for r in rs],
                    has=[kvl_s[r] > 0 for r in rs], off=off_s[0], scale_k=sc_s[0], scale_v=sc_s[1],
                    sm_scale=HD**-0.5, interpret=interpret or True,
                    recent=(rk[q], rv[q], [okr[r] != 0 for r in rs], oka[q] != 0), dcp=dcp,
                )
                if mode == 0:
                    res, nt = ua.attend_request(ids=ids, phys=False, **common), 0
                elif xreq:
                    q1 = jnp.minimum(q + 1, nreq - 1)
                    res, nt = ua.attend_request_pipe(
                        ids=rows3, ids_row=ctl["ids_row"], pipe=mode, base=ctl["base"], inflight=ctl["inflight"],
                        nxt=ua.PipeNext(ids_row=ctl["next_row"], n_union=un_s[lyr, q1], table_row=q1,
                                        has=ctl["has_next"]),
                        **common,
                    )
                else:
                    res, nt = ua.attend_request_pipe(ids=ids, ids_row=row, pipe=mode, **common)
                for t, r in enumerate(rs):
                    if dcp > 1:
                        att, l_ = res[t]
                        out[lyr, r] = att
                        lse[lyr, r] = jnp.broadcast_to(l_, (NH, 128))
                    else:
                        out[lyr, r] = res[t]
                return carry, nt

            if xreq:

                def ids_copies(q, k, lyr=lyr):  # the block's helper (`_PipeMixin._attend_body`)
                    return ua.pipe_ids_copies(uids.at[lyr, q], qiz, rows3, xsem, k)

                ua.pipe_request_loop(nreq=nreq, nslots=NSLOTS, pipe=mode, ids_copies=ids_copies, body=body,
                                     init=jnp.int32(0))
            else:  # D08 S1 B3's union-id double buffer (`_LeanMixin._attend_body`)

                def ucopies(q, b, lyr=lyr):
                    return [
                        pltpu.make_async_copy(uids.at[lyr, q], rowb.at[pl.ds(b, 1), pl.ds(0, KU)], bsem.at[b]),
                        pltpu.make_async_copy(qiz, rowb.at[pl.ds(b, 1), pl.ds(KU, 128)], bsem.at[b]),
                    ]

                for c in ucopies(0, 0):
                    c.start()

                def req(q, carry, lyr=lyr, ucopies=ucopies, body=body):
                    b = jax.lax.rem(q, 2)
                    for c in ucopies(q, b):
                        c.wait()

                    @pl.when(q + 1 < nreq)
                    def _():
                        for c in ucopies(q + 1, 1 - b):
                            c.start()

                    return body(q, None, carry, ids=rowb.at[pl.ds(b, 1)], row=0)[0]

                jax.lax.fori_loop(0, nreq, req, jnp.int32(0))

    smem = lambda: pl.BlockSpec(memory_space=pltpu.SMEM)  # noqa: E731
    vmem = lambda: pl.BlockSpec(memory_space=pltpu.VMEM)  # noqa: E731
    call = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((L, M, NH, HD), jnp.float32), jax.ShapeDtypeStruct((L, M, NH, 128), jnp.float32)),
        in_specs=[smem(), smem(), smem(), smem(), smem(), pl.BlockSpec(memory_space=pltpu.HBM)] + [vmem() for _ in range(9)],
        out_specs=(vmem(), vmem()),
        scratch_shapes=[
            pltpu.VMEM((NSLOTS, TILE, 1, 4, HD), F8),
            pltpu.SemaphoreType.DMA((NSLOTS,)),
            pltpu.SMEM((2, KU + 128), jnp.int32),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SMEM((ua.PIPE_IDS_BUFS, KU + 128), jnp.int32),
            pltpu.SemaphoreType.DMA((ua.PIPE_IDS_BUFS,)),
        ],
        interpret=interpret or pltpu.InterpretParams(),
    )
    args = [d["off"], d["scales"], d["un"], d["tab"].reshape(L, nreq, -1), d["kvl"], d["pool"], d["uids"], d["ucodes"],
            d["qv"], d["gv"], d["rk"], d["rv"], d["okr"], d["oka"], np.zeros((1, 128), np.int32)]
    o, l_ = call(*[jnp.asarray(a) for a in args])
    return np.asarray(o), np.asarray(l_)


def _reference(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """Float64 NumPy attention of every (layer, row): its union groups (code bit t) plus its recent tokens."""
    nreq, dcp, NH = d["nreq"], d["dcp"], d["NH"]
    M = nreq * T
    pool = np.asarray(jnp.asarray(d["pool"]).astype(jnp.float32), np.float64)
    off = int(d["off"][0])
    sk, sv = (float(x) for x in d["scales"])
    shift = PER_PAGE.bit_length() - 1
    cshift = shift + dcp.bit_length() - 1
    out = np.zeros((L, M, NH, HD))
    lse = np.full((L, M, NH), -np.inf)
    for lyr in range(L):
        for q in range(nreq):
            n = int(d["un"][lyr, q])
            g = d["uids"][lyr, q, 0, :n].astype(np.int64)
            code = d["ucodes"][lyr, q, 0, :n]
            page = d["tab"][lyr, q][g >> cshift]
            micro = g & (PER_PAGE - 1)
            rows = (micro[:, None] * MICRO + np.arange(MICRO)[None, :]).reshape(-1)
            pages = np.repeat(page, MICRO)
            K = pool[pages, rows, 0, 2 * off, :] * sk
            V = pool[pages, rows, 0, 2 * off + 1, :] * sv
            tcode = np.repeat(code, MICRO)
            for t in range(T):
                r = q * T + t
                sel = ((tcode >> t) & 1) != 0
                ok = d["okr"][r, 0] != 0
                keys = np.concatenate([K[sel], d["rk"][q][ok].astype(np.float64) * sk])
                okany = d["oka"][q, :, 0] != 0
                vals = np.concatenate([V[sel], np.where(okany[:, None], d["rv"][q] * sv, 0.0)[ok]])
                if len(keys) == 0 or (dcp == 1 and not (d["kvl"][r] > 0 or ok.any())):
                    continue  # no token (or, at DCP 1, `row_has` false: the kernel zeroes the gated row)
                s = d["qv"][r].astype(np.float64) @ keys.T * HD**-0.5
                mx = s.max(axis=1, keepdims=True)
                p = np.exp(s - mx)
                att = (p @ vals) / p.sum(axis=1, keepdims=True)
                lse[lyr, r] = mx[:, 0] + np.log(p.sum(axis=1))
                if dcp > 1:
                    out[lyr, r] = att
                else:
                    gate = 1.0 / (1.0 + np.exp(-d["gv"][r].astype(np.float64)))
                    out[lyr, r] = att * gate
    return out, lse


def _cover(d: dict) -> None:
    tiles = [[_tiles(int(n)) for n in row] for row in d["un"]]
    flat = [x for row in tiles for x in row]
    print(f"L2-UAPIPE COVER {d['scen']}: nreq {d['nreq']} DCP {d['dcp']} heads/row {d['NH']} layers {L}; tiles per "
          f"request-layer {tiles}; empty {flat.count(0)}, 1-tile {flat.count(1)}, 2-tile {flat.count(2)}, 5-6-tile "
          f"{sum(1 for x in flat if x in (5, 6))}; last requests {[row[-1] for row in tiles]}", flush=True)
    assert flat.count(0) >= 1 and flat.count(1) >= 1 and sum(1 for x in flat if x in (5, 6)) >= 4


def _check_reference(d: dict, o: np.ndarray, l_: np.ndarray) -> None:
    ref, ref_lse = _reference(d)
    assert np.isfinite(o).all(), "base output has non-finite values"
    if d["dcp"] > 1:
        np.testing.assert_allclose(o, ref, rtol=2e-3, atol=2e-3, err_msg="base vs float64 reference (attn)")
        lv = l_[..., 0]
        fin = np.isfinite(ref_lse)
        assert (np.isfinite(lv) == fin).all(), "base lse finiteness differs from the reference"
        np.testing.assert_allclose(lv[fin], ref_lse[fin], rtol=1e-4, atol=1e-3, err_msg="base vs reference (lse)")
    else:
        live = np.abs(ref).sum(axis=-1) > 0
        np.testing.assert_allclose(o[live], ref[live], rtol=2e-2, atol=2e-2, err_msg="base vs reference (gated)")
    assert np.abs(o).max() > 0.05, "base output is (near) zero: vacuous"


_BASE_CACHE: dict = {}


def _base(scen: str):
    if scen not in _BASE_CACHE:
        d = _inputs(scen)
        o, l_ = _run(d, 0)
        _cover(d)
        _check_reference(d, o, l_)
        _BASE_CACHE[scen] = (d, o, l_)
    return _BASE_CACHE[scen]


def _run_eager(d: dict, mode: int):
    """`_run` with every DMA executed at its start (`dma_execution_mode='eager'`): data lands as early as possible, and
    a copy that is started but never waited over-signals its semaphore, which the interpreter reports at kernel exit
    ("Semaphore <id> has non-zero count ... at kernel exit"). Returns (outputs, lse, those report lines)."""
    import contextlib
    import io
    import sys

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        o, l_ = _run(d, mode, interpret=pltpu.InterpretParams(dma_execution_mode="eager"))
    text = buf.getvalue()
    sys.stdout.write(text)
    return o, l_, [x for x in text.splitlines() if "non-zero count" in x]


def _same(scen: str, mode: int, eager: bool = False, **kw) -> None:
    d, o0, l0 = _base(scen)
    if eager:
        o, l_, left = _run_eager(d, mode)
        assert not left, f"{scen} pipe={mode} (eager DMAs): semaphores not drained at kernel exit: {left[:4]}"
    else:
        o, l_ = _run(d, mode, **kw)
    np.testing.assert_array_equal(o, o0, err_msg=f"{scen} pipe={mode}: attn / gated rows differ from the base")
    if d["dcp"] > 1:
        np.testing.assert_array_equal(l_[..., 0], l0[..., 0], err_msg=f"{scen} pipe={mode}: lse differs from the base")
    how = "eager DMAs, every semaphore drained at kernel exit" if eager else "DMAs run on wait, last in first out"
    print(f"L2-UAPIPE IDENTITY {scen} pipe={mode}: bitwise identical to attend_request (both layers, every row; {how})",
          flush=True)


@pytest.mark.parametrize("scen", list(SCEN))
@pytest.mark.parametrize("mode", MODES)
def test_uapipe_bitwise(scen, mode):
    """Each X-UA-PIPE mode is bitwise identical to `attend_request` on the scenario."""
    _same(scen, mode)


@pytest.mark.parametrize("scen", list(SCEN))
@pytest.mark.parametrize("mode", MODES)
def test_uapipe_bitwise_eager(scen, mode):
    """The same with every DMA landing at its start; no copy may be left unwaited (no semaphore over-signaled)."""
    _same(scen, mode, eager=True)


@pytest.mark.parametrize("scen", list(SCEN))
def test_uapipe_base_eager_control(scen):
    """Controls of the eager leg: the base drains every semaphore and its outputs do not depend on the DMA mode."""
    d, o0, l0 = _base(scen)
    o, l_, left = _run_eager(d, 0)
    assert not left, left[:4]
    np.testing.assert_array_equal(o, o0)
    np.testing.assert_array_equal(l_[..., 0], l0[..., 0])
    print(f"L2-UAPIPE CONTROL {scen}: base drains every semaphore with eager DMAs; outputs independent of the DMA mode",
          flush=True)


def test_uapipe_eager_detects_an_orphan_copy():
    """Positive control of the drained-semaphore check: one extra start (a tile's 64 copies) that nobody waits must be
    reported at kernel exit."""
    d = _inputs("b16_d4")

    def kernel(pool, out, ring, sems):
        for j in range(ua.GROUPS_PER_TILE):
            pltpu.make_async_copy(pool.at[1, pl.ds(4 * j, MICRO)], ring.at[0, pl.ds(j * MICRO, MICRO)], sems.at[0]).start()
        out[...] = jnp.zeros(out.shape, out.dtype)

    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        call = pl.pallas_call(
            kernel, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)], out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[pltpu.VMEM((NSLOTS, TILE, 1, 4, HD), F8), pltpu.SemaphoreType.DMA((NSLOTS,))],
            interpret=pltpu.InterpretParams(dma_execution_mode="eager"),
        )
        np.asarray(call(jnp.asarray(d["pool"])))
    left = [x for x in buf.getvalue().splitlines() if "non-zero count" in x]
    print(f"L2-UAPIPE CONTROL orphan copy reported: {left[:1]}", flush=True)
    assert left, buf.getvalue()[-2000:]


def test_uapipe_modes_validated():
    """Only the documented masks are accepted (PIPE_POST needs PIPE_COISSUE); 0 = off."""
    assert ua.pipe_mode(0) == 0 and ua.pipe_mode(None) == 0
    for m in ua.PIPE_MODES:
        assert ua.pipe_mode(m) == m
    for bad in (4, 6, 8, 9):
        with pytest.raises(ValueError):
            ua.pipe_mode(bad)


def test_uapipe_reference_catches_a_wrong_page():
    """Positive control of the harness's reference check: the base run on inputs whose table row of one request is
    rotated must fail `_check_reference` against the unrotated inputs (the reference reads the pages it claims)."""
    d, o, l_ = _base("b16_d4")
    bad = dict(d)
    bad["tab"] = d["tab"].copy()
    bad["tab"][0, 0, :4] = np.roll(bad["tab"][0, 0, :4], 1)
    ob, lb = _run(bad, 0)
    with pytest.raises(AssertionError):
        _check_reference(d, ob, lb)

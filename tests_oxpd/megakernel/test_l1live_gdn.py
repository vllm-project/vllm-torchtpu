# SPDX-License-Identifier: Apache-2.0
"""q38 X-K-LIVE, L1 GDN half (VLLM_TPU_MK_LIVE_ROWS / `ExecSpec.live_rows`), CPU interpret. Two modes:
  mode 1 (`live_rows` = 1, the flag alone): pad requests skip only their pool-gather DMAs; everything else runs as with
    the flag off. With served pads (null read block, no prior state) every output equals the flag off bit for bit, pad
    rows included (`test_l1live_mode1_*`).
  mode 2 (`live_rows` = 2, + VLLM_TPU_MK_LIVE_GDN_COMPUTE): pad requests also skip the GDN recurrence and their state
    copies. The checks below (A / P / B) are mode 2's.

Per configuration: one flag-off run, one flag-on run with every request live ("all") and flag-on runs with pad requests,
all on the same inputs. Checks, in this order:
  A. pad independence, bitwise, always asserted: every output of the live rows of a flag-on run with pads (residual rows,
     GDN checkpoints of the live requests, and in the whole step the QSA outputs of the live rows / requests) equals the
     flag-on "all" run's, bit for bit. Same compiled program; only the live-row mask differs.
  P. pad state: pad residual rows finite (their GDN output is zero); on the whole-state path (<= 16 rows) a pad request's
     rec_ckpt rows are the same bytes in every GDN layer (never written), and the flag-off kernel's differ (positive
     control); the production-shaped case adds a zero-fill run where every pad rec_ckpt value must be exactly 0.
  B. against flags-off. Always asserted: the first GDN layer's checkpoints of the live rows within a tolerance of
     flag-off (`_tol`; deeper outputs can differ more, as a rounding flip propagates: composed proof c1 j-4e01803f
     measured 616 of 32768 segment residual elements, max 0.0625, between the mode-2 all-live run and flag-off). With Q38_LIVE_STRICT_OFF=1 every live output must equal flag-off
     bit for bit; the harness runs those legs with the XLA CPU ISA capped below FMA (--xla_cpu_max_isa=SSE4_2), testing
     the hypothesis that the default-flag differences are XLA:CPU contracting multiply-adds into FMAs differently inside
     the interpreter's conditional branches than in straight-line code (proofs i1 j-81f97d95 / i2 j-3d943109: 1-2 bf16
     ulp in 4-41 rec_ckpt elements, residual equal on the segment kernel). REFUTED in composed proof c1 j-4e01803f:
     under --xla_cpu_max_isa=SSE4_2 the same 1-2 ulp differences remain (4-6 rec_ckpt elements), so the cause is another
     XLA:CPU compilation difference across the conditional. Otherwise the differences are printed.
Pads in the middle: the MoE's first-seen expert order (so a live row's f32 accumulation order) changes once a pad row ahead
of a live row has a different hidden state; there A and B assert the first GDN layer's checkpoints only and report the rest.

Segment kernel (`qwen38.segment_spec(pool_gdn=True)`, R = 2, the served pool-GDN block): 16 rows at 4 / 8 / 12 / 16 live
rows plus a non-suffix pattern; production-shaped pads (null read block, no prior state); 32 rows (chunked) on the
head-major, whole-state-update and chunk-gather variants.
Whole step (`wholestep_spec`, R = 4, the D14 configuration of test_wholestep.py's DCP-4 tests): main-KV DCP 1 / 2 / 4 at
4 / 8 / 12 / 16 live rows in the 16-row bucket; DCP 4 in the 32-row bucket.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.blocks.hc import RESIDUAL_OUT
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.live_rows import LIVE_ROWS, live_rows_from_qsl
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.models.qwen38_inputs import pack_segment, random_segment_inputs
from vllm_torchtpu.megakernel.models.qwen38_pack import shard_step_inputs

T = 4
G = qwen38.TINY
TILES = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)
LANES = 256


def _qsl(live_reqs) -> np.ndarray:
    q = np.where(np.asarray(live_reqs, bool), T, 0).astype(np.int32)
    return np.concatenate([[0], np.cumsum(q)]).astype(np.int32)


def _mask(live_reqs) -> np.ndarray:
    return np.asarray(live_rows_from_qsl(jnp.asarray(_qsl(live_reqs)), len(live_reqs), T))


def _bits(a) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(a))
    return a.view(np.uint8).reshape(a.shape + (-1,)) if a.dtype.itemsize > 1 else a.view(np.uint8)


def _same(a, b) -> bool:
    return a.shape == b.shape and np.array_equal(_bits(a), _bits(b))


def _eq(a, b, what: str) -> None:
    assert a.shape == b.shape, (what, a.shape, b.shape)
    if not _same(a, b):
        fa, fb = np.asarray(a, np.float32), np.asarray(b, np.float32)
        d = np.abs(fa - fb)
        raise AssertionError(f"{what}: not bit-identical ({int((_bits(a) != _bits(b)).any(-1).sum()) if a.dtype.itemsize > 1 else 'n/a'}"
                             f" elements differ; max |diff| {np.nanmax(d) if d.size else 0})")


def _suffix(live_reqs) -> bool:
    lv = list(np.asarray(live_reqs, bool))
    return lv == sorted(lv, reverse=True)


STRICT_OFF = os.environ.get("Q38_LIVE_STRICT_OFF", "0") == "1"


def _diff(a, b) -> str:
    if _same(a, b):
        return "identical"
    fa, fb = np.asarray(a, np.float32), np.asarray(b, np.float32)
    n = int((_bits(a) != _bits(b)).any(-1).sum()) if a.dtype.itemsize > 1 else -1
    return f"{n} of {a.size} differ, max |diff| {float(np.nanmax(np.abs(fa - fb))):.3g}"


def _tol(a, b, what: str) -> None:
    """Close to flag-off: finite wherever flag-off is, at most max(64, 1e-4 of the elements) differ, and no difference
    above 2^-6 of the largest |flag-off| value. Proof i2 j-3d943109 measured (M) at most 41 of 18.9M rec_ckpt elements
    differing by <= 4.9e-4 (states ~ 0.05-0.5), conv_ckpt / residual equal on the segment kernel; a broken guard
    (mutants M3 / M5 / M7) moves whole rows."""
    fa, fb = np.asarray(a, np.float32), np.asarray(b, np.float32)
    bad_fin = int((np.isfinite(fb) & ~np.isfinite(fa)).sum())
    d = np.where(np.isfinite(fa) & np.isfinite(fb), np.abs(fa - fb), 0.0)
    n = int((d > 0).sum())
    lim = 2.0**-6 * float(np.max(np.abs(np.where(np.isfinite(fb), fb, 0.0)))) if fb.size else 0.0
    ok = bad_fin == 0 and n <= max(64, int(1e-4 * fa.size)) and float(d.max() if d.size else 0.0) <= lim
    assert ok, (f"{what}: not within tolerance of flag-off ({bad_fin} non-finite where flag-off is finite, {n} of "
                f"{fa.size} differ, max |diff| {float(d.max() if d.size else 0.0):.3g} vs limit {lim:.3g})")


def _first_gdn(r: dict, rows, lv) -> dict:
    return {"rec_ckpt[0]": r["rec"][:, :1][:, :, rows], "conv_ckpt[0]": r["conv"][:, :1][:, :, lv]}


def _live_outputs(r: dict, rows, lv, first_only: bool) -> dict:
    """The live rows' outputs of run `r` (GDN checkpoints of the first GDN layer only when `first_only`)."""
    out = {}
    if not first_only:
        out["residual"] = r["res"][:, rows]
    rec, conv = (r["rec"][:, :1], r["conv"][:, :1]) if first_only else (r["rec"], r["conv"])
    out["rec_ckpt"] = rec[:, :, rows]
    out["conv_ckpt"] = conv[:, :, lv]
    if not first_only:
        for k in r.get("extra", ()):
            out[k] = r[k][r["extra_sel"][k](rows, lv)]
    return out


def _check(off: dict, allr: dict, on: dict, live_reqs, *, pad_unwritten: bool, tag: str) -> None:
    lv = np.asarray(live_reqs, bool)
    rows = np.repeat(lv, T)
    assert rows.any() and (~rows).any(), tag  # every checked case has live and pad rows
    first = not _suffix(live_reqs)
    got, ref = _live_outputs(on, rows, lv, first), _live_outputs(allr, rows, lv, first)
    for k in got:  # A
        _eq(got[k], ref[k], f"{tag} {k} live rows vs flag-on all-live (pad independence)")
    assert np.isfinite(np.asarray(off["res"][:, ~rows], np.float32)).all(), f"{tag}: flag-off pad residual not finite"
    assert np.isfinite(np.asarray(on["res"][:, ~rows], np.float32)).all(), f"{tag}: pad residual rows not finite (never written check)"
    if pad_unwritten:  # P
        pr_on, pr_off = on["rec"][:, :, ~rows], off["rec"][:, :, ~rows]
        NL = pr_on.shape[1]
        assert NL >= 2, tag
        for l in range(1, NL):
            _eq(pr_on[:, l], pr_on[:, 0], f"{tag} pad rec_ckpt rows layer {l} vs 0 (flag on: never written)")
        assert any(not _same(pr_off[:, l], pr_off[:, 0]) for l in range(1, NL)), (
            f"{tag}: positive control: the flag-off kernel's pad rec_ckpt rows are the same in every layer")
    if first:
        deep = {k: _same(a, b) for k, a, b in ((k, *v) for k, v in {
            "residual": (on["res"][:, rows], allr["res"][:, rows]), "rec_ckpt": (on["rec"][:, :, rows], allr["rec"][:, :, rows]),
            "conv_ckpt": (on["conv"][:, :, lv], allr["conv"][:, :, lv])}.items())}
        print(f"Q38LIVE {tag}: pads in the middle: first GDN layer asserted; all layers vs all-live (informational) {deep}", flush=True)
    gf, of_ = _first_gdn(on, rows, lv), _first_gdn(off, rows, lv)  # B-tol: always asserted (first GDN layer: no propagation)
    for k in gf:
        _tol(gf[k], of_[k], f"{tag} {k} live rows vs flag-off (tolerance)")
    offv = _live_outputs(off, rows, lv, first)
    rep = {k: _diff(got[k], offv[k]) for k in got}
    print(f"Q38LIVE {tag}: A pad independence PASS ({int(rows.sum())} live / {rows.size} rows); vs flag-off "
          f"{'(asserted)' if STRICT_OFF else '(informational)'}: {rep}", flush=True)
    if STRICT_OFF:  # B
        for k in got:
            _eq(got[k], offv[k], f"{tag} {k} live rows vs flag-off (strict)")


def _strict_probe() -> None:
    """Under Q38_LIVE_STRICT_OFF=1 the harness caps the XLA CPU ISA below FMA (XLA_FLAGS --xla_cpu_max_isa=SSE4_2), so no
    multiply-add can be contracted into an FMA in either program; a strict leg without that flag is invalid."""
    if not STRICT_OFF:
        return
    xf = os.environ.get("XLA_FLAGS", "")
    print(f"Q38LIVE strict: XLA_FLAGS={xf!r}", flush=True)
    assert "--xla_cpu_max_isa=" in xf, f"strict leg invalid: no --xla_cpu_max_isa in XLA_FLAGS {xf!r}"


def _check_all(off: dict, allr: dict, tag: str, n_req: int) -> None:
    """The flag-on all-live run against flag-off: every row (B)."""
    _strict_probe()
    lv = np.ones(n_req, bool)
    rows = np.repeat(lv, T)
    got, offv = _live_outputs(allr, rows, lv, False), _live_outputs(off, rows, lv, False)
    gf, of_ = _first_gdn(allr, rows, lv), _first_gdn(off, rows, lv)
    for k in gf:
        _tol(gf[k], of_[k], f"{tag} all-live {k} vs flag-off (tolerance)")
    rep = {k: _diff(got[k], offv[k]) for k in got}
    print(f"Q38LIVE {tag} all-live vs flag-off {'(asserted)' if STRICT_OFF else '(informational)'}: {rep}", flush=True)
    if STRICT_OFF:
        for k in got:
            _eq(got[k], offv[k], f"{tag} all-live {k} vs flag-off (strict)")


# ------------------------------------------------------------------------------- segment kernel, pool GDN

def _interp(zero_init: bool):
    """True (the kernel's default TPU interpreter, uninitialized memory = its default), or an interpreter whose
    uninitialized scratch reads as zero (so a never-written state slot reads exactly 0)."""
    if not zero_init:
        return True
    import dataclasses as dc

    names = {f.name for f in dc.fields(pltpu.InterpretParams)}
    assert "uninitialized_memory" in names, sorted(names)
    return pltpu.InterpretParams(uninitialized_memory="zero")


def _pool_step(monkeypatch, B: int, live_reqs, *, on: bool, chunk_gather: str = "0", head_major: bool = False,
               pads_like=None, seed: int = 5, zero_init: bool = False) -> dict:
    """Residual and GDN checkpoints of a pool-GDN segment step (R = 2) at `B` rows. Every request has its own pool
    states (requests without a prior state mixed in), except the pad requests of `pads_like` (a request pattern),
    which read the null block 0 with no prior state, as `gdn_blocks` gives them. The inputs depend only on `B`,
    `seed` and `pads_like`, never on the kernel mask `live_reqs` or `on`. `zero_init`: uninitialized scratch reads
    as zero."""
    from tests.megakernel.test_qwen38_gdn_pool import _encode
    from vllm_torchtpu.megakernel.models import qwen38_gdn_pool

    monkeypatch.setattr(qwen38_gdn_pool, "CHUNK_GATHER", chunk_gather)
    R, L = 2, qwen38.SEGMENT_LAYERS
    NS, NL = B // T, L - 1
    assert len(live_reqs) == NS
    tiles = TILES.with_blocks({"gdn": {"gdn_head_major": True}}) if head_major else TILES
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=_interp(zero_init), tiles=tiles, live_rows=int(on))
    assert ex.WIDE == (B > 16)
    spec = qwen38.segment_spec(G, R, L=L, pool_gdn=True)
    inp = random_segment_inputs(jax.random.PRNGKey(seed), G, ex, L, gain=0.25)
    rng = np.random.default_rng(seed + 1)
    nmb = 1 + NL * NS
    rb = 1 + rng.permutation(NL * NS).astype(np.int32).reshape(NL, NS)
    hi = (np.arange(NL * NS).reshape(NL, NS) % 3 != 1).astype(np.int32)
    if pads_like is not None:
        lv = np.asarray(pads_like, bool)[None]
        rb, hi = np.where(lv, rb, 0).astype(np.int32), np.where(lv, hi, 0).astype(np.int32)
    raws = []
    for _ in range(R):
        raw = np.zeros((nmb * 4, 128, 1, 4, LANES), np.uint8)
        for l in range(NL):
            for s in range(NS):
                ssm = rng.standard_normal((6, 128, 128)).astype(np.float32) * 0.05
                conv = rng.standard_normal((3, 1280)).astype(np.float32) * 0.05
                if rb[l, s]:
                    _encode(raw, rb[l, s], ssm, conv)
        raws.append(raw)
    w_mk, _, attn = pack_segment(spec, G, ex, inp)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos, hid, w_mk, st = shard_step_inputs(mesh, ex, inp.pos, inp.x, w_mk, {})
    tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    pool = jnp.concatenate([jax.lax.bitcast_convert_type(jnp.asarray(r), jnp.float8_e4m3fn) for r in raws])
    extras = [
        jax.device_put(kernel.fold(attn), tp),
        jax.device_put(pool, tp),
        jax.device_put(jnp.asarray(rb), rep),
        jax.device_put(jnp.asarray(hi), rep),
    ]
    names = [x.name for x in kernel.extra_inputs(spec, ex)]
    if on:
        assert names[-1] == LIVE_ROWS and len(names) == 5, names
        extras.append(jax.device_put(jnp.asarray(_mask(live_reqs)), rep))
    else:
        assert LIVE_ROWS not in names and len(names) == 4, names
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos, hid, w_mk, st, *extras)
    return dict(
        res=np.asarray(outs[RESIDUAL_OUT]).reshape(R, ex.M, -1)[:, :B],
        rec=np.asarray(outs["rec_ckpt"]).reshape(R, NL, B, -1),
        conv=np.asarray(outs["conv_ckpt"]).reshape(R, NL, NS, -1),
    )


POOL16 = {"r4": [1, 0, 0, 0], "r8": [1, 1, 0, 0], "r12": [1, 1, 1, 0], "r8mid": [0, 1, 0, 1]}


def test_l1live_pool16_live_rows_exact(monkeypatch):
    """16 rows (whole-state path): 4 / 8 / 12 live rows and a non-suffix 8 against the 16-live-row run."""
    off = _pool_step(monkeypatch, 16, [1, 1, 1, 1], on=False)
    allr = _pool_step(monkeypatch, 16, [1, 1, 1, 1], on=2)
    _check_all(off, allr, "pool16 r16", 4)
    for tag, lr in POOL16.items():
        on = _pool_step(monkeypatch, 16, lr, on=2)
        _check(off, allr, on, lr, pad_unwritten=True, tag=f"pool16 {tag}")


def test_l1live_pool16_prod_shape(monkeypatch):
    """16 rows, 8 live, pads as served (null read block 0, no prior state; the same inputs in every run): the mutant
    target. Plus a flag-on run whose uninitialized scratch reads as zero: there every pad rec_ckpt value must be exactly
    0 in every GDN layer (no state computed for a pad row, whatever the interpreter's fill)."""
    lr = [1, 1, 0, 0]
    off = _pool_step(monkeypatch, 16, [1, 1, 1, 1], on=False, pads_like=lr, seed=7)
    allr = _pool_step(monkeypatch, 16, [1, 1, 1, 1], on=2, pads_like=lr, seed=7)
    on = _pool_step(monkeypatch, 16, lr, on=2, pads_like=lr, seed=7)
    _check(off, allr, on, lr, pad_unwritten=True, tag="pool16 prod r8")
    onz = _pool_step(monkeypatch, 16, lr, on=2, pads_like=lr, seed=7, zero_init=True)
    rows = np.repeat(np.asarray(lr, bool), T)
    pz = np.asarray(onz["rec"][:, :, ~rows], np.float32)
    nz = int(np.count_nonzero(pz))
    assert nz == 0, f"pool16 prod r8 zero-init: {nz} non-zero pad rec_ckpt values (flag on: never written)"
    po = np.asarray(off["rec"][:, :, ~rows], np.float32)
    assert np.count_nonzero(po) > 0  # positive control: the flag-off kernel computes pad states (non-zero)
    print(f"Q38LIVE pool16 prod r8 zero-init: pad rec_ckpt all zero ({pz.size} values); live rows vs the default-fill "
          f"flag-on run: residual {_diff(onz['res'][:, rows], on['res'][:, rows])} (informational)", flush=True)


POOL32 = {"hm": dict(head_major=True), "ws": dict(head_major=False), "cg": dict(head_major=True, chunk_gather="1")}


@pytest.mark.parametrize("variant", list(POOL32))
def test_l1live_pool32_live_rows_exact(monkeypatch, variant):
    """32 rows (chunked state ring, two requests per chunk): 12 live rows (suffix pads; the boundary inside a chunk) and
    20 with pads in the middle (chunks mixing pad and live requests both ways) against the 32-live-row run, on the
    head-major update (served), the whole-state update and the chunk gather."""
    kw = POOL32[variant]
    off = _pool_step(monkeypatch, 32, [1] * 8, on=False, **kw)
    allr = _pool_step(monkeypatch, 32, [1] * 8, on=2, **kw)
    _check_all(off, allr, f"pool32 {variant} r32", 8)
    for tag, lr in (("r12", [1, 1, 1, 0, 0, 0, 0, 0]), ("r20mid", [1, 0, 1, 1, 0, 1, 0, 1])):
        on = _pool_step(monkeypatch, 32, lr, on=2, **kw)
        _check(off, allr, on, lr, pad_unwritten=False, tag=f"pool32 {variant} {tag}")


# ------------------------------------------------------------------------------- whole step (D14 configuration)

def _ws(monkeypatch, dcp: int, nreq: int, live_reqs=None, mode: int = 2) -> dict:
    """One whole-step run (test_wholestep._run_c with the DCP-4 test's configuration, R = 4) at main-KV DCP `dcp`."""
    from tests.megakernel.test_wholestep import CTX_D4, D4_KW, _run_c
    from vllm_torchtpu.models.vllm.qwen4_exp import pool_layout

    if dcp > 1:
        monkeypatch.setenv(pool_layout.MAIN_KV_DCP_ENV, str(dcp))
    else:
        monkeypatch.delenv(pool_layout.MAIN_KV_DCP_ENV, raising=False)
    kw = dict(D4_KW, nreq=nreq, ctx=CTX_D4)
    if dcp > 1:
        kw["dcp"] = dcp
    live = None if live_reqs is None else _mask(live_reqs)
    res, _ref, _x0, ck, ring, _steps, iq, ids, d = _run_c(**kw, live=live, live_mode=mode)
    R = ids.shape[0]
    rec, conv = d["_gdn"]["rec_ckpt"], d["_gdn"]["conv_ckpt"]
    NL = rec.shape[0] // R
    B = nreq * T
    out = dict(
        res=res,
        rec=rec.reshape(R, NL, B, -1),
        conv=conv.reshape(R, NL, nreq, -1),
        ck=ck, ring=ring, iq=iq, ids=ids, kr=d["_kr_new"], ir=d["_ir_new"],
    )
    out["extra"] = ("ck", "ring", "iq", "ids", "kr", "ir")
    out["extra_sel"] = {
        "ck": lambda rows, lv: (slice(None), lv),
        "ring": lambda rows, lv: (slice(None), lv),
        "iq": lambda rows, lv: (slice(None), rows),
        "ids": lambda rows, lv: (slice(None), slice(None), rows),
        "kr": lambda rows, lv: (slice(None), slice(None), rows),
        "ir": lambda rows, lv: (slice(None), slice(None), rows),
    }
    return out


@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_l1live_wholestep16_live_rows_exact(monkeypatch, dcp):
    """The 16-row bucket (four requests) at 4 / 8 / 12 live rows against 16, main-KV DCP `dcp`."""
    off = _ws(monkeypatch, dcp, 4)
    allr = _ws(monkeypatch, dcp, 4, [1, 1, 1, 1])
    _check_all(off, allr, f"ws16 dcp{dcp} r16", 4)
    for n in (1, 2, 3):
        lr = [1] * n + [0] * (4 - n)
        on = _ws(monkeypatch, dcp, 4, lr)
        _check(off, allr, on, lr, pad_unwritten=True, tag=f"ws16 dcp{dcp} r{4 * n}")


def test_l1live_wholestep32_dcp4(monkeypatch):
    """The 32-row bucket (eight requests, chunked GDN path) at DCP 4, 20 live rows (the live / pad boundary inside a
    two-request state chunk) against 32."""
    off = _ws(monkeypatch, 4, 8)
    allr = _ws(monkeypatch, 4, 8, [1] * 8)
    _check_all(off, allr, "ws32 dcp4 r32", 8)
    lr = [1, 1, 1, 1, 1, 0, 0, 0]
    on = _ws(monkeypatch, 4, 8, lr)
    _check(off, allr, on, lr, pad_unwritten=False, tag="ws32 dcp4 r20")


# ------------------------------------------------------------------------------- mode 1: gather DMAs only, bitwise

MODE1 = {"pool16": (16, [1, 1, 0, 0], {}), "pool16r4": (16, [1, 0, 0, 0], {}), "pool32hm": (32, [1, 1, 1, 0, 0, 0, 0, 0], dict(head_major=True)),
         "pool32cg": (32, [1, 1, 1, 0, 0, 0, 0, 0], dict(head_major=True, chunk_gather="1"))}


@pytest.mark.parametrize("case", list(MODE1))
def test_l1live_mode1_pool_bitwise(monkeypatch, case):
    """Mode 1 with served pads: every output of the live rows (residual, rec_ckpt, conv_ckpt) equals the flag off bit for
    bit under the default CPU compilation (the guarded GDN code is the gather's DMA issue / wait only), and the pad rows'
    recurrence still runs (first GDN layer: finite, non-zero pad states; mode 1 does not skip compute). Pad rows are
    reported, not asserted: on the composed tree L3's MoE half changes them by design (no routed experts)."""
    B, lr, kw = MODE1[case]
    off = _pool_step(monkeypatch, B, [1] * (B // T), on=0, pads_like=lr, seed=11, **kw)
    on1 = _pool_step(monkeypatch, B, lr, on=1, pads_like=lr, seed=11, **kw)
    rows, lv = np.repeat(np.asarray(lr, bool), T), np.asarray(lr, bool)
    assert rows.any() and (~rows).any()
    _eq(on1["res"][:, rows], off["res"][:, rows], f"mode1 {case} residual live rows vs flag-off")
    _eq(on1["rec"][:, :, rows], off["rec"][:, :, rows], f"mode1 {case} rec_ckpt live rows vs flag-off")
    _eq(on1["conv"][:, :, lv], off["conv"][:, :, lv], f"mode1 {case} conv_ckpt live requests vs flag-off")
    p0 = np.asarray(on1["rec"][:, :1][:, :, ~rows], np.float32)
    assert np.isfinite(p0).all() and np.count_nonzero(p0) > 0, f"mode1 {case}: pad recurrence not computed (first GDN layer)"
    pads = {k: _diff(on1[k][:, ~rows] if k == "res" else on1[k][:, :, ~rows], off[k][:, ~rows] if k == "res" else off[k][:, :, ~rows])
            for k in ("res", "rec")}
    print(f"Q38LIVE mode1 {case}: live rows bit-identical to flag-off ({int(rows.sum())} live / {rows.size} rows); pad rows vs "
          f"flag-off (informational) {pads}", flush=True)


def test_l1live_mode1_wholestep_bitwise(monkeypatch):
    """Mode 1, whole step at DCP 4, 8 live rows of 16: every live output equals the flag off bit for bit (with L3's MoE
    half on the composed tree, its pad rows add no routed experts)."""
    off = _ws(monkeypatch, 4, 4)
    lr = [1, 1, 0, 0]
    on1 = _ws(monkeypatch, 4, 4, lr, mode=1)
    rows, lv = np.repeat(np.asarray(lr, bool), T), np.asarray(lr, bool)
    got, ref = _live_outputs(on1, rows, lv, False), _live_outputs(off, rows, lv, False)
    for k in got:
        _eq(got[k], ref[k], f"mode1 ws16 dcp4 r8 {k} live rows vs flag-off")
    print(f"Q38LIVE mode1 ws16 dcp4 r8: every live output bit-identical to flag-off: {sorted(got)}", flush=True)

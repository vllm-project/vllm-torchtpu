# SPDX-License-Identifier: Apache-2.0
"""q38 L1-GDNINK (VLLM_TPU_MK_GDN_INK, a bit mask; default off), CPU interpret.

  bit 1 GDN-REC-WSHM : the whole-state GDN recurrence (`gdn_rec`, <= 16 rows) takes the head-major per-head update
                       (`recurrence.gdn_rec_wshm`; read from the env at trace time);
  bit 2 GDN-REL-FETCH: `wholestep_tiles` builds the whole step with MM's `Tiles.rel_fetch=True` (dense weight-stream refills
                       decoded at trace time, `TwoRing.fetch_rel`, instead of `TwoRing.fetch`'s run-time decode).

Claim (GDN-INKERNEL = bit 2, GDN-REL-FETCH): every output equals the flag off bit for bit. Each case runs the same inputs twice
in one process, flag off then flag on. Bit 1 (GDN-REC-WSHM) is NOT bitwise on CPU interpret (proof i1 j-50f9f5d3: 1978
residual elements differ, max 0.0625, at pool 16 rows with bits 1 + 2, while bit 2 alone passes at 32 / 64 rows): its node
here (`test_gdnink_wshm_report`) only reports the difference and asserts its engagement; it is not part of the claim. The test kernels take explicit test tiles (not `wholestep_tiles`), so bit 2 is applied here as the tiles it selects
(`rel_fetch=True` on the test tiles); `test_gdnink_wholestep_tiles_rel_fetch` proves the env -> tiles plumbing. Asserted:
  E. engagement: WSHM's trace counter (`recurrence.GDNINK_TRACES`) rises during the flag-on build only, and only where it
     applies (whole-state path, B <= 16; 0 on the chunked path); with bit 2, `TwoRing.fetch_rel` is called during the flag-on
     build and never during the flag-off build (spy on the class method; a cached or ignored flag cannot pass);
  B. bitwise: residual rows, every GDN layer's rec_ckpt / conv_ckpt (and, in the whole step, the QSA outputs: compressed
     keys, ring, index queries / ids, new key / index rows) of flag on == flag off. With X-K-LIVE mode 1
     (VLLM_TPU_MK_LIVE_ROWS=1, served pads: null read block, no prior state) the live rows / requests are asserted and the pad
     rows reported (mode 1 skips pad requests' state gathers, so their first-layer state is the interpreter's
     uninitialized scratch).
Segment kernel (pool-GDN, R = 2; the served GDN block, per-rank vh 6 / kh 2 / KD = VD = 128): 8 / 16 / 32 / 48 / 64 rows; the
dense bank at 768 KiB gives `lin_in` 2 K-tiles (refills cross tiles and layers; i1's 384 KiB fit no 128-row bf16 tile). Whole step (`wholestep_spec`, R = 4,
test_wholestep's D14 DCP-4 configuration): main-KV DCP 1 / 2 / 4 at 16 rows, DCP 4 at 8 / 32 / 64 rows; X-K-LIVE mode 0 and
1. Helpers: tests/megakernel/test_l1live_gdn.py (L1-LIVE), unchanged.

r2 (L1-GDNINK r2, the verifier's gaps): `Tiles.rel_fetch` reroutes every refill of the dense weight ring issued inside a
layer (not only the GDN projections), so the bitwise claim is now made on the WHOLE step output:
  W. every leaf the step returns (`kernel.make_step`'s (hidden, new states) pytree: the hidden output, the residual, every
     GDN layer's rec_ckpt / conv_ckpt, the main-KV rows for the cache write (`qsa_kv_new`), the compressed keys / ring /
     new key / index rows, the debug index queries / ids, and every pass-through state), all ranks, all rows (pads too),
     byte for byte (`_bitwise_all`; the leaf inventory is pinned, `WS_LEAVES` / `POOL_LEAVES`, so an empty capture fails);
     `test_gdnink_bitwise_all_selftest` proves the comparison catches one flipped bit in any leaf;
  S. refill scope (printed, `Q38GINK SCOPE`): per call site (first vllm_torchtpu frame outside stream.py) the flag-off
     `TwoRing.fetch` refills and the flag-on `fetch_rel` refills with the dense weight kinds they decode; asserted: no
     `fetch_rel` with the flag off; with it on, every decoded kind is a dense (static) ring kind, never an expert kind, and
     every remaining `fetch` site is a flag-off `fetch` site.
"""

from __future__ import annotations

import collections
import dataclasses
import os
import sys

import jax
import numpy as np
import pytest

from tests.megakernel import test_l1live_gdn as _l1live
from tests.megakernel import test_wholestep as _wst
from tests.megakernel.test_l1live_gdn import T, _diff, _eq, _live_outputs, _pool_step, _ws
from vllm_torchtpu.megakernel import kernel as _kernel
from vllm_torchtpu.megakernel import stream as _stream
from vllm_torchtpu.megakernel.blocks.gdn import recurrence as _rec
from vllm_torchtpu.megakernel.blocks.hc import RESIDUAL_OUT
from vllm_torchtpu.megakernel.wholestep.qsa_full import KV_NEW
from vllm_torchtpu.megakernel.wholestep.qsa_full_b import DBG_IDS, DBG_IQ
from vllm_torchtpu.megakernel.wholestep.qsa_full_c import CK_NEW, IR_NEW, KR_NEW, RING_NEW

ENV = "VLLM_TPU_MK_GDN_INK"
_CALLS = {"fetch_rel": 0, "fetch": 0}
# r2: (method, call site, decoded kind) -> trace-time calls; ring kinds seen (static = dense ring, dynamic = expert ring)
_SITES: collections.Counter = collections.Counter()
_KINDS: dict = {"static": set(), "dynamic": set()}
_STEP_OUT: list = []
# r2: the step-output leaves that must be present (pinned to the current kernels; `hid` = make_step's hidden output)
WS_LEAVES = ("hid", RESIDUAL_OUT, "rec_ckpt", "conv_ckpt", KV_NEW, CK_NEW, RING_NEW, KR_NEW, IR_NEW, DBG_IQ, DBG_IDS)
POOL_LEAVES = ("hid", RESIDUAL_OUT, "rec_ckpt", "conv_ckpt")


def _site() -> str:
    """The first vllm_torchtpu frame outside megakernel/stream.py (the block code whose ring consumption issued the refill)."""
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename.replace(os.sep, "/")
        if "/vllm_torchtpu/" in fn and not fn.endswith("/megakernel/stream.py"):
            return f"{fn.split('/vllm_torchtpu/', 1)[1]}:{f.f_code.co_name}:{f.f_lineno}"
        f = f.f_back
    return "?"


@pytest.fixture(autouse=True)
def _spy(monkeypatch):
    """Count TwoRing.fetch_rel / TwoRing.fetch calls (trace time); the originals run unchanged. r2: also record each call's
    site and, for a `fetch_rel` leaf call (static layer type), the dense kinds it decodes (`StreamParams.rel_decode`, the
    same decode the original runs); and capture every step's full output (`kernel.make_step` wrapped; the step runs
    unchanged)."""
    orig_rel, orig_fetch, orig_make = _stream.TwoRing.fetch_rel, _stream.TwoRing.fetch, _kernel.make_step

    def _ring_kinds(sp):
        _KINDS["static"].update(sp.static_kinds)
        _KINDS["dynamic"].update(sp.dynamic_kinds)

    def fetch_rel(self, g, blk, lt, r, dt=None, dt_vals=(0,), lts=None):
        _CALLS["fetch_rel"] += 1
        if isinstance(lt, int):
            _ring_kinds(self.sp)
            site = _site()
            for kd in sorted({self.sp.rel_decode(lt, r + v)[4] for v in ((0,) if dt is None else dt_vals)}):
                _SITES[("fetch_rel", site, kd)] += 1
        return orig_rel(self, g, blk, lt, r, dt, dt_vals, lts)

    def fetch(self, *a, **k):
        _CALLS["fetch"] += 1
        _ring_kinds(self.sp)
        _SITES[("fetch", _site(), "?")] += 1
        return orig_fetch(self, *a, **k)

    def make_step(*a, **k):
        step, info = orig_make(*a, **k)

        class _Cap:
            def __call__(self, *x, **y):
                out = step(*x, **y)
                _STEP_OUT.append(_leaves(out))
                return out

            def lower(self, *x, **y):
                return step.lower(*x, **y)

        return _Cap(), info

    monkeypatch.setattr(_stream.TwoRing, "fetch_rel", fetch_rel)
    monkeypatch.setattr(_stream.TwoRing, "fetch", fetch)
    monkeypatch.setattr(_kernel, "make_step", make_step)
    yield


def _leaves(out) -> dict:
    """make_step's (hidden, new states) -> {leaf name: host array}; `hid` for the hidden output, the state key (plus [i] for
    a tupled state) otherwise."""
    hid, states = out
    res = {"hid": np.asarray(hid)}
    for k, v in states.items():
        for path, x in jax.tree_util.tree_flatten_with_path(v)[0]:
            res[k + "".join(f"[{getattr(p, 'idx', getattr(p, 'key', p))}]" for p in path)] = np.asarray(x)
    return res


def _bitwise_all(off: dict, on: dict, tag: str, want=()) -> None:
    """Every leaf of the flag-on step output equals the flag-off leaf byte for byte (shape, dtype, bytes; NaN payloads and
    signed zeros included); the same leaf names on both sides; every name of `want` present."""
    missing = [k for k in want if k not in off]
    assert not missing, f"{tag}: step output lacks pinned leaves {missing} (have {sorted(off)})"
    assert set(off) == set(on), f"{tag}: leaf names differ flag on vs off: {sorted(set(off) ^ set(on))}"
    bad, nbytes = [], 0
    for k in sorted(off):
        a, b = np.ascontiguousarray(off[k]), np.ascontiguousarray(on[k])
        if a.shape != b.shape or a.dtype != b.dtype:
            bad.append(f"{k}: {a.shape} {a.dtype} vs {b.shape} {b.dtype}")
            continue
        nbytes += a.nbytes
        if a.tobytes() != b.tobytes():
            ua = np.frombuffer(a.tobytes(), np.uint8).reshape(a.size, a.dtype.itemsize)
            ub = np.frombuffer(b.tobytes(), np.uint8).reshape(b.size, b.dtype.itemsize)
            bad.append(f"{k}: {int((ua != ub).any(-1).sum())} of {a.size} elements differ")
    if bad:
        raise AssertionError(f"{tag} step output flag on vs off: not bit-identical ({'; '.join(bad[:8])})")
    print(f"Q38GINK {tag}: whole output W PASS ({len(off)} leaves, {nbytes} bytes, all ranks / rows, byte-equal): "
          f"{ {k: (tuple(off[k].shape), str(off[k].dtype)) for k in sorted(off)} }", flush=True)


def _scope(s_off: collections.Counter, s_on: collections.Counter, tag: str) -> None:
    """Refill scope of GDN-REL-FETCH at this build (see the module docstring, S)."""
    off_rel = {k: n for k, n in s_off.items() if k[0] == "fetch_rel"}
    assert not off_rel, f"{tag}: fetch_rel ran with the flag off: {off_rel}"
    off_f = collections.Counter({k[1]: n for k, n in s_off.items() if k[0] == "fetch"})
    on_f = collections.Counter({k[1]: n for k, n in s_on.items() if k[0] == "fetch"})
    on_rel: dict = collections.defaultdict(collections.Counter)
    for (m, site, kd), n in s_on.items():
        if m == "fetch_rel":
            on_rel[site][kd] += n
    assert off_f and on_rel, f"{tag}: positive controls: flag-off fetch sites {dict(off_f)}, flag-on fetch_rel sites {dict(on_rel)}"
    kinds = set().union(*on_rel.values())
    assert kinds <= _KINDS["static"] and not kinds & _KINDS["dynamic"], (tag, kinds, _KINDS)
    assert set(on_f) <= set(off_f), f"{tag}: new plain fetch sites with the flag on: {set(on_f) - set(off_f)}"
    print(f"Q38GINK SCOPE {tag}: flag off TwoRing.fetch sites {dict(sorted(off_f.items()))}", flush=True)
    print(f"Q38GINK SCOPE {tag}: flag on fetch_rel (rerouted, trace-time decode) "
          f"{ {s: dict(sorted(c.items())) for s, c in sorted(on_rel.items())} }", flush=True)
    print(f"Q38GINK SCOPE {tag}: flag on TwoRing.fetch (NOT rerouted) {dict(sorted(on_f.items()))}; dense kinds rerouted "
          f"{sorted(kinds)} of the dense ring's {sorted(_KINDS['static'])}; expert-ring kinds {sorted(_KINDS['dynamic'])} untouched",
          flush=True)


def _counts() -> tuple[int, int, int]:
    return _rec.GDNINK_TRACES.get("GDN-REC-WSHM", 0), _CALLS["fetch_rel"], _CALLS["fetch"]


def _run(monkeypatch, ink: int, fn, tiles_mod, tiles_name: str, full: bool = False):
    """`fn()` with VLLM_TPU_MK_GDN_INK=`ink` (unset for 0) and, for bit 2, `rel_fetch=True` on the module's test tiles;
    returns (result, counter deltas (wshm traces, fetch_rel calls, fetch calls)); with `full`, also the run's site deltas
    and its one step's full output leaves."""
    base = getattr(tiles_mod, tiles_name)
    if ink:
        monkeypatch.setenv(ENV, str(ink))
    else:
        monkeypatch.delenv(ENV, raising=False)
    if ink & 2:
        monkeypatch.setattr(tiles_mod, tiles_name, dataclasses.replace(base, rel_fetch=True))
    c0, s0, n0 = _counts(), collections.Counter(_SITES), len(_STEP_OUT)
    try:
        out = fn()
    finally:
        monkeypatch.setattr(tiles_mod, tiles_name, base)
        monkeypatch.delenv(ENV, raising=False)
    c1 = _counts()
    d = tuple(b - a for a, b in zip(c0, c1))
    if not full:
        return out, d
    assert len(_STEP_OUT) == n0 + 1, f"expected one captured step call, got {len(_STEP_OUT) - n0}"
    return out, d, _SITES - s0, _STEP_OUT[-1]


def _engaged(d_off, d_on, ink: int, B: int, tag: str) -> None:
    assert d_off[0] == 0 and d_off[1] == 0, f"{tag}: a GDNINK variant ran with the flag off: {d_off}"
    assert d_off[2] > 0, f"{tag}: positive control: the flag-off build issued no TwoRing.fetch refills ({d_off})"
    ws = bool(ink & 1) and B <= 16
    rf = bool(ink & 2)
    assert (d_on[0] > 0) == ws, f"{tag}: GDN-REC-WSHM traces {d_on[0]} (want {'> 0' if ws else '0'})"
    assert (d_on[1] > 0) == rf, f"{tag}: GDN-REL-FETCH: TwoRing.fetch_rel calls {d_on[1]} (want {'> 0' if rf else '0'})"
    print(f"Q38GINK {tag}: engagement E PASS (flag on: wshm traces {d_on[0]}, fetch_rel {d_on[1]}, fetch {d_on[2]}; "
          f"flag off: fetch_rel 0, fetch {d_off[2]})", flush=True)


def _outputs(r: dict, rows, lv) -> dict:
    """`_live_outputs` (live rows / requests); an extra whose selector does not fit this request count (the 2-request whole
    step: i1 j-50f9f5d3 IndexError) is compared whole (stricter)."""
    out = _live_outputs({k: v for k, v in r.items() if k not in ("extra", "extra_sel")}, rows, lv, False)
    for k in r.get("extra", ()):
        try:
            out[k] = r[k][r["extra_sel"][k](rows, lv)]
        except IndexError:
            out[k + "(whole)"] = r[k]
    return out


def _compare(off: dict, on: dict, lr, tag: str) -> None:
    lv = np.asarray(lr, bool)
    rows = np.repeat(lv, T)
    got, ref = _outputs(on, rows, lv), _outputs(off, rows, lv)
    for k in got:
        _eq(got[k], ref[k], f"{tag} {k} (live rows) flag on vs off")
    pads = {}
    if (~rows).any():
        pads = {k: _diff(on[k][:, ~rows] if k == "res" else on[k][:, :, ~rows],
                         off[k][:, ~rows] if k == "res" else off[k][:, :, ~rows]) for k in ("res", "rec")}
    print(f"Q38GINK {tag}: bitwise B PASS ({int(rows.sum())} live / {rows.size} rows; {sorted(got)}); pad rows "
          f"(informational) {pads or 'none'}", flush=True)


POOL = [(8, 0, 2, 768), (8, 1, 2, 768), (16, 0, 2, 768), (16, 1, 2, 768), (16, 0, 2, 1024), (32, 0, 2, 768), (32, 1, 2, 768),
        (48, 0, 2, 768), (64, 0, 2, 768)]


@pytest.mark.parametrize(("B", "mode", "ink", "bank_kib"), POOL, ids=[f"b{b}_m{m}_ink{i}_bank{k}" for b, m, i, k in POOL])
def test_gdnink_pool_bitwise(monkeypatch, B, mode, ink, bank_kib):
    """Segment kernel (pool GDN, R = 2) at `B` rows, X-K-LIVE mode `mode` (1: half the requests are served pads), dense bank
    `bank_kib` KiB (768: `lin_in` [256, 2176] bf16 here in 2 K-tiles; 1024: 1 tile). Chunk gather as served above 32 rows."""
    NS = B // T
    lr = [1] * NS if mode == 0 else [1] * (NS // 2) + [0] * (NS - NS // 2)
    monkeypatch.setattr(_l1live, "TILES", dataclasses.replace(_l1live.TILES, bank=bank_kib << 10))
    kw = dict(on=mode, seed=13, head_major=B > 16, chunk_gather="1" if B > 32 else "0")
    if mode:
        kw["pads_like"] = lr
    tag = f"pool b{B} mode{mode} ink{ink} bank{bank_kib}K"
    off, d_off, s_off, w_off = _run(monkeypatch, 0, lambda: _pool_step(monkeypatch, B, lr, **kw), _l1live, "TILES", True)
    on, d_on, s_on, w_on = _run(monkeypatch, ink, lambda: _pool_step(monkeypatch, B, lr, **kw), _l1live, "TILES", True)
    _engaged(d_off, d_on, ink, B, tag)
    _compare(off, on, lr, tag)
    _bitwise_all(w_off, w_on, tag, POOL_LEAVES)
    _scope(s_off, s_on, tag)


WS = [(1, 4, 0), (2, 4, 0), (4, 4, 0), (1, 4, 1), (2, 4, 1), (4, 4, 1), (4, 2, 0), (4, 2, 1), (4, 8, 0), (4, 8, 1), (4, 16, 0),
      (4, 12, 0)]


@pytest.mark.parametrize(("dcp", "nreq", "mode"), WS, ids=[f"dcp{d}_b{4 * n}_m{m}" for d, n, m in WS])
def test_gdnink_wholestep_bitwise(monkeypatch, dcp, nreq, mode):
    """Whole step (R = 4) at main-KV DCP `dcp`, `nreq` requests (4 rows each), X-K-LIVE mode `mode`; VLLM_TPU_MK_GDN_INK=2.
    r2: the whole step output (W) byte-equal and the refill scope (S) printed / asserted."""
    ink = 2
    lr = [1] * nreq if mode == 0 else [1] * (nreq // 2) + [0] * (nreq - nreq // 2)
    args = (monkeypatch, dcp, nreq) + ((lr, 1) if mode else (None,))
    tag = f"ws dcp{dcp} b{4 * nreq} mode{mode} ink{ink}"
    off, d_off, s_off, w_off = _run(monkeypatch, 0, lambda: _ws(*args), _wst, "_TILES", True)
    on, d_on, s_on, w_on = _run(monkeypatch, ink, lambda: _ws(*args), _wst, "_TILES", True)
    _engaged(d_off, d_on, ink, 4 * nreq, tag)
    _compare(off, on, lr, tag)
    _bitwise_all(w_off, w_on, tag, WS_LEAVES)
    _scope(s_off, s_on, tag)


@pytest.mark.parametrize("where", ["pool16", "ws16"])
def test_gdnink_wshm_report(monkeypatch, where):
    """Bit 1 alone (GDN-REC-WSHM), REPORT ONLY (not part of the claim): engagement asserted, the difference vs flag off printed."""
    if where == "pool16":
        lr, kw = [1] * 4, dict(on=0, seed=13, head_major=False, chunk_gather="0")
        off, d_off = _run(monkeypatch, 0, lambda: _pool_step(monkeypatch, 16, lr, **kw), _l1live, "TILES")
        on, d_on = _run(monkeypatch, 1, lambda: _pool_step(monkeypatch, 16, lr, **kw), _l1live, "TILES")
    else:
        lr = [1] * 4
        off, d_off = _run(monkeypatch, 0, lambda: _ws(monkeypatch, 4, 4, None), _wst, "_TILES")
        on, d_on = _run(monkeypatch, 1, lambda: _ws(monkeypatch, 4, 4, None), _wst, "_TILES")
    _engaged(d_off, d_on, 1, 16, f"wshm {where}")
    rows, lv = np.repeat(np.asarray(lr, bool), T), np.asarray(lr, bool)
    got, ref = _outputs(on, rows, lv), _outputs(off, rows, lv)
    first = {"rec_ckpt[0]": _diff(on["rec"][:, :1], off["rec"][:, :1]), "conv_ckpt[0]": _diff(on["conv"][:, :1], off["conv"][:, :1])}
    print(f"Q38GINK wshm {where} REPORT (bit 1 alone vs flag off, not asserted): first GDN layer {first}; all outputs "
          f"{ {k: _diff(got[k], ref[k]) for k in got} }", flush=True)


def test_gdnink_mode_parse(monkeypatch):
    """The flag's parse: unset / "0" / junk = off; bit 1 selects WSHM."""
    for v, want in (("", 0), ("0", 0), ("x", 0), ("1", 1), ("2", 2), ("3", 3)):
        monkeypatch.setenv(ENV, v)
        assert _rec.gdnink_mode() == want, (v, _rec.gdnink_mode())
        assert _rec.gdnink_ws(None) == bool(want & 1), v
    monkeypatch.delenv(ENV, raising=False)
    assert _rec.gdnink_mode() == 0 and not _rec.gdnink_ws(None)


@pytest.mark.parametrize("B", [8, 16, 32, 48, 64])
def test_gdnink_wholestep_tiles_rel_fetch(monkeypatch, caplog, B):
    """Bit 2 -> `wholestep_tiles(B).rel_fetch` True, every other field unchanged; unset / "0" / "1" -> the flag-off tiles
    exactly; one engagement line per bucket (logger 'vllm.' + module, megakernel_runtime/qwen38_wholestep_call.py)."""
    import logging

    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    monkeypatch.delenv(ENV, raising=False)
    off = wsc.wholestep_tiles(B)
    assert off.rel_fetch is False, off
    for v in ("0", "1", "x"):
        monkeypatch.setenv(ENV, v)
        assert wsc.wholestep_tiles(B) == off, v
    monkeypatch.setattr(wsc, "_GDNINK_RF_LOGGED", set())
    with caplog.at_level(logging.INFO, logger="vllm." + wsc.__name__):
        for v in ("2", "3"):
            monkeypatch.setenv(ENV, v)
            on = wsc.wholestep_tiles(B)
            assert on.rel_fetch is True and dataclasses.replace(on, rel_fetch=False) == off, (v, on, off)
    lines = [r.getMessage() for r in caplog.records if "q38 L1-GDNINK GDN-REL-FETCH on" in r.getMessage()]
    assert len(lines) == 1 and f"whole step at {B} rows" in lines[0], lines
    monkeypatch.delenv(ENV, raising=False)
    print(f"Q38GINK tiles b{B}: rel_fetch off -> on only with bit 2; engagement line: {lines[0]}", flush=True)


def test_gdnink_bitwise_all_selftest():
    """r2 harness control: `_bitwise_all` passes identical outputs (NaN payloads included) and FAILS one flipped bit in ANY
    leaf (f32 / bf16 / fp8 / int32), a signed zero, a missing pinned leaf, an extra leaf and a shape change."""
    import ml_dtypes

    rng = np.random.default_rng(3)
    base = {"hid": rng.standard_normal((4, 8, 16)).astype(np.float32),
            RESIDUAL_OUT: rng.standard_normal((4, 32)).astype(ml_dtypes.bfloat16),
            KV_NEW: rng.standard_normal((2, 3, 8)).astype(ml_dtypes.float8_e4m3fn),
            DBG_IDS: rng.integers(0, 1 << 20, (2, 5), dtype=np.int32)}
    base["hid"][0, 0, 0] = np.nan
    cp = lambda: {k: v.copy() for k, v in base.items()}  # noqa: E731
    _bitwise_all(base, cp(), "selftest identical", tuple(base))
    caught = []
    for k in base:
        on = cp()
        u = on[k].view(np.uint8).reshape(-1)
        u[-1] ^= 1
        with pytest.raises(AssertionError, match="not bit-identical") as ei:
            _bitwise_all(base, on, f"selftest flip {k}")
        assert k in str(ei.value), (k, str(ei.value))
        caught.append(k)
    z, nz = cp(), cp()
    z["hid"][1, 0, 0], nz["hid"][1, 0, 0] = 0.0, -0.0
    with pytest.raises(AssertionError, match="not bit-identical"):
        _bitwise_all(z, nz, "selftest signed zero")
    with pytest.raises(AssertionError, match="lacks pinned leaves"):
        _bitwise_all({k: v for k, v in base.items() if k != KV_NEW}, {k: v for k, v in cp().items() if k != KV_NEW}, "s", tuple(base))
    with pytest.raises(AssertionError, match="leaf names differ"):
        _bitwise_all(base, {**cp(), "extra": np.zeros(1, np.float32)}, "selftest extra leaf")
    sh = cp()
    sh["hid"] = sh["hid"].reshape(8, 4, 16)
    with pytest.raises(AssertionError, match="not bit-identical"):
        _bitwise_all(base, sh, "selftest shape")
    print(f"Q38GINK selftest: _bitwise_all catches a one-bit flip in every leaf {caught}, a signed zero, a missing pinned "
          f"leaf, an extra leaf and a shape change; identical outputs (NaN included) pass", flush=True)

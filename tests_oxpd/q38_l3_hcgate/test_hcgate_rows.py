# SPDX-License-Identifier: Apache-2.0
"""HCGATE: the fused decode HC row gate (q38 L3-HCGATE), on L3 tree v1 (ee289379f + HCX + D68).

VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS = N or N:M (unset = off): with TPU_QWEN4_HC_DECODE_FUSED
armed, the fused decode HC kernel (mix and the combine_mix prologue) runs only for HC inputs
of <= N rows (and, with N:M, of >= M rows); a disallowed input takes exactly the path of the
same module with the fused ops unbuilt (FINAL - HC fused), which is HCX's XLA path where HCX
applies. Floor (cycle-4 ruling): N < 128 needs HCX >= 128, else build raises. Guard neutrality
(T95u fix 521ca02f7): only 128 and 128:256 are accepted, and rows >= 256 is decided as
tokens % 128 == 0 (no value bound on the traced token count); see test_hcgate_symint.py.

The real GatedResidual is driven (hcx_load: real packages in the CPU image, package stubs
plus vLLM v0.31.0's common file on a Mac) with recording stand-ins for the fused decode
and prefill-tile ops at small dims (the stand-ins' outputs differ from the XLA path, so a
wrong route changes the bits). Checked:

1. helpers: off by default, the limit and the band, bad values raise, the floor, off never
   compares the rows.
2. Off: every path, call and output equal to FINAL 5d368f166's file at every row count;
   set on a module without fused ops (FINAL - HC fused, the MTP draft): also FINAL's path.
3. On, per bucket (16 / 96 / 128 / 160 / 192 / 224 / 256 / 512) at several limits: <= N
   bit-equal (and call-equal) to FINAL with HC fused; > N bit-equal to FINAL without HC
   fused; both sides of the threshold (N and the next row count); mix, combine and
   combine_mix (the next HC's gate decides the prologue).
4. Composition with HCX (VLLM_TPU_HC_XLA_MAX_ROWS): > N rows at 128 / 256 take XLA and
   equal FINAL - HC fused + HCX; <= N keeps the fused kernel (fused precedence).
5. Guards: no row read on unarmed modules; read once at build; 2-D only; the host-side
   engagement counter (hc_engagement.count_step) follows the gate.
6. tpu_compiler._tpu_compile_env_factors keys the value only when set; not in envs.py.

Run with --noconftest on a Mac; mutation-checked by q38_l3_hcgate_mutrun.py.
"""

import ast
import logging
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "q38_l3_hcx"))
import hcx_load  # noqa: E402

ENV = "VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS"
HCX_ENV = "VLLM_TPU_HC_XLA_MAX_ROWS"
HC, H, LR = 4, 8, 4
NORM_T, UP_T, FULL_T = 128, 128, 256
BUCKETS = (16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 512)
ROWS = (1, 16, 64, 96, 112, 128, 129, 160, 192, 208, 224, 256, 384, 512, 1024)


@pytest.fixture
def m(monkeypatch):
    for key in (ENV, HCX_ENV, "VLLM_TPU_DECODE_TP_AR"):
        monkeypatch.delenv(key, raising=False)
    for key in list(os.environ):
        if key.startswith("TPU_QWEN4_HC_"):
            monkeypatch.delenv(key)
    return hcx_load.load()


# ---------------------------------------------------------------- builders
def _config(mod):
    return mod.HyperConnectionConfig(
        hc_count=HC, hidden_size=H, params_dtype=torch.bfloat16, hc_lowrank=LR,
        rms_norm_eps=1e-6, hc_per_branch_norm=True,
    )


def build(m, *, tiles=True, inject=True, fused=False, seed=0, cls_mod=None, like=None):
    """A GatedResidual; tile and fused stand-ins are attached after construction (the
    real Pallas ops need the TPU adapter). ``like``: copy that module's weights."""
    mod = (cls_mod or m.hc).GatedResidual(_config(cls_mod or m.hc), use_combine=inject)
    with torch.no_grad():
        if like is not None:
            mod.load_state_dict(like.state_dict())
        else:
            g = torch.Generator().manual_seed(seed)
            for p in mod.parameters():
                p.copy_((torch.randn(p.shape, generator=g) * 0.4).to(p.dtype))
    mod.calls = []
    if tiles:
        _attach_tiles(mod)
    if fused:
        _attach_fused(mod)
    return mod


def _rec(mod, name, fn):
    def op(*args):
        mod.calls.append(name)
        return fn(*args)

    return op


def _const(rows, value, width=H):
    return torch.full((rows, width), value, dtype=torch.bfloat16)


def _attach_tiles(mod):
    mod.prefill_norm_tile, mod.prefill_up_tile, mod.prefill_full_tile = NORM_T, UP_T, FULL_T
    mod._prefill_tile_step = 2 * FULL_T  # the served default (no ODD_TILES / FUSED_MIX)
    mod._prefill_norm_op = _rec(mod, "norm_tile", lambda x, w: (x.float() * 0.5).to(x.dtype))
    mod._prefill_up_op = _rec(mod, "up_tile", lambda g, n, w: _const(g.shape[0], 0.25))
    mod._prefill_full_op = _rec(mod, "full", lambda x, w, d, u: (_const(x.shape[0], 0.125), x))
    if hasattr(mod, "block_inject_weight"):
        mod._prefill_combine_op = _rec(mod, "combine_tile", lambda r, b, g: (r.float() + 1).to(r.dtype))


def _attach_fused(mod):
    """What TPU_QWEN4_HC_DECODE_FUSED builds (the decode mix and the combine_mix prologue)."""
    inject = hasattr(mod, "block_inject_weight")

    def mix_op(x, *w):
        mod.calls.append("decode_fused")
        mixed = _const(x.shape[0], 0.0625)
        return (mixed, torch.full((x.shape[0], HC), 0.5)) if inject else mixed

    mod._decode_mix_op = mix_op
    if inject:
        mod._decode_combine_mix_op = _rec(
            mod, "decode_combine_mix",
            lambda h, b, g, *w: (_const(h.shape[0], 0.0625), torch.full((h.shape[0], HC), 0.5), h),
        )


def _x(rows, seed=1):
    g = torch.Generator().manual_seed(rows * 7 + seed)
    return torch.randn(rows, HC * H, generator=g).bfloat16()


def _block(rows):
    g = torch.Generator().manual_seed(rows * 11)
    return torch.randn(rows, H, generator=g).bfloat16()


def _equal(a, b):
    if isinstance(a, tuple | list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


def _run_all(mod, nxt, rows):
    """mix, combine and combine_mix of one row count: (outputs, calls of mod, calls of nxt)."""
    mod.calls.clear()
    nxt.calls.clear()
    x = _x(rows)
    mixed, residual = mod.mix(x)
    outs = [mixed, residual]
    if hasattr(mod, "block_inject_weight"):
        outs.append(mod.combine(_block(rows), residual))
        outs.append(mod.combine_mix(_block(rows), residual, nxt))
    return outs, list(mod.calls), list(nxt.calls)


def _pair(m, monkeypatch, *, gate=None, hcx=None, fused=True, like=None, **kw):
    """(mod, nxt) built under the given env (both read it once at build)."""
    for key, value in ((ENV, gate), (HCX_ENV, hcx)):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    a = build(m, fused=fused, like=like[0] if like else None, **kw)
    b = build(m, fused=fused, seed=5, like=like[1] if like else None, **kw)
    for key in (ENV, HCX_ENV):
        monkeypatch.delenv(key, raising=False)
    return a, b


def _final(m, monkeypatch, like, **kw):
    """FINAL (HC fused armed, gate unset) with the same weights."""
    return _pair(m, monkeypatch, fused=True, like=like, **kw)


def _nohcf(m, monkeypatch, like, hcx=None, **kw):
    """FINAL - HC fused (TPU_QWEN4_HC_DECODE_FUSED unset), optionally + HCX."""
    return _pair(m, monkeypatch, fused=False, hcx=hcx, like=like, **kw)


# ---------------------------------------------------------------- 1. helpers
def test_helper_off_by_default(m, monkeypatch):
    assert m.hcd.decode_fused_max_rows() == (0, 0)
    monkeypatch.setenv(ENV, "  ")
    assert m.hcd.decode_fused_max_rows() == (0, 0)
    assert all(m.hcd.decode_fused_allows(t, 0) for t in (1, 128, 224, 512, 4096))
    assert all(m.hcd.decode_fused_allows(t, (0, 0)) for t in (1, 128, 224, 512, 4096))


def test_helper_limit(m, monkeypatch):
    monkeypatch.setenv(ENV, "128")
    assert m.hcd.decode_fused_max_rows() == (128, 0)
    for lim in (128, (128, 0)):
        assert all(m.hcd.decode_fused_allows(t, lim) for t in (1, 16, 96, 127, 128))
        assert not any(m.hcd.decode_fused_allows(t, lim) for t in (129, 160, 224, 256, 512))
    monkeypatch.setenv(ENV, " 128:256 ")
    assert m.hcd.decode_fused_max_rows() == (128, 256)


def test_helper_band(m, monkeypatch):
    monkeypatch.setenv(ENV, "128:256")
    assert m.hcd.decode_fused_max_rows() == (128, 256)
    assert all(m.hcd.decode_fused_allows(t, (128, 256)) for t in (1, 96, 128, 256, 384, 512))
    assert not any(m.hcd.decode_fused_allows(t, (128, 256)) for t in (129, 160, 192, 224, 255))
    # the band's upper side is "a multiple of the 128-row tile above 128" (guard-neutral): equal to
    # rows >= 256 at every bucket of the ladder
    ladder = [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 512, 1024, 2048]
    for t in ladder:
        assert m.hcd.decode_fused_allows(t, (128, 256)) == (t <= 128 or t >= 256), t


@pytest.mark.parametrize("value", ["160", "192", "224", "512", "128:144", "128:224", "128:512", "160:256", "4096"])
def test_only_guard_neutral_values_are_accepted(m, monkeypatch, value):
    """T95u fix: any N other than 128 or M other than 256 would put a bound on the traced token
    count that decode_token_tile does not already put there; build refuses it."""
    monkeypatch.setenv(HCX_ENV, "256")  # so the floor is not what refuses it
    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError, match="only 128 and 128:256"):
        m.hcd.decode_fused_max_rows()
    with pytest.raises(ValueError, match="only 128 and 128:256"):
        build(m)


@pytest.mark.parametrize("bad", ["0", "-1", "x", "1.5", "128rows", "128:", ":256", "128:143", "128:1024",
                                 "256:128", "128:256:512", "128:2x6", "0:256"])
def test_helper_bad_values(m, monkeypatch, bad):
    monkeypatch.setenv(ENV, bad)
    with pytest.raises(ValueError):
        m.hcd.decode_fused_max_rows()
    with pytest.raises(ValueError):
        build(m)


FLOOR_MSG = "needs VLLM_TPU_HC_XLA_MAX_ROWS >= 128"
GUARD_MSG = "only 128 and 128:256"


@pytest.mark.parametrize(("gate", "hcx", "want"), [
    ("128", None, None), ("128:256", None, None), ("128", "256", None),
    ("127", None, FLOOR_MSG), ("96", None, FLOOR_MSG), ("1", None, FLOOR_MSG), ("96:256", None, FLOOR_MSG),
    ("96", "127", "only 128, 256 and 512"),   # HCX 127 is itself refused now (T95u generalization)
    # passed by the floor, refused as not guard-neutral (T95u fix)
    ("96", "128", GUARD_MSG), ("127", "256", GUARD_MSG), ("112:256", "128", GUARD_MSG), ("160", None, GUARD_MSG)])
def test_floor_below_128_needs_hcx_128(m, monkeypatch, gate, hcx, want):
    """Cycle-4 ruling: N < 128 without HCX >= 128 would send the 128 bucket to the
    prefill-tile path; build refuses it (the floor is checked before guard neutrality, so its
    own message shows)."""
    monkeypatch.setenv(ENV, gate)
    if hcx is not None:
        monkeypatch.setenv(HCX_ENV, hcx)
    if want is None:
        assert build(m)._hcf_max_rows[0] == int(gate.split(":")[0])
    else:
        with pytest.raises(ValueError, match=want):
            m.hcd.decode_fused_max_rows()
        with pytest.raises(ValueError, match=want):
            build(m)


@pytest.mark.parametrize("bad", ["128:143", "256:128", "128:1024", "128:128"])
def test_band_gap_checked_before_guard_neutrality(m, monkeypatch, bad):
    monkeypatch.setenv(ENV, bad)
    with pytest.raises(ValueError, match="M must satisfy"):
        m.hcd.decode_fused_max_rows()


@pytest.mark.parametrize("bad", ["0", "-1", "0:256", "-128:256"])
def test_nonpositive_n_raises_even_with_hcx(m, monkeypatch, bad):
    """N < 1 is malformed on its own, not only through the floor: with HCX 256 set the floor
    does not apply, and N = 0 must still raise rather than read as off (proof-4 j-b497a55c:
    the zero_accepted mutant survived because the floor masked this check)."""
    monkeypatch.setenv(HCX_ENV, "256")
    monkeypatch.setenv(ENV, bad)
    with pytest.raises(ValueError, match=">= 1"):
        m.hcd.decode_fused_max_rows()
    with pytest.raises(ValueError, match=">= 1"):
        build(m)


@pytest.mark.parametrize("tokens", [16, 96, 128, 160, 224, 256, 384, 512, 768, 1024, 2048, 6400])
def test_prefill_tile_tests_are_divisibility_only(m, tokens):
    """T95u generalization: the served-step prefill tests are divisibility alone, equal to the old
    `>=` form for every positive row count (so no value bound on the traced token count)."""
    mod = build(m)
    assert mod._prefill_full_rows(tokens) == (tokens >= 2 * FULL_T and tokens % (2 * FULL_T) == 0)
    assert (tokens % NORM_T == 0) == (tokens >= NORM_T and tokens % NORM_T == 0)


class _NoCompare(int):
    """A row count that fails the test if it is compared (a token-count guard while tracing)."""

    def __le__(self, other):
        raise AssertionError("row count compared")

    __lt__ = __gt__ = __ge__ = __le__


def test_helper_off_never_compares_the_rows(m):
    assert m.hcd.decode_fused_allows(_NoCompare(224), 0) is True
    assert m.hcd.decode_fused_allows(_NoCompare(224), (0, 0)) is True
    with pytest.raises(AssertionError, match="compared"):
        m.hcd.decode_fused_allows(_NoCompare(224), 128)


@pytest.mark.parametrize("tokens", [16, 96, 128, 160, 224, 256, 512, 1024, 2048])
def test_hcx_and_prefill_rows_never_compare_the_rows(m, tokens):
    """T95u generalization: HCX and the served-step _prefill_full_rows decide by divisibility
    only, so no ordering comparison (no value bound) is made on the row count."""
    for limit in (128, 256, 512):
        assert m.hcd.decode_xla_applies(_NoCompare(tokens), limit) == (tokens % (2 * limit) != 0)
    mod = build(m)
    assert mod._prefill_tile_step == 2 * FULL_T
    assert mod._prefill_full_rows(_NoCompare(tokens)) == (tokens % (2 * FULL_T) == 0)


# ---------------------------------------------------------------- 2. off = FINAL 5d368f166
VARIANTS = {
    "tiles": dict(tiles=True),
    "tiles_fused": dict(tiles=True, fused=True),
    "fused_no_tiles": dict(tiles=False, fused=True),
    "no_tiles": dict(tiles=False),
    "fused_no_inject": dict(tiles=True, fused=True, inject=False),
}


@pytest.mark.parametrize(
    ("gate", "variant"),
    [(None, v) for v in sorted(VARIANTS)] + [("128", "tiles"), ("128:256", "tiles"), ("128", "no_tiles"), ("128:256", "no_tiles")],
)
def test_off_or_unarmed_matches_final(m, monkeypatch, variant, gate):
    """Unset: every row count runs exactly FINAL's path (same calls, same bits). Set, on a
    module without the fused ops (FINAL - HC fused; the MTP draft): also FINAL's path."""
    kw = VARIANTS[variant]
    if gate is not None:
        monkeypatch.setenv(ENV, gate)
    new, new_n = build(m, **kw), build(m, seed=5, **kw)
    old = build(m, cls_mod=m.base, like=new, **kw)
    old_n = build(m, cls_mod=m.base, like=new_n, **kw)
    for rows in ROWS:
        got, got_calls, got_n = _run_all(new, new_n, rows)
        ref, ref_calls, ref_n = _run_all(old, old_n, rows)
        assert (got_calls, got_n) == (ref_calls, ref_n), (variant, rows)
        assert _equal(got, ref), (variant, rows)


def test_off_fuses_every_decode_bucket(m, monkeypatch):
    """FINAL's routing (the reference for section 3): fused at every bucket <= 512."""
    mod, nxt = _pair(m, monkeypatch)
    for rows in BUCKETS:
        assert _run_all(mod, nxt, rows)[1:] == (["decode_fused"], ["decode_combine_mix"]), rows


# ---------------------------------------------------------------- 3. on, per bucket
def _fused_side(gate, rows):
    """The gate's own side of a row count (C, the spec): <= N, or (N:M, M = 256) a multiple of 128."""
    lo, _, hi = gate.partition(":")
    return rows <= int(lo) or (bool(hi) and rows % 128 == 0)


def _hcx_for(gate):
    """The floor: N < 128 is only valid with HCX >= 128."""
    return "128" if int(gate.split(":")[0]) < 128 else None


def _check_bucket(m, monkeypatch, gate, rows, hcx=None, **kw):
    """Gate arm vs its references at one row count. Returns 'fused' or 'unfused'."""
    g = _pair(m, monkeypatch, gate=gate, hcx=hcx, **kw)
    got, ga, gb = _run_all(*g, rows)
    fin = _final(m, monkeypatch, like=g, hcx=hcx, **kw)
    ref_f, fa, fb = _run_all(*fin, rows)
    noh = _nohcf(m, monkeypatch, like=g, hcx=hcx, **kw)
    ref_n, na, nb = _run_all(*noh, rows)
    if _fused_side(gate, rows):
        assert (ga, gb) == (fa, fb), (gate, rows, ga, gb)
        assert _equal(got, ref_f), (gate, rows)
        return "fused"
    assert (ga, gb) == (na, nb), (gate, rows, ga, gb)
    assert _equal(got, ref_n), (gate, rows)
    return "unfused"


@pytest.mark.parametrize("rows", BUCKETS)
def test_gate_128_per_bucket(m, monkeypatch, rows):
    """The proposed default: fused at 16 ... 128, FINAL - HC fused at 160 / 192 / 224 / 256 / 512."""
    want = "fused" if rows <= 128 else "unfused"
    assert _check_bucket(m, monkeypatch, "128", rows) == want


@pytest.mark.parametrize("gate", ["128", "128:256"])
@pytest.mark.parametrize("rows", [16, 96, 112, 128, 160, 192, 224, 256, 384, 512])
def test_gate_limits_at_the_required_buckets(m, monkeypatch, gate, rows):
    _check_bucket(m, monkeypatch, gate, rows, hcx=_hcx_for(gate))


@pytest.mark.parametrize("gate", [128])
def test_both_sides_of_the_threshold(m, monkeypatch, gate):
    """N rows: the fused kernel when N has a decode tile; N + 1 and the next bucket: unfused."""
    hcx = _hcx_for(str(gate))
    nxt_bucket = min(b for b in BUCKETS if b > gate)
    for rows in (gate, gate + 1, nxt_bucket):
        _check_bucket(m, monkeypatch, str(gate), rows, hcx=hcx)
    if gate % 16 == 0:  # N itself has a decode tile: the gate keeps it fused
        g = _pair(m, monkeypatch, gate=str(gate), hcx=hcx)
        assert _run_all(*g, gate)[1:] == (["decode_fused"], ["decode_combine_mix"])
    g = _pair(m, monkeypatch, gate=str(gate), hcx=hcx)
    assert "decode_fused" not in _run_all(*g, nxt_bucket)[1]


@pytest.mark.parametrize("gate", ["128:256"])
def test_both_sides_of_the_band_top(m, monkeypatch, gate):
    """128:256: 240 (and 255) unfused, 256 fused (the 256 route kept on the fused kernel)."""
    hi = int(gate.split(":")[1])
    for rows in (hi - 16, hi - 1, hi):
        _check_bucket(m, monkeypatch, gate, rows)
    g = _pair(m, monkeypatch, gate=gate)
    assert _run_all(*g, hi)[1:] == (["decode_fused"], ["decode_combine_mix"])
    assert "decode_fused" not in _run_all(*g, hi - 16)[1]


@pytest.mark.parametrize("rows", BUCKETS)
def test_band_128_256_per_bucket(m, monkeypatch, rows):
    """The car's gate value: fused at 16 ... 128 and at 256 / 512, FINAL - HC fused at 160 / 192 / 224."""
    want = "unfused" if 128 < rows < 256 else "fused"
    assert _check_bucket(m, monkeypatch, "128:256", rows) == want


def test_unfused_route_differs_from_fused(m, monkeypatch):
    """The bit comparison can see a wrong route: at 224 FINAL (fused) and FINAL - HC fused differ."""
    g = _pair(m, monkeypatch, gate="128")
    fin = _final(m, monkeypatch, like=g)
    noh = _nohcf(m, monkeypatch, like=g)
    for rows in (96, 224):
        assert not _equal(_run_all(*fin, rows)[0][0], _run_all(*noh, rows)[0][0]), rows


def test_which_unfused_path(m, monkeypatch):
    """Which unfused path at FINAL's tiles 128 / 128 / 256. Gate 128, HCX unset: 160 / 192 /
    224 the XLA path (no tile op), 256 the norm and up-epilogue tiles. Gate 128 + HCX 256: 256 the XLA path. Gate
    128:256: 256 fused."""
    g = _pair(m, monkeypatch, gate="128")
    for rows in (160, 192, 224):
        assert _run_all(*g, rows)[1:] == ([], []), rows
    assert _run_all(*g, 256)[1:] == (["norm_tile", "up_tile"], ["norm_tile", "up_tile"])
    g = _pair(m, monkeypatch, gate="128", hcx="256")
    assert _run_all(*g, 256)[1:] == ([], [])
    g = _pair(m, monkeypatch, gate="128:256")
    assert _run_all(*g, 256)[1:] == (["decode_fused"], ["decode_combine_mix"])


def test_combine_mix_prologue_follows_the_next_hcs_gate(m, monkeypatch):
    """The combine_mix prologue is the next HC's fused op: gated by the next HC's limit."""
    a, _ = _pair(m, monkeypatch, gate=None)
    _, b = _pair(m, monkeypatch, gate="128")
    for rows, want in ((96, ["decode_combine_mix"]), (224, [])):
        a.calls.clear()
        b.calls.clear()
        _, residual = a.mix(_x(rows))
        a.combine_mix(_block(rows), residual, b)
        assert b.calls == want, rows


def test_no_inject_module(m, monkeypatch):
    for rows in (96, 224):
        _check_bucket(m, monkeypatch, "128", rows, inject=False)


# ---------------------------------------------------------------- 4. with HCX
@pytest.mark.parametrize(("gate", "rows"), [("128", 256), ("128", 224), ("128", 96), ("128", 128), ("128", 512),
                                           ("128:256", 256), ("128:256", 224), ("128:256", 128)])
def test_composes_with_hcx(m, monkeypatch, gate, rows):
    """Gate + HCX 256: > N rows equal FINAL - HC fused + HCX: no tile op (the XLA path) up to
    the HCX limit, FINAL - HC fused's tile path above it (512); <= N keeps the fused kernel (it
    precedes HCX). proof-6 j-4ee90701 caught the old expectation of no tile op at 512."""
    got = _check_bucket(m, monkeypatch, gate, rows, hcx="256")
    g = _pair(m, monkeypatch, gate=gate, hcx="256")
    calls = _run_all(*g, rows)[1:]
    if got == "fused":
        assert calls == (["decode_fused"], ["decode_combine_mix"])
    elif rows <= 256:
        assert calls == ([], [])
    else:
        assert calls == (["full", "combine_tile", "combine_tile"], ["full"])


# ---------------------------------------------------------------- 5. guards
def test_no_row_check_on_unarmed_modules(m, monkeypatch):
    """FINAL - HC fused and the MTP draft (disable_decode_fused): the gate never reads rows."""

    def boom(*_):
        raise AssertionError("decode_fused_allows evaluated on a module without fused ops")

    noh = _pair(m, monkeypatch, gate="128", fused=False)
    draft = _pair(m, monkeypatch, gate="128", fused=True)
    for mod in draft:
        mod.disable_decode_fused()
    monkeypatch.setattr(m.hc, "decode_fused_allows", boom)
    for pair in (noh, draft):
        for rows in (96, 224):
            _run_all(*pair, rows)
    assert noh[0]._hcf_max_rows == (128, 0)
    # The patch point is live: an armed module does reach it.
    armed = _pair(m, monkeypatch, gate="128")
    with pytest.raises(AssertionError, match="without fused ops"):
        armed[0].mix(_x(96))


def test_limit_is_read_once_at_build(m, monkeypatch):
    g = _pair(m, monkeypatch, gate="128")
    monkeypatch.setenv(ENV, "512")
    assert "decode_fused" not in _run_all(*g, 224)[1]
    monkeypatch.delenv(ENV)
    g = _pair(m, monkeypatch, gate=None)
    monkeypatch.setenv(ENV, "128")
    assert _run_all(*g, 224)[1] == ["decode_fused"]


def test_only_2d_inputs(m, monkeypatch):
    mod, _ = _pair(m, monkeypatch, gate="128")
    assert mod._decode_fused(_x(96))
    assert not mod._decode_fused(_x(96).unsqueeze(0))
    assert not mod._decode_fused(_x(224))
    assert mod._decode_rows(96) and mod._decode_rows(128)
    assert not mod._decode_rows(129) and not mod._decode_rows(224)
    band, _ = _pair(m, monkeypatch, gate="128:256")
    assert band._decode_rows(256) and band._decode_rows(128) and not band._decode_rows(224)


def test_engagement_counter_follows_the_gate(m, monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp import hc_engagement as E

    E.reset()
    try:
        E.record_built(True)
        assert E.count_step(224, 0) is True and E.count_step(96, 0) is True
        monkeypatch.setenv(ENV, "128")
        assert E.count_step(224, 0) is False and E.count_step(160, 0) is False
        assert E.count_step(128, 0) is True and E.count_step(96, 0) is True
        assert E.count_step(256, 0) is False
        monkeypatch.setenv(ENV, "128:256")
        assert E.count_step(256, 0) is True and E.count_step(224, 0) is False
        assert E.snapshot() == {"modules": 1, "up_t": 1, "steps": 9, "eligible": 5}
    finally:
        E.reset()


# ---------------------------------------------------------------- 6. compile key
def _factors(m):
    path = os.path.join(m.root, "vllm_torchtpu", "compilation", "tpu_compiler.py")
    src = open(path).read()
    tree = ast.parse(src)
    ignored = next(
        ast.literal_eval(n.value) for n in tree.body
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "_TPU_COMPILE_ENV_IGNORED" for t in n.targets)
    )
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_tpu_compile_env_factors")
    ns = {"envs": m.envs, "_TPU_COMPILE_ENV_IGNORED": ignored, "logger": logging.getLogger("hcgate"),
          "os": os, "Any": object}
    exec(ast.get_source_segment(src, fn), ns)
    return ns["_tpu_compile_env_factors"]


def test_factors_unchanged_when_unset_and_keyed_when_set(m, monkeypatch):
    factors = _factors(m)
    off = factors()
    assert ENV not in off
    monkeypatch.setenv(ENV, "")
    assert factors() == off
    monkeypatch.setenv(ENV, "128")
    on = factors()
    assert on == {**off, ENV: "128"}
    monkeypatch.setenv(ENV, "160")
    assert factors()[ENV] == "160"
    monkeypatch.setenv(ENV, "128:256")
    assert factors()[ENV] == "128:256"
    monkeypatch.setenv(HCX_ENV, "256")
    both = factors()
    assert both[ENV] == "128:256" and both[HCX_ENV] == "256"


def test_not_registered_in_envs(m):
    """Registered variables are hashed set or not, which would move every compile key."""
    assert ENV not in m.envs.environment_variables


def test_compile_hash_keys_the_gate_only_when_set():
    """compute_tpu_compilation_hash (the real one; CDK CPU image only)."""
    if hcx_load.mode() != "image":
        pytest.skip("needs vLLM (the CDK CPU image)")
    sys.path.insert(0, os.path.join(hcx_load.TREE, "tests", "compilation"))
    import test_tpu_compile_key_registration as reg

    off = reg._hash_with({ENV: None})
    assert reg._hash_with({ENV: ""}) == off
    on = reg._hash_with({ENV: "128"})
    assert on != off
    assert reg._hash_with({ENV: "160"}) not in (on, off)
    assert reg._hash_with({ENV: "128:256"}) not in (on, off)

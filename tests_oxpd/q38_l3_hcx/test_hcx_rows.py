# SPDX-License-Identifier: Apache-2.0
"""HCX on PD's FINAL tree (q38 L3-HCX; port of q38mm 20e6fcf23, MM DSTOCK HCX).

VLLM_TPU_HC_XLA_MAX_ROWS (unset = off): an HC of <= N rows on a module that has
prefill-tile ops skips every one of them and runs GatedResidual.mix's XLA path.
The real GatedResidual is driven (hcx_load: real packages in the CPU image,
package stubs plus vLLM v0.31.0's common file on a Mac), with recording stand-ins
for the tile and fused ops at small dims. Checked:

1. hc_decode helpers: off by default, the limit, bad values raise.
2. Off: every path, call and output equal to FINAL 5d368f166's file at every
   row count, with tile ops, fused decode ops, fused prefill ops or none.
3. On: no tile op at <= N rows (128 / 256 with N = 256), and the outputs equal
   the explicit XLA path bit for bit; 96 rows (the placebo) is unchanged; > N
   keeps the tiles; the combine and the fused prefill combine_mix are covered.
4. The fused decode branch keeps precedence; no token-count read on modules
   without tile ops (the MTP draft); 2-D only; the injection gate is still
   computed inside mix (block_inject_weight's first use); read once at build.
5. tpu_compiler._tpu_compile_env_factors keys the value only when set; the
   variable is not registered in envs.py.

Run with --noconftest on a Mac; mutation-checked by q38_l3_hcx_mutrun.py.
"""

import ast
import logging
import os

import hcx_load
import pytest
import torch

ENV = "VLLM_TPU_HC_XLA_MAX_ROWS"
HC, H, LR = 4, 8, 4
NORM_T, UP_T, FULL_T = 128, 128, 256
ROWS = (1, 16, 64, 96, 128, 160, 224, 256, 384, 512, 1024)


@pytest.fixture
def m(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv("VLLM_TPU_DECODE_TP_AR", raising=False)
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


def build(m, *, tiles=True, inject=True, fused=False, fused_mix=False, full_tile=FULL_T,
          seed=0, cls_mod=None, like=None):
    """A GatedResidual (tiles are attached after construction: building the real
    Pallas ops needs the TPU adapter). ``like``: copy that module's weights."""
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
        _attach_tiles(mod, full_tile)
    if fused:
        _attach_fused(mod)
    if fused_mix:
        _attach_fused_mix(mod, full_tile)
    return mod


def _rec(mod, name, fn):
    def op(*args):
        mod.calls.append(name)
        return fn(*args)

    return op


def _const(rows, value, width=H):
    return torch.full((rows, width), value, dtype=torch.bfloat16)


def _attach_tiles(mod, full_tile):
    mod.prefill_norm_tile, mod.prefill_up_tile, mod.prefill_full_tile = NORM_T, UP_T, full_tile
    mod._prefill_tile_step = 2 * full_tile  # the served default (no ODD_TILES / FUSED_MIX)
    mod._prefill_norm_op = _rec(mod, "norm_tile", lambda x, w: (x.float() * 0.5).to(x.dtype))
    mod._prefill_up_op = _rec(mod, "up_tile", lambda g, n, w: _const(g.shape[0], 0.25))
    mod._prefill_full_op = _rec(mod, "full", lambda x, w, d, u: (_const(x.shape[0], 0.125), x))
    if hasattr(mod, "block_inject_weight"):
        mod._prefill_combine_op = _rec(mod, "combine_tile", lambda r, b, g: (r.float() + 1).to(r.dtype))


def _attach_fused(mod):
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


def _attach_fused_mix(mod, full_tile):
    mod._prefill_tile_step = full_tile  # TPU_QWEN4_HC_PREFILL_FUSED_MIX allows any count >= 2
    mod._prefill_inj_mix_op = _rec(
        mod, "prefill_inj_mix", lambda x, *w: (_const(x.shape[0], 0.75), torch.full((x.shape[0], HC), 0.25))
    )
    mod._prefill_combine_mix_op = _rec(
        mod, "prefill_combine_mix",
        lambda h, b, g, *w: (_const(h.shape[0], 0.75), torch.full((h.shape[0], HC), 0.25), h),
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


# ---------------------------------------------------------------- 1. helpers
def test_helper_off_by_default(m, monkeypatch):
    assert m.hcd.decode_xla_max_rows() == 0
    assert not any(m.hcd.decode_xla_applies(t) for t in (1, 64, 128, 256))
    monkeypatch.setenv(ENV, "  ")
    assert m.hcd.decode_xla_max_rows() == 0
    assert not m.hcd.decode_xla_applies(128)


def test_helper_limit(m, monkeypatch):
    """q38 L3-HCGATE (T95u generalization): decided as tokens % (2 x limit) != 0, which over the
    decode bucket ladder is tokens <= limit for every row count that has a tile op."""
    monkeypatch.setenv(ENV, "256")
    assert m.hcd.decode_xla_max_rows() == 256
    assert all(m.hcd.decode_xla_applies(t) for t in (1, 96, 128, 255, 256))
    assert not any(m.hcd.decode_xla_applies(t) for t in (0, 512, 1024, 2048))
    ladder = [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 512, 1024, 2048]
    for limit in (128, 256, 512):
        for t in ladder:
            if t % 128 == 0:   # the rows that have tile ops
                assert m.hcd.decode_xla_applies(t, limit) == (t <= limit), (limit, t)


def test_helper_explicit_limit_wins_over_env(m, monkeypatch):
    assert m.hcd.decode_xla_applies(128, 256) and not m.hcd.decode_xla_applies(128, 0)
    monkeypatch.setenv(ENV, "128")
    assert m.hcd.decode_xla_applies(256, 256) and not m.hcd.decode_xla_applies(256)


@pytest.mark.parametrize(("bad", "why"), [
    ("0", "must be >= 1"), ("-1", "must be >= 1"), ("x", "must be an integer"), ("1.5", "must be an integer"),
    ("256.0", "must be an integer"), ("256rows", "must be an integer"),
    ("64", "only 128, 256 and 512"), ("160", "only 128, 256 and 512"), ("4096", "only 128, 256 and 512")])
def test_helper_bad_values(m, monkeypatch, bad, why):
    """Each bad value is refused for its own reason (full-8 j-f4ccd3ca: with only `raises(ValueError)` the
    128 / 256 / 512 allowlist masked the integer and >= 1 checks, so zero_accepted / float_accepted survived)."""
    monkeypatch.setenv(ENV, bad)
    with pytest.raises(ValueError, match=why):
        m.hcd.decode_xla_max_rows()
    with pytest.raises(ValueError, match=why):   # the refusal contract: GatedResidual build reads it once and raises
        build(m)


# ---------------------------------------------------------------- 2. off = FINAL 5d368f166
VARIANTS = {
    "tiles": dict(tiles=True),
    "tiles_fused": dict(tiles=True, fused=True),
    "tiles_fused_mix": dict(tiles=True, fused_mix=True),
    "tiles_full128": dict(tiles=True, full_tile=128),
    "no_tiles": dict(tiles=False),
    "no_inject": dict(tiles=True, inject=False),
}


@pytest.mark.parametrize(
    ("hcx", "variant"), [(None, v) for v in sorted(VARIANTS)] + [("256", "no_tiles"), ("512", "no_tiles")]
)
def test_off_or_no_tiles_matches_final(m, monkeypatch, variant, hcx):
    """Unset: every row count runs exactly FINAL's path (same calls, same bits). Set, on a
    module without tile ops (the MTP draft): also exactly FINAL's path."""
    kw = VARIANTS[variant]
    if hcx is not None:
        monkeypatch.setenv(ENV, hcx)
    new, new_n = build(m, **kw), build(m, seed=5, **kw)
    old = build(m, cls_mod=m.base, like=new, **kw)
    old_n = build(m, cls_mod=m.base, like=new_n, **kw)
    for rows in ROWS:
        got, got_calls, got_n = _run_all(new, new_n, rows)
        ref, ref_calls, ref_n = _run_all(old, old_n, rows)
        assert (got_calls, got_n) == (ref_calls, ref_n), (variant, rows)
        assert _equal(got, ref), (variant, rows)


def test_off_routing_at_the_served_tiles(m):
    """FINAL's decode tiles (norm 128 / up 128 / full 256): 128 and 256 rows take the
    norm and up-epilogue tiles, 512 the full mix and the combine tile, 96 none."""
    mod, nxt = build(m), build(m, seed=5)
    expect = {96: ([], []), 128: (["norm_tile", "up_tile"], ["norm_tile", "up_tile"]),
              256: (["norm_tile", "up_tile"], ["norm_tile", "up_tile"]),
              512: (["full", "combine_tile", "combine_tile"], ["full"])}
    for rows, calls in expect.items():
        _, a, b = _run_all(mod, nxt, rows)
        assert (a, b) == calls, rows


# ---------------------------------------------------------------- 3. on
def _xla_twin(m, mod, **kw):
    """The same weights with no tile op: the explicit XLA path."""
    kw = {k: v for k, v in kw.items() if k not in ("tiles", "fused_mix", "full_tile")}
    return build(m, tiles=False, like=mod, **kw)


@pytest.mark.parametrize("rows", [128, 256])
def test_on_routes_128_and_256_to_xla_bit_equal(m, monkeypatch, rows):
    monkeypatch.setenv(ENV, "256")
    mod, nxt = build(m), build(m, seed=5)
    twin, twin_n = _xla_twin(m, mod), _xla_twin(m, nxt)
    got, a, b = _run_all(mod, nxt, rows)
    assert (a, b) == ([], [])
    ref, _, _ = _run_all(twin, twin_n, rows)
    assert _equal(got, ref)
    # The XLA path itself is FINAL's: same bits as 5d368f166's module without tiles.
    old = build(m, cls_mod=m.base, tiles=False, like=mod)
    old_n = build(m, cls_mod=m.base, tiles=False, like=nxt)
    assert _equal(got, _run_all(old, old_n, rows)[0])
    # And it differs from the tile path, so the comparison can see a wrong route.
    monkeypatch.delenv(ENV)
    off, off_n = build(m, like=mod), build(m, like=nxt)
    assert not _equal(got[0], _run_all(off, off_n, rows)[0][0])


@pytest.mark.parametrize("rows", [1, 16, 64, 96, 160, 224])
def test_on_is_a_placebo_where_no_tile_applies(m, monkeypatch, rows):
    mod, nxt = build(m), build(m, seed=5)
    ref, ra, rb = _run_all(mod, nxt, rows)
    monkeypatch.setenv(ENV, "256")
    on, on_n = build(m, like=mod), build(m, like=nxt)
    got, ga, gb = _run_all(on, on_n, rows)
    assert (ga, gb) == (ra, rb) == ([], [])
    assert _equal(got, ref)


def test_on_leaves_rows_above_the_limit_on_the_tiles(m, monkeypatch):
    monkeypatch.setenv(ENV, "128")
    mod, nxt = build(m), build(m, seed=5)
    assert _run_all(mod, nxt, 128)[1:] == ([], [])
    assert _run_all(mod, nxt, 256)[1:] == (["norm_tile", "up_tile"], ["norm_tile", "up_tile"])
    monkeypatch.setenv(ENV, "256")
    mod, nxt = build(m), build(m, seed=5)
    assert _run_all(mod, nxt, 512)[1:] == (["full", "combine_tile", "combine_tile"], ["full"])


def test_on_covers_the_full_mix_and_the_combine_tile(m, monkeypatch):
    """A limit reaching the full-mix rows (512 at the served tile) skips the full mix and
    the gated-residual combine tile too, and matches the explicit XLA path."""
    monkeypatch.setenv(ENV, "512")
    mod, nxt = build(m), build(m, seed=5)
    got, a, b = _run_all(mod, nxt, 512)
    assert (a, b) == ([], [])
    assert _equal(got, _run_all(_xla_twin(m, mod), _xla_twin(m, nxt), 512)[0])


def test_on_covers_the_full_mix_at_a_128_full_tile(m, monkeypatch):
    monkeypatch.setenv(ENV, "256")
    mod, nxt = build(m, full_tile=128), build(m, seed=5, full_tile=128)
    assert _run_all(mod, nxt, 256)[1:] == ([], [])
    monkeypatch.delenv(ENV)
    mod, nxt = build(m, full_tile=128), build(m, seed=5, full_tile=128)
    assert _run_all(mod, nxt, 256)[1:] == (["full", "combine_tile", "combine_tile"], ["full"])


def test_on_covers_the_fused_prefill_mix_and_combine_mix(m, monkeypatch):
    mod, nxt = build(m, fused_mix=True), build(m, seed=5, fused_mix=True)
    assert _run_all(mod, nxt, 512)[1:] == (["prefill_inj_mix", "combine_tile"], ["prefill_combine_mix"])
    monkeypatch.setenv(ENV, "512")
    mod, nxt = build(m, fused_mix=True), build(m, seed=5, fused_mix=True)
    got, a, b = _run_all(mod, nxt, 512)
    assert (a, b) == ([], [])
    assert _equal(got, _run_all(_xla_twin(m, mod), _xla_twin(m, nxt), 512)[0])


def test_on_combine_alone_when_it_is_the_only_tile_op(m, monkeypatch):
    """A module whose only tile op is the combine still counts as tiled."""
    monkeypatch.setenv(ENV, "512")
    mod = build(m)
    mod._prefill_norm_op = mod._prefill_up_op = mod._prefill_full_op = None
    x = _x(512)
    _, residual = mod.mix(x)
    mod.calls.clear()
    mod.combine(_block(512), residual)
    assert mod.calls == []


# ---------------------------------------------------------------- 4. precedence and guards
def test_fused_decode_branch_keeps_precedence(m, monkeypatch):
    monkeypatch.setenv(ENV, "256")
    mod, nxt = build(m, fused=True), build(m, seed=5, fused=True)
    for rows in (16, 128, 256):
        mod.calls.clear()
        nxt.calls.clear()
        mixed, residual = mod.mix(_x(rows))
        assert mod.calls == ["decode_fused"], rows
        mod.combine_mix(_block(rows), residual, nxt)
        assert nxt.calls == ["decode_combine_mix"], rows


def test_no_row_check_on_modules_without_tile_ops(m, monkeypatch):
    """The MTP draft's HC has no tile ops: the row check must not run there."""
    monkeypatch.setenv(ENV, "256")

    def boom(*_):
        raise AssertionError("decode_xla_applies evaluated on a module without tile ops")

    draft, nxt = build(m, tiles=False), build(m, seed=5, tiles=False)
    monkeypatch.setattr(m.hc, "decode_xla_applies", boom)
    _run_all(draft, nxt, 128)
    assert draft._hcx_max_rows == 256
    # The patch point is live: a tiled module does reach it.
    with pytest.raises(AssertionError, match="without tile ops"):
        build(m).mix(_x(128))


def test_no_row_check_when_unset(m, monkeypatch):
    def boom(*_):
        raise AssertionError("row count compared with HCX unset")

    monkeypatch.setattr(m.hc, "decode_xla_applies", boom)
    for rows in (96, 128, 512):
        _run_all(build(m), build(m, seed=5), rows)


def test_only_2d_inputs(m, monkeypatch):
    monkeypatch.setenv(ENV, "256")
    mod = build(m)
    assert mod._hcx_xla(_x(128))
    assert not mod._hcx_xla(_x(128).unsqueeze(0))
    assert not mod._hcx_xla(_x(512))


def test_injection_gate_is_computed_inside_mix(m, monkeypatch):
    """block_inject_weight's first use stays inside mix on the HCX path: mix returns the
    [T, hc] FP32 gate, not the normalized input (the _residual docstring)."""
    monkeypatch.setenv(ENV, "256")
    mod = build(m)
    _, (hyper, gate) = mod.mix(_x(128))
    assert gate.shape == (128, HC) and gate.dtype == torch.float32
    _, (hyper, normed) = build(m, inject=False).mix(_x(128))
    assert normed.shape == (128, HC * H)


def test_limit_is_read_once_at_build(m, monkeypatch):
    monkeypatch.setenv(ENV, "256")
    mod, nxt = build(m), build(m, seed=5)
    monkeypatch.delenv(ENV)
    assert _run_all(mod, nxt, 128)[1:] == ([], [])
    mod, nxt = build(m), build(m, seed=5)
    monkeypatch.setenv(ENV, "256")
    assert _run_all(mod, nxt, 128)[1:] == (["norm_tile", "up_tile"], ["norm_tile", "up_tile"])
    monkeypatch.setenv(ENV, "zero")
    with pytest.raises(ValueError):
        build(m)


# ---------------------------------------------------------------- 5. compile key
def _factors(m):
    path = os.path.join(m.root, "vllm_torchtpu", "compilation", "tpu_compiler.py")
    src = open(path).read()
    tree = ast.parse(src)
    ignored = next(
        ast.literal_eval(n.value) for n in tree.body
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "_TPU_COMPILE_ENV_IGNORED" for t in n.targets)
    )
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_tpu_compile_env_factors")
    ns = {"envs": m.envs, "_TPU_COMPILE_ENV_IGNORED": ignored, "logger": logging.getLogger("hcx"),
          "os": os, "Any": object}
    exec(ast.get_source_segment(src, fn), ns)
    return ns["_tpu_compile_env_factors"]


def test_factors_unchanged_when_unset_and_keyed_when_set(m, monkeypatch):
    factors = _factors(m)
    off = factors()
    assert ENV not in off
    monkeypatch.setenv(ENV, "")
    assert factors() == off
    monkeypatch.setenv(ENV, "256")
    on = factors()
    assert on == {**off, ENV: "256"}
    monkeypatch.setenv(ENV, "96")
    assert factors()[ENV] == "96"


def test_not_registered_in_envs(m):
    """Registered variables are hashed set or not, which would move every compile key."""
    assert ENV not in m.envs.environment_variables

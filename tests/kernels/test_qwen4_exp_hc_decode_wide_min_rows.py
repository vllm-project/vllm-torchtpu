# SPDX-License-Identifier: Apache-2.0
"""TPU_QWEN4_HC_DECODE_WIDE_MIN_ROWS (q38 HCK floor key): hc_decode_wide mode 1 below its 160-row floor (64-144 rows).

Fast checks: flag reads / refusals / compile key; wide_applies with a floor (rows >= floor, a multiple of 16, every row block
>= two 32-row pieces; unset = 160-511 exactly as before); the adapter's refusals, op tags (l<rows>) and engagement line; the
flag-off engagement row (WIDE=1, floor unset: 96 / 112 rows still trace hc_decode.py's kernel, no _w1*); dispatch (the floor
is passed on only below 160, so calls at >= 160 are unchanged); the kernel's mode-1-only refusal and trace line; the kernel
body independent of the floor at >= 160 (== the pinned K3b.2 copy); and small interpret-mode numerics at 64 / 96 / 112 / 144
rows (DMAs on wait / eager, NaN for uninitialised memory, race detection): W1 bitwise hc_decode_mix (the served kernel), the
schedule bitwise W1 / W1F. Full-size legs, deviceless compiles and code mutants run in the HCK CPU proof."""

import hashlib
import importlib.machinery
import importlib.util
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import hc_decode_wide
from vllm_torchtpu.kernels.qwen4_exp.hc_decode import hc_decode_mix
from vllm_torchtpu.kernels.qwen4_exp.hc_decode_wide import (
    FLOOR_CHOICES,
    MAX_TOKENS,
    MIN_TOKENS,
    hc_decode_mix_wide,
    hc_decode_mix_wide_auto,
    row_blocks,
    wide_applies,
)

FLAG = "TPU_QWEN4_HC_DECODE_WIDE_MIN_ROWS"
WIDE = "TPU_QWEN4_HC_DECODE_WIDE"
FOLD = "TPU_QWEN4_HC_DECODE_INJECT_FOLD"
SCHED = "TPU_QWEN4_HC_DECODE_W1_SCHED"
BRANCHES, HIDDEN, LOWRANK, EPS = 4, 2560, 320, 1e-6
WIDTH = BRANCHES * HIDDEN
K3B2_REF = Path(__file__).parent / "q38_ref" / "hc_decode_wide_k3b2_d26816eaf6.py.txt"
K3B2_SHA256 = "2b89a458da13108ca66b17c98ec8ac2360b598cd957ebfd47f1e6ea6e5fa3b39"
CHOICES = ["64", "80", "96", "112", "128", "144"]
_ENV_KEYS = (FLAG, WIDE, FOLD, SCHED, "TPU_QWEN4_HC_DECODE_COMPARE", "TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID",
             "TPU_QWEN4_HC_DECODE_TILE_ROWS")


def _set_env(monkeypatch, **values):
    for key in _ENV_KEYS:
        envs.__dict__.pop(key, None)
        monkeypatch.delenv(key, raising=False)
    for key, value in values.items():
        monkeypatch.setenv(key, str(value))


# ---------------------------------------------------------------- flag
@pytest.mark.parametrize("value,expected", [(None, ""), ("", "")] + [(c, c) for c in CHOICES])
def test_flag_reads(monkeypatch, value, expected):
    _set_env(monkeypatch)
    if value is not None:
        monkeypatch.setenv(FLAG, value)
    assert envs.environment_variables[FLAG]() == expected
    assert getattr(envs, FLAG) == expected


@pytest.mark.parametrize("value", ["0", "1", "48", "72", "160", "true"])
def test_flag_refuses_other_values(monkeypatch, value):
    _set_env(monkeypatch)
    monkeypatch.setenv(FLAG, value)
    with pytest.raises(ValueError):
        envs.environment_variables[FLAG]()


def test_choices_match_the_kernel():
    assert tuple(int(c) for c in CHOICES) == FLOOR_CHOICES


def test_flag_is_a_compile_key(monkeypatch):
    from vllm_torchtpu.compilation import tpu_compiler

    assert FLAG not in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    _set_env(monkeypatch)
    unset = tpu_compiler._tpu_compile_env_factors()
    monkeypatch.setenv(FLAG, "96")
    one = tpu_compiler._tpu_compile_env_factors()
    assert unset[FLAG] == ""
    assert {k for k in one if one[k] != unset.get(k)} == {FLAG}


# ---------------------------------------------------------------- gate
def test_wide_applies_unset_is_unchanged():
    assert (MIN_TOKENS, MAX_TOKENS) == (160, 511)
    assert [t for t in range(1, 1100) if wide_applies(t)] == list(range(160, 512, 16))
    assert [t for t in range(1, 1100) if wide_applies(t, None)] == list(range(160, 512, 16))


@pytest.mark.parametrize("floor", FLOOR_CHOICES)
def test_wide_applies_with_a_floor(floor):
    # Every row block of 64-144 rows holds >= two 32-row pieces (checked by hand: 64 = 2, 80 = 2 + 16, 96 = 3,
    # 112 = 3 + 16, 128 = 4, 144 = 2 + (2 + 16)), so the admitted rows are floor .. 496 in steps of 16.
    assert [t for t in range(1, 1100) if wide_applies(t, floor)] == list(range(floor, 512, 16))
    for t in range(floor, MIN_TOKENS, 16):
        assert all(rows // 32 >= 2 for _, rows in row_blocks(t)), t


@pytest.mark.parametrize("floor", [0, 16, 48, 72, 160, "96"])
def test_wide_applies_refuses_other_floors(floor):
    with pytest.raises(ValueError, match="floor"):
        wide_applies(112, floor)


def test_piece_rule_is_enforced(monkeypatch):
    """A floor whose rows would leave a row block with one 32-row piece is not admitted (48 = 1 piece + 16)."""
    monkeypatch.setattr(hc_decode_wide, "FLOOR_CHOICES", (32, 48) + FLOOR_CHOICES)
    assert not hc_decode_wide.wide_applies(48, 48) and not hc_decode_wide.wide_applies(32, 32)
    assert hc_decode_wide.wide_applies(64, 48) and hc_decode_wide.wide_applies(64, 32)


# ---------------------------------------------------------------- adapter
class _FakeOp:
    def __init__(self, name, core):
        self.name, self.core = name, core

    def register_fake(self, fn):
        return fn


def _ops(monkeypatch, **env):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    _set_env(monkeypatch, TPU_QWEN4_HC_DECODE_TILE_ROWS=512, **env)
    monkeypatch.setattr(vllm_adapter.pallas, "jax_op", lambda name, core: _FakeOp(name, core))
    ops = {}
    for kind, build in (("mix", vllm_adapter.build_decode_mix_op), ("combine", vllm_adapter.build_decode_combine_mix_op)):
        for inject in (True, False):
            ops[kind, inject] = build.__wrapped__(BRANCHES, HIDDEN, EPS, inject)
    return ops


@pytest.mark.parametrize("env", [{FLAG: 96}, {FLAG: 96, WIDE: 0}, {FLAG: 96, WIDE: 2}, {FLAG: 96, WIDE: 3}])
def test_adapter_refuses_the_floor_outside_w1(monkeypatch, env):
    with pytest.raises(ValueError, match="TPU_QWEN4_HC_DECODE_WIDE_MIN_ROWS=96 needs TPU_QWEN4_HC_DECODE_WIDE=1"):
        _ops(monkeypatch, **env)


@pytest.mark.parametrize("base,tag", [({WIDE: 1}, "_w1"), ({WIDE: 1, FOLD: 1}, "_w1f"), ({WIDE: 1, SCHED: 1}, "_w1s"),
                                      ({WIDE: 1, FOLD: 1, SCHED: 1}, "_w1fs")])
def test_op_names_tag_the_floor(monkeypatch, base, tag):
    off = {k: o.name for k, o in _ops(monkeypatch, **base).items()}
    offe = {k: o.name for k, o in _ops(monkeypatch, **base, **{FLAG: ""}).items()}
    on = {k: o.name for k, o in _ops(monkeypatch, **base, **{FLAG: 96}).items()}
    assert offe == off
    for key in off:
        assert off[key].endswith(f"_t512{tag}_v1"), off[key]
        assert on[key] == off[key].replace(f"{tag}_v1", f"{tag}l96_v1"), (on[key], off[key])


def test_engagement_line(monkeypatch, capsys):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    vllm_adapter._log_wide_floor.cache_clear()
    _ops(monkeypatch, **{WIDE: 1, FOLD: 1, FLAG: 96})
    out = capsys.readouterr().out
    assert out.count("q38 HCK: TPU_QWEN4_HC_DECODE_WIDE_MIN_ROWS=96 engaged") == 1 and "op tag _w1fl96" in out
    vllm_adapter._log_wide_floor.cache_clear()
    _ops(monkeypatch, **{WIDE: 1, FOLD: 1})
    assert "WIDE_MIN_ROWS" not in capsys.readouterr().out


def _core_args(rows, kind, inject):
    s = jax.ShapeDtypeStruct
    bf = jnp.bfloat16
    args = [s((rows, WIDTH), bf)]
    if kind == "combine":
        args += [s((rows, HIDDEN), bf), s((rows, BRANCHES), jnp.float32)]
    args += [s((WIDTH,), bf), s((LOWRANK, WIDTH), bf), s((WIDTH, LOWRANK), bf)]
    if inject:
        args.append(s((BRANCHES, WIDTH), bf))
    return args


def _mask(txt):
    txt = re.sub(r"\S+\.py(?:\.txt)?:\d+", "<src>", txt)
    return re.sub(r"0x[0-9a-f]+", "<addr>", txt)


def _core_trace(op, rows, kind, inject):
    return _mask(str(jax.make_jaxpr(op.core)(*_core_args(rows, kind, inject))))


@pytest.mark.parametrize("rows", [96, 112])
def test_flag_off_engagement_row(monkeypatch, rows):
    """WIDE=1 + INJ, floor unset: 96 / 112-row calls trace hc_decode.py's kernel exactly as WIDE=0 does (the default keeps
    160); with the floor they trace the wide kernel (_w1f / _w1)."""
    served = _ops(monkeypatch)
    wide = _ops(monkeypatch, **{WIDE: 1, FOLD: 1})
    floor = _ops(monkeypatch, **{WIDE: 1, FOLD: 1, FLAG: 64})
    for (kind, inject), op in wide.items():
        t_wide = _core_trace(op, rows, kind, inject)
        assert t_wide == _core_trace(served[kind, inject], rows, kind, inject), (kind, inject)
        assert "_w1" not in t_wide
        t_floor = _core_trace(floor[kind, inject], rows, kind, inject)
        assert t_floor != t_wide
        assert ("_w1f" in t_floor) == inject and (re.search(r"_w1\b", t_floor) is not None) == (not inject)


@pytest.mark.parametrize("rows", [160, 288, 496])
def test_floor_leaves_rows_from_160_unchanged(monkeypatch, rows):
    wide = _ops(monkeypatch, **{WIDE: 1, FOLD: 1, SCHED: 1})
    floor = _ops(monkeypatch, **{WIDE: 1, FOLD: 1, SCHED: 1, FLAG: 64})
    for key, op in wide.items():
        assert _core_trace(floor[key], rows, *key) == _core_trace(op, rows, *key), key


# ---------------------------------------------------------------- dispatch
def _specs(tokens, inject=True, prologue=True, hidden=HIDDEN, lowrank=LOWRANK):
    s = jax.ShapeDtypeStruct
    bf = jnp.bfloat16
    width = BRANCHES * hidden
    args = [s((tokens, width), bf), s((width,), bf), s((lowrank, width), bf)]
    args += [s((width, lowrank), bf), s((BRANCHES, width), bf) if inject else None]
    extra = {}
    if prologue:
        extra = dict(block=s((tokens, hidden), bf), prev_injection=s((tokens, BRANCHES), jnp.float32))
    return args, extra


@pytest.mark.parametrize("tokens,floor,want", [(48, 64, "served"), (64, 64, "wide"), (80, 96, "served"),
                                               (96, 96, "wide"), (112, 96, "wide"), (144, 144, "wide"),
                                               (128, 144, "served"), (104, 96, "served"), (16, 64, "served")])
def test_auto_takes_the_floor(tokens, floor, want, monkeypatch):
    calls = []
    monkeypatch.setattr(hc_decode_wide, "hc_decode_mix", lambda *a, **k: calls.append(("served", k)))
    monkeypatch.setattr(hc_decode_wide, "hc_decode_mix_wide", lambda *a, **k: calls.append(("wide", k)))
    args, extra = _specs(tokens)
    hc_decode_mix_wide_auto(*args, branches=BRANCHES, hidden=HIDDEN, eps=EPS, mode=1, max_tile=512, fold=True,
                            min_rows=floor, **extra)
    assert [c[0] for c in calls] == [want]
    if want == "wide":
        assert calls[0][1]["min_rows"] == floor
    else:
        assert "min_rows" not in calls[0][1]


@pytest.mark.parametrize("tokens", [160, 288, 496])
def test_auto_does_not_pass_the_floor_from_160(tokens, monkeypatch):
    calls = []
    monkeypatch.setattr(hc_decode_wide, "hc_decode_mix_wide", lambda *a, **k: calls.append(k))
    args, extra = _specs(tokens)
    for floor in (None, 64):
        hc_decode_mix_wide_auto(*args, branches=BRANCHES, hidden=HIDDEN, eps=EPS, mode=1, max_tile=512, fold=True,
                                sched=True, min_rows=floor, **extra)
    assert calls[0] == calls[1] and "min_rows" not in calls[0]


# ---------------------------------------------------------------- kernel API
@pytest.mark.parametrize("mode", [2, 3])
def test_floor_needs_mode_1(mode):
    args, extra = _specs(112)

    def f(*a):
        return hc_decode_mix_wide(*a, branches=BRANCHES, hidden=HIDDEN, eps=EPS, mode=mode, min_rows=64, **extra)

    with pytest.raises(ValueError, match="HCK floor needs mode 1"):
        jax.eval_shape(f, *args)


@pytest.mark.parametrize("tokens,floor", [(48, 64), (72, 64), (96, 112), (512, 64)])
def test_kernel_refuses_rows_outside_the_floor(tokens, floor):
    args, extra = _specs(tokens)

    def f(*a):
        return hc_decode_mix_wide(*a, branches=BRANCHES, hidden=HIDDEN, eps=EPS, mode=1, min_rows=floor, **extra)

    with pytest.raises(ValueError, match="wide"):
        jax.eval_shape(f, *args)


def _wide_trace(module, tokens, mode, inject, prologue, **extra):
    args, arrays = _specs(tokens, inject, prologue)
    names = list(arrays)

    def f(*a):
        kw = dict(zip(names, a[len(args):]))
        return module.hc_decode_mix_wide(*a[: len(args)], branches=BRANCHES, hidden=HIDDEN, eps=EPS, mode=mode, **kw, **extra)

    arr = [a for a in args if a is not None]
    if not inject:
        def g(*a):
            return f(*a[:4], None, *a[4:])

        return _mask(str(jax.make_jaxpr(g)(*arr, *arrays.values())))
    return _mask(str(jax.make_jaxpr(f)(*arr, *arrays.values())))


@pytest.mark.parametrize("tokens", [64, 112, 144])
def test_trace_line(capsys, tokens):
    jax.clear_caches()
    capsys.readouterr()
    _wide_trace(hc_decode_wide, 160, 1, True, True, fold=True)
    assert "WIDE_MIN_ROWS traced" not in capsys.readouterr().out
    jax.clear_caches()
    _wide_trace(hc_decode_wide, tokens, 1, True, True, fold=True, min_rows=64)
    out = capsys.readouterr().out
    want = f"q38 HCK WIDE_MIN_ROWS traced: rows {tokens}, kernel qwen4exp_hc_decode_combine_w1f, floor 64, blocks {len(row_blocks(tokens))}"
    assert out.count(want) == 1, out


def _load_ref():
    data = K3B2_REF.read_bytes()
    assert hashlib.sha256(data).hexdigest() == K3B2_SHA256
    loader = importlib.machinery.SourceFileLoader("q38_hc_decode_wide_k3b2_mr", str(K3B2_REF))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.mark.parametrize("tokens", [160, 288])
@pytest.mark.parametrize("inject,prologue,fold", [(True, True, True), (True, True, False), (False, False, False)])
def test_kernel_body_ignores_the_floor_from_160(tokens, inject, prologue, fold):
    """A floor passed at >= 160 rows changes nothing: the trace equals K3b.2's kernel (the pinned copy)."""
    fx = {"fold": True} if fold else {}
    ref = _wide_trace(_load_ref(), tokens, 1, inject, prologue, **fx)
    assert _wide_trace(hc_decode_wide, tokens, 1, inject, prologue, min_rows=64, **fx) == ref


# ---------------------------------------------------------------- small numerics
SHAPES = [(True, True, True), (True, True, False), (False, True, False), (True, False, True), (False, False, False)]


def _bits(x):
    return np.asarray(jnp.asarray(x, jnp.float32)).view(np.uint32)


def _same(a, b, label):
    a = [_bits(o) for o in a if o is not None]
    b = [_bits(o) for o in b if o is not None]
    assert len(a) == len(b), label
    for i, (u, v) in enumerate(zip(a, b)):
        assert np.array_equal(u, v), (label, i, int((u != v).sum()))


@pytest.mark.parametrize("tokens", [64, 96, 112, 144])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dma_mode", ["on_wait", "eager"])
def test_floor_numerics_small_bitwise(tokens, shape, dma_mode):
    """Interpret mode, small widths (hidden 256, low rank 64), NaN for uninitialised memory, race detection: below 160 rows
    W1 (via the floor) is bitwise the served hc_decode_mix (max_tile 512) in mixed, injection gate and h, and the
    schedule (W1_SCHED, both switches) is bitwise W1 / W1F."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc

    inject, prologue, fold = shape
    hidden, lowrank = 256, 64
    width = BRANCHES * hidden
    g = np.random.default_rng(2000 + tokens)
    n = lambda *s: g.standard_normal(s).astype(np.float32)  # noqa: E731
    bf = lambda a: jnp.asarray(a, jnp.bfloat16)  # noqa: E731
    x, w = bf(4.0 * n(tokens, width)), bf(0.3 * n(width))
    down, up = bf(0.08 * n(lowrank, width)), bf(0.2 * n(width, lowrank))
    inj = bf(0.02 * n(BRANCHES, width)) if inject else None
    common = dict(branches=BRANCHES, hidden=hidden, eps=EPS)
    if prologue:
        common["block"] = bf(n(tokens, hidden))
        common["prev_injection"] = jnp.asarray(2.0 / (1.0 + np.exp(-n(tokens, BRANCHES))), jnp.float32)
    interp = pltpu.InterpretParams(dma_execution_mode=dma_mode, uninitialized_memory="nan", detect_races=True)
    served = hc_decode_mix(x, w, down, up, inj, max_tile=512, interpret=interp, **common)
    assert all(np.isfinite(np.asarray(jnp.asarray(o, jnp.float32))).all() for o in served if o is not None)
    w1 = hc_decode_mix_wide(x, w, down, up, inj, mode=1, min_rows=64, interpret=interp, **common)
    assert ipc.races is not None and not ipc.races.races_found
    _same(w1, served, "W1 vs served")
    fx = {"fold": True} if fold else {}
    base = hc_decode_mix_wide(x, w, down, up, inj, mode=1, min_rows=64, interpret=interp, **fx, **common) if fold else w1
    sched = hc_decode_mix_wide(x, w, down, up, inj, mode=1, min_rows=64, sched_dma=True, sched_onepass=True,
                               interpret=interp, **fx, **common)
    assert not ipc.races.races_found
    _same(sched, base, "schedule vs W1" + ("F" if fold else ""))

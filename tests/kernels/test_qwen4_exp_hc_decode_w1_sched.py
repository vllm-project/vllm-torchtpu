# SPDX-License-Identifier: Apache-2.0
"""TPU_QWEN4_HC_DECODE_W1_SCHED (q38 HCK): the W1 decode HC kernel's schedule v2 (``..._w1s`` / ``..._w1fs``).

Fast checks: flag reads, compile key, the adapter's refusals, op names and engagement line, dispatch, the flag-off trace
(identical to K3b.2's hc_decode_wide.py, a pinned copy in q38_ref/), the schedule's structure (the DMA switch leaves every
copy and compute equation identical and starts every input copy before the first wait; the one-pass switch runs one up
projection per branch), and small interpret-mode numerics
(DMAs executed on wait, NaN for uninitialised memory, race detection): bitwise on vs off. The full-size bitwise legs, the
deviceless VMEM compiles and the code mutants run in the HCK CPU proof (scratch/hck/cpu)."""

import hashlib
import importlib.machinery
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import hc_decode_wide
from vllm_torchtpu.kernels.qwen4_exp.hc_decode_wide import (
    hc_decode_mix_wide,
    hc_decode_mix_wide_auto,
    kernel_name,
)

FLAG = "TPU_QWEN4_HC_DECODE_W1_SCHED"
WIDE = "TPU_QWEN4_HC_DECODE_WIDE"
FOLD = "TPU_QWEN4_HC_DECODE_INJECT_FOLD"
BRANCHES, HIDDEN, LOWRANK, EPS = 4, 2560, 320, 1e-6
WIDTH = BRANCHES * HIDDEN
K3B2_REF = Path(__file__).parent / "q38_ref" / "hc_decode_wide_k3b2_d26816eaf6.py.txt"
K3B2_SHA256 = "2b89a458da13108ca66b17c98ec8ac2360b598cd957ebfd47f1e6ea6e5fa3b39"
_ENV_KEYS = (
    FLAG,
    WIDE,
    FOLD,
    "TPU_QWEN4_HC_DECODE_COMPARE",
    "TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID",
    "TPU_QWEN4_HC_DECODE_TILE_ROWS",
)


def _set_env(monkeypatch, **values):
    for key in _ENV_KEYS:
        envs.__dict__.pop(key, None)
        monkeypatch.delenv(key, raising=False)
    for key, value in values.items():
        monkeypatch.setenv(key, str(value))


# ---------------------------------------------------------------- flag
@pytest.mark.parametrize("value,expected", [(None, ""), ("", ""), ("0", "0"), ("1", "1")])
def test_flag_reads(monkeypatch, value, expected):
    _set_env(monkeypatch)
    if value is not None:
        monkeypatch.setenv(FLAG, value)
    assert envs.environment_variables[FLAG]() == expected
    assert getattr(envs, FLAG) == expected


@pytest.mark.parametrize("value", ["2", "true", "on"])
def test_flag_refuses_other_values(monkeypatch, value):
    _set_env(monkeypatch)
    monkeypatch.setenv(FLAG, value)
    with pytest.raises(ValueError):
        envs.environment_variables[FLAG]()


def test_flag_is_a_compile_key(monkeypatch):
    from vllm_torchtpu.compilation import tpu_compiler

    assert FLAG not in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    _set_env(monkeypatch)
    unset = tpu_compiler._tpu_compile_env_factors()
    monkeypatch.setenv(FLAG, "1")
    one = tpu_compiler._tpu_compile_env_factors()
    assert unset[FLAG] == ""
    assert {k for k in one if one[k] != unset.get(k)} == {FLAG}


# ---------------------------------------------------------------- adapter
class _FakeOp:
    def __init__(self, name, core):
        self.name, self.core = name, core

    def register_fake(self, fn):
        return fn


def _ops(monkeypatch, **env):
    """{(kind, inject): _FakeOp} built by the adapter under ``env``."""
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    _set_env(monkeypatch, TPU_QWEN4_HC_DECODE_TILE_ROWS=512, **env)
    monkeypatch.setattr(vllm_adapter.pallas, "jax_op", lambda name, core: _FakeOp(name, core))
    ops = {}
    for kind, build in (
        ("mix", vllm_adapter.build_decode_mix_op),
        ("combine", vllm_adapter.build_decode_combine_mix_op),
    ):
        for inject in (True, False):
            ops[kind, inject] = build.__wrapped__(BRANCHES, HIDDEN, EPS, inject)
    return ops


@pytest.mark.parametrize(
    "env,match",
    [
        ({FLAG: 1}, "needs TPU_QWEN4_HC_DECODE_WIDE=1"),
        ({FLAG: 1, WIDE: 0}, "needs TPU_QWEN4_HC_DECODE_WIDE=1"),
        ({FLAG: 1, WIDE: 2}, "needs TPU_QWEN4_HC_DECODE_WIDE=1"),
        ({FLAG: 1, WIDE: 3}, "needs TPU_QWEN4_HC_DECODE_WIDE=1"),
    ],
)
def test_adapter_refuses_sched_outside_w1(monkeypatch, env, match):
    with pytest.raises(ValueError, match=match):
        _ops(monkeypatch, **env)


def test_op_names_tag_sched(monkeypatch):
    w1 = {k: o.name for k, o in _ops(monkeypatch, **{WIDE: 1}).items()}
    w1f = {k: o.name for k, o in _ops(monkeypatch, **{WIDE: 1, FOLD: 1}).items()}
    off0 = {k: o.name for k, o in _ops(monkeypatch, **{WIDE: 1, FOLD: 1, FLAG: 0}).items()}
    offe = {k: o.name for k, o in _ops(monkeypatch, **{WIDE: 1, FOLD: 1, FLAG: ""}).items()}
    s1 = {k: o.name for k, o in _ops(monkeypatch, **{WIDE: 1, FLAG: 1}).items()}
    s1f = {k: o.name for k, o in _ops(monkeypatch, **{WIDE: 1, FOLD: 1, FLAG: 1}).items()}
    assert off0 == w1f and offe == w1f
    for key in w1:
        assert s1[key].endswith("_t512_w1s_v1"), s1[key]
        assert s1f[key].endswith("_t512_w1fs_v1"), s1f[key]
        assert s1[key].replace("_w1s_v1", "_w1_v1") == w1[key]
        assert s1f[key].replace("_w1fs_v1", "_w1f_v1") == w1f[key]


def test_engagement_line(monkeypatch, capsys):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    vllm_adapter._log_w1_sched.cache_clear()
    _ops(monkeypatch, **{WIDE: 1, FOLD: 1, FLAG: 1})
    out = capsys.readouterr().out
    assert out.count("q38 HCK: TPU_QWEN4_HC_DECODE_W1_SCHED=1 engaged") == 1 and "op tag _w1fs" in out
    vllm_adapter._log_w1_sched.cache_clear()
    _ops(monkeypatch, **{WIDE: 1, FOLD: 1})
    assert "q38 HCK" not in capsys.readouterr().out


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
    # Source locations and object addresses are not the program.
    txt = re.sub(r"\S+\.py(?:\.txt)?:\d+", "<src>", txt)
    return re.sub(r"0x[0-9a-f]+", "<addr>", txt)


def _core_trace(op, rows, kind, inject):
    return _mask(str(jax.make_jaxpr(op.core)(*_core_args(rows, kind, inject))))


@pytest.mark.parametrize("rows", [144, 160, 288, 496])
def test_adapter_cores_take_the_sched_kernel(monkeypatch, rows):
    base = _ops(monkeypatch, **{WIDE: 1, FOLD: 1})
    on = _ops(monkeypatch, **{WIDE: 1, FOLD: 1, FLAG: 1})
    for (kind, inject), op in on.items():
        t_on = _core_trace(op, rows, kind, inject)
        t_base = _core_trace(base[kind, inject], rows, kind, inject)
        if rows < 160:
            assert t_on == t_base, (kind, inject, rows)
            assert "_w1fs" not in t_on and "_w1s" not in t_on
        else:
            assert t_on != t_base
            assert ("_w1fs" in t_on) == inject and ("_w1s" in t_on) == (not inject)


# ---------------------------------------------------------------- kernel API
def test_kernel_name():
    assert kernel_name(True, 1, fold=True, sched=True) == "qwen4exp_hc_decode_combine_w1fs"
    assert kernel_name(False, 1, fold=True, sched=True) == "qwen4exp_hc_decode_w1fs"
    assert kernel_name(True, 1, sched=True) == "qwen4exp_hc_decode_combine_w1s"
    assert kernel_name(True, 1, "dma", True, True) == "qwen4exp_hc_decode_combine_w1fs_dma"
    assert kernel_name(True, 1, fold=True) == "qwen4exp_hc_decode_combine_w1f"


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


@pytest.mark.parametrize("mode", [2, 3])
@pytest.mark.parametrize("switch", ["sched_dma", "sched_onepass"])
def test_sched_needs_mode_1(mode, switch):
    args, extra = _specs(288)

    def f(*a):
        return hc_decode_mix_wide(*a, branches=BRANCHES, hidden=HIDDEN, eps=EPS, mode=mode, **{switch: True}, **extra)

    with pytest.raises(ValueError, match="HCK schedule needs mode 1"):
        jax.eval_shape(f, *args)


@pytest.mark.parametrize("tokens", [160, 288, 496])
def test_trace_line(capsys, tokens):
    """The trace-time engagement line prints (once per traced shape) with the schedule and never without it."""
    jax.clear_caches()   # a shape traced by an earlier test would not trace (or print) again
    capsys.readouterr()
    for prologue in (True, False):
        _wide_trace(hc_decode_wide, tokens, 1, True, prologue, fold=True)
    assert "q38 HCK W1_SCHED traced" not in capsys.readouterr().out
    jax.clear_caches()
    _wide_trace(hc_decode_wide, tokens, 1, True, True, fold=True, sched_dma=True, sched_onepass=True)
    out = capsys.readouterr().out
    b2 = 1 if tokens <= 320 else 2
    want = (f"q38 HCK W1_SCHED traced: rows {tokens}, kernel qwen4exp_hc_decode_combine_w1fs, sched_dma 1, sched_onepass 1, "
            f"b2 blocks {b2}")
    assert out.count(want) == 1, out


@pytest.mark.parametrize("tokens", [16, 144, 512])
def test_auto_keeps_served_kernel_outside_wide_rows(tokens, monkeypatch):
    calls = []
    monkeypatch.setattr(hc_decode_wide, "hc_decode_mix", lambda *a, **k: calls.append(("served", k)))
    monkeypatch.setattr(hc_decode_wide, "hc_decode_mix_wide", lambda *a, **k: calls.append(("wide", k)))
    args, extra = _specs(tokens)
    hc_decode_mix_wide_auto(
        *args, branches=BRANCHES, hidden=HIDDEN, eps=EPS, mode=1, max_tile=512, fold=True, sched=True, **extra
    )
    assert [c[0] for c in calls] == ["served"] and "sched_dma" not in calls[0][1] and "sched" not in calls[0][1]


@pytest.mark.parametrize("tokens", [160, 288, 496])
def test_auto_passes_sched_to_the_wide_kernel(tokens, monkeypatch):
    calls = []
    monkeypatch.setattr(hc_decode_wide, "hc_decode_mix_wide", lambda *a, **k: calls.append(k))
    args, extra = _specs(tokens)
    for sched in (False, True):
        hc_decode_mix_wide_auto(
            *args, branches=BRANCHES, hidden=HIDDEN, eps=EPS, mode=1, max_tile=512, fold=True, sched=sched, **extra
        )
    assert "sched_dma" not in calls[0] and "sched_onepass" not in calls[0]
    assert calls[1]["sched_dma"] is True and calls[1]["sched_onepass"] is True


# ---------------------------------------------------------------- flag-off trace
def _load_ref():
    data = K3B2_REF.read_bytes()
    assert hashlib.sha256(data).hexdigest() == K3B2_SHA256
    loader = importlib.machinery.SourceFileLoader("q38_hc_decode_wide_k3b2", str(K3B2_REF))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _wide_trace(module, tokens, mode, inject, prologue, **extra):
    args, arrays = _specs(tokens, inject, prologue)
    names = list(arrays)

    def f(*a):
        kw = dict(zip(names, a[len(args):]))
        return module.hc_decode_mix_wide(
            *a[: len(args)], branches=BRANCHES, hidden=HIDDEN, eps=EPS, mode=mode, **kw, **extra
        )

    arr = [a for a in args if a is not None]
    if not inject:
        def g(*a):
            return f(*a[:4], None, *a[4:])

        return _mask(str(jax.make_jaxpr(g)(*arr, *arrays.values())))
    return _mask(str(jax.make_jaxpr(f)(*arr, *arrays.values())))


FORMS = [
    (tokens, mode, inject, prologue, fold)
    for tokens in (160, 288, 320, 496)
    for mode in (1, 2, 3)
    for inject in (True, False)
    for prologue in (True, False)
    for fold in ((False, True) if mode == 1 else (False,))
]


def _form_id(form):
    tokens, mode, inject, prologue, fold = form
    return f"t{tokens}m{mode}{'i' if inject else 'n'}{'c' if prologue else 'x'}{'f' if fold else ''}"


@pytest.mark.parametrize("form", FORMS, ids=_form_id)
def test_flag_off_trace_is_k3b2(form):
    """Both switches off (default and explicit False) trace K3b.2's kernel; on, mode 1 changes."""
    tokens, mode, inject, prologue, fold = form
    ref_mod = _load_ref()
    fx = {"fold": True} if fold else {}
    ref = _wide_trace(ref_mod, tokens, mode, inject, prologue, **fx)
    assert _wide_trace(hc_decode_wide, tokens, mode, inject, prologue, **fx) == ref
    off = _wide_trace(hc_decode_wide, tokens, mode, inject, prologue, sched_dma=False, sched_onepass=False, **fx)
    assert off == ref
    if mode == 1:
        on = _wide_trace(hc_decode_wide, tokens, 1, inject, prologue, sched_dma=True, sched_onepass=True, **fx)
        assert on != ref and re.search(r"_w1f?s\b", on)


# ---------------------------------------------------------------- structure
def _kernel_jaxpr(tokens, inject, prologue, **extra):
    args, arrays = _specs(tokens, inject, prologue)
    names = list(arrays)

    def f(*a):
        kw = dict(zip(names, a[len(args):]))
        pos = list(a[: len(args)])
        return hc_decode_mix_wide(*pos, branches=BRANCHES, hidden=HIDDEN, eps=EPS, mode=1, **kw, **extra)

    arr = [a for a in args if a is not None]
    if not inject:
        def g(*a):
            return f(*a[:4], None, *a[4:])

        closed = jax.make_jaxpr(g)(*arr, *arrays.values())
    else:
        closed = jax.make_jaxpr(f)(*arr, *arrays.values())
    found = []

    def walk(jx):
        for eqn in jx.eqns:
            if eqn.primitive.name == "pallas_call":
                found.append(eqn.params["jaxpr"])
            for v in eqn.params.values():
                for sub in v if isinstance(v, (list, tuple)) else (v,):
                    inner = getattr(sub, "jaxpr", None)
                    if inner is not None and hasattr(inner, "eqns"):
                        walk(inner)
                    elif hasattr(sub, "eqns"):
                        walk(sub)

    walk(closed.jaxpr)
    assert len(found) == 1
    return found[0]


def _compute_seq(kjx):
    """(primitive, output avals) of every non-DMA equation, nested bodies included, in program order."""
    seq = []

    def walk(jx):
        for eqn in jx.eqns:
            if eqn.primitive.name not in ("dma_start", "dma_wait"):
                seq.append((eqn.primitive.name, tuple(str(v.aval) for v in eqn.outvars)))
            for v in eqn.params.values():
                for sub in v if isinstance(v, (list, tuple)) else (v,):
                    inner = getattr(sub, "jaxpr", None)
                    if inner is not None and hasattr(inner, "eqns"):
                        walk(inner)
                    elif hasattr(sub, "eqns"):
                        walk(sub)

    walk(kjx)
    return seq


def _count(kjx, prim):
    return sum(1 for p, _ in _compute_seq_all(kjx) if p == prim)


def _compute_seq_all(kjx):
    seq = []

    def walk(jx):
        for eqn in jx.eqns:
            seq.append((eqn.primitive.name, ()))
            for v in eqn.params.values():
                for sub in v if isinstance(v, (list, tuple)) else (v,):
                    inner = getattr(sub, "jaxpr", None)
                    if inner is not None and hasattr(inner, "eqns"):
                        walk(inner)
                    elif hasattr(sub, "eqns"):
                        walk(sub)

    walk(kjx)
    return seq


SHAPES = [(True, True, True), (True, True, False), (False, True, False), (True, False, True)]  # (inject, prologue, fold)


def _starts_before_first_wait(kjx):
    """dma_start equations before the kernel's first dma_wait (top level, program order)."""
    n = 0
    for eqn in kjx.eqns:
        if eqn.primitive.name == "dma_wait":
            return n
        n += eqn.primitive.name == "dma_start"
    return n


@pytest.mark.parametrize("tokens", [160, 288, 320, 496])
@pytest.mark.parametrize("shape", SHAPES)
def test_sched_dma_keeps_every_compute_equation(tokens, shape):
    """Data movement only: the same copies and the same compute equations in the same order; every input copy (norm weight,
    inject rows, previous gate, row blocks, block rows, down, up) is started before the first wait."""
    inject, prologue, fold = shape
    fx = {"fold": True} if fold else {}
    off = _kernel_jaxpr(tokens, inject, prologue, **fx)
    dma = _kernel_jaxpr(tokens, inject, prologue, sched_dma=True, **fx)
    assert _compute_seq(dma) == _compute_seq(off)
    assert _count(dma, "dma_start") == _count(off, "dma_start")
    assert _count(dma, "dma_wait") == _count(off, "dma_wait")
    blocks = len(hc_decode_wide.row_blocks(tokens))
    n_in = 3 + blocks + int(inject) + (1 + blocks if prologue else 0)
    assert _starts_before_first_wait(dma) == n_in
    assert _starts_before_first_wait(off) < n_in


@pytest.mark.parametrize("tokens", [160, 288, 320, 496])
@pytest.mark.parametrize("shape", SHAPES)
def test_sched_onepass_runs_one_up_projection_per_branch(tokens, shape):
    inject, prologue, fold = shape
    fx = {"fold": True} if fold else {}
    off = _kernel_jaxpr(tokens, inject, prologue, **fx)
    one = _kernel_jaxpr(tokens, inject, prologue, sched_onepass=True, **fx)
    b2 = len(hc_decode_wide.row_blocks(tokens, hc_decode_wide.ONEPASS_ROWS))
    assert b2 == (1 if tokens <= 320 else 2)
    assert _count(one, "dot_general") == 1 + BRANCHES * b2
    assert _count(off, "dot_general") == 1 + BRANCHES * len(hc_decode_wide.row_blocks(tokens))


# ---------------------------------------------------------------- small numerics (bitwise on vs off)
def _bits(x):
    return np.asarray(jnp.asarray(x, jnp.float32)).view(np.uint32)


def _run(variant, x, w, down, up, inj, extra, interp):
    sw = {"off": {}, "dma": {"sched_dma": True}, "one": {"sched_onepass": True},
          "both": {"sched_dma": True, "sched_onepass": True}}[variant]
    return hc_decode_mix_wide(x, w, down, up, inj, interpret=interp, **sw, **extra)


@pytest.mark.parametrize("tokens", [160, 288, 320])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dma_mode", ["on_wait", "eager"])
def test_sched_numerics_small_bitwise(tokens, shape, dma_mode):
    """Interpret mode, small widths (hidden 256, low rank 64): every variant bitwise equal to the flag-off kernel in mixed,
    injection gate and h; DMAs executed on wait (a read before its wait reads NaN) or eagerly, races detected."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc

    inject, prologue, fold = shape
    hidden, lowrank = 256, 64
    width = BRANCHES * hidden
    g = np.random.default_rng(1000 + tokens)
    n = lambda *s: g.standard_normal(s).astype(np.float32)  # noqa: E731
    bf = lambda a: jnp.asarray(a, jnp.bfloat16)  # noqa: E731
    x, w = bf(4.0 * n(tokens, width)), bf(0.3 * n(width))
    down, up = bf(0.08 * n(lowrank, width)), bf(0.2 * n(width, lowrank))
    inj = bf(0.02 * n(BRANCHES, width)) if inject else None
    extra = dict(branches=BRANCHES, hidden=hidden, eps=EPS, mode=1)
    if fold:
        extra["fold"] = True
    if prologue:
        extra["block"] = bf(n(tokens, hidden))
        extra["prev_injection"] = jnp.asarray(2.0 / (1.0 + np.exp(-n(tokens, BRANCHES))), jnp.float32)
    interp = pltpu.InterpretParams(dma_execution_mode=dma_mode, uninitialized_memory="nan", detect_races=True)
    ref = _run("off", x, w, down, up, inj, extra, interp)
    ref = [_bits(o) for o in ref if o is not None]
    assert all(np.isfinite(np.asarray(r).view(np.float32)).all() for r in ref)
    for variant in ("dma", "one", "both"):
        out = _run(variant, x, w, down, up, inj, extra, interp)
        assert ipc.races is not None and not ipc.races.races_found, (variant, "race")
        got = [_bits(o) for o in out if o is not None]
        assert len(got) == len(ref)
        for i, (a, b) in enumerate(zip(got, ref)):
            assert np.array_equal(a, b), (variant, i, int((a != b).sum()))


# ---------------------------------------------------------------- real wrapper
_REAL_WRAPPER = r"""
import os, sys
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
from torch_tpu._internal import pallas
from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter as va
assert va.pallas is pallas and pallas.jax_op.__module__.startswith("torch_tpu")
os.environ["TPU_QWEN4_HC_DECODE_WIDE"] = "1"
os.environ["TPU_QWEN4_HC_DECODE_INJECT_FOLD"] = "1"
os.environ["TPU_QWEN4_HC_DECODE_W1_SCHED"] = "1"
os.environ["TPU_QWEN4_HC_DECODE_TILE_ROWS"] = "512"
names = []
for build in (va.build_decode_mix_op, va.build_decode_combine_mix_op):
    for inject in (True, False):
        op = build.__wrapped__(4, 2560, 1e-6, inject)
        names.append(str(getattr(op, "name", op)))
print("REAL WRAPPER OK built %d %s" % (len(names), names))
"""


def test_real_torch_tpu_wrapper_accepts_the_sched_ops():
    """Build the schedule ops through the real torch_tpu pallas.jax_op in a fresh process."""
    pytest.importorskip("torch_tpu")
    src = str(Path(hc_decode_wide.__file__).resolve().parents[3])
    path = src + os.pathsep + os.environ.get("PYTHONPATH", "")
    env = dict(os.environ, PYTHONPATH=path)
    proc = subprocess.run([sys.executable, "-c", _REAL_WRAPPER], env=env, capture_output=True, text=True, timeout=600)
    out = proc.stdout
    assert proc.returncode == 0 and "REAL WRAPPER OK built 4" in out, out[-2000:] + proc.stderr[-3000:]
    assert "q38 HCK: TPU_QWEN4_HC_DECODE_W1_SCHED=1 engaged" in out

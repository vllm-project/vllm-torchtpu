# SPDX-License-Identifier: Apache-2.0
"""TPU_QWEN4_HC_DECODE_INJECT_FOLD (q38 INJ-MXU): the W1 decode HC kernel with
the injection logits folded into its down projection (``..._w1f``).

Fast checks: flag reads, compile key, the adapter's refusals and op names,
dispatch, the flag-off trace (identical to K2's hc_decode_wide.py, a pinned
copy in q38_ref/), the fold kernel's structure, a small interpret-mode
numerics check and the real torch_tpu wrapper. The numerics of record run
under reports/prereg/q38_injmxu_bound_LOCKED.md (a separate CPU proof)."""

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

FLAG = "TPU_QWEN4_HC_DECODE_INJECT_FOLD"
WIDE = "TPU_QWEN4_HC_DECODE_WIDE"
BRANCHES, HIDDEN, LOWRANK, EPS = 4, 2560, 320, 1e-6
WIDTH = BRANCHES * HIDDEN
K2_REF = Path(__file__).parent / "q38_ref" / "hc_decode_wide_k2_3d617a559.py.txt"
K2_SHA256 = "de08c97f3188bdfc08ddf8fbfb53f396e6027e354314e890b8ceffafb28be2b3"
_ENV_KEYS = (
    FLAG,
    WIDE,
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
@pytest.mark.parametrize(
    "value,expected", [(None, False), ("0", False), ("1", True), ("true", True)]
)
def test_flag_reads(monkeypatch, value, expected):
    _set_env(monkeypatch)
    if value is not None:
        monkeypatch.setenv(FLAG, value)
    assert envs.environment_variables[FLAG]() is expected
    assert getattr(envs, FLAG) is expected


def test_flag_is_a_compile_key(monkeypatch):
    from vllm_torchtpu.compilation import tpu_compiler

    assert FLAG not in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    _set_env(monkeypatch)
    unset = tpu_compiler._tpu_compile_env_factors()
    monkeypatch.setenv(FLAG, "0")
    zero = tpu_compiler._tpu_compile_env_factors()
    monkeypatch.setenv(FLAG, "1")
    one = tpu_compiler._tpu_compile_env_factors()
    assert unset == zero and unset[FLAG] is False
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
    monkeypatch.setattr(
        vllm_adapter.pallas, "jax_op", lambda name, core: _FakeOp(name, core)
    )
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
        ({FLAG: 1, "TPU_QWEN4_HC_DECODE_COMPARE": 1}, "needs"),
        ({FLAG: 1, WIDE: 1, "TPU_QWEN4_HC_DECODE_COMPARE": 1}, "exclusive"),
        ({FLAG: 1, WIDE: 1, "TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID": 1}, "exclusive"),
    ],
)
def test_adapter_refuses_fold_outside_w1(monkeypatch, env, match):
    with pytest.raises(ValueError, match=match):
        _ops(monkeypatch, **env)


def test_op_names_tag_w1f(monkeypatch):
    base = {k: o.name for k, o in _ops(monkeypatch).items()}
    w1 = {k: o.name for k, o in _ops(monkeypatch, **{WIDE: 1}).items()}
    off = {k: o.name for k, o in _ops(monkeypatch, **{WIDE: 1, FLAG: 0}).items()}
    on = {k: o.name for k, o in _ops(monkeypatch, **{WIDE: 1, FLAG: 1}).items()}
    assert off == w1
    for key in base:
        assert w1[key].endswith("_t512_w1_v1"), w1[key]
        assert on[key].endswith("_t512_w1f_v1"), on[key]
        assert on[key].replace("_w1f_v1", "_w1_v1") == w1[key]


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
def test_adapter_cores_take_the_fold_kernel(monkeypatch, rows):
    w1 = _ops(monkeypatch, **{WIDE: 1})
    on = _ops(monkeypatch, **{WIDE: 1, FLAG: 1})
    for (kind, inject), op in on.items():
        t_on = _core_trace(op, rows, kind, inject)
        t_w1 = _core_trace(w1[kind, inject], rows, kind, inject)
        if rows < 160 or not inject:
            # Below W1's rows, or without an injection gate: unchanged.
            assert t_on == t_w1, (kind, inject, rows)
            assert "_w1f" not in t_on
        else:
            assert t_on != t_w1 and "_w1f" in t_on and "_w1f" not in t_w1


# ---------------------------------------------------------------- kernel API
def test_kernel_name():
    assert kernel_name(True, 1, fold=True) == "qwen4exp_hc_decode_combine_w1f"
    assert kernel_name(False, 1, fold=True) == "qwen4exp_hc_decode_w1f"
    assert kernel_name(True, 1) == "qwen4exp_hc_decode_combine_w1"


def _specs(tokens, inject=True, prologue=True, hidden=HIDDEN, lowrank=LOWRANK):
    s = jax.ShapeDtypeStruct
    bf = jnp.bfloat16
    width = BRANCHES * hidden
    args = [s((tokens, width), bf), s((width,), bf), s((lowrank, width), bf)]
    args += [s((width, lowrank), bf), s((BRANCHES, width), bf) if inject else None]
    extra = {}
    if prologue:
        extra = dict(
            block=s((tokens, hidden), bf),
            prev_injection=s((tokens, BRANCHES), jnp.float32),
        )
    return args, extra


@pytest.mark.parametrize("mode", [2, 3])
def test_fold_needs_mode_1(mode):
    args, extra = _specs(288)

    def f(*a):
        return hc_decode_mix_wide(
            *a, branches=BRANCHES, hidden=HIDDEN, eps=EPS, mode=mode, fold=True, **extra
        )

    with pytest.raises(ValueError, match="fold needs mode 1"):
        jax.eval_shape(f, *args)


@pytest.mark.parametrize("tokens", [16, 144, 512])
def test_auto_keeps_served_kernel_outside_wide_rows(tokens, monkeypatch):
    calls = []
    monkeypatch.setattr(
        hc_decode_wide, "hc_decode_mix", lambda *a, **k: calls.append(("served", k))
    )
    monkeypatch.setattr(
        hc_decode_wide, "hc_decode_mix_wide", lambda *a, **k: calls.append(("wide", k))
    )
    args, extra = _specs(tokens)
    hc_decode_mix_wide_auto(
        *args, branches=BRANCHES, hidden=HIDDEN, eps=EPS, mode=1, max_tile=512,
        fold=True, **extra,
    )
    assert [c[0] for c in calls] == ["served"] and "fold" not in calls[0][1]


@pytest.mark.parametrize("tokens", [160, 288, 496])
def test_auto_passes_fold_to_the_wide_kernel(tokens, monkeypatch):
    calls = []
    monkeypatch.setattr(
        hc_decode_wide, "hc_decode_mix_wide", lambda *a, **k: calls.append(k)
    )
    args, extra = _specs(tokens)
    for fold in (False, True):
        hc_decode_mix_wide_auto(
            *args, branches=BRANCHES, hidden=HIDDEN, eps=EPS, mode=1, max_tile=512,
            fold=fold, **extra,
        )
    assert [(k["mode"], k["fold"]) for k in calls] == [(1, False), (1, True)]


# ---------------------------------------------------------------- flag-off trace
def _load_k2():
    data = K2_REF.read_bytes()
    assert hashlib.sha256(data).hexdigest() == K2_SHA256
    loader = importlib.machinery.SourceFileLoader("q38_hc_decode_wide_k2", str(K2_REF))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _wide_trace(module, tokens, mode, inject, prologue, **extra):
    args, arrays = _specs(tokens, inject, prologue)
    names = list(arrays)

    def f(*a):
        kw = dict(zip(names, a[len(args) :]))
        return module.hc_decode_mix_wide(
            *a[: len(args)], branches=BRANCHES, hidden=HIDDEN, eps=EPS, mode=mode,
            **kw, **extra,
        )

    arr = [a for a in args if a is not None]
    if not inject:
        def g(*a):
            return f(*a[:4], None, *a[4:])

        return _mask(str(jax.make_jaxpr(g)(*arr, *arrays.values())))
    return _mask(str(jax.make_jaxpr(f)(*arr, *arrays.values())))


FORMS = [
    (tokens, mode, inject, prologue)
    for tokens in (160, 288, 496)
    for mode in (1, 2, 3)
    for inject in (True, False)
    for prologue in (True, False)
]


def _form_id(form):
    tokens, mode, inject, prologue = form
    return f"t{tokens}m{mode}{'i' if inject else 'n'}{'c' if prologue else 'x'}"


@pytest.mark.parametrize("form", FORMS, ids=_form_id)
def test_flag_off_trace_is_k2(form):
    """fold off (default and explicit False) traces K2's kernel; fold on
    changes only mode 1 with the injection gate."""
    tokens, mode, inject, prologue = form
    k2 = _load_k2()
    ref = _wide_trace(k2, tokens, mode, inject, prologue)
    assert _wide_trace(hc_decode_wide, tokens, mode, inject, prologue) == ref
    off = _wide_trace(hc_decode_wide, tokens, mode, inject, prologue, fold=False)
    assert off == ref
    if mode == 1:
        on = _wide_trace(hc_decode_wide, tokens, 1, inject, prologue, fold=True)
        if inject:
            assert on != ref and "_w1f" in on
        else:
            assert on == ref


@pytest.mark.parametrize("prologue", [True, False])
def test_fold_kernel_structure(prologue):
    """No extra matrix pass; the down rows grow to a 128-row tile and the
    separate inject scratch is gone."""
    on = _wide_trace(hc_decode_wide, 288, 1, True, prologue, fold=True)
    off = _wide_trace(hc_decode_wide, 288, 1, True, prologue)
    assert on.count("dot_general") == off.count("dot_general") > 0
    assert "bf16[384,10240]" in on and "bf16[384,10240]" not in off
    # The inject rows: an input in both, a separate scratch only without fold.
    assert off.count("bf16[4,10240]") > on.count("bf16[4,10240]") > 0


# ---------------------------------------------------------------- small numerics
def _sig(a):
    return 1.0 / (1.0 + np.exp(-a))


def _bf16(x):
    import ml_dtypes

    return np.asarray(x, np.float64).astype(ml_dtypes.bfloat16).astype(np.float64)


@pytest.mark.parametrize("prologue", [True, False])
def test_fold_numerics_small(prologue):
    """Interpret mode, small widths (hidden 256, low rank 32): the folded gate
    is a bf16-logit gate within the order-change allowance of the served one,
    and h (formed before the down projection) is bitwise."""
    hidden, lowrank, tokens = 256, 32, 160
    width = BRANCHES * hidden
    g = np.random.default_rng(11)
    n = lambda *s: g.standard_normal(s).astype(np.float32)  # noqa: E731
    bf = lambda a: jnp.asarray(a, jnp.bfloat16)  # noqa: E731
    x, w = bf(4.0 * n(tokens, width)), bf(0.3 * n(width))
    # Larger down / up scales than the model's so the mixed rows depend
    # visibly on the gate columns (a wrong gate slice moves them by ~25 %).
    down, up = bf(0.08 * n(lowrank, width)), bf(0.2 * n(width, lowrank))
    inj = bf(0.02 * n(BRANCHES, width))
    extra = dict(branches=BRANCHES, hidden=hidden, eps=EPS, mode=1)
    extra["interpret"] = pltpu.InterpretParams(uninitialized_memory="nan")
    if prologue:
        extra["block"] = bf(n(tokens, hidden))
        pinj = 2.0 * _sig(n(tokens, BRANCHES))
        extra["prev_injection"] = jnp.asarray(pinj, jnp.float32)
    a = hc_decode_mix_wide(x, w, down, up, inj, **extra)
    e = hc_decode_mix_wide(x, w, down, up, inj, fold=True, **extra)
    ga = np.asarray(a[1], np.float64)
    ge = np.asarray(e[1], np.float64)
    assert np.isfinite(ge).all()
    if prologue:
        ha, he = (np.asarray(o[2]).view(np.uint16) for o in (a, e))
        assert np.array_equal(ha, he)
    # The normalized rows the kernel formed (h, or x without a prologue).
    h = np.asarray(jnp.asarray(e[2] if prologue else x, jnp.float32), np.float64)
    wt = np.asarray(jnp.asarray(w, jnp.float32), np.float64)
    norm = []
    for b in range(BRANCHES):
        v = h[:, b * hidden : (b + 1) * hidden]
        r = 1.0 / np.sqrt(np.mean(v * v, axis=1, keepdims=True) + EPS)
        norm.append(_bf16(v * r * (1.0 + wt[b * hidden : (b + 1) * hidden])))
    norm = np.concatenate(norm, axis=1)
    iw = np.asarray(jnp.asarray(inj, jnp.float32), np.float64)
    L, S = norm @ iw.T, np.abs(norm) @ np.abs(iw).T
    tau = 2.0**-14 * S * (1 + 2.0**-6)
    m = (np.abs(L) * (1 + 2.0**-6)) + tau
    ulp = np.exp2(np.floor(np.log2(np.maximum(m, 2.0**-126))) - 7.0)
    # |b_E - b_A| <= tau + ulp, and the gate moves at most 1/8 of that.
    assert (np.abs(ge - ga) <= (tau + ulp) / 8.0 + 2.0**-22).all()
    # Every folded gate is 2 sigmoid(b / 4) of a bf16 logit b.
    import ml_dtypes

    b = BRANCHES * np.log(ge / (2.0 - ge))
    best = np.full(ge.shape, np.inf)
    bits = b.astype(ml_dtypes.bfloat16).view(np.uint16).astype(np.int64)
    for step in (-1, 0, 1):
        cand = np.clip(bits + step, 0, 0xFFFF).astype(np.uint16)
        cand = cand.view(ml_dtypes.bfloat16).astype(np.float64)
        best = np.minimum(best, np.abs(2.0 * _sig(cand / BRANCHES) - ge))
    assert (best <= 4 * 2.0**-24).all()
    # The mixed rows, loosely: a CPU dot may round the gate columns
    # differently once the down rows grow (the device leg holds mixed
    # bitwise); a wrong gate slice is far outside this.
    ma = np.asarray(jnp.asarray(a[0], jnp.float32), np.float64)
    me = np.asarray(jnp.asarray(e[0], jnp.float32), np.float64)
    d = np.abs(me - ma)
    assert d.max() <= 2.0**-7 * np.abs(ma).max()
    assert np.linalg.norm(d) <= 1e-2 * np.linalg.norm(ma)


# ---------------------------------------------------------------- real wrapper
_REAL_WRAPPER = r"""
import os, sys
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
from torch_tpu._internal import pallas
from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter as va
assert va.pallas is pallas and pallas.jax_op.__module__.startswith("torch_tpu")
os.environ["TPU_QWEN4_HC_DECODE_WIDE"] = "1"
os.environ["TPU_QWEN4_HC_DECODE_INJECT_FOLD"] = "1"
os.environ["TPU_QWEN4_HC_DECODE_TILE_ROWS"] = "512"
names = []
for build in (va.build_decode_mix_op, va.build_decode_combine_mix_op):
    for inject in (True, False):
        op = build.__wrapped__(4, 2560, 1e-6, inject)
        names.append(str(getattr(op, "name", op)))
print("REAL WRAPPER OK built %d %s" % (len(names), names))
"""


def test_real_torch_tpu_wrapper_accepts_the_fold_ops():
    """Methodology rule 12: build the fold ops through the real torch_tpu
    pallas.jax_op in a fresh process."""
    pytest.importorskip("torch_tpu")
    src = str(Path(hc_decode_wide.__file__).resolve().parents[3])
    path = src + os.pathsep + os.environ.get("PYTHONPATH", "")
    env = dict(os.environ, PYTHONPATH=path)
    proc = subprocess.run(
        [sys.executable, "-c", _REAL_WRAPPER], env=env, capture_output=True, text=True,
        timeout=600,
    )
    out = proc.stdout
    assert proc.returncode == 0 and "REAL WRAPPER OK built 4" in out, (
        out[-2000:] + proc.stderr[-3000:]
    )

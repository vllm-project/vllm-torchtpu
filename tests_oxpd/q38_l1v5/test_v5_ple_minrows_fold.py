# SPDX-License-Identifier: Apache-2.0
"""STACK-D v5 composition (q38 L1-V5): glue32's PLE min-rows gate (VLLM_TPU_PLE_DECODE_BOUNDED_MIN_ROWS, q38 L1-GLUEFIX)
x PLE-D14-FOLD (VLLM_TPU_PLE_D14_FOLD, q38 L1-PLE) on the one PLE speculative conv (CPU).

Both members add a 4th parameter to the PLE conv builder; the composed signature is
build_pooled_speculative_short_conv_op(dilation, state_len, bounded=0, min_rows=0, fold=False) and
pooled_short_conv_speculative(..., bounded, bounded_min_rows, fold). With both on, a bucket of fewer than min_rows tokens
traces the default (bounded=0) conv: the fold is not taken (same jaxpr as the default op, same bits); a bucket of
min_rows or more takes the fold exactly as the fold alone does (same jaxpr as fold-only, the fold program engaged, bits
equal to the default op). The op builder names the composed op _bnd<b>_m<n>_fold with all three keywords; bind_pool
passes (d, s, b, min_rows, True) with both envs set and (d, s, b, 0, True) with the fold alone.
Pallas runs in interpret mode with the fold test's CPU writer stand-in (tests/kernels/test_qwen4_exp_ple_d14_fold.py).
"""

import functools
import importlib.util
import os
from types import SimpleNamespace

import jax
import numpy as np
import pytest
import torch
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.qwen4_exp import ple_d14_fold
from vllm_torchtpu.kernels.qwen4_exp import pool as pool_mod

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "q38_l1ple_fold_suite", os.path.join(_HERE, "..", "kernels", "test_qwen4_exp_ple_d14_fold.py"))
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
BND, MINR, FOLD = "VLLM_TPU_PLE_DECODE_BOUNDED", "VLLM_TPU_PLE_DECODE_BOUNDED_MIN_ROWS", "VLLM_TPU_PLE_D14_FOLD"


@pytest.fixture
def fold_calls(monkeypatch):
    """The fold suite's CPU setup (interpret-mode Pallas, write_region stand-in) plus a spy on the fold program."""
    real = pl.pallas_call

    def call(*args, **kwargs):
        kwargs["interpret"] = True
        return real(*args, **kwargs)

    monkeypatch.setattr(pl, "pallas_call", call)
    monkeypatch.setattr(pool_mod, "write_region", F._write_region_cpu)
    calls = []
    real_fold = ple_d14_fold.uniform_window_conv_fold

    def spy(*args, **kwargs):
        calls.append(args[0].shape[0])
        return real_fold(*args, **kwargs)

    monkeypatch.setattr(ple_d14_fold, "uniform_window_conv_fold", spy)
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    return calls


def _fn(bounded, min_rows=0, fold=False):
    kw = dict(dilation=F.DILATION, state_len=F.STATE_LEN, bounded=bounded)
    if min_rows:
        kw["bounded_min_rows"] = min_rows
    if fold:
        kw["fold"] = True
    return functools.partial(pool_mod.pooled_short_conv_speculative, **kw)


def _args(case):
    return tuple(case[k] for k in F.ORDER)


def _run(fn, case):
    pool, output = jax.jit(fn)(*_args(case))
    return np.asarray(pool).view(np.uint8), np.asarray(output.view(np.uint16))


# (count = max-num-seqs, tokens = bucket, live requests, min rows, fold taken)
CASES = [
    (48, 8, 2, 32, False),
    (48, 16, 4, 32, False),
    (48, 16, 3, 32, False),
    (48, 32, 8, 48, False),
    (48, 32, 8, 32, True),
    (48, 48, 12, 32, True),
    (48, 64, 16, 32, True),
    (48, 48, 12, 48, True),
    (48, 16, 4, 16, True),
]


@pytest.mark.parametrize("bounded", [1, 2])
@pytest.mark.parametrize("count,tokens,live,min_rows,taken", CASES,
                         ids=[f"c{c}-t{t}-n{n}-m{m}" for c, t, n, m, _ in CASES])
def test_min_rows_gate_then_fold(fold_calls, count, tokens, live, min_rows, taken, bounded):
    case = F._case(count * 1000 + tokens * 10 + live + min_rows, count=count, tokens=tokens, live=live)
    before = np.asarray(case["pool"]).view(np.uint8)
    pool0, out0 = _run(_fn(0), case)
    assert fold_calls == []
    poolc, outc = _run(_fn(bounded, min_rows, fold=True), case)
    assert fold_calls == ([tokens] if taken else []), fold_calls
    # Not vacuous: the default op wrote checkpoints and produced live outputs.
    assert np.any(pool0 != before) and np.any(out0[: live * F.WINDOW] != 0)
    np.testing.assert_array_equal(outc, out0, err_msg="composed output vs default")
    np.testing.assert_array_equal(poolc, pool0, err_msg="composed pool vs default")
    fold_calls.clear()
    jc = str(jax.make_jaxpr(_fn(bounded, min_rows, fold=True))(*_args(case)))
    jd = str(jax.make_jaxpr(_fn(0))(*_args(case)))
    jf = str(jax.make_jaxpr(_fn(bounded, fold=True))(*_args(case)))
    assert jf != jd  # not vacuous: the fold changes this bucket's program
    if taken:
        assert jc == jf, "at / above min rows the composed op must trace exactly the fold"
    else:
        assert jc == jd, "below min rows the composed op must trace exactly the default conv"


def test_composed_op_builds_with_its_own_name(monkeypatch):
    from vllm_torchtpu.layers.adapter.custom_ops import ple_op

    ops = [ple_op.build_pooled_speculative_short_conv_op(3, 9, *b) for b in ((1,), (1, 32), (1, 0, True), (1, 32, True))]
    assert all(callable(op) for op in ops) and len({id(op) for op in ops}) == 4
    seen = []

    class _Op:
        def __init__(self, name, fn):
            kw = dict(fn.keywords)
            assert (kw.pop("dilation"), kw.pop("state_len")) == (3, 9), kw
            seen.append((name, kw))

        def register_fake(self, f):
            return f

    monkeypatch.setattr(ple_op.pallas, "jax_op", lambda name, fn, **kw: _Op(name, fn))
    raw = ple_op.build_pooled_speculative_short_conv_op.__wrapped__
    raw(3, 9, 1, 32, True)
    raw(3, 9, 2, 0, True)
    raw(3, 9, 1, 32)
    raw(3, 9, 1)
    assert seen == [
        ("qwen4_exp::pooled_spec_short_conv_3_9_bnd1_m32_fold", {"bounded": 1, "bounded_min_rows": 32, "fold": True}),
        ("qwen4_exp::pooled_spec_short_conv_3_9_bnd2_fold", {"bounded": 2, "fold": True}),
        ("qwen4_exp::pooled_spec_short_conv_3_9_bnd1_m32", {"bounded": 1, "bounded_min_rows": 32}),
        ("qwen4_exp::pooled_spec_short_conv_3_9_bnd1", {"bounded": 1}),
    ], seen


@pytest.mark.parametrize("env,want", [
    ({BND: "1", MINR: "32", FOLD: "1"}, [(3, 9), (3, 9, 1, 32, True)]),
    ({BND: "2", MINR: "48", FOLD: "1"}, [(3, 9), (3, 9, 2, 48, True)]),
    ({BND: "1", FOLD: "1"}, [(3, 9), (3, 9, 1, 0, True)]),
    ({BND: "1", MINR: "32"}, [(3, 9), (3, 9, 1, 32)]),
    ({BND: "1"}, [(3, 9), (3, 9, 1)]),
    ({MINR: "32", FOLD: "1"}, [(3, 9)]),
])
def test_bind_pool_passes_min_rows_and_fold(monkeypatch, env, want):
    from vllm_torchtpu.models.vllm.qwen4_exp import pooled_ple

    for k in (BND, MINR, FOLD, "VLLM_TPU_PLE_FUSED", "VLLM_TPU_PLE_CONV_ONEHOT", "Q38_PROBE_PLE_ONEHOT_MUTANT"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    built = []
    monkeypatch.setattr(pooled_ple, "_FOLD_NOTED", [])
    monkeypatch.setattr(pooled_ple, "build_pooled_short_conv_op", lambda *a: ("conv", a))
    monkeypatch.setattr(pooled_ple, "build_pooled_speculative_short_conv_op", lambda *a: built.append(a) or ("spec", a))
    monkeypatch.setattr(pooled_ple.pdense_engagement, "register_ple", lambda *a, **k: None)
    layer = SimpleNamespace(conv_dilation=3, conv_state_len=9, prefix="model.layers.0.ple",
                            key_proj=SimpleNamespace(weight=torch.zeros(1)), value_proj=SimpleNamespace(weight=torch.zeros(1)))
    pooled_ple.Qwen4ExpPooledPLE.bind_pool(layer, torch.zeros(1))
    assert built == want, built
    assert (layer._speculative_conv_bounded is None) == (BND not in env)

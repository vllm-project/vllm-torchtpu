# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PLE_DECODE_BOUNDED_MIN_ROWS (q38 L1-GLUEFIX): the per-bucket gate of D07-b2 (CPU).

With ``bounded`` set and ``bounded_min_rows`` N, a megakernel bucket of fewer than N tokens traces the default
(``bounded=0``) conv: the same jaxpr as the default op, the same pool / output bits, the same history reads (all
max-num-seqs rows); a bucket of N or more tokens traces the D07-b2 bounded conv unchanged (same jaxpr as the ungated
op, reads only ceil(T / 4) rows). N = 0 is the D07-b2 behaviour. The op builder gives the gated op its own name and
leaves the D07-b2 op's construction unchanged when N = 0; the env defaults to 0 and refuses negatives.
Pure JAX on CPU with the plain-XLA pool I/O stand-ins of tests/kernels/test_qwen4_exp_ple_decode_bounded.py.
"""

import functools
import importlib.util
import os

import jax
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp import pool as pool_mod

_spec = importlib.util.spec_from_file_location(
    "q38_d07b2_suite", os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_qwen4_exp_ple_decode_bounded.py"))
K = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(K)


@pytest.fixture(autouse=True)
def _xla_pool_io(monkeypatch):
    monkeypatch.setattr(pool_mod, "read_region", K._read_region_ref)
    monkeypatch.setattr(pool_mod, "write_region", K._write_region_ref)
    K._READS.clear()
    K._WRITES.clear()


def _fn(bounded, min_rows=0):
    kw = dict(dilation=K.DILATION, state_len=K.STATE_LEN, bounded=bounded)
    if min_rows:
        kw["bounded_min_rows"] = min_rows
    return functools.partial(pool_mod.pooled_short_conv_speculative, **kw)


def _args(case):
    return tuple(case[k] for k in ("x", "weight", "pool", "starts", "ids", "initial", "read_ids", "ckpt_ids", "windowed"))


def _jaxpr(fn, case):
    """(jaxpr text, history reads recorded while tracing). Callers pass a FRESH partial each time: JAX caches traces per
    function object, so a second trace of the same object records no stand-in calls (pb2 j-7200e5a8: reads [] after a
    make_jaxpr of the same partial)."""
    K._READS.clear()
    K._WRITES.clear()
    return str(jax.make_jaxpr(fn)(*_args(case))), list(K._READS)


def _run(fn, case):
    pool, output = jax.jit(fn)(*_args(case))
    return np.asarray(pool.view(np.uint16)), np.asarray(output.view(np.uint16))


@pytest.mark.parametrize("bounded", [1, 2])
@pytest.mark.parametrize("tokens,live,min_rows", [(16, 2, 32), (16, 3, 32), (16, 4, 32), (16, 4, 17), (32, 7, 48)])
def test_below_min_rows_is_the_default_conv(tokens, live, min_rows, bounded):
    case = K._case(tokens * 31 + live, count=64, tokens=tokens, live=live)
    jg, rg = _jaxpr(_fn(bounded, min_rows), case)
    jd, rd = _jaxpr(_fn(0), case)
    ju, ru = _jaxpr(_fn(bounded), case)
    assert jg == jd
    assert ju != jd  # not vacuous: D07-b2 changes this bucket's program
    assert rg == rd == [64] and ru == [tokens // K.WINDOW], (rg, rd, ru)  # gated reads every max-num-seqs row, as default
    pg, og = _run(_fn(bounded, min_rows), case)
    pd, od = _run(_fn(0), case)
    np.testing.assert_array_equal(pg, pd)
    np.testing.assert_array_equal(og, od)
    assert (od[: live * K.WINDOW] != 0).any()


@pytest.mark.parametrize("bounded", [1, 2])
@pytest.mark.parametrize("tokens,live,min_rows", [(32, 7, 32), (48, 12, 32), (64, 16, 32), (16, 4, 16), (16, 3, 8)])
def test_at_or_above_min_rows_is_the_bounded_conv(tokens, live, min_rows, bounded):
    case = K._case(tokens * 37 + live, count=64, tokens=tokens, live=live)
    jg, rg = _jaxpr(_fn(bounded, min_rows), case)
    ju, ru = _jaxpr(_fn(bounded), case)
    jd, rd = _jaxpr(_fn(0), case)
    assert jg == ju and jg != jd
    assert rg == ru == [tokens // K.WINDOW] and rd == [64], (rg, ru, rd)  # bounded: ceil(T / 4) request rows read
    pg, og = _run(_fn(bounded, min_rows), case)
    pu, ou = _run(_fn(bounded), case)
    pd, od = _run(_fn(0), case)
    np.testing.assert_array_equal(pg, pu)
    np.testing.assert_array_equal(og, ou)
    np.testing.assert_array_equal(pg, pd)  # D07-b2's own identity still holds


def test_min_rows_zero_is_the_d07b2_conv():
    case = K._case(5, count=64, tokens=16, live=4)
    j0, r0 = _jaxpr(_fn(1, 0), case)
    j1, r1 = _jaxpr(_fn(1), case)
    assert j0 == j1 and r0 == r1 == [4], (r0, r1)


def test_env_default_and_refusal(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.delenv("VLLM_TPU_PLE_DECODE_BOUNDED_MIN_ROWS", raising=False)
    assert envs.VLLM_TPU_PLE_DECODE_BOUNDED_MIN_ROWS == 0
    monkeypatch.setenv("VLLM_TPU_PLE_DECODE_BOUNDED_MIN_ROWS", "32")
    assert envs.VLLM_TPU_PLE_DECODE_BOUNDED_MIN_ROWS == 32
    monkeypatch.setenv("VLLM_TPU_PLE_DECODE_BOUNDED_MIN_ROWS", "-4")
    with pytest.raises(ValueError, match="must be >= 0"):
        envs.VLLM_TPU_PLE_DECODE_BOUNDED_MIN_ROWS


def test_gated_op_builds_with_its_own_name(monkeypatch):
    """The gated op builds through torch_tpu's real pallas.jax_op next to the D07-b2 one; with min rows 0 the builder
    returns the D07-b2 op itself (same cache entry). With a recording jax_op (the builder's cache bypassed): the gated
    op's name ends in _bnd1_m32 and its function carries bounded=1, bounded_min_rows=32; min rows 0 gives exactly the
    D07-b2 name and keywords."""
    from vllm_torchtpu.layers.adapter.custom_ops import ple_op

    d07b2 = ple_op.build_pooled_speculative_short_conv_op(3, 9, 1)
    gated = ple_op.build_pooled_speculative_short_conv_op(3, 9, 1, 32)
    assert callable(d07b2) and callable(gated) and gated is not d07b2
    assert ple_op.build_pooled_speculative_short_conv_op(3, 9, 1) is d07b2
    seen = []

    class _Op:
        def __init__(self, name, fn):
            kw = dict(fn.keywords)  # nested partials flatten into one keyword dict
            assert (kw.pop("dilation"), kw.pop("state_len")) == (3, 9), kw
            seen.append((name, kw))

        def register_fake(self, f):
            return f

    monkeypatch.setattr(ple_op.pallas, "jax_op", lambda name, fn, **kw: _Op(name, fn))
    raw = ple_op.build_pooled_speculative_short_conv_op.__wrapped__
    raw(3, 9, 1, 32)
    raw(3, 9, 1)
    raw(3, 9, 1, 0)
    assert seen[0] == ("qwen4_exp::pooled_spec_short_conv_3_9_bnd1_m32", {"bounded": 1, "bounded_min_rows": 32}), seen[0]
    assert seen[1] == seen[2] == ("qwen4_exp::pooled_spec_short_conv_3_9_bnd1", {"bounded": 1}), seen[1:]

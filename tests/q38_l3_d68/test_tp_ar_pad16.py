# SPDX-License-Identifier: Apache-2.0
"""q38 L3 D68: VLLM_TPU_DECODE_TP_AR_PAD16 pads decode all-reduce row counts that are not a multiple of 16 so H8
takes them (8 host devices, Pallas TPU interpreter, Qwen3.8-Flash hidden 2560, mesh in the engine's device order).

Pre-stated checks:
- pad on, rows 4 / 8 / 12 / 24 / 40 / 250: every member's output is bitwise H8's numpy order model ``reference`` of
  the UNPADDED partial (the zero rows change no real row), all members bitwise equal, f32 wire bitwise
  ``reference_f32``; the compiled program has no XLA all-reduce (H8 runs).
- pad off (default), the same rows: the old ``lax.psum`` fallback (an all-reduce in the program).
- multiples of 16 are untouched by the flag: the traced program text is identical with pad on and off.
- a padded count above the cap falls back to psum (rows 72 with cap 64; rows 260 with cap 256).
- flag parsing ('' off, '1' on, anything else raises) and the compile factors (PAD16 keyed only when set).
- positive control: one changed input element moves exactly that output element on every member, at 8 rows.

Needs 8 host devices: JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 (skipped otherwise).
"""
from __future__ import annotations

import importlib.util
import sys
import types
import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

if importlib.util.find_spec("vllm") is None:  # Mac dev venv: stub vLLM's logger
    import logging

    _v = types.ModuleType("vllm")
    _vl = types.ModuleType("vllm.logger")
    _vl.init_logger = logging.getLogger
    _v.logger = _vl
    sys.modules.setdefault("vllm", _v)
    sys.modules.setdefault("vllm.logger", _vl)

from vllm_torchtpu.kernels.collectives import tp8_hier_all_reduce as h8  # noqa: E402
from vllm_torchtpu.kernels.collectives import tp8_ar_decode_fn as ar  # noqa: E402

H = 2560
ENGINE_ORDER = (0, 1, 4, 5, 6, 7, 2, 3)
PAD_ROWS = (4, 8, 12, 24, 40, 250)


@pytest.fixture(scope="module")
def mesh_pairs():
    if jax.default_backend() != "cpu" or jax.device_count() < 8:
        pytest.skip("needs 8 host devices (see module docstring)")
    devs = {d.id: d for d in jax.devices()[:8]}
    return Mesh(np.asarray([devs[i] for i in ENGINE_ORDER]), ("d",)), ar.pairs_from_device_ids(ENGINE_ORDER)


def _bits(a):
    return np.asarray(a, np.float32).view(np.uint32)


def _normal(rows, seed):
    return h8._bf16_round(np.random.default_rng(seed).standard_normal((8, rows, H)).astype(np.float32))


def _fn(mesh, pairs, wire, pad, cap=h8.MAX_ROWS):
    return ar.global_fn(mesh, axis="d", pairs=pairs, wire=wire, max_rows_=cap, interpret=True, pad16_=pad)


def _run(mesh, pairs, x, wire="bf16", pad=True, cap=h8.MAX_ROWS):
    xs = jax.device_put(jnp.asarray(x.reshape(8 * x.shape[1], H), jnp.bfloat16), NamedSharding(mesh, P("d")))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = jax.jit(_fn(mesh, pairs, wire, pad, cap))(xs)
    return np.asarray(out).astype(np.float32).reshape(8, x.shape[1], H)


def _text(mesh, pairs, rows, pad, cap=h8.MAX_ROWS, wire="bf16"):
    arg = jax.ShapeDtypeStruct((8 * rows, H), jnp.bfloat16, sharding=NamedSharding(mesh, P("d")))
    return jax.jit(_fn(mesh, pairs, wire, pad, cap)).lower(arg).as_text()


def _has_ar(txt):
    return "all_reduce" in txt or "all-reduce" in txt


@pytest.mark.parametrize("rows", PAD_ROWS)
@pytest.mark.parametrize("wire", ["bf16", "f32"])
def test_pad_on_matches_h8_order_on_real_rows(mesh_pairs, rows, wire):
    mesh, pairs = mesh_pairs
    x = _normal(rows, 500 + rows)
    out = _run(mesh, pairs, x, wire, pad=True)
    assert out.shape == (8, rows, H)
    assert all(np.array_equal(_bits(out[0]), _bits(out[r])) for r in range(8)), "members disagree"
    ref = h8.reference(x, pairs, wire)
    bad = int((_bits(out[0]) != _bits(ref)).sum())
    assert bad == 0, f"{bad} elements differ from H8's {wire} order on the real rows"


@pytest.mark.parametrize("rows", PAD_ROWS)
def test_engagement_pad_on_kernel_pad_off_psum(mesh_pairs, rows):
    mesh, pairs = mesh_pairs
    assert not _has_ar(_text(mesh, pairs, rows, pad=True)), f"pad on still lowers an all-reduce at {rows} rows"
    assert _has_ar(_text(mesh, pairs, rows, pad=False)), f"pad off lost the psum fallback at {rows} rows"


@pytest.mark.parametrize("rows", (16, 64, 256))
def test_multiples_of_16_unchanged_by_the_flag(mesh_pairs, rows):
    mesh, pairs = mesh_pairs
    assert _text(mesh, pairs, rows, pad=True) == _text(mesh, pairs, rows, pad=False)


@pytest.mark.parametrize("rows,cap", [(72, 64), (260, 256)])
def test_padded_count_above_cap_falls_back(mesh_pairs, rows, cap):
    mesh, pairs = mesh_pairs
    assert _has_ar(_text(mesh, pairs, rows, pad=True, cap=cap))
    x = _normal(rows, 9 + rows)
    out = _run(mesh, pairs, x, "bf16", pad=True, cap=cap)
    assert all(np.array_equal(_bits(out[0]), _bits(out[r])) for r in range(8))
    exact = h8._bf16_round(np.asarray(x, np.float64).sum(0).astype(np.float32))
    assert float((h8.ulp_distance(out[0], exact) <= 2).mean()) >= 0.999


def test_padded_rows_helper():
    bf = jnp.bfloat16
    assert ar.padded_rows(16, H, bf, 256, True) == 16
    assert ar.padded_rows(8, H, bf, 256, True) == 16
    assert ar.padded_rows(24, H, bf, 256, True) == 32
    assert ar.padded_rows(250, H, bf, 256, True) == 256
    assert ar.padded_rows(8, H, bf, 256, False) == 0
    assert ar.padded_rows(72, H, bf, 64, True) == 0
    assert ar.padded_rows(260, H, bf, 256, True) == 0
    assert ar.padded_rows(8, H, jnp.float32, 256, True) == 0  # H8 is bf16 only
    assert ar.padded_rows(0, H, bf, 256, True) == 0


def test_flag_parse_and_compile_factors(monkeypatch):
    monkeypatch.delenv(ar.PAD16_ENV, raising=False)
    monkeypatch.delenv(ar.MAX_ROWS_ENV, raising=False)
    monkeypatch.delenv(ar.ENV, raising=False)
    assert ar.pad16() is False
    assert ar.compile_factors() == {}
    monkeypatch.setenv(ar.PAD16_ENV, "1")
    assert ar.pad16() is True
    assert ar.compile_factors() == {}  # the AR flag itself is off: no factors at all
    monkeypatch.setenv(ar.ENV, "h8")
    assert ar.compile_factors() == {ar.ENV: "h8", ar.MAX_ROWS_ENV: "256", ar.PAD16_ENV: "1"}
    monkeypatch.setenv(ar.PAD16_ENV, "")
    assert ar.pad16() is False
    assert ar.compile_factors() == {ar.ENV: "h8", ar.MAX_ROWS_ENV: "256"}
    for bad in ("0", "true", "yes", "2"):
        monkeypatch.setenv(ar.PAD16_ENV, bad)
        with pytest.raises(ValueError):
            ar.pad16()


def test_positive_control(mesh_pairs):
    mesh, pairs = mesh_pairs
    x = _normal(8, 3)
    y = x.copy()
    y[5, 7, 1000] += 64.0
    a, b = _run(mesh, pairs, x), _run(mesh, pairs, y)
    moved = _bits(a) != _bits(b)
    assert moved[:, 7, 1000].all() and int(moved.sum()) == 8

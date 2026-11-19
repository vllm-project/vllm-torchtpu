# SPDX-License-Identifier: Apache-2.0
"""q38 AR-FIX: the decode TP all-reduce op's JAX function (``tp8_ar_decode_fn.global_fn``) on 8 host devices (CPU).

The op body runs in the Pallas TPU interpreter under shard_map at Qwen3.8-Flash's hidden 2560, at the PD decode
buckets (16 / 32 / 48 / 64 rows, MTP k=3 at 4 / 8 / 12 / 16 requests) and 128 / 256, with the mesh in the engine's
device order (0, 1, 4, 5, 6, 7, 2, 3: the served all-reduce's replica group) and in id order.

Pre-stated tolerances (the reduction order differs from XLA's ring, so XLA is not the bitwise reference):
- every member's output is bitwise equal (one result per step on every rank);
- h8 (bf16 wire) is bitwise H8's numpy order model ``reference(wire="bf16")``; h8x (f32 wire) is bitwise
  ``reference_f32``, the single-rounding f32 sum;
- against the exact sum (float64, rounded once to bf16): h8x within 1 bf16 ulp everywhere, h8 within 1 ulp on at
  least 85 % of N(0, 1) elements (two roundings; T39 measured 11.9 % > 1 ulp on device, M);
- rows the kernel does not take (not a multiple of 16, or above the cap) come from ``lax.psum`` inside the op:
  equal on every member and within 2 bf16 ulps of the exact sum on 99.9 % of elements (XLA:CPU's own order);
- a positive control: one changed input element moves exactly that output element on every member.

Needs 8 host devices: JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 (skipped otherwise).
The JAX half lives in kernels/collectives/tp8_ar_decode_fn.py (no torch); vLLM's logger is stubbed when vLLM is absent.
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
ROWS = (16, 32, 48, 64)


@pytest.fixture(scope="module")
def devs():
    if jax.default_backend() != "cpu" or jax.device_count() < 8:
        pytest.skip("needs 8 host devices (see module docstring)")
    return {d.id: d for d in jax.devices()[:8]}


def _mesh(devs, order):
    return Mesh(np.asarray([devs[i] for i in order]), ("d",))


def _bits(a):
    return np.asarray(a, np.float32).view(np.uint32)


def _run(mesh, pairs, x, wire, max_rows=h8.MAX_ROWS):
    fn = ar.global_fn(mesh, axis="d", pairs=pairs, wire=wire, max_rows_=max_rows, interpret=True)
    xs = jax.device_put(jnp.asarray(x.reshape(8 * x.shape[1], H), jnp.bfloat16), NamedSharding(mesh, P("d")))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = jax.jit(fn)(xs)
    return np.asarray(out).astype(np.float32).reshape(8, x.shape[1], H)


def _normal(rows, seed):
    rng = np.random.default_rng(seed)
    return h8._bf16_round(rng.standard_normal((8, rows, H)).astype(np.float32))


def _exact(x):
    return h8._bf16_round(np.asarray(x, np.float64).sum(0).astype(np.float32))


def _ulps(a, b):
    return h8.ulp_distance(a, b)


def _pairs(order):
    return ar.pairs_from_device_ids(order)


def test_pairs_from_device_ids():
    assert _pairs(ENGINE_ORDER) == ((0, 1), (6, 7), (2, 3), (4, 5))
    assert _pairs(tuple(range(8))) == ((0, 1), (2, 3), (4, 5), (6, 7))
    with pytest.raises(RuntimeError):
        _pairs((0, 2, 4, 6, 8, 10, 12, 14))  # one core per chip
    with pytest.raises(RuntimeError):
        _pairs((0, 1, 2, 3, 4, 5))  # three chips


@pytest.mark.parametrize("order", [ENGINE_ORDER, tuple(range(8))], ids=["engine", "id"])
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("wire", ["bf16", "f32"])
def test_kernel_rows_match_declared_order(devs, order, rows, wire):
    mesh, pairs = _mesh(devs, order), _pairs(order)
    x = _normal(rows, 1000 + rows)
    out = _run(mesh, pairs, x, wire)
    assert all(np.array_equal(_bits(out[0]), _bits(out[r])) for r in range(8)), "members disagree"
    ref = h8.reference(x, pairs, wire)
    bad = int((_bits(out[0]) != _bits(ref)).sum())
    assert bad == 0, f"{bad} elements differ from H8's {wire}-wire order"
    d = _ulps(out[0], _exact(x))
    if wire == "f32":
        assert int(d.max()) <= 1, f"h8x max {int(d.max())} ulp from the exact sum"
    else:
        assert float((d <= 1).mean()) >= 0.85, f"h8 only {float((d <= 1).mean()):.3f} within 1 ulp"


@pytest.mark.parametrize("rows,cap", [(8, 256), (24, 256), (80, 64)])
def test_rows_outside_the_kernel_use_xla_psum(devs, rows, cap):
    mesh, pairs = _mesh(devs, ENGINE_ORDER), _pairs(ENGINE_ORDER)
    x = _normal(rows, 7 + rows)
    out = _run(mesh, pairs, x, "bf16", max_rows=cap)
    assert all(np.array_equal(_bits(out[0]), _bits(out[r])) for r in range(8))
    d = _ulps(out[0], _exact(x))
    assert float((d <= 2).mean()) >= 0.999


def test_engagement_kernel_vs_psum_in_hlo(devs):
    """The kernel is in the program at a kernel row count and absent at a fallback row count (and vice versa
    for the XLA all-reduce)."""
    mesh, pairs = _mesh(devs, ENGINE_ORDER), _pairs(ENGINE_ORDER)
    for rows, want_kernel in ((16, True), (64, True), (24, False)):
        fn = ar.global_fn(mesh, axis="d", pairs=pairs, wire="bf16", max_rows_=256, interpret=True)
        arg = jax.ShapeDtypeStruct((8 * rows, H), jnp.bfloat16, sharding=NamedSharding(mesh, P("d")))
        txt = jax.jit(fn).lower(arg).as_text()
        has_ar = "all_reduce" in txt or "all-reduce" in txt
        # Interpret mode lowers the kernel to its interpreter body; the psum path lowers to an all-reduce.
        assert has_ar != want_kernel, (rows, has_ar)


def test_positive_control(devs):
    mesh, pairs = _mesh(devs, ENGINE_ORDER), _pairs(ENGINE_ORDER)
    x = _normal(16, 3)
    y = x.copy()
    y[5, 7, 1000] += 64.0
    a, b = _run(mesh, pairs, x, "bf16"), _run(mesh, pairs, y, "bf16")
    moved = _bits(a) != _bits(b)
    assert moved[:, 7, 1000].all() and int(moved.sum()) == 8


def test_back_to_back_calls_in_one_program(devs):
    """Barrier and DMA semaphores reused across consecutive calls (the decode forward issues ~50 per step)."""
    mesh, pairs = _mesh(devs, ENGINE_ORDER), _pairs(ENGINE_ORDER)
    rows = 16
    x = np.ldexp(_normal(rows, 11), -40).astype(np.float32)  # scaled so 3 chained sums stay finite and exact-ish
    body = ar.global_fn(mesh, axis="d", pairs=pairs, wire="f32", max_rows_=256, interpret=True)
    xs = jax.device_put(jnp.asarray(x.reshape(8 * rows, H), jnp.bfloat16), NamedSharding(mesh, P("d")))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = np.asarray(jax.jit(lambda v: body(body(body(v))))(xs)).astype(np.float32).reshape(8, rows, H)
    r1 = h8.reference_f32(x, pairs)
    r2 = h8.reference_f32(np.broadcast_to(r1, (8, rows, H)), pairs)
    r3 = h8.reference_f32(np.broadcast_to(r2, (8, rows, H)), pairs)
    assert all(np.array_equal(_bits(out[0]), _bits(out[r])) for r in range(8))
    assert int((_bits(out[0]) != _bits(r3)).sum()) == 0

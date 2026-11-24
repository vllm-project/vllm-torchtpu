# SPDX-License-Identifier: Apache-2.0
"""MXFP4 transforms of VLLM_TPU_OFFLOAD_FP4 (offload/fp4.py), on CPU JAX."""

import importlib.util
import pathlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
lax = jax.lax

# Load fp4.py on its own: the package imports pull in the TPU runtime.
_SPEC = importlib.util.spec_from_file_location(
    "fp4", pathlib.Path(__file__).parents[2] / "src/vllm_torchtpu/offload/fp4.py"
)


@pytest.fixture(scope="module")
def fp4(monkeypatch_module=None):
    import sys
    import types

    logger_mod = types.ModuleType("vllm_torchtpu.logger")
    logger_mod.init_logger = lambda name: __import__("logging").getLogger(name)
    sys.modules.setdefault("vllm_torchtpu", types.ModuleType("vllm_torchtpu"))
    sys.modules["vllm_torchtpu.logger"] = logger_mod
    if "torch" not in sys.modules:
        stub = types.ModuleType("torch")
        stub.Tensor = object  # annotations are evaluated (no __future__ import in fp4.py)
        sys.modules["torch"] = stub
    mod = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(mod)
    return mod


def _fp8_pages(rng, n, rows, scale=4.0):
    vals = rng.standard_normal((n, rows, 4, 256)).astype(np.float32) * scale
    f8 = jnp.asarray(vals).astype(jnp.float8_e4m3fn)
    return np.asarray(lax.bitcast_convert_type(f8, jnp.uint8)), np.asarray(f8.astype(jnp.float32))


def test_geometry(fp4):
    assert fp4.page_rows(3_276_800) == 3200
    assert fp4.fp4_page_bytes(3_276_800) == 1_740_800
    assert fp4.staging_shape(3_276_800, 7) == (7, 13_600, 128)
    assert 13_600 * 128 == fp4.fp4_page_bytes(3_276_800)


def test_round_trip_error(fp4):
    rng = np.random.default_rng(0)
    rows = 32
    pages, ref = _fp8_pages(rng, 3, rows)
    staged = fp4.quantize_pages(jnp.asarray(pages), jnp, lax)
    assert staged.shape == (3, rows * 4 + rows // 4, 128) and staged.dtype == jnp.uint8
    back_u8 = fp4.dequantize_pages(staged, rows, jnp, lax)
    assert back_u8.shape == pages.shape
    back = np.asarray(lax.bitcast_convert_type(back_u8, jnp.float8_e4m3fn).astype(jnp.float32))
    assert np.isfinite(back).all()
    rel = np.sqrt(np.mean((back - ref) ** 2)) / np.sqrt(np.mean(ref**2))
    # e2m1 with a per-32 power-of-two scale: ~10% RMS on Gaussian values.
    assert rel < 0.16, rel
    # Each value lands on the grid scaled by its group's power of two.
    g = ref.reshape(3, rows, 4, 8, 32)
    amax = np.abs(g).max(-1, keepdims=True)
    assert (np.abs(back.reshape(g.shape)) <= 2 * amax + 1e-6).all()


def test_codes_match_reference(fp4):
    rng = np.random.default_rng(1)
    rows = 4
    pages, ref = _fp8_pages(rng, 1, rows)
    staged = np.asarray(fp4.quantize_pages(jnp.asarray(pages), jnp, lax))
    packed = staged[0, : rows * 4].reshape(rows, 4, 128)
    nib = np.stack([packed & 0xF, packed >> 4], -1).reshape(rows, 4, 8, 32)
    codes, e = fp4.quantize_reference(ref[0].reshape(rows, 4, 8, 32).astype(np.float64))
    np.testing.assert_array_equal(nib, codes)
    scales = staged[0, rows * 4 :].reshape(rows, 4, 8, 1).astype(np.int64) - fp4.SCALE_BIAS
    np.testing.assert_array_equal(scales, e)


def test_extremes_and_zeros(fp4):
    rows = 4
    vals = np.zeros((1, rows, 4, 256), np.float32)
    vals[0, 0, 0, :32] = 448.0  # FP8 max: must not overflow on the way back
    vals[0, 0, 1, :32] = -0.001953125  # FP8 smallest subnormal region
    f8 = jnp.asarray(vals).astype(jnp.float8_e4m3fn)
    pages = lax.bitcast_convert_type(f8, jnp.uint8)
    back = fp4.dequantize_pages(fp4.quantize_pages(pages, jnp, lax), rows, jnp, lax)
    out = np.asarray(lax.bitcast_convert_type(back, jnp.float8_e4m3fn).astype(jnp.float32))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[0, 0, 0, :32], 448.0)
    assert (out[0, 1:] == 0).all() and (out[0, 0, 2:] == 0).all()


def test_slot_pool(fp4):
    pool = fp4.SlotPool(4)
    a = pool.take(3)
    assert sorted(a) == [0, 1, 2] and pool.free == 1
    assert pool.take(2) is None
    pool.give(a)
    assert pool.free == 4
    with pytest.raises(AssertionError):
        pool.give([0])

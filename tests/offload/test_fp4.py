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
    assert fp4.packed_rows(3200) == 1700  # 3200 * 544 B / 1024 B
    assert fp4.stage_rows(3200) == 1700
    assert fp4.fp4_page_bytes(3_276_800) == 1_740_800
    assert fp4.staging_shape(3_276_800, 7) == (7, 1700, 4, 256)


def test_staging_packing_round_trip(fp4):
    # The programs' reshape: [n, rows*4 + rows//4, 128] -> [n, packed_rows, 4, 256],
    # zero-padded to stage_rows, then sliced and reshaped back before dequantizing.
    rng = np.random.default_rng(2)
    rows = 64
    pages, _ = _fp8_pages(rng, 2, rows)
    q = fp4.quantize_pages(jnp.asarray(pages), jnp, lax)
    prow, srow = fp4.packed_rows(rows), fp4.stage_rows(rows)
    staged = jnp.pad(q.reshape(2, prow, 4, 256), ((0, 0), (0, srow - prow), (0, 0), (0, 0)))
    back = staged[:, :prow].reshape(q.shape)
    np.testing.assert_array_equal(np.asarray(back), np.asarray(q))
    a = fp4.dequantize_pages(back, rows, jnp, lax)
    b = fp4.dequantize_pages(q, rows, jnp, lax)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


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
    rows = 32
    pages, ref = _fp8_pages(rng, 1, rows)
    staged = np.asarray(fp4.quantize_pages(jnp.asarray(pages), jnp, lax))
    packed = staged[0, : rows * 4].reshape(rows, 4, 128)
    nib = np.stack([packed & 0xF, packed >> 4], -1).reshape(rows, 4, 8, 32)
    codes, e = fp4.quantize_reference(ref[0].reshape(rows, 4, 8, 32).astype(np.float64))
    np.testing.assert_array_equal(nib, codes)
    scales = staged[0, rows * 4 :].reshape(rows, 4, 8, 1).astype(np.int64) - fp4.SCALE_BIAS
    np.testing.assert_array_equal(scales, e)


def test_extremes_and_zeros(fp4):
    rows = 32
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


def test_fp8_input_matches_bytes(fp4):
    # The device path quantizes FP8 values directly; the bytes path must agree.
    rng = np.random.default_rng(3)
    pages, _ = _fp8_pages(rng, 1, 32)
    f8 = lax.bitcast_convert_type(jnp.asarray(pages), jnp.float8_e4m3fn)
    a = fp4.quantize_pages(f8, jnp, lax)
    b = fp4.quantize_pages(jnp.asarray(pages), jnp, lax)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    d8 = fp4.dequantize_pages(a, 32, jnp, lax, as_fp8=True)
    du = fp4.dequantize_pages(a, 32, jnp, lax)
    np.testing.assert_array_equal(np.asarray(lax.bitcast_convert_type(d8, jnp.uint8)), np.asarray(du))


def test_page_scatter_interpret(fp4):
    # page_scatter's kernel in Pallas TPU interpret mode on CPU, when available.
    pl = pytest.importorskip("jax.experimental.pallas")
    target = jnp.zeros((5, 8, 4, 256), jnp.uint8)
    values = jnp.arange(2 * 8 * 4 * 256, dtype=jnp.int32).astype(jnp.uint8).reshape(2, 8, 4, 256)
    try:
        from jax.experimental.pallas import tpu as pltpu  # noqa: F401
        orig = pl.pallas_call
        pl.pallas_call = lambda *a, **k: orig(*a, **{**k, "interpret": True})
        out = fp4.page_scatter(target, values, jnp.asarray([3, 1], jnp.int32))
    except Exception as exc:  # pragma: no cover - interpret mode not supported here
        pytest.skip(f"interpret mode unavailable: {exc}")
    finally:
        pl.pallas_call = orig
    out = np.asarray(out)
    np.testing.assert_array_equal(out[3], np.asarray(values[0]))
    np.testing.assert_array_equal(out[1], np.asarray(values[1]))
    assert (out[[0, 2, 4]] == 0).all()


def test_page_gather_interpret(fp4):
    pl = pytest.importorskip("jax.experimental.pallas")
    src = jnp.arange(4 * 8 * 4 * 256, dtype=jnp.int32).astype(jnp.uint8).reshape(4, 8, 4, 256)
    orig = pl.pallas_call
    try:
        pl.pallas_call = lambda *a, **k: orig(*a, **{**k, "interpret": True})
        out = fp4.page_gather(src, jnp.asarray([2, 0, 2], jnp.int32))
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"interpret mode unavailable: {exc}")
    finally:
        pl.pallas_call = orig
    np.testing.assert_array_equal(np.asarray(out), np.asarray(src)[[2, 0, 2]])


# -- VLLM_TPU_OFFLOAD_FP4_GDN ------------------------------------------------------


def test_gdn_geometry(fp4):
    assert fp4.gdn_packed_rows(3200) == 425  # 3200 rows x 136 B / 1024 B
    assert fp4.fp4_gdn_page_bytes(3_276_800) == 435_200
    with pytest.raises(ValueError):
        fp4.gdn_packed_rows(96)  # 96 x 136 B is not whole 1,024-B rows


def test_slot_layout(fp4):
    assert fp4.slot_layout(1024, 0) == {"main": (0, 1024)}
    assert fp4.slot_layout(1024, 512) == {"main": (0, 1024), "gdn": (1024, 512)}


def test_gdn_rows_typed_view_round_trip(fp4):
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 8, 256)).astype(np.float32)
    rows = fp4.gdn_rows_from_f32(x)
    assert rows.shape == (2, 8, 4, 256) and rows.dtype == np.uint8
    np.testing.assert_array_equal(fp4.gdn_f32_from_rows(rows), x)
    # The JAX view agrees with the NumPy one (pd_staging._typed's convention).
    w = np.asarray(fp4._words(jnp.asarray(rows), jnp, lax))
    np.testing.assert_array_equal(w, x)
    np.testing.assert_array_equal(np.asarray(fp4._unwords(jnp.asarray(x), jnp, lax)), rows)


def test_gdn_round_trip_error(fp4):
    rng = np.random.default_rng(5)
    rows = 128
    x = rng.standard_normal((2, rows, 256)).astype(np.float32)
    x *= np.float32(10.0) ** rng.uniform(-4, 2, (2, rows, 1)).astype(np.float32)
    raw = jnp.asarray(fp4.gdn_rows_from_f32(x))
    q = fp4.quantize_gdn_pages(raw, jnp, lax)
    assert q.shape == (2, fp4.gdn_packed_rows(rows), 4, 256) and q.dtype == jnp.uint8
    back = fp4.gdn_f32_from_rows(np.asarray(fp4.dequantize_gdn_pages(q, rows, jnp, lax)))
    assert np.isfinite(back).all()
    row_rel = np.sqrt(np.mean((back - x) ** 2, -1)) / np.sqrt(np.mean(x**2, -1))
    assert row_rel.mean() < 0.16 and row_rel.max() < 0.3, (row_rel.mean(), row_rel.max())


def test_gdn_grid_values_are_exact(fp4):
    # Values already on an e2m1 grid times a power of two survive unchanged; zeros stay 0.
    rows = 128
    grid = np.array(fp4.E2M1_GRID + tuple(-v for v in fp4.E2M1_GRID), np.float32)
    x = np.resize(grid, (1, rows, 256)).astype(np.float32) * np.float32(2.0**-20)
    x[0, 5] = 0.0
    raw = jnp.asarray(fp4.gdn_rows_from_f32(x))
    back = fp4.gdn_f32_from_rows(
        np.asarray(fp4.dequantize_gdn_pages(fp4.quantize_gdn_pages(raw, jnp, lax), rows, jnp, lax))
    )
    np.testing.assert_array_equal(back, x)


def test_page_gather_u8_interpret(fp4):
    pl = pytest.importorskip("jax.experimental.pallas")
    src_u8 = (np.arange(3 * 64 * 1 * 4 * 256) % 251).astype(np.uint8).reshape(3, 64, 1, 4, 256)
    src = lax.bitcast_convert_type(jnp.asarray(src_u8), jnp.float8_e4m3fn)
    orig = pl.pallas_call
    try:
        pl.pallas_call = lambda *a, **k: orig(*a, **{**k, "interpret": True})
        out = fp4.page_gather_u8(src, jnp.asarray([2, 0], jnp.int32))
    except Exception as exc:  # pragma: no cover - interpret mode lacks the vreg bitcast
        pytest.skip(f"interpret mode unavailable: {exc}")
    finally:
        pl.pallas_call = orig
    want = src_u8[[2, 0]].reshape(-1, 4, 256)
    np.testing.assert_array_equal(np.asarray(out), want)

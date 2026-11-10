"""TPU_QMM_ACT_F32_AMAX: xla_quantized_matmul with the activation row max on f32
gives the same bits as the bf16 row max, and the flag-off lowering is untouched."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.quantized_matmul import util as U

FP8 = jnp.float8_e4m3fn


def _cases():
    rng = np.random.default_rng(20261102)
    out = [rng.standard_normal((r, c)).astype(np.float32) * s
           for r, c, s in [(64, 512, 1.0), (40, 256, 3.0), (7, 128, 1e3), (128, 256, 1e-3)]]
    x = rng.standard_normal((64, 512)).astype(np.float32)
    x[np.arange(64), rng.integers(0, 512, 64)] = rng.choice([-1, 1], 64) * rng.uniform(5, 500, 64)
    out.append(x)
    edge = np.zeros((10, 256), np.float32)
    edge[0] = -rng.uniform(0, 1, 256); edge[0, 3] = -7.5
    edge[2] = -0.0; edge[3, 100] = 1e-3
    edge[4] = rng.uniform(-1, 1, 256) * 1e-39; edge[5] = rng.uniform(-1, 1, 256) * 1e4
    edge[6] = rng.uniform(-1, 1, 256) * 3e38; edge[7] = 448.0; edge[7, 0] = -448.0
    edge[8] = rng.uniform(-1, 1, 256); edge[8, 9] = 2.0 ** -126; edge[9] = rng.standard_normal(256) * 1e-30
    out.append(edge)
    sweep = rng.uniform(-1, 1, (512, 128)).astype(np.float32)
    mags = (2.0 ** rng.uniform(-60, 60, 512)).astype(np.float32)
    sweep *= mags[:, None]
    sweep[np.arange(512), rng.integers(0, 128, 512)] = mags * rng.choice([-1, 1], 512)
    out.append(sweep)
    return [jnp.asarray(c, jnp.bfloat16) for c in out]


def _w(k, n, seed=3):
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.standard_normal((k, n)) * 0.5, jnp.float32).astype(FP8),
            jnp.asarray(rng.uniform(0.5, 2.0, n), jnp.float32))


def _u(a, dt):
    return np.asarray(jax.lax.bitcast_convert_type(a, dt))


@pytest.mark.parametrize("jit", [False, True])
def test_quantize_activation_bitwise(monkeypatch, jit):
    monkeypatch.setenv("TPU_QMM_ACT_F32_AMAX", "1")
    rows = 0
    for x in _cases():
        new = (jax.jit(lambda v: U._quantize_activation(v, FP8)) if jit else (lambda v: U._quantize_activation(v, FP8)))(x)
        ref = (jax.jit(lambda v: U.quantize_tensor(v, FP8)) if jit else (lambda v: U.quantize_tensor(v, FP8)))(x)
        np.testing.assert_array_equal(_u(new[1], jnp.uint32), _u(ref[1], jnp.uint32))
        np.testing.assert_array_equal(_u(new[0], jnp.uint8), _u(ref[0], jnp.uint8))
        rows += x.shape[0]
    assert rows >= 800


def test_matmul_bitwise_flag_on_vs_off(monkeypatch):
    for x in _cases()[:5]:
        w, s = _w(x.shape[1], 384)
        monkeypatch.delenv("TPU_QMM_ACT_F32_AMAX", raising=False)
        off = jax.jit(lambda a, b, c: U.xla_quantized_matmul(a, b, c))(x, w, s)
        monkeypatch.setenv("TPU_QMM_ACT_F32_AMAX", "1")
        on = jax.jit(lambda a, b, c: U.xla_quantized_matmul(a, b, c))(x, w, s)
        np.testing.assert_array_equal(_u(on, jnp.uint16), _u(off, jnp.uint16))


def _lower(fn, *a):
    return jax.jit(lambda *v: fn(*v)).lower(*a).as_text()


def test_lowering_flag_off_identical_flag_on_engages(monkeypatch):
    x = jnp.zeros((4352, 512), jnp.bfloat16); w, s = _w(512, 256)
    monkeypatch.delenv("TPU_QMM_ACT_F32_AMAX", raising=False)
    off = _lower(U.xla_quantized_matmul, x, w, s)
    ref = _lower(lambda a, b, c: _ref_matmul(a, b, c), x, w, s)
    assert off == ref and "-> tensor<4352xbf16>" in off
    monkeypatch.setenv("TPU_QMM_ACT_F32_AMAX", "1")
    on = _lower(U.xla_quantized_matmul, x, w, s)
    assert "-> tensor<4352xbf16>" not in on and "-> tensor<4352xf32>" in on
    # non-bf16 rows and integer weights keep the original path
    xf = jnp.zeros((64, 512), jnp.float32)
    assert _lower(U.xla_quantized_matmul, xf, w, s) == _lower(lambda a, b, c: _ref_matmul(a, b, c), xf, w, s)
    wi = jnp.zeros((512, 256), jnp.int8)
    assert _lower(U.xla_quantized_matmul, x, wi, s) == _lower(lambda a, b, c: _ref_matmul(a, b, c), x, wi, s)


def _ref_matmul(x, w_q, w_scale):
    """The original xla_quantized_matmul body (quantize_activation=True)."""
    acc = jnp.int32 if jnp.issubdtype(w_q.dtype, jnp.integer) else jnp.float32
    x_q, x_scale = U.quantize_tensor(x, w_q.dtype)
    out = jax.lax.dot_general(x_q, w_q, dimension_numbers=(((1,), (0,)), ((), ())),
                              preferred_element_type=acc).astype(jnp.float32)
    out *= x_scale
    out *= jnp.expand_dims(w_scale, 0)
    return out.astype(x.dtype)


def test_post_reduce_arithmetic_matches_quantize_tensor(monkeypatch):
    """Same ops after the row max as quantize_block (XLA:CPU here rewrites x / c
    as x * (1 / c), so a reciprocal-multiply mutant is invisible numerically on
    CPU; on TPU the division differs, so check the ops)."""
    monkeypatch.setenv("TPU_QMM_ACT_F32_AMAX", "1")
    x = jnp.zeros((64, 512), jnp.bfloat16)
    new = _lower(lambda v: U._quantize_activation(v, FP8), x)
    ref = _lower(lambda v: U.quantize_tensor(v, FP8), x)
    for op in ("stablehlo.divide", "stablehlo.multiply", "stablehlo.select", "stablehlo.compare",
               "stablehlo.maximum", "stablehlo.minimum"):
        assert new.count(op) == ref.count(op), (op, new.count(op), ref.count(op))

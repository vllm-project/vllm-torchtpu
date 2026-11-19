# SPDX-License-Identifier: Apache-2.0
"""split_e4m3_matmul (TPU_MINIMAX_M3_FP8_SPLIT_MAX_ROWS) against an f64 oracle and the
incumbent bf16-activation path (x @ bf16(w) * scale)."""

import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_PATH = (pathlib.Path(__file__).parents[3] / "src/vllm_torchtpu/models/minimax_m3/fp8_split.py")
_spec = importlib.util.spec_from_file_location("fp8_split", _PATH)
fp8_split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp8_split)


def _case(rows, k, n, seed, outliers=False):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, k)).astype(np.float32)
    if outliers:
        x[:, rng.choice(k, max(1, k // 200), replace=False)] *= 40.0
    x = jnp.asarray(x, jnp.bfloat16)
    w = jnp.asarray(rng.standard_normal((k, n)) * 0.5, jnp.float8_e4m3fn)
    s = jnp.asarray(2.0 ** rng.integers(-8, -3, n), jnp.bfloat16)
    return x, w, s


def _oracle(x, w, s):
    return (np.asarray(x, np.float64) @ np.asarray(w, np.float64)) * np.asarray(s, np.float64)


def _rel(a, ref):
    return float(np.linalg.norm(np.asarray(a, np.float64) - ref) / np.linalg.norm(ref))


@pytest.mark.parametrize("rows", [1, 8, 32, 64])
@pytest.mark.parametrize("outliers", [False, True])
def test_split_matches_bf16_path_within_bf16_rounding(rows, outliers):
    x, w, s = _case(rows, 1024, 512, rows, outliers)
    ref = _oracle(x, w, s)
    split = fp8_split.split_e4m3_matmul(x, w, s)
    inc = (jnp.dot(x, w.astype(jnp.bfloat16), preferred_element_type=jnp.float32) * s.astype(jnp.float32)).astype(jnp.bfloat16)
    assert split.shape == (rows, 512) and split.dtype == jnp.bfloat16
    e_split, e_inc = _rel(split, ref), _rel(inc, ref)
    # The incumbent only rounds the output to bf16 (~1.6e-3 rel); the split adds the
    # activation's 2-term rounding (~2^-8 per element, averaged over K).
    assert e_split < 4e-3, (e_split, e_inc)
    assert e_split < 2.5 * max(e_inc, 1e-3), (e_split, e_inc)


def test_zero_row_and_3d_shape():
    x, w, s = _case(6, 256, 128, 7)
    x = x.at[2].set(0)
    out = fp8_split.split_e4m3_matmul(x.reshape(2, 3, 256), w, s)
    assert out.shape == (2, 3, 128)
    assert bool(jnp.all(out.reshape(6, 128)[2] == 0))
    assert bool(jnp.all(jnp.isfinite(out.astype(jnp.float32))))


def test_fault_injection_single_term_fails():
    """Positive control: dropping the low term (plain per-row W8A8) must fail the gate."""
    x, w, s = _case(32, 1024, 512, 3)
    ref = _oracle(x, w, s)
    x2 = x.astype(jnp.float32)
    sc = jnp.exp2(jnp.ceil(jnp.log2(jnp.max(jnp.abs(x2), -1, keepdims=True) / 448.0)))
    hi = (x2 / sc).astype(jnp.float8_e4m3fn)
    one = (jnp.dot(hi, w, preferred_element_type=jnp.float32) * sc * s.astype(jnp.float32)).astype(jnp.bfloat16)
    assert _rel(one, ref) > 4e-3

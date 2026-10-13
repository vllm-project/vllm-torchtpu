# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One-pass Pallas (fused add +) RMSNorm vs ``vllm.ir.ops`` numerics.

The new residual must be bit-identical. The normalized output may differ only
where the f32 summation order of ``mean(s^2)`` moves ``rsqrt`` by an ulp: at
most two bf16 ulp (one in the rounded normed value, carried through the
weight multiply), on a tiny fraction of elements. ``test_checker_rejects_*``
are the positive controls: the same checker must fail a wrong kernel.
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from vllm_torchtpu.kernels import fused_add_rmsnorm as K

EPS = 1e-5
MAX_ULP_FRACTION = 1e-3


def _inputs(rows, hidden, seed=0, res_scale=4.0):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    x = (jax.random.normal(k1, (rows, hidden), jnp.float32) * 0.7).astype(jnp.bfloat16)
    r = (jax.random.normal(k2, (rows, hidden), jnp.float32) * res_scale).astype(
        jnp.bfloat16
    )
    w = (1 + 0.2 * jax.random.normal(k3, (hidden,), jnp.float32)).astype(jnp.bfloat16)
    return x, r, w


def _oracle(x, r, w, eps=EPS):
    """``vllm.ir.ops`` torch semantics on the host, every bf16 rounding kept.

    Not XLA: a jitted JAX reference lets XLA drop the intermediate rounding
    (excess precision), which is not what the torch graph specifies.
    """
    bf = ml_dtypes.bfloat16
    s = np.asarray(x, np.float32)
    if r is not None:
        s = s + np.asarray(r, np.float32)
    var = np.mean(s.astype(np.float64) ** 2, axis=-1, keepdims=True).astype(np.float32)
    inv = (1.0 / np.sqrt(var + np.float32(eps))).astype(np.float32)
    normed = (s * inv).astype(bf)
    y = (normed.astype(np.float32) * np.asarray(w, np.float32)).astype(bf)
    return (y, s.astype(bf)) if r is not None else y


def _ulp_diff(a, b):
    ai = np.asarray(a).view(np.int16).astype(np.int32)
    bi = np.asarray(b).view(np.int16).astype(np.int32)
    return np.abs(ai - bi)


def _check(y, y_ref, res=None, res_ref=None):
    """True iff the kernel output is within the documented contract."""
    if res is not None and not np.array_equal(
        np.asarray(res).view(np.int16), np.asarray(res_ref).view(np.int16)
    ):
        return False
    d = _ulp_diff(y, y_ref)
    # one ulp of rsqrt -> one ulp of the rounded normed value -> <= 2 ulp after * weight
    return bool(d.max() <= 2 and (d > 0).mean() <= MAX_ULP_FRACTION)


def _run(x, r, w, interpret, **kw):
    return K.fused_add_rms_norm(x, r, w, eps=EPS, interpret=interpret, **kw)


@pytest.mark.parametrize("has_res", [True, False])
@pytest.mark.parametrize("rows,hidden,block", [(256, 512, 64), (384, 1024, 128)])
def test_interpret_matches_reference(has_res, rows, hidden, block):
    if jax.devices()[0].platform == "tpu":
        # interpret mode on a TPU host runs the body through XLA, which drops
        # the intermediate bf16 rounding (excess precision); the compiled
        # kernel is gated by the test_tpu_* cases instead.
        pytest.skip("interpret mode is a host-only check")
    x, r, w = _inputs(rows, hidden, seed=rows)
    r = r if has_res else None
    got = _run(x, r, w, True, block_rows=block)
    ref = _oracle(x, r, w)
    if has_res:
        assert _check(got[0], ref[0], got[1], ref[1])
    else:
        assert _check(got, ref)


def test_supports():
    assert K.supports(4096, 6144)
    assert not K.supports(16, 6144)
    assert not K.supports(4096 + 8, 6144)
    assert not K.supports(4096, 6100)


def test_checker_rejects_wrong_residual_and_wrong_norm():
    x, r, w = _inputs(256, 512)
    y_ref, res_ref = _oracle(x, r, w)
    # residual off by one ulp in one element
    bad_res = res_ref.copy()
    bad_res.view(np.int16)[3, 7] += 1
    assert not _check(y_ref, y_ref, bad_res, res_ref)
    # a norm that ignores the residual
    y_nores = _oracle(x, None, w)
    assert not _check(y_nores, y_ref, res_ref, res_ref)
    # a norm with the weight applied before the bf16 rounding (one rounding fewer)
    s = np.asarray(x, np.float32) + np.asarray(r, np.float32)
    var = np.mean(s.astype(np.float64) ** 2, -1, keepdims=True).astype(np.float32)
    inv = (1.0 / np.sqrt(var + np.float32(EPS))).astype(np.float32)
    y_onefewer = (s * inv * np.asarray(w, np.float32)).astype(ml_dtypes.bfloat16)
    assert not _check(y_onefewer, y_ref, res_ref, res_ref)


def _require_tpu():
    # Checked inside the test, not in a module-level skipif: touching the
    # backend at collection claims the TPU for the whole pytest process.
    if jax.devices()[0].platform != "tpu":
        pytest.skip("compiled Mosaic kernel needs a TPU")


@pytest.mark.parametrize("has_res", [True, False])
@pytest.mark.parametrize("rows", [512, 1024, 2048, 4096])
def test_tpu_matches_reference_at_served_shapes(has_res, rows):
    _require_tpu()
    x, r, w = _inputs(rows, 6144, seed=rows + int(has_res))
    r = r if has_res else None
    got = jax.block_until_ready(_run(x, r, w, False))
    ref = _oracle(x, r, w)
    if has_res:
        assert _check(got[0], ref[0], got[1], ref[1])
    else:
        assert _check(got, ref)


def test_tpu_large_magnitude_residual():
    """Residual-stream magnitudes late in a deep model (|r| ~ 1e3)."""
    _require_tpu()
    x, r, w = _inputs(1024, 6144, seed=7, res_scale=1500.0)
    got = _run(x, r, w, False)
    ref = _oracle(x, r, w)
    assert _check(got[0], ref[0], got[1], ref[1])

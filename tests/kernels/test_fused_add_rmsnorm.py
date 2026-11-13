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
weight multiply), on at most 0.1% of the elements (MAX_ULP_FRACTION).
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from vllm_torchtpu.kernels import fused_add_rmsnorm as K
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor

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


def test_supports(monkeypatch):
    # 64 MiB, the VMEM of one TPU v7x TensorCore, so the VMEM cases also run
    # on a host without a TPU.
    monkeypatch.setattr(K, "_vmem_capacity_bytes", lambda: 64 * 1024 * 1024)
    assert K.supports(4096, 6144)
    assert not K.supports(16, 6144)
    assert not K.supports(4096 + 8, 6144)
    assert not K.supports(4096, 6100)
    # block_rows must be a multiple of 8 or equal num_rows.
    assert not K.supports(24, 6144, block_rows=12)
    assert K.supports(12, 6144, block_rows=12)
    # Estimated working set: 56 MiB at hidden 7168, 128 MiB at hidden 16384.
    assert K.supports(4096, 7168)
    assert not K.supports(4096, 16384)


def test_supports_skips_vmem_check_without_tpu():
    if jax.devices()[0].platform == "tpu":
        pytest.skip("checks the behavior on a host without a TPU")
    assert K._vmem_capacity_bytes() is None
    assert K.supports(4096, 16384)


def test_rejects_rows_not_multiple_of_block():
    x, r, w = _inputs(192, 512)
    with pytest.raises(ValueError, match="multiple of block_rows"):
        _run(x, r, w, True, block_rows=128)


def test_rejects_hidden_not_multiple_of_128():
    x, r, w = _inputs(64, 200)
    with pytest.raises(ValueError, match="hidden=200 must be a multiple of 128"):
        _run(x, r, w, True, block_rows=64)


def test_rejects_block_rows_not_multiple_of_8():
    x, r, w = _inputs(24, 512)
    with pytest.raises(ValueError, match="block_rows=12 must be a multiple of 8"):
        _run(x, r, w, True, block_rows=12)


def test_rejects_block_larger_than_vmem(monkeypatch):
    # The check runs before the kernel lowers, so it runs without a TPU. The
    # estimated working set at [256, 1024] bf16 with a residual is 8 MiB.
    monkeypatch.setattr(K, "_vmem_capacity_bytes", lambda: 1024 * 1024)
    x, r, w = _inputs(256, 1024)
    with pytest.raises(ValueError, match="bytes of VMEM"):
        _run(x, r, w, False, block_rows=256)


def _require_tpu():
    # Checked inside the test, not in a module-level skipif: touching the
    # backend at collection claims the TPU for the whole pytest process.
    if jax.devices()[0].platform != "tpu":
        pytest.skip("compiled Mosaic kernel needs a TPU")


@pytest.mark.tpu_test
@pytest.mark.parametrize("has_res", [True, False])
@pytest.mark.parametrize(
    "rows,hidden,block",
    [
        (512, 6144, 256),
        (1024, 6144, 256),
        (2048, 6144, 256),
        (4096, 6144, 256),
        (512, 4096, 256),
        (4096, 4096, 256),
        (512, 7168, 256),
        (4096, 7168, 256),
        # A block smaller than the default, at a decode-sized row count.
        (16, 4096, 16),
    ],
)
def test_tpu_matches_reference(has_res, rows, hidden, block):
    _require_tpu()
    x, r, w = _inputs(rows, hidden, seed=rows + hidden + int(has_res))
    r = r if has_res else None
    got = jax.block_until_ready(_run(x, r, w, False, block_rows=block))
    ref = _oracle(x, r, w)
    if has_res:
        assert _check(got[0], ref[0], got[1], ref[1])
    else:
        assert _check(got, ref)


@pytest.mark.tpu_test
def test_tpu_large_magnitude_residual():
    """Residual-stream magnitudes late in a deep model (|r| ~ 1e3)."""
    _require_tpu()
    x, r, w = _inputs(1024, 6144, seed=7, res_scale=1500.0)
    got = _run(x, r, w, False)
    ref = _oracle(x, r, w)
    assert _check(got[0], ref[0], got[1], ref[1])


def _require_host():
    if jax.devices()[0].platform == "tpu":
        # On a TPU host, XLA can drop the intermediate bf16 rounding (excess
        # precision), which these host checks rely on.
        pytest.skip("host-only check")


@pytest.mark.parametrize("has_res", [True, False])
def test_reference_matches_oracle(has_res):
    _require_host()
    x, r, w = _inputs(256, 512, seed=3)
    r = r if has_res else None
    got = jax.jit(K.reference, static_argnames=("eps",))(x, r, w, eps=EPS)
    ref = _oracle(x, r, w)
    if has_res:
        assert _check(got[0], ref[0], got[1], ref[1])
    else:
        assert _check(got, ref)


def test_quant_interpret_matches_quantize_tensor():
    """Interpret mode: (y_q, scale) equal quantize_tensor of the kernel's own y."""
    _require_host()
    x, r, w = _inputs(256, 512, seed=5)
    y, y_q, scale, res = K.fused_add_rms_norm_quant(
        x, r, w, eps=EPS, block_rows=64, interpret=True
    )
    q_ref, s_ref = jax.jit(quantize_tensor, static_argnums=1)(y, jnp.float8_e4m3fn)
    y_ref, res_ref = _oracle(x, r, w)
    assert _check(y, y_ref, res, res_ref)
    q_got, q_ref = np.asarray(y_q).view(np.uint8), np.asarray(q_ref).view(np.uint8)
    assert np.array_equal(q_got, q_ref)
    s_got, s_ref = np.asarray(scale).view(np.int32), np.asarray(s_ref).view(np.int32)
    assert np.array_equal(s_got, s_ref)


def test_quant_rejects_hidden_not_multiple_of_128():
    x, r, w = _inputs(64, 200)
    with pytest.raises(ValueError, match="hidden=200 must be a multiple of 128"):
        K.fused_add_rms_norm_quant(x, r, w, eps=EPS, block_rows=64, interpret=True)


def test_quant_rejects_block_rows_not_multiple_of_8():
    x, r, w = _inputs(24, 512)
    with pytest.raises(ValueError, match="block_rows=12 must be a multiple of 8"):
        K.fused_add_rms_norm_quant(x, r, w, eps=EPS, block_rows=12, interpret=True)


def test_quant_rejects_integer_quant_dtype():
    x, r, w = _inputs(64, 512)
    with pytest.raises(ValueError, match="quant_dtype=.* is not supported"):
        K.fused_add_rms_norm_quant(
            x, r, w, eps=EPS, quant_dtype=jnp.int8, block_rows=64, interpret=True
        )

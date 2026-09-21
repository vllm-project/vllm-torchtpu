"""The fused gated norm and fp8 quantisation (kernels/gated_norm_quant.py) against the two-step path, in Pallas
interpret mode on CPU: identical fp8 rows and scales, and the pre-quantised matmul equals the XLA per-channel path."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import gated_norm_quant as g
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul


@pytest.fixture(autouse=True)
def _on_the_host():
    """Every array on the CPU device: the kernel runs in interpret mode and
    the reference must round the same way, even in a session that has already
    brought up the TPU backend (the reference's XLA fusions on the chip round
    differently). A session of the host-only tests alone is pinned to the CPU
    by the collection hook in tests/conftest.py."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _inputs(tokens=64, heads=4, dim=128, seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 3)
    o = (jax.random.normal(k[0], (tokens, heads, dim), jnp.float32) *
         3).astype(jnp.bfloat16)
    z = jax.random.normal(k[1], (tokens, heads, dim),
                          jnp.float32).astype(jnp.bfloat16)
    w = (1.0 + 0.1 * jax.random.normal(k[2], (dim, ), jnp.float32)).astype(
        jnp.bfloat16)
    return o, z, w


def test_kernel_matches_the_two_step_path():
    o, z, w = _inputs()
    q, s = g.gated_norm_quant(o, z, w, 1e-6, tile=16, interpret=True)
    q_ref, s_ref = g.reference(o, z, w, 1e-6)
    np.testing.assert_array_equal(np.asarray(s), np.asarray(s_ref))
    np.testing.assert_array_equal(np.asarray(q, np.float32),
                                  np.asarray(q_ref, np.float32))


def test_zero_rows_get_unit_scale():
    o, z, w = _inputs(tokens=16, heads=2)
    o = o.at[3].set(0)
    q, s = g.gated_norm_quant(o, z, w, 1e-6, tile=16, interpret=True)
    assert float(s[3, 0]) == 1.0 and float(
        jnp.abs(q[3].astype(jnp.float32)).max()) == 0.0


def test_prequant_matmul_equals_the_xla_path():
    o, z, w = _inputs(tokens=32, heads=2)
    q_ref, s_ref = g.reference(o, z, w, 1e-6)
    x = (o.astype(jnp.float32) * jax.lax.rsqrt(
        jnp.mean(o.astype(jnp.float32)**2, -1, keepdims=True) + 1e-6) *
         w.astype(jnp.float32) *
         (z.astype(jnp.float32) *
          jax.nn.sigmoid(z.astype(jnp.float32)))).astype(jnp.bfloat16)
    x2 = x.reshape(x.shape[0], -1)
    k = jax.random.PRNGKey(7)
    w_q = (jax.random.normal(k, (x2.shape[1], 96), jnp.float32) * 0.05).astype(
        g.FP8)
    w_s = jnp.full((96, ), 0.02, jnp.float32)
    ref = xla_quantized_matmul(x2, w_q, w_s)
    out = g.prequant_matmul_in_out(q_ref.reshape(x2.shape),
                                   s_ref,
                                   w_q,
                                   w_s,
                                   out_dtype=x2.dtype)
    np.testing.assert_array_equal(np.asarray(out, np.float32),
                                  np.asarray(ref, np.float32))


def test_row_counts_that_the_tile_does_not_divide():
    for tokens in (
            48, 12, 4, 20, 36, 40, 44, 52,
            80):  # 36, 44, 52: no multiple-of-8 divisor, one whole block
        o, z, w = _inputs(tokens=tokens, heads=2)
        q, s = g.gated_norm_quant(o, z, w, 1e-6, tile=32, interpret=True)
        q_ref, s_ref = g.reference(o, z, w, 1e-6)
        np.testing.assert_array_equal(np.asarray(q, np.float32),
                                      np.asarray(q_ref, np.float32))
        np.testing.assert_array_equal(np.asarray(s), np.asarray(s_ref))

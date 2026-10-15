# SPDX-License-Identifier: Apache-2.0
"""QSA native row packing."""

import jax.numpy as jnp
import numpy as np
import pytest
import qwen4_exp_qsa_reference as qsa

from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("heads", [1, 2])
def test_head_packing_preserves_values(fp8, heads):
    k = jnp.arange(3 * heads * 256).reshape(3, heads, 256).astype(jnp.bfloat16) / 32
    v = -k
    packed = qsa_rpa.pack_rows(k, v, fp8=fp8, k_scale=0.375, v_scale=2.5)
    assert packed.dtype == (jnp.float8_e4m3fn if fp8 else jnp.bfloat16)
    decoded = packed.reshape(3, -1, 256)
    expected_k = qsa.quantize_kv(k, 0.375) if fp8 else k
    expected_v = qsa.quantize_kv(v, 2.5) if fp8 else v
    np.testing.assert_array_equal(
        np.asarray(decoded[:, : 2 * heads : 2], np.float32),
        np.asarray(expected_k, np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(decoded[:, 1 : 2 * heads : 2], np.float32),
        np.asarray(expected_v, np.float32),
    )
    if fp8 and heads == 1:
        np.testing.assert_array_equal(np.asarray(decoded[:, 2:], np.float32), 0)

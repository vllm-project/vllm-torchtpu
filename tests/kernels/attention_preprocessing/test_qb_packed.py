# SPDX-License-Identifier: Apache-2.0
"""Packed head-pair writeback reproduces the serial kernel's Q-B/RoPE output.

The kernel views its HBM output as u32 head pairs (a TPU tiled-layout fact);
interpret mode cannot discharge that ref bitcast, so this runs on TPU only.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_serial import make_inputs

from vllm_torchtpu.kernels.attention_preprocessing import qb_packed, reference, serial


@pytest.mark.parametrize(("tokens", "heads"), [(256, 16), (512, 32)])
def test_matches_serial_bitwise(tokens, heads):
    if jax.default_backend() != "tpu":
        pytest.skip("Requires a TPU backend")
    args, weights = make_inputs(tokens=tokens, head_dim=512, heads=heads)
    qr, _ = reference.qa_kv_norm_swa(*args)
    cos, sin = args[-2:]
    q = jax.jit(qb_packed.fused_qb_rope_packed)(qr, *weights, cos, sin)
    expected = jax.jit(serial.fused_qb_rope_serial)(qr, *weights, cos, sin)
    assert q.shape == (tokens, heads, 512) and q.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(q, np.float32), np.asarray(expected, np.float32)
    )

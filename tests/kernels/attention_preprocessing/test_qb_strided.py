# SPDX-License-Identifier: Apache-2.0
"""Strided head staging reproduces the serial kernel's Q-B/RoPE output."""

import jax.numpy as jnp
import numpy as np
import pytest
from test_serial import make_inputs

from vllm_torchtpu.kernels.attention_preprocessing import qb_strided, reference, serial


@pytest.mark.cpu_test
@pytest.mark.parametrize(("tokens", "heads"), [(256, 16), (512, 32)])
def test_matches_serial_bitwise(tokens, heads):
    args, weights = make_inputs(tokens=tokens, head_dim=512, heads=heads)
    qr, _ = reference.qa_kv_norm_swa(*args)
    cos, sin = args[-2:]
    q = qb_strided.fused_qb_rope_strided(qr, *weights, cos, sin, interpret=True)
    expected = serial.fused_qb_rope_serial(qr, *weights, cos, sin, interpret=True)
    assert q.shape == (tokens, heads, 512) and q.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(q, np.float32), np.asarray(expected, np.float32)
    )
    np.testing.assert_array_equal(np.asarray(q[0]), 0)

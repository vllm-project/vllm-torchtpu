# SPDX-License-Identifier: Apache-2.0
"""Quantization scratch is reused across heads without changing RoPE order."""

import jax.numpy as jnp
import numpy as np
import pytest
from test_serial import assert_close, make_inputs

from vllm_torchtpu.kernels.attention_preprocessing import reference, streamed


@pytest.mark.cpu_test
@pytest.mark.parametrize(("tokens", "head_dim"), [(256, 128), (512, 512)])
def test_cached_quantization_and_streamed_rope(tokens, head_dim):
    args, weights = make_inputs(tokens=tokens, head_dim=head_dim, heads=32)
    qr, _ = reference.qa_kv_norm_swa(*args)
    cos, sin = args[-2:]
    q = streamed.fused_qb_rope_streamed(
        qr, *weights, cos, sin, head_dim=head_dim, interpret=True
    )
    expected = reference.qb_rope(qr, *weights, cos, sin, head_dim=head_dim)
    assert_close(q, expected)
    np.testing.assert_array_equal(np.asarray(q[0]), 0)
    identity = streamed.fused_qb_rope_streamed(
        qr * jnp.bfloat16(0.75),
        *weights,
        jnp.ones_like(cos),
        jnp.zeros_like(sin),
        head_dim=head_dim,
        interpret=True,
    )
    projected = reference.projection(qr * jnp.bfloat16(0.75), *weights)
    assert_close(identity, projected.reshape(identity.shape))

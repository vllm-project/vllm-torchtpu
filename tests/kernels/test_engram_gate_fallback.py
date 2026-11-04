# SPDX-License-Identifier: Apache-2.0
"""Non-decode shapes must use the native expression behind the JAX bridge."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.experimental.engram_gate import gate_residual


@pytest.mark.parametrize("tokens", [0, 17, 32])
def test_gate_fallback_matches_torch_and_float64(tokens):
    rng = np.random.default_rng(tokens)
    shapes = [(tokens, 4, 5120), (tokens, 25600), (4, 5120), (4, 5120)]
    inputs = [
        torch.from_numpy(rng.normal(size=s).astype(np.float32)).bfloat16()
        for s in shapes
    ]
    mask = torch.ones(tokens)
    mask[::3] = 0

    def reference(dtype):
        h, kv, qw, kw = (x.to(dtype) for x in inputs)
        keys = kv[:, :20480].reshape(tokens, 4, 5120)
        values = kv[:, 20480:]
        dot = (h * qw * kw * keys).sum(-1)
        dot *= torch.rsqrt(h.square().mean(-1) + 1e-20)
        dot *= torch.rsqrt(keys.square().mean(-1) + 1e-20) * 5120**-0.5
        root = dot.abs().clamp(min=1e-6).sqrt()
        gate = torch.where(dot < 0, -root, root).sigmoid() * mask[:, None]
        return (h + gate[:, :, None] * values[:, None]).bfloat16().float().numpy()

    with jax.default_device(jax.devices("cpu")[0]):
        arrays = [jnp.asarray(x.float().numpy(), jnp.bfloat16) for x in inputs]
        actual = np.asarray(
            jax.jit(gate_residual)(*arrays, jnp.asarray(mask.numpy()))
        ).astype(np.float32)
    for expected in (reference(torch.float32), reference(torch.float64)):
        np.testing.assert_allclose(actual, expected, rtol=1 / 128, atol=2e-6)

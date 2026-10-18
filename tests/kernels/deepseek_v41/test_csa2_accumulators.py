# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check async accumulator reuse against synchronous attention math."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.csa2 import csa2


@pytest.mark.parametrize("heads", [8, 16, 32, 64, 96, 104, 128])
def test_csa2_accumulator_pipeline_matches_interpret(heads: int) -> None:
    """Cover both accumulator layouts and every one-to-six-query tail."""
    if jax.default_backend() != "tpu":
        pytest.skip("Compiled accumulator operations require TPU")
    rng = np.random.default_rng(107)
    starts = np.array([0, 1, 3, 6, 10, 15, 21], np.int32)
    tokens = 23
    positions = np.zeros(tokens, np.int32)
    for start, end in zip(starts[:-1], starts[1:], strict=True):
        positions[start:end] = np.arange(20, 20 + end - start)

    def random(shape: tuple[int, ...]) -> jax.Array:
        return jnp.asarray(rng.normal(size=shape) * 0.2, jnp.bfloat16)

    q = random((tokens, heads, 512))
    main = random((tokens, 512, 4, 128)).astype(jnp.float8_e4m3fn)
    scales = jnp.ones((tokens, 512), jnp.float32)
    # At these positions all windows start at zero. Reuse the same KV rows
    # within each request, as required by the grouped SWA DMA contract.
    swa = jnp.broadcast_to(random((1, 128, 512)), (tokens, 128, 512))
    angles = jnp.repeat(jnp.asarray(rng.normal(size=(tokens, 32)) * 0.1), 2, -1)
    signed_sin = jnp.sin(angles) * jnp.tile(jnp.array([1, -1]), 32)
    rope = jnp.stack(
        (
            jnp.concatenate((jnp.ones_like(angles), jnp.cos(angles)), -1),
            jnp.concatenate((jnp.zeros_like(angles), signed_sin), -1),
        ),
        1,
    )
    lengths = jnp.resize(jnp.array([0, 1, 127, 511, 512], jnp.int32), tokens)
    args = (
        q,
        main,
        scales,
        swa,
        jnp.asarray(starts),
        jnp.asarray(positions),
        jnp.linspace(-1, 1, heads, dtype=jnp.float32),
        rope,
    )
    expected = csa2(*args, bq=6, main_lengths=lengths, interpret=True)
    actual = csa2(*args, bq=6, main_lengths=lengths)
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        np.asarray(expected, np.float32),
        atol=3e-3,
        rtol=2e-2,
    )
    np.testing.assert_array_equal(actual[21:], 0)

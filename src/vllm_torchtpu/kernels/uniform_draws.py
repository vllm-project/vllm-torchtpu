# SPDX-License-Identifier: Apache-2.0
"""Uniform draws from an explicit generator state, inside the calling program.

The state is XLA's ``RngBitGenerator`` state (four 32-bit words, carried as
int32 so torch can hold it) and the draws use its default algorithm, which on
TPU is the one the backend implements natively. The caller keeps the returned
state for its next call, so the stream advances only through these calls and
every replica that starts from the same state and makes the same calls draws
the same numbers.
"""

import jax
import jax.numpy as jnp
import numpy as np

# float32 1.0: OR-ed into 23 random mantissa bits gives a value in [1, 2).
_ONE_BITS = np.uint32(0x3F800000)


def initial_state(seed: int) -> np.ndarray:
    """Four int32 state words from ``seed`` by splitmix64, so nearby seeds give
    unrelated states."""
    words = []
    x = int(seed) & 0xFFFFFFFFFFFFFFFF
    for _ in range(2):
        x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z ^= z >> 31
        words += [z & 0xFFFFFFFF, z >> 32]
    return np.array(words, dtype=np.uint32).view(np.int32)


def _uniform(state: jax.Array, shape: tuple[int, ...]) -> tuple[jax.Array, jax.Array]:
    state, bits = jax.lax.rng_bit_generator(state, shape, dtype=jnp.uint32)
    one_to_two = jax.lax.bitcast_convert_type((bits >> 9) | _ONE_BITS, jnp.float32)
    return state, one_to_two - 1.0


def uniform_draws(
    state: jax.Array, rows: int, cols: int, count: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """F32 [rows, cols] and [count] uniforms in [0, 1), and the next state."""
    key = jax.lax.bitcast_convert_type(state, jnp.uint32)
    key, matrix = _uniform(key, (rows, cols))
    key, vector = _uniform(key, (count,))
    return matrix, vector, jax.lax.bitcast_convert_type(key, jnp.int32)

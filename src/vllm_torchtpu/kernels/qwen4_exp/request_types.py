# SPDX-License-Identifier: Apache-2.0
"""QSA request segmentation using the runner's decode-first ragged ordering."""

import jax
import jax.numpy as jnp


def decode_prefix(starts: jax.Array) -> jax.Array:
    """Count the leading one-token requests; leave later/empty rows in place."""
    return jnp.sum(jnp.cumprod((jnp.diff(starts) == 1).astype(jnp.int32)))

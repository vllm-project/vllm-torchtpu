# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JAX attention reference for CSA2 prefill validation."""

import jax
import jax.numpy as jnp
from jax import lax


def prefill_reference(
    q: jax.Array,
    words: jax.Array,
    scales: jax.Array,
    swa_kv: jax.Array | tuple[jax.Array, jax.Array],
    starts: jax.Array,
    positions: jax.Array,
    sink: jax.Array,
    rope: jax.Array,
    *,
    swa_metadata: jax.Array,
    main_lengths: jax.Array,
    swa_bounds: jax.Array | None = None,
    scale: float = 512**-0.5,
) -> jax.Array:
    """Independent attention math for validating either SWA visibility mode."""
    banks = swa_kv if isinstance(swa_kv, tuple) else (swa_kv,)
    pages, bounds, lengths = swa_metadata, swa_bounds, main_lengths
    tokens = q.shape[0]
    request = jnp.minimum(
        jnp.searchsorted(starts[1:], jnp.arange(tokens), side="right"),
        pages.shape[0] - 1,
    )
    page_size = banks[0].shape[1]
    left = jnp.maximum(0, positions - 127) if bounds is None else bounds[request, 0]
    right = positions + 1 if bounds is None else bounds[request, 1]
    keys = left[:, None] + jnp.arange(128 if bounds is None else 192)
    columns = (keys - pages[request, 0, None]) // page_size
    valid = (keys < right[:, None]) & (keys >= pages[request, 1, None])
    valid &= (columns >= 0) & (columns < pages.shape[1] - 2)
    columns = jnp.clip(columns, 0, pages.shape[1] - 3)
    page_ids = pages[request[:, None], columns + 2]
    swa = banks[0][jnp.clip(page_ids, 0, banks[0].shape[0] - 1), keys % page_size]
    if len(banks) == 2:
        high = banks[1][
            jnp.clip(page_ids - banks[0].shape[0], 0, banks[1].shape[0] - 1),
            keys % page_size,
        ]
        swa = jnp.where((page_ids < banks[0].shape[0])[..., None], swa, high)
    swa = jnp.where(valid[..., None], swa, 0)
    bits = (words[:, :, None, :] >> (8 * jnp.arange(4))[None, None, :, None]).astype(
        jnp.uint8
    )
    main = lax.bitcast_convert_type(bits, jnp.float8_e4m3fn).astype(jnp.bfloat16)
    main = main.reshape(tokens, 512, 512)
    main_logits = jnp.einsum(
        "thd,tkd->thk", q, main, preferred_element_type=jnp.float32
    )
    main_logits *= scales[:, None, :] * scale
    main_logits = jnp.where(
        jnp.arange(512) < lengths[:, None, None], main_logits, -jnp.inf
    )
    swa_logits = (
        jnp.einsum("thd,tkd->thk", q, swa, preferred_element_type=jnp.float32) * scale
    )
    swa_logits = jnp.where(valid[:, None, :], swa_logits, -jnp.inf)
    logits = jnp.concatenate((main_logits, swa_logits), axis=-1)
    maximum = jnp.maximum(logits.max(axis=-1, keepdims=True), sink[None, :, None])
    weights = jnp.exp(logits - maximum)
    weights /= weights.sum(axis=-1, keepdims=True) + jnp.exp(
        sink[None, :, None] - maximum
    )
    main_p = (weights[:, :, :512] * scales[:, None, :]).astype(jnp.bfloat16)
    swa_p = weights[:, :, 512:].astype(jnp.bfloat16)
    output = jnp.einsum(
        "thk,tkd->thd", main_p, main, preferred_element_type=jnp.float32
    )
    output += jnp.einsum("thk,tkd->thd", swa_p, swa, preferred_element_type=jnp.float32)
    tail = output[:, :, 384:]
    tail = (
        tail * rope[:, None, 0, :]
        + tail[:, :, jnp.arange(128) ^ 1] * rope[:, None, 1, :]
    )
    output = jnp.concatenate((output[:, :, :384], tail), axis=-1).astype(jnp.bfloat16)
    return jnp.where((jnp.arange(tokens) < starts[-1])[:, None, None], output, 0)

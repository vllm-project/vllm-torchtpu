# SPDX-License-Identifier: Apache-2.0
"""Frozen pre-expansion CSA dispatcher for the integrated A/B benchmark only.

Copied verbatim (dispatcher and _requests) from production csa2/cache.py before
its guard extension. Do not replace with an import of the live dispatcher.
Source SHA256: cd755dc47f925169a20dfd33d7cc11b07af25edc5a67c8b91df5afd2286ff280
"""

import functools

import jax
import jax.numpy as jnp


def _requests(starts: jax.Array, tokens: int) -> jax.Array:
    return jnp.repeat(
        jnp.arange(starts.shape[0] - 1, dtype=jnp.int32),
        starts[1:] - starts[:-1],
        total_repeat_length=tokens,
    )


@functools.partial(jax.jit, static_argnames=("page_size",))
def prepare_main_records(
    selected: jax.Array,
    physical_pages: jax.Array,
    starts: jax.Array,
    *,
    page_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Resolve already-compacted logical selections; preserve stage gather reuse.

    Page IDs address contiguous records in concatenated physical banks. Caller
    guarantees all nonnegative selections are in range and form a valid prefix.
    Returns fixed [T,512] physical IDs and authoritative [T] valid-prefix counts.
    This operation belongs to the first gather for a source/index group only.
    """
    if selected.ndim != 2 or selected.shape[1] > 512 or page_size < 1:
        raise ValueError("CSA2 main selection width must be at most 512")
    if (
        jax.default_backend() == "tpu"
        and selected.shape == (288, 512)
        and physical_pages.shape[0] == 48
        and physical_pages.shape[1] in (386, 772)
    ):
        from vllm_torchtpu.kernels.experimental.decode_preparation import (
            main_records_call,
        )

        inputs = (selected, physical_pages, starts)
        return main_records_call(inputs, page_size=page_size)(*inputs)
    tokens, width = selected.shape
    req = _requests(starts, tokens)
    valid = (selected >= 0) & (jnp.arange(tokens)[:, None] < starts[-1])
    logical = jnp.maximum(selected, 0)
    if jax.default_backend() == "tpu":
        from vllm_torchtpu.kernels.deepseek_v4.core_attention import sparse_mla

        # Preserve the epic optimized page lookup. Ordinary JAX indexing is
        # especially expensive for single-request, long-context prefill.
        padded = jnp.pad(logical, ((0, 0), (0, 512 - width)))
        pages = sparse_mla.gather_page_ids(
            physical_pages.reshape(-1),
            padded // page_size,
            req,
            physical_pages.shape[0],
        )[:, :width]
    else:
        pages = physical_pages[req[:, None], logical // page_size]
    records = jnp.where(valid, pages * page_size + logical % page_size, -1)
    records = jnp.pad(records, ((0, 0), (0, 512 - width)), constant_values=-1)
    return records.astype(jnp.int32), jnp.sum(valid, axis=1, dtype=jnp.int32)

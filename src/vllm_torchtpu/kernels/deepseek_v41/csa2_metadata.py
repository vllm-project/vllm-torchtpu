# SPDX-License-Identifier: Apache-2.0
"""Device address preparation for CSA2; never materializes KV features."""

import functools

import jax
import jax.numpy as jnp


def _requests(starts: jax.Array, tokens: int) -> jax.Array:
    return jnp.repeat(
        jnp.arange(starts.shape[0] - 1, dtype=jnp.int32),
        starts[1:] - starts[:-1],
        total_repeat_length=tokens,
    )


@functools.partial(
    jax.jit,
    static_argnames=("page_size", "page_stride", "row_offset", "non_causal"),
)
def prepare_swa_metadata(
    block_table: jax.Array,
    starts: jax.Array,
    positions: jax.Array,
    minimum_visible: jax.Array,
    *,
    page_size: int,
    page_stride: int,
    row_offset: int,
    non_causal: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return request page metadata, explicit bounds, and physical write records.

    Block table entries are allocator block IDs. Stride/offset are in BF16
    512-feature record units and must preserve whole BF16 pages. Starts and
    positions obey csa2's contiguous-request contract; noncausal requests have
    at most 64 queries. Minimum visibility carries replay/window-start state.
    Bounds are passed to csa2 ONLY for noncausal calls; causal visibility stays
    query-dependent. Padding records are -1. Empty requests have empty bounds.
    """
    tokens = positions.shape[0]
    requests = block_table.shape[0]
    if (
        block_table.ndim != 2
        or block_table.shape[1] < 1
        or starts.shape != (requests + 1,)
        or minimum_visible.shape != (requests,)
        or not tokens
        or page_size not in (8, 16, 32, 64)
        and (page_size < 128 or page_size % 128)
        or page_stride % page_size
        or row_offset % page_size
    ):
        raise ValueError("CSA2 requires nonempty, whole-page aligned SWA metadata")
    first = positions[jnp.minimum(starts[:-1], tokens - 1)]
    count = starts[1:] - starts[:-1]
    left = jnp.maximum(first - (128 if non_causal else 127), 0)
    right = first + count
    left = jnp.where(count > 0, left, 0)
    right = jnp.where(count > 0, right, 0)
    first_page = left // page_size
    # An unaligned first page can add one page to ceil((queries+history)/S).
    columns = (tokens + 128 + page_size - 1) // page_size + 1
    logical_pages = first_page[:, None] + jnp.arange(columns, dtype=jnp.int32)
    blocks = jnp.take_along_axis(
        block_table, jnp.minimum(logical_pages, block_table.shape[1] - 1), axis=1
    )
    physical = blocks * (page_stride // page_size) + row_offset // page_size
    visible = (
        (count[:, None] > 0)
        & (logical_pages < block_table.shape[1])
        & (logical_pages * page_size < right[:, None])
    )
    metadata = jnp.concatenate(
        (
            (first_page * page_size)[:, None],
            minimum_visible[:, None],
            jnp.where(visible, physical, 0),
        ),
        axis=1,
    ).astype(jnp.int32)
    req = _requests(starts, tokens)
    current_blocks = block_table[
        req, jnp.minimum(positions // page_size, block_table.shape[1] - 1)
    ]
    records = row_offset + current_blocks * page_stride + positions % page_size
    records = jnp.where(jnp.arange(tokens) < starts[-1], records, -1)
    return metadata, jnp.stack((left, right), axis=1), records.astype(jnp.int32)


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

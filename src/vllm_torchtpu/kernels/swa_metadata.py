# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal SWA metadata and final physical attention records in one Pallas call.

All address arithmetic is int32. Logical-only APIs remain for inspection.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


class CommonSwaMetadata(NamedTuple):
    """T token slots, R requests, W window positions; page labels are 2/1/0."""

    request_ids: jax.Array  # int32[T]
    valid_tokens: jax.Array  # bool[T]
    token_window_starts: jax.Array  # int32[T]
    current_pages: jax.Array  # int32[T], block-table column for current KV
    previous_pages: jax.Array  # int32[T]
    oldest_pages: jax.Array  # int32[T]
    current_offsets: jax.Array  # int32[T]
    page_offsets: jax.Array  # int32[T, W]
    page_selector: jax.Array  # int32[T, W], current=2, previous=1, oldest=0
    read_allowed: jax.Array  # bool[T, W], excludes layer-specific slot validity
    query_counts: jax.Array  # int32[R]
    nonempty_requests: jax.Array  # bool[R]


class SwaAttentionMetadata(NamedTuple):
    """Only the logical fields consumed by three-page causal SWA addressing."""

    request_ids: jax.Array
    valid_tokens: jax.Array
    current_pages: jax.Array
    previous_pages: jax.Array
    oldest_pages: jax.Array
    current_offsets: jax.Array
    page_offsets: jax.Array
    is_current_page: jax.Array
    is_previous_page: jax.Array
    read_allowed: jax.Array


class SwaAttentionRecords(NamedTuple):
    """Final addresses consumed by current-KV insertion and SWA attention."""

    write_slots: jax.Array  # int32[T], logical KV slots; -1 for padding
    write_valid: jax.Array  # bool[T], write_slots >= 0
    read_records: jax.Array  # int32[T, W], physical row indices; -1 if masked


def supports_swa_attention_metadata(
    tokens: int,
    requests: int,
    window_size: int,
    page_size: int,
    max_queries_per_request: int,
) -> bool:
    """Small causal decode envelope for a whole-batch VMEM program."""
    return (
        window_size == 128
        and page_size == 64
        and 0 < requests <= window_size
        and 0 < tokens <= requests * max_queries_per_request
    )


def _swa_metadata_values(
    query_start_loc: jax.Array,
    positions: jax.Array,
    window_starts: jax.Array,
    *,
    window_size: int,
    page_size: int,
    table_columns: int,
) -> tuple[SwaAttentionMetadata, jax.Array, jax.Array]:
    """Shared expressions; no physical lookup or integer wider than 32 bits."""
    tokens = positions.shape[0]
    requests = window_starts.shape[0]
    token_rows = jnp.arange(tokens, dtype=jnp.int32)
    request_rows = jnp.arange(requests, dtype=jnp.int32)

    raw_request_ids = jnp.sum(
        token_rows[:, None] >= query_start_loc[None, 1:],
        axis=1,
        dtype=jnp.int32,
    )
    valid_tokens = raw_request_ids < requests
    # Padded token rows bucketize to R; their metadata is masked below.
    request_ids = jnp.where(valid_tokens, raw_request_ids, requests - 1)
    belongs_to_request = request_ids[:, None] == request_rows[None, :]
    token_window_starts = jnp.sum(
        jnp.where(belongs_to_request, window_starts[None, :], 0),
        axis=1,
        dtype=jnp.int32,
    )

    # Padding can retain stale positions from a larger bucket. Make physical
    # table indices safe before gathering; valid positions are scheduler-bounded.
    positions = jnp.where(valid_tokens, positions, 0)
    window_columns = jnp.arange(window_size, dtype=jnp.int32)
    logical = positions[:, None] - (window_size - 1) + window_columns[None, :]
    max_position = table_columns * page_size - 1
    safe = jnp.clip(logical, 0, max_position)
    page_shift = page_size.bit_length() - 1
    page_mask = page_size - 1
    pages = safe >> page_shift

    current_pages = positions >> page_shift
    previous_pages = jnp.maximum(current_pages - 1, 0)
    oldest_pages = jnp.maximum(current_pages - 2, 0)
    # Nonnegative window starts also exclude negative logical positions.
    # Broadcast int32 instead of reshaping a bool vector, which Mosaic cannot
    # lower for this layout. Reuse the request-validity calculation.
    read_allowed = (
        jax.lax.broadcast_in_dim(raw_request_ids, logical.shape, (0,)) < requests
    ) & (logical >= token_window_starts[:, None])
    metadata = SwaAttentionMetadata(
        request_ids=request_ids,
        valid_tokens=valid_tokens,
        current_pages=current_pages,
        previous_pages=previous_pages,
        oldest_pages=oldest_pages,
        current_offsets=positions & page_mask,
        page_offsets=safe & page_mask,
        is_current_page=pages == current_pages[:, None],
        is_previous_page=pages == previous_pages[:, None],
        read_allowed=read_allowed,
    )
    return metadata, token_window_starts, logical


def _common_swa_metadata_values(
    query_start_loc: jax.Array,
    positions: jax.Array,
    window_starts: jax.Array,
    *,
    window_size: int,
    page_size: int,
    table_columns: int,
) -> CommonSwaMetadata:
    """Expanded review interface, including request counts and window starts."""
    metadata, token_window_starts, _ = _swa_metadata_values(
        query_start_loc,
        positions,
        window_starts,
        window_size=window_size,
        page_size=page_size,
        table_columns=table_columns,
    )
    query_counts = query_start_loc[1:] - query_start_loc[:-1]
    return CommonSwaMetadata(
        request_ids=metadata.request_ids,
        valid_tokens=metadata.valid_tokens,
        token_window_starts=token_window_starts,
        current_pages=metadata.current_pages,
        previous_pages=metadata.previous_pages,
        oldest_pages=metadata.oldest_pages,
        current_offsets=metadata.current_offsets,
        page_offsets=metadata.page_offsets,
        page_selector=jnp.where(
            metadata.is_current_page,
            jnp.int32(2),
            jnp.where(metadata.is_previous_page, jnp.int32(1), jnp.int32(0)),
        ),
        read_allowed=metadata.read_allowed,
        query_counts=query_counts,
        nonempty_requests=query_counts > 0,
    )


def _common_swa_metadata_kernel(
    query_start_loc_ref,
    positions_ref,
    window_starts_ref,
    request_ids_ref,
    valid_tokens_ref,
    token_window_starts_ref,
    current_pages_ref,
    previous_pages_ref,
    oldest_pages_ref,
    current_offsets_ref,
    page_offsets_ref,
    page_selector_ref,
    read_allowed_ref,
    query_counts_ref,
    nonempty_requests_ref,
    *,
    window_size,
    page_size,
    table_columns,
):
    metadata = _common_swa_metadata_values(
        query_start_loc_ref[...],
        positions_ref[...],
        window_starts_ref[...],
        window_size=window_size,
        page_size=page_size,
        table_columns=table_columns,
    )
    outputs = (
        request_ids_ref,
        valid_tokens_ref,
        token_window_starts_ref,
        current_pages_ref,
        previous_pages_ref,
        oldest_pages_ref,
        current_offsets_ref,
        page_offsets_ref,
        page_selector_ref,
        read_allowed_ref,
        query_counts_ref,
        nonempty_requests_ref,
    )
    for output_ref, value in zip(outputs, metadata):
        output_ref[...] = value


def _swa_attention_metadata_kernel(
    query_start_loc_ref,
    positions_ref,
    window_starts_ref,
    request_ids_ref,
    valid_tokens_ref,
    current_pages_ref,
    previous_pages_ref,
    oldest_pages_ref,
    current_offsets_ref,
    page_offsets_ref,
    is_current_page_ref,
    is_previous_page_ref,
    read_allowed_ref,
    *,
    window_size,
    page_size,
    table_columns,
):
    metadata, _, _ = _swa_metadata_values(
        query_start_loc_ref[...],
        positions_ref[...],
        window_starts_ref[...],
        window_size=window_size,
        page_size=page_size,
        table_columns=table_columns,
    )
    outputs = (
        request_ids_ref,
        valid_tokens_ref,
        current_pages_ref,
        previous_pages_ref,
        oldest_pages_ref,
        current_offsets_ref,
        page_offsets_ref,
        is_current_page_ref,
        is_previous_page_ref,
        read_allowed_ref,
    )
    for output_ref, value in zip(outputs, metadata):
        output_ref[...] = value


def _swa_attention_records_kernel(
    query_start_loc_ref,
    positions_ref,
    window_starts_ref,
    block_table_ref,
    write_slots_ref,
    write_valid_ref,
    read_records_ref,
    *,
    window_size,
    page_size,
    page_stride,
    row_offset,
    table_columns,
):
    metadata, token_window_starts, logical = _swa_metadata_values(
        query_start_loc_ref[...],
        positions_ref[...],
        window_starts_ref[...],
        window_size=window_size,
        page_size=page_size,
        table_columns=table_columns,
    )

    table = block_table_ref[...]
    tokens = metadata.request_ids.shape[0]
    request_rows = jnp.arange(table.shape[0], dtype=jnp.int32)
    belongs = request_rows[:, None] == metadata.request_ids[None, :]
    dnums = jax.lax.GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(1,),
        start_index_map=(1,),
        operand_batching_dims=(0,),
        start_indices_batching_dims=(0,),
    )
    lanes = pltpu.get_tpu_info().num_lanes
    selected_pages = []
    for pages in (
        metadata.current_pages,
        metadata.previous_pages,
        metadata.oldest_pages,
    ):
        selected = jnp.zeros((table.shape[0], tokens), dtype=jnp.int32)
        # Mosaic dynamic gather requires one source vreg along its gather axis.
        for begin in range(0, table.shape[1], lanes):
            stop = min(begin + lanes, table.shape[1])
            local_columns = jax.lax.broadcast_in_dim(
                pages - begin, selected.shape, (1,)
            )
            indices = jnp.clip(local_columns, 0, stop - begin - 1)[..., None]
            values = jax.lax.gather(
                table[:, begin:stop],
                indices,
                dnums,
                slice_sizes=(1, 1),
                mode="promise_in_bounds",
            )
            active = (local_columns >= 0) & (local_columns < stop - begin)
            selected = jnp.where(active, values, selected)
        selected_pages.append(
            jnp.sum(jnp.where(belongs, selected, 0), axis=0, dtype=jnp.int32)
        )
    current_blocks, previous_blocks, oldest_blocks = selected_pages
    write_slots = jnp.where(
        metadata.valid_tokens,
        current_blocks * page_size + metadata.current_offsets,
        -1,
    )
    blocks = jnp.where(
        metadata.is_current_page,
        current_blocks[:, None],
        jnp.where(
            metadata.is_previous_page,
            previous_blocks[:, None],
            oldest_blocks[:, None],
        ),
    )
    # The current write slot also gates reads, matching the attention consumer.
    # Broadcasting int32 slots avoids the unsupported bool shape cast.
    read_allowed = (jax.lax.broadcast_in_dim(write_slots, logical.shape, (0,)) >= 0) & (
        logical >= token_window_starts[:, None]
    )
    read_records = jnp.where(
        read_allowed,
        row_offset + blocks * page_stride + metadata.page_offsets,
        -1,
    )
    write_slots_ref[...] = write_slots
    write_valid_ref[...] = write_slots >= 0
    read_records_ref[...] = read_records


def _validate_inputs(
    query_start_loc, positions, window_starts, window_size, page_size, table_columns
):
    assert positions.ndim == query_start_loc.ndim == window_starts.ndim == 1
    assert positions.dtype == query_start_loc.dtype == window_starts.dtype == jnp.int32
    assert positions.shape[0] > 0 and window_starts.shape[0] > 0
    assert query_start_loc.shape == (window_starts.shape[0] + 1,)
    assert page_size > 0 and page_size & (page_size - 1) == 0
    assert 0 < window_size <= 2 * page_size
    assert 0 < table_columns * page_size <= jnp.iinfo(jnp.int32).max


@functools.partial(
    jax.jit,
    static_argnames=(
        "window_size",
        "page_size",
        "page_stride",
        "row_offset",
        "interpret",
    ),
)
def swa_attention_records(
    query_start_loc: jax.Array,
    positions: jax.Array,
    window_starts: jax.Array,
    block_table: jax.Array,
    *,
    window_size: int,
    page_size: int,
    page_stride: int,
    row_offset: int,
    interpret: bool = False,
) -> SwaAttentionRecords:
    """Fuse logical metadata, physical lookup, and final attention addresses.

    Inputs are int32: query_start_loc[R+1], positions[T], window_starts[R],
    block_table[R,C]. Boundaries are sorted, start at zero and end at <= T.
    Real token positions lie in [0, C*page_size); padding positions are ignored.
    Window starts are nonnegative. Physical cache addresses must fit int32.

    Write slots use page_size; read records use page_stride plus row_offset.
    This reads the small block table, not the KV cache. Attention reads KV later.
    """
    assert block_table.ndim == 2 and block_table.dtype == jnp.int32
    assert block_table.shape[0] == window_starts.shape[0]
    assert block_table.size <= jnp.iinfo(jnp.int32).max
    assert 0 < page_stride <= jnp.iinfo(jnp.int32).max
    assert 0 <= row_offset <= jnp.iinfo(jnp.int32).max
    _validate_inputs(
        query_start_loc,
        positions,
        window_starts,
        window_size,
        page_size,
        block_table.shape[1],
    )
    tokens = positions.shape[0]
    shapes = (
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
        jax.ShapeDtypeStruct((tokens,), jnp.bool_),
        jax.ShapeDtypeStruct((tokens, window_size), jnp.int32),
    )
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    outputs = pl.pallas_call(
        functools.partial(
            _swa_attention_records_kernel,
            window_size=window_size,
            page_size=page_size,
            page_stride=page_stride,
            row_offset=row_offset,
            table_columns=block_table.shape[1],
        ),
        out_shape=shapes,
        grid=(),
        in_specs=(vmem,) * 4,
        out_specs=(vmem,) * 3,
        interpret=interpret,
        name="swa_attention_records",
    )(query_start_loc, positions, window_starts, block_table)
    return SwaAttentionRecords(*outputs)


@functools.partial(
    jax.jit,
    static_argnames=("window_size", "page_size", "table_columns", "interpret"),
)
def swa_attention_metadata(
    query_start_loc: jax.Array,
    positions: jax.Array,
    window_starts: jax.Array,
    *,
    window_size: int,
    page_size: int,
    table_columns: int,
    interpret: bool = False,
) -> SwaAttentionMetadata:
    """Build only the ten fields used by causal three-page SWA addressing.

    Input layout matches ``common_swa_metadata``. Request window starts remain
    internal; page comparison masks are written directly for block selection.
    """
    _validate_inputs(
        query_start_loc,
        positions,
        window_starts,
        window_size,
        page_size,
        table_columns,
    )
    tokens = positions.shape[0]
    token_int = jax.ShapeDtypeStruct((tokens,), jnp.int32)
    token_bool = jax.ShapeDtypeStruct((tokens,), jnp.bool_)
    window_int = jax.ShapeDtypeStruct((tokens, window_size), jnp.int32)
    window_bool = jax.ShapeDtypeStruct((tokens, window_size), jnp.bool_)
    shapes = (
        token_int,
        token_bool,
        token_int,
        token_int,
        token_int,
        token_int,
        window_int,
        window_bool,
        window_bool,
        window_bool,
    )
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    outputs = pl.pallas_call(
        functools.partial(
            _swa_attention_metadata_kernel,
            window_size=window_size,
            page_size=page_size,
            table_columns=table_columns,
        ),
        out_shape=shapes,
        grid=(),
        in_specs=(vmem, vmem, vmem),
        out_specs=(vmem,) * len(shapes),
        interpret=interpret,
        name="swa_attention_metadata",
    )(query_start_loc, positions, window_starts)
    return SwaAttentionMetadata(*outputs)


@functools.partial(
    jax.jit,
    static_argnames=("window_size", "page_size", "table_columns", "interpret"),
)
def common_swa_metadata(
    query_start_loc: jax.Array,
    positions: jax.Array,
    window_starts: jax.Array,
    *,
    window_size: int,
    page_size: int,
    table_columns: int,
    interpret: bool = False,
) -> CommonSwaMetadata:
    """Prepare logical metadata shared by causal SWA layers with the same layout.

    Args:
        query_start_loc: Sorted int32[R+1] batch boundaries, from 0 to at most T.
        positions: int32[T] positions, real tokens in [0, table_columns*page_size).
        window_starts: Nonnegative int32[R] lower bounds of available KV per request.
        window_size: Number of window positions, at most two pages.
        page_size: Records per page, a positive power of two.
        table_columns: Number of logical pages per request's block table.
        interpret: Execute Pallas in interpret mode for correctness checks.

    Returns:
        Common metadata without physical block lookup or the final attention mask.

    This expanded interface is retained for the metadata walkthrough. Model
    integration uses ``swa_attention_records`` and stores final physical addresses.
    """
    _validate_inputs(
        query_start_loc,
        positions,
        window_starts,
        window_size,
        page_size,
        table_columns,
    )
    tokens = positions.shape[0]
    requests = window_starts.shape[0]

    token_int = jax.ShapeDtypeStruct((tokens,), jnp.int32)
    token_bool = jax.ShapeDtypeStruct((tokens,), jnp.bool_)
    window_int = jax.ShapeDtypeStruct((tokens, window_size), jnp.int32)
    shapes = (
        token_int,
        token_bool,
        token_int,
        token_int,
        token_int,
        token_int,
        token_int,
        window_int,
        window_int,
        jax.ShapeDtypeStruct((tokens, window_size), jnp.bool_),
        jax.ShapeDtypeStruct((requests,), jnp.int32),
        jax.ShapeDtypeStruct((requests,), jnp.bool_),
    )
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    outputs = pl.pallas_call(
        functools.partial(
            _common_swa_metadata_kernel,
            window_size=window_size,
            page_size=page_size,
            table_columns=table_columns,
        ),
        out_shape=shapes,
        grid=(),
        in_specs=(vmem, vmem, vmem),
        out_specs=(vmem,) * len(shapes),
        interpret=interpret,
        name="swa_common_metadata",
    )(query_start_loc, positions, window_starts)
    return CommonSwaMetadata(*outputs)

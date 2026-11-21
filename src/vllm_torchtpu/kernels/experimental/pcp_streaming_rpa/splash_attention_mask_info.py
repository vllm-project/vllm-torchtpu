# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""MaskInfo creation and block-sparsity metadata utilities for Splash attention.

Source provenance:
  Ported from https://github.com/openxla/tokamax
  (`tokamax/_src/ops/splash_attention/splash_attention_mask_info.py`),
  commit `5dff06a967d941abc5035edff68766e22a315430` (2026-10-20).
  Local modifications vs upstream Tokamax:
  - Updated `splash_attention_mask` import path to `vllm_torchtpu`.
  - Added `BlockMaskCode` enum and shared `downcast_index_dtype` helper.
  - Cast `find_bounds` outputs to `jnp.int8` and typed `arr` as accepting
    `None`.
  - Typed `MaskInfo.q_sequence` as `np.ndarray | jax.Array | None` and
    `MaskCallable` as `Callable[..., np.ndarray]`.
  - Replaced `hasattr` probing with `isinstance(mask, mask_lib.ComputableMask)`.
  - Cleared `q_sequence` to `None` alongside `mask_function=None` when all
    blocks are `FULL`.
  - Pruned backward/dKV paths (`is_dkv`, `process_mask_dkv`,
    `process_dynamic_mask_dkv`, `dkv_q_blocks_are_contiguous`,
    `_q_block_stats`, `_classify_mask_blocks`), unused `kv_seq_shards` /
    `return_dynamic_grid=False` options, and unused `_check_mask`.
  - Removed legacy mypy error-suppression directive and applied repository
    formatting and lint cleanup.
"""

import enum
import functools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
    splash_attention_mask as mask_lib,
)

lax = jax.lax
MaskCallable = Callable[..., np.ndarray]
_EXPECTED_MASK_NDIM = 2


class BlockMaskCode(enum.IntEnum):
    """Block-level sparsity codes stored in `MaskInfo.block_mask`."""

    EMPTY = 0
    PARTIAL = 1
    FULL = 2
    SEGMENT_ONLY = 3


def find_bounds(
    arr: jax.Array | np.ndarray | None,
) -> tuple[jax.Array | np.ndarray | None, jax.Array | np.ndarray | None]:
    # Find the first and last block of a row to determine when to initialize/store
    # the output.

    if arr is None:
        return None, None

    bounds_start = (arr != jnp.roll(arr, shift=1, axis=-1)).astype(jnp.int8)
    bounds_end = (arr != jnp.roll(arr, shift=-1, axis=-1)).astype(jnp.int8)
    bounds_start = bounds_start.at[0].set(1)
    bounds_end = bounds_end.at[-1].set(1)

    return bounds_start, bounds_end


# Logic for processing NumPy masks for kernels
class MaskInfo(NamedTuple):
    """Contains runtime masking information for the Splash attention kernel.

    The arrays, mask_next and block_mask are placed in TPU
    scalar-memory. This is a scarse resource so the mask creation logic attempts
    to shrink the data-type of these arrays to the smallest possible one.
    This can be: np.int32, np.int16 or np.int8.

    Attributes:
      mask_next: An integer[num_active_blocks] NumPy array where each entry
        contains the next mask block index in `partial_mask_blocks` to prefetch.
      active_rows: An integer[num_active_blocks] NumPy array where each entry
        contains the row index of the corresponding active block in the original
        mask.
      active_cols: An integer[num_active_blocks] NumPy array where each entry
        contains the column index of the corresponding active block in the
        original mask.
      block_mask: An integer[num_active_blocks] NumPy array whose entries are
        `BlockMaskCode` values: `PARTIAL` (1) applies `partial_mask_blocks`,
        `FULL` (2) requires no mask, `SEGMENT_ONLY` (3) is a base-full block
        demoted by segment refinement that applies only segment masking without
        reading `partial_mask_blocks`, and `EMPTY` (0) is entirely masked out
        (used as a placeholder step so an inactive row still initializes and
        writes zero output).
      num_active_blocks: An integer[1] NumPy array containing the number of
        leading entries of the compacted arrays above that are valid. The
        remaining entries are padding.
      partial_mask_blocks: An int8[num_partial_blocks, block_q, block_kv] NumPy
        array that contains the blocks of the original mask that contained both
        zeros and ones. The entries in `mask_next` point to indices in the first
        axis of this array.
      q_sequence: A i32[q_sequence_length] NumPy array. When using causal masking,
        this contains the list of indices that correspond to q tokens. For plain
        causal this is just np.arange(q_sequence_length).
    """

    mask_next: np.ndarray | jax.Array | None
    active_rows: np.ndarray | jax.Array | None
    active_cols: np.ndarray | jax.Array | None
    block_mask: np.ndarray | jax.Array | None
    num_active_blocks: np.ndarray | jax.Array | None
    partial_mask_blocks: np.ndarray | jax.Array | None
    q_sequence: np.ndarray | jax.Array | None


def downcast_index_dtype(
    max_index: int,
    *,
    downcast_smem_data: bool = True,
) -> jnp.dtype:
    """Select the smallest signed integer dtype that holds `max_index`.

    Args:
      max_index: Largest non-negative index value stored in the array.
      downcast_smem_data: Whether to downcast below `int32` when possible.

    Returns:
      `jnp.dtype(jnp.int8)`, `jnp.dtype(jnp.int16)`, or `jnp.dtype(jnp.int32)`.
    """
    if not downcast_smem_data:
        return jnp.dtype(jnp.int32)
    if max_index <= np.iinfo(np.int8).max:
        return jnp.dtype(jnp.int8)
    if max_index <= np.iinfo(np.int16).max:
        return jnp.dtype(jnp.int16)
    return jnp.dtype(jnp.int32)


def _downcast_to_small_type(array: np.ndarray) -> np.ndarray:
    """Downcast numpy array.

    If possible, downcast the data-type of the input array to the smallest numpy
    type (among np.int16 and np.int8) that fits the content of the array.

    Args:
      array: the array to downcast

    Returns:
      The downcasted array.

    Raises:
      ValueError: if the input array is not np.int32 or if its elements are not
      all positive.
    """
    if array.dtype != np.int32:
        raise ValueError(f"Expected int32 input, but got {array.dtype}.")

    if not np.all(array >= -1):
        # Allow -1 for padding.
        raise ValueError("Expected non-negative array.")

    if array.size == 0:
        return array

    max_value = int(np.max(array))
    return array.astype(downcast_index_dtype(max_value))


class _HashableNDArray:
    """Helper to make a numpy array hashable: can be added associative containers.

    Attributes:
      array: The underlying numpy array.
    """

    __slots__ = ("array", "_hash")
    array: np.ndarray

    def __init__(self, array: np.ndarray):
        self.array = array
        self._hash = hash(array.tobytes())

    def __hash__(self):
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _HashableNDArray):
            return NotImplemented
        return np.array_equal(self.array, other.array, equal_nan=True)


def _generate_shard_metadata(
    block_mask: np.ndarray,
    partial_blocks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Build compacted per-shard block schedule arrays.

    Args:
      block_mask: 2D array of `BlockMaskCode` entries for one Q shard.
      partial_blocks: 2D array of partial block IDs (`-1` when not partial).

    Returns:
      Tuple `(active_rows, active_cols, mask_next, flat_block_mask, grid_size)`.
    """
    active_mask = block_mask > BlockMaskCode.EMPTY
    active_indices = np.argwhere(active_mask)
    active_rows = active_indices[:, 0].astype(np.int32)
    active_cols = active_indices[:, 1].astype(np.int32)
    block_mask = block_mask[active_mask]
    grid_size = active_rows.size

    partial_coords = np.argwhere(partial_blocks != -1)
    if partial_coords.size > 0:
        mask_next = []
        mask_coords_iter = iter([tuple(c) for c in partial_coords])
        first_m = coord_m = next(mask_coords_iter)

        for idx in active_indices:
            is_next_mask = tuple(idx) > tuple(coord_m)
            if is_next_mask:
                try:
                    coord_m = next(mask_coords_iter)
                except StopIteration:
                    coord_m = first_m
            mask_next.append(partial_blocks[coord_m])
    else:
        mask_next = np.full(block_mask.size, -1, dtype=np.int32)

    mask_next = np.array(mask_next, dtype=np.int32)
    flat_block_mask = block_mask.flatten()

    return active_rows, active_cols, mask_next, flat_block_mask, grid_size


def process_dynamic_mask(
    mask: jax.Array,
    block_shape: tuple[int, int],
    *,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
) -> MaskInfo:
    """Process a dynamic mask to compute its local sparsity data.

    Note that this operates on a single shard of the mask.

    Args:
      mask: [q_seq_len, kv_seq_len] jax.Array representing a dense mask to
        process.
      block_shape: A Tuple[int, int] representing the shape of the Pallas grid
        block.
      downcast_smem_data: If True, downcast the scalar-memory data of MaskInfo to
        a data type smaller than np.int32 (if possible).
      partial_mask_blocks_dtype: Element dtype for stored partial mask blocks.

    Returns:
      `MaskInfo`, a sparse representation of the dense mask.

    Raises:
      ValueError: if the input mask is invalid or the block sizes are not
      compatible with the mask sizes.
    """
    if len(mask.shape) != 2:
        raise ValueError(f"Expected a 2-dim mask, instead got: {mask.shape}.")

    q_seq_len, kv_seq_len = mask.shape
    q_block_size, kv_block_size = block_shape
    q_blocks_count, q_mod = divmod(q_seq_len, q_block_size)
    kv_blocks_count, kv_mod = divmod(kv_seq_len, kv_block_size)

    if q_mod != 0:
        raise ValueError(f"{q_block_size=} should divide {q_seq_len=}.")
    if kv_mod != 0:
        raise ValueError(f"{kv_block_size=} should divide {kv_seq_len=}.")

    # Tile the last 2 dimensions of the mask into 2D tiles of size `block_shape`.
    mask_blocks = (
        mask.reshape(
            q_blocks_count,
            q_block_size,
            kv_blocks_count,
            kv_block_size,
        )
        .swapaxes(-2, -3)
        .astype(partial_mask_blocks_dtype)
    )

    any_mask = jnp.any(mask_blocks, axis=(-1, -2)).astype(np.int32)
    all_mask = jnp.all(mask_blocks, axis=(-1, -2)).astype(np.int32)
    block_mask = any_mask + all_mask

    block_ids = jnp.arange(block_mask.size, dtype=np.int32).reshape(block_mask.shape)
    active_mask = block_mask > BlockMaskCode.EMPTY

    num_active_blocks = active_mask.flatten().sum(keepdims=True)
    active_indices = jnp.argwhere(active_mask, size=active_mask.size, fill_value=-1)
    active_rows = active_indices[:, 0].astype(np.int32)
    active_cols = active_indices[:, 1].astype(np.int32)

    block_mask = block_mask[active_rows, active_cols]
    mask_next = block_ids.at[active_rows, active_cols].get(wrap_negative_indices=False)
    mask_next = jnp.where(block_mask == BlockMaskCode.PARTIAL, mask_next, 0)

    # Mask out the blocks that aren't active.
    mask = (jnp.arange(block_mask.size) < num_active_blocks).astype(np.int32)
    block_mask = block_mask * mask

    # Collapsing because the block ids are linearized.
    mask_blocks = lax.collapse(mask_blocks, 0, 2)

    if downcast_smem_data:
        block_mask = block_mask.astype(np.int8)
        if mask_next.size > 0:
            mask_next = mask_next.astype(
                downcast_index_dtype(
                    q_blocks_count * kv_blocks_count,
                    downcast_smem_data=downcast_smem_data,
                )
            )

    return MaskInfo(
        mask_next=mask_next,
        active_rows=active_rows,
        active_cols=active_cols,
        block_mask=block_mask,
        num_active_blocks=num_active_blocks,
        partial_mask_blocks=mask_blocks,
        q_sequence=None,
    )


def _validate_mask_shape_and_shards(
    mask: mask_lib.Mask,
    block_shape: tuple[int, int],
    q_seq_shards: int,
) -> tuple[int, int, int]:
    """Validate mask rank, block divisibility, and Q shard divisibility.

    Args:
      mask: 2D mask to validate.
      block_shape: Tile shape `(q_block_size, kv_block_size)`.
      q_seq_shards: Number of shards along the Q sequence dimension.

    Returns:
      Tuple `(q_blocks_count, kv_blocks_count, q_blocks_per_shard)`.
    """
    if len(mask.shape) != _EXPECTED_MASK_NDIM:
        msg = f"Expected a 2-dim mask, instead got: {mask.shape=}"
        raise ValueError(msg)

    q_seq_len, kv_seq_len = mask.shape
    q_block_size, kv_block_size = block_shape
    q_blocks_count, q_mod = divmod(q_seq_len, q_block_size)
    kv_blocks_count, kv_mod = divmod(kv_seq_len, kv_block_size)

    if q_mod != 0:
        msg = f"{q_block_size=} should divide {q_seq_len=}."
        raise ValueError(msg)
    if kv_mod != 0:
        msg = f"{kv_block_size=} should divide {kv_seq_len=}."
        raise ValueError(msg)

    q_seq_len_per_shard, mod = divmod(q_seq_len, q_seq_shards)
    if mod != 0:
        msg = f"{q_seq_shards=} should divide {q_seq_len=}."
        raise ValueError(msg)

    q_blocks_per_shard, mod = divmod(q_seq_len_per_shard, q_block_size)
    if mod != 0:
        msg = f"{q_block_size=} should divide {q_seq_len_per_shard=}."
        raise ValueError(msg)

    return q_blocks_count, kv_blocks_count, q_blocks_per_shard


def _categorize_mask_blocks(
    mask: mask_lib.Mask,
    block_shape: tuple[int, int],
    q_blocks_count: int,
    kv_blocks_count: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Classify dense mask blocks into `BlockMaskCode` entries and unique partials.

    Args:
      mask: 2D mask to partition.
      block_shape: Tile shape `(q_block_size, kv_block_size)`.
      q_blocks_count: Total Q block count.
      kv_blocks_count: Total KV block count.

    Returns:
      Tuple `(state_grid, partial_id_grid, unique_chunks)`.
    """
    q_block_size, kv_block_size = block_shape
    blocked_shape = (q_blocks_count, kv_blocks_count)
    state_grid = np.full(blocked_shape, BlockMaskCode.EMPTY, dtype=np.int32)
    partial_id_grid = np.full(blocked_shape, -1, dtype=np.int32)
    partial_blocks_map: dict[_HashableNDArray, int] = {}
    unique_chunks: list[np.ndarray] = []

    for q_idx, kv_idx in np.ndindex(blocked_shape):
        chunk = mask[
            (
                slice(q_idx * q_block_size, (q_idx + 1) * q_block_size),
                slice(kv_idx * kv_block_size, (kv_idx + 1) * kv_block_size),
            )
        ]
        if chunk.any():
            if chunk.all():
                state_grid[q_idx, kv_idx] = BlockMaskCode.FULL
            else:
                state_grid[q_idx, kv_idx] = BlockMaskCode.PARTIAL
                key = _HashableNDArray(chunk)
                chunk_id = partial_blocks_map.get(key)
                if chunk_id is None:
                    chunk_id = len(unique_chunks)
                    partial_blocks_map[key] = chunk_id
                    unique_chunks.append(chunk)
                partial_id_grid[q_idx, kv_idx] = chunk_id

    return state_grid, partial_id_grid, unique_chunks


def _assemble_sharded_mask_info(
    state_grid: np.ndarray,
    partial_id_grid: np.ndarray,
    q_blocks_per_shard: int,
    q_seq_shards: int,
    *,
    downcast_smem_data: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build padded per-shard schedule arrays concatenated across Q shards.

    Args:
      state_grid: 2D grid of `BlockMaskCode` values.
      partial_id_grid: 2D grid of deduplicated partial block indices.
      q_blocks_per_shard: Number of Q blocks per shard.
      q_seq_shards: Number of Q sequence shards.
      downcast_smem_data: Whether to downcast schedule arrays below `int32`.

    Returns:
      Tuple `(active_rows, active_cols, mask_next, block_mask, num_active_blocks)`.
    """
    all_shards_metadata = [
        _generate_shard_metadata(
            state_grid[
                q_shard_idx * q_blocks_per_shard : (q_shard_idx + 1)
                * q_blocks_per_shard,
                :,
            ],
            partial_id_grid[
                q_shard_idx * q_blocks_per_shard : (q_shard_idx + 1)
                * q_blocks_per_shard,
                :,
            ],
        )
        for q_shard_idx in range(q_seq_shards)
    ]
    (
        active_rows_slices,
        active_cols_slices,
        mask_next_slices,
        block_mask_slices,
        num_active_blocks_tuple,
    ) = zip(*all_shards_metadata, strict=True)

    max_size = max(num_active_blocks_tuple)

    def _pad_slice(arr: np.ndarray) -> np.ndarray:
        """Pad a 1D per-shard array to `max_size` with `-1` sentinels."""
        return np.pad(
            arr, (0, max_size - arr.shape[0]), mode="constant", constant_values=-1
        )

    active_rows = np.concatenate(list(map(_pad_slice, active_rows_slices)), axis=0)
    active_cols = np.concatenate(list(map(_pad_slice, active_cols_slices)), axis=0)
    mask_next = np.concatenate(list(map(_pad_slice, mask_next_slices)), axis=0)
    block_mask = np.concatenate(list(map(_pad_slice, block_mask_slices)), axis=0)
    num_active_blocks = np.array(num_active_blocks_tuple, dtype=np.int32)

    if downcast_smem_data:
        active_rows = _downcast_to_small_type(active_rows)
        active_cols = _downcast_to_small_type(active_cols)
        mask_next = _downcast_to_small_type(mask_next)
        block_mask = _downcast_to_small_type(block_mask)

    return active_rows, active_cols, mask_next, block_mask, num_active_blocks


# When used in a transformer network with multiple layers, the SplashAttention
# kernel is created several times with the same mask. Cache MaskInfo to avoid
# blowing up compile times. Ideally the size of the cache should be determined
# by the client.
@functools.lru_cache(maxsize=12)
def process_mask(
    mask: mask_lib.Mask,  # [q_seq_len, kv_seq_len]
    block_shape: tuple[int, int],
    *,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
    q_seq_shards: int = 1,
) -> tuple[MaskInfo, MaskCallable | None]:
    """Transform a dense mask into a sparse representation.

    The number of Q sequence shards is needed to create a `MaskInfo`
    object that is partitionable (with `shard_map`) along that dimension.

    Args:
      mask: Dense mask to process.
      block_shape: Shape of the Pallas grid block.
      downcast_smem_data: If True, downcast the SMEM data of MaskInfo to a data
        type smaller than `int32` if possible.
      partial_mask_blocks_dtype: Element dtype for stored partial mask blocks.
      q_seq_shards: Number of Q sequence shards of the mesh in which the kernel is
        launched.

    Returns:
      Tuple `(mask_info, mask_function)` where `mask_info` holds sparse block
      metadata and `mask_function` computes partial masks in-kernel when `mask`
      is a `ComputableMask`.

    Raises:
      ValueError: if the input mask is invalid or the block sizes are not
      compatible with the mask sizes.
    """
    q_blocks_count, kv_blocks_count, q_blocks_per_shard = (
        _validate_mask_shape_and_shards(mask, block_shape, q_seq_shards)
    )

    if isinstance(mask, mask_lib.ComputableMask):
        q_sequence = mask.q_sequence
        mask_function = mask.mask_function
    else:
        q_sequence = mask_function = None

    state_grid, partial_id_grid, unique_chunks = _categorize_mask_blocks(
        mask, block_shape, q_blocks_count, kv_blocks_count
    )

    if (state_grid == BlockMaskCode.FULL).all():
        return MaskInfo(
            mask_next=None,
            active_rows=None,
            active_cols=None,
            block_mask=None,
            num_active_blocks=None,
            partial_mask_blocks=None,
            q_sequence=None,
        ), None

    partial_mask_blocks = (
        np.stack(unique_chunks).astype(partial_mask_blocks_dtype)
        if unique_chunks
        else None
    )
    active_rows, active_cols, mask_next, block_mask, num_active_blocks = (
        _assemble_sharded_mask_info(
            state_grid,
            partial_id_grid,
            q_blocks_per_shard,
            q_seq_shards,
            downcast_smem_data=downcast_smem_data,
        )
    )

    if partial_mask_blocks is None:
        mask_next = None

    return (
        MaskInfo(
            mask_next=mask_next if mask_function is None else None,
            active_rows=active_rows,
            active_cols=active_cols,
            block_mask=block_mask,
            num_active_blocks=num_active_blocks,
            partial_mask_blocks=partial_mask_blocks if mask_function is None else None,
            q_sequence=q_sequence,
        ),
        mask_function,
    )

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Block-major unified KV pool layout calculations for TPU.

Folds per-layer unified pools into one physical allocation ordered
`(num_blocks, rows_per_block, ...)`.

Strictly zero dependencies on vLLM.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


def is_block_major_layout_applicable(
    block_major_flag: bool, is_unified_kv_layout: bool, is_hybrid: bool
) -> bool:
    """Returns whether unified hybrid pools should be merged into one
    block-major pool.
    """
    if not block_major_flag:
        return False
    return is_unified_kv_layout and is_hybrid


@dataclass(frozen=True)
class BlockMajorPoolLayout:
    """Geometry of the merged block-major unified pool.

    Attributes:
        num_blocks: Number of scheduler (manager) blocks (`shape[0]`).
        num_pools: Number of layer regions packed into each block row (`P`).
        split: Number of kernel blocks per manager block within one region.
        pool_index_by_layer: Mapping from layer name to its region index `[0, P)`.
    """

    num_blocks: int
    num_pools: int
    split: int
    pool_index_by_layer: Mapping[str, int]

    @property
    def rows_per_block(self) -> int:
        """Total kernel blocks per scheduler block row (`shape[1]`)."""
        return self.num_pools * self.split


def kernel_view(pool: Any) -> Any:
    """Folds the merged pool's leading `(num_blocks, rows_per_block)` axes into dim 0.

    Reshapes `(num_blocks, rows_per_block, *page)` into
    `(num_blocks * rows_per_block, *page)` so kernels can index flat kernel
    blocks along dim 0. Operates on both JAX and PyTorch tensors and compiles
    to a zero-copy bitcast on the donated buffer.
    """
    return pool.reshape((pool.shape[0] * pool.shape[1],) + tuple(pool.shape[2:]))


def flat_manager_ids(
    ids: torch.Tensor, pool_index: int, num_pools: int
) -> torch.Tensor:
    """Maps region-local manager block IDs to flat row indices in the merged pool."""
    if num_pools == 1:
        return ids
    return ids * num_pools + pool_index


def flat_kernel_ids(
    ids: torch.Tensor, pool_index: int, num_pools: int, split: int
) -> torch.Tensor:
    """Maps region-local kernel block IDs to flat `kernel_view(pool)` block IDs.

    Each manager block spans `split` consecutive kernel blocks in both
    region-local and merged numbering, so only the manager-block quotient is
    strided by `num_pools` while the intra-block offset is preserved. Padding
    entries (ID 0) remain within the reserved null manager block (block 0).
    """
    if num_pools == 1:
        return ids
    if split == 1:
        return ids * num_pools + pool_index
    manager = ids // split
    return (manager * num_pools + pool_index) * split + (ids - manager * split)


BLOCK_MAJOR_LAYOUT_VERSION = 1


@dataclass(frozen=True)
class BlockMajorContract:
    """Immutable specification of the bundled KV layout agreed upon across ranks.

    Attributes:
        fragment_count: Number of physical fragments folded into the bundle's layer
            dimension (dim 1).
        fragment_row_bytes: Byte size of a single fragment's row for one kernel block.
        bundle_row_bytes: Total contiguous byte length of one bundled kernel block
            across all fragments (= F * fragment_row_bytes).
        logical_fingerprint: Deterministic SHA-256 digest of layout parameters; must
            match across all serving peers.
    """

    fragment_count: int
    fragment_row_bytes: int
    bundle_row_bytes: int
    logical_fingerprint: str


def canonical_layout_fingerprint(value: Mapping[str, Any]) -> str:
    """Computes a deterministic SHA-256 fingerprint from a canonical
    JSON layout payload."""
    import hashlib
    import json

    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()

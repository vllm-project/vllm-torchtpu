# Copyright 2026 Google LLC
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
"""Block-major layout for the unified hybrid KV pool.

The unified layout gives a hybrid model `P` attention-shaped pools, one per
layer region (a full-attention layer and the GDN layers paired with it). The
block-major layout is one pool of shape `(num_blocks, P * split, *page)`:
- Row `b` holds scheduler block `b` for all `P` regions.
- Region `p` occupies kernel blocks `[p * split, (p + 1) * split)` of each row.

A scheduler block is one contiguous `P * page_bytes` row, so the offload tier
and the transfer plane move it as one DMA and register the pool as allocated.

Attention kernels, pooled GDN ops and Mamba state seed copies index kernel
blocks along one leading dimension: each consumer folds the leading axes with
`kernel_view` (a zero-copy bitcast inside the compiled op) and remaps its block
IDs before the kernel runs, which keeps in-place donation and aliasing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm_torchtpu import envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def unified_block_major_enabled(vllm_config: VllmConfig) -> bool:
    """Returns whether unified hybrid pools are merged into one block-major pool.

    Resolved purely from `VllmConfig` so the scheduler process, runner buffer
    allocation, and layer kernel construction remain consistent. Enabled only
    for hybrid models on the unified KV layout when `VLLM_TPU_BLOCK_MAJOR_KV`
    is set; dense models retain per-layer caches and the dense block-major bundle.
    """
    if not envs.VLLM_TPU_BLOCK_MAJOR_KV:
        return False
    from vllm_torchtpu.platforms.tpu_block_size_utils import unified_kv_layout_enabled

    return unified_kv_layout_enabled(vllm_config) and vllm_config.model_config.is_hybrid


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


def kernel_view(pool):
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

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Block tables that hold kernel block ids.

When a KV cache group's manager block is larger than the attention kernel's
page, each manager block is ``split`` consecutive kernel pages in the pool.
The TPU attention metadata copies block-table rows straight into the kernels'
page tables, so those rows must hold kernel block ids.

vLLM 0.31 did that split inside ``BlockTable`` (``kernel_block_size``).
vLLM 0.32 removed it (https://github.com/vllm-project/vllm/pull/57169):
``BlockTable`` now holds manager ids, and the GPU runner maps them per
attention group while it builds metadata. The TPU runner builds its own
metadata, so ``KernelBlockTable`` keeps the 0.31 behavior for the groups that
split.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable


class KernelBlockTable(BlockTable):
    """A ``BlockTable`` whose rows hold kernel block ids.

    Manager block ``b`` becomes kernel blocks ``b * split .. b * split +
    split - 1``. ``block_size`` and ``max_num_blocks_per_req`` are in kernel
    blocks; ``kv_cache_block_size`` is the manager block size.
    """

    kv_cache_block_size: int
    blocks_per_kv_block: int

    @classmethod
    def from_block_table(
        cls, table: BlockTable, kernel_block_size: int
    ) -> KernelBlockTable:
        """An empty kernel-block copy of ``table``."""
        split, remainder = divmod(table.block_size, kernel_block_size)
        if remainder or split < 2:
            raise ValueError(
                f"kernel_block_size {kernel_block_size} must divide the "
                f"manager block size {table.block_size} into 2 or more blocks"
            )
        if table.num_blocks_per_row.any():
            raise RuntimeError("Cannot replace a block table that holds requests.")
        kernel_table = cls(
            kernel_block_size,
            table.max_num_reqs,
            table.max_num_blocks_per_req * split,
            table.max_num_batched_tokens,
            table.pin_memory,
            table.device,
            table.cp_kv_cache_interleave_size,
            slot_mapping_mode=table.slot_mapping_mode,
        )
        kernel_table.kv_cache_block_size = table.block_size
        kernel_table.blocks_per_kv_block = split
        return kernel_table

    def append_row(self, block_ids: list[int], row_idx: int) -> None:
        if not block_ids:
            return
        split = self.blocks_per_kv_block
        kernel_ids = np.asarray(block_ids, dtype=np.int64)[:, None] * split
        kernel_ids = kernel_ids + np.arange(split, dtype=np.int64)
        super().append_row(kernel_ids.reshape(-1).tolist(), row_idx)


def install_kernel_block_tables(
    block_tables: MultiGroupBlockTable, kernel_block_sizes: Sequence[int]
) -> None:
    """Give each group whose kernel block is smaller than its manager block a
    ``KernelBlockTable``.

    ``kernel_block_sizes`` is ``prepare_kernel_block_sizes`` output, one entry
    per block table. Call this before any request is added.
    """
    tables = block_tables.block_tables
    if len(kernel_block_sizes) != len(tables):
        raise ValueError(
            f"Got {len(kernel_block_sizes)} kernel block sizes for "
            f"{len(tables)} block tables."
        )
    for gid, (table, kernel_block_size) in enumerate(zip(tables, kernel_block_sizes)):
        if isinstance(table, KernelBlockTable):
            if table.block_size != kernel_block_size:
                raise ValueError(
                    f"Block table {gid} already splits into "
                    f"{table.block_size}-token blocks, not {kernel_block_size}."
                )
            continue
        if kernel_block_size != table.block_size:
            tables[gid] = KernelBlockTable.from_block_table(table, kernel_block_size)

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
"""Block tables that hold kernel block ids for split KV cache groups."""

import pytest
import torch
from vllm.v1.worker.block_table import MultiGroupBlockTable, SlotMappingMode

from vllm_torchtpu.runner.kernel_block_table import (
    KernelBlockTable,
    install_kernel_block_tables,
)

MAX_NUM_REQS = 4


def _block_tables(block_sizes, slot_mapping_modes=None) -> MultiGroupBlockTable:
    return MultiGroupBlockTable(
        max_num_reqs=MAX_NUM_REQS,
        max_num_batched_tokens=64,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=block_sizes,
        max_num_blocks=[8] * len(block_sizes),
        slot_mapping_modes=slot_mapping_modes,
    )


def test_only_split_groups_get_kernel_block_tables():
    # An attention group split 1024 -> 256, one that is not split, and a
    # Mamba group, whose state blocks are never split.
    tables = _block_tables(
        [1024, 256, 1024],
        [
            SlotMappingMode.TOKEN_TO_KV_SLOT,
            SlotMappingMode.TOKEN_TO_KV_SLOT,
            SlotMappingMode.NONE,
        ],
    )
    unsplit, mamba = tables.block_tables[1], tables.block_tables[2]
    manager_width = tables.block_tables[0].max_num_blocks_per_req

    install_kernel_block_tables(tables, [256, 256, 1024])

    split = tables.block_tables[0]
    assert isinstance(split, KernelBlockTable)
    assert split.block_size == 256
    assert split.kv_cache_block_size == 1024
    assert split.blocks_per_kv_block == 4
    assert split.max_num_blocks_per_req == manager_width * 4
    assert split.get_cpu_tensor().shape == (MAX_NUM_REQS, manager_width * 4)
    assert tables.block_tables[1] is unsplit
    assert tables.block_tables[2] is mamba


def test_rows_hold_kernel_block_ids():
    tables = _block_tables([512])
    install_kernel_block_tables(tables, [256])

    tables.add_row(([3, 7],), 1)
    tables.append_row(([2],), 1)
    row = tables.block_tables[0].get_numpy_array()[1]
    assert row[:6].tolist() == [6, 7, 14, 15, 4, 5]
    assert tables.block_tables[0].num_blocks_per_row[1] == 6

    tables.move_row(1, 0)
    assert tables.block_tables[0].get_numpy_array()[0, :6].tolist() == [
        6,
        7,
        14,
        15,
        4,
        5,
    ]
    tables.append_row(([],), 0)
    assert tables.block_tables[0].num_blocks_per_row[0] == 6


def test_install_is_idempotent():
    tables = _block_tables([512])
    install_kernel_block_tables(tables, [256])
    kernel_table = tables.block_tables[0]

    install_kernel_block_tables(tables, [256])
    assert tables.block_tables[0] is kernel_table

    with pytest.raises(ValueError, match="already splits"):
        install_kernel_block_tables(tables, [128])


def test_install_rejects_bad_inputs():
    tables = _block_tables([512])
    with pytest.raises(ValueError, match="1 block tables"):
        install_kernel_block_tables(tables, [256, 256])
    with pytest.raises(ValueError, match="must divide"):
        install_kernel_block_tables(tables, [384])

    tables.add_row(([1],), 0)
    with pytest.raises(RuntimeError, match="holds requests"):
        install_kernel_block_tables(tables, [256])

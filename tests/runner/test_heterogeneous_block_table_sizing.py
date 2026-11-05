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
"""Regression test for heterogeneous MultiGroupBlockTable max_num_blocks_per_req."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def test_may_reinitialize_input_batch_uses_max_across_heterogeneous_groups():
    """When group 0 has a larger block size than group 1 (e.g. [512, 256]),
    block_table[0].max_num_blocks_per_req is smaller than block_table[1]'s.
    block_table_cpu must be sized to the maximum across all groups.
    """
    bt0 = SimpleNamespace(max_num_blocks_per_req=16)
    bt1 = SimpleNamespace(max_num_blocks_per_req=32)

    class FakeMultiGroupBlockTable:
        def __init__(self):
            self.block_tables = [bt0, bt1]

        def __bool__(self):
            return True

        def __getitem__(self, idx: int):
            return self.block_tables[idx]

    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.max_num_reqs = 8
    runner.max_num_blocks_per_req = 16
    runner.block_table_cpu = torch.zeros((8, 16), dtype=torch.int32)
    runner.input_batch = SimpleNamespace(block_table=FakeMultiGroupBlockTable())

    with (
        patch.object(
            TPUModelRunner.__mro__[1],
            "may_reinitialize_input_batch",
            return_value=None,
        ),
        patch(
            "vllm_torchtpu.runner.tpu_runner.install_kernel_block_tables",
            return_value=None,
        ),
        patch.object(
            TPUModelRunner,
            "_refresh_smem_request_caps",
            return_value=None,
        ),
    ):
        runner.may_reinitialize_input_batch(
            SimpleNamespace(kv_cache_groups=[]), [256, 256]
        )

    assert runner.max_num_blocks_per_req == 32
    assert runner.block_table_cpu.shape == (8, 32)

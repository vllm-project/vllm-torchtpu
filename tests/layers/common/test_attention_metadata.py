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

from unittest.mock import MagicMock

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_torchtpu.layers.common.attention_metadata import (
    AttentionMetadataBuilder, AttentionMetadataBuilderContext)
from vllm_torchtpu.layers.common.sequence_layout import (
    PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL, SequenceLayoutDescriptor,
    SequenceLayoutKind)


class TestAttentionMetadataBuilderPlumbing:

    def _make_runner_mock(self,
                          most_model_len=None,
                          num_groups=1,
                          max_num_blocks_per_req=4,
                          max_num_reqs=4):
        runner = MagicMock()
        runner.device = torch.device("cpu")
        runner.block_size = 16
        runner.max_num_reqs = max_num_reqs
        runner.most_model_len = most_model_len
        runner.position_ids = torch.full((8, ), 42, dtype=torch.int32)

        block_tables = []
        for gid in range(num_groups):
            bt = MagicMock()
            bt.max_num_blocks_per_req = max_num_blocks_per_req
            bt.get_cpu_tensor.return_value = (
                torch.arange(max_num_reqs * max_num_blocks_per_req,
                             dtype=torch.int32).reshape(
                                 max_num_reqs, max_num_blocks_per_req) +
                gid * 100)
            block_tables.append(bt)
        runner.input_batch.block_table = block_tables
        return runner

    def _make_builder(self, runner, kv_cache_group_id=0):
        spec = FullAttentionSpec(block_size=16,
                                 num_kv_heads=2,
                                 head_size=128,
                                 dtype=torch.bfloat16,
                                 page_size_padded=16384)
        return AttentionMetadataBuilder(
            kv_cache_spec=spec,
            layer_names=["attn.0"],
            vllm_config=MagicMock(),
            device=runner.device,
            runner=runner,
            kv_cache_group_id=kv_cache_group_id,
        )

    def _make_mamba_builder(self, runner, kv_cache_group_id=0):
        spec = MambaSpec(block_size=16,
                         shapes=[(4, 128), (8, 64, 32)],
                         dtypes=[torch.bfloat16, torch.float32],
                         page_size_padded=16384)
        return AttentionMetadataBuilder(
            kv_cache_spec=spec,
            layer_names=["gdn.0"],
            vllm_config=MagicMock(),
            device=runner.device,
            runner=runner,
            kv_cache_group_id=kv_cache_group_id,
        )

    def _make_cm(self, num_reqs):
        cm = MagicMock()
        cm.num_reqs = num_reqs
        return cm

    def test_build_position_ids_override_uses_zero_block_table(self):
        runner = self._make_runner_mock()
        builder = self._make_builder(runner)

        position_override = torch.zeros((3, 8), dtype=torch.int32)
        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=4,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.ones((4, ), dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
            position_ids_override=position_override,
        )

        meta = builder.build(common_prefix_len=0,
                             common_attn_metadata=self._make_cm(4))

        assert meta.input_positions is position_override
        runner.input_batch.block_table[0].get_cpu_tensor.assert_not_called()
        assert torch.equal(meta.block_tables, torch.zeros(16,
                                                          dtype=torch.int32))

    def test_build_carries_sequence_layout_metadata(self):
        runner = self._make_runner_mock()
        builder = self._make_builder(runner)

        descriptor = SequenceLayoutDescriptor(
            kind=SequenceLayoutKind.PARTIAL,
            protocol=PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL,
            version=7,
        )
        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=2,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.ones((4, ), dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
            sequence_layout_descriptor=descriptor,
        )

        meta = builder.build(common_prefix_len=0,
                             common_attn_metadata=self._make_cm(4))

        assert meta.sequence_layout_kind == SequenceLayoutKind.PARTIAL.value
        assert (meta.sequence_layout_protocol ==
                PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL)
        assert meta.sequence_layout_version == 7

    def test_build_mamba_state_indices_from_current_block_table_entry(self):
        runner = self._make_runner_mock(max_num_blocks_per_req=4)
        builder = self._make_mamba_builder(runner)

        seq_lens = torch.tensor([1, 16, 17, 64], dtype=torch.int32)
        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=4,
            start_index=0,
            use_max_model_len=True,
            seq_lens=seq_lens,
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
        )

        meta = builder.build(common_prefix_len=0,
                             common_attn_metadata=self._make_cm(4))

        block_tables = (
            runner.input_batch.block_table[0].get_cpu_tensor.return_value)
        expected = torch.tensor([
            block_tables[0, 0],
            block_tables[1, 0],
            block_tables[2, 1],
            block_tables[3, 3],
        ],
                                dtype=torch.int32)
        assert torch.equal(meta.mamba_state_indices, expected)

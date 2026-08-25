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
        runner._unified_kv_layout = True
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

    def test_ckpt_window_names_the_blocks_after_the_state_block(self):
        """Checkpoint t is the row entry `state_block_offset + t`.

        The manager allocates the `window - 1` speculative blocks by
        inflating the request's token count, so they sit immediately after
        the positional state block; checkpoint 0 must stay the state block
        itself, keeping `mamba_state_indices` its column 0.
        """
        window = 3
        runner = self._make_runner_mock(max_num_blocks_per_req=6)
        runner._unified_kv_layout = True
        builder = self._make_mamba_builder(runner)

        seq_lens = torch.tensor([1, 17, 33, 33], dtype=torch.int32)
        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=4,
            start_index=0,
            use_max_model_len=True,
            seq_lens=seq_lens,
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
            mamba_ckpt_window=window,
        )

        meta = builder.build(common_prefix_len=0,
                             common_attn_metadata=self._make_cm(4))

        block_tables = (
            runner.input_batch.block_table[0].get_cpu_tensor.return_value)
        # (seq_len - 1) // 16 for a 16-token block: 0, 1, 2, 2.
        base_cols = [0, 1, 2, 2]
        assert meta.mamba_ckpt_indices is not None
        assert meta.mamba_ckpt_indices.shape == (4, window)
        for row, col0 in enumerate(base_cols):
            for t in range(window):
                assert (meta.mamba_ckpt_indices[row,
                                                t] == block_tables[row, col0 +
                                                                   t]), (row,
                                                                         t)
        # Checkpoint 0 is the state block the affine path also uses.
        assert torch.equal(meta.mamba_ckpt_indices[:, 0],
                           meta.mamba_state_indices)

    def test_no_ckpt_indices_without_a_window(self):
        runner = self._make_runner_mock(max_num_blocks_per_req=4)
        runner._unified_kv_layout = True
        builder = self._make_mamba_builder(runner)
        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=2,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.tensor([1, 16], dtype=torch.int32),
            query_start_loc=torch.arange(3, dtype=torch.int32),
            request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
        )
        meta = builder.build(common_prefix_len=0,
                             common_attn_metadata=self._make_cm(2))
        assert meta.mamba_ckpt_indices is None

    def test_most_model_len_row_covers_the_checkpoint_blocks(self):
        """The `most_model_len` row must include the speculative blocks.

        `MambaSpec.max_num_blocks_per_req` is
        `cdiv(max_model_len, B) + num_speculative_blocks`, which the
        `use_max_model_len` branch picks up for free. Deriving the shorter
        row from `most_model_len` alone stops before the checkpoint group,
        and the in-row mask then silently redirects checkpoints to the null
        block -- rollback accuracy lost with no error.
        """
        window = 3
        runner = self._make_runner_mock(most_model_len=32,
                                        max_num_blocks_per_req=6)
        runner._unified_kv_layout = True
        builder = self._make_mamba_builder(runner)

        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=4,
            start_index=0,
            use_max_model_len=False,
            seq_lens=torch.full((4, ), 17, dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
            mamba_ckpt_window=window,
        )
        meta = builder.build(common_prefix_len=0,
                             common_attn_metadata=self._make_cm(4))

        # cdiv(32, 16) = 2 positional columns + (window - 1) checkpoints.
        assert meta.block_tables.numel() == 4 * 4
        block_tables = (
            runner.input_batch.block_table[0].get_cpu_tensor.return_value)
        # (17 - 1) // 16 = column 1, so checkpoints are columns 1, 2, 3.
        for row in range(4):
            for t in range(window):
                assert (meta.mamba_ckpt_indices[row, t] == block_tables[row,
                                                                        1 + t])
        # Nothing was masked onto the null block.
        assert int(meta.mamba_ckpt_indices.min()) > 0

    def test_most_model_len_row_unchanged_without_a_window(self):
        # No speculative blocks are allocated without spec decode, so the
        # row stays exactly cdiv(most_model_len, block_size) wide.
        runner = self._make_runner_mock(most_model_len=32,
                                        max_num_blocks_per_req=6)
        runner._unified_kv_layout = True
        builder = self._make_mamba_builder(runner)
        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=4,
            start_index=0,
            use_max_model_len=False,
            seq_lens=torch.full((4, ), 17, dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
        )
        meta = builder.build(common_prefix_len=0,
                             common_attn_metadata=self._make_cm(4))
        assert meta.block_tables.numel() == 4 * 2

    def test_unified_spec_stashes_state_indices_for_offset_scatter(self):
        # With spec decoding the runner seeds unified_mamba_state_indices
        # with an empty list; each mamba group's builder appends the state
        # blocks it derived so the post-sampling read-offset scatter can
        # address them.
        runner = self._make_runner_mock(max_num_blocks_per_req=4)
        runner._unified_kv_layout = True
        builder = self._make_mamba_builder(runner)

        ctx = AttentionMetadataBuilderContext(
            num_reqs=4,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.tensor([1, 16, 17, 64], dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
            unified_mamba_state_indices=[],
        )
        runner._attn_metadata_builder_ctx = ctx

        meta = builder.build(common_prefix_len=0,
                             common_attn_metadata=self._make_cm(4))

        assert len(ctx.unified_mamba_state_indices) == 1
        assert ctx.unified_mamba_state_indices[0] is meta.mamba_state_indices

    def test_unified_without_spec_does_not_stash_state_indices(self):
        runner = self._make_runner_mock(max_num_blocks_per_req=4)
        runner._unified_kv_layout = True
        builder = self._make_mamba_builder(runner)

        ctx = AttentionMetadataBuilderContext(
            num_reqs=4,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.tensor([1, 16, 17, 64], dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
        )
        runner._attn_metadata_builder_ctx = ctx

        builder.build(common_prefix_len=0,
                      common_attn_metadata=self._make_cm(4))

        assert ctx.unified_mamba_state_indices is None

    def test_unified_none_mode_derives_state_indices_from_block_table(self):
        runner = self._make_runner_mock(max_num_blocks_per_req=4)
        runner._unified_kv_layout = True
        builder = self._make_mamba_builder(runner)

        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=4,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.tensor([1, 16, 17, 64], dtype=torch.int32),
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

    def test_mamba_groups_share_row_plan_but_not_block_tables(self):
        """The shared row plan must not leak one group's blocks into another.

        Every mamba group derives the same row offsets from `seq_lens`, so the
        derivation runs once per chunk and all of them reuse it — only the
        gather into each group's own block table stays per-group. This pins
        that sharing the offsets does not also share the blocks.
        """
        window = 3
        runner = self._make_runner_mock(num_groups=2, max_num_blocks_per_req=6)
        runner._unified_kv_layout = True
        ctx = AttentionMetadataBuilderContext(
            num_reqs=4,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.tensor([1, 17, 33, 33], dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
            mamba_ckpt_window=window,
        )
        runner._attn_metadata_builder_ctx = ctx

        metas = [
            self._make_mamba_builder(runner, kv_cache_group_id=gid).build(
                common_prefix_len=0, common_attn_metadata=self._make_cm(4))
            for gid in range(2)
        ]

        # Derived once, reused by the second group.
        assert len(ctx.mamba_row_plans) == 1

        # (seq_len - 1) // 16 for a 16-token block: 0, 1, 2, 2.
        base_cols = [0, 1, 2, 2]
        for gid, meta in enumerate(metas):
            block_tables = (runner.input_batch.block_table[gid].get_cpu_tensor.
                            return_value)
            for row, col0 in enumerate(base_cols):
                for t in range(window):
                    assert (
                        meta.mamba_ckpt_indices[row,
                                                t] == block_tables[row, col0 +
                                                                   t]), (gid,
                                                                         row,
                                                                         t)

        # Group 1's block ids sit 100 above group 0's; a plan that carried
        # blocks rather than offsets would have collapsed them together.
        assert not torch.equal(metas[0].mamba_state_indices,
                               metas[1].mamba_state_indices)

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

from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionMetadataBuilderContext,
    _gather_prefill_ckpt_ids,
    stage_block_table_uploads,
)
from vllm_torchtpu.layers.core.sequence_layout import (
    PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL,
    SequenceLayoutDescriptor,
    SequenceLayoutKind,
)


class TestAttentionMetadataBuilderPlumbing:
    def _make_runner_mock(
        self,
        most_model_len=None,
        num_groups=1,
        max_num_blocks_per_req=4,
        max_num_reqs=4,
    ):
        runner = MagicMock()
        runner.device = torch.device("cpu")
        runner.block_size = 16
        runner.max_num_reqs = max_num_reqs
        runner.most_model_len = most_model_len
        runner.position_ids = torch.full((8,), 42, dtype=torch.int32)
        runner._block_table_stage_cpu = None
        runner.attn_groups = []
        runner._unified_kv_layout = False

        block_tables = []
        for gid in range(num_groups):
            bt = MagicMock()
            bt.max_num_blocks_per_req = max_num_blocks_per_req
            bt.get_cpu_tensor.return_value = (
                torch.arange(
                    max_num_reqs * max_num_blocks_per_req, dtype=torch.int32
                ).reshape(max_num_reqs, max_num_blocks_per_req)
                + gid * 100
            )
            block_tables.append(bt)
        runner.input_batch.block_table = block_tables
        return runner

    def _make_builder(self, runner, kv_cache_group_id=0):
        spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            page_size_padded=16384,
        )
        return AttentionMetadataBuilder(
            kv_cache_spec=spec,
            layer_names=["attn.0"],
            vllm_config=MagicMock(),
            device=runner.device,
            runner=runner,
            kv_cache_group_id=kv_cache_group_id,
        )

    def _make_mamba_builder(self, runner, kv_cache_group_id=0):
        spec = MambaSpec(
            block_size=16,
            shapes=[(4, 128), (8, 64, 32)],
            dtypes=[torch.bfloat16, torch.float32],
            page_size_padded=16384,
        )
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
            seq_lens=torch.ones((4,), dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
            position_ids_override=position_override,
        )

        meta = builder.build(common_prefix_len=0, common_attn_metadata=self._make_cm(4))

        assert meta.input_positions is position_override
        runner.input_batch.block_table[0].get_cpu_tensor.assert_not_called()
        assert torch.equal(meta.block_tables, torch.zeros(16, dtype=torch.int32))

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
            seq_lens=torch.ones((4,), dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
            sequence_layout_descriptor=descriptor,
        )

        meta = builder.build(common_prefix_len=0, common_attn_metadata=self._make_cm(4))

        assert meta.sequence_layout_kind == SequenceLayoutKind.PARTIAL.value
        assert meta.sequence_layout_protocol == PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL
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

        meta = builder.build(common_prefix_len=0, common_attn_metadata=self._make_cm(4))

        block_tables = runner.input_batch.block_table[0].get_cpu_tensor.return_value
        expected = torch.tensor(
            [
                block_tables[0, 0],
                block_tables[1, 0],
                block_tables[2, 1],
                block_tables[3, 3],
            ],
            dtype=torch.int32,
        )
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

        meta = builder.build(common_prefix_len=0, common_attn_metadata=self._make_cm(4))

        block_tables = runner.input_batch.block_table[0].get_cpu_tensor.return_value
        # (seq_len - 1) // 16 for a 16-token block: 0, 1, 2, 2.
        base_cols = [0, 1, 2, 2]
        assert meta.mamba_ckpt_indices is not None
        assert meta.mamba_ckpt_indices.shape == (4, window)
        for row, col0 in enumerate(base_cols):
            for t in range(window):
                assert meta.mamba_ckpt_indices[row, t] == block_tables[row, col0 + t], (
                    row,
                    t,
                )
        # Checkpoint 0 is the state block the affine path also uses.
        assert torch.equal(meta.mamba_ckpt_indices[:, 0], meta.mamba_state_indices)

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
        meta = builder.build(common_prefix_len=0, common_attn_metadata=self._make_cm(2))
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
        runner = self._make_runner_mock(most_model_len=32, max_num_blocks_per_req=6)
        runner._unified_kv_layout = True
        builder = self._make_mamba_builder(runner)

        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=4,
            start_index=0,
            use_max_model_len=False,
            seq_lens=torch.full((4,), 17, dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
            mamba_ckpt_window=window,
        )
        meta = builder.build(common_prefix_len=0, common_attn_metadata=self._make_cm(4))

        # cdiv(32, 16) = 2 positional columns + (window - 1) checkpoints.
        assert meta.block_tables.numel() == 4 * 4
        block_tables = runner.input_batch.block_table[0].get_cpu_tensor.return_value
        # (17 - 1) // 16 = column 1, so checkpoints are columns 1, 2, 3.
        for row in range(4):
            for t in range(window):
                assert meta.mamba_ckpt_indices[row, t] == block_tables[row, 1 + t]
        # Nothing was masked onto the null block.
        assert int(meta.mamba_ckpt_indices.min()) > 0

    def test_most_model_len_row_unchanged_without_a_window(self):
        # No speculative blocks are allocated without spec decode, so the
        # row stays exactly cdiv(most_model_len, block_size) wide.
        runner = self._make_runner_mock(most_model_len=32, max_num_blocks_per_req=6)
        runner._unified_kv_layout = True
        builder = self._make_mamba_builder(runner)
        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=4,
            start_index=0,
            use_max_model_len=False,
            seq_lens=torch.full((4,), 17, dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
        )
        meta = builder.build(common_prefix_len=0, common_attn_metadata=self._make_cm(4))
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

        meta = builder.build(common_prefix_len=0, common_attn_metadata=self._make_cm(4))

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

        builder.build(common_prefix_len=0, common_attn_metadata=self._make_cm(4))

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

        meta = builder.build(common_prefix_len=0, common_attn_metadata=self._make_cm(4))

        block_tables = runner.input_batch.block_table[0].get_cpu_tensor.return_value
        expected = torch.tensor(
            [
                block_tables[0, 0],
                block_tables[1, 0],
                block_tables[2, 1],
                block_tables[3, 3],
            ],
            dtype=torch.int32,
        )
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
                common_prefix_len=0, common_attn_metadata=self._make_cm(4)
            )
            for gid in range(2)
        ]

        # Derived once, reused by the second group.
        assert len(ctx.mamba_row_plans) == 1

        # (seq_len - 1) // 16 for a 16-token block: 0, 1, 2, 2.
        base_cols = [0, 1, 2, 2]
        for gid, meta in enumerate(metas):
            block_tables = runner.input_batch.block_table[
                gid
            ].get_cpu_tensor.return_value
            for row, col0 in enumerate(base_cols):
                for t in range(window):
                    assert (
                        meta.mamba_ckpt_indices[row, t] == block_tables[row, col0 + t]
                    ), (gid, row, t)

        # Group 1's block ids sit 100 above group 0's; a plan that carried
        # blocks rather than offsets would have collapsed them together.
        assert not torch.equal(
            metas[0].mamba_state_indices, metas[1].mamba_state_indices
        )

    def _make_attn_groups(self, runner, builders):
        """Wrap builders in the runner.attn_groups structure the walks use."""
        groups = []
        for builder in builders:
            group = MagicMock()
            group.metadata_builders = [builder]
            group.layer_names = list(builder.layer_names)
            groups.append([group])
        runner.attn_groups = groups
        return groups

    def test_staged_build_matches_unstaged_build(self):
        """One batched upload must hand every group exactly what it would
        have uploaded itself — including the batched mamba derivation, which
        engages when at least two uniform mamba groups are staged. Padded
        rows (seq_lens == 0) are included to pin the null-block handling."""
        window = 3
        num_reqs = 4

        def make(staged):
            runner = self._make_runner_mock(num_groups=3, max_num_blocks_per_req=6)
            runner._unified_kv_layout = True
            runner._block_table_stage_cpu = None
            attn = self._make_builder(runner, kv_cache_group_id=0)
            mambas = [
                self._make_mamba_builder(runner, kv_cache_group_id=gid)
                for gid in (1, 2)
            ]
            self._make_attn_groups(runner, [attn] + mambas)
            ctx = AttentionMetadataBuilderContext(
                num_reqs=num_reqs,
                start_index=0,
                use_max_model_len=True,
                seq_lens=torch.tensor([0, 17, 33, 33], dtype=torch.int32),
                query_start_loc=torch.arange(5, dtype=torch.int32),
                request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
                mamba_ckpt_window=window,
                unified_mamba_state_indices=[],
            )
            runner._attn_metadata_builder_ctx = ctx
            if staged:
                stage_block_table_uploads(runner, ctx, num_reqs)
                assert ctx.staged_block_tables is not None
                assert set(ctx.staged_block_tables) == {0, 1, 2}
                assert ctx.staged_mamba_products is not None
                assert set(ctx.staged_mamba_products) == {1, 2}
            metas = [
                b.build(
                    common_prefix_len=0, common_attn_metadata=self._make_cm(num_reqs)
                )
                for b in [attn] + mambas
            ]
            assert len(ctx.unified_mamba_state_indices) == 2
            return metas

        for meta_staged, meta_plain in zip(make(staged=True), make(staged=False)):
            assert torch.equal(meta_staged.block_tables, meta_plain.block_tables)
            assert (meta_staged.mamba_state_indices is None) == (
                meta_plain.mamba_state_indices is None
            )
            if meta_plain.mamba_state_indices is not None:
                assert torch.equal(
                    meta_staged.mamba_state_indices, meta_plain.mamba_state_indices
                )
                assert torch.equal(
                    meta_staged.mamba_ckpt_indices, meta_plain.mamba_ckpt_indices
                )
                assert (
                    meta_staged.mamba_state_indices.dtype
                    == meta_plain.mamba_state_indices.dtype
                )
                assert (
                    meta_staged.mamba_ckpt_indices.dtype
                    == meta_plain.mamba_ckpt_indices.dtype
                )

    def test_single_group_fast_path(self):
        """Single-group pure attention must stage directly without mamba derivation."""
        runner = self._make_runner_mock(num_groups=1)
        builder = self._make_builder(runner, kv_cache_group_id=0)
        self._make_attn_groups(runner, [builder])
        ctx = AttentionMetadataBuilderContext(
            num_reqs=2,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.ones((4,), dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
        )
        runner._attn_metadata_builder_ctx = ctx
        stage_block_table_uploads(runner, ctx, 4)
        assert ctx.staged_block_tables is not None
        assert set(ctx.staged_block_tables) == {0}
        assert ctx.staged_mamba_products is None
        assert ctx.staged_block_tables[0].numel() == 4 * 4

    def test_staged_build_skips_per_group_upload(self):
        """With views staged, build() must not touch the CPU block table."""
        runner = self._make_runner_mock(num_groups=1)
        runner._block_table_stage_cpu = None
        builder = self._make_builder(runner, kv_cache_group_id=0)
        self._make_attn_groups(runner, [builder])
        ctx = AttentionMetadataBuilderContext(
            num_reqs=2,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.ones((4,), dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
        )
        runner._attn_metadata_builder_ctx = ctx
        stage_block_table_uploads(runner, ctx, 4)
        source = runner.input_batch.block_table[0].get_cpu_tensor
        assert source.call_count == 1  # staging read it once

        builder.build(common_prefix_len=0, common_attn_metadata=self._make_cm(4))
        assert source.call_count == 1  # build() did not read it again

    def test_staging_respects_gid_subset_and_dummy_override(self):
        runner = self._make_runner_mock(num_groups=2)
        runner._block_table_stage_cpu = None
        builders = [
            self._make_builder(runner, kv_cache_group_id=gid) for gid in range(2)
        ]
        self._make_attn_groups(runner, builders)
        ctx = AttentionMetadataBuilderContext(
            num_reqs=2,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.ones((4,), dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
        )
        runner._attn_metadata_builder_ctx = ctx

        # Subset: only gid 1 is staged (the draft's targeted walk).
        stage_block_table_uploads(runner, ctx, 4, gids={1})
        assert set(ctx.staged_block_tables) == {1}

        # Dummy runs stage the same layout with zero-filled tables (so
        # warmup dispatches the same program sequence serving does) and
        # never read the CPU source tables.
        source_calls_before = runner.input_batch.block_table[
            0
        ].get_cpu_tensor.call_count
        override = torch.zeros((3, 8), dtype=torch.int32)
        ctx_dummy = AttentionMetadataBuilderContext(
            num_reqs=2,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.ones((4,), dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
            position_ids_override=override,
        )
        stage_block_table_uploads(runner, ctx_dummy, 4)
        assert set(ctx_dummy.staged_block_tables) == {0, 1}
        for staged in ctx_dummy.staged_block_tables.values():
            assert torch.equal(staged, torch.zeros_like(staged))
        assert (
            runner.input_batch.block_table[0].get_cpu_tensor.call_count
            == source_calls_before
        )

        # And build() must consume the staged zero table while keeping the
        # override positions.
        runner._attn_metadata_builder_ctx = ctx_dummy
        meta = builders[0].build(
            common_prefix_len=0, common_attn_metadata=self._make_cm(4)
        )
        assert meta.input_positions is override
        assert torch.equal(meta.block_tables, torch.zeros_like(meta.block_tables))

        # Pre-init runs (no attn_groups yet) skip staging entirely.
        runner.attn_groups = []
        ctx_preinit = AttentionMetadataBuilderContext(
            num_reqs=2,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.ones((4,), dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
        )
        stage_block_table_uploads(runner, ctx_preinit, 4)
        assert ctx_preinit.staged_block_tables is None


class TestPrefillCheckpointIds:
    """Tests for per-group internal prefill checkpoint block ID gathering."""

    NUM_REQS = 5
    TARGET_NUM_REQS = 6
    NUM_BLOCKS = 6
    # 16-token blocks. Row r's table is r * 6 + col (+ 100 * gid), so
    # group 0 row 0 column 0 holds the null block.
    SEQ_LENS = [17, 40, 40, 10, 96, 64]
    # Row 5 is padding: its seq_len is stale and only the zero offset masks it.
    OFFSETS = [8, 16, 0, 4, 32, 0]
    # Row 0: column 0 is the null block. Row 1: column 1. Row 2: no offset.
    # Row 3: column -1. Row 4: column 4. Row 5: padding.
    EXPECTED_GID0 = [-1, 7, -1, -1, 28, -1]

    def _runner(self, num_groups):
        return TestAttentionMetadataBuilderPlumbing()._make_runner_mock(
            num_groups=num_groups,
            max_num_blocks_per_req=self.NUM_BLOCKS,
            max_num_reqs=self.TARGET_NUM_REQS,
        )

    def _mamba_builder(self, runner, gid, *, declares=True, num_speculative_blocks=0):
        spec = MambaSpec(
            block_size=16,
            shapes=[(4, 128), (8, 64, 32)],
            dtypes=[torch.bfloat16, torch.float32],
            page_size_padded=16384,
            mamba_cache_mode="align",
            num_speculative_blocks=num_speculative_blocks,
            num_prefill_checkpoint_blocks=1 if declares else 0,
            prefill_checkpoint_alignment=64 if declares else None,
        )
        return AttentionMetadataBuilder(
            kv_cache_spec=spec,
            layer_names=[f"gdn.{gid}"],
            vllm_config=MagicMock(),
            device=runner.device,
            runner=runner,
            kv_cache_group_id=gid,
        )

    def _ctx(self, offsets, **kwargs):
        return AttentionMetadataBuilderContext(
            num_reqs=self.NUM_REQS,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.tensor(self.SEQ_LENS, dtype=torch.int32),
            query_start_loc=torch.arange(self.TARGET_NUM_REQS + 1, dtype=torch.int32),
            request_distribution=torch.tensor([0, 0, 5], dtype=torch.int32),
            mamba_prefill_ckpt_offsets=offsets,
            **kwargs,
        )

    def _offsets(self):
        return torch.tensor(self.OFFSETS, dtype=torch.int32)

    def _build(self, builder):
        return builder.build(
            common_prefix_len=0,
            common_attn_metadata=SimpleNamespace(num_reqs=self.TARGET_NUM_REQS),
        )

    def _walk(self, *, staged, mamba_declares, offsets, ckpt_window=1):
        runner = self._runner(num_groups=1 + len(mamba_declares))
        runner._unified_kv_layout = True
        plumbing = TestAttentionMetadataBuilderPlumbing()
        attn = plumbing._make_builder(runner, kv_cache_group_id=0)
        mambas = [
            self._mamba_builder(runner, gid, declares=declares)
            for gid, declares in enumerate(mamba_declares, start=1)
        ]
        plumbing._make_attn_groups(runner, [attn] + mambas)
        ctx = self._ctx(offsets, mamba_ckpt_window=ckpt_window)
        runner._attn_metadata_builder_ctx = ctx
        if staged:
            stage_block_table_uploads(runner, ctx, self.TARGET_NUM_REQS)
            assert ctx.staged_mamba_products is not None
        return [self._build(b) for b in [attn] + mambas]

    def test_fallback_gathers_the_column_before_the_state_block(self):
        runner = self._runner(num_groups=1)
        runner._unified_kv_layout = True
        builder = self._mamba_builder(runner, 0)
        offsets = self._offsets()
        runner._attn_metadata_builder_ctx = self._ctx(offsets)

        meta = self._build(builder)

        assert meta.mamba_prefill_ckpt_indices.tolist() == self.EXPECTED_GID0
        assert meta.mamba_prefill_ckpt_indices.dtype == torch.int32
        assert meta.mamba_prefill_ckpt_offsets is offsets
        # The state ids are the column after the checkpoint (row 3 has no
        # checkpoint; row 5 is padding and resolves to the null block).
        assert meta.mamba_state_indices.tolist() == [1, 8, 14, 18, 29, 0]

    @pytest.mark.parametrize("ckpt_window", [1, 2])
    def test_staged_walk_matches_the_fallback(self, ckpt_window):
        # Group 1 and 3 declare checkpoint blocks, group 2 does not.
        declares = [True, False, True]
        staged = self._walk(
            staged=True,
            mamba_declares=declares,
            offsets=self._offsets(),
            ckpt_window=ckpt_window,
        )
        plain = self._walk(
            staged=False,
            mamba_declares=declares,
            offsets=self._offsets(),
            ckpt_window=ckpt_window,
        )
        for gid, (meta_staged, meta_plain) in enumerate(zip(staged, plain)):
            if gid in (0, 2):
                # Attention group, and a Mamba group without checkpoint blocks.
                assert meta_staged.mamba_prefill_ckpt_indices is None
                assert meta_plain.mamba_prefill_ckpt_indices is None
                assert meta_staged.mamba_prefill_ckpt_offsets is None
                assert meta_plain.mamba_prefill_ckpt_offsets is None
                continue
            assert torch.equal(
                meta_staged.mamba_prefill_ckpt_indices,
                meta_plain.mamba_prefill_ckpt_indices,
            )
            assert torch.equal(
                meta_staged.mamba_state_indices, meta_plain.mamba_state_indices
            )
            assert meta_staged.mamba_prefill_ckpt_offsets is not None
            if ckpt_window > 1:
                assert torch.equal(
                    meta_staged.mamba_ckpt_indices, meta_plain.mamba_ckpt_indices
                )
        # Group 1's table is group 0's plus 100; group 3's plus 300, so row 0
        # column 0 holds a real block there.
        assert staged[1].mamba_prefill_ckpt_indices.tolist() == [
            100,
            107,
            -1,
            -1,
            128,
            -1,
        ]
        assert staged[3].mamba_prefill_ckpt_indices.tolist() == [
            300,
            307,
            -1,
            -1,
            328,
            -1,
        ]

    @pytest.mark.parametrize("staged", [True, False])
    def test_disabled_checkpoints_leave_both_fields_none(self, staged):
        metas = self._walk(staged=staged, mamba_declares=[True], offsets=None)
        for meta in metas:
            assert meta.mamba_prefill_ckpt_offsets is None
            assert meta.mamba_prefill_ckpt_indices is None

    @pytest.mark.parametrize("staged", [True, False])
    def test_dummy_run_zero_offsets_give_no_checkpoint(self, staged):
        runner = self._runner(num_groups=2)
        runner._unified_kv_layout = True
        plumbing = TestAttentionMetadataBuilderPlumbing()
        attn = plumbing._make_builder(runner, kv_cache_group_id=0)
        mamba = self._mamba_builder(runner, 1)
        plumbing._make_attn_groups(runner, [attn, mamba])
        ctx = self._ctx(
            torch.zeros(self.TARGET_NUM_REQS, dtype=torch.int32),
            position_ids_override=torch.zeros(8, dtype=torch.int32),
        )
        runner._attn_metadata_builder_ctx = ctx
        if staged:
            stage_block_table_uploads(runner, ctx, self.TARGET_NUM_REQS)

        meta = self._build(mamba)

        assert meta.mamba_prefill_ckpt_indices.tolist() == [-1] * 6
        assert meta.mamba_prefill_ckpt_offsets is ctx.mamba_prefill_ckpt_offsets

    def test_a_block_moved_to_the_state_or_a_speculative_column_is_no_checkpoint(
        self,
    ):
        # One speculative block: the tail is the state column and the next.
        table = torch.tensor(
            [
                [0, 5, 7, 9, 0, 0],
                # The checkpoint column's block also sits in the tail: the
                # manager moved it there and left the column null.
                [0, 5, 7, 5, 0, 0],
                [0, 5, 5, 9, 0, 0],
                # The null block.
                [0, 0, 7, 8, 0, 0],
                # The tail runs past the table: only the in-row column counts.
                [3, 4, 5, 6, 7, 4],
                # An equal id outside the tail does not count.
                [1, 2, 3, 4, 1, 0],
            ],
            dtype=torch.int32,
        )
        state_cols = torch.tensor([2, 2, 2, 2, 5, 1], dtype=torch.int64)
        offsets = torch.tensor([64, 64, 64, 64, 64, 64], dtype=torch.int32)
        ids = _gather_prefill_ckpt_ids(table, state_cols, offsets, 1)
        assert ids.tolist() == [5, -1, -1, -1, 7, 1]
        # Without speculative blocks only the state column is compared.
        ids = _gather_prefill_ckpt_ids(table, state_cols, offsets, 0)
        assert ids.tolist() == [5, 5, -1, -1, 7, 1]

    @pytest.mark.parametrize("staged", [True, False])
    def test_builder_compares_the_groups_speculative_columns(self, staged):
        runner = self._runner(num_groups=2)
        runner._unified_kv_layout = True
        table = runner.input_batch.block_table[1].get_cpu_tensor.return_value
        # Row 1 (state column 2, checkpoint column 1): the checkpoint
        # column's block also sits two columns after the state column.
        table[1, 4] = table[1, 1]
        plumbing = TestAttentionMetadataBuilderPlumbing()
        attn = plumbing._make_builder(runner, kv_cache_group_id=0)
        mamba = self._mamba_builder(runner, 1, num_speculative_blocks=2)
        plumbing._make_attn_groups(runner, [attn, mamba])
        ctx = self._ctx(self._offsets())
        runner._attn_metadata_builder_ctx = ctx
        if staged:
            stage_block_table_uploads(runner, ctx, self.TARGET_NUM_REQS)
            assert ctx.staged_mamba_products is not None

        meta = self._build(mamba)

        assert meta.mamba_prefill_ckpt_indices.tolist() == [100, -1, -1, -1, 128, -1]

    def test_metadata_pytree_carries_the_checkpoint_fields(self):
        offsets = jnp.array([0, 3072], dtype=jnp.int32)
        ids = jnp.array([-1, 5], dtype=jnp.int32)
        meta = AttentionMetadata(
            input_positions=jnp.zeros(4, dtype=jnp.int32),
            mamba_prefill_ckpt_offsets=offsets,
            mamba_prefill_ckpt_indices=ids,
        )
        leaves = jax.tree_util.tree_leaves(meta)
        assert any(leaf is offsets for leaf in leaves)
        assert any(leaf is ids for leaf in leaves)

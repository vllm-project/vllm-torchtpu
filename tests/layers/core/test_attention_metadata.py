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

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadataBuilder,
    AttentionMetadataBuilderContext,
    stage_block_table_uploads,
    stage_h2d_upload,
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
        runner._unfenced_h2d_sources = []
        runner._free_h2d_sources = {}
        runner._registered_h2d_sources = set()
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

    @pytest.mark.parametrize("fast", [False, True])
    @pytest.mark.parametrize("reuse", [False, True])
    def test_stacked_walk_matches_per_group_walk(self, fast, reuse):
        """VLLM_TPU_STAGED_WALK_STACKED: every mamba group's state and
        checkpoint ids read from the stack equal the per-group clones bit for
        bit, the walk returns one [G, n] / [G, n, W] stack per class, and the
        read-offset list carries the stack once (same flattened ids)."""
        from vllm_torchtpu.layers.core.attention_metadata import (
            StackedMambaAttentionMetadata,
            mamba_ckpt_of,
            mamba_state_of,
        )
        from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

        window = 3
        num_reqs = 4

        def make(stacked):
            env = {
                "VLLM_TPU_STAGED_WALK_CLASS_GATHER": "1",
                "VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE": "1" if reuse else "0",
            }
            if stacked:
                env["VLLM_TPU_STAGED_WALK_STACKED"] = "1"
            with patch.dict(os.environ, env):
                runner = self._make_runner_mock(num_groups=4, max_num_blocks_per_req=6)
                runner._unified_kv_layout = True
                runner._block_table_stage_cpu = None
                runner._staged_walk_stacked_logged = False
                attn = self._make_builder(runner, kv_cache_group_id=0)
                mambas = [
                    self._make_mamba_builder(runner, kv_cache_group_id=gid)
                    for gid in (1, 2, 3)
                ]
                groups = self._make_attn_groups(runner, [attn] + mambas)
                for i, (group,) in enumerate(groups):
                    group.layer_names = [f"layer.{i}"]
                out = None
                for _ in range(2 if reuse else 1):
                    ctx = AttentionMetadataBuilderContext(
                        num_reqs=num_reqs,
                        start_index=0,
                        use_max_model_len=True,
                        seq_lens=torch.tensor([0, 17, 33, 49], dtype=torch.int32),
                        query_start_loc=torch.arange(5, dtype=torch.int32),
                        request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
                        mamba_slot_read_offsets=torch.zeros(8, dtype=torch.int32),
                        mamba_ckpt_window=window,
                        mamba_request_distribution=torch.tensor([4, 4, 4]),
                        unified_mamba_state_indices=[],
                    )
                    runner._attn_metadata_builder_ctx = ctx
                    stage_block_table_uploads(runner, ctx, num_reqs)
                    if fast:
                        out = TPUModelRunner._build_attention_metadata_staged(
                            runner, ctx
                        )
                        assert out is not None
                        metas = [out[f"layer.{i}"] for i in range(4)]
                    else:
                        metas = [
                            b.build(
                                common_prefix_len=0,
                                common_attn_metadata=self._make_cm(num_reqs),
                            )
                            for b in [attn] + mambas
                        ]
                return metas, ctx

        (plain, c_plain), (stk, c_stk) = make(False), make(True)
        assert not isinstance(plain[1], StackedMambaAttentionMetadata)
        for k in (1, 2, 3):
            a, b = stk[k], plain[k]
            assert isinstance(a, StackedMambaAttentionMetadata)
            assert a.mamba_state_stack.shape == (3, num_reqs)
            assert a.mamba_ckpt_stack.shape == (3, num_reqs, window)
            assert a.mamba_stack_row == k - 1
            assert a.mamba_state_stack is stk[1].mamba_state_stack
            assert a.mamba_state_indices is None and a.mamba_ckpt_indices is None
            for get, f in (
                (mamba_state_of, "mamba_state_indices"),
                (mamba_ckpt_of, "mamba_ckpt_indices"),
            ):
                x, y = get(a), getattr(b, f)
                assert x.dtype == y.dtype and torch.equal(x, y), (k, f)
                assert torch.equal(get(b), y)
            assert torch.equal(a.block_tables, b.block_tables)
        assert stk[0].mamba_state_indices is None
        assert len(c_stk.unified_mamba_state_indices) == 1
        assert torch.equal(
            c_stk.unified_mamba_state_indices[0].reshape(-1),
            torch.cat(c_plain.unified_mamba_state_indices),
        )

    @pytest.mark.parametrize("stacked", [False, True])
    def test_dummy_and_real_builds_trace_the_same_inputs(self, stacked):
        """A program traced on a dummy run's metadata is replayed on real
        metadata (guard-free AOT), so both must expose the same mamba-index
        structure: same metadata class per group, same non-None fields, and
        the same number of dynamo graph inputs for a forward that reads the
        ids through mamba_state_of / mamba_ckpt_of. Flag on and off."""
        from vllm_torchtpu.layers.core.attention_metadata import (
            StackedMambaAttentionMetadata,
            mamba_ckpt_of,
            mamba_state_of,
        )

        window = 3
        num_reqs = 4
        fields = (
            "mamba_state_indices",
            "mamba_ckpt_indices",
            "mamba_state_stack",
            "mamba_ckpt_stack",
        )

        def build(dummy):
            env = {
                "VLLM_TPU_STAGED_WALK_CLASS_GATHER": "1",
                "VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE": "0",
                "VLLM_TPU_MD_STRUCT_LOG": "1",
            }
            if stacked:
                env["VLLM_TPU_STAGED_WALK_STACKED"] = "1"
            with patch.dict(os.environ, env):
                runner = self._make_runner_mock(num_groups=4, max_num_blocks_per_req=6)
                runner._unified_kv_layout = True
                runner._block_table_stage_cpu = None
                runner._staged_walk_stacked_logged = False
                attn = self._make_builder(runner, kv_cache_group_id=0)
                mambas = [
                    self._make_mamba_builder(runner, kv_cache_group_id=gid)
                    for gid in (1, 2, 3)
                ]
                groups = self._make_attn_groups(runner, [attn] + mambas)
                for i, (group,) in enumerate(groups):
                    group.layer_names = [f"layer.{i}"]
                ctx = AttentionMetadataBuilderContext(
                    num_reqs=num_reqs,
                    start_index=0,
                    use_max_model_len=True,
                    seq_lens=torch.tensor([1, 17, 33, 49], dtype=torch.int32),
                    query_start_loc=torch.arange(5, dtype=torch.int32),
                    request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
                    position_ids_override=(
                        torch.zeros(num_reqs, dtype=torch.int32) if dummy else None
                    ),
                    mamba_slot_read_offsets=torch.zeros(8, dtype=torch.int32),
                    mamba_ckpt_window=window,
                    mamba_request_distribution=torch.tensor([4, 4, 4]),
                    unified_mamba_state_indices=[],
                )
                runner._attn_metadata_builder_ctx = ctx
                stage_block_table_uploads(runner, ctx, num_reqs)
                return [
                    b.build(
                        common_prefix_len=0,
                        common_attn_metadata=self._make_cm(num_reqs),
                    )
                    for b in mambas
                ]

        def structure(metas):
            return [
                (type(m).__name__,)
                + tuple(getattr(m, f, None) is not None for f in fields)
                for m in metas
            ]

        def num_graph_inputs(metas):
            counts = []

            def backend(gm, example_inputs):
                counts.append(
                    sum(1 for n in gm.graph.nodes if n.op == "placeholder")
                )
                return gm

            def forward(x, mds):
                for md in mds:
                    x = x + mamba_state_of(md).sum() + mamba_ckpt_of(md).sum()
                return x

            torch._dynamo.reset()
            torch.compile(forward, backend=backend, fullgraph=True)(
                torch.zeros(()), metas
            )
            torch._dynamo.reset()
            return counts[0]

        dummy, real = build(True), build(False)
        assert structure(dummy) == structure(real)
        assert all(
            isinstance(m, StackedMambaAttentionMetadata) == stacked
            for m in dummy + real
        )
        assert num_graph_inputs(dummy) == num_graph_inputs(real)

    def test_stacked_read_offset_update_matches_per_group(self):
        """`_update_read_offsets_impl` on the [G, width] stack writes the
        same slot buffer as on the per-group list."""
        from vllm_torchtpu.runner.tpu_runner import _update_read_offsets_impl

        groups = [torch.tensor([3, 5, 0, 0]), torch.tensor([7, 9, 0, 0])]
        nt = torch.tensor([[1, 2, -1], [4, -1, -1]])
        a = torch.full((16,), -5, dtype=torch.int32)
        b = a.clone()
        _update_read_offsets_impl(a, groups, nt, torch.tensor(2))
        _update_read_offsets_impl(b, [torch.stack(groups)], nt, torch.tensor(2))
        assert torch.equal(a, b)

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

    @pytest.mark.parametrize("num_groups", [1, 3])
    def test_each_staged_walk_owns_its_host_buffer(self, num_groups):
        """TorchTPU reads a non-blocking H2D source until the copy drains, so a
        later chunk, draft or dummy walk must neither refill nor drop an earlier
        walk's buffer before the staging fence releases it."""
        runner = self._make_runner_mock(num_groups=num_groups)
        self._make_attn_groups(
            runner, [self._make_builder(runner, gid) for gid in range(num_groups)]
        )
        snapshots = []
        for start, count, padded, dummy in [
            (0, 2, 4, False),
            (1, 1, 2, False),
            (0, 3, 4, True),
        ]:
            ctx = AttentionMetadataBuilderContext(
                num_reqs=count,
                start_index=start,
                use_max_model_len=True,
                seq_lens=torch.ones(padded, dtype=torch.int32),
                query_start_loc=torch.arange(padded + 1, dtype=torch.int32),
                request_distribution=torch.tensor(
                    [count, count, count], dtype=torch.int32
                ),
                position_ids_override=torch.zeros(1) if dummy else None,
            )
            stage_block_table_uploads(runner, ctx, padded)
            snapshots.append(runner._unfenced_h2d_sources[-1][0].clone())

        sources = [src for src, _ in runner._unfenced_h2d_sources]
        assert len(sources) == 3
        assert len({src.data_ptr() for src in sources}) == 3
        for src, snapshot in zip(sources, snapshots, strict=True):
            torch.testing.assert_close(src, snapshot, rtol=0, atol=0)
        # The first walk's rows survived the smaller batch and the dummy walk.
        expected = torch.zeros((4, 4), dtype=torch.int32)
        expected[:2] = runner.input_batch.block_table[0].get_cpu_tensor()[:2]
        torch.testing.assert_close(
            sources[0][: 4 * 4].view(4, 4), expected, rtol=0, atol=0
        )

    def test_each_unstaged_build_owns_its_host_buffer(self):
        """The unstaged build() fallback must not reuse a shared block table
        buffer across walks either."""
        runner = self._make_runner_mock(num_groups=1)
        builder = self._make_builder(runner, kv_cache_group_id=0)
        snapshots = []
        for start, count, padded in [(0, 2, 4), (1, 1, 2)]:
            runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
                num_reqs=count,
                start_index=start,
                use_max_model_len=True,
                seq_lens=torch.ones(padded, dtype=torch.int32),
                query_start_loc=torch.arange(padded + 1, dtype=torch.int32),
                request_distribution=torch.tensor(
                    [count, count, count], dtype=torch.int32
                ),
            )
            builder.build(
                common_prefix_len=0, common_attn_metadata=self._make_cm(padded)
            )
            snapshots.append(runner._unfenced_h2d_sources[-1][0].clone())

        sources = [src for src, _ in runner._unfenced_h2d_sources]
        assert len(sources) == 2
        assert sources[0].data_ptr() != sources[1].data_ptr()
        for src, snapshot in zip(sources, snapshots, strict=True):
            torch.testing.assert_close(src, snapshot, rtol=0, atol=0)

    def test_stage_h2d_upload_rejects_non_contiguous_source(self):
        runner = self._make_runner_mock(num_groups=1)
        non_contig = torch.zeros((4, 4), dtype=torch.int32)[:, :2]
        assert not non_contig.is_contiguous()
        with pytest.raises(AssertionError):
            stage_h2d_upload(runner, non_contig)
        assert runner._unfenced_h2d_sources == []

    def test_recycled_free_list_buffer_is_reused_and_zeroed(self):
        runner = self._make_runner_mock(num_groups=1)
        builder = self._make_builder(runner, kv_cache_group_id=0)
        self._make_attn_groups(runner, [builder])
        recycled = torch.full((4 * 4,), 99, dtype=torch.int32)
        ptr = recycled.data_ptr()
        runner._free_h2d_sources[(4 * 4, torch.int32)] = [recycled]

        ctx = AttentionMetadataBuilderContext(
            num_reqs=2,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.ones(4, dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
        )
        stage_block_table_uploads(runner, ctx, 4)

        assert runner._free_h2d_sources[(4 * 4, torch.int32)] == []
        used, release_cb = runner._unfenced_h2d_sources[-1]
        assert id(used) in runner._registered_h2d_sources
        assert used.data_ptr() == ptr
        # Padded tail rows (2..3) were zeroed before refilling rows 0..1.
        assert used.view(4, 4)[2:].tolist() == [[0, 0, 0, 0], [0, 0, 0, 0]]
        release_cb()
        assert runner._free_h2d_sources[(4 * 4, torch.int32)] == [used]

    def test_staged_build_skips_per_group_upload(self):
        """With views staged, build() must not touch the CPU block table."""
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
        source = runner.input_batch.block_table[0].get_cpu_tensor
        assert source.call_count == 1  # staging read it once

        builder.build(common_prefix_len=0, common_attn_metadata=self._make_cm(4))
        assert source.call_count == 1  # build() did not read it again

    @patch.dict(os.environ, {"VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE": "1"})
    def test_unchanged_tables_skip_repacking(self):
        """With no BlockTable mutation since the last walk and the same walk
        geometry, staging reuses the previous upload without reading the CPU
        tables; a mutation or a different window re-packs."""
        from vllm_torchtpu.layers.core import attention_metadata as am

        runner = self._make_runner_mock(num_groups=2)
        runner._block_table_stage_cpu = None
        builders = [
            self._make_builder(runner, kv_cache_group_id=gid) for gid in range(2)
        ]
        self._make_attn_groups(runner, builders)

        def ctx(num_reqs=2):
            c = AttentionMetadataBuilderContext(
                num_reqs=num_reqs,
                start_index=0,
                use_max_model_len=True,
                seq_lens=torch.ones((4,), dtype=torch.int32),
                query_start_loc=torch.arange(5, dtype=torch.int32),
                request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
            )
            runner._attn_metadata_builder_ctx = c
            return c

        source = runner.input_batch.block_table[0].get_cpu_tensor
        c1 = ctx()
        stage_block_table_uploads(runner, c1, 4)
        calls = source.call_count
        c2 = ctx()
        stage_block_table_uploads(runner, c2, 4)
        assert source.call_count == calls  # reused, not re-packed
        for gid in (0, 1):
            assert torch.equal(c1.staged_block_tables[gid], c2.staged_block_tables[gid])

        am._BLOCK_TABLE_VERSION[0] += 1  # any BlockTable mutation
        stage_block_table_uploads(runner, ctx(), 4)
        assert source.call_count == calls + 1

        stage_block_table_uploads(runner, ctx(num_reqs=1), 4)  # new window
        assert source.call_count == calls + 2

    @patch.dict(os.environ, {"VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE": "1"})
    def test_content_reuse_returns_the_packed_buffer_to_the_pool(self):
        """A walk whose re-packed tables equal the previous upload reuses that
        upload; the pooled buffer it packed into is never uploaded, so it must
        go back on the free list instead of leaking a registered id."""
        from vllm_torchtpu.layers.core import attention_metadata as am

        runner = self._make_runner_mock(num_groups=1)
        builder = self._make_builder(runner, kv_cache_group_id=0)
        self._make_attn_groups(runner, [builder])

        def walk():
            c = AttentionMetadataBuilderContext(
                num_reqs=2,
                start_index=0,
                use_max_model_len=True,
                seq_lens=torch.ones((4,), dtype=torch.int32),
                query_start_loc=torch.arange(5, dtype=torch.int32),
                request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
            )
            runner._attn_metadata_builder_ctx = c
            stage_block_table_uploads(runner, c, 4)
            return c

        key = (4 * 4, torch.int32)
        first = walk()
        assert len(runner._unfenced_h2d_sources) == 1
        uploaded = runner._unfenced_h2d_sources[0][0]
        cached = runner._block_table_stage_prev[1]

        # A version bump with unchanged contents re-packs, then reuses.
        am._BLOCK_TABLE_VERSION[0] += 1
        second = walk()
        assert torch.equal(second.staged_block_tables[0], first.staged_block_tables[0])
        assert len(runner._unfenced_h2d_sources) == 1  # no second upload
        # The reuse keeps the cached host copy rather than cloning a new one.
        assert runner._block_table_stage_prev[1] is cached
        (packed,) = runner._free_h2d_sources[key]
        assert packed is not uploaded
        assert id(packed) in runner._registered_h2d_sources

        # The next reused walk packs into that same buffer again.
        am._BLOCK_TABLE_VERSION[0] += 1
        walk()
        assert runner._free_h2d_sources[key] == [packed]
        assert runner._registered_h2d_sources == {id(uploaded), id(packed)}

    @patch.dict(os.environ, {"VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE": "1"})
    def test_reused_upload_reuses_tables_and_rederives_mamba_indices(self):
        """A reused upload keeps the previous walk's table outputs but must
        re-derive the mamba indices from the new seq_lens."""
        window = 3
        runner = self._make_runner_mock(num_groups=3, max_num_blocks_per_req=6)
        runner._unified_kv_layout = True
        runner._block_table_stage_cpu = None
        attn = self._make_builder(runner, kv_cache_group_id=0)
        mambas = [
            self._make_mamba_builder(runner, kv_cache_group_id=gid) for gid in (1, 2)
        ]
        self._make_attn_groups(runner, [attn] + mambas)

        def walk(seq_lens, staged):
            ctx = AttentionMetadataBuilderContext(
                num_reqs=4,
                start_index=0,
                use_max_model_len=True,
                seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
                query_start_loc=torch.arange(5, dtype=torch.int32),
                request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
                mamba_ckpt_window=window,
                unified_mamba_state_indices=[],
            )
            runner._attn_metadata_builder_ctx = ctx
            if staged:
                stage_block_table_uploads(runner, ctx, 4)
            metas = [
                b.build(common_prefix_len=0, common_attn_metadata=self._make_cm(4))
                for b in [attn] + mambas
            ]
            return ctx, metas

        ctx1, _ = walk([0, 17, 33, 33], staged=True)
        ctx2, metas2 = walk([0, 18, 49, 33], staged=True)  # 49 crosses a block
        assert ctx2.staged_block_tables is ctx1.staged_block_tables
        _, plain = walk([0, 18, 49, 33], staged=False)
        for a, b in zip(metas2, plain):
            assert torch.equal(a.block_tables, b.block_tables)
            if b.mamba_state_indices is not None:
                assert torch.equal(a.mamba_state_indices, b.mamba_state_indices)
                assert torch.equal(a.mamba_ckpt_indices, b.mamba_ckpt_indices)

    @patch.dict(
        os.environ,
        {
            "VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE": "1",
            "VLLM_TPU_BLOCK_TABLE_PACK_DELTA": "1",
        },
    )
    def test_delta_pack_recopies_only_changed_tables(self):
        """PACK_DELTA: after a mutation only the changed table is re-read, and
        the staged tables still match the CPU tables; a new window re-packs
        everything."""
        from vllm_torchtpu.layers.core import attention_metadata as am

        runner = self._make_runner_mock(num_groups=2)
        runner._block_table_stage_cpu = None
        tables = runner.input_batch.block_table
        for bt in tables:
            bt._tpu_version = 0
            bt.get_numpy_array.return_value = bt.get_cpu_tensor.return_value.numpy()
        builders = [
            self._make_builder(runner, kv_cache_group_id=gid) for gid in range(2)
        ]
        self._make_attn_groups(runner, builders)

        def walk(num_reqs=2):
            c = AttentionMetadataBuilderContext(
                num_reqs=num_reqs,
                start_index=0,
                use_max_model_len=True,
                seq_lens=torch.ones((4,), dtype=torch.int32),
                query_start_loc=torch.arange(5, dtype=torch.int32),
                request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
            )
            runner._attn_metadata_builder_ctx = c
            stage_block_table_uploads(runner, c, 4)
            for gid, bt in enumerate(tables):
                want = torch.zeros((4, 4), dtype=torch.int32)
                want[:num_reqs] = bt.get_cpu_tensor.return_value[:num_reqs]
                assert torch.equal(c.staged_block_tables[gid], want.reshape(-1))
            return c

        walk()
        reads = [bt.get_numpy_array.call_count for bt in tables]
        am._BLOCK_TABLE_VERSION[0] += 1
        tables[1]._tpu_version = am._BLOCK_TABLE_VERSION[0]
        tables[1].get_cpu_tensor.return_value[1, 2] = 999
        walk()
        assert tables[0].get_numpy_array.call_count == reads[0]
        assert tables[1].get_numpy_array.call_count == reads[1] + 1
        walk(num_reqs=1)  # new window: full re-pack
        assert tables[0].get_numpy_array.call_count == reads[0] + 1

    @pytest.mark.parametrize("unified", [False, True])
    def test_fast_staged_build_matches_builder_walk(self, unified):
        """VLLM_TPU_FAST_METADATA_BUILD: the inlined staged walk returns the
        same metadata per layer, and the same unified-pool state list, as one
        build() call per group."""
        from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

        runner = self._make_runner_mock(num_groups=3, max_num_blocks_per_req=6)
        runner._unified_kv_layout = unified
        runner._block_table_stage_cpu = None
        attn = self._make_builder(runner, kv_cache_group_id=0)
        mambas = [
            self._make_mamba_builder(runner, kv_cache_group_id=gid) for gid in (1, 2)
        ]
        groups = self._make_attn_groups(runner, [attn] + mambas)
        for i, (group,) in enumerate(groups):
            group.layer_names = [f"layer.{i}.a", f"layer.{i}.b"]

        def ctx():
            c = AttentionMetadataBuilderContext(
                num_reqs=4,
                start_index=0,
                use_max_model_len=True,
                seq_lens=torch.tensor([0, 17, 33, 49], dtype=torch.int32),
                query_start_loc=torch.arange(5, dtype=torch.int32),
                request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
                mamba_state_indices=None if unified else torch.arange(4),
                mamba_slot_read_offsets=torch.zeros(8, dtype=torch.int32),
                mamba_ckpt_window=3 if unified else 1,
                mamba_request_distribution=torch.tensor([4, 4, 4]),
                unified_mamba_state_indices=[] if unified else None,
            )
            runner._attn_metadata_builder_ctx = c
            stage_block_table_uploads(runner, c, 4)
            return c

        c_ref = ctx()
        ref = {}
        for (group,) in groups:
            built = group.metadata_builders[0].build(
                common_prefix_len=0, common_attn_metadata=self._make_cm(4)
            )
            for name in group.layer_names:
                ref[name] = built
        c_fast = ctx()
        fast = TPUModelRunner._build_attention_metadata_staged(runner, c_fast)
        assert fast is not None and fast.keys() == ref.keys()
        for name in ref:
            a, b = fast[name], ref[name]
            for f in (
                "input_positions",
                "block_tables",
                "seq_lens",
                "query_start_loc",
                "request_distribution",
                "mamba_state_indices",
                "mamba_slot_read_offsets",
                "mamba_ckpt_indices",
                "mamba_request_distribution",
            ):
                x, y = getattr(a, f), getattr(b, f)
                assert (x is None) == (y is None), (name, f)
                if x is not None:
                    assert torch.equal(x, y), (name, f)
            assert a.sequence_layout_kind == b.sequence_layout_kind
        if unified:
            assert len(c_fast.unified_mamba_state_indices) == 2
            for x, y in zip(
                c_fast.unified_mamba_state_indices, c_ref.unified_mamba_state_indices
            ):
                assert torch.equal(x, y)

    def test_block_table_mutations_bump_the_version(self):
        from types import SimpleNamespace

        import numpy as np
        from vllm.v1.worker.block_table import BlockTable

        from vllm_torchtpu.layers.core import attention_metadata as am

        bt = SimpleNamespace(
            num_blocks_per_row=np.array([2, 0], dtype=np.int32),
            block_table=SimpleNamespace(np=np.ones((2, 4), dtype=np.int32)),
            use_hybrid_blocks=False,
        )
        v = am._BLOCK_TABLE_VERSION[0]
        BlockTable.append_row(bt, [], 0)  # no blocks: contents unchanged
        assert am._BLOCK_TABLE_VERSION[0] == v
        BlockTable.append_row(bt, [7], 1)
        assert am._BLOCK_TABLE_VERSION[0] > v
        v = am._BLOCK_TABLE_VERSION[0]
        BlockTable.move_row(bt, 1, 0)
        assert am._BLOCK_TABLE_VERSION[0] > v
        v = am._BLOCK_TABLE_VERSION[0]
        BlockTable.clear_row(bt, 0)
        assert am._BLOCK_TABLE_VERSION[0] > v

    def test_staging_respects_gid_subset_and_dummy_override(self):
        runner = self._make_runner_mock(num_groups=2)
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

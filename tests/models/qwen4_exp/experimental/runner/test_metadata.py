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
"""The model's copy of the staged attention metadata walk (runner/metadata.py).

Ported from the tests of #2606 #2626 #2721 #3217 #2649
(tests/layers/core/test_attention_metadata.py), pointed at the copy.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadataBuilder,
    AttentionMetadataBuilderContext,
)
from vllm_torchtpu.models.qwen4_exp.experimental.runner.metadata import (
    _build,
    _build_attention_metadata_staged,
    _build_group,
    build_attention_metadata,
    stage_block_table_uploads,
)


class TestStagedMetadataCopy:
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
        runner.pcp_manager = None

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

    def _make_mamba_builder(self, runner, kv_cache_group_id=0, block_size=16):
        spec = MambaSpec(
            block_size=block_size,
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

    def test_unchanged_tables_skip_repacking(self):
        """With no BlockTable mutation since the last walk and the same walk
        geometry, staging reuses the previous upload without reading the CPU
        tables; a mutation or a different window re-packs."""
        from vllm_torchtpu.models.qwen4_exp.experimental.runner import metadata as am

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

    def test_reused_upload_reuses_tables_and_rederives_mamba_indices(self):
        """A reused upload keeps the previous walk's table outputs and
        re-derives mamba state and checkpoint indices."""
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
                _build_group(
                    b, common_prefix_len=0, common_attn_metadata=self._make_cm(4)
                )
                for b in [attn] + mambas
            ]
            return ctx, metas

        ctx1, metas1 = walk([0, 17, 33, 33], staged=True)
        _, plain1 = walk([0, 17, 33, 33], staged=False)
        for a, b in zip(metas1, plain1):
            assert torch.equal(a.block_tables, b.block_tables)
            if b.mamba_state_indices is not None:
                assert torch.equal(a.mamba_state_indices, b.mamba_state_indices)
                assert torch.equal(a.mamba_ckpt_indices, b.mamba_ckpt_indices)

        ctx2, metas2 = walk([0, 18, 49, 33], staged=True)  # 49 crosses a block
        assert ctx2.staged_block_tables is ctx1.staged_block_tables
        _, plain2 = walk([0, 18, 49, 33], staged=False)
        for a, b in zip(metas2, plain2):
            assert torch.equal(a.block_tables, b.block_tables)
            if b.mamba_state_indices is not None:
                assert torch.equal(a.mamba_state_indices, b.mamba_state_indices)
                assert torch.equal(a.mamba_ckpt_indices, b.mamba_ckpt_indices)

    def test_delta_pack_recopies_only_changed_tables(self):
        """With upload reuse active, after a mutation only the changed table is
        re-read and the staged tables still match the CPU tables; a new window
        re-packs everything."""
        from vllm_torchtpu.models.qwen4_exp.experimental.runner import metadata as am

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
            assert c.staged_block_tables is not None
            for gid, bt in enumerate(tables):
                want = torch.zeros((4, 4), dtype=torch.int32)
                want[:num_reqs] = bt.get_cpu_tensor.return_value[:num_reqs]
                assert torch.equal(c.staged_block_tables[gid], want.reshape(-1))
            return c

        c0 = walk()
        assert c0.staged_block_tables is not None
        reads = [bt.get_numpy_array.call_count for bt in tables]
        # Global version bumps (e.g. another table outside this walk), while
        # neither participating table's _tpu_version changed: _pack_delta runs
        # and reuses state[2] with zero table reads.
        am._BLOCK_TABLE_VERSION[0] += 1
        c_unchanged = walk()
        assert c_unchanged.staged_block_tables is not None
        assert [bt.get_numpy_array.call_count for bt in tables] == reads
        assert torch.equal(
            c_unchanged.staged_block_tables[0], c0.staged_block_tables[0]
        )

        am._BLOCK_TABLE_VERSION[0] += 1
        tables[1]._tpu_version = am._BLOCK_TABLE_VERSION[0]
        tables[1].get_cpu_tensor.return_value[1, 2] = 999
        walk()
        assert tables[0].get_numpy_array.call_count == reads[0]
        assert tables[1].get_numpy_array.call_count == reads[1] + 1
        walk(num_reqs=1)  # new window: full re-pack
        assert tables[0].get_numpy_array.call_count == reads[0] + 1

    def test_block_table_mutations_bump_the_version(self, monkeypatch):
        from types import SimpleNamespace

        import numpy as np
        from vllm.v1.worker.block_table import BlockTable

        from vllm_torchtpu.models.qwen4_exp.experimental.runner import metadata as am

        for name in (
            "append_row",
            "add_row",
            "clear_row",
            "move_row",
            "swap_row",
            "clear",
        ):
            monkeypatch.setattr(BlockTable, name, getattr(BlockTable, name))
        monkeypatch.setattr(
            BlockTable,
            "_tpu_version_patch",
            getattr(BlockTable, "_tpu_version_patch", False),
            raising=False,
        )
        if hasattr(BlockTable, "_tpu_version_patch"):
            delattr(BlockTable, "_tpu_version_patch")
        am._install_block_table_version()

        bt = SimpleNamespace(
            num_blocks_per_row=np.array([2, 0], dtype=np.int32),
            block_table=SimpleNamespace(np=np.ones((2, 4), dtype=np.int32)),
            use_hybrid_blocks=False,
        )
        v = am._BLOCK_TABLE_VERSION[0]
        BlockTable.append_row(bt, [], 0)  # no blocks: contents unchanged
        assert am._BLOCK_TABLE_VERSION[0] == v
        assert not hasattr(bt, "_tpu_version")
        BlockTable.append_row(bt, [7], 1)
        assert am._BLOCK_TABLE_VERSION[0] > v
        assert bt._tpu_version == am._BLOCK_TABLE_VERSION[0]
        v = am._BLOCK_TABLE_VERSION[0]
        BlockTable.move_row(bt, 1, 0)
        assert am._BLOCK_TABLE_VERSION[0] > v
        assert bt._tpu_version == am._BLOCK_TABLE_VERSION[0]
        v = am._BLOCK_TABLE_VERSION[0]
        BlockTable.clear_row(bt, 0)
        assert am._BLOCK_TABLE_VERSION[0] > v
        assert bt._tpu_version == am._BLOCK_TABLE_VERSION[0]

    def test_staged_walk_class_gather_matches_per_group(self):
        """Class-batched mamba gathers produce bit-identical results across multiple
        distinct (num_blocks, block_size) geometries compared to eager execution."""
        window = 3
        runner = self._make_runner_mock(num_groups=4, max_num_blocks_per_req=6)
        # Group 3 uses a distinct (num_blocks, block_size) class from groups 1 & 2.
        runner.input_batch.block_table[3].max_num_blocks_per_req = 4
        runner.input_batch.block_table[3].get_cpu_tensor.return_value = (
            torch.arange(runner.max_num_reqs * 4, dtype=torch.int32).reshape(
                runner.max_num_reqs, 4
            )
            + 300
        )
        runner._unified_kv_layout = True
        runner._block_table_stage_cpu = None
        attn = self._make_builder(runner, kv_cache_group_id=0)
        mambas = [
            self._make_mamba_builder(runner, kv_cache_group_id=1, block_size=16),
            self._make_mamba_builder(runner, kv_cache_group_id=2, block_size=16),
            self._make_mamba_builder(runner, kv_cache_group_id=3, block_size=32),
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
                _build_group(
                    b, common_prefix_len=0, common_attn_metadata=self._make_cm(4)
                )
                for b in [attn] + mambas
            ]
            return ctx, metas

        _, metas_staged = walk([0, 17, 33, 65], staged=True)
        _, metas_plain = walk([0, 17, 33, 65], staged=False)
        for a, b in zip(metas_staged, metas_plain):
            assert torch.equal(a.block_tables, b.block_tables)
            if b.mamba_state_indices is not None:
                assert torch.equal(a.mamba_state_indices, b.mamba_state_indices)
                assert torch.equal(a.mamba_ckpt_indices, b.mamba_ckpt_indices)

    def test_stage_entries_cache_reuses_walk_and_invalidates_on_key_change(self):
        runner = self._make_runner_mock(num_groups=2, most_model_len=64)
        builders = [
            self._make_mamba_builder(runner, kv_cache_group_id=gid) for gid in range(2)
        ]
        self._make_attn_groups(runner, builders)

        walk_counts = [0, 0]
        for gid, builder in enumerate(builders):
            orig = builder._block_table_geometry

            def wrapped(c, _orig=orig, _gid=gid):
                walk_counts[_gid] += 1
                return _orig(c)

            builder._block_table_geometry = wrapped

        def make_ctx(*, use_max_model_len=True, mamba_ckpt_window=1):
            return AttentionMetadataBuilderContext(
                num_reqs=2,
                start_index=0,
                use_max_model_len=use_max_model_len,
                seq_lens=torch.ones((4,), dtype=torch.int32),
                query_start_loc=torch.arange(5, dtype=torch.int32),
                request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
                mamba_ckpt_window=mamba_ckpt_window,
            )

        stage_block_table_uploads(runner, make_ctx(), 4)
        assert walk_counts == [1, 1]

        # Identical walk key reuses cached entries without re-walking.
        stage_block_table_uploads(runner, make_ctx(), 4)
        assert walk_counts == [1, 1]

        # Changing gids, use_max_model_len, or mamba_ckpt_window invalidates the cache.
        stage_block_table_uploads(runner, make_ctx(), 4, gids={1})
        assert walk_counts == [1, 2]

        stage_block_table_uploads(runner, make_ctx(use_max_model_len=False), 4)
        assert walk_counts == [2, 3]

        stage_block_table_uploads(runner, make_ctx(mamba_ckpt_window=3), 4)
        assert walk_counts == [3, 4]

    @pytest.mark.parametrize("unified", [False, True])
    def test_fast_staged_build_matches_builder_walk(self, unified):
        """The inlined staged walk returns the same metadata per layer, and the
        same unified-pool state list, as one build() call per group, while
        returning None when a group lacks staged products."""
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
            built = _build_group(
                group.metadata_builders[0],
                common_prefix_len=0,
                common_attn_metadata=self._make_cm(4),
            )
            for name in group.layer_names:
                ref[name] = built
        c_fast = ctx()
        fast = _build_attention_metadata_staged(runner, c_fast)
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

        # Negative control: missing staged group table falls back cleanly.
        c_partial = ctx()
        assert c_partial.staged_block_tables is not None
        c_partial.staged_block_tables.pop(1)
        assert _build_attention_metadata_staged(runner, c_partial) is None
        if unified:
            c_no_prod = ctx()
            c_no_prod.staged_mamba_products = None
            assert _build_attention_metadata_staged(runner, c_no_prod) is None
            assert c_no_prod.unified_mamba_state_indices == []


def test_build_dispatch_keeps_a_custom_builder_build():
    """A builder subclass keeps its own build(); a plain builder takes the
    copy's."""

    class Custom(AttentionMetadataBuilder):
        def __init__(self):
            pass

        def build(self, **kwargs):
            return ("custom", kwargs["common_prefix_len"])

    assert _build(Custom(), common_prefix_len=0, common_attn_metadata=None) == (
        "custom",
        0,
    )


def test_build_attention_metadata_stages_then_takes_the_inline_walk(monkeypatch):
    """An unstaged ctx is staged once, then built by the inline walk; a
    position override (dummy run) takes the per-group builds."""
    from vllm_torchtpu.models.qwen4_exp.experimental.runner import metadata

    calls = []
    runner = SimpleNamespace(attn_groups=[])

    def stage(r, c, n):
        calls.append(("stage", n))
        c.staged_block_tables = {}

    monkeypatch.setattr(metadata, "stage_block_table_uploads", stage)
    monkeypatch.setattr(
        metadata,
        "_build_attention_metadata_staged",
        lambda r, c: calls.append("inline") or {"layer": "fast"},
    )
    ctx = SimpleNamespace(staged_block_tables=None, position_ids_override=None)
    assert build_attention_metadata(runner, ctx, 8) == {"layer": "fast"}
    assert calls == [("stage", 8), "inline"]

    calls.clear()
    ctx = SimpleNamespace(staged_block_tables={}, position_ids_override=object())
    assert build_attention_metadata(runner, ctx, 8) == {}
    assert calls == []

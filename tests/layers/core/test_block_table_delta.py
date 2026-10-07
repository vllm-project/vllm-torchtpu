"""incremental (delta) staged block-table uploads.

Every walk staged incrementally must hand each group exactly the table a full
upload would have produced, across decode appends, new and resumed requests,
row moves/swaps/clears, batch shrink/growth, active-column bucket changes,
chunk offsets, drafter subsets and dummy runs.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.worker.block_table import BlockTable

from vllm_torchtpu.layers.core import block_table_delta as btd
from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadataBuilder,
    AttentionMetadataBuilderContext,
    stage_block_table_uploads,
)

MAX_REQS = 8
MAX_BLOCKS = 64


class _Builder:
    """The slice of AttentionMetadataBuilder that staging reads."""

    _active_block_table_columns = AttentionMetadataBuilder._active_block_table_columns

    def __init__(self, gid: int, bt: BlockTable, block_size: int):
        self.kv_cache_group_id = gid
        self.bt = bt
        self.target_block_size = block_size
        self.is_mamba_group = False

    def _block_table_geometry(self, ctx):
        return self.bt, self.bt.max_num_blocks_per_req


def _make_block_table(block_size: int) -> BlockTable:
    return BlockTable(
        block_size=block_size,
        max_num_reqs=MAX_REQS,
        max_num_blocks_per_req=MAX_BLOCKS,
        max_num_batched_tokens=64,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=block_size,
        cp_kv_cache_interleave_size=1,
    )


def _make_runner(builders):
    groups = [[SimpleNamespace(metadata_builders=[b])] for b in builders]
    return SimpleNamespace(
        attn_groups=groups,
        _block_table_stage_cpu=None,
        device=torch.device("cpu"),
        _unified_kv_layout=False,
    )


def _ctx(num_reqs, seq_lens, start_index=0, override=None):
    lens = torch.tensor(seq_lens, dtype=torch.int32)
    return AttentionMetadataBuilderContext(
        num_reqs=num_reqs,
        start_index=start_index,
        use_max_model_len=True,
        seq_lens=lens,
        query_start_loc=torch.arange(len(seq_lens) + 1, dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, num_reqs], dtype=torch.int32),
        position_ids_override=override,
        seq_lens_cpu=lens,
    )


def _reference(builders, num_reqs, seq_lens, padded, start_index, gids=None):
    """Full-upload tables, computed independently of the staging code."""
    ctx = _ctx(num_reqs, seq_lens, start_index)
    out = {}
    for b in builders:
        if gids is not None and b.kv_cache_group_id not in gids:
            continue
        tnb = b.bt.max_num_blocks_per_req
        cols = b._active_block_table_columns(ctx, tnb)
        tbl = torch.zeros((padded, tnb), dtype=torch.int32)
        src = b.bt.get_cpu_tensor()
        tbl[:num_reqs, :cols] = src[start_index : start_index + num_reqs, :cols]
        out[b.kv_cache_group_id] = tbl.reshape(-1)
    return out


def _stage(runner, num_reqs, seq_lens, padded, start_index=0, gids=None):
    ctx = _ctx(num_reqs, seq_lens, start_index)
    stage_block_table_uploads(runner, ctx, padded, gids=gids)
    return ctx.staged_block_tables


@pytest.fixture
def builders():
    return [
        _Builder(gid, _make_block_table(bs), bs) for gid, bs in enumerate((4, 8, 16))
    ]


def test_randomized_walks_match_full_upload(builders, monkeypatch):
    monkeypatch.setenv("TPU_INCREMENTAL_BLOCK_TABLE_UPLOAD", "1")
    monkeypatch.setenv("TPU_VERIFY_INCREMENTAL_BLOCK_TABLE", "0")
    runner = _make_runner(builders)
    rng = np.random.default_rng(0)
    next_block = [1000, 2000, 3000]
    lens = [0] * MAX_REQS
    num_reqs = 4

    def grow(row, tokens):
        old = lens[row]
        lens[row] = old + tokens
        for g, b in enumerate(builders):
            bs = b.target_block_size
            new = -(-lens[row] // bs) - -(-old // bs)
            ids = list(range(next_block[g], next_block[g] + new))
            next_block[g] += new
            b.bt.append_row(ids, row)

    for row in range(num_reqs):
        grow(row, int(rng.integers(1, 120)))

    for step in range(400):
        op = rng.random()
        if op < 0.70:  # decode: every live request gains a token
            for row in range(num_reqs):
                grow(row, 1)
        elif op < 0.78 and num_reqs < MAX_REQS:  # new request
            row = num_reqs
            num_reqs += 1
            for b in builders:
                b.bt.add_row([], row)
            lens[row] = 0
            grow(row, int(rng.integers(1, 200)))
        elif op < 0.84 and num_reqs > 1:  # finish last request
            num_reqs -= 1
            for b in builders:
                b.bt.clear_row(num_reqs)
            lens[num_reqs] = 0
        elif op < 0.89 and num_reqs > 1:  # condense: move last into a hole
            src, dst = num_reqs - 1, int(rng.integers(0, num_reqs - 1))
            for b in builders:
                b.bt.move_row(src, dst)
            lens[dst], lens[src] = lens[src], 0
            num_reqs -= 1
        elif op < 0.94 and num_reqs > 1:  # reorder
            a, c = rng.choice(num_reqs, 2, replace=False)
            for b in builders:
                b.bt.swap_row(int(a), int(c))
            lens[a], lens[c] = lens[c], lens[a]
        elif num_reqs > 0:  # preempt + resume in place with fresh blocks
            row = int(rng.integers(0, num_reqs))
            for g, b in enumerate(builders):
                bs = b.target_block_size
                n = -(-lens[row] // bs)
                b.bt.add_row(list(range(next_block[g], next_block[g] + n)), row)
                next_block[g] += n

        padded = 8 if num_reqs > 4 else 4
        start = 0
        gids = None
        if step % 7 == 3 and num_reqs > 2:  # second chunk of a split walk
            start = 1
        if step % 11 == 5:  # drafter's targeted walk
            gids = {0, 2}
        n = num_reqs - start
        seq = lens[start : start + n] + [0] * (padded - n)
        got = _stage(runner, n, seq, padded, start, gids)
        want = _reference(builders, n, seq, padded, start, gids)
        assert set(got) == set(want)
        for gid in want:
            assert torch.equal(got[gid], want[gid]), (step, gid)

    stager = runner._block_table_delta_stager
    assert stager.num_full > 0
    assert stager.num_delta > 0
    assert stager.num_reuse > 0
    assert stager.num_delta + stager.num_reuse > 3 * stager.num_full


def test_verify_mode_and_untracked_write_detection(builders, monkeypatch):
    monkeypatch.setenv("TPU_INCREMENTAL_BLOCK_TABLE_UPLOAD", "1")
    monkeypatch.setenv("TPU_VERIFY_INCREMENTAL_BLOCK_TABLE", "1")
    runner = _make_runner(builders)
    for b in builders:
        b.bt.append_row([1, 2, 3], 0)
        b.bt.append_row([4, 5], 1)
    seq = [12, 8, 0, 0]
    _stage(runner, 2, seq, 4)  # full
    for b in builders:
        b.bt.append_row([6], 1)
    _stage(runner, 2, seq, 4)  # delta, verified
    _stage(runner, 2, seq, 4)  # reuse, verified
    # A write that bypasses the tracked mutators must be caught.
    builders[1].bt.get_cpu_tensor()[0, 0] = 77
    with pytest.raises(RuntimeError, match="mirror diverged"):
        _stage(runner, 2, seq, 4)


def test_device_copy_divergence_is_detected(builders, monkeypatch):
    monkeypatch.setenv("TPU_INCREMENTAL_BLOCK_TABLE_UPLOAD", "1")
    monkeypatch.setenv("TPU_VERIFY_INCREMENTAL_BLOCK_TABLE", "1")
    runner = _make_runner(builders)
    for b in builders:
        b.bt.append_row([1, 2], 0)
    seq = [8, 0]
    _stage(runner, 1, seq, 2)
    layout = next(iter(runner._block_table_delta_stager._layouts.values()))
    layout.dev = layout.dev.clone()
    layout.dev[0] = 99
    with pytest.raises(RuntimeError, match="device copy diverged"):
        _stage(runner, 1, seq, 2)


def test_capacity_overflow_falls_back_to_full(builders, monkeypatch):
    monkeypatch.setenv("TPU_INCREMENTAL_BLOCK_TABLE_UPLOAD", "1")
    monkeypatch.setenv("TPU_VERIFY_INCREMENTAL_BLOCK_TABLE", "0")
    monkeypatch.setattr(btd, "DELTA_CAPACITY", 4)
    runner = _make_runner(builders)
    seq = [64, 0]
    _stage(runner, 1, seq, 2)
    for b in builders:
        b.bt.append_row(list(range(10, 13)), 0)  # 9 changed entries > 4
    got = _stage(runner, 1, seq, 2)
    want = _reference(builders, 1, seq, 2, 0)
    for gid in want:
        assert torch.equal(got[gid], want[gid])
    stager = runner._block_table_delta_stager
    assert stager.num_full == 2 and stager.num_delta == 0
    for b in builders:
        b.bt.append_row([20], 0)  # 3 entries <= 4: delta
    got = _stage(runner, 1, seq, 2)
    want = _reference(builders, 1, seq, 2, 0)
    for gid in want:
        assert torch.equal(got[gid], want[gid])
    assert stager.num_delta == 1


def test_disabled_dummy_and_single_group_paths(builders, monkeypatch):
    monkeypatch.setenv("TPU_INCREMENTAL_BLOCK_TABLE_UPLOAD", "0")
    runner = _make_runner(builders)
    for b in builders:
        b.bt.append_row([1], 0)
    _stage(runner, 1, [4, 0], 2)
    assert not isinstance(
        getattr(runner, "_block_table_delta_stager", None), btd.BlockTableDeltaStager
    )

    monkeypatch.setenv("TPU_INCREMENTAL_BLOCK_TABLE_UPLOAD", "1")
    # Dummy run: zero tables, warms the scatter, records nothing.
    ctx = _ctx(1, [4, 0], override=torch.zeros((1,), dtype=torch.int32))
    stage_block_table_uploads(runner, ctx, 2)
    stager = runner._block_table_delta_stager
    assert not stager._layouts
    assert all(int(t.abs().sum()) == 0 for t in ctx.staged_block_tables.values())

    # One group without mamba never caches (its view goes straight to build()).
    single = _make_runner(builders[:1])
    _stage(single, 1, [4, 0], 2)
    _stage(single, 1, [4, 0], 2)
    assert not isinstance(
        getattr(single, "_block_table_delta_stager", None), btd.BlockTableDeltaStager
    )


def test_layout_lru_eviction_and_tracking_idempotent(builders, monkeypatch):
    monkeypatch.setenv("TPU_INCREMENTAL_BLOCK_TABLE_UPLOAD", "1")
    monkeypatch.setenv("TPU_VERIFY_INCREMENTAL_BLOCK_TABLE", "0")
    runner = _make_runner(builders)
    for b in builders:
        b.bt.append_row([1, 2], 0)
    for padded in range(2, 2 + btd.MAX_LAYOUTS + 1):
        _stage(runner, 1, [8] + [0] * (padded - 1), padded)
    stager = runner._block_table_delta_stager
    assert len(stager._layouts) == btd.MAX_LAYOUTS
    token = builders[0].bt._bt_delta_token
    assert btd.ensure_tracked(builders[0].bt) == token
    assert type(builders[0].bt).__name__ == "VersionedBlockTable"
    assert btd._versioned_subclass(BlockTable) is type(builders[0].bt)
    builders[0].bt.clear()
    assert (builders[0].bt._row_version > 0).all()


def test_apply_block_table_delta_cpu_and_tpu_dispatch(monkeypatch):
    staged = torch.arange(6, dtype=torch.int32)
    update = torch.tensor([[1, 4, 1], [10, 40, 10]], dtype=torch.int32)
    out = btd.apply_block_table_delta(staged, update)
    assert out.tolist() == [0, 10, 2, 3, 40, 5]
    assert staged.tolist() == list(range(6))  # functional

    calls = []

    def fake_compiled(s, u):
        calls.append((s, u))
        return s

    monkeypatch.setattr(btd, "_apply_block_table_delta_compiled", fake_compiled)
    tpu_like = SimpleNamespace(device=SimpleNamespace(type="tpu"))
    assert btd.apply_block_table_delta(tpu_like, update) is tpu_like
    assert len(calls) == 1


def test_warm_reuses_zero_update_and_device_copy_not_cloned(builders):
    stager = btd.BlockTableDeltaStager()
    staged = torch.zeros((8,), dtype=torch.int32)
    stager.warm(staged)
    zero = stager._zero_update[staged.device]
    stager.warm(staged)
    assert stager._zero_update[staged.device] is zero

    # A non-CPU device copy is already distinct from the host scratch and is
    # kept as is (no defensive clone).
    for b in builders:
        b.bt.append_row([1], 0)
    entries = [
        (b.kv_cache_group_id, b.bt, b.bt.max_num_blocks_per_req) for b in builders
    ]
    key = stager.layout_key(entries, 1, 0)
    flat = torch.zeros((sum(t for _, _, t in entries),), dtype=torch.int32)
    dev = torch.empty_like(flat, device="meta")
    stager.record_full(key, _ctx(1, [4]), entries, builders, 1, flat, dev)
    assert stager._layouts[key].dev is dev

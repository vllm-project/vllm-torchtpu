# SPDX-License-Identifier: Apache-2.0
"""q38 BT-INC: incremental block-table staging (VLLM_TPU_BLOCK_TABLE_STAGE_INC).

The gate: over random sequences of BlockTable mutations (append / add / clear /
move / swap rows, clear(), batch-size changes, dummy runs) the staged bytes of
the incremental pack equal a full pack's bytes at every step, and the
incremental path is actually taken.
"""

import random
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker.block_table import BlockTable

from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.layers.core import attention_metadata as am

MAX_REQS = 8
WIDTHS = [12, 4, 12, 7]


def _tables():
    return [
        BlockTable(16, MAX_REQS, w, 64, False, torch.device("cpu"), 16, 1)
        for w in WIDTHS
    ]


def _reference(bts, ctx, nreq_padded):
    parts = []
    for bt, w in zip(bts, WIDTHS):
        rows = torch.zeros((nreq_padded, w), dtype=torch.int32)
        rows[: ctx.num_reqs] = bt.get_cpu_tensor()[
            ctx.start_index : ctx.start_index + ctx.num_reqs, :w
        ]
        parts.append(rows.flatten())
    return torch.cat(parts)


def _pack(runner, bts, ctx, nreq_padded):
    entries = [(i, bt, w) for i, (bt, w) in enumerate(zip(bts, WIDTHS))]
    geom, off = [], 0
    for _, _, w in entries:
        geom.append((off, nreq_padded, w))
        off += nreq_padded * w
    key = (am._BLOCK_TABLE_VERSION[0], ctx.start_index, ctx.num_reqs, tuple(geom))
    return am._pack_and_upload(runner, ctx, entries, nreq_padded, geom, key).clone()


@pytest.fixture
def inc(monkeypatch):
    # On CPU an "upload" can alias the host scratch; return a copy, as a device
    # transfer would, so a stale reused upload is visible.
    monkeypatch.setattr(am, "_inc_upload", lambda src, device: src.clone())
    monkeypatch.setattr(am, "_STAGE_INC", True)
    monkeypatch.setattr(am, "_STAGE_INC_CHECK", [0])
    monkeypatch.setattr(
        am, "_STAGE_INC_STATS", {"inc": 0, "full": 0, "rows": 0, "logged": False}
    )


def _random_op(rng, bts, num_reqs):
    bt = rng.choice(bts)
    w = bt.max_num_blocks_per_req
    op = rng.random()
    row = rng.randrange(MAX_REQS)
    if op < 0.45:
        room = w - int(bt.num_blocks_per_row[row])
        if room > 0:
            bt.append_row([rng.randrange(1, 9999) for _ in range(rng.randint(1, room))], row)
    elif op < 0.6:
        bt.add_row([rng.randrange(1, 9999) for _ in range(rng.randint(0, w))], row)
    elif op < 0.7:
        bt.clear_row(row)
    elif op < 0.8:
        bt.move_row(row, rng.randrange(MAX_REQS))
    elif op < 0.9:
        bt.swap_row(row, rng.randrange(MAX_REQS))
    elif op < 0.92:
        bt.clear()
    # else: no mutation this step (the pack must reuse the previous upload)


@pytest.mark.parametrize("reuse,start", [("1", 0), ("0", 0), ("1", 2)])
def test_incremental_pack_equals_full_pack_over_random_mutations(inc, monkeypatch, reuse, start):
    monkeypatch.setenv("VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE", reuse)
    rng = random.Random(1234 + start)
    bts = _tables()
    runner = SimpleNamespace(_block_table_stage_cpu=None, device=torch.device("cpu"))
    num_reqs = 5 - start
    for step in range(600):
        for _ in range(rng.randint(0, 3)):
            _random_op(rng, bts, num_reqs)
        if rng.random() < 0.05:
            num_reqs = rng.randint(1, MAX_REQS - start)  # geometry change: full pack
        ctx = SimpleNamespace(position_ids_override=None, start_index=start, num_reqs=num_reqs)
        got = _pack(runner, bts, ctx, MAX_REQS)
        assert torch.equal(got, _reference(bts, ctx, MAX_REQS)), f"step {step}"
    st = am._STAGE_INC_STATS
    # The incremental path carried most packs, copying few rows each.
    assert st["inc"] > 300 and st["full"] >= 1
    assert st["rows"] < st["inc"] * 3


def test_dummy_run_resets_and_next_pack_is_full(inc):
    bts = _tables()
    runner = SimpleNamespace(_block_table_stage_cpu=None, device=torch.device("cpu"))
    bts[0].add_row([5, 6], 1)
    ctx = SimpleNamespace(position_ids_override=None, start_index=0, num_reqs=3)
    _pack(runner, bts, ctx, MAX_REQS)
    dummy = SimpleNamespace(position_ids_override=torch.zeros(1), start_index=0, num_reqs=3)
    assert not _pack(runner, bts, dummy, MAX_REQS).any()  # dummy runs stage zeros
    assert runner._bt_inc_state[0] is None
    full_before = am._STAGE_INC_STATS["full"]
    bts[2].append_row([9], 0)
    got = _pack(runner, bts, ctx, MAX_REQS)
    assert am._STAGE_INC_STATS["full"] == full_before + 1
    assert torch.equal(got, _reference(bts, ctx, MAX_REQS))


def test_check_mode_catches_a_write_that_bypasses_blocktable(inc, monkeypatch):
    monkeypatch.setattr(am, "_STAGE_INC_CHECK", [5])
    bts = _tables()
    runner = SimpleNamespace(_block_table_stage_cpu=None, device=torch.device("cpu"))
    ctx = SimpleNamespace(position_ids_override=None, start_index=0, num_reqs=4)
    bts[0].add_row([1, 2, 3], 0)
    _pack(runner, bts, ctx, MAX_REQS)  # full
    bts[1].append_row([7], 2)
    _pack(runner, bts, ctx, MAX_REQS)  # incremental, checked, clean
    bts[2].get_cpu_tensor()[3, 0] = 4242  # not through BlockTable: no dirty mark
    bts[1].append_row([8], 2)
    with pytest.raises(RuntimeError, match="BT-INC"):
        _pack(runner, bts, ctx, MAX_REQS)


def test_off_records_nothing(monkeypatch):
    monkeypatch.setattr(am, "_STAGE_INC", False)
    bts = _tables()
    runner = SimpleNamespace(_block_table_stage_cpu=None, device=torch.device("cpu"))
    bts[0].add_row([1, 2], 0)
    bts[0].move_row(0, 3)
    bts[1].clear()
    assert not bts[0].__dict__.get("_q38_dirty") and "_q38_dirty_all" not in bts[1].__dict__
    ctx = SimpleNamespace(position_ids_override=None, start_index=0, num_reqs=4)
    assert torch.equal(_pack(runner, bts, ctx, MAX_REQS), _reference(bts, ctx, MAX_REQS))
    assert "_bt_inc_state" not in runner.__dict__


def test_flags_are_compile_neutral():
    assert "VLLM_TPU_BLOCK_TABLE_STAGE_INC" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    assert "VLLM_TPU_BLOCK_TABLE_STAGE_INC_CHECK" in tpu_compiler._TPU_COMPILE_ENV_IGNORED

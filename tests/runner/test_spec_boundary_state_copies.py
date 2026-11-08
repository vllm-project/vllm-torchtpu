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
"""Unit tests for the align-mode boundary-state copies of verify steps.

A verify window starting at `start` writes checkpoint t (the state after
`start + t + 1` tokens) to the block t columns past its state block, so the
block a boundary completes is last written with an earlier window's
checkpoint 0. ``TPUModelRunner._plan_spec_boundary_copies`` and
``_copy_spec_boundary_states`` copy the exact boundary checkpoint into that
block after sampling. These tests drive both against a CPU stand-in for the
GDN checkpoint writes in which every state block holds the token count its
state covers, so a boundary block must end up holding the boundary itself.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_torchtpu.runner.tpu_runner as runner_mod
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

BLOCK = 16
NUM_SPEC = 3
NUM_COLS = 16
DIM = 4
UNWRITTEN = -1.0


def _copy_stand_in(pools, src, dst):
    for pool in pools:
        pool[dst.long()] = pool[src.long()].clone()


@pytest.fixture(autouse=True)
def _cpu_copies(monkeypatch):
    monkeypatch.setattr(runner_mod, "copy_mamba_state_blocks", _copy_stand_in)


def _make_runner(
    *,
    num_reqs=1,
    num_groups=1,
    split=1,
    window=NUM_SPEC + 1,
    async_scheduling=False,
    prefix_caching=True,
    kv_transfer_config=None,
    max_num_reqs=8,
):
    """A stand-in runner: row r of group g owns block `1 + (g * num_reqs + r)
    * NUM_COLS + col` at every column, each `split` pool rows of one pool."""
    num_blocks = 1 + num_groups * num_reqs * NUM_COLS
    tables, raws = [SimpleNamespace(get_cpu_tensor=lambda: None)], []
    for g in range(num_groups):
        cpu = torch.zeros((num_reqs, NUM_COLS), dtype=torch.int32)
        for r in range(num_reqs):
            first = 1 + (g * num_reqs + r) * NUM_COLS
            cpu[r] = torch.arange(first, first + NUM_COLS, dtype=torch.int32)
        tables.append(SimpleNamespace(get_cpu_tensor=lambda cpu=cpu: cpu))
        raws.append(torch.full((num_blocks * split, DIM), UNWRITTEN))
    req_ids = [f"r{r}" for r in range(num_reqs)]
    ns = SimpleNamespace(
        _mamba_ckpt_window=window,
        mamba_slot_read_offsets=torch.zeros(num_blocks, dtype=torch.int32),
        _mamba_copy_plan=[(g + 1, [raws[g]]) for g in range(num_groups)],
        _mamba_state_block_size=BLOCK,
        _pool_block_split=split,
        bm_pool_layout=None,
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
        vllm_config=SimpleNamespace(kv_transfer_config=kv_transfer_config),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        speculative_config=SimpleNamespace(num_speculative_tokens=NUM_SPEC),
        max_num_reqs=max_num_reqs,
        device="cpu",
        _unfenced_h2d_sources=[],
        _registered_h2d_sources=set(),
        input_batch=SimpleNamespace(
            req_ids=req_ids,
            num_computed_tokens_cpu=np.zeros(num_reqs, dtype=np.int32),
            block_table=tables,
        ),
    )
    ns._bucket_len = TPUModelRunner._bucket_len
    for name in (
        "_pad_dev_to_bucket",
        "_expand_pool_split",
        "_copy_mamba_state_blocks",
        "_copies_spec_boundary_states",
        "_mamba_column_stride",
        "_plan_spec_boundary_copies",
        "_copy_spec_boundary_states",
    ):
        setattr(ns, name, getattr(TPUModelRunner, name).__get__(ns))
    return ns, raws


def _block_value(runner, raw, row, col, group=0):
    block = int(runner.input_batch.block_table[group + 1].get_cpu_tensor()[row, col])
    split = runner._pool_block_split
    values = raw[block * split : (block + 1) * split]
    assert torch.all(values == values.flatten()[0]), values
    return float(values.flatten()[0])


def _verify_step(runner, raws, starts, drafts, accepted, host_ahead):
    """One verify step over every row: stage, write the window checkpoints,
    sample `accepted[r]` tokens (accepted drafts + 1), copy."""
    num_reqs = len(starts)
    runner.input_batch.num_computed_tokens_cpu[:] = [
        s + a for s, a in zip(starts, host_ahead)
    ]
    sched = SimpleNamespace(
        scheduled_spec_decode_tokens={
            f"r{r}": [0] * drafts[r] for r in range(num_reqs) if drafts[r]
        },
        num_scheduled_tokens={f"r{r}": drafts[r] + 1 for r in range(num_reqs)},
    )
    seq_lens = torch.tensor(
        [s + d + 1 for s, d in zip(starts, drafts)] + [0] * 3, dtype=torch.int32
    )
    plan = runner._plan_spec_boundary_copies(sched, 0, num_reqs, seq_lens)
    for g, (_, (raw,)) in enumerate(runner._mamba_copy_plan):
        bt = runner.input_batch.block_table[g + 1].get_cpu_tensor()
        split = runner._pool_block_split
        for r, (start, d) in enumerate(zip(starts, drafts)):
            state_col = (start + d) // BLOCK
            for t in range(d + 1):
                block = int(bt[r, state_col + t])
                raw[block * split : (block + 1) * split] = start + t + 1
    next_tokens = torch.full(
        (num_reqs + 2, NUM_SPEC + 1), runner_mod.INVALID_TOKEN_ID, dtype=torch.int64
    )
    for r, a in enumerate(accepted):
        next_tokens[r, :a] = 7
    before = [raw.clone() for raw in raws]
    runner._copy_spec_boundary_states(plan, next_tokens)
    return plan, before


PATTERNS = {
    "accept_all": lambda step: NUM_SPEC + 1,
    "reject_all": lambda step: 1,
    "accept_one": lambda step: 2,
    "accept_two": lambda step: 3,
    "mixed": lambda step: (1, 4, 2, 4, 3, 1, 4)[step % 7],
}


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("pattern", sorted(PATTERNS))
@pytest.mark.parametrize("phase", range(4))
def test_boundary_block_holds_the_boundary_state(async_scheduling, pattern, phase):
    runner, raws = _make_runner(async_scheduling=async_scheduling)
    accept = PATTERNS[pattern]
    start, ahead, step = BLOCK - 7 + phase, 0, 0
    first = start
    while start + NUM_SPEC + 1 <= 4 * BLOCK:
        # Every fifth step carries fewer drafts, so windows of every size
        # reach boundaries.
        drafts = NUM_SPEC if step % 5 else step % 4
        accepted = min(accept(step), drafts + 1)
        _, before = _verify_step(
            runner,
            raws,
            [start],
            [drafts],
            [accepted],
            [ahead if async_scheduling else 0],
        )
        # Only blocks a boundary completed this step change.
        changed = torch.nonzero((raws[0] != before[0]).any(dim=1)).flatten().tolist()
        reached = [
            t for t in range(BLOCK, 5 * BLOCK, BLOCK) if start < t <= start + accepted
        ]
        assert changed == [1 + t // BLOCK - 1 for t in reached if drafts]
        ahead = drafts + 1 - accepted
        start += accepted
        step += 1
    for boundary in range(BLOCK * (first // BLOCK + 1), start + 1, BLOCK):
        col = boundary // BLOCK - 1
        assert _block_value(runner, raws[0], 0, col) == boundary, (boundary, col)


def test_no_copies_without_block_hashes():
    runner, _ = _make_runner(prefix_caching=False)
    sched = SimpleNamespace(
        scheduled_spec_decode_tokens={"r0": [0] * NUM_SPEC},
        num_scheduled_tokens={"r0": NUM_SPEC + 1},
    )
    runner.input_batch.num_computed_tokens_cpu[0] = BLOCK - 1
    seq_lens = torch.tensor([BLOCK + NUM_SPEC], dtype=torch.int32)
    assert runner._plan_spec_boundary_copies(sched, 0, 1, seq_lens) is None
    runner.vllm_config.kv_transfer_config = object()
    assert runner._plan_spec_boundary_copies(sched, 0, 1, seq_lens) is not None


def test_no_copies_without_a_checkpoint_window():
    runner, _ = _make_runner(window=1)
    sched = SimpleNamespace(
        scheduled_spec_decode_tokens={"r0": [0] * NUM_SPEC},
        num_scheduled_tokens={"r0": NUM_SPEC + 1},
    )
    runner.input_batch.num_computed_tokens_cpu[0] = BLOCK - 1
    seq_lens = torch.tensor([BLOCK + NUM_SPEC], dtype=torch.int32)
    assert runner._plan_spec_boundary_copies(sched, 0, 1, seq_lens) is None


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_rows_away_from_boundaries_are_not_staged(async_scheduling):
    runner, raws = _make_runner(num_reqs=3, async_scheduling=async_scheduling)
    # Row 1 cannot reach a boundary even with its whole window; rows 0 and 2
    # can, and only row 2 does.
    starts = [BLOCK - 2, BLOCK + 4, 2 * BLOCK - 1]
    plan, _ = _verify_step(runner, raws, starts, [NUM_SPEC] * 3, [1, 4, 1], [0, 0, 0])
    assert plan is not None
    staged = plan.entries[:, :2].tolist()
    assert [0, BLOCK] in staged and [2, 2 * BLOCK] in staged
    assert all(row != 1 for row, boundary in staged if boundary)
    assert _block_value(runner, raws[0], 0, 0) == UNWRITTEN
    assert _block_value(runner, raws[0], 2, 1) == 2 * BLOCK


def test_every_group_and_pool_row_gets_the_copy():
    runner, raws = _make_runner(num_groups=2, split=3)
    _verify_step(runner, raws, [BLOCK - 3], [NUM_SPEC], [NUM_SPEC + 1], [0])
    for g in range(2):
        assert _block_value(runner, raws[g], 0, 0, group=g) == BLOCK


def test_candidates_pad_within_max_num_reqs():
    runner, raws = _make_runner(num_reqs=9, max_num_reqs=9)
    plan, _ = _verify_step(
        runner, raws, [BLOCK - 1] * 9, [NUM_SPEC] * 9, [1] * 9, [0] * 9
    )
    assert plan is not None and plan.entries.shape[0] == 9
    for r in range(9):
        assert _block_value(runner, raws[0], r, 0) == BLOCK


def test_windows_inside_a_block_stage_nothing():
    runner, raws = _make_runner()
    plan, before = _verify_step(runner, raws, [3], [NUM_SPEC], [4], [0])
    assert plan is None
    assert all(torch.equal(raw, b) for raw, b in zip(raws, before))


@pytest.mark.parametrize("ahead", range(NUM_SPEC + 1))
def test_async_host_count_ahead_still_finds_the_boundary(ahead):
    runner, raws = _make_runner(async_scheduling=True)
    # The host count runs `ahead` tokens past the device count, which alone
    # determines the copy.
    start = BLOCK - 2
    _verify_step(runner, raws, [start], [NUM_SPEC], [2], [ahead])
    assert _block_value(runner, raws[0], 0, 0) == BLOCK

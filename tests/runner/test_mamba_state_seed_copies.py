"""Unit tests for the align-mode Mamba state-block seed copies.

These cover the decision logic in
``TPUModelRunner._collect_mamba_state_seed_copies`` — which state block is
seeded from which when a request's last-token block advances (chunked-prefill
boundary, decode crossing, or prefix-cache resume). The method stages
``(raw, src, dst)`` copies into ``_pending_mamba_state_copies``; the tests
assert on those staged pairs (CPU only, no TPU op), and one test drives
``_flush_mamba_state_seed_copies`` through a CPU stand-in to check the pairs
are applied.

The seed copies are the mamba half of prefix-cache correctness: a wrong pair
here reads stale recurrent state on a cache hit. End-to-end this is covered by
MMLU-with-prefix-caching parity; this file pins the block-selection logic
directly and cheaply.
"""
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_torchtpu.runner.tpu_runner as runner_mod
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

BLOCK_SIZE = 256
STATE_DIM = 8
NUM_BLOCKS = 32

_HAS_SPLIT = "_pool_block_split" in inspect.getsource(
    TPUModelRunner._collect_mamba_state_seed_copies)


def _table(block_rows, num_rows=16, num_cols=8):
    cpu = torch.zeros((num_rows, num_cols), dtype=torch.int32)
    for row, blocks in enumerate(block_rows):
        cpu[row, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    return SimpleNamespace(get_cpu_tensor=lambda cpu=cpu: cpu)


def _make_self(req_ids,
               num_computed,
               group_block_rows,
               *,
               state_pos=None,
               split=1):
    """Build a minimal fake runner ``self`` for the collector.

    ``group_block_rows`` is a list (one entry per mamba group) of block-table
    rows; each group gets its own block table and its own raw pool buffer.
    """
    input_batch = SimpleNamespace(
        req_id_to_index={
            rid: i
            for i, rid in enumerate(req_ids)
        },
        num_computed_tokens_cpu=np.array(num_computed + [0] *
                                         (16 - len(num_computed))),
        req_ids=list(req_ids) + [None] * (16 - len(req_ids)),
        # block_table is indexed by group id; give attn gid 0 a dummy table
        # so mamba gids start at 1 (matching a real hybrid model layout).
        block_table=[_table([])] + [_table(rows) for rows in group_block_rows],
    )
    raws = [
        torch.zeros((NUM_BLOCKS, STATE_DIM), dtype=torch.float32)
        for _ in group_block_rows
    ]
    plan = [(gid + 1, [raws[gid]]) for gid in range(len(group_block_rows))]
    return SimpleNamespace(
        _mamba_copy_plan=plan,
        _mamba_state_pos=dict(state_pos or {}),
        _pool_block_split=split,
        block_size=BLOCK_SIZE,
        device="cpu",
        _pending_mamba_state_copies=[],
        input_batch=input_batch,
    ), raws


def _sched(num_scheduled):
    return SimpleNamespace(num_scheduled_tokens=num_scheduled)


def _real_pairs(staged):
    """Non-padding (src, dst) pairs from one staged (raw, src_t, dst_t)."""
    _raw, src_t, dst_t = staged
    return [(int(s), int(d)) for s, d in zip(src_t.tolist(), dst_t.tolist())
            if (s, d) != (0, 0)]


def _collect(fake, sched, num_reqs=None):
    if num_reqs is None:
        num_reqs = len([r for r in fake.input_batch.req_ids if r is not None])
    TPUModelRunner._collect_mamba_state_seed_copies(fake, sched, 0, num_reqs)


# --- crossing detection -------------------------------------------------


def test_first_prefill_chunk_no_copy():
    # computed=0 => prev block index is -1 (no prior state) => never a copy.
    fake, _ = _make_self(["a"], [0], [[[5, 6, 7, 8]]])
    _collect(fake, _sched({"a": 2 * BLOCK_SIZE}))
    assert fake._pending_mamba_state_copies == []
    assert fake._mamba_state_pos["a"] == 1


def test_chunk_advance_copies_state():
    fake, _ = _make_self(["a"], [2 * BLOCK_SIZE], [[[5, 6, 7, 8]]],
                         state_pos={"a": 1})
    _collect(fake, _sched({"a": BLOCK_SIZE}))
    assert fake._mamba_state_pos["a"] == 2
    assert len(fake._pending_mamba_state_copies) == 1
    # block col 1 -> col 2 in the table [5,6,7,8] => seed block 7 from 6.
    assert _real_pairs(fake._pending_mamba_state_copies[0]) == [(6, 7)]


def test_pcp_uses_rank_local_state_block_size(monkeypatch):
    monkeypatch.setattr(runner_mod, "get_total_cp_world_size", lambda: 8)
    # A PCP rank's two table columns cover 16 logical manager blocks. The
    # third global chunk is still in local column 0 and must not index column 2.
    fake, _ = _make_self(["a"], [2 * BLOCK_SIZE], [[[5, 6]]])
    _collect(fake, _sched({"a": BLOCK_SIZE}))
    assert fake._mamba_state_pos["a"] == 0
    assert fake._pending_mamba_state_copies == []


def test_pcp_local_state_block_crossing_copies(monkeypatch):
    monkeypatch.setattr(runner_mod, "get_total_cp_world_size", lambda: 8)
    fake, _ = _make_self(["a"], [8 * BLOCK_SIZE], [[[5, 6]]],
                         state_pos={"a": 0})
    _collect(fake, _sched({"a": BLOCK_SIZE}))
    assert fake._mamba_state_pos["a"] == 1
    assert _real_pairs(fake._pending_mamba_state_copies[0]) == [(5, 6)]


def test_cache_hit_resume_copies_from_checkpoint():
    # No prior state_pos: a resumed request's prev block is derived from its
    # computed-token count (the cached prefix boundary), then seeded forward.
    fake, _ = _make_self(["b"], [2 * BLOCK_SIZE], [[[9, 10, 11, 12]]])
    _collect(fake, _sched({"b": BLOCK_SIZE}))
    assert fake._mamba_state_pos["b"] == 2
    assert _real_pairs(fake._pending_mamba_state_copies[0]) == [(10, 11)]


def test_decode_within_block_no_copy():
    fake, _ = _make_self(["a"], [3 * BLOCK_SIZE - 1], [[[5, 6, 7, 8]]],
                         state_pos={"a": 2})
    _collect(fake, _sched({"a": 1}))
    assert fake._pending_mamba_state_copies == []
    assert fake._mamba_state_pos["a"] == 2


def test_decode_boundary_crossing_copies():
    fake, _ = _make_self(["a"], [3 * BLOCK_SIZE], [[[5, 6, 7, 8]]],
                         state_pos={"a": 2})
    _collect(fake, _sched({"a": 1}))
    assert fake._mamba_state_pos["a"] == 3
    assert _real_pairs(fake._pending_mamba_state_copies[0]) == [(7, 8)]


# --- request lifecycle / windowing --------------------------------------


def test_stale_request_positions_are_forgotten():
    # A req_id no longer in the batch is dropped from _mamba_state_pos.
    fake, _ = _make_self(["live"], [2 * BLOCK_SIZE], [[[5, 6, 7, 8]]],
                         state_pos={
                             "live": 1,
                             "gone": 3
                         })
    _collect(fake, _sched({"live": BLOCK_SIZE}))
    assert "gone" not in fake._mamba_state_pos
    assert "live" in fake._mamba_state_pos


def test_start_index_window_skips_other_rows():
    # num_reqs covers only a sub-window; rows outside it are untouched.
    fake, _ = _make_self(["a", "b"], [2 * BLOCK_SIZE, 2 * BLOCK_SIZE],
                         [[[5, 6, 7, 8], [9, 10, 11, 12]]],
                         state_pos={
                             "a": 1,
                             "b": 1
                         })
    TPUModelRunner._collect_mamba_state_seed_copies(
        fake, _sched({
            "a": BLOCK_SIZE,
            "b": BLOCK_SIZE
        }), 1, 1)
    # only row 1 (req "b") processed
    assert fake._mamba_state_pos["b"] == 2
    assert fake._mamba_state_pos["a"] == 1
    assert _real_pairs(fake._pending_mamba_state_copies[0]) == [(10, 11)]


# --- pool / group structure ---------------------------------------------


def test_no_plan_is_noop():
    # Non-align modes leave the copy plan empty => collector does nothing.
    fake, _ = _make_self(["a"], [2 * BLOCK_SIZE], [[[5, 6, 7, 8]]],
                         state_pos={"a": 1})
    fake._mamba_copy_plan = []
    _collect(fake, _sched({"a": BLOCK_SIZE}))
    assert fake._pending_mamba_state_copies == []


def test_null_block_pairs_filtered():
    # Block 0 is vLLM's null block; a pair touching it is skipped, not copied.
    fake, _ = _make_self(["a"], [BLOCK_SIZE], [[[3, 0]]], state_pos={"a": 0})
    _collect(fake, _sched({"a": 1}))
    assert fake._mamba_state_pos["a"] == 1
    # src=3, dst=0 (col1) => filtered => nothing staged.
    assert fake._pending_mamba_state_copies == []


def test_multiple_mamba_groups_get_independent_copies():
    fake, raws = _make_self(["a"], [2 * BLOCK_SIZE],
                            [[[5, 6, 7, 8]], [[15, 16, 17, 18]]],
                            state_pos={"a": 1})
    _collect(fake, _sched({"a": BLOCK_SIZE}))
    staged = {
        id(s[0]): _real_pairs(s)
        for s in fake._pending_mamba_state_copies
    }
    assert staged[id(raws[0])] == [(6, 7)]
    assert staged[id(raws[1])] == [(16, 17)]


def test_pairs_padded_to_bucket_ladder():
    # 3 real pairs pad up to the first bucket (8); 9 pairs to the next (32).
    fake, _ = _make_self(["a", "b", "c"], [2 * BLOCK_SIZE] * 3,
                         [[[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]],
                         state_pos={
                             "a": 1,
                             "b": 1,
                             "c": 1
                         })
    _collect(fake, _sched({"a": BLOCK_SIZE, "b": BLOCK_SIZE, "c": BLOCK_SIZE}))
    _raw, src_t, _dst_t = fake._pending_mamba_state_copies[0]
    assert len(src_t) == 8
    assert _real_pairs(fake._pending_mamba_state_copies[0]) == [(6, 7),
                                                                (10, 11),
                                                                (14, 15)]


# --- apply path ---------------------------------------------------------


def test_flush_applies_and_clears(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runner_mod, "copy_mamba_state_blocks",
        lambda raw, src, dst: calls.append(
            (id(raw), src.tolist(), dst.tolist())))
    fake, raws = _make_self(["a"], [2 * BLOCK_SIZE], [[[5, 6, 7, 8]]],
                            state_pos={"a": 1})
    _collect(fake, _sched({"a": BLOCK_SIZE}))
    TPUModelRunner._flush_mamba_state_seed_copies(fake)
    assert len(calls) == 1
    assert calls[0][0] == id(raws[0])
    assert fake._pending_mamba_state_copies == []


# --- split expansion (batched-RPA kernel-granular pool birth, #405) ------


@pytest.mark.skipif(not _HAS_SPLIT,
                    reason="split expansion only exists on the batched-RPA PR")
def test_split_expansion_fans_out_pairs():
    # split=3: manager pair (6,7) fans out to 3 consecutive kernel-block pairs
    # (6*3+j, 7*3+j) for j in 0..2.
    fake, _ = _make_self(["a"], [2 * BLOCK_SIZE], [[[5, 6, 7, 8]]],
                         state_pos={"a": 1},
                         split=3)
    _collect(fake, _sched({"a": BLOCK_SIZE}))
    assert _real_pairs(fake._pending_mamba_state_copies[0]) == [(18, 21),
                                                                (19, 22),
                                                                (20, 23)]

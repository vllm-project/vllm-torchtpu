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
               split=1,
               num_cols=8,
               state_block_size=BLOCK_SIZE,
               ckpt_window=1,
               read_offsets=None):
    """Build a minimal fake runner ``self`` for the collector.

    ``group_block_rows`` is a list (one entry per mamba group) of block-table
    rows; each group gets its own block table and its own raw pool buffer.
    ``state_block_size`` is the mamba groups' spec block size
    (``_mamba_state_block_size``, captured from the kv-cache config when the
    copy plan is built); the collector strides columns by it times the CP
    world size, never by the attention ``block_size``.
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
        block_table=[_table([], num_cols=num_cols)] +
        [_table(rows, num_cols=num_cols) for rows in group_block_rows],
    )
    raws = [
        torch.zeros((NUM_BLOCKS, STATE_DIM), dtype=torch.float32)
        for _ in group_block_rows
    ]
    plan = [(gid + 1, [raws[gid]]) for gid in range(len(group_block_rows))]
    ns = SimpleNamespace(
        _mamba_copy_plan=plan,
        _mamba_state_pos=dict(state_pos or {}),
        _mamba_state_block_size=state_block_size,
        _pool_block_split=split,
        # 1 = no speculative checkpoint group; > 1 takes the sliding-group
        # path exercised at the bottom of this file.
        _mamba_ckpt_window=ckpt_window,
        # The collector reads the manager block size from cache_config, not
        # the runner's init-time snapshot (the executor finalizes it after
        # the runner is constructed).
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
        block_size=BLOCK_SIZE,
        device="cpu",
        mamba_slot_read_offsets=read_offsets,
        _pending_mamba_state_copies=[],
        input_batch=input_batch,
    )
    # Padding/expansion/source-selection helpers the collector calls on
    # self; bind the real implementations onto the stand-in. The static and
    # class methods are already callable as-is.
    ns._bucket_len = TPUModelRunner._bucket_len
    ns._pad_to_bucket = TPUModelRunner._pad_to_bucket
    for name in ("_pad_dev_to_bucket", "_expand_pool_split",
                 "_spec_seed_sources"):
        setattr(ns, name, getattr(TPUModelRunner, name).__get__(ns))
    return ns, raws


def _sched(num_scheduled):
    return SimpleNamespace(num_scheduled_tokens=num_scheduled)


def _real_pairs(staged):
    """Non-padding (src, dst) pairs from one staged (raws, src_t, dst_t)."""
    _raws, src_t, dst_t = staged
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
    monkeypatch.setattr(runner_mod, "get_dcp_group",
                        lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(runner_mod, "get_pcp_group",
                        lambda: SimpleNamespace(world_size=1))
    # A PCP rank's two table columns cover 16 logical manager blocks. The
    # third global chunk is still in local column 0 and must not index column 2.
    fake, _ = _make_self(["a"], [2 * BLOCK_SIZE], [[[5, 6]]])
    _collect(fake, _sched({"a": BLOCK_SIZE}))
    assert fake._mamba_state_pos["a"] == 0
    assert fake._pending_mamba_state_copies == []


def test_pcp_local_state_block_crossing_copies(monkeypatch):
    monkeypatch.setattr(runner_mod, "get_dcp_group",
                        lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(runner_mod, "get_pcp_group",
                        lambda: SimpleNamespace(world_size=1))
    fake, _ = _make_self(["a"], [8 * BLOCK_SIZE], [[[5, 6]]],
                         state_pos={"a": 0})
    _collect(fake, _sched({"a": BLOCK_SIZE}))
    assert fake._mamba_state_pos["a"] == 1
    assert _real_pairs(fake._pending_mamba_state_copies[0]) == [(5, 6)]


def test_pcp_disagg_mamba_block_table_dimensions(monkeypatch):
    monkeypatch.setattr(runner_mod, "get_dcp_group",
                        lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(runner_mod, "get_pcp_group",
                        lambda: SimpleNamespace(world_size=1))
    # Simulate disaggregated serving where Mamba physical block size is 2048
    # (rounded up power-of-two fit size for TP=2 decode).
    # With max_model_len = 4096 and mamba_block_size = 2048, the block table
    # is allocated with exactly 2 columns ([0, 1]).
    # Under CP=8, each logical column spans 2048 * 8 = 16384 logical tokens.
    max_model_len = 4096
    mamba_block_size = 2048
    num_cols = max_model_len // mamba_block_size  # 2 columns

    fake, _ = _make_self(["a"], [32], [[[5, 6]]],
                         num_cols=num_cols,
                         state_block_size=mamba_block_size)
    _collect(fake, _sched({"a": 1}))

    assert fake._mamba_state_pos["a"] == 0
    assert fake._pending_mamba_state_copies == []


def test_pcp_disagg_mamba_block_stride_comparison(monkeypatch):
    """Documents the two historical wrong strides against the correct one.

    The collector must stride mamba tables by the mamba groups' physical
    block size times the CP world; the two bugs both strode by an
    attention-derived size instead. Each wrong stride is simulated here by
    installing it as ``_mamba_state_block_size``.
    """
    monkeypatch.setattr(runner_mod, "get_dcp_group",
                        lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(runner_mod, "get_pcp_group",
                        lambda: SimpleNamespace(world_size=1))

    # Wrong stride 1 (pre-PR172): attention block_size=16 with no cp factor,
    # i.e. an effective stride of 16 logical tokens per column.
    # At token 32, curr = 32 // 16 = 2 -> IndexError on a 2-column table.
    fake_bug, _ = _make_self(["a"], [32], [[[5, 6]]],
                             num_cols=2,
                             state_block_size=16 // 8)
    with pytest.raises(IndexError):
        _collect(fake_bug, _sched({"a": 1}))

    # Wrong stride 2 (PR 172): attention block_size * cp = 16 * 8 = 128.
    # Survives token 32 (32 // 128 = 0) -- the CI's short prompts hid it...
    fake_pr172, _ = _make_self(["a"], [32], [[[5, 6]]],
                               num_cols=2,
                               state_block_size=16)
    _collect(fake_pr172, _sched({"a": 1}))
    assert fake_pr172._mamba_state_pos["a"] == 0

    # ...but any sequence past 2 columns * 128 tokens crashes again:
    # curr = 256 // 128 = 2 -> IndexError.
    fake_pr172_large, _ = _make_self(["a"], [256], [[[5, 6]]],
                                     num_cols=2,
                                     state_block_size=16)
    with pytest.raises(IndexError):
        _collect(fake_pr172_large, _sched({"a": 1}))

    # Correct stride: the mamba groups' physical block size (2048), giving
    # 2048 * 8 = 16384 logical tokens per column.
    fake_mamba_true, _ = _make_self(["a"], [256], [[[5, 6]]],
                                    num_cols=2,
                                    state_block_size=2048)
    _collect(fake_mamba_true, _sched({"a": 1}))
    assert fake_mamba_true._mamba_state_pos["a"] == 0


def test_cache_hit_resume_copies_from_checkpoint():
    # No prior state_pos: a resumed request's prev block is derived from its
    # computed-token count (the cached prefix boundary), then seeded forward.
    fake, _ = _make_self(["b"], [2 * BLOCK_SIZE], [[[9, 10, 11, 12]]])
    _collect(fake, _sched({"b": BLOCK_SIZE}))
    assert fake._mamba_state_pos["b"] == 2
    assert _real_pairs(fake._pending_mamba_state_copies[0]) == [(10, 11)]


def test_cache_hit_uses_mamba_group_block_size():
    mamba_block_size = 768
    fake, _ = _make_self(["b"], [2 * mamba_block_size],
                         [[[5, 6, 7, 8, 9, 10]]],
                         state_block_size=mamba_block_size)
    # The runner's constructor-time attention scalars can still contain the
    # input value after the platform derives the physical Mamba geometry.
    fake.block_size = 16

    _collect(fake, _sched({"b": mamba_block_size}))

    assert fake._mamba_state_pos["b"] == 2
    assert _real_pairs(fake._pending_mamba_state_copies[0]) == [(6, 7)]


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
    # Each group lives on its own raw buffer here, so each gets its own
    # program; groups sharing a buffer set would be merged into one.
    staged = {
        tuple(id(r) for r in s[0]): _real_pairs(s)
        for s in fake._pending_mamba_state_copies
    }
    assert staged[(id(raws[0]), )] == [(6, 7)]
    assert staged[(id(raws[1]), )] == [(16, 17)]


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
    _raws, src_t, _dst_t = fake._pending_mamba_state_copies[0]
    assert len(src_t) == 8
    assert _real_pairs(fake._pending_mamba_state_copies[0]) == [(6, 7),
                                                                (10, 11),
                                                                (14, 15)]


# --- apply path ---------------------------------------------------------


def test_flush_applies_and_clears(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runner_mod, "copy_mamba_state_blocks",
        lambda raws, src, dst: calls.append(
            ([id(r) for r in raws], src.tolist(), dst.tolist())))
    fake, raws = _make_self(["a"], [2 * BLOCK_SIZE], [[[5, 6, 7, 8]]],
                            state_pos={"a": 1})
    _collect(fake, _sched({"a": BLOCK_SIZE}))
    TPUModelRunner._flush_mamba_state_seed_copies(fake)
    assert len(calls) == 1
    assert calls[0][0] == [id(raws[0])]
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


# --- speculative decoding: the checkpoint group slides with the state ----
#
# With `num_speculative_blocks = window - 1`, vLLM's MambaManager appends a
# block and never shifts, so when the state column advances the whole
# checkpoint group slides one column right: what was checkpoint 1 becomes the
# state block, checkpoint 2 becomes checkpoint 1, and a fresh (stateless)
# block appears at the end. Seeding the new state block from the OLD state
# block would leave the request resuming one checkpoint late. What has to
# move is the checkpoint it is about to resume from, which lands as
# checkpoint 0 of the new group -- so the read offset resets to 0.


def _spec_self(offset, *, window=3, row=(10, 11, 12, 13), split=1):
    """One request about to cross from state column 0 to column 1.

    Pre-crossing the group is blocks (10, 11, 12) with the state at 10;
    post-crossing the manager has appended 13 and the group is (11, 12, 13).
    ``offset`` is the read offset parked on the old state block, i.e. the
    checkpoint of the request's last accepted token.
    """
    read_offsets = torch.zeros(NUM_BLOCKS, dtype=torch.int32)
    read_offsets[row[0]] = offset
    return _make_self(["a"], [BLOCK_SIZE], [[list(row)]],
                      state_pos={"a": 0},
                      ckpt_window=window,
                      read_offsets=read_offsets,
                      split=split)


@pytest.mark.parametrize("offset,expected_src", [(0, 10), (1, 11), (2, 12)])
def test_spec_crossing_seeds_from_the_resumed_checkpoint(offset, expected_src):
    fake, _ = _spec_self(offset)
    _collect(fake, _sched({"a": 1}))
    assert _real_pairs(fake._pending_mamba_state_copies[0]) == [(expected_src,
                                                                 11)]


def test_spec_crossing_does_not_carry_the_stale_state_block():
    # Regression: seeding from the old state block (10) regardless of the
    # offset left the request one checkpoint late, and at the top of the
    # window pointed it at the freshly appended block, which holds no state.
    fake, _ = _spec_self(2)
    _collect(fake, _sched({"a": 1}))
    srcs = [s for s, _ in _real_pairs(fake._pending_mamba_state_copies[0])]
    assert srcs == [12] and 10 not in srcs


@pytest.mark.parametrize("offset", [0, 1, 2])
def test_spec_crossing_resets_the_read_offset(offset):
    # The seeded checkpoint IS checkpoint 0 of the post-crossing group.
    fake, _ = _spec_self(offset)
    _collect(fake, _sched({"a": 1}))
    assert int(fake.mamba_slot_read_offsets[11]) == 0


def test_spec_crossing_offset_past_the_row_falls_back_to_the_state_block():
    # A row too short to hold the whole group must not gather out of range.
    fake, _ = _spec_self(2, window=3, row=(10, 11))
    _collect(fake, _sched({"a": 1}))
    assert _real_pairs(fake._pending_mamba_state_copies[0]) == [(10, 11)]


@pytest.mark.skipif(not _HAS_SPLIT,
                    reason="split expansion only exists on the batched-RPA PR")
def test_spec_crossing_fans_out_pool_split():
    # The device-side source selection still expands to kernel blocks.
    fake, _ = _spec_self(2, split=3)
    _collect(fake, _sched({"a": 1}))
    assert _real_pairs(fake._pending_mamba_state_copies[0]) == [(36, 33),
                                                                (37, 34),
                                                                (38, 35)]


def test_non_spec_crossing_keeps_the_state_block_source():
    # window == 1: no checkpoint group, so the old state block is the source
    # and its offset migrates to the new state block unchanged.
    read_offsets = torch.zeros(NUM_BLOCKS, dtype=torch.int32)
    read_offsets[7] = 3
    fake, _ = _make_self(["a"], [3 * BLOCK_SIZE], [[[5, 6, 7, 8]]],
                         state_pos={"a": 2},
                         ckpt_window=1,
                         read_offsets=read_offsets)
    _collect(fake, _sched({"a": 1}))
    assert _real_pairs(fake._pending_mamba_state_copies[0]) == [(7, 8)]
    assert int(fake.mamba_slot_read_offsets[8]) == 3

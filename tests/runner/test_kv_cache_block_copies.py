"""Unit tests for the scheduler's copy-on-write pool block copies.

``TPUModelRunner._apply_kv_cache_block_copies`` executes
``SchedulerOutput.kv_cache_block_copies``: the private-block copies vLLM
queues when a prefix-cache hit ends inside a block (``prefix_match_unit``
finer than the block size). Without them the resuming request reads the
shared tail block's stale bytes and generates silently wrong output.

The tests drive the method with a fake runner ``self`` (CPU only, no TPU op)
and assert on the pool block index pairs it hands the copy op.
"""
from types import SimpleNamespace

import pytest
import torch

import vllm_torchtpu.runner.tpu_runner as runner_mod
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

NUM_BLOCKS = 32
BLOCK_ROW = 8


def _copy(src, dst):
    return SimpleNamespace(src_block_id=src, dst_block_id=dst)


def _make_self(num_raw_tensors=1, split=1):
    return SimpleNamespace(
        kv_cache_raw_tensors=[
            torch.zeros((NUM_BLOCKS, BLOCK_ROW), dtype=torch.float32)
            for _ in range(num_raw_tensors)
        ],
        _pool_block_split=split,
        device="cpu",
    )


def _sched(copies):
    return SimpleNamespace(kv_cache_block_copies=copies)


def _apply(fake, copies, monkeypatch):
    calls = []
    monkeypatch.setattr(
        runner_mod, "copy_mamba_state_blocks",
        lambda raw, src, dst: calls.append(
            (id(raw), src.tolist(), dst.tolist())))
    TPUModelRunner._apply_kv_cache_block_copies(fake, _sched(copies))
    return calls


def _real_pairs(call):
    _raw, srcs, dsts = call
    return [(s, d) for s, d in zip(srcs, dsts) if (s, d) != (0, 0)]


def test_no_copies_is_noop(monkeypatch):
    fake = _make_self()
    assert _apply(fake, None, monkeypatch) == []
    assert _apply(fake, [], monkeypatch) == []


def test_single_copy_runs_on_the_pool(monkeypatch):
    fake = _make_self()
    calls = _apply(fake, [_copy(6, 7)], monkeypatch)
    assert len(calls) == 1
    assert calls[0][0] == id(fake.kv_cache_raw_tensors[0])
    assert _real_pairs(calls[0]) == [(6, 7)]


def test_every_pool_buffer_is_copied(monkeypatch):
    # A manager block id is global: the block's bytes live at the same row in
    # every pool buffer, so all of them must be copied.
    fake = _make_self(num_raw_tensors=3)
    calls = _apply(fake, [_copy(6, 7)], monkeypatch)
    assert [call[0] for call in calls
            ] == [id(raw) for raw in fake.kv_cache_raw_tensors]
    assert all(_real_pairs(call) == [(6, 7)] for call in calls)


def test_split_expansion_fans_out_pairs(monkeypatch):
    # split=3: a manager block is 3 consecutive kernel-granular pool blocks.
    fake = _make_self(split=3)
    calls = _apply(fake, [_copy(6, 7)], monkeypatch)
    assert _real_pairs(calls[0]) == [(18, 21), (19, 22), (20, 23)]


def test_pairs_padded_to_bucket_ladder(monkeypatch):
    fake = _make_self()
    calls = _apply(fake, [_copy(i, i + 10) for i in range(1, 4)], monkeypatch)
    assert len(calls[0][1]) == 8
    assert _real_pairs(calls[0]) == [(1, 11), (2, 12), (3, 13)]

    calls = _apply(fake, [_copy(i, i + 10) for i in range(1, 10)], monkeypatch)
    assert len(calls[0][1]) == 32


def test_copies_without_the_pool_are_rejected(monkeypatch):
    # The per-layer layout has no pool rows to copy; fail instead of dropping
    # the copies and reading the shared tail block.
    fake = _make_self(num_raw_tensors=0)
    with pytest.raises(AssertionError, match="unified KV pool"):
        _apply(fake, [_copy(6, 7)], monkeypatch)

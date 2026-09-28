"""Tests for page-local prefix-cache copy-on-write block copies."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

import vllm_torchtpu.runner.tpu_runner as runner_mod
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.utils import tpu_bind_kv_cache

pytestmark = pytest.mark.cpu_test


def _runner_with_raw_tensors(raw_tensors):
    runner = object.__new__(TPUModelRunner)
    runner.requests = {}
    runner._unified_kv_layout = bool(raw_tensors)
    runner.kv_cache_raw_tensors = raw_tensors
    runner._install_spec_token_room_guard = lambda: None
    return runner


def test_update_states_routes_pooled_copies_away_from_generic_path(monkeypatch):
    """Verify that CoW block copies in a unified pool bypass generic GPU copy.

    When unified KV pool is active (`kv_cache_raw_tensors` present),
    `TPUModelRunner._update_states` must:
    1. Intercept `scheduler_output.kv_cache_block_copies` and pass
       `kv_cache_block_copies=None` to `super()._update_states()`, avoiding
       upstream vLLM's generic GPU in-place copy (`blocks[dst] = blocks[src]`
       and `blocks.set_(storage)`) which triggers PyTorch/XLA OOM on TPU.
    2. Retain all other fields (e.g., `new_block_ids_to_zero`) for base updates.
    3. Route the copies to `_apply_kv_cache_block_copies` after base state updates.
    """
    base_outputs = []
    events = []
    result_marker = object()

    def base_update(_runner, scheduler_output):
        events.append("base_zero_and_state")
        base_outputs.append(scheduler_output)
        return result_marker

    monkeypatch.setattr(GPUModelRunner, "_update_states", base_update)
    runner = _runner_with_raw_tensors([torch.empty(1)])
    local_copies = []

    def apply_page_local(copies):
        events.append("page_local_copy")
        local_copies.append(copies)

    runner._apply_kv_cache_block_copies = apply_page_local

    scheduler_output = SchedulerOutput.make_empty()
    copies = [KVCacheBlockCopy(src_block_id=6, dst_block_id=7)]
    scheduler_output.kv_cache_block_copies = copies
    scheduler_output.new_block_ids_to_zero = [7]

    result = TPUModelRunner._update_states(runner, scheduler_output)

    assert result is result_marker
    assert len(base_outputs) == 1
    assert base_outputs[0] is not scheduler_output
    assert base_outputs[0].kv_cache_block_copies is None
    assert base_outputs[0].new_block_ids_to_zero == [7]
    assert scheduler_output.kv_cache_block_copies == copies
    assert local_copies == [copies]
    assert events == ["base_zero_and_state", "page_local_copy"]


def test_update_states_keeps_generic_copy_for_non_pooled_layout(monkeypatch):
    """Verify fallback to generic copy when unified KV pool is not active.

    If `kv_cache_raw_tensors` is empty (non-pooled layout), `_update_states`
    must pass `scheduler_output` intact to `super()._update_states()` and must
    not invoke `_apply_kv_cache_block_copies`.
    """
    base_outputs = []
    monkeypatch.setattr(
        GPUModelRunner,
        "_update_states",
        lambda _runner, scheduler_output: base_outputs.append(scheduler_output),
    )
    runner = _runner_with_raw_tensors([])
    local_copies = []
    runner._apply_kv_cache_block_copies = local_copies.append

    scheduler_output = SchedulerOutput.make_empty()
    copies = [KVCacheBlockCopy(src_block_id=6, dst_block_id=7)]
    scheduler_output.kv_cache_block_copies = copies

    TPUModelRunner._update_states(runner, scheduler_output)

    assert base_outputs == [scheduler_output]
    assert local_copies == []


def test_page_local_copy_expands_split_and_updates_every_pool(monkeypatch):
    """Verify manager block fan-out, bucket padding, and pool kernel invocation.

    In the unified pool:
    1. A logical scheduler manager block may span `_pool_block_split` physical
       kernel-granular blocks. Verifies each `(src, dst)` pair is fanned out
       by `_pool_block_split`.
    2. Pairs are padded to a power-of-4 bucket ladder with `(0, 0)` null-copies
       to prevent TPU shape recompilations.
    3. `copy_mamba_state_blocks` is called on `kv_cache_raw_tensors`, copying
       the memory in-place across all raw buffers.
    """
    raw_tensors = [
        torch.arange(32 * 4, dtype=torch.float32).reshape(32, 4),
        torch.arange(32 * 4, dtype=torch.float32).reshape(32, 4) + 1000,
    ]
    before = [raw.clone() for raw in raw_tensors]
    fake = SimpleNamespace(
        kv_cache_raw_tensors=raw_tensors,
        _unified_kv_layout=True,
        is_unified_pool_used=lambda: True,
        _pool_block_split=3,
        _pad_to_bucket=TPUModelRunner._pad_to_bucket,
        device="cpu",
        mamba_slot_read_offsets=None,
        bm_pool_layout=None,
    )
    fake._copy_mamba_state_blocks = TPUModelRunner._copy_mamba_state_blocks.__get__(
        fake
    )
    calls = []

    def cpu_page_copy(raws, src, dst):
        calls.append((id(raws), src.tolist(), dst.tolist()))
        for raw in raws:
            raw[dst.long()] = raw[src.long()]

    monkeypatch.setattr(runner_mod, "copy_mamba_state_blocks", cpu_page_copy)

    TPUModelRunner._apply_kv_cache_block_copies(
        fake, [KVCacheBlockCopy(src_block_id=6, dst_block_id=7)]
    )

    assert len(calls) == 1
    assert calls[0][0] == id(raw_tensors)
    assert calls[0][1] == [18, 19, 20, 0, 0, 0, 0, 0]
    assert calls[0][2] == [21, 22, 23, 0, 0, 0, 0, 0]
    for raw, old in zip(raw_tensors, before):
        torch.testing.assert_close(raw[21:24], old[18:21])
        torch.testing.assert_close(raw[24], old[24])


def test_tpu_bind_kv_cache_unpacks_lists():
    """Verify that `tpu_bind_kv_cache` flattens nested lists and tuples in
    `runner_kv_caches`.

    Hybrid models (e.g. Attention + GDN/Mamba) store recurrent layer states
    as lists or tuples of tensors `[conv_state, ssm_state]`. If appended directly,
    `runner.kv_caches` would contain nested sequences, breaking downstream callers
    expecting `list[torch.Tensor]` with `AttributeError: 'list' object has no
    attribute 'device'`.

    This test verifies that:
    1. `runner_kv_caches` receives flat `torch.Tensor` objects only (unpacks both lists
       and tuples).
    2. `forward_context` retains the original sequence structure for model layers.
    """
    t1 = torch.empty(1)
    t2 = torch.empty(1)
    t3 = torch.empty(1)
    t4 = torch.empty(1)
    kv_caches = {
        "layer.0": t1,
        "layer.1": [t2],
        "layer.2": (t3, t4),
    }
    forward_context = {
        "layer.0": SimpleNamespace(kv_cache=None),
        "layer.1": SimpleNamespace(kv_cache=None),
        "layer.2": SimpleNamespace(kv_cache=None),
    }
    runner_kv_caches = []
    tpu_bind_kv_cache(kv_caches, forward_context, runner_kv_caches)

    assert runner_kv_caches == [t1, t2, t3, t4]
    assert all(isinstance(t, torch.Tensor) for t in runner_kv_caches)
    assert forward_context["layer.1"].kv_cache == [t2]
    assert forward_context["layer.2"].kv_cache == (t3, t4)


def test_apply_kv_cache_block_copies_migrates_mamba_offsets(monkeypatch):
    """Verify that speculative decoding mamba read offsets are migrated
    during CoW copy."""
    raw_tensors = [torch.zeros(32, 4)]
    offsets = torch.arange(10, dtype=torch.long)
    fake = SimpleNamespace(
        kv_cache_raw_tensors=raw_tensors,
        _unified_kv_layout=True,
        is_unified_pool_used=lambda: True,
        mamba_slot_read_offsets=offsets,
        _pool_block_split=2,
        _pad_to_bucket=TPUModelRunner._pad_to_bucket,
        device="cpu",
        bm_pool_layout=None,
    )
    fake._copy_mamba_state_blocks = TPUModelRunner._copy_mamba_state_blocks.__get__(
        fake
    )
    migrated = []

    def mock_migrate(read_offsets, src_t, dst_t):
        migrated.append((src_t.tolist(), dst_t.tolist()))
        read_offsets[dst_t] = read_offsets[src_t]

    monkeypatch.setattr(runner_mod, "_rollback_offsets_migrate", mock_migrate)
    monkeypatch.setattr(runner_mod, "copy_mamba_state_blocks", lambda *args: None)

    copies = [KVCacheBlockCopy(src_block_id=3, dst_block_id=5)]
    TPUModelRunner._apply_kv_cache_block_copies(fake, copies)

    assert len(migrated) == 1
    assert migrated[0][0] == [3, 0, 0, 0, 0, 0, 0, 0]
    assert migrated[0][1] == [5, 0, 0, 0, 0, 0, 0, 0]
    assert offsets[5].item() == 3


def test_precompile_mamba_state_seed_copies_covers_all_groups_and_raw_tensors(
    monkeypatch,
):
    """Verify precompilation includes raw_tensors and scales limit to all kv groups."""
    raw1 = torch.zeros(16, 4)
    raw2 = torch.zeros(16, 4)
    kv_raw = [raw1, raw2]
    copy_plan = [(0, [raw1])]

    @contextmanager
    def mock_timed(_name):
        yield

    fake = SimpleNamespace(
        _mamba_copy_plan=copy_plan,
        kv_cache_raw_tensors=kv_raw,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object(), object()]),
        max_num_reqs=3,
        _pool_block_split=2,
        _bucket_len=TPUModelRunner._bucket_len,
        _precompile_timed=mock_timed,
        device="cpu",
        bm_pool_layout=None,
    )
    fake._copy_mamba_state_blocks = TPUModelRunner._copy_mamba_state_blocks.__get__(
        fake
    )
    compiled_raws = []
    compiled_lens = []

    def mock_copy(raws, src, dst):
        compiled_raws.append(tuple(id(r) for r in raws))
        compiled_lens.append(len(src))

    monkeypatch.setattr(runner_mod, "copy_mamba_state_blocks", mock_copy)
    monkeypatch.setattr(runner_mod, "synchronize_tensors", lambda *args: None)

    TPUModelRunner._precompile_mamba_state_seed_copies(fake)

    assert tuple(id(r) for r in [raw1]) in compiled_raws
    assert tuple(id(r) for r in kv_raw) in compiled_raws
    # With len(copy_plan)=1, len(kv_groups)=2, max_num_reqs=3, split=2:
    # Old single-group formula gave _bucket_len(1 * 3 * 2 = 6) = 8.
    # New multi-group formula gives _bucket_len(2 * 3 * 2 = 12) = 32.
    assert max(compiled_lens) == 32


def test_is_unified_pool_used():
    """Verify is_unified_pool_used checks whether kv_cache_raw_tensors
    is present and non-empty."""
    runner = object.__new__(TPUModelRunner)
    # Attribute missing
    assert not runner.is_unified_pool_used()

    # Raw tensors empty (non-pooled layout)
    runner.kv_cache_raw_tensors = []
    assert not runner.is_unified_pool_used()

    # Raw tensors allocated (unified pool active)
    runner.kv_cache_raw_tensors = [torch.empty(1)]
    assert runner.is_unified_pool_used()

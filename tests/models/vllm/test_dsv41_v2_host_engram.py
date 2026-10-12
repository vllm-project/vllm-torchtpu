# SPDX-License-Identifier: Apache-2.0
"""Host-side Engram hashing matches the device path on real rows."""

import types

import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.models.vllm.deepseek_v41.v2.engram import NgramHashState
from vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram import (
    capture_host_engram_inputs,
)

LAYERS, NGRAM, HEADS, DEPTH, BLOCK = 2, 3, 4, 4, 8


def _state(use_slot_cache, cache_slots=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    state = object.__new__(NgramHashState)
    torch.nn.Module.__init__(state)
    state.token_map = torch.randint(1, 50, (100,), generator=g)
    state.multipliers = torch.randint(1, 1 << 20, (LAYERS, NGRAM), generator=g)
    state.primes = torch.randint(1000, 5000, (LAYERS, NGRAM - 1, HEADS), generator=g)
    state.offsets = torch.randint(0, 9999, (LAYERS, (NGRAM - 1) * HEADS), generator=g)
    state.lookback_depth = DEPTH
    state.pad_id = 0
    state.use_slot_cache = use_slot_cache
    state.block_size = BLOCK
    # Garbage history: the host path must never need it for prompt rows.
    state._cache = torch.randint(1, 50, (cache_slots,), generator=g)
    return state


def _all_prompt_step():
    """Two requests mid-prompt and from start, padded to 16 rows."""
    prompts = [list(range(10, 30)), list(range(40, 55))]
    starts, counts = [8, 0], [6, 5]
    ids, pos = [], []
    for p, s, n in zip(prompts, starts, counts, strict=True):
        ids += p[s : s + n]
        pos += list(range(s, s + n))
    real = len(ids)
    ids += [0] * (16 - real)
    pos += [0] * (16 - real)
    qsl = torch.tensor([0, 6, 11, 11], dtype=torch.int32)
    lookback = torch.full((4, DEPTH), -1, dtype=torch.int32)
    for r, (p, s) in enumerate(zip(prompts, starts, strict=True)):
        for j in range(DEPTH):
            if 0 <= s - 1 - j < len(p):
                lookback[r, j] = p[s - 1 - j]
    ids = torch.tensor(ids, dtype=torch.int32)
    return ids, torch.tensor(pos), qsl, lookback, real


def test_hash_on_host_matches_device_path_on_real_rows():
    ids, pos, qsl, lookback, real = _all_prompt_step()
    dead = torch.zeros_like(ids, dtype=torch.bool)
    dead_lb = torch.zeros_like(lookback, dtype=torch.bool)
    for use_slot_cache in (False, True):
        state = _state(use_slot_cache)
        slot_mapping = torch.arange(ids.numel()) if use_slot_cache else None
        block_table = torch.arange(16).reshape(4, 4)
        device = state.forward(
            ids, pos, qsl, dead, lookback, dead_lb, slot_mapping, block_table
        )
        host = _state(use_slot_cache).hash_on_host(
            ids, pos, qsl, dead, lookback, dead_lb
        )
        assert host.dtype == device.dtype == torch.int32
        assert host.shape == device.shape
        assert torch.equal(host[:real], device[:real])


def test_hash_on_host_caches_cpu_tables_once():
    ids, pos, qsl, lookback, _ = _all_prompt_step()
    state = _state(False)
    dead = torch.zeros_like(ids, dtype=torch.bool)
    state.hash_on_host(ids, pos, qsl, dead, lookback, dead.new_zeros(lookback.shape))
    tables = state._host_tables
    state.hash_on_host(ids, pos, qsl, dead, lookback, dead.new_zeros(lookback.shape))
    assert state._host_tables is tables


def _runner(seq_lens, num_prompt, kind=SequenceLayoutKind.ALL):
    return types.SimpleNamespace(
        _last_sequence_layout_plan=types.SimpleNamespace(kind=kind),
        input_batch=types.SimpleNamespace(num_prompt_tokens=np.array(num_prompt)),
        seq_lens_np=np.array(seq_lens),
        input_ids_cpu=torch.zeros(8, dtype=torch.int32),
        positions_cpu=torch.zeros(8, dtype=torch.int64),
        query_start_loc_cpu=torch.zeros(5, dtype=torch.int32),
        lookback_token_ids=types.SimpleNamespace(
            cpu=torch.zeros(4, DEPTH, dtype=torch.int32)
        ),
    )


def test_capture_flags_all_prompt_only_when_no_generated_token():
    assert capture_host_engram_inputs(_runner([8, 3], [20, 3]), 2).all_prompt
    # Request 1 is decoding (seq_len past its prompt): all_prompt is False.
    assert not capture_host_engram_inputs(_runner([8, 4], [20, 3]), 2).all_prompt
    other = next(k for k in SequenceLayoutKind if k is not SequenceLayoutKind.ALL)
    assert capture_host_engram_inputs(_runner([8], [20], kind=other), 1) is None


def test_decode_and_mixed_step_host_hash_matches_device_path():
    """Decode/mixed steps resolve async input_ids and full lookback on CPU."""
    # Request 0: decoding at pos 12 (num_prompt=10, num_computed=12, 1 token).
    # Request 1: mixed past prompt at pos 4..5 (num_prompt=4, num_computed=4).
    req0_tokens = list(range(10, 23))  # 13 tokens (0..12), token[12]=22 from t-1
    req1_tokens = list(range(40, 46))  # 6 tokens (0..5), tokens[4:6]=[44, 45]
    token_ids_cpu = np.zeros((4, 32), dtype=np.int32)
    token_ids_cpu[0, :12] = req0_tokens[:12]  # token 12 not in token_ids_cpu yet
    token_ids_cpu[1, :6] = req1_tokens

    # Runner input_ids_cpu has -1 placeholder for req0's async token at index 0.
    input_ids_cpu = torch.tensor([-1, 44, 45, 0, 0, 0, 0, 0], dtype=torch.int32)
    positions_cpu = torch.tensor([12, 4, 5, 0, 0, 0, 0, 0], dtype=torch.int64)
    qsl_cpu = torch.tensor([0, 1, 3, 3, 3], dtype=torch.int32)
    # Runner's default lookback_token_ids only filled prompt positions (< num_prompt).
    stale_lookback = torch.full((4, DEPTH), -1, dtype=torch.int32)

    pre_async = types.SimpleNamespace(
        req_id_to_index_copy={"req-0": 0},
        wait_for_copy=lambda: None,
        copy_state=types.SimpleNamespace(
            chunks_cpu=[torch.tensor([[22]], dtype=torch.int32)]
        ),
        spec_decode_num_rejected_tokens=None,
    )
    runner = types.SimpleNamespace(
        _last_sequence_layout_plan=types.SimpleNamespace(
            kind=SequenceLayoutKind.ALL, start_index=0
        ),
        input_batch=types.SimpleNamespace(
            num_prompt_tokens=np.array([10, 4]),
            num_computed_tokens_cpu=np.array([12, 4]),
            token_ids_cpu=token_ids_cpu,
            req_ids=["req-0", "req-1"],
        ),
        seq_lens_np=np.array([13, 6]),
        input_ids_cpu=input_ids_cpu,
        positions_cpu=positions_cpu,
        query_start_loc_cpu=qsl_cpu,
        lookback_token_ids=types.SimpleNamespace(cpu=stale_lookback),
        _pre_async_results=pre_async,
        speculative_config=None,
    )

    captured = capture_host_engram_inputs(runner, 2)
    assert captured is not None
    assert not captured.all_prompt
    assert captured.can_hash_on_host
    assert captured.input_ids[:3].tolist() == [22, 44, 45]
    assert captured.lookback_token_ids[0].tolist() == [21, 20, 19, 18]
    assert captured.lookback_token_ids[1].tolist() == [43, 42, 41, 40]

    real = 3
    dead = torch.zeros_like(captured.input_ids, dtype=torch.bool)
    dead_lb = torch.zeros_like(captured.lookback_token_ids, dtype=torch.bool)
    host_hashes = _state(False).hash_on_host(
        captured.input_ids,
        captured.positions,
        captured.query_start_loc,
        dead,
        captured.lookback_token_ids,
        dead_lb,
    )
    dev_hashes = _state(False).forward(
        captured.input_ids,
        captured.positions,
        captured.query_start_loc,
        dead,
        captured.lookback_token_ids,
        dead_lb,
        None,
        torch.arange(16).reshape(4, 4),
    )
    assert torch.equal(host_hashes[:real], dev_hashes[:real])


def test_multi_chunk_execute_model_uses_attn_metadata_builder_ctx():
    """Chunk 1 (start_index=1, chunk_num_reqs=1) slices its own request and lookback."""
    token_ids_cpu = np.zeros((2, 16), dtype=np.int32)
    token_ids_cpu[0, :6] = [10, 11, 12, 13, 14, 15]
    token_ids_cpu[1, :8] = [50, 51, 52, 53, 54, 55, 56, 57]

    # Chunk 1 has 1 request (index 1 in input_batch) with 3 prompt tokens (pos 4..6).
    runner = types.SimpleNamespace(
        _last_sequence_layout_plan=types.SimpleNamespace(kind=SequenceLayoutKind.ALL),
        _attn_metadata_builder_ctx=types.SimpleNamespace(start_index=1, num_reqs=1),
        input_batch=types.SimpleNamespace(
            num_prompt_tokens=np.array([5, 10]),
            num_computed_tokens_cpu=np.array([6, 4]),
            token_ids_cpu=token_ids_cpu,
            req_ids=["req-0", "req-1"],
        ),
        seq_lens_np=np.array([7, 0]),  # chunk 1 seq_len = 4 + 3 = 7 <= 10 (prompt)
        input_ids_cpu=torch.tensor([54, 55, 56, 0], dtype=torch.int32),
        positions_cpu=torch.tensor([4, 5, 6, 0], dtype=torch.int64),
        query_start_loc_cpu=torch.tensor([0, 3, 3], dtype=torch.int32),
        lookback_token_ids=types.SimpleNamespace(
            cpu=torch.full((2, DEPTH), -1, dtype=torch.int32)
        ),
        _pre_async_results=types.SimpleNamespace(
            req_id_to_index_copy={"req-0": 0},
            wait_for_copy=lambda: None,
            copy_state=types.SimpleNamespace(
                chunks_cpu=[torch.tensor([[99]], dtype=torch.int32)]
            ),
            spec_decode_num_rejected_tokens=None,
        ),
        speculative_config=None,
    )

    # forward_model passes full input_batch.num_reqs=2, while the chunk has
    # num_reqs=1 at start_index=1.
    captured = capture_host_engram_inputs(runner, 2)
    assert captured is not None
    assert captured.all_prompt is True
    assert captured.can_hash_on_host is True
    assert captured.input_ids[:3].tolist() == [54, 55, 56]
    assert captured.lookback_token_ids[0].tolist() == [53, 52, 51, 50]


@pytest.mark.cpu_test
def test_host_dp_slot_uses_snapshot_without_device_scalar_read(monkeypatch):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import host_engram

    # The Engram subgroup covers global ranks 2 and 3, including an idle rank.
    context = types.SimpleNamespace(
        additional_kwargs={"tpu_num_tokens_across_dp_cpu": (8192, 8192, 0, 2048)},
        dp_metadata=object(),
    )
    monkeypatch.setattr(host_engram, "get_forward_context", lambda: context)
    monkeypatch.setattr(
        host_engram, "get_dp_group", lambda: types.SimpleNamespace(rank_in_group=3)
    )
    monkeypatch.setattr(
        host_engram,
        "get_engram_dp_group",
        lambda: types.SimpleNamespace(rank_in_group=1, world_size=2),
    )

    def device_read():
        raise AssertionError("Host preparation must not read a device scalar")

    monkeypatch.setattr(host_engram, "upstream_engram_gathered_num_tokens", device_read)
    assert host_engram.engram_gathered_num_tokens() == 2048

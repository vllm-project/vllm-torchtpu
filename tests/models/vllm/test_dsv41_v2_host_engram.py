# SPDX-License-Identifier: Apache-2.0
"""Host-side Engram hashing matches the device path on real rows."""

import types

import numpy as np
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
    # Request 1 is decoding (seq_len past its prompt): host ids may be
    # async placeholders, so the device path must be used.
    assert not capture_host_engram_inputs(_runner([8, 4], [20, 3]), 2).all_prompt
    other = next(k for k in SequenceLayoutKind if k is not SequenceLayoutKind.ALL)
    assert capture_host_engram_inputs(_runner([8], [20], kind=other), 1) is None

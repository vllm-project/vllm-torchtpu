# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the aggregated (mega-agg) K3 program's host logic: the agg
role and its checks, the scheduler's held-back last prompt token, the step
split between the programs, and the prefill's page plan for a decoder under
attention DP."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_torchtpu import hold_back_last_prompt_token
from vllm_torchtpu.megakernel import backend
from vllm_torchtpu.megakernel import program as mk
from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
from vllm_torchtpu.megakernel.kimi_k3 import program as k3
from vllm_torchtpu.megakernel.kimi_k3.agg import agg_configs, split_rows
from vllm_torchtpu.megakernel.kimi_k3.prefill import rank_page_plan


# ---------------------------------------------------------------- role / check
def _vllm_config(kv=None, extra=None):
    return SimpleNamespace(
        load_config=SimpleNamespace(
            load_format="megakernel",
            model_loader_extra_config=extra or {"presharded": "/p"},
        ),
        kv_transfer_config=kv,
        model_config=None,
        speculative_config=None,
    )


def test_role_without_kv_transfer_is_agg():
    config = _vllm_config(extra={"presharded": "/p", "attention_dp": 2})
    assert backend.role(config) == "agg"
    assert backend.attention_dp(config) == 2
    assert backend.draft(config) is None
    assert backend.draft_ring_rows(config) == 0


def test_role_with_kv_transfer_is_unchanged():
    producer = SimpleNamespace(is_kv_producer=True, is_kv_consumer=False)
    consumer = SimpleNamespace(is_kv_producer=False, is_kv_consumer=True)
    assert backend.role(_vllm_config(producer)) == "prefill"
    assert backend.role(_vllm_config(consumer)) == "decode"
    both = SimpleNamespace(is_kv_producer=True, is_kv_consumer=True)
    with pytest.raises(ValueError):
        backend.role(_vllm_config(both))


def test_agg_rejects_prefill_only_loader_keys():
    with pytest.raises(ValueError):
        backend.role(_vllm_config(extra={"presharded": "/p", "draft": "/d"}))


def _agg(**fields):
    base = dict(
        role="agg",
        tensor_parallel_size=32,
        decode_context_parallel_size=16,
        block_size=512,
        kv_cache_dtype="fp8",
        cp_kv_cache_interleave_size=512,
        max_num_seqs=128,
        max_model_len=16384,
        max_num_batched_tokens=8192 + 128,
        async_scheduling=True,
        attention_dp=2,
    )
    base.update(fields)
    return MegakernelEngineConfig(**base)


def test_agg_check_accepts_the_serving_config():
    config = _agg()
    k3.check(config)
    caps = k3.capabilities(config)
    assert caps.roles == frozenset({"agg"})
    assert caps.page_aligned_chunks and not caps.block_copies
    assert caps.max_step_tokens == 8192 + 128


@pytest.mark.parametrize(
    "fields",
    [
        dict(max_num_batched_tokens=8192),  # no room for the decode batch
        dict(async_scheduling=False),
        dict(decode_context_parallel_size=32),  # DCP must be 32 / attention_dp
        dict(mamba_ssm_cache_dtype="bfloat16"),  # the prefill writes FP32 state
        dict(num_speculative_tokens=3),
    ],
)
def test_agg_check_rejects(fields):
    with pytest.raises(NotImplementedError):
        k3.check(_agg(**fields))


def test_agg_configs_split_the_roles():
    prefill, decode = agg_configs(_agg())
    assert prefill.role == "prefill" and prefill.attention_dp == 1
    assert prefill.decode_context_parallel_size == 32
    assert prefill.max_num_batched_tokens == k3.MAX_STEP_TOKENS
    assert decode.role == "decode" and decode.attention_dp == 2
    assert decode.decode_context_parallel_size == 16


# ---------------------------------------------------------------- hold-back
@pytest.mark.parametrize(
    "start,new,prompt,expected",
    [
        (0, 8192, 8192, 8191),  # whole prompt: stop before its last token
        (8191, 1, 8192, 1),  # the last prompt token alone (the decoder's)
        (0, 1024, 8192, 1024),  # a chunk well before it
        (4096, 4096, 8192, 4095),
        (0, 1, 1, 1),  # a one-token prompt: nothing to hold back
        (8192, 1, 8192, 1),  # decoding
        (0, 300, 200, 199),  # a chunk running past the prompt
    ],
)
def test_hold_back_last_prompt_token(start, new, prompt, expected):
    assert hold_back_last_prompt_token(start, new, prompt) == expected


# ---------------------------------------------------------------- step split
def _row(req, computed, scheduled, prompt, num_tokens=None):
    num_tokens = num_tokens if num_tokens is not None else max(prompt, computed + 1)
    null = np.zeros(1, np.int32)
    return mk.RequestInput(
        req_id=req,
        computed=computed,
        scheduled=scheduled,
        token_ids=np.zeros(num_tokens, np.int32),
        num_tokens=num_tokens,
        num_prompt_tokens=prompt,
        block_tables=(null, null, null, null),
    )


def test_split_rows_routes_prompt_chunks_to_the_prefill():
    rows = [
        _row("chunk", 0, 8191, 8192),  # prompt minus its last token
        _row("first-decode", 8191, 1, 8192),  # the held-back last token
        _row("decoding", 9000, 1, 8192, num_tokens=9001),
        _row("idle", 100, 0, 8192),
        _row("partial", 0, 4096, 8192),
    ]
    prefill, decode = split_rows(rows)
    assert [r.req_id for r in prefill] == ["chunk", "partial"]
    # The prefill sees a prompt one token shorter, so it finishes at 8191.
    assert all(r.num_prompt_tokens == 8191 for r in prefill)
    assert prefill[0].end == prefill[0].num_prompt_tokens
    assert [r.req_id for r in decode] == [r.req_id for r in rows]
    assert [r.scheduled for r in decode] == [0, 1, 1, 0, 0]


def test_split_rows_rejects_a_chunk_across_the_last_prompt_token():
    with pytest.raises(RuntimeError):
        split_rows([_row("bad", 0, 8192, 8192)])


def test_split_rows_one_token_prompt_goes_to_the_decoder():
    prefill, decode = split_rows([_row("tiny", 0, 1, 1)])
    assert prefill == [] and decode[0].scheduled == 1


# ---------------------------------------------------------------- page plan
def _original_plan(row, first, pages, skip, cp_rank):
    """``KimiPrefiller._cp_operands`` before the attention DP layout."""
    cp = 32
    owned = [int(row[k // cp]) for k in range(cp_rank, first, cp)]
    writes = [
        (p, int(row[(first + p) // cp]))
        for p in range(skip, pages)
        if (first + p) % cp == cp_rank
    ]
    return owned, writes


@pytest.mark.parametrize("first,pages,skip", [(0, 16, 0), (32, 16, 0), (40, 24, 3)])
def test_page_plan_without_dp_is_the_p_d_layout(first, pages, skip):
    row = np.arange(10, 30, dtype=np.int32)
    for cp_rank in range(32):
        assert rank_page_plan(
            row, first, pages, skip, cp_rank=cp_rank, position=cp_rank
        ) == _original_plan(row, first, pages, skip, cp_rank)


@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("first,pages", [(0, 16), (16, 16), (5, 11)])
def test_page_plan_dp2_matches_the_decoders_owners(group, first, pages):
    """Every page has exactly one owner: mesh position ``16 g + k % 16`` (the
    decoder's DCP member, ``mla_dcp``: ``rank % group``), at physical row
    ``id % N`` of logical block ``k // 16``."""
    num_physical, shared = 1000, 100
    logical = np.array([shared + 7, shared + 9, shared + 11], np.int32)
    row = np.concatenate(
        [logical + group * num_physical, np.zeros(5, np.int32)]
    )  # ids g * N + r (megakernel.dp_blocks), null-padded
    writers: dict[int, list[int]] = {}
    prefix_owner: dict[int, int] = {}
    for position in range(32):
        owned, writes = rank_page_plan(
            row,
            first,
            pages,
            0,
            cp_rank=position % 32,
            position=position,
            groups=2,
            num_physical=num_physical,
        )
        if position // 16 != group:
            assert owned == [] and writes == []
            continue
        member = position % 16
        assert owned == [int(logical[k // 16]) for k in range(member, first, 16)]
        for k in range(member, first, 16):
            assert k not in prefix_owner
            prefix_owner[k] = position
        for p, block in writes:
            k = first + p
            assert k % 16 == member
            assert block == int(logical[k // 16])  # physical row, group stripped
            writers.setdefault(k, []).append(position)
    assert sorted(writers) == list(range(first, first + pages))
    assert all(len(w) == 1 and w[0] == 16 * group + k % 16 for k, w in writers.items())
    assert sorted(prefix_owner) == list(range(first))


def test_page_plan_dp_needs_the_physical_block_count():
    with pytest.raises(ValueError):
        rank_page_plan(np.ones(2, np.int32), 0, 2, 0, cp_rank=0, position=0, groups=2)


def test_engine_config_is_a_dataclass():
    # agg_configs relies on dataclasses.replace over MegakernelEngineConfig.
    assert dataclasses.is_dataclass(MegakernelEngineConfig)

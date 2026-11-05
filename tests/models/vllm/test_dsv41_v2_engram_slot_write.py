# SPDX-License-Identifier: Apache-2.0
"""TPU_DEEPSEEK_V41_ENGRAM_INT32_SLOT_WRITE: same slots and cache contents."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import cache as v2_cache
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import engram as v2_engram
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention import _cache_slots

REQS, TOKENS, PAGE, COLS, PAGES = 48, 288, 64, 64, 8192


def _metadata(device, live_tokens=TOKENS - 6):
    gen = torch.Generator().manual_seed(0)
    seq = torch.randint(PAGE, PAGE * COLS, (REQS,), generator=gen)
    positions = torch.cat([torch.arange(s - 6, s) for s in seq.tolist()]).int()
    qsl = torch.arange(0, TOKENS + 1, 6, dtype=torch.int32)
    qsl[-1] = live_tokens
    tables = torch.randperm(PAGES, generator=gen)[: REQS * COLS].int()
    metadata = SimpleNamespace(
        block_tables=tables.to(device),
        seq_lens=seq.int().to(device),
        query_start_loc=qsl.to(device),
    )
    return metadata, positions.to(device)


@pytest.mark.parametrize("onehot", [True, False])
def test_int32_slots_match_int64(device, monkeypatch, onehot):
    monkeypatch.setattr("vllm_torchtpu.envs.TPU_ONEHOT_TABLE_LOOKUP", onehot)
    metadata, positions = _metadata(device)
    kv_cache = torch.empty((PAGES, PAGE), dtype=torch.int8, device=device)
    ref = _cache_slots(kv_cache, metadata, positions)
    out = v2_engram.engram_cache_slots(kv_cache, metadata, positions)
    assert out.dtype == torch.int32
    assert torch.equal(out.cpu().long(), ref.cpu())
    assert (out.cpu()[-6:] == -1).all()


def test_int32_slot_write_matches_write_cache(device, monkeypatch):
    if device.type != "tpu":
        pytest.skip("the cache write op runs on TPU")
    from vllm_torchtpu.models.vllm import vllm_model_wrapper_context

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("tp",))
    monkeypatch.setattr(
        vllm_model_wrapper_context,
        "get_vllm_model_wrapper_context",
        lambda: SimpleNamespace(mesh=mesh),
    )
    v2_cache.initialize_cache_ops()
    metadata, positions = _metadata(device)
    kv_cache = torch.empty((PAGES, PAGE), dtype=torch.int8, device=device)
    gen = torch.Generator().manual_seed(1)
    token_map = torch.randint(0, 1 << 20, (4096,), generator=gen).int().to(device)
    input_ids = torch.randint(0, 4096, (TOKENS,), generator=gen).int().to(device)
    dead = (torch.rand(TOKENS, generator=gen) < 0.1).to(device)
    initial = torch.randint(0, 1 << 20, (PAGES * PAGE,), generator=gen).int()

    ref_cache = initial.to(device)
    slots = _cache_slots(kv_cache, metadata, positions)
    mapped = token_map[input_ids.long()].masked_fill(dead, v2_engram.upstream.DEAD_ID)
    v2_cache.write_cache(ref_cache, slots, mapped, slots >= 0)

    state = SimpleNamespace(token_map=token_map)
    new_cache = initial.to(device)
    slots32 = v2_engram.engram_cache_slots(kv_cache, metadata, positions)
    v2_engram.NgramHashState.write_token_ids(
        state, new_cache, slots32, input_ids, dead
    )
    assert torch.equal(new_cache.cpu(), ref_cache.cpu())
    assert not torch.equal(new_cache.cpu(), initial)

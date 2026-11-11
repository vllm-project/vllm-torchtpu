# SPDX-License-Identifier: Apache-2.0
"""TPU_DEEPSEEK_V41_ENGRAM_FUSED_PREP: one op call, same hashes/mask/cache."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import torch
from vllm.models.deepseek_v41.common import engram as upstream
from vllm.models.deepseek_v41.common.mm_preprocess import (
    IMAGE_SENTINEL_BASE_ID,
    image_sentinel_mask,
)

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import cache as v2_cache
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import engram as v2_engram

VOCAB, COMPRESSED, BLOCK, PAGES, COLS = 129280, 99092, 64, 2048, 1600


def _state(device):
    config = SimpleNamespace(
        engram_layer_ids=[1, 14],
        engram_num_embeddings=[384006168, 384016682],
        engram_max_ngram_size=4,
        engram_vocab_size=16000000,
        engram_n_heads=8,
        engram_head_dim=256,
        engram_compressed_vocab_size=COMPRESSED,
        engram_pad_token_id=2,
    )
    layout = upstream.EngramLayout(config)
    rng = np.random.default_rng(0)
    token_map = torch.tensor(rng.integers(0, COMPRESSED, VOCAB), dtype=torch.int32)
    state = v2_engram.NgramHashState.__new__(v2_engram.NgramHashState)
    torch.nn.Module.__init__(state)
    state.token_map = token_map.to(device)
    state.multipliers = upstream.compute_hash_multipliers(
        layout.layer_ids, layout.max_ngram_size, COMPRESSED
    ).to(device)
    state.primes = torch.as_tensor(layout.primes).to(device)
    state.offsets = torch.as_tensor(layout.offsets).to(device)
    state.pad_id = int(token_map[2])
    state.lookback_depth, state.block_size, state.use_slot_cache = 3, BLOCK, True
    cache = rng.integers(0, COMPRESSED, PAGES * BLOCK).astype(np.int32)
    state._cache = torch.from_numpy(cache).to(device)
    return state


def _inputs(device, requests, per_request, padding):
    rng = np.random.default_rng(requests * 31 + padding)
    count = requests * per_request + padding
    lengths = rng.integers(1, per_request + 1, requests)
    starts = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    positions = np.zeros(count, np.int32)
    for r in range(requests):
        first = rng.integers(0, (COLS - 1) * BLOCK)
        positions[starts[r] : starts[r + 1]] = first + np.arange(lengths[r])
    ids = rng.integers(0, VOCAB, count)
    ids[rng.random(count) < 0.05] = IMAGE_SENTINEL_BASE_ID
    lookback = rng.integers(0, VOCAB, (requests, 3))
    lookback[rng.random(lookback.shape) < 0.1] = IMAGE_SENTINEL_BASE_ID
    lookback[rng.random(lookback.shape) < 0.1] = -1
    tables = rng.integers(0, PAGES, requests * COLS)
    arrays = dict(
        input_ids=ids,
        positions=positions,
        query_start_loc=starts,
        lookback=lookback,
        block_tables=tables,
    )
    data = {k: torch.tensor(v, dtype=torch.int32).to(device) for k, v in arrays.items()}
    data["seq_lens"] = torch.full((requests,), 1, dtype=torch.int32, device=device)
    return data


@pytest.mark.parametrize("tiles", [True, False])
@pytest.mark.parametrize("requests,per_request,padding", [(48, 6, 0), (5, 6, 7)])
def test_fused_prep_matches_separate_calls(
    device, monkeypatch, tiles, requests, per_request, padding
):
    if device.type != "tpu":
        pytest.skip("the fused prep op runs on TPU")
    from vllm_torchtpu.models.vllm import vllm_model_wrapper_context

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("tp",))
    monkeypatch.setattr(
        vllm_model_wrapper_context,
        "get_vllm_model_wrapper_context",
        lambda: SimpleNamespace(mesh=mesh),
    )
    for name in (
        "FUSED_HASH",
        "MASKS_IN_HASH",
        "INT32_SLOT_WRITE",
        "FUSED_PREP",
    ):
        monkeypatch.setattr(f"vllm_torchtpu.envs.TPU_DEEPSEEK_V41_ENGRAM_{name}", True)
    monkeypatch.setattr(
        "vllm_torchtpu.envs.TPU_DEEPSEEK_V41_ENGRAM_TILE_SCATTER", tiles
    )
    v2_cache.initialize_cache_ops()
    separate, fused = _state(device), _state(device)
    data = _inputs(device, requests, per_request, padding)
    kv_cache = torch.empty((PAGES, BLOCK), dtype=torch.int8, device=device)
    metadata = SimpleNamespace(
        block_tables=data["block_tables"],
        seq_lens=data["seq_lens"],
        query_start_loc=data["query_start_loc"],
    )
    for _ in range(2):
        ids, positions = data["input_ids"], data["positions"]
        ref = separate(
            ids,
            positions,
            data["query_start_loc"],
            None,
            data["lookback"],
            None,
            v2_engram.engram_cache_slots(kv_cache, metadata, positions),
            data["block_tables"].reshape(requests, -1),
        )
        hashes, keep = fused.fused_prep(
            ids,
            positions,
            data["query_start_loc"],
            data["lookback"],
            data["block_tables"],
            requests,
            BLOCK,
        )
        assert torch.equal(hashes.cpu(), ref.cpu())
        assert torch.equal(keep.cpu(), (~image_sentinel_mask(ids)).cpu())
        assert torch.equal(fused._cache.cpu(), separate._cache.cpu())
        data["positions"] = positions + per_request

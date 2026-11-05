# SPDX-License-Identifier: Apache-2.0
"""The fused Engram hash equals the int64 reference bit for bit."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.models.deepseek_v41.common import engram as upstream

from vllm_torchtpu.kernels.engram_hash import engram_hash, hash_tables
from vllm_torchtpu.models.vllm.deepseek_v41.common.engram import NgramHashState

VOCAB, COMPRESSED, BLOCK = 129280, 99092, 64


@pytest.fixture(scope="module")
def tables():
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
    multipliers = upstream.compute_hash_multipliers(
        layout.layer_ids, layout.max_ngram_size, COMPRESSED
    )
    primes = torch.tensor(layout.primes)
    return token_map, multipliers, primes, layout.offsets, int(token_map[2])


def _inputs(rng, requests, per_request, padding, use_cache):
    count = requests * per_request + padding
    lengths = rng.integers(0, per_request + 1, requests)
    lengths[: requests // 2] = per_request
    starts = np.concatenate(([0], np.cumsum(lengths)))
    positions = np.zeros(count, np.int32)
    for r in range(requests):
        first = rng.integers(0, 100000)
        positions[starts[r] : starts[r + 1]] = first + np.arange(lengths[r])
    positions[: min(3, count)] = np.arange(min(3, count))  # a request near position 0
    lookback = rng.integers(0, VOCAB, (requests, 3)).astype(np.int32)
    lookback[rng.random(lookback.shape) < 0.2] = -1
    cache = rng.integers(0, COMPRESSED, 4096 * BLOCK).astype(np.int32)
    cache[rng.random(cache.shape) < 0.05] = upstream.DEAD_ID
    return dict(
        input_ids=rng.integers(0, VOCAB, count).astype(np.int32),
        positions=positions,
        query_start_loc=np.concatenate((starts, [starts[-1]] * 0)).astype(np.int32),
        dead_mask=rng.random(count) < 0.05,
        lookback_token_ids=lookback,
        lookback_dead_mask=rng.random(lookback.shape) < 0.05,
        cache=cache,
        block_table=rng.integers(0, 4096, (requests, 1600)).astype(np.int32),
        use_cache=use_cache,
    )


def _reference(tables, data):
    token_map, multipliers, primes, offsets, pad_id = tables
    state = SimpleNamespace(lookback_depth=3, pad_id=pad_id)
    cache = torch.from_numpy(data["cache"])
    block_table = torch.from_numpy(data["block_table"])

    def fallback(past, reqs, window):
        if not data["use_cache"]:
            return torch.full_like(window, pad_id)
        logical = past.clamp(min=0, max=block_table.shape[1] * BLOCK - 1).long()
        blocks = block_table[reqs, logical // BLOCK].long()
        slots = (blocks * BLOCK + logical % BLOCK).clamp(min=0, max=cache.numel() - 1)
        return cache[slots]

    return NgramHashState._hash(
        state,
        (token_map, multipliers, primes, offsets),
        torch.from_numpy(data["input_ids"]),
        torch.from_numpy(data["positions"]),
        torch.from_numpy(data["query_start_loc"]),
        torch.from_numpy(data["dead_mask"]),
        torch.from_numpy(data["lookback_token_ids"]),
        torch.from_numpy(data["lookback_dead_mask"]),
        fallback,
    ).numpy()


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize(
    "requests,per_request,padding", [(48, 6, 0), (5, 6, 3), (1, 1, 0), (7, 3, 11)]
)
def test_fused_hash_matches_reference(
    tables, use_cache, requests, per_request, padding
):
    rng = np.random.default_rng(requests * 100 + per_request + padding)
    data = _inputs(rng, requests, per_request, padding, use_cache)
    token_map, multipliers, primes, offsets, pad_id = tables
    limbs, primes32, offsets32 = hash_tables(
        token_map.numpy(), multipliers.numpy(), primes.numpy(), offsets.numpy(), pad_id
    )
    fused = engram_hash(
        jnp.asarray(token_map.numpy()),
        jnp.asarray(limbs),
        jnp.asarray(primes32),
        jnp.asarray(offsets32),
        jnp.asarray(data["input_ids"]),
        jnp.asarray(data["positions"]),
        jnp.asarray(data["query_start_loc"]),
        jnp.asarray(data["dead_mask"]),
        jnp.asarray(data["lookback_token_ids"]),
        jnp.asarray(data["lookback_dead_mask"]),
        jnp.asarray(data["cache"]),
        jnp.asarray(data["block_table"]),
        lookback_depth=3,
        block_size=BLOCK,
        pad_id=pad_id,
        dead_id=upstream.DEAD_ID,
        use_cache=use_cache,
    )
    np.testing.assert_array_equal(np.asarray(fused), _reference(tables, data))


def test_hash_tables_reject_out_of_range(tables):
    token_map, multipliers, primes, offsets, pad_id = tables
    with pytest.raises(ValueError):
        hash_tables(
            token_map.numpy(),
            multipliers.numpy(),
            primes.numpy() * 2,
            offsets.numpy(),
            pad_id,
        )


@pytest.mark.parametrize("use_cache", [False, True])
def test_state_fused_hash_matches_state_hash(tables, monkeypatch, use_cache):
    # The NgramHashState plumbing (tables, statics, operand order) of the fused
    # path, with the TPU op replaced by a direct call of the JAX function.
    from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as common

    token_map, multipliers, primes, offsets, pad_id = tables
    rng = np.random.default_rng(7)
    data = _inputs(rng, 48, 6, 0, use_cache)
    state = NgramHashState.__new__(NgramHashState)
    torch.nn.Module.__init__(state)
    state.token_map, state.multipliers, state.primes = token_map, multipliers, primes
    state.offsets, state.pad_id = offsets, token_map[2]
    state.lookback_depth, state.block_size = 3, BLOCK
    state.use_slot_cache = use_cache
    state._cache = torch.from_numpy(data["cache"]) if use_cache else None

    def direct(*args):
        arrays = [jnp.asarray(a.numpy()) for a in args[:12]]
        depth, block_size, pad, dead, cache_on = args[12:]
        return torch.from_numpy(
            np.asarray(
                engram_hash(
                    *arrays,
                    lookback_depth=depth,
                    block_size=block_size,
                    pad_id=pad,
                    dead_id=dead,
                    use_cache=cache_on,
                )
            )
        )

    monkeypatch.setattr(common, "_fused_hash_op", lambda: direct)
    fused = state._fused_hash(
        torch.from_numpy(data["input_ids"]).long(),
        torch.from_numpy(data["positions"]),
        torch.from_numpy(data["query_start_loc"]),
        torch.from_numpy(data["dead_mask"]),
        torch.from_numpy(data["lookback_token_ids"]),
        torch.from_numpy(data["lookback_dead_mask"]),
        torch.from_numpy(data["block_table"]),
    )
    assert fused.dtype == torch.int32
    np.testing.assert_array_equal(fused.numpy(), _reference(tables, data))

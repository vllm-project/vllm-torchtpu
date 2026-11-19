# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_PREFILL_QUERY_TILE: a query subtile for the paired-page prefill tiles, bitwise.

Query rows are independent, so splitting a query block into subtiles that share
one decoded KV tile must leave every output bit unchanged; checked on a packed
multi-request step (unaligned requests, own pages). The flag selects the
subtile only for paired-page entries that leave it to the kernel. Off TPU the
Pallas kernels run in interpret mode.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as bsp

RATIO, TOPK, PAGE = 4, 512, 3200


@pytest.fixture
def kernels(monkeypatch):
    if jax.default_backend() != "tpu":
        from jax._src import tpu_info

        monkeypatch.setitem(
            tpu_info.registry,
            jax.devices()[0].device_kind,
            lambda: tpu_info._get_tpu_info_impl(tpu_info.ChipVersion.TPU_7X, 1),
        )
        original = pl.pallas_call

        def interpreted(*args, **kwargs):
            kwargs["interpret"] = True
            return original(*args, **kwargs)

        monkeypatch.setattr(bsp.pl, "pallas_call", interpreted)
    yield
    jax.clear_caches()


def _packed_step(seed, rows, cached):
    """One 3,200-token step of len(rows) requests; each request owns pages of a permuted pool."""
    rng = np.random.default_rng(seed)
    pages_per_seq = max(-(-(c + n) // PAGE) for n, c in zip(rows, cached))
    pool_pages = pages_per_seq * len(rows)
    tokens = int(sum(rows))
    starts = np.concatenate([[0], np.cumsum(rows)]).astype(np.int32)
    owner = rng.permutation(pool_pages)
    lengths = np.zeros(tokens, np.int32)
    blocks = np.full((tokens, TOPK), -1, np.int32)
    table = np.zeros((tokens, pages_per_seq), np.int32)
    for r, (n, ctx) in enumerate(zip(rows, cached)):
        pages = owner[r * pages_per_seq:(r + 1) * pages_per_seq]
        for i in range(n):
            t = starts[r] + i
            length = ctx + i + 1
            lengths[t] = length
            table[t] = pages
            complete = length // RATIO
            if complete:
                local = list(range(max(0, complete - 32), complete))
                rest = rng.permutation(complete)[: TOPK - len(local)]
                pick = np.unique(np.concatenate([local, rest]))[:TOPK]
                blocks[t, : len(pick)] = pick
    q = jnp.asarray(rng.normal(0, 0.5, (tokens, 24, 256)), jnp.bfloat16)
    gate = jnp.asarray(rng.uniform(-3, 3, (tokens, 24, 256)), jnp.bfloat16)
    raw = rng.integers(0, 0x40, size=(pool_pages, PAGE, 1, 4, 256), dtype=np.uint8)
    raw |= rng.integers(0, 2, size=raw.shape, dtype=np.uint8) << 7
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    return q, gate, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table), pool, jnp.asarray(starts)


from vllm_torchtpu.kernels.qwen4_exp import prefill_tiles


@pytest.mark.parametrize("query_block,query_tile", [(256, 64), (256, 32), (128, 64)])
def test_query_subtile_is_bitwise_on_packed_steps(kernels, query_block, query_tile):
    rows, cached = (37, 300, 129, 1000, 511, 223, 1000), (0, 1500, 4000, 700, 2900, 100, 5000)
    q, gate, blocks, lengths, table, pool, starts = _packed_step(7, rows, cached)

    def run(tile):
        jax.clear_caches()
        out, _ = bsp.block_sparse_gqa_attention(
            q, gate, blocks, lengths, table, pool, starts, kv_heads=2, ratio=RATIO,
            query_block=query_block, kv_tile=512, **({"query_tile": tile} if tile else {}),
        )
        return np.asarray(out)

    whole, split = run(None), run(query_tile)
    assert np.abs(whole.astype(np.float32)).max() > 0
    np.testing.assert_array_equal(split.view(np.uint16), whole.view(np.uint16))


@pytest.mark.parametrize("flag", [0, 64])
def test_flag_selects_the_subtile_only_for_paired_entries(kernels, monkeypatch, flag):
    from vllm_torchtpu import envs

    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFILL_QUERY_TILE", flag, raising=False)
    # the default-tile branch logs through vLLM's logger; keep the test independent of vLLM
    monkeypatch.setattr(
        prefill_tiles, "logger", type("L", (), {"warning_once": staticmethod(lambda *a, **k: None)})()
    )
    cache = jax.ShapeDtypeStruct((4, PAGE, 1, 4, 256), jnp.float8_e4m3fn)
    tiles = {}
    for tokens in (8192, 6400, 3200):
        query = jax.ShapeDtypeStruct((tokens, 24, 256), jnp.bfloat16)
        tiles[tokens] = tuple(prefill_tiles.get_sparse_prefill_tiles(
            query, cache, kv_heads=2, ratio=RATIO, with_query_tile=True))
    if not flag:
        assert tiles[8192][2] is None and tiles[6400][2] is None
    else:
        # paired Q256/KV512 entries get the subtile; the 3200 bucket's unpaired KV128 fallback does not
        assert tiles[8192] == (256, 512, 64) and tiles[6400] == (256, 512, 64)
        assert tiles[3200][1] == 128 and tiles[3200][2] is None


def test_subtile_with_the_q128_bucket_flag(kernels, monkeypatch):
    """With VLLM_TPU_QSA_PREFILL_Q128_FALLBACK on, the 3,200-token bucket runs Q128/KV512 and takes the subtile too."""
    from vllm_torchtpu import envs

    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFILL_QUERY_TILE", 64, raising=False)
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_Q128_FALLBACK", "1")
    monkeypatch.setattr(
        prefill_tiles, "logger", type("L", (), {"warning_once": staticmethod(lambda *a, **k: None)})()
    )
    cache = jax.ShapeDtypeStruct((4, PAGE, 1, 4, 256), jnp.float8_e4m3fn)
    query = jax.ShapeDtypeStruct((3200, 24, 256), jnp.bfloat16)
    tiles = tuple(prefill_tiles.get_sparse_prefill_tiles(query, cache, kv_heads=2, ratio=RATIO, with_query_tile=True))
    assert tiles == (128, 512, 64)

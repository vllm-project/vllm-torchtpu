# SPDX-License-Identifier: Apache-2.0
"""Serving reuses the pipeline's gathered KV with one attention launch per layer."""

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import cache as cache_module
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows


def _swa_spec(block_size=32):
    return SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
    )


def _config(specs, blocks, page_bytes, swa_offset=0):
    return KVCacheConfig(
        num_blocks=blocks,
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec)
            for name, spec in specs.items()
        ],
        kv_cache_tensors=[
            KVCacheTensor(
                size=blocks * page_bytes * (2 if swa_offset else 1),
                block_stride=page_bytes,
                offset=swa_offset if name.startswith("swa") else 0,
                layer_stride=blocks * page_bytes,
                layers=[name],
            )
            for name in specs
        ],
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("split", [False, True])
def test_swa_and_main_share_word_pool(monkeypatch, split):
    blocks, page_size = 8, 32
    page_bytes = page_size * 1024
    specs = {
        "swa": _swa_spec(page_size),
        "main": MLAAttentionSpec(
            block_size=2 * page_size,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
        ),
    }
    if split:
        monkeypatch.setattr(cache_module, "SPARSECORE_MAX_POOL_BYTES", 4 * page_bytes)
    materialized = cache_module.materialize_upstream_kv_cache(
        _config(specs, blocks, page_bytes),
        "cpu",
        KVCacheLayout.LBHNC,
        word_rows=True,
    )
    raw = materialized.native_raw_cache["swa"]
    assert raw is materialized.native_raw_cache["main"]
    banks = raw if isinstance(raw, tuple) else (raw,)
    assert len(banks) == (2 if split else 1)
    assert all(bank.dtype == torch.int32 and bank.shape[1] == 128 for bank in banks)
    assert sum(bank.nbytes for bank in banks) == blocks * page_bytes
    cache = PagedCacheRows(
        raw, materialized.native_cache_layouts["swa"], swa_words=True
    )
    assert cache.shape == (blocks, page_size, 512)
    indices = torch.tensor([1, 3, 4, 7])
    slots = indices * page_size
    values = torch.arange(4 * 512).reshape(4, 512).to(torch.bfloat16)
    cache.write(slots, values)
    torch.testing.assert_close(cache[indices, torch.zeros_like(indices)], values)
    torch.testing.assert_close(cache.row_indices(slots), slots)
    torch.testing.assert_close(cache.pages(indices), indices)
    physical = torch.cat(banks).reshape(-1, 512)
    bits = values.view(torch.int16).to(torch.int32).bitwise_and(65535)
    torch.testing.assert_close(physical[slots // 2], bits)
    assert not physical[: page_size // 2].any()


@pytest.mark.cpu_test
@pytest.mark.parametrize("tokens", [1, 513])
def test_chunk_cache_reuse_and_stage_invalidation(monkeypatch, tokens):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
        attention_kernels,
        csa2_adapter,
    )

    gathered, consumed = [], []

    def gather(query, raw, high, scales, high_scales, selected, *args):
        gathered.append(selected)
        return (
            query * 2,
            raw.new_empty((tokens, 512, 128)),
            scales.new_empty((tokens, 512)),
            (selected >= 0).sum(1).int(),
        )

    def attend(query, words, scales, raw, high, starts, *args):
        consumed.append((words, scales, starts, args[-1]))
        return query * 2

    monkeypatch.setattr(
        csa2_adapter, "_ops", {"gather_attend": gather, "attend": attend}
    )
    raw = torch.zeros((128, 128), dtype=torch.int32)
    main = PagedCacheRows(
        raw, ((4, 32, 512), torch.uint8, 32 * 512, 0), scale=torch.ones(128)
    )
    swa = PagedCacheRows(
        torch.zeros((2 * 32 * 2, 128), dtype=torch.int32),
        ((2, 32, 512), torch.bfloat16, 32 * 1024, 0),
    )
    q = torch.ones((tokens, 8, 512), dtype=torch.bfloat16)
    records = torch.zeros((tokens, 512), dtype=torch.int32)
    starts = torch.tensor([0, min(17, tokens - 1), tokens - 1], dtype=torch.int32)

    def attention(selected):
        return csa2_adapter.attention(
            q,
            swa,
            main,
            selected,
            None,
            None,
            starts,
            torch.arange(tokens),
            torch.zeros(8),
            torch.empty((tokens, 64)),
            torch.empty((2, 4)),
            torch.empty((2, 2)),
            2,
            512**-0.5,
            False,
        )

    with attention_kernels.gathered_kv_cache_scope():
        torch.testing.assert_close(attention(records), q * 2)
        cache = attention_kernels._active_gathered_kv
        assert cache is not None and cache.values is not None
        cached = cache.values
        torch.testing.assert_close(attention(records), q * 2)
        assert len(gathered) == 1
        assert len(consumed) == 1
        words, scales, reused_starts, bq = consumed[0]
        assert words is cached[0] and scales is cached[1]
        torch.testing.assert_close(reused_starts, starts)
        assert bq == (6 if tokens <= 384 else 16)
        attention(records.clone())
        assert len(gathered) == 2
    with attention_kernels.gathered_kv_cache_scope():
        attention(records)
        assert len(gathered) == 3
    assert attention_kernels._active_gathered_kv is None

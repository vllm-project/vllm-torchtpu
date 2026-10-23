# SPDX-License-Identifier: Apache-2.0
"""Serving reuses the pipeline's gathered KV with one attention launch per layer."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels, csa2_adapter
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import cache as cache_module
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows


@pytest.mark.cpu_test
@pytest.mark.parametrize("split", [False, True])
def test_native_swa_allocation_budgets_overlaid_groups_and_binds_pages(
    monkeypatch, split
):
    blocks, page_size = 4, 32
    specs = {
        name: MLAAttentionSpec(
            block_size=page_size, num_kv_heads=1, head_size=512, dtype=dtype
        )
        for name, dtype in (
            ("swa", torch.bfloat16),
            ("swa_alias", torch.bfloat16),
            ("main", torch.uint8),
        )
    }
    swa_layers = frozenset(("swa", "swa_alias"))
    specs = cache_module.budget_native_swa(specs, swa_layers)
    assert cache_module.budget_native_swa(specs, swa_layers) == specs
    page_bytes = (1024 + 512) * page_size
    assert {spec.page_size_bytes for spec in specs.values()} == {page_bytes}
    views = {
        name: torch.empty((blocks, 1, page_size, 512), dtype=spec.dtype, device="meta")
        for name, spec in specs.items()
    }
    monkeypatch.setattr(cache_module, "allocate_upstream_kv_cache", lambda *a: views)
    if split:
        monkeypatch.setattr(
            cache_module, "SPARSECORE_MAX_POOL_BYTES", 2 * page_size * 1024
        )
    config = SimpleNamespace(
        num_blocks=blocks,
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=spec) for spec in specs.values()
        ],
        kv_cache_tensors=[
            SimpleNamespace(
                size=blocks * page_bytes,
                block_stride=page_bytes,
                offset=0,
                layer_stride=0,
                layers=[name],
            )
            for name in specs
        ],
    )
    materialized = cache_module.materialize_upstream_kv_cache(
        config,
        "cpu",
        SimpleNamespace(is_layer_compact=True),
        word_rows=True,
        csa2_layers=frozenset(("main",)),
        native_swa_layers=swa_layers,
    )
    raw = materialized.native_raw_cache["swa"]
    assert raw is materialized.native_raw_cache["swa_alias"]
    banks = raw if isinstance(raw, tuple) else (raw,)
    assert len(banks) == (2 if split else 1)
    assert all(
        bank.dtype == torch.bfloat16 and bank.shape[1:] == (page_size, 512)
        for bank in banks
    )
    assert sum(bank.nbytes for bank in materialized.raw_tensors) == (
        blocks * page_bytes + blocks * page_size * 4
    )
    assert materialized.native_raw_cache["main"].dtype == torch.int32
    cache = PagedCacheRows(raw, materialized.native_cache_layouts["swa"])
    for i, bank in enumerate(banks):
        bank.fill_(i + 1)
    indices = torch.tensor([1, 2, 3])
    expected = torch.tensor([1, 2, 2] if split else [1, 1, 1], dtype=torch.bfloat16)
    torch.testing.assert_close(
        cache[indices, torch.zeros_like(indices)][:, 0], expected
    )
    torch.testing.assert_close(
        cache.row_indices(indices * page_size), indices * page_size
    )
    torch.testing.assert_close(cache.pages(indices), indices)


@pytest.mark.cpu_test
@pytest.mark.parametrize("tokens", [1, 513])
@pytest.mark.parametrize("decoder", [False, True])
def test_chunk_cache_reuse_and_stage_invalidation(monkeypatch, tokens, decoder):
    gathered, consumed = [], []

    def gather(query, raw, high, scales, high_scales, selected, *args):
        assert args[-1] is decoder
        gathered.append(selected)
        return (
            query * 2,
            raw.new_empty((tokens, 512, 128)),
            scales.new_empty((tokens, 512)),
            (selected >= 0).sum(1).int(),
        )

    def attend(query, words, scales, raw, high, starts, *args):
        assert args[-1] is decoder
        consumed.append((words, scales, starts, args[-2]))
        return query * 2

    monkeypatch.setattr(
        csa2_adapter, "_ops", {"gather_attend": gather, "attend": attend}
    )
    raw = torch.zeros((128, 128), dtype=torch.int32)
    main = PagedCacheRows(
        raw, ((4, 32, 512), torch.uint8, 32 * 512, 0), scale=torch.ones(128)
    )
    swa = PagedCacheRows(
        torch.zeros((2, 32, 512), dtype=torch.bfloat16),
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
            decoder=decoder,
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

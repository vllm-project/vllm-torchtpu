# SPDX-License-Identifier: Apache-2.0
"""Serving reuses the pipeline's gathered KV with one attention launch per layer."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_configs,
    unify_kv_cache_spec_page_size,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import cache as cache_module
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows
from vllm_torchtpu.models.vllm.deepseek_v41.v2.swa_cache import (
    DeepseekV41SWASpec,
    register_swa_specs,
    swa_pool_bytes,
)


def _swa_spec(block_size=32, sequences=1, in_flight=32, lookahead=0):
    return DeepseekV41SWASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
        bounded_replay=True,
        max_num_seqs=sequences,
        max_in_flight_tokens=in_flight,
        num_lookahead_tokens=lookahead,
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
def test_swa_owners_hold_pool_pages_and_bind_block_ids(monkeypatch, split):
    """Each SWA layer owns pool_capacity BF16 pages; block IDs index them."""
    blocks, page_size = 16, 32
    swa = _swa_spec(page_size)
    capacity = swa.pool_capacity
    assert capacity == 8 < blocks
    specs = {
        "swa": swa,
        "swa_alias": swa,
        "main": MLAAttentionSpec(
            block_size=page_size, num_kv_heads=1, head_size=512, dtype=torch.uint8
        ),
    }
    page_bytes = 1024 * page_size
    if split:
        monkeypatch.setattr(cache_module, "SPARSECORE_MAX_POOL_BYTES", 4 * page_bytes)
    materialized = cache_module.materialize_upstream_kv_cache(
        _config(specs, blocks, page_bytes),
        "cpu",
        KVCacheLayout.LBHNC,
        word_rows=True,
        csa2_layers=frozenset(("main",)),
    )
    raw = materialized.native_raw_cache["swa"]
    # Layers never alias SWA pages: each request window is per layer.
    assert raw is not materialized.native_raw_cache["swa_alias"]
    banks = raw if isinstance(raw, tuple) else (raw,)
    assert len(banks) == (2 if split else 1)
    assert sum(bank.shape[0] for bank in banks) == capacity
    assert all(
        bank.dtype == torch.bfloat16 and bank.shape[1:] == (page_size, 512)
        for bank in banks
    )
    layout = materialized.native_cache_layouts["swa"]
    assert layout[0] == (capacity, page_size, 512)
    assert materialized.kv_caches["swa"].shape[0] == capacity
    main = materialized.native_raw_cache["main"]
    main_banks = main if isinstance(main, tuple) else (main,)
    assert all(bank.dtype == torch.int32 for bank in main_banks)
    assert sum(bank.nbytes for bank in main_banks) == blocks * page_size * 512
    cache = PagedCacheRows(raw, layout)
    for i, bank in enumerate(banks):
        bank.fill_(i + 1)
    indices = torch.tensor([1, 3, 4, 7])
    expected = torch.tensor(
        [1, 1, 2, 2] if split else [1, 1, 1, 1], dtype=torch.bfloat16
    )
    torch.testing.assert_close(
        cache[indices, torch.zeros_like(indices)][:, 0], expected
    )
    torch.testing.assert_close(
        cache.row_indices(indices * page_size), indices * page_size
    )
    torch.testing.assert_close(cache.pages(indices), indices)


@pytest.mark.cpu_test
def test_swa_pool_admits_million_token_context(record_property):
    """SWA pages leave the shared pool, so its blocks only pay for main KV."""
    KVCacheSpecRegistry.check_kv_cache_spec_registry({})
    register_swa_specs()
    specs: dict[str, KVCacheSpec] = {
        f"swa.{layer}": SlidingWindowMLASpec(
            block_size=16,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            sliding_window=128,
        )
        for layer in range(43)
    }
    swa_layers = [f"swa.{layer}" for layer in range(2, 40)]
    for name in swa_layers:
        specs[name] = _swa_spec(16, sequences=48, in_flight=4096, lookahead=5)
    for layer, ratio in ((2, 2), (8, 2), (14, 2), (20, 1)):
        for kind, width, dtype in (
            ("main", 512, torch.uint8),
            ("index", 128, torch.bfloat16),
        ):
            specs[f"{kind}.{layer}"] = MLAAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=width,
                dtype=dtype,
                tokens_per_state=ratio,
            )
        if ratio == 2:
            specs[f"ring.{layer}"] = CircularBufferSpec(
                block_size=8,
                num_kv_heads=1,
                head_size=1024,
                head_size_v=0,
                dtype=torch.float32,
            )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=1048576, original_max_model_len=1048576
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, decode_context_parallel_size=1
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        attention_config=SimpleNamespace(hisparse_config=None),
        speculative_config=None,
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBHNC,
        ),
        max_in_flight_tokens=4096,
        num_prefill_lookahead_tokens=0,
    )
    unified = unify_kv_cache_spec_page_size(specs)
    reservation = swa_pool_bytes(unified)
    caches = get_kv_cache_configs(config, [unified], [(24 << 30) - reservation])[0]
    record_property("blocks", caches.num_blocks)
    record_property("swa_reserved_bytes", reservation)
    assert caches.num_blocks > 0
    assert unified["swa.2"].block_size == 32
    # Check full-model owner sizing without allocating multi-GiB CPU buffers.
    materialized = cache_module.materialize_upstream_kv_cache(
        caches,
        "meta",
        KVCacheLayout.LBHNC,
        word_rows=True,
        csa2_layers=frozenset(f"main.{layer}" for layer in (2, 8, 14, 20)),
    )
    assert len(materialized.kv_caches) == len(specs)
    assert sum(raw.nbytes for raw in materialized.raw_tensors) <= 30 << 30
    for name in swa_layers:
        spec = unified[name]
        assert isinstance(spec, DeepseekV41SWASpec)
        raw = materialized.native_raw_cache[name]
        assert isinstance(raw, torch.Tensor)
        assert raw.shape[0] == spec.pool_capacity < caches.num_blocks // 10


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

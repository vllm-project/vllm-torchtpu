# SPDX-License-Identifier: Apache-2.0
"""Serving reuses the pipeline's gathered KV with one attention launch per layer."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
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
    page_bytes = 2 * 1024 * page_size
    assert {spec.page_size_bytes for spec in specs.values()} == {page_bytes}
    assert specs["main"].block_size == 2 * page_size
    if split:
        monkeypatch.setattr(
            cache_module, "SPARSECORE_MAX_POOL_BYTES", 2 * page_size * 1024
        )
    config = KVCacheConfig(
        num_blocks=blocks,
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec)
            for name, spec in specs.items()
        ],
        kv_cache_tensors=[
            KVCacheTensor(
                size=blocks * page_bytes,
                block_stride=page_bytes,
                offset=0,
                layer_stride=blocks * page_bytes,
                layers=[name],
            )
            for name in specs
        ],
    )
    materialized = cache_module.materialize_upstream_kv_cache(
        config,
        "cpu",
        KVCacheLayout.LBHNC,
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
        blocks * page_bytes + blocks * specs["main"].num_states * 4
    )
    main = materialized.native_raw_cache["main"]
    main_banks = main if isinstance(main, tuple) else (main,)
    assert all(bank.dtype == torch.int32 for bank in main_banks)
    assert sum(bank.nbytes for bank in main_banks) == blocks * page_size * 1024
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
def test_native_swa_budget_admits_million_token_context(record_property):
    """Typed SWA owners must not erase compressed-page packing at admission."""
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
    native_swa = frozenset(f"swa.{layer}" for layer in range(2, 40))
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
    budgeted = cache_module.budget_native_swa(specs, native_swa)
    assert cache_module.budget_native_swa(budgeted, native_swa) == budgeted
    # Use the real startup admission check from the failing serving run.
    caches = get_kv_cache_configs(config, [budgeted], [24 << 30])
    assert caches[0].num_blocks > 0
    assert config.model_config.max_model_len == 1048576
    assert budgeted["main.2"].block_size == 128
    assert budgeted["index.2"].block_size == 256
    assert budgeted["swa.2"].block_size == 32
    assert budgeted["ring.2"].block_size == 8
    assert cache_module.budget_native_swa(specs, frozenset()) is specs
    # Check full-model owner sizing without allocating multi-GiB CPU buffers.
    materialized = cache_module.materialize_upstream_kv_cache(
        caches[0],
        "meta",
        KVCacheLayout.LBHNC,
        word_rows=True,
        csa2_layers=frozenset(f"main.{layer}" for layer in (2, 8, 14, 20)),
        native_swa_layers=native_swa,
    )
    assert len(materialized.kv_caches) == len(specs)
    assert sum(raw.nbytes for raw in materialized.raw_tensors) <= 30 << 30

    config.cache_config.enable_prefix_caching = False
    config.scheduler_config.max_num_seqs = 48
    config.num_speculative_tokens = 5
    config.max_concurrent_batches = 2
    unified = cache_module.unify_kv_cache_spec_page_size(specs)
    compact = cache_module.NativeSWACache(unified, native_swa, config)
    compact_config = get_kv_cache_configs(
        config, [unified], [(24 << 30) - compact.reserve_bytes]
    )[0]
    record_property("old_blocks", caches[0].num_blocks)
    record_property("compact_blocks", compact_config.num_blocks)
    record_property("swa_reserved_bytes", compact.reserve_bytes)
    assert compact_config.num_blocks > 1.8 * caches[0].num_blocks
    compact_materialized = cache_module.materialize_upstream_kv_cache(
        compact_config,
        "meta",
        KVCacheLayout.LBHNC,
        word_rows=True,
        csa2_layers=frozenset(f"main.{layer}" for layer in (2, 8, 14, 20)),
        native_swa_layers=native_swa,
        compact_swa=compact,
    )
    assert sum(raw.nbytes for raw in compact_materialized.raw_tensors) <= 30 << 30
    for name in native_swa:
        raw = compact_materialized.native_raw_cache[name]
        assert isinstance(raw, torch.Tensor)
        assert raw.shape[0] == compact.layers[name].capacity
        assert raw.shape[0] < compact_config.num_blocks // 10


def _compact_config(sequences=2, tokens=256, speculative=5):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=sequences),
        num_speculative_tokens=speculative,
        max_concurrent_batches=2,
        num_prefill_lookahead_tokens=0,
        max_in_flight_tokens=2 * tokens,
        cache_config=SimpleNamespace(enable_prefix_caching=True),
    )


def _compact_spec(page=32):
    return SlidingWindowMLASpec(
        block_size=page,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
        bounded_replay=True,
    )


@pytest.mark.cpu_test
def test_compact_swa_preserves_windows_through_rejection_and_recycling():
    """Physical pages preserve committed tokens while discarded drafts rewind."""
    window = cache_module.NativeSWAWindow(_compact_spec(), _compact_config())
    storage = torch.full((window.capacity, 32), -1, dtype=torch.int32)
    starts = {"a": 0, "b": 0}
    for step in range(100):
        window.advance(set())
        for req in ("a", "b") if step % 2 else ("b", "a"):
            start = starts[req]
            size = 128 if not start else 6
            pages = window.map_request(req, start, start + size, 0)
            tag = 10000 if req == "b" else 0
            for token in range(max(0, start - 127), start):
                assert storage[pages[token // 32], token % 32] == tag + token
            for token in range(start, start + size):
                storage[pages[token // 32], token % 32] = tag + token
            starts[req] += size if not start else 1 + step % 6
        live = [page for pages in window.requests.values() for page in pages.values()]
        assert len(live) == len(set(live))
        assert 0 not in live
    window.advance({"a"})
    assert "a" not in window.requests
    # A replayed prefix starts in a different logical region of a recycled pool.
    pages = window.map_request("a", 9000, 9128, 9000)
    assert min(pages) == 9000 // 32
    assert set(pages.values()).isdisjoint(window.requests["b"].values())


@pytest.mark.cpu_test
def test_compact_swa_maps_history_before_optimistic_cpu_window():
    window = cache_module.NativeSWAWindow(_compact_spec(), _compact_config())
    storage = torch.full((window.capacity, 32), -1, dtype=torch.int32)
    committed = 0
    optimistic = 0
    for step in range(200):
        window.advance(set())
        size = 128 if not step else 6
        pages = window.map_request("a", optimistic, optimistic + size, 0)
        # Device-side rejection corrects positions after CPU metadata is built.
        for token in range(max(0, committed - 127), committed):
            assert storage[pages[token // 32], token % 32] == token
        for token in range(committed, committed + size):
            storage[pages[token // 32], token % 32] = token
        optimistic = committed + size
        committed += size if not step else 1 + step % 6


@pytest.mark.cpu_test
def test_compact_swa_reservation_depends_on_working_set():
    config = _compact_config(sequences=48, tokens=2048)
    cache = cache_module.NativeSWACache(
        {"swa": _compact_spec()}, frozenset({"swa"}), config
    )
    window = cache.layers["swa"]
    # One shared in-flight budget, not max_batch_tokens for every request.
    assert window.capacity < 512
    assert cache.reserve_bytes < 16 << 20
    for sequence in range(48):
        window.map_request(str(sequence), 100000, 100006, 100000)
    assert len(window.requests) == 48
    config.cache_config.enable_prefix_caching = True
    with pytest.raises(ValueError, match="bounded prefix replay"):
        cache_module.NativeSWACache(
            {"swa": dataclasses.replace(_compact_spec(), bounded_replay=False)},
            frozenset({"swa"}),
            config,
        )


@pytest.mark.cpu_test
def test_compact_swa_metadata_reorders_and_versions_live_pages():
    from vllm.v1.worker.block_table import BlockTable

    spec = _compact_spec()
    config = _compact_config()
    compact = cache_module.NativeSWACache({"swa": spec}, frozenset({"swa"}), config)
    source = BlockTable(32, 2, 512, 256, False, torch.device("cpu"), 32, 1)
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(block_table=[source], req_ids=["a", "b"])
    )
    kv_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["swa"], kv_cache_spec=spec)]
    )
    compact.initialize(runner, kv_config)
    ctx = SimpleNamespace(
        position_ids_override=None,
        num_reqs=2,
        start_index=0,
        query_start_loc_cpu=(0, 6, 12),
        seq_lens_cpu=(10006, 10006),
        replay_starts_cpu=(10000, 10000),
    )
    compact.prepare(runner, ctx)
    table = compact.group_tables[0]
    previous = table.get_cpu_tensor().clone()
    versions = table._row_version.copy()
    compact.prepare(runner, ctx)
    np.testing.assert_array_equal(table._row_version, versions)
    runner.input_batch.req_ids.reverse()
    compact.prepare(runner, ctx)
    torch.testing.assert_close(table.get_cpu_tensor(), previous.flip(0))
    assert (table._row_version > versions).all()
    # Dummy compilation neither allocates pages nor changes request ownership.
    free = list(compact.layers["swa"].free)
    ctx.position_ids_override = torch.zeros(12, dtype=torch.int32)
    compact.prepare(runner, ctx)
    assert compact.layers["swa"].free == free


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

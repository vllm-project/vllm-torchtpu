# SPDX-License-Identifier: Apache-2.0
"""Unified pool sized from profiled HBM, after profiling on a scratch pool.

The pool used to be fixed before profiling at min(max_num_seqs x the blocks
of a max_model_len request, QWEN4_EXP_MAX_POOL_BYTES), so with 32 sequences
it was the 48 GiB default cap. Profiling then compiled its forward with
that pool allocated, and every compiled forward holds a second pool-sized
buffer: a 40 GiB pool failed the profiling compile at 96.89 GiB of 94.74.
"""

from types import SimpleNamespace as NS

import pytest
import torch
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups

from tests.runner.test_qwen4_exp_pool import pool_config
from vllm_torchtpu.models.vllm.qwen4_exp import config as qconfig
from vllm_torchtpu.models.vllm.qwen4_exp import pooled_cache
from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs
from vllm_torchtpu.utils import HbmBudget

GiB = 1024**3
PAGE = 512 * 1024


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "QWEN4_EXP_POOL_SIZING",
        "QWEN4_EXP_POOL_GRAPH_COPIES",
        "QWEN4_EXP_POOL_HEADROOM_BYTES",
        "QWEN4_EXP_MAX_POOL_BYTES",
        "QWEN4_EXP_PACK_COMPRESSED_K",
    ):
        monkeypatch.delenv(name, raising=False)


def test_pool_reserves_headroom_but_no_graph_copy_by_default():
    # Measured after an 8192-token profile on v7x: 15.9 GiB resident.
    blocks = qconfig.pooled_pool_blocks(
        PAGE,
        cap_available=int(87.16 * GiB) - int(15.9 * GiB),
        physical_available=int(94.74 * GiB) - int(15.9 * GiB),
        min_blocks=1,
    )
    assert blocks == (int(87.16 * GiB) - int(15.9 * GiB) - 6 * GiB) // PAGE


def test_without_graph_copies_the_utilization_cap_binds():
    blocks = qconfig.pooled_pool_blocks(
        PAGE, cap_available=70 * GiB, physical_available=78 * GiB, min_blocks=1
    )
    assert blocks == 64 * GiB // PAGE


def test_a_reserved_graph_copy_halves_the_physical_budget(monkeypatch):
    monkeypatch.setenv("QWEN4_EXP_POOL_GRAPH_COPIES", "1")
    blocks = qconfig.pooled_pool_blocks(
        PAGE,
        cap_available=int(87.16 * GiB) - int(15.9 * GiB),
        physical_available=int(94.74 * GiB) - int(15.9 * GiB),
        min_blocks=1,
    )
    assert blocks == (int(94.74 * GiB) - int(15.9 * GiB) - 6 * GiB) // 2 // PAGE


def test_max_pool_bytes_still_caps(monkeypatch):
    monkeypatch.setenv("QWEN4_EXP_MAX_POOL_BYTES", str(20 * GiB))
    blocks = qconfig.pooled_pool_blocks(
        PAGE, cap_available=70 * GiB, physical_available=78 * GiB, min_blocks=1
    )
    assert blocks == 20 * GiB // PAGE


def test_pool_must_hold_one_max_model_len_request():
    with pytest.raises(ValueError, match="max_model_len"):
        qconfig.pooled_pool_blocks(
            PAGE, cap_available=8 * GiB, physical_available=8 * GiB, min_blocks=9000
        )


def test_hbm_sizing_is_default_and_has_static_escape_hatches(monkeypatch):
    monkeypatch.setattr(qconfig, "use_unified_pool", lambda: True)
    cfg = pool_config()
    assert qconfig.hbm_pool_sizing(cfg)
    monkeypatch.setenv("QWEN4_EXP_POOL_SIZING", "static")
    assert not qconfig.hbm_pool_sizing(cfg)
    monkeypatch.delenv("QWEN4_EXP_POOL_SIZING")
    cfg.cache_config.num_gpu_blocks_override = 1000
    assert not qconfig.hbm_pool_sizing(cfg)
    cfg.cache_config.num_gpu_blocks_override = None
    cfg.kv_transfer_config = object()
    assert not qconfig.hbm_pool_sizing(cfg)


class _Attention:
    cache_owners = ()
    compressed_owner = None

    def bind_pool(self, pool, compressed_owner=None):
        self.cache_owners = (pool,)
        self.compressed_owner = compressed_owner


def _model(text):
    layers = []
    for index, kind in enumerate(text.layer_types):
        ple = _Attention() if index + 1 in text.ple_layer_ids else None
        if kind == "linear_attention":
            layers.append(NS(layer_type=kind, linear_attn=NS(kv_cache=None), ple=ple))
        else:
            layers.append(NS(layer_type=kind, self_attn=_Attention(), ple=ple))
    return NS(model=NS(layers=layers))


def _cache(monkeypatch):
    monkeypatch.setattr(qconfig, "use_unified_pool", lambda: True)
    monkeypatch.setattr(pooled_cache, "hbm_pool_sizing", qconfig.hbm_pool_sizing)
    monkeypatch.setattr(pooled_cache, "synchronize_tensors", lambda *a, **k: None)
    monkeypatch.setattr(pooled_cache, "AttentionMetadataBuilder", lambda *a, **k: None)
    cfg = pool_config()
    layout, _ = pooled_cache_specs(cfg)
    cfg.cache_config.block_size = cfg.cache_config.mamba_block_size = layout.block_size
    cfg.cache_config.gpu_memory_utilization = 0.9
    cfg.compilation_config = NS(static_forward_context={})
    runner = NS(
        vllm_config=cfg,
        cache_config=cfg.cache_config,
        max_model_len=cfg.model_config.max_model_len,
        device=torch.device("cpu"),
        kv_cache_dtype=torch.float8_e4m3fn,
        enforce_eager=True,
        may_reinitialize_input_batch=lambda *a: None,
        _build_mamba_copy_plan=lambda *a: None,
        _precompile_mamba_state_seed_copies=lambda: None,
    )
    model = _model(cfg.model_config.hf_text_config)
    return pooled_cache.Qwen4ExpPooledCache(runner, model), runner, model


def _budget(monkeypatch, used):
    budget = HbmBudget(
        total_limit=100 * GiB,
        total_used=used,
        cap=90 * GiB,
        available=90 * GiB - used,
    )
    monkeypatch.setattr(pooled_cache, "compute_hbm_budget", lambda *a: budget)


def test_dummy_pages_is_exactly_what_warmup_addresses(monkeypatch):
    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

    cache, _, _ = _cache(monkeypatch)
    metadata = AttentionMetadata(
        input_positions=torch.arange(700, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 700] + [700] * 6, dtype=torch.int32),
        seq_lens=torch.tensor([1, 699] + [0] * 6, dtype=torch.int32),
    )
    cache.pages = cache.dummy_pages([1, 699], active_rows=2)
    cache.dummy_metadata(metadata, active_rows=2)
    cache.pages -= 1
    with pytest.raises(AssertionError, match="dummy owner pages"):
        cache.dummy_metadata(metadata, active_rows=2)


def test_profile_on_scratch_then_planned_pool(monkeypatch):
    cache, runner, model = _cache(monkeypatch)
    _budget(monkeypatch, used=10 * GiB)
    # The dummy forward sizes a scratch pool to what it addresses.
    cache.allocate(16)
    assert (cache.pages, cache.final) == (16, False)
    scratch = cache.materialized.raw_tensors[0]
    attention = next(
        layer.self_attn for layer in model.model.layers if hasattr(layer, "self_attn")
    )
    assert attention.cache_owners[0].untyped_storage().data_ptr() == (
        scratch.untyped_storage().data_ptr()
    )
    cache.allocate(8)  # a smaller dummy reuses it
    assert cache.materialized.raw_tensors[0] is scratch

    # The worker's budget adds the scratch pool back (initialize frees it).
    budget = cache.memory_budget()
    scratch_bytes = 16 * cache.block_bytes
    physical = 90 * GiB + scratch_bytes
    # No graph-time pool copy by default: the utilization cap binds.
    expected_blocks = min(80 * GiB + scratch_bytes - 6 * GiB, physical - 6 * GiB)
    assert budget == expected_blocks // cache.block_bytes * cache.block_bytes

    # The engine plans from that budget; initialize swaps in the planned pool.
    planned = get_kv_cache_config_from_groups(
        runner.vllm_config, cache.groups, 64 * cache.block_bytes
    )
    planned.kv_cache_layout = runner.cache_config.kv_cache_layout
    _budget(monkeypatch, used=10 * GiB)
    cache.initialize(planned)
    assert (cache.pages, cache.final) == (64, True)
    pool = cache.materialized.raw_tensors[0]
    assert pool is not scratch
    assert attention.cache_owners[0].untyped_storage().data_ptr() == (
        pool.untyped_storage().data_ptr()
    )
    assert runner.kv_cache_raw_tensors == [pool]
    # Later dummy forwards (graph capture) run on the planned pool.
    cache.allocate(16)
    assert cache.materialized.raw_tensors[0] is pool


def test_planned_pool_refuses_what_its_graph_copies_cannot_fit(monkeypatch):
    monkeypatch.setenv("QWEN4_EXP_POOL_GRAPH_COPIES", "1")
    cache, runner, _ = _cache(monkeypatch)
    _budget(monkeypatch, used=10 * GiB)
    cache.allocate(4)
    planned = get_kv_cache_config_from_groups(
        runner.vllm_config, cache.groups, 64 * cache.block_bytes
    )
    planned.kv_cache_layout = runner.cache_config.kv_cache_layout
    # Free HBM holds one pool but not the pool plus its graph-time copy.
    _budget(monkeypatch, used=100 * GiB - 96 * cache.block_bytes)
    with pytest.raises(ValueError, match="graph-time copies"):
        cache.initialize(planned)


def test_static_sizing_keeps_the_fixed_pool(monkeypatch):
    monkeypatch.setenv("QWEN4_EXP_POOL_SIZING", "static")
    cache, _, _ = _cache(monkeypatch)
    assert cache.pages == qconfig.native_cache_capacity(
        cache.runner.vllm_config, cache.specs
    )
    assert cache.pages > 0


def test_packed_pool_plans_fewer_groups_and_binds_every_slot(monkeypatch):
    """Profiling scratch and planned pools bind packed owners and slots."""
    cache, runner, model = _cache(monkeypatch)
    unpacked_groups = len(cache.groups)
    monkeypatch.setenv("QWEN4_EXP_PACK_COMPRESSED_K", "1")
    cache, runner, model = _cache(monkeypatch)
    attention = {
        f"model.layers.{i}.self_attn": layer.self_attn
        for i, layer in enumerate(model.model.layers)
        if hasattr(layer, "self_attn")
    }
    slots = cache.layout.compressed_slots
    assert len(cache.groups) == unpacked_groups - len(attention) + -(
        -len(attention) // slots
    )
    assert cache.block_bytes == cache.layout.page_bytes
    _budget(monkeypatch, used=10 * GiB)
    cache.allocate(16)
    planned = get_kv_cache_config_from_groups(
        runner.vllm_config, cache.groups, 64 * cache.block_bytes
    )
    planned.kv_cache_layout = runner.cache_config.kv_cache_layout
    for config in (None, planned):
        if config is not None:
            cache.initialize(config)
        pool = cache.materialized.raw_tensors[0]
        owners = {}
        for name, layer in attention.items():
            assert layer.cache_owners[0] is pool
            assert layer.compressed_owner == cache.layout.compressed_owner(name)
            owner, slot = layer.compressed_owner
            assert owner in cache.materialized.kv_caches
            owners.setdefault(owner, []).append(slot)
        assert all(s == list(range(len(s))) for s in owners.values())
        assert max(map(len, owners.values())) == min(slots, len(attention))

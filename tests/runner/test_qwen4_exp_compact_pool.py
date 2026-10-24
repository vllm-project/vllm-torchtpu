# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native planner, real storage and block-lifecycle compact pool regressions."""

from types import SimpleNamespace as NS

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)

from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
from tests.runner.test_qwen4_exp_pool import pool_config
from vllm_torchtpu.models.vllm.qwen4_exp.config import native_cache_capacity
from vllm_torchtpu.models.vllm.qwen4_exp.pool_allocation import (
    materialize_pool,
    validate_pool_tiling,
)
from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs


def compact_config(
    *,
    kv="bfloat16",
    kv_heads=2,
    head_dim=256,
    index_dim=128,
    ratio=4,
    prefix=True,
    gdn=True,
    ple=True,
    length=512,
    layers=2,
):
    cfg = pool_config(kv=kv)
    kinds = (["linear_attention"] if gdn else []) + ["indexed_attention"] * layers
    cfg.model_config.hf_text_config = q4exp_text_config(
        hidden_size=128,
        num_attention_heads=kv_heads * 2,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        indexer_n_heads=1,
        indexer_head_dim=index_dim,
        indexer_compress_ratio=ratio,
        indexer_budget=64,
        num_hidden_layers=len(kinds),
        layer_types=kinds,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        ngram_size=2,
        heads_per_ngram=1,
        ple_embed_dim=128,
        ple_conv_kernel_size=4,
        ple_layer_ids=[1] if ple else [],
    )
    cfg.model_config.max_model_len = length
    cfg.scheduler_config.max_num_seqs = 2
    cfg.cache_config.enable_prefix_caching = prefix
    cfg.cache_config.mamba_cache_mode = "align" if prefix else "none"
    return cfg


def native_allocation(cfg):
    layout, specs = pooled_cache_specs(cfg)
    cfg.cache_config.block_size = layout.block_size
    cfg.cache_config.mamba_block_size = (
        layout.block_size if cfg.cache_config.enable_prefix_caching else None
    )
    pages = native_cache_capacity(cfg, specs)
    cfg.cache_config.num_gpu_blocks_override = pages
    groups = get_kv_cache_groups(cfg, specs)
    plan = get_kv_cache_config_from_groups(cfg, groups, 1 << 30)
    return layout, specs, plan, materialize_pool(cfg, plan, layout, torch.device("cpu"))


@pytest.mark.parametrize(
    "kv,heads,dim,index,ratio,prefix,gdn,ple,layers",
    [
        ("bfloat16", 2, 256, 128, 4, True, True, True, 2),
        ("fp8_e4m3", 2, 256, 128, 4, True, True, True, 2),
        ("bfloat16", 1, 128, 128, 2, False, False, False, 3),
        ("fp8_e4m3", 3, 256, 256, 8, False, True, False, 2),
        ("bfloat16", 4, 256, 192, 4, True, False, True, 2),
        ("fp8_e4m3", 1, 256, 128, 16, True, False, False, 1),
    ],
)
def test_native_backing_and_semantic_capacity(
    kv, heads, dim, index, ratio, prefix, gdn, ple, layers
):
    cfg = compact_config(
        kv=kv,
        kv_heads=heads,
        head_dim=dim,
        index_dim=index,
        ratio=ratio,
        prefix=prefix,
        gdn=gdn,
        ple=ple,
        layers=layers,
        length=513,
    )
    layout, specs, plan, actual = native_allocation(cfg)
    assert plan.num_blocks == native_cache_capacity(cfg, specs)
    assert layout.gdn_state_bytes > 0 if gdn else layout.gdn_state_bytes == 0
    assert layout.block_size % 128 == 0
    # An independent semantic byte count and actual backing storage oracle.
    semantic = layout.block_size // ratio * index * 2
    regions = max(len(g.layer_names) for g in plan.kv_cache_groups)
    storages = {
        t.untyped_storage()._cdata: t.untyped_storage().nbytes()
        for t in actual.raw_tensors
    }
    assert len(storages) == len(actual.raw_tensors) == regions * 2
    assert sum(storages.values()) == max(t.size for t in plan.kv_cache_tensors)
    assert sum(storages.values()) == plan.num_blocks * regions * (
        layout.main_page_bytes + layout.index_page_bytes
    )
    for name, spec in specs.items():
        if name.endswith(".self_attn"):
            main = actual.kv_caches[name]
            index_owner = actual.kv_caches[name + ".compressed_k"]
            ring = actual.kv_caches[name + ".raw_ring"]
            assert index_owner is ring and index_owner is not main
            assert index_owner.dtype == torch.bfloat16
            assert index_owner[0].nbytes >= semantic
            assert index_owner[0].nbytes < main[0].nbytes
            assert index_owner.shape[-1] >= index
            assert name + ".compressed_k" not in specs
            assert spec.block_size == layout.block_size
    # No extra physical allocation for merely reading a typed index view.
    pointers = [t.untyped_storage()._cdata for t in actual.raw_tensors]
    assert pointers == list(storages)


def test_compact_index_eliminates_separate_scheduler_owner():
    cfg = compact_config(layers=1, gdn=True, ple=False)
    layout, specs, plan, actual = native_allocation(cfg)
    name = "model.layers.1.self_attn"
    # This also fails on the old planner, which registers a separate padded
    # compressed owner and cannot make its data follow main prefix ownership.
    assert name + ".compressed_k" not in specs
    index = actual.kv_caches[name + ".compressed_k"]
    assert index[0].nbytes == 8192
    assert actual.kv_caches[name][0].nbytes == 256 << 10
    assert layout.block_size == 128


@pytest.mark.parametrize("prefix", [False, True])
def test_real_block_pool_reuse_preserves_both_prefix_planes(prefix):
    cfg = compact_config(gdn=True, ple=True, layers=2, prefix=prefix)
    layout, specs, plan, actual = native_allocation(cfg)
    pool = BlockPool(
        plan.num_blocks, enable_caching=prefix, hash_block_size=layout.block_size
    )
    blocks = pool.get_new_blocks(8)
    assert all(b.block_id > 0 for b in blocks)
    ids = [b.block_id for b in blocks][::-1]
    main = actual.kv_caches["model.layers.1.self_attn"]
    index = actual.kv_caches["model.layers.1.self_attn.compressed_k"]
    main.fill_(7)
    index.fill_(11)
    before_main = main.clone()
    before_index = index.clone()
    for ordinal, page in enumerate(ids[::2]):
        main[page, 3, 0, 0, 0] = ordinal + 20
        index[page, 1, 0, 0, 0] = ordinal + 40
    # Actual native retain/free/refcount operations: a shared prefix remains
    # owned until both users release it; planes are keyed by one retained ID.
    retained = blocks[2]
    pool.touch([retained])
    assert retained.ref_cnt == 2
    pool.free_blocks(blocks)
    assert retained.ref_cnt == 1
    allocated = pool.get_new_blocks(3)
    assert retained.block_id not in {b.block_id for b in allocated}
    assert index[retained.block_id, 1, 0, 0, 0] == (
        ids.index(retained.block_id) // 2 + 40 if retained.block_id in ids[::2] else 11
    )
    pool.free_blocks(allocated)
    pool.free_blocks([retained])
    assert retained.ref_cnt == 0
    replacements = pool.get_new_blocks(plan.num_blocks - 1)
    assert {b.block_id for b in replacements} == set(range(1, plan.num_blocks))
    for page in set(range(plan.num_blocks)) - set(ids[::2]):
        assert torch.equal(main[page], before_main[page])
        assert torch.equal(index[page], before_index[page])
    pool.free_blocks(replacements)


def test_copy_plan_uses_actual_main_row_not_paired_budget(monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp.pooled_cache import Qwen4ExpPooledCache
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    cfg = compact_config(layers=1)
    layout, specs, plan, actual = native_allocation(cfg)
    runner = NS(
        device=torch.device("cpu"),
        cache_config=cfg.cache_config,
        enforce_eager=True,
        may_reinitialize_input_batch=lambda *args: None,
        _precompile_mamba_state_seed_copies=lambda: None,
        _pool_block_split=1,
    )
    runner._build_mamba_copy_plan = TPUModelRunner._build_mamba_copy_plan.__get__(
        runner
    )
    cache = object.__new__(Qwen4ExpPooledCache)
    cache.runner = runner
    cache.groups = plan.kv_cache_groups
    cache.native_config = plan
    cache.layout = layout
    cache.kernel_block_sizes = [g.kv_cache_spec.block_size for g in cache.groups]
    cache.attn_groups = [
        [
            NS(
                kv_cache_spec=g.kv_cache_spec,
                layer_names=g.layer_names,
                metadata_builders=[],
            )
        ]
        for g in cache.groups
    ]
    cache.materialized = actual
    cache.allocate = lambda: None
    runner.vllm_config = cfg
    monkeypatch.setattr(
        "vllm_torchtpu.models.vllm.qwen4_exp.pooled_cache.AttentionMetadataBuilder",
        lambda *a, **kw: None,
    )
    cache.initialize(plan)
    assert runner._pool_block_split == 1
    copied = [t for _, tensors in runner._mamba_copy_plan for t in tensors]
    assert copied and all(t.shape[1] == layout.block_size for t in copied)
    assert all(t.dtype == torch.bfloat16 for t in copied)
    assert all(t[0].nbytes == layout.main_page_bytes for t in copied)
    assert all(t[0].nbytes < layout.page_bytes for t in copied)


@pytest.mark.parametrize("capacity", [0, -1, 2**31])
def test_capacity_rejects_invalid_int32_extent(capacity):
    layout, _ = pooled_cache_specs(compact_config())
    with pytest.raises(ValueError, match="int32"):
        layout.validate_capacity(capacity)


@pytest.mark.parametrize("vmem", [16 << 20, 64 << 20, 128 << 20])
def test_worker_layout_facts_are_checked_without_initializing_a_backend(vmem):
    layout, _ = pooled_cache_specs(compact_config())
    hardware = NS(
        num_lanes=128, get_sublane_tiling=lambda _: 16, vmem_capacity_bytes=vmem
    )
    validate_pool_tiling(layout, hardware)
    with pytest.raises(ValueError, match="known TPU"):
        validate_pool_tiling(layout, None)
    with pytest.raises(ValueError, match="lane/sublane"):
        validate_pool_tiling(
            layout,
            NS(
                num_lanes=256, get_sublane_tiling=lambda _: 16, vmem_capacity_bytes=vmem
            ),
        )
    with pytest.raises(ValueError, match="VMEM"):
        validate_pool_tiling(
            layout,
            NS(
                num_lanes=128, get_sublane_tiling=lambda _: 16, vmem_capacity_bytes=1024
            ),
        )


@pytest.mark.parametrize("tp,kv_heads", [(1, 3), (2, 4), (4, 4)])
def test_local_head_geometry_is_not_a_tp2_policy(tp, kv_heads):
    cfg = compact_config(kv_heads=kv_heads, layers=1)
    cfg.parallel_config.tensor_parallel_size = tp
    cfg.model_config.hf_text_config.linear_num_key_heads = tp
    cfg.model_config.hf_text_config.linear_num_value_heads = 2 * tp
    layout, specs, plan, actual = native_allocation(cfg)
    main = actual.kv_caches["model.layers.1.self_attn"]
    packing = 2
    assert main.shape[2] == (2 * max(kv_heads // tp, 1) + packing - 1) // packing


@pytest.mark.parametrize("gdn", [False, True])
def test_native_manager_prefix_hit_and_request_reuse(monkeypatch, gdn):
    from vllm import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request

    cfg = compact_config(gdn=gdn, ple=gdn, layers=1)
    layout, specs, plan, actual = native_allocation(cfg)
    monkeypatch.setattr(
        kv_cache_utils,
        "NONE_HASH",
        kv_cache_utils.BlockHash(sha256(b"compact-pool-test")),
        raising=False,
    )
    hasher = kv_cache_utils.get_request_block_hasher(layout.block_size, sha256)
    manager = KVCacheManager(
        plan,
        max_model_len=512,
        scheduler_block_size=layout.block_size,
        hash_block_size=layout.block_size,
        enable_caching=True,
    )
    prompt = list(range(2 * layout.block_size + 17))

    def request(name):
        return Request(
            name, prompt, SamplingParams(max_tokens=8), None, block_hasher=hasher
        )

    first, second = request("first"), request("second")
    assert manager.allocate_slots(first, 2 * layout.block_size) is not None
    first.num_computed_tokens = 2 * layout.block_size
    manager.cache_blocks(first, first.num_computed_tokens)
    initial = manager.get_block_ids(first.request_id)
    # The native Mamba manager rejects checkpoints generated in this same
    # scheduling step. The producer has completed before prefix adoption.
    manager.new_step_starts()
    # This fixture drives KVCacheManager directly. In serving, the TPU
    # Scheduler.schedule wrapper clears the previous forward's unwritten
    # set at this boundary; a manager-only test must model that boundary too.
    manager.block_pool.__dict__.get("_tpu_unwritten_block_ids", set()).clear()
    hits, count, boundary = manager.get_computed_blocks(second)
    assert count == 2 * layout.block_size, {
        "hashes_equal": first.block_hashes == second.block_hashes,
        "hash_count": len(first.block_hashes),
        "lookup": manager.prefix_cache_lookup_enabled(second),
        "cache": repr(manager.block_pool.cached_block_hash_to_block._cache),
        "request_tokens": second.num_tokens,
        "group_blocks": [g.kv_cache_spec.block_size for g in plan.kv_cache_groups],
    }
    assert (
        manager.allocate_slots(
            second, 1, num_new_computed_tokens=count, new_computed_blocks=hits
        )
        is not None
    )
    reused = manager.get_block_ids(second.request_id)
    main_gid = next(
        i
        for i, g in enumerate(plan.kv_cache_groups)
        if any(n.endswith(".self_attn") for n in g.layer_names)
    )
    ring_gid = next(
        i
        for i, g in enumerate(plan.kv_cache_groups)
        if any(n.endswith(".raw_ring") for n in g.layer_names)
    )
    assert initial[main_gid][:2] == reused[main_gid][:2]
    assert set(initial[ring_gid]).isdisjoint(reused[ring_gid])
    retained = set(reused[main_gid][:2])
    manager.free(first)
    for block in retained:
        assert manager.block_pool.blocks[block].ref_cnt == 1
    manager.free(second)
    manager.new_step_starts()
    # This fixture drives KVCacheManager directly. In serving, the TPU
    # Scheduler.schedule wrapper clears the previous forward's unwritten
    # set at this boundary; a manager-only test must model that boundary too.
    manager.block_pool.__dict__.get("_tpu_unwritten_block_ids", set()).clear()
    third = request("third")
    prefix, matched, _ = manager.get_computed_blocks(third)
    assert matched == count
    assert (
        manager.allocate_slots(
            third, 1, num_new_computed_tokens=matched, new_computed_blocks=prefix
        )
        is not None
    )
    assert manager.get_block_ids(third.request_id)[main_gid][:2] == list(
        reused[main_gid][:2]
    )
    manager.free(third)


@pytest.mark.parametrize("gdn", [False, True])
def test_native_tables_real_runner_metadata_use_one_prefix_owner(monkeypatch, gdn):
    from tests.runner.qwen4_pool_test_utils import prepare_native_pool_inputs

    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
    cfg = compact_config(gdn=gdn, ple=gdn, layers=1)
    layout, specs, plan, actual = native_allocation(cfg)
    # Public BlockPool gives disjoint physical IDs; real native BlockTable
    # handles group widths/padding, rather than a stub table accessor.
    pool = BlockPool(
        plan.num_blocks, enable_caching=True, hash_block_size=layout.block_size
    )
    per_req = []
    for row in range(2):
        groups = []
        for group in plan.kv_cache_groups:
            count = group.kv_cache_spec.max_num_blocks_per_req(cfg, 512)
            count = (
                min(count, 2) if not group.layer_names[0].endswith(".raw_ring") else 1
            )
            groups.append([b.block_id for b in pool.get_new_blocks(count)][::-1])
        per_req.append(tuple(groups))
    runner, kwargs, metadata = prepare_native_pool_inputs(
        cfg, plan, per_req, [128, 37], [3, 5], 16
    )
    assert torch.equal(kwargs["main_block_table"], kwargs["compressed_block_table"])
    assert kwargs["query_start_loc"].tolist() == [0, 3, 8, 8, 8, 8, 8, 8, 8]
    assert kwargs["context_lens"].tolist() == [128, 37, 0, 0, 0, 0, 0, 0]
    assert kwargs["valid_tokens"].tolist() == [True] * 8 + [False] * 8
    assert all(".compressed_k" not in name for name in metadata)
    if gdn:
        state = metadata["model.layers.0.linear_attn"].mamba_state_indices
        assert state[0] > 0 and state[1] > 0 and torch.count_nonzero(state[2:]) == 0

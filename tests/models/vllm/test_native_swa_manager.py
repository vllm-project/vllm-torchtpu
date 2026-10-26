# SPDX-License-Identifier: Apache-2.0
"""Exercise private SWA pools through the upstream KV-cache control plane."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.request import Request

from vllm_torchtpu.models.vllm.deepseek_v41.v2.swa_cache import (
    CompactSWAManager,
    CompactSWASpec,
    native_swa_reservation,
    register_native_swa_specs,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def register_specs():
    KVCacheSpecRegistry.check_kv_cache_spec_registry({})
    register_native_swa_specs()


def spec(sequences=2, in_flight=512):
    return CompactSWASpec(
        block_size=32,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
        bounded_replay=True,
        max_num_seqs=sequences,
        max_in_flight_tokens=in_flight,
    )


def manager(swa_spec=None, *, caching=False, groups=1):
    swa_spec = swa_spec or spec()
    config = KVCacheConfig(
        num_blocks=4096,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["main"],
                FullAttentionSpec(
                    block_size=32,
                    num_kv_heads=1,
                    head_size=512,
                    dtype=torch.bfloat16,
                ),
            ),
            *[KVCacheGroupSpec([f"swa.{i}"], swa_spec) for i in range(groups)],
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=32768,
        scheduler_block_size=32,
        hash_block_size=32,
        max_in_flight_tokens=swa_spec.max_in_flight_tokens,
        enable_caching=caching,
    )


def request(name="request"):
    return Request(
        name,
        prompt_token_ids=[1] * 32768,
        sampling_params=SamplingParams(max_tokens=128),
        pooling_params=None,
    )


def test_admission_charges_the_correct_pool_and_rejects_exhaustion():
    cache = manager(spec(sequences=1, in_flight=32))
    swa = cache.coordinator.single_type_managers[1]
    assert isinstance(swa, CompactSWAManager)
    assert swa.block_pool is not cache.block_pool
    free = swa.block_pool.get_num_free_blocks()
    first = request("first")
    assert cache.allocate_slots(first, free * 32) is not None
    assert cache.block_pool.get_num_free_blocks() == 4095 - free
    assert swa.block_pool.get_num_free_blocks() == 0
    shared_free = cache.block_pool.get_num_free_blocks()
    assert cache.allocate_slots(request("second"), 32) is None
    assert cache.block_pool.get_num_free_blocks() == shared_free
    cache.free(first)
    assert cache.allocate_slots(request("second"), 32) is not None


def test_swa_reclamation_waits_for_inflight_prefill():
    cache = manager()
    req = request()
    assert cache.allocate_slots(req, 256) is not None
    swa = cache.coordinator.single_type_managers[1]
    old = list(swa.req_to_blocks[req.request_id][:4])
    req.num_computed_tokens = 256
    req.num_in_flight_tokens = 256
    assert cache.allocate_slots(req, 6) is not None
    assert all(block.ref_cnt == 1 for block in old)
    req.num_in_flight_tokens = 0
    assert cache.allocate_slots(req, 6) is not None
    assert all(block.ref_cnt == 0 for block in old)
    assert all(block.is_null for block in swa.req_to_blocks[req.request_id][:4])
    assert not any(
        block.is_null
        for block in cache.coordinator.single_type_managers[0].req_to_blocks[
            req.request_id
        ]
    )


def test_rejection_preserves_cross_page_history():
    cache = manager()
    req = request()
    assert cache.allocate_slots(req, 160) is not None
    swa = cache.coordinator.single_type_managers[1]
    history = swa.req_to_blocks[req.request_id][0]
    # CPU positions advance to 160 while the preceding six queries are pending.
    req.num_computed_tokens = 160
    req.num_in_flight_tokens = 6
    assert cache.allocate_slots(req, 6) is not None
    assert cache.get_block_ids(req.request_id)[1][0] == history.block_id
    # Rejection rewinds to 158; token 31 remains in the 128-token window.
    req.num_computed_tokens = 158
    req.num_in_flight_tokens = 0
    assert cache.allocate_slots(req, 6) is not None
    assert history.ref_cnt == 1
    assert cache.get_block_ids(req.request_id)[1][0] == history.block_id
    req.num_computed_tokens = 166
    assert cache.allocate_slots(req, 6) is not None
    assert history.ref_cnt == 0


def test_deferred_free_returns_each_block_to_its_own_pool():
    cache = manager(groups=2)
    req = request()
    pools = [mgr.block_pool for mgr in cache.coordinator.single_type_managers]
    before = [pool.get_num_free_blocks() for pool in pools]
    assert len(set(map(id, pools))) == 3
    assert cache.allocate_slots(req, 128) is not None
    blocks = cache.pop_blocks_for_free(req)
    assert [pool.get_num_free_blocks() for pool in pools] == [n - 4 for n in before]
    cache.block_pool.free_blocks(reversed(blocks))
    assert [pool.get_num_free_blocks() for pool in pools] == before
    assert all(pool.null_block.is_null for pool in pools)


def test_private_ids_do_not_enter_shared_zeroing_or_copy_commands():
    cache = manager()
    req = request()
    assert cache.allocate_slots(req, 128) is not None
    swa = cache.coordinator.single_type_managers[1]
    assert not swa.records_new_block_ids
    assert swa.take_new_block_ids() == []
    assert swa.take_pending_cow_copies() == []


def test_bounded_replay_keeps_upstream_prefix_policy():
    cache = manager(caching=True)
    assert len(cache.coordinator.attention_groups) == 1
    assert cache.coordinator.attention_groups[0].group_ids == [0]
    with pytest.raises(ValueError, match="bounded prefix replay"):
        manager(replace(spec(), bounded_replay=False), caching=True)


def test_spec_merge_and_page_unification_keep_pool_capacity_inputs():
    original = spec()
    merged = CompactSWASpec.merge([original, original])
    assert merged == original
    larger_pages = replace(merged, block_size=64)
    assert larger_pages.pool_capacity < original.pool_capacity
    assert larger_pages.pool_capacity * 64 >= 2 * 127 + 512


def test_upstream_grouping_preserves_private_pool_contract():
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=32768, original_max_model_len=32768),
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
        max_in_flight_tokens=512,
        num_prefill_lookahead_tokens=0,
    )
    specs = {
        "main": MLAAttentionSpec(
            block_size=32,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            tokens_per_state=2,
        ),
        "swa.0": spec(),
        "swa.1": spec(),
    }
    reservation = native_swa_reservation(specs)
    result = get_kv_cache_configs(config, [specs], [(256 << 20) - reservation])[0]
    swa_groups = [
        g for g in result.kv_cache_groups if isinstance(g.kv_cache_spec, CompactSWASpec)
    ]
    assert sum(len(g.layer_names) for g in swa_groups) == 2
    assert all(
        g.kv_cache_spec.pool_capacity == spec().pool_capacity for g in swa_groups
    )
    assert all(g.kv_cache_spec.max_num_seqs == 2 for g in swa_groups)

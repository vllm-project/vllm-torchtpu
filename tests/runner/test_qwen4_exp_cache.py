# SPDX-License-Identifier: Apache-2.0
"""Actual native planner contracts without allocating model/checkpoint memory."""
from types import SimpleNamespace as NS

import pytest
import torch
from vllm.config.cache import CacheConfig
from vllm.v1.core.kv_cache_utils import (get_kv_cache_config_from_groups,
                                         get_kv_cache_groups)
from vllm.v1.kv_cache_interface import (CircularBufferSpec, FullAttentionSpec,
                                        MLAAttentionSpec)

from vllm_torchtpu.models.vllm.qwen4_exp.cache import Qwen4ExpCache


@pytest.mark.parametrize('seqs,context,expected', [(1, 4096, 44),
                                                   (8, 8192, 601),
                                                   (64, 8192, 4801)])
def test_native_groups_last_blocks_and_real_capacity(seqs, context, expected,
                                                     monkeypatch):
    cc = CacheConfig(block_size=128,
                     enable_prefix_caching=False,
                     mamba_cache_mode='none')
    cfg = NS(cache_config=cc,
             scheduler_config=NS(disable_hybrid_kv_cache_manager=False),
             speculative_config=None,
             kv_transfer_config=None,
             model_config=NS(max_model_len=context),
             parallel_config=NS(decode_context_parallel_size=1,
                                prefill_context_parallel_size=1))
    qsa = NS(get_kv_cache_spec=lambda _: FullAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=256, dtype=torch.bfloat16),
             get_auxiliary_cache_specs=lambda: {
                 'compressed_k':
                 MLAAttentionSpec(block_size=128,
                                  num_kv_heads=1,
                                  head_size=128,
                                  dtype=torch.bfloat16,
                                  tokens_per_state=4),
                 'raw_ring':
                 CircularBufferSpec(block_size=4,
                                    num_kv_heads=1,
                                    head_size=132,
                                    head_size_v=0,
                                    dtype=torch.bfloat16,
                                    page_size_padded=8192)
             })
    gdn = NS(get_state_shape=lambda: ((3, 1280), (6, 128, 128)),
             get_state_dtype=lambda: (torch.float32, torch.float32))
    ple = NS(get_state_shape=lambda: ((9, 10240), ),
             get_state_dtype=lambda: (torch.bfloat16, ))
    layers = [
        NS(layer_type='full_attention' if i % 4 == 3 else 'linear_attention',
           self_attn=qsa,
           linear_attn=gdn,
           ple=ple if i == 1 else None) for i in range(48)
    ]
    runner = NS(vllm_config=cfg,
                cache_config=cc,
                max_num_reqs=max(seqs, 8),
                max_model_len=context,
                scheduler_config=NS(max_num_seqs=seqs))
    cache = Qwen4ExpCache(runner, NS(model=NS(layers=layers)))
    specs = cache.get_specs()
    groups = get_kv_cache_groups(cfg, specs)
    native = get_kv_cache_config_from_groups(cfg, groups, 32 * 1024**3)
    assert (len(specs), len(groups), len(native.kv_cache_tensors)) == (73, 12,
                                                                       13)
    assert native.num_blocks == expected
    assert cache.slots == max(seqs, 8) + 1
    for descriptor in native.kv_cache_tensors:
        for i, name in enumerate(descriptor.layers):
            assert descriptor.offset + i * descriptor.layer_stride + (
                expected - 1) * descriptor.block_stride + specs[
                    name].page_size_bytes <= descriptor.size
    needed = sum(
        g.kv_cache_spec.max_num_blocks_per_req(cfg, context) for g in groups)
    assert needed * seqs + 1 == expected

    # Static max_num_blocks_per_req is not the allocator's admission rule.
    # Exercise full-ISL admission, chunk growth, exhaustion and actual reuse.
    from vllm import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config, resolve_kv_cache_block_sizes)
    from vllm.v1.request import Request
    scheduler_config = generate_scheduler_kv_cache_config([native])
    scheduler_block, hash_block = resolve_kv_cache_block_sizes(
        scheduler_config, cfg)
    manager = KVCacheManager(scheduler_config,
                             max_model_len=context,
                             scheduler_block_size=scheduler_block,
                             hash_block_size=hash_block,
                             max_in_flight_tokens=128,
                             enable_caching=False)
    requests = [
        Request(str(i), [17] * context, SamplingParams(max_tokens=1), None)
        for i in range(seqs + 1)
    ]
    for request in requests[:-1]:
        for offset in range(0, context, 128):
            chunk = min(128, context - offset)
            assert manager.allocate_slots(
                request, chunk, full_sequence_must_fit=True) is not None
            request.num_computed_tokens += chunk
    assert manager.block_pool.get_num_free_blocks() == 0
    assert manager.allocate_slots(requests[-1],
                                  128,
                                  full_sequence_must_fit=True) is None
    ids = {
        block.block_id
        for m in manager.coordinator.single_type_managers
        for blocks in m.req_to_blocks.values()
        for block in blocks
    }
    assert ids == set(range(1, expected))
    for request in requests[:-1]:
        manager.free(request)
    assert manager.block_pool.get_num_free_blocks() == expected - 1
    assert manager.allocate_slots(requests[-1],
                                  128,
                                  full_sequence_must_fit=True) is not None
    manager.free(requests[-1])
    assert manager.block_pool.get_num_free_blocks() == expected - 1

    # Exercise the real native-wrapper -> metadata-builder boundary. A
    # UniformTypeKVCacheSpecs wrapper must not erase the GDN slot tensor.
    from vllm_torchtpu.layers.core.attention_metadata import \
        AttentionMetadataBuilderContext
    from vllm_torchtpu.models.vllm.qwen4_exp import cache as cache_module
    monkeypatch.setattr(cache, 'allocate', lambda: None)
    monkeypatch.setattr(cache_module, 'compute_hbm_budget',
                        lambda *args: NS(available=32 * 1024**3))
    runner.device = 'cpu'
    runner.input_batch = NS(
        block_table=[NS(max_num_blocks_per_req=32) for _ in groups])
    runner.may_reinitialize_input_batch = lambda *args: None
    runner._init_mamba_slot_pool = lambda slots: None
    runner.position_ids = torch.arange(16, dtype=torch.int32)
    cache.initialize(native)
    rows = runner.max_num_reqs
    slots = torch.arange(1, rows + 1, dtype=torch.int32)
    runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
        num_reqs=rows,
        start_index=0,
        use_max_model_len=True,
        seq_lens=torch.ones(rows, dtype=torch.int32),
        query_start_loc=torch.arange(rows + 1, dtype=torch.int32),
        request_distribution=torch.tensor([rows] * 3, dtype=torch.int32),
        mamba_state_indices=slots,
        staged_block_tables={
            gid: torch.zeros(rows * 32, dtype=torch.int32)
            for gid in range(len(groups))
        })
    seen_recurrent = 0
    for group_list in runner.attn_groups:
        for group in group_list:
            metadata = group.metadata_builders[0].build(0, NS(num_reqs=rows))
            for name in group.layer_names:
                if name.endswith(('.linear_attn', '.ple')):
                    assert metadata.mamba_state_indices is slots
                    seen_recurrent += 1
                else:
                    assert metadata.mamba_state_indices is None
    assert seen_recurrent == 37

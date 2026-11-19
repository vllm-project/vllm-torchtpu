# SPDX-License-Identifier: Apache-2.0
"""Actual native planner contracts without allocating model/checkpoint memory."""

from types import SimpleNamespace as NS

import pytest
import torch
from vllm.config.attention import AttentionConfig
from vllm.config.cache import CacheConfig
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import CircularBufferSpec, MLAAttentionSpec

from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
from vllm_torchtpu.models.vllm.qwen4_exp.cache import Qwen4ExpCache
from vllm_torchtpu.models.vllm.qwen4_exp.config import native_cache_capacity
from vllm_torchtpu.models.vllm.qwen4_exp.qsa import Qwen4ExpQSA


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", ["auto", "fp8_e4m3"])
@pytest.mark.parametrize("seqs", [1, 8, 64])
@pytest.mark.parametrize("context", [128, 4096, 8192])
def test_native_groups_last_blocks_and_real_capacity(
    tp, dtype, seqs, context, monkeypatch
):
    # TP1/2 require 36 native recurrent groups, TP4 eighteen and TP8 nine.
    # The actual planner/allocator below independently verifies
    # the model capacity formula, including full exhaustion and reuse.
    cc = CacheConfig(
        block_size=128,
        cache_dtype=dtype,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        mamba_ssm_cache_dtype="float32",
    )
    cc.kv_cache_layout = "BLHNC"
    cfg = NS(
        additional_config={},
        attention_config=AttentionConfig(),
        cache_config=cc,
        scheduler_config=NS(disable_hybrid_kv_cache_manager=False, max_num_seqs=seqs),
        speculative_config=None,
        kv_transfer_config=None,
        model_config=NS(
            max_model_len=context,
            dtype=torch.bfloat16,
            hf_text_config=q4exp_text_config(),
        ),
        parallel_config=NS(
            tensor_parallel_size=tp,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )
    expected = native_cache_capacity(cfg)
    qsa = NS(
        config=cfg.model_config.hf_text_config,
        tp_size=tp,
        block_size=128,
        head_dim=256,
        index_head_dim=128,
        compress_ratio=4,
        ring_size=8,
        get_cache_owner_layouts=lambda pages, slots: (
            Qwen4ExpQSA.get_cache_owner_layouts(qsa, pages, slots)
        ),
        num_kv_heads=max(2 // tp, 1),
        fp8_kv=dtype == "fp8_e4m3",
        cache_dtype=dtype,
        get_kv_cache_spec=lambda cfg: Qwen4ExpQSA.get_kv_cache_spec(qsa, cfg),
        get_auxiliary_cache_specs=lambda: {
            "compressed_k": MLAAttentionSpec(
                block_size=128,
                num_kv_heads=1,
                head_size=128,
                state_content_bytes=256,
                dtype=torch.bfloat16,
                tokens_per_state=4,
            ),
            "raw_ring": CircularBufferSpec(
                block_size=4,
                num_kv_heads=1,
                head_size=132,
                head_size_v=0,
                dtype=torch.bfloat16,
                page_size_padded=8192,
            ),
        },
    )
    gdn = NS(
        get_state_shape=lambda: ((3, 1, 10240 // tp), (48 // tp, 128, 128)),
        get_state_dtype=lambda: (torch.float32, torch.float32),
    )
    ple = NS(state_shape=(9, 10240))
    layers = [
        NS(
            layer_type="full_attention" if i % 4 == 3 else "linear_attention",
            self_attn=qsa,
            linear_attn=gdn,
            ple=ple if i == 1 else None,
        )
        for i in range(48)
    ]
    runner = NS(
        vllm_config=cfg,
        cache_config=cc,
        max_num_reqs=max(seqs, 8),
        max_model_len=context,
        scheduler_config=NS(max_num_seqs=seqs),
    )
    cache = Qwen4ExpCache(runner, NS(model=NS(layers=layers)))
    specs = cache.get_specs()
    groups = get_kv_cache_groups(cfg, specs)
    native = get_kv_cache_config_from_groups(cfg, groups, 32 * 1024**3)
    expected_groups = ({1: 39, 2: 39, 4: 21, 8: 12})[tp]
    assert (len(specs), len(groups), len(native.kv_cache_tensors)) == (
        73,
        expected_groups,
        expected_groups + 1,
    )
    assert native.num_blocks == expected
    assert cache.slots == max(seqs, 8) + 1
    for descriptor in native.kv_cache_tensors:
        for i, name in enumerate(descriptor.layers):
            assert (
                descriptor.offset
                + i * descriptor.layer_stride
                + (expected - 1) * descriptor.block_stride
                + specs[name].page_size_bytes
                <= descriptor.size
            )
    needed = sum(g.kv_cache_spec.max_num_blocks_per_req(cfg, context) for g in groups)
    assert needed * seqs + 1 == expected
    # The allowance must cover all logical owners, with actual TP widths.
    main_bytes = 128 * 1024 * (1 if qsa.fp8_kv else qsa.num_kv_heads)
    logical = (
        12 * expected * (main_bytes + 32 * 128 * 2)
        + 36 * cache.slots * (3 * (10240 // tp) + (48 // tp) * 128 * 128) * 4
        + cache.slots * 9 * 10240 * 2
        + 12 * cache.slots * (8 * 128 * 2 + 8 * 8)
    )
    assert cache.allocation_allowance_bytes() >= logical

    # Static max_num_blocks_per_req is not the allocator's admission rule.
    # Exercise full-ISL admission, chunk growth, exhaustion and actual reuse.
    from vllm import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        resolve_kv_cache_block_sizes,
    )
    from vllm.v1.request import Request

    scheduler_config = generate_scheduler_kv_cache_config([native])
    scheduler_block, hash_block = resolve_kv_cache_block_sizes(scheduler_config, cfg)
    manager = KVCacheManager(
        scheduler_config,
        max_model_len=context,
        scheduler_block_size=scheduler_block,
        hash_block_size=hash_block,
        max_in_flight_tokens=128,
        enable_caching=False,
    )
    requests = [
        Request(str(i), [17] * context, SamplingParams(max_tokens=1), None)
        for i in range(seqs + 1)
    ]
    for request in requests[:-1]:
        for offset in range(0, context, 128):
            chunk = min(128, context - offset)
            assert (
                manager.allocate_slots(request, chunk, full_sequence_must_fit=True)
                is not None
            )
            request.num_computed_tokens += chunk
    assert manager.block_pool.get_num_free_blocks() == 0
    assert (
        manager.allocate_slots(requests[-1], 128, full_sequence_must_fit=True) is None
    )
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
    assert (
        manager.allocate_slots(requests[-1], 128, full_sequence_must_fit=True)
        is not None
    )
    manager.free(requests[-1])
    assert manager.block_pool.get_num_free_blocks() == expected - 1

    # Exercise the real native-wrapper -> metadata-builder boundary. A
    # UniformTypeKVCacheSpecs wrapper must not erase the GDN slot tensor.
    from vllm_torchtpu.layers.core.attention_metadata import (
        AttentionMetadataBuilderContext,
    )
    from vllm_torchtpu.models.vllm.qwen4_exp import cache as cache_module

    # Exercise the actual allocation/binding path without allocating GiBs of
    # CPU memory. Each stand-in records the requested shape/dtype and owns a
    # distinct storage identity, just as the private allocator contract asks.
    class Owner:
        def __init__(self, shape, dtype):
            self.shape, self.dtype = shape, dtype

        def untyped_storage(self):
            return NS(data_ptr=lambda: id(self))

    def bind_qsa(owners):
        auxiliary_shapes = [
            (expected, 32, 1, 128),
            (cache.slots, 8, 128),
            (cache.slots, 8),
        ]
        auxiliary_dtypes = [torch.bfloat16, torch.bfloat16, torch.int64]
        if qsa.fp8_kv:
            shapes = [
                (expected, 128, 1, 4, 256),
                *auxiliary_shapes,
            ]
            dtypes = [torch.float8_e4m3fn, *auxiliary_dtypes]
        else:
            shapes = [(expected, 128, qsa.num_kv_heads, 2, 256)] + auxiliary_shapes
            dtypes = [torch.bfloat16] + auxiliary_dtypes
        assert [x.shape for x in owners] == shapes
        assert [x.dtype for x in owners] == dtypes

    qsa.bind_cache_owners = bind_qsa
    ple.bind_history = lambda owner: None
    runner.device = "cpu"
    with monkeypatch.context() as allocation_patch:
        allocation_patch.setattr(
            torch, "full", lambda shape, fill, dtype, device: Owner(shape, dtype)
        )
        allocation_patch.setattr(
            cache_module, "synchronize_tensors", lambda *args: None
        )
        allocation_patch.setattr(
            cache_module,
            "compute_hbm_budget",
            lambda *args: NS(available=64 * 1024**3, total_used=0),
        )
        cache.allocate()
    assert len(cache.owners) == 121
    assert cache.owners["model.layers.0.linear_attn.conv"].dtype == torch.float32
    assert cache.owners["model.layers.0.linear_attn.recurrent"].dtype == torch.float32
    assert cache.owners["model.layers.0.linear_attn.conv"].shape == (
        cache.slots,
        3,
        1,
        10240 // tp,
    )
    assert cache.owners["model.layers.0.linear_attn.recurrent"].shape == (
        cache.slots,
        48 // tp,
        128,
        128,
    )
    assert cache.owners["model.layers.1.ple.history"].shape == (cache.slots, 9, 10240)
    monkeypatch.setattr(cache, "allocate", lambda: None)
    monkeypatch.setattr(
        cache_module, "compute_hbm_budget", lambda *args: NS(available=32 * 1024**3)
    )
    runner.device = "cpu"
    runner.input_batch = NS(block_table=[NS(max_num_blocks_per_req=32) for _ in groups])
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
            gid: torch.zeros(rows * 32, dtype=torch.int32) for gid in range(len(groups))
        },
    )
    seen_recurrent = 0
    for group_list in runner.attn_groups:
        for group in group_list:
            metadata = group.metadata_builders[0].build(0, NS(num_reqs=rows))
            for name in group.layer_names:
                if name.endswith((".linear_attn", ".ple")):
                    assert metadata.mamba_state_indices is slots
                    seen_recurrent += 1
                else:
                    assert metadata.mamba_state_indices is None
    assert seen_recurrent == 37


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("ssm_dtype", ["auto", "bfloat16", "float32"])
def test_gdn_planner_matches_existing_adapter(monkeypatch, tp, ssm_dtype):
    from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
        VllmGatedDeltaNetAttention,
    )
    from vllm_torchtpu.models.vllm.qwen4_exp.config import gdn_cache_spec

    # This planner describes private state; pooled state has its own contract.
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "0")
    # This is the post-normalization configuration consumed by both the
    # planner and adapter. BF16 is also checked to catch a forced FP32 SSM.
    cfg = NS(
        model_config=NS(
            dtype=torch.bfloat16,
            architecture="Qwen4ExpForCausalLM",
            max_model_len=8192,
            hf_text_config=q4exp_text_config(),
        ),
        cache_config=CacheConfig(
            mamba_cache_dtype="auto", mamba_ssm_cache_dtype=ssm_dtype
        ),
        parallel_config=NS(tensor_parallel_size=tp),
    )
    # Only state metadata is needed: avoid allocating weights or building
    # device ops when exercising the real adapter's inherited methods.
    layer = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.model_config, layer.cache_config = cfg.model_config, cfg.cache_config
    t = cfg.model_config.hf_text_config
    layer.tp_size, layer.num_spec = tp, 0
    layer.num_k_heads, layer.num_v_heads = (
        t.linear_num_key_heads,
        t.linear_num_value_heads,
    )
    layer.head_k_dim, layer.head_v_dim = (
        t.linear_key_head_dim,
        t.linear_value_head_dim,
    )
    layer.conv_kernel_size = t.linear_conv_kernel_dim

    spec = gdn_cache_spec(cfg)
    assert spec.shapes == layer.get_state_shape()
    assert spec.dtypes == layer.get_state_dtype()
    assert spec.dtypes[1] == (
        torch.float32 if ssm_dtype == "float32" else torch.bfloat16
    )

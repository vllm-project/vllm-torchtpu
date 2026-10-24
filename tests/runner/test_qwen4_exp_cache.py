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

from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
from vllm_torchtpu.models.vllm.qwen4_exp.config import native_cache_capacity
from vllm_torchtpu.models.vllm.qwen4_exp.pooled_cache import Qwen4ExpPooledCache


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
    runner = NS(vllm_config=cfg, cache_config=cc)
    cache = Qwen4ExpPooledCache(runner, model=None)
    cc.block_size = cache.layout.block_size
    specs = cache.get_specs()
    groups = get_kv_cache_groups(cfg, specs)
    expected = native_cache_capacity(cfg, specs)
    cc.num_gpu_blocks_override = expected
    native = get_kv_cache_config_from_groups(cfg, groups, 32 * 1024**3)
    assert native.num_blocks == cache.pages == expected
    assert len(specs) == 61  # compressed keys share twelve main logical owners.
    assert all(
        name + ".compressed_k" not in specs
        for name in specs
        if name.endswith(".self_attn")
    )
    assert {spec.page_size_bytes for spec in specs.values()} == {
        cache.layout.page_bytes
    }
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
    assert (
        cache.layout.main_page_bytes + cache.layout.index_page_bytes
        == cache.layout.page_bytes
    )
    assert cache.layout.main_page_bytes >= max(
        cache.layout.gdn_state_bytes, cache.layout.ple_state_bytes
    )

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

    # Real native metadata builders retain scheduler-owned state IDs, rather
    # than the retired shared private-slot allocator. Storage shape/binding is
    # exercised by the dual-store materializer cases in test_qwen4_exp_pool_storage.
    from tests.runner.qwen4_pool_test_utils import prepare_native_pool_inputs

    rows = [
        tuple(
            [gid + 1] * g.kv_cache_spec.max_num_blocks_per_req(cfg, context)
            for gid, g in enumerate(groups)
        )
    ]
    _, kwargs, metadata = prepare_native_pool_inputs(cfg, native, rows, [0], [1], 8)
    assert kwargs["main_block_table"] is kwargs["compressed_block_table"]
    for gid, group in enumerate(groups):
        for name in group.layer_names:
            if name.endswith((".linear_attn", ".ple")):
                assert metadata[name].mamba_state_indices[0] == gid + 1
                expected_dtype = torch.bfloat16
                assert specs[name].dtypes[0] == expected_dtype


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("ssm_dtype", ["auto", "bfloat16", "float32"])
def test_gdn_planner_matches_existing_adapter(monkeypatch, tp, ssm_dtype):
    from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
        VllmGatedDeltaNetAttention,
    )
    from vllm_torchtpu.models.vllm.qwen4_exp.config import gdn_cache_spec

    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
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

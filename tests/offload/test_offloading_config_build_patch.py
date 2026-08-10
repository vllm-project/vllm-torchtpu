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
"""Unit tests for _patch_vllm_offloading_config_build.

The patch wraps vLLM's build_offloading_config so that offloading token
quantities span all PCP ranks (upstream only scales by DCP) and
tokens_per_hash matches the runtime (PCP-patched) resolve.
"""

from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheTensor,
                                        MambaSpec)

from vllm_torchtpu import (_patch_vllm_hybrid_pcp_block_sizes,
                           _patch_vllm_offloading_config_build)


def _vllm_config(*, pcp: int, block_size: int = 4096):
    return SimpleNamespace(
        model_config=SimpleNamespace(model="test-model", use_mla=False),
        cache_config=SimpleNamespace(
            block_size=block_size,
            cache_dtype="fp8",
            enable_prefix_caching=True,
            prefix_match_unit=None,
        ),
        parallel_config=SimpleNamespace(
            rank=0,
            world_size=pcp,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=1,
            data_parallel_index=0,
        ),
        kv_transfer_config=SimpleNamespace(
            engine_id="test-engine",
            kv_connector_extra_config={"cpu_bytes_to_use": 1 << 30},
        ),
        kv_events_config=None,
        use_v2_model_runner=False,
        speculative_config=None,
    )


def _kv_cache_config(block_size: int = 4096):
    attn = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.bfloat16,
    )
    mamba = MambaSpec(
        block_size=block_size,
        shapes=((3, 8192), ),
        dtypes=(torch.bfloat16, ),
        mamba_cache_mode="align",
    )
    return KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[
            KVCacheTensor(size=128 * 1024, shared_by=["attn"]),
            KVCacheTensor(size=128 * 1024, shared_by=["gdn"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], attn),
            KVCacheGroupSpec(["gdn"], mamba),
        ],
    )


def _patched_offloading_config_module():
    _patch_vllm_hybrid_pcp_block_sizes()
    _patch_vllm_offloading_config_build()
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import \
        config as offloading_config_module
    return offloading_config_module


def test_pcp_scales_tokens_per_block_and_hash():
    module = _patched_offloading_config_module()
    resolve_before = module.resolve_kv_cache_block_sizes

    vllm_config = _vllm_config(pcp=8)
    kv_cache_config = _kv_cache_config()
    config = module.build_offloading_config(vllm_config, kv_cache_config)

    # Logical (all-PCP-rank) token spans: 4096 rank-local * pcp 8.
    assert tuple(g.tokens_per_block for g in config.groups) == (32768, 32768)
    assert config.cache.tokens_per_hash == 32768
    # Per-rank physical bytes are untouched by the PCP scaling.
    assert config.worker_kv_bytes_per_block == (256 * 1024) // 128
    # The temporary unpatched-resolve swap must be restored.
    assert module.resolve_kv_cache_block_sizes is resolve_before


def test_no_pcp_is_passthrough():
    module = _patched_offloading_config_module()

    vllm_config = _vllm_config(pcp=1)
    kv_cache_config = _kv_cache_config()
    config = module.build_offloading_config(vllm_config, kv_cache_config)

    assert tuple(g.tokens_per_block for g in config.groups) == (4096, 4096)
    assert config.cache.tokens_per_hash == 4096


def test_patch_is_idempotent():
    module = _patched_offloading_config_module()
    build_first = module.build_offloading_config
    _patch_vllm_offloading_config_build()
    assert module.build_offloading_config is build_first

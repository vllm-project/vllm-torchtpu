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
tokens_per_hash matches the runtime (PCP-patched) resolve. Under DCP it also
scales groups that mix attention cache shapes (UniformTypeKVCacheSpecs).
"""

from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_torchtpu import (
    _patch_vllm_hybrid_pcp_block_sizes,
    _patch_vllm_offloading_config_build,
)


def _vllm_config(*, pcp: int, block_size: int = 4096, dcp: int = 1):
    return SimpleNamespace(
        model_config=SimpleNamespace(model="test-model", use_mla=dcp > 1),
        cache_config=SimpleNamespace(
            block_size=block_size,
            cache_dtype="fp8",
            enable_prefix_caching=True,
            prefix_match_unit=None,
            kv_cache_layout="LBNHC",
        ),
        parallel_config=SimpleNamespace(
            rank=0,
            world_size=max(pcp, dcp),
            tensor_parallel_size=dcp,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=dcp,
            cp_kv_cache_interleave_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=0,
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
        shapes=((3, 8192),),
        dtypes=(torch.bfloat16,),
        mamba_cache_mode="align",
    )
    return KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[
            KVCacheTensor(
                size=256 * 1024,
                layers=["attn"],
                layer_stride=128 * 1024,
                block_stride=1024,
            ),
            KVCacheTensor(
                size=256 * 1024,
                layers=["gdn"],
                layer_stride=128 * 1024,
                block_stride=1024,
                offset=128 * 1024,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], attn),
            KVCacheGroupSpec(["gdn"], mamba),
        ],
    )


def _mixed_mla_kv_cache_config(block_size: int = 256):
    """One group holding MLA latent and sparse indexer K pages (GLM-5.2)."""
    mla = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
    )
    indexer = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.uint8,
    )
    spec = UniformTypeKVCacheSpecs(
        block_size=block_size,
        kv_cache_specs={"attn": mla, "indexer": indexer},
    )
    return KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[
            KVCacheTensor(
                size=128 * spec.page_size_bytes,
                layers=["attn", "indexer"],
                layer_stride=128 * mla.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(["attn", "indexer"], spec)],
    )


def _patched_offloading_config_module():
    _patch_vllm_hybrid_pcp_block_sizes()
    _patch_vllm_offloading_config_build()
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
        config as offloading_config_module,
    )

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


def test_dcp_scales_mixed_mla_group():
    """A UniformTypeKVCacheSpecs group spans block_size * dcp tokens."""
    module = _patched_offloading_config_module()
    resolve_before = module.resolve_kv_cache_block_sizes

    vllm_config = _vllm_config(pcp=1, block_size=256, dcp=16)
    kv_cache_config = _mixed_mla_kv_cache_config()
    config = module.build_offloading_config(vllm_config, kv_cache_config)

    assert tuple(g.tokens_per_block for g in config.groups) == (4096,)
    assert config.cache.tokens_per_hash == 4096
    assert config.cache.blocks_per_chunk == 1
    assert config.parallel.dcp_size == 16
    # Per-rank physical bytes are untouched by the DCP scaling.
    assert config.worker_kv_bytes_per_block == (
        kv_cache_config.kv_cache_tensors[0].size // 128
    )
    assert module.resolve_kv_cache_block_sizes is resolve_before


def test_dcp_rescales_chunk_block_size():
    """A kv_connector_extra_config block_size counts logical DCP tokens."""
    module = _patched_offloading_config_module()

    vllm_config = _vllm_config(pcp=1, block_size=256, dcp=16)
    vllm_config.kv_transfer_config.kv_connector_extra_config = {
        "cpu_bytes_to_use": 1 << 30,
        "block_size": 8192,
    }
    config = module.build_offloading_config(vllm_config, _mixed_mla_kv_cache_config())

    assert tuple(g.tokens_per_block for g in config.groups) == (4096,)
    assert config.cache.blocks_per_chunk == 2

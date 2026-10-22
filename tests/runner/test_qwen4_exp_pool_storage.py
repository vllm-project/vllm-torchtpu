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
"""Actual pool allocation regression, runnable against the preceding source."""

import math
from types import SimpleNamespace as NS

import torch

from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
from tests.runner.test_qwen4_exp_pool import pool_config
from vllm_torchtpu.models.vllm.qwen4_exp.pooled_cache import Qwen4ExpPooledCache


def test_actual_pool_backing_does_not_allocate_main_sized_compressed_pages(monkeypatch):
    cfg = pool_config(kv="bfloat16")
    cfg.model_config.hf_text_config = q4exp_text_config(
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=256,
        indexer_head_dim=128,
        indexer_n_heads=1,
        indexer_compress_ratio=4,
        num_hidden_layers=2,
        layer_types=["linear_attention", "indexed_attention"],
        ple_layer_ids=[],
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    cfg.model_config.max_model_len = 512
    cfg.scheduler_config.max_num_seqs = 2
    runner = NS(
        vllm_config=cfg,
        cache_config=cfg.cache_config,
        kv_cache_dtype="bfloat16",
        device=torch.device("cpu"),
    )
    bound = []
    # No checkpoint/model computation: exercise the production planner,
    # allocator and binding with two logical layers, not a mocked allocator.
    layer0 = NS(layer_type="linear_attention", linear_attn=NS(kv_cache=None), ple=None)
    layer1 = NS(
        layer_type="indexed_attention",
        self_attn=NS(bind_pool=lambda *owners: bound.append(owners)),
        ple=None,
    )
    model = NS(model=NS(layers=[layer0, layer1]))
    monkeypatch.setattr(
        "vllm_torchtpu.models.vllm.qwen4_exp.pooled_cache.compute_hbm_budget",
        lambda *a, **kw: NS(available=1 << 40),
    )
    cache = Qwen4ExpPooledCache(runner, model)
    cfg.cache_config.block_size = cache.layout.block_size
    cfg.cache_config.mamba_block_size = cache.layout.block_size
    cfg.cache_config.num_gpu_blocks_override = cache.pages
    cache.allocate()
    assert bound
    # Two requests: four full-attention blocks, four recurrent checkpoints,
    # one non-cacheable ring each, plus the native null block. The compressed
    # index belongs to those same attention blocks, not four more blocks.
    expected_blocks = 1 + 2 * (math.ceil(512 / 128) * 2 + 1)
    main_per_block = 128 * 2 * 2 * 256 * 2
    index_per_block = max(8192, (128 // 4) * 128 * 2)
    expected_bytes = expected_blocks * (main_per_block + index_per_block)
    unique = {
        t.untyped_storage()._cdata: t.untyped_storage().nbytes()
        for t in cache.materialized.raw_tensors
    }
    assert sum(unique.values()) == expected_bytes
    assert cache.native_config.num_blocks == expected_blocks
    assert len(bound[0]) == 2

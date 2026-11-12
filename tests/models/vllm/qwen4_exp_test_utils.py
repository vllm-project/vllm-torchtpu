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
"""Checkpoint fixture values, kept out of production model defaults."""

from copy import deepcopy
from types import SimpleNamespace

_TEXT_CONFIG = {
    "norm_topk_prob": True,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "hidden_size": 2560,
    "num_hidden_layers": 48,
    "hc_count": 4,
    "hc_lowrank": 320,
    "vocab_size": 248320,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "output_gate_type": "sigmoid",
    "indexer_n_heads": 4,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 128,
    "indexer_budget": 2048,
    "indexer_compress_ratio": 4,
    "ple_layer_ids": [2],
    "ple_embed_dim": 2560,
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "ngram_vocab_size_base": 20000000,
    "make_ngram_vocab_size_divisible_by": 128,
    "split_ngram_parts": 128,
    "eos_token_id": 248044,
    "rms_norm_eps": 1e-06,
    "hidden_act": "silu",
    "partial_rotary_factor": 0.25,
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    },
    "max_position_embeddings": 262144,
    "layer_types": [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "indexed_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "indexed_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "indexed_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "indexed_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "indexed_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "indexed_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "indexed_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "indexed_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "indexed_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "indexed_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "indexed_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "indexed_attention",
    ],
    "attention_bias": False,
    "mtp_num_hidden_layers": 1,
    "mtp": {
        "num_hidden_layers": 1,
        "layer_types": ["full_attention"],
        "mtp_use_hidden_state_from_layer": None,
        "rope_theta": 10000000,
    },
    "mtp_use_dedicated_embeddings": False,
}


def q4exp_text_config(**overrides):
    values = deepcopy(_TEXT_CONFIG)
    values.update(overrides)
    if (
        "rope_parameters" not in overrides
        and ("head_dim" in overrides or "partial_rotary_factor" in overrides)
        and isinstance(values.get("rope_parameters"), dict)
        and "mrope_section" in values["rope_parameters"]
    ):
        half = int(values["head_dim"] * values["partial_rotary_factor"]) // 2
        if sum(values["rope_parameters"]["mrope_section"]) != half and half > 0:
            third = half // 3
            values["rope_parameters"]["mrope_section"] = [
                half - 2 * third,
                third,
                third,
            ]
    return SimpleNamespace(**values)


def qsa_test_pools(model, pages, *, block_size=128):
    """Actual 3-owner shapes from the configured native pool plan."""
    import math

    import torch
    from vllm.config.cache import CacheConfig

    from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs

    cfg = SimpleNamespace(
        speculative_config=None,
        model_config=SimpleNamespace(
            hf_text_config=model.config, dtype=torch.bfloat16, max_model_len=8192
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=model.tp_size),
        cache_config=CacheConfig(
            block_size=block_size,
            cache_dtype=model.cache_dtype,
            enable_prefix_caching=False,
            mamba_cache_mode="none",
        ),
    )
    layout, _ = pooled_cache_specs(cfg)
    packing = 4 if model.fp8_kv else 2
    main = torch.zeros(
        (
            pages,
            layout.block_size,
            math.ceil(2 * model.num_kv_heads / packing),
            packing,
            math.ceil(model.head_dim / 128) * 128,
        ),
        dtype=torch.float8_e4m3fn if model.fp8_kv else torch.bfloat16,
    )
    index = torch.zeros(
        (pages, layout.index_rows // 2, 1, 2, layout.compressed_width),
        dtype=torch.bfloat16,
    )
    ring = torch.zeros(
        (max(pages, 64), layout.ring_carrier_rows, 1, 2, layout.compressed_width),
        dtype=torch.bfloat16,
    )
    return (main, index, ring), layout


def qsa_forward_context(
    monkeypatch, model, metadata, *, request_distribution, main_block_table, state_slots
):
    from vllm_torchtpu.models.vllm.qwen4_exp import qsa

    main = SimpleNamespace(
        block_tables=main_block_table,
        request_distribution=request_distribution,
        qsa_ring_slots=state_slots,
    )
    monkeypatch.setattr(
        qsa,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={model.prefix: main}),
    )


def encode_pool_history(pool, history):
    """Independent CPU Mosaic packing for a BF16 history prefix in either carrier."""
    import torch

    assert pool.device.type == history.device.type == "cpu"
    count, width = pool.shape[0], pool.shape[-1]
    rows = (history[0].numel() + width - 1) // width
    padded = torch.zeros((count, rows * width), dtype=torch.bfloat16)
    padded[:, : history[0].numel()] = history.reshape(count, -1)
    if pool.dtype == torch.bfloat16:
        pool.reshape(count, -1)[:, : rows * width] = padded
    else:
        assert pool.dtype == torch.float8_e4m3fn
        bits = padded.view(torch.uint16).to(torch.int32).reshape(count, rows, width)
        planes = torch.stack((bits & 255, bits >> 8), dim=-2).to(torch.uint8)
        pool.view(torch.uint8).reshape(count, -1, 2, width)[:, :rows] = planes
    return pool


def decode_pool_history(pool, shape):
    """Read BF16 lane words without calling production region readers."""
    import torch

    count, width = pool.shape[0], pool.shape[-1]
    if pool.dtype == torch.bfloat16:
        values = pool.reshape(count, -1)
    else:
        assert pool.dtype == torch.float8_e4m3fn
        planes = pool.view(torch.uint8).reshape(count, -1, 2, width).to(torch.int32)
        words = (planes[:, :, 0] | (planes[:, :, 1] << 8)).to(torch.uint16)
        values = words.view(torch.bfloat16).reshape(count, -1)
    return values[:, : shape[1] * shape[2]].reshape(shape)

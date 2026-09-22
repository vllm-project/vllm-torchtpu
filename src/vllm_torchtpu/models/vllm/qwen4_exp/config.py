# SPDX-License-Identifier: Apache-2.0
"""Single admission boundary for the FP8, text-only TPU S1 implementation."""
import math

import torch
from vllm.model_executor.models.config import (Qwen4ExpForCausalLMConfig,
                                               VerifyAndUpdateConfig)

from vllm_torchtpu import envs


class TPUQwen4ExpConfig(VerifyAndUpdateConfig):

    @staticmethod
    def verify_and_update_config(vllm_config):
        # Reuse upstream's text/config normalization, including removal of
        # mrope even though this class truthfully does not support vision.
        Qwen4ExpForCausalLMConfig.verify_and_update_config(vllm_config)
        validate_config(vllm_config)
        vllm_config.cache_config.kv_cache_layout = "BLHNC"
        pages = 1 + vllm_config.scheduler_config.max_num_seqs * (
            math.ceil(vllm_config.model_config.max_model_len / 128) + 11)
        if vllm_config.cache_config.num_gpu_blocks_override not in (None,
                                                                    pages):
            raise ValueError(
                f"Qwen4Exp S1 fixes native capacity to {pages} blocks")
        vllm_config.cache_config.num_gpu_blocks_override = pages


def validate_config(cfg):
    m, p, c, s = (cfg.model_config, cfg.parallel_config, cfg.cache_config,
                  cfg.scheduler_config)
    t = m.hf_text_config
    quant = m.hf_config.quantization_config
    expected = dict(hidden_size=2560,
                    num_hidden_layers=48,
                    hc_count=4,
                    hc_lowrank=320,
                    vocab_size=248320,
                    num_attention_heads=24,
                    num_key_value_heads=2,
                    head_dim=256,
                    num_experts=512,
                    num_experts_per_tok=10,
                    moe_intermediate_size=640,
                    shared_expert_intermediate_size=640,
                    linear_num_key_heads=16,
                    linear_num_value_heads=48,
                    linear_key_head_dim=128,
                    linear_value_head_dim=128,
                    linear_conv_kernel_dim=4,
                    output_gate_type="sigmoid",
                    indexer_n_heads=4,
                    indexer_kv_heads=1,
                    indexer_head_dim=128,
                    indexer_budget=2048,
                    indexer_compress_ratio=4,
                    ple_layer_ids=[2],
                    ple_embed_dim=2560,
                    ple_conv_kernel_size=4,
                    ngram_size=3,
                    heads_per_ngram=8,
                    ngram_vocab_size_base=20000000,
                    make_ngram_vocab_size_divisible_by=128,
                    eos_token_id=248044,
                    rms_norm_eps=1e-6,
                    hidden_act="silu",
                    partial_rotary_factor=0.25)
    mismatches = {
        key: getattr(t, key)
        for key, value in expected.items() if getattr(t, key) != value
    }
    if t.rope_parameters["rope_theta"] != 10000000:
        raise ValueError("Qwen4Exp TPU S1 requires rope_theta=10000000")
    if mismatches or t.layer_types != [
            "full_attention" if i % 4 == 3 else "linear_attention"
            for i in range(48)
    ]:
        raise ValueError(
            f"Qwen4Exp TPU S1 checkpoint geometry mismatch: {mismatches}")
    if (m.quantization != "fp8" or quant["quant_method"] != "fp8"
            or quant["weight_block_size"] != [128, 128]
            or quant["activation_scheme"] != "dynamic"):
        raise ValueError(
            "Qwen4Exp TPU S1 requires the block128 FP8 checkpoint")
    if (p.tensor_parallel_size != 8 or p.data_parallel_size != 1
            or not p.enable_expert_parallel or p.pipeline_parallel_size != 1
            or p.prefill_context_parallel_size != 1
            or p.decode_context_parallel_size != 1 or p.enable_dbo
            or p.enable_eplb or p.use_sequence_parallel_moe):
        raise ValueError(
            "Qwen4Exp TPU S1 requires TP8/DP1/EP8 and PP/CP1, no DBO/EPLB")
    if (m.dtype != torch.bfloat16 or c.cache_dtype not in ("auto", "bfloat16")
            or c.block_size != 128 or not 1 <= m.max_model_len <= 8192
            or not 1 <= s.max_num_batched_tokens <= 128
            or not 1 <= s.max_num_seqs <= 64):
        raise ValueError(
            "Qwen4Exp TPU S1 requires BF16/P128, context<=8192, step<=128, seqs<=64"
        )
    if m.max_model_len < 128:
        raise ValueError(
            "Qwen4Exp TPU S1 requires context>=128 for full activation profiling"
        )
    if (c.enable_prefix_caching or s.async_scheduling
            or s.disable_hybrid_kv_cache_manager
            or cfg.speculative_config is not None
            or cfg.kv_transfer_config is not None
            or cfg.lora_config is not None
            or c.kv_cache_memory_bytes is not None):
        raise ValueError(
            "Qwen4Exp TPU S1 requires profiled hybrid caches, no APC/async/MTP/transfer/LoRA"
        )
    if (envs.TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL is True
            or envs.VLLM_TPU_BLOCK_MAJOR_KV):
        raise ValueError("Qwen4Exp TPU S1 requires private typed cache owners")
    engram = cfg.additional_config.get("qwen4_exp",
                                       {}).get("engram_config", {})
    if engram.get("cpu_offload", False) or engram.get("embedding_across_dp",
                                                      False):
        raise ValueError("Qwen4Exp TPU S1 requires resident PLE embedding TP8")

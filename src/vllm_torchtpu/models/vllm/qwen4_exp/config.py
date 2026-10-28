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
"""Single admission boundary for the FP8, text-only TPU S1 implementation."""

import math

import torch
from vllm.model_executor.models.config import (
    Qwen4ExpForCausalLMConfig,
    VerifyAndUpdateConfig,
)
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_torchtpu import envs
from vllm_torchtpu.gdn_pool_layout import (
    DEFAULT_POOLED_GDN_CONV_STATE_DTYPE,
    unified_kv_layout_enabled_for_architecture,
)
from vllm_torchtpu.kernels.qwen4_exp.hc_prefill_policy import validate_hc_prefill_mode
from vllm_torchtpu.utils import torch_dtype_from_name

from .ple_host import host_available_memory


def require_unified_pool() -> None:
    """Reject explicit opt-out before constructing Qwen4 cache owners."""
    if not unified_kv_layout_enabled_for_architecture("Qwen4ExpForCausalLM"):
        raise ValueError(
            "Qwen4Exp requires the dual-storage pool; remove "
            "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0 or set it to 1"
        )


def gdn_cache_spec(cfg):
    """Logical GDN geometry used before native pool materialization."""
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateDtypeCalculator,
        MambaStateShapeCalculator,
    )
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import MambaSpec

    t = cfg.model_config.hf_text_config
    _, ssm_dtype = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
        cfg.model_config.dtype,
        cfg.cache_config.mamba_cache_dtype,
        cfg.cache_config.mamba_ssm_cache_dtype,
    )
    conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        cfg.parallel_config.tensor_parallel_size,
        t.linear_num_key_heads,
        t.linear_num_value_heads,
        t.linear_key_head_dim,
        t.linear_value_head_dim,
        t.linear_conv_kernel_dim,
    )
    return MambaSpec(
        block_size=cfg.model_config.max_model_len,
        # The TPU adapter inserts a singleton axis before conv channels.
        shapes=(conv_shape[:-1] + (1, conv_shape[-1]), ssm_shape),
        # The Qwen4 pool stores BF16 projected-input history; SSM storage
        # follows the resolved upstream cache configuration.
        dtypes=(
            torch_dtype_from_name(DEFAULT_POOLED_GDN_CONV_STATE_DTYPE),
            ssm_dtype,
        ),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )


def native_cache_specs(cfg) -> dict[str, KVCacheSpec]:
    """Describe logical regions before native dual-store materialization."""
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import MambaSpec

    from .ple_geometry import ple_state_shape
    from .qsa import get_qsa_cache_specs

    t = cfg.model_config.hf_text_config
    main, auxiliary = get_qsa_cache_specs(
        t, cfg.parallel_config.tensor_parallel_size, cfg.cache_config.cache_dtype
    )
    specs: dict[str, KVCacheSpec] = {}
    for index, layer_type in enumerate(t.layer_types):
        prefix = f"model.layers.{index}"
        if layer_type == "linear_attention":
            specs[prefix + ".linear_attn"] = gdn_cache_spec(cfg)
        else:
            specs[prefix + ".self_attn"] = main
            specs.update(
                {
                    prefix + ".self_attn." + role: spec
                    for role, spec in auxiliary.items()
                }
            )
        if index + 1 in t.ple_layer_ids:
            specs[prefix + ".ple"] = MambaSpec(
                block_size=cfg.model_config.max_model_len,
                shapes=(ple_state_shape(t),),
                dtypes=(torch.bfloat16,),
                tp_replicated=True,
                mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
            )
    return specs


def native_cache_capacity(cfg, specs):
    """Size scheduler block IDs with the same upstream planner used at runtime."""
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

    groups = get_kv_cache_groups(cfg, specs)
    blocks_per_request = sum(
        group.kv_cache_spec.max_num_blocks_per_req(cfg, cfg.model_config.max_model_len)
        for group in groups
    )
    # One null block is reserved by the scheduler's pool.
    return 1 + cfg.scheduler_config.max_num_seqs * blocks_per_request


class TPUQwen4ExpConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config):
        # Reuse upstream's text/config normalization, including removal of
        # mrope even though this class truthfully does not support vision.
        Qwen4ExpForCausalLMConfig.verify_and_update_config(vllm_config)
        tile = envs.VLLM_TPU_HC_PREFILL_TILE
        validate_hc_prefill_mode("auto" if tile is None else tile, full_mix=True)
        validate_config(vllm_config)
        qwen = vllm_config.additional_config.get("qwen4_exp", {})
        if qwen.get("engram_config", {}).get("cpu_offload", False):
            # Host PLE lookup stays eager; its device producer and decoder
            # consumers use the existing piecewise compilation backend.
            compilation = vllm_config.compilation_config
            if compilation.splitting_ops is None:
                compilation.splitting_ops = []
            if "qwen4_exp::host_lookup" not in compilation.splitting_ops:
                compilation.splitting_ops.append("qwen4_exp::host_lookup")
        if qwen.get("engram_config", {}).get("dp_shared_memory", False):
            # This hook runs before engine copies / TPU worker DP collapse.
            # Runtime-only ModelConfig attributes survive deepcopy/pickling
            # but are outside its dataclass hash factors. A launch identity
            # in additional_config would invalidate compilation caches.
            vllm_config.model_config._qwen4_exp_ple_host_launch = dict(
                instance=vllm_config.instance_id,
                participants=vllm_config.parallel_config.data_parallel_size,
                checkpoint=vllm_config.model_config.model,
                revision=vllm_config.model_config.revision,
            )
        vllm_config.cache_config.kv_cache_layout = "BLHNC"
        from .pool_layout import pooled_cache_specs

        layout, specs = pooled_cache_specs(vllm_config)
        vllm_config.cache_config.block_size = layout.block_size
        # Upstream reserves this override for prefix caching. The model's
        # native pool specs carry the physical state-page span themselves.
        vllm_config.cache_config.mamba_block_size = (
            layout.block_size
            if vllm_config.cache_config.enable_prefix_caching
            else None
        )
        vllm_config.cache_config.mamba_page_size_padded = layout.page_bytes
        pages = native_cache_capacity(vllm_config, specs)
        if vllm_config.cache_config.num_gpu_blocks_override not in (None, pages):
            raise ValueError(f"Qwen4Exp S1 fixes native capacity to {pages} blocks")
        vllm_config.cache_config.num_gpu_blocks_override = pages


def validate_config(cfg):
    m, p, c, s = (
        cfg.model_config,
        cfg.parallel_config,
        cfg.cache_config,
        cfg.scheduler_config,
    )
    retired = sorted(
        cfg.additional_config.get("qwen4_exp", {}).keys()
        & {"qsa_prefill_backend", "qsa_sparse_backend", "qsa_dense_threshold"}
    )
    if retired:
        raise ValueError(
            f"Remove retired QSA options {retired}; prefill uses block-sparse "
            "attention automatically."
        )
    t = m.hf_text_config
    require_unified_pool()
    quant = m.hf_config.quantization_config
    # Model dimensions are input data. Only actual implementation contracts
    # belong here; do not admit a single checkpoint by matching its literals.
    positive = (
        "hidden_size",
        "num_hidden_layers",
        "hc_count",
        "hc_lowrank",
        "vocab_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "indexer_n_heads",
        "indexer_head_dim",
        "indexer_budget",
        "indexer_compress_ratio",
        "ple_embed_dim",
        "ple_conv_kernel_size",
        "heads_per_ngram",
        "ngram_vocab_size_base",
        "make_ngram_vocab_size_divisible_by",
        "max_position_embeddings",
    )
    if any(getattr(t, field) <= 0 for field in positive):
        raise ValueError("Qwen4Exp model geometry dimensions must be positive")
    if (
        len(t.layer_types) != t.num_hidden_layers
        or any(
            kind not in ("indexed_attention", "linear_attention")
            for kind in t.layer_types
        )
        or "indexed_attention" not in t.layer_types
        or any(not 1 <= layer <= t.num_hidden_layers for layer in t.ple_layer_ids)
    ):
        raise ValueError(
            "Qwen4Exp layer geometry requires valid layer types and one-based PLE IDs"
        )
    if (
        t.hc_count <= 1
        or t.output_gate_type != "sigmoid"
        or t.hidden_act != "silu"
        or t.attention_bias
        or t.num_experts_per_tok > t.num_experts
    ):
        raise ValueError(
            "Qwen4Exp requires HC, sigmoid gates, SiLU and unbiased "
            "projections with valid top-k"
        )
    if t.decoder_sparse_step != 1 or t.mlp_only_layers:
        raise ValueError(
            "Qwen4Exp TPU currently requires routed MoE in every decoder layer"
        )
    tp = p.tensor_parallel_size
    if tp <= 0 or p.data_parallel_size <= 0:
        raise ValueError("Qwen4Exp parallel sizes must be positive")
    expert_shards = tp * p.data_parallel_size if p.enable_expert_parallel else 1
    if (
        t.num_experts % expert_shards
        or (not p.enable_expert_parallel and t.moe_intermediate_size % tp)
        or t.shared_expert_intermediate_size % tp
    ):
        raise ValueError(
            "Qwen4Exp experts and intermediate must divide their parallel groups"
        )
    if (
        t.linear_num_key_heads % tp
        or t.linear_num_value_heads % tp
        or t.linear_num_value_heads % t.linear_num_key_heads
    ):
        raise ValueError(
            "Qwen4Exp GDN requires integral TP heads and value/key head groups"
        )
    # The selected GDN prefill loader slices whole 128-lane head tiles.
    # The private-state wrapper also requires non-empty convolution history.
    if (
        t.linear_key_head_dim % 128
        or t.linear_value_head_dim % 128
        or t.linear_conv_kernel_dim <= 1
    ):
        raise ValueError(
            "Qwen4Exp GDN requires head dimensions divisible by 128 and conv kernel > 1"
        )
    if (
        t.num_attention_heads % t.num_key_value_heads
        or t.num_attention_heads % tp
        or (
            t.num_key_value_heads % tp
            if t.num_key_value_heads >= tp
            else tp % t.num_key_value_heads
        )
    ):
        raise ValueError(
            "Qwen4Exp QSA requires integral query/KV head sharding or replication"
        )
    rotary_dim = int(t.head_dim * t.partial_rotary_factor)
    theta = t.rope_parameters["rope_theta"]
    if (
        t.indexer_kv_heads != 1
        or c.block_size % t.indexer_compress_ratio
        or t.indexer_budget % t.indexer_compress_ratio
        or rotary_dim <= 0
        or rotary_dim % 2
        or rotary_dim > min(t.head_dim, t.indexer_head_dim)
        or not math.isfinite(theta)
        or theta <= 0
        or not math.isfinite(t.rms_norm_eps)
        or t.rms_norm_eps <= 0
        or (c.cache_dtype in ("fp8", "fp8_e4m3") and t.head_dim % 2)
    ):
        raise ValueError(
            "Qwen4Exp QSA geometry violates compression, RoPE or FP8 packing "
            "constraints"
        )
    if m.max_model_len > t.max_position_embeddings:
        raise ValueError("Qwen4Exp context exceeds model position capacity")
    if (
        t.ngram_size < 2
        or t.ple_embed_dim % ((t.ngram_size - 1) * t.heads_per_ngram)
        or t.ple_embed_dim % 4
        or t.ple_conv_kernel_size <= 1
        or not isinstance(t.eos_token_id, int)
    ):
        raise ValueError(
            "Qwen4Exp PLE geometry requires complete heads, packed words and "
            "convolution history"
        )
    if (
        m.quantization != "fp8"
        or quant["quant_method"] != "fp8"
        or quant["weight_block_size"] != [128, 128]
        or quant["activation_scheme"] != "dynamic"
    ):
        raise ValueError("Qwen4Exp TPU S1 requires the block128 FP8 checkpoint")
    # use_sequence_parallel_moe is derived by upstream before this model
    # hook. TPU's existing platform_activation patch disables it afterward;
    # rejecting that intermediate property would block every mixed TP/DP.
    if (
        p.pipeline_parallel_size != 1
        or p.prefill_context_parallel_size != 1
        or p.decode_context_parallel_size != 1
        or p.enable_dbo
        or p.enable_eplb
        or p.expert_placement_strategy != "linear"
        or p.enable_elastic_ep
    ):
        raise ValueError(
            "Qwen4Exp TPU requires PP/CP1, "
            "linear expert placement and no DBO/EPLB/elastic EP"
        )
    # SparseCore top-k writes 16-lane vectors; this is a kernel boundary,
    # independent of checkpoint head counts and compression geometry.
    block_topk = t.indexer_budget // t.indexer_compress_ratio
    if c.block_size // t.indexer_compress_ratio % 2:
        raise ValueError("Qwen4Exp BF16 index pages require pairs of compressed rows")
    if block_topk % 16 or not 16 <= block_topk <= 4096:
        raise ValueError(
            "Qwen4Exp StreamIndex top-k must be aligned to 16 in [16,4096]"
        )
    if c.cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
        raise ValueError(
            "Qwen4Exp QSA requires BF16 or per-tensor FP8 E4M3 KV (auto, "
            "bfloat16, fp8, fp8_e4m3)"
        )
    # Request slots are sized from max_num_seqs and checked against HBM in
    # Qwen4ExpPooledCache.allocate. Active work remains bounded by the token budget;
    # the initial 64-slot bring-up was a test configuration, not a kernel cap.
    if (
        m.dtype != torch.bfloat16
        or c.block_size != 128
        or not 1 <= s.max_num_batched_tokens <= 8192
        or s.max_num_seqs < 1
    ):
        raise ValueError(
            "Qwen4Exp TPU S1 requires BF16/P128, step<=8192 and positive max_num_seqs"
        )
    if m.max_model_len < 128:
        raise ValueError(
            "Qwen4Exp TPU S1 requires context>=128 for full activation profiling"
        )
    if (
        s.disable_hybrid_kv_cache_manager
        or cfg.speculative_config is not None
        or cfg.lora_config is not None
        or c.kv_cache_memory_bytes is not None
    ):
        raise ValueError(
            "Qwen4Exp TPU requires profiled hybrid caches, and no MTP/LoRA"
        )
    transfer = cfg.kv_transfer_config
    if transfer is not None:
        connector_module = "vllm_torchtpu.models.vllm.qwen4_exp.pd_connector"
        if (
            (p.tensor_parallel_size, p.data_parallel_size) != (1, 8)
            or not p.enable_expert_parallel
            or transfer.kv_connector != "Qwen4ExpRaidenConnector"
            or transfer.kv_connector_module_path not in (None, connector_module)
            or transfer.kv_role not in ("kv_producer", "kv_consumer")
            or c.cache_dtype not in ("fp8", "fp8_e4m3")
            or c.enable_prefix_caching
            or c.mamba_ssm_cache_dtype not in ("bfloat16", "float32")
            or envs.TPU_KV_RESHARD_TRANSPORT != "raiden"
            or any(
                (
                    envs.TPU_RAIDEN_QWEN35_ADMISSION,
                    envs.TPU_RAIDEN_KIMIK3_ADMISSION,
                    envs.TPU_RAIDEN_GLM_ADMISSION,
                    envs.TPU_RAIDEN_DSV4_ADMISSION,
                )
            )
        ):
            raise ValueError(
                "Qwen4Exp PD requires its model-owned Raiden connector, a producer "
                "or consumer role, DP8/TP1/EP8, unified FP8 pages, "
                "BF16/FP32 SSM and APC off"
            )
    if envs.VLLM_TPU_BLOCK_MAJOR_KV:
        raise ValueError("Qwen4Exp requires its native HEAD_ALONG_SUBLANE cache layout")
    if c.enable_prefix_caching and c.mamba_cache_mode != "align":
        raise ValueError("Qwen4Exp unified pool requires mamba_cache_mode='align'")
    engram = cfg.additional_config.get("qwen4_exp", {}).get("engram_config", {})
    for flag in ("cpu_offload", "embedding_across_dp", "dp_shared_memory"):
        if type(engram.get(flag, False)) is not bool:
            raise ValueError(f"Qwen4Exp engram_config.{flag} must be bool")
    offload = engram.get("cpu_offload", False)
    shared = engram.get("dp_shared_memory", False)
    across = engram.get("embedding_across_dp", False)
    if shared and not offload:
        raise ValueError("Qwen4Exp dp_shared_memory requires cpu_offload")
    if shared and (
        across
        or p.data_parallel_size <= 1
        or p.nnodes != 1
        or p.data_parallel_size_local != p.data_parallel_size
    ):
        raise ValueError(
            "Qwen4Exp shared PLE requires local DP>1, one host, and "
            "embedding_across_dp=false"
        )
    from .ple_geometry import ple_vocab_rows

    etp = tp * p.data_parallel_size if across else tp
    ple_width = t.ple_embed_dim // ((t.ngram_size - 1) * t.heads_per_ngram)
    ple_rows = [ple_vocab_rows(t, layer - 1) for layer in sorted(set(t.ple_layer_ids))]
    for rows in ple_rows:
        if (
            rows % etp
            or (rows // etp * ple_width) % 128
            or rows >= 2**31
            or rows // etp * ple_width // 128 >= 2**31
        ):
            raise ValueError(
                "Qwen4Exp PLE table must fit aligned embedding shards and INT32 indices"
            )
    if offload and len(ple_rows) > 1:
        raise ValueError(
            "Qwen4Exp host offload currently requires at most one PLE layer"
        )
    if offload:
        if p.nnodes != 1 or p.data_parallel_size_local != p.data_parallel_size:
            raise ValueError("Qwen4Exp host PLE requires local DP on one host")
        # BF16 upper bound, including all DP replicas for private TP tables.
        # The 8-worker loader has measured ~135GiB of aggregate peak RSS
        # beyond PLE staging; reserve160GiB as a conservative launch allowance.
        copies = 1 if shared or across else p.data_parallel_size
        required = sum(ple_rows) * ple_width * 2 * copies + 160 * 1024**3
        available = host_available_memory()
        if required > available:
            raise ValueError(
                f"Qwen4Exp host PLE capacity estimate needs {required} bytes "
                f"including loader allowance; only {available} bytes available. "
                "Use dp_shared_memory or embedding_across_dp to avoid DP replicas."
            )

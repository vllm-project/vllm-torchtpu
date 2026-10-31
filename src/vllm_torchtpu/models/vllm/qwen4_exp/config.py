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
import os

import torch
from vllm.model_executor.models.config import (
    Qwen4ExpForCausalLMConfig,
    VerifyAndUpdateConfig,
)
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_torchtpu import envs
from vllm_torchtpu.gdn_pool_layout import unified_kv_layout_enabled_for_architecture

from .ple_host import host_available_memory

# Model-owned prefill/decode disaggregation (see pd_connector.py).
PD_CONNECTOR = "Qwen4ExpRaidenConnector"
PD_CONNECTOR_MODULE = "vllm_torchtpu.models.vllm.qwen4_exp.pd_connector"
# Host-DRAM prefix-cache offload through the shared Raiden store.
OFFLOAD_CONNECTOR = "TPURaidenOffloadingConnector"
OFFLOAD_CONNECTOR_MODULE = "vllm_torchtpu.offload.raiden_connector"


def pd_transfer_enabled(cfg) -> bool:
    """Whether this engine is one side of a Qwen4Exp PD pair."""
    transfer = getattr(cfg, "kv_transfer_config", None)
    return getattr(transfer, "kv_connector", None) == PD_CONNECTOR


def host_offload_enabled(cfg) -> bool:
    """Whether prefix-cache pages are offloaded to host DRAM (Raiden store)."""
    transfer = getattr(cfg, "kv_transfer_config", None)
    return getattr(transfer, "kv_connector", None) == OFFLOAD_CONNECTOR


def use_unified_pool() -> bool:
    """Use the shared architecture policy, including an explicit global override."""
    return unified_kv_layout_enabled_for_architecture("Qwen4ExpForCausalLM")


def num_speculative_tokens(cfg) -> int:
    """Draft tokens verified per step; 0 when speculative decoding is off."""
    spec = getattr(cfg, "speculative_config", None)
    return 0 if spec is None else int(spec.num_speculative_tokens)


def mtp_layer_prefixes(cfg) -> list[str]:
    """Module prefixes of the checkpoint MTP layers the draft model owns.

    Like upstream, MTP layer ``i`` is numbered after the target's layers so its
    cache and forward-context names never collide with a target layer.
    """
    if num_speculative_tokens(cfg) == 0:
        return []
    t = cfg.model_config.hf_text_config
    count = int(getattr(t, "mtp_num_hidden_layers", 0) or 0)
    return [f"mtp.layers.{t.num_hidden_layers + i}" for i in range(count)]


def qsa_ring_rows(t) -> int:
    """Raw indexer-key rows each request owns (TPU eight-row alignment)."""
    return (int(t.indexer_compress_ratio) + 7) // 8 * 8


def max_qsa_speculative_tokens(t) -> int:
    """Largest verify window whose rejected tokens cannot evict ring history.

    The raw ring holds the newest ``rows`` positions modulo ``rows``. After a
    step that wrote ``[c, c + k + 1)`` and accepted at least one token, the
    next step re-reads the partial compression group before its new context
    ``c' >= c + 1``, i.e. positions ``>= c' - ratio + 1``. Every position
    written since lies below ``c' + k``, so no lane aliases while
    ``rows >= k + ratio - 1``.
    """
    return qsa_ring_rows(t) - int(t.indexer_compress_ratio) + 1


def gdn_cache_spec(cfg):
    """Match the existing TPU GDN adapter's private-cache contract."""
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
        # The existing non-pooled TPU adapter stores convolution history in
        # FP32; SSM storage follows the resolved upstream cache configuration.
        dtypes=(torch.float32, ssm_dtype),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        # Rollback keeps one whole-state checkpoint block per verify position.
        num_speculative_blocks=num_speculative_tokens(cfg),
    )


def native_cache_specs(cfg) -> dict[str, KVCacheSpec]:
    """Describe model-owned caches before allocating any model weights."""
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import MambaSpec

    from .ple_geometry import ple_state_shape
    from .qsa import get_qsa_cache_specs

    t = cfg.model_config.hf_text_config
    main, auxiliary = get_qsa_cache_specs(
        t, cfg.parallel_config.tensor_parallel_size, cfg.cache_config.cache_dtype
    )
    specs: dict[str, KVCacheSpec] = {}

    def add_qsa(prefix):
        specs[prefix + ".self_attn"] = main
        specs.update(
            {prefix + ".self_attn." + role: spec for role, spec in auxiliary.items()}
        )

    for index, layer_type in enumerate(t.layer_types):
        prefix = f"model.layers.{index}"
        if layer_type == "linear_attention":
            specs[prefix + ".linear_attn"] = gdn_cache_spec(cfg)
        else:
            add_qsa(prefix)
        if index + 1 in t.ple_layer_ids:
            specs[prefix + ".ple"] = MambaSpec(
                block_size=cfg.model_config.max_model_len,
                shapes=(ple_state_shape(t),),
                dtypes=(torch.bfloat16,),
                tp_replicated=True,
                mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
                # PLE conv history checkpoints per verify position, like GDN.
                num_speculative_blocks=num_speculative_tokens(cfg),
            )
    # The MTP draft's full-attention QSA layers own ordinary request pages in
    # the same native pool (main KV, compressed keys and raw ring).
    for prefix in mtp_layer_prefixes(cfg):
        add_qsa(prefix)
    return specs


def native_cache_capacity(cfg, specs=None):
    """Size scheduler block IDs with the same upstream planner used at runtime."""
    from vllm.v1.core.kv_cache_utils import (
        _get_kv_cache_bytes_per_block,
        get_kv_cache_groups,
    )

    groups = get_kv_cache_groups(
        cfg, native_cache_specs(cfg) if specs is None else specs
    )
    blocks_per_request = sum(
        group.kv_cache_spec.max_num_blocks_per_req(cfg, cfg.model_config.max_model_len)
        for group in groups
    )
    # One null block is reserved by the scheduler's pool.
    pages = 1 + cfg.scheduler_config.max_num_seqs * blocks_per_request
    bytes_per_block = _get_kv_cache_bytes_per_block(groups)
    max_pool_bytes = int(os.environ.get("QWEN4_EXP_MAX_POOL_BYTES", str(48 * 1024**3)))
    if bytes_per_block > 0 and max_pool_bytes > 0:
        max_pages = max_pool_bytes // bytes_per_block
        if max_pages > 1:
            pages = min(pages, max_pages)
    return pages


# Default reserve beside the pool (and its graph-time copies) for
# activations and the program regions of every compiled bucket. The
# 8192-token profiling forward needs ~1.3 GiB of non-pool temporaries; with
# one graph copy this plans ~36.6 GiB on v7x, the 36 GiB pool already
# serving AgentX.
POOL_HEADROOM_BYTES = 6 * 1024**3


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def hbm_pool_sizing(cfg) -> bool:
    """Size the unified pool from profiled HBM instead of a static formula.

    The default for the unified pool. ``QWEN4_EXP_POOL_SIZING=static``
    restores ``native_cache_capacity``; PD keeps it (both sides must agree
    on the pool before profiling), and so does an explicit
    ``--num-gpu-blocks-override``. Host offload sizes from HBM like a plain
    server: its host capacity follows the planned pool, not the reverse.
    """
    return (
        use_unified_pool()
        and (
            getattr(cfg, "kv_transfer_config", None) is None
            or host_offload_enabled(cfg)
        )
        and getattr(cfg.cache_config, "num_gpu_blocks_override", None) is None
        and os.environ.get("QWEN4_EXP_POOL_SIZING", "hbm").strip().lower() != "static"
    )


def pool_graph_copies() -> int:
    """Extra pool-sized buffers to reserve per compiled forward; none by default.

    The pooled ops donate the pool and write it back with ``copy_``, so the
    compiled forward aliases its pool input to its output and holds a single
    pool. A conditional that returns the pool breaks that aliasing: QSA once
    routed mixed and prefill buckets through a ``lax.cond`` carrying it, and
    each of those programs needed a second full pool (the 40 GiB-pool profile
    needed 96.89 GiB = 15.6 GiB resident + 40 + 40). Set
    ``QWEN4_EXP_POOL_GRAPH_COPIES=1`` to reserve such a copy again.
    """
    return max(0, _env_int("QWEN4_EXP_POOL_GRAPH_COPIES", 0))


def pool_headroom_bytes() -> int:
    return max(0, _env_int("QWEN4_EXP_POOL_HEADROOM_BYTES", POOL_HEADROOM_BYTES))


def pooled_pool_blocks(
    bytes_per_block: int,
    *,
    cap_available: int,
    physical_available: int,
    min_blocks: int,
) -> int:
    """Unified pool blocks that fit the measured HBM.

    ``cap_available``: bytes free under ``gpu_memory_utilization`` with no
    pool allocated; ``physical_available``: bytes free on the device. The
    pool plus the headroom must fit the former; the pool, its graph-time
    copies and the headroom the latter. ``QWEN4_EXP_MAX_POOL_BYTES`` caps
    the pool when set (no cap by default).
    """
    headroom = pool_headroom_bytes()
    budget = min(
        cap_available - headroom,
        (physical_available - headroom) // (1 + pool_graph_copies()),
    )
    cap = _env_int("QWEN4_EXP_MAX_POOL_BYTES", 0)
    if cap > 0:
        budget = min(budget, cap)
    blocks = max(0, budget) // bytes_per_block
    if blocks < min_blocks:
        raise ValueError(
            f"Qwen4Exp unified pool fits {blocks} blocks in the measured HBM "
            f"(cap_available={cap_available}, physical={physical_available}, "
            f"headroom={headroom}, graph copies={pool_graph_copies()}), fewer "
            f"than the {min_blocks} one max_model_len request needs; lower "
            "max_model_len or the headroom"
        )
    return blocks


class TPUQwen4ExpConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config):
        # Reuse upstream's text/config normalization, including removal of
        # mrope even though this class truthfully does not support vision.
        Qwen4ExpForCausalLMConfig.verify_and_update_config(vllm_config)
        validate_config(vllm_config)
        qwen = vllm_config.additional_config.get("qwen4_exp", {})
        if qwen.get("engram_config", {}).get("cpu_offload", False):
            # Host PLE lookup stays eager; its device producer and decoder
            # consumers use the existing piecewise compilation backend.
            compilation = vllm_config.compilation_config
            if compilation.splitting_ops is None:
                compilation.splitting_ops = []
            ops = ["qwen4_exp::host_lookup"]
            if envs.VLLM_TPU_PLE_HOST_OVERLAP:
                ops += ["qwen4_exp::host_lookup_start", "qwen4_exp::host_lookup_finish"]
            for op in ops:
                if op not in compilation.splitting_ops:
                    compilation.splitting_ops.append(op)
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
        if use_unified_pool():
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
            if hbm_pool_sizing(vllm_config):
                # Workers report the pool budget after profiling
                # (Qwen4ExpPooledCache.memory_budget); vLLM plans the blocks.
                return
            pages = native_cache_capacity(vllm_config, specs)
        else:
            pages = native_cache_capacity(vllm_config)
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
    pooled = use_unified_pool()
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
            kind not in ("full_attention", "linear_attention") for kind in t.layer_types
        )
        or "full_attention" not in t.layer_types
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
    from .pool_layout import MAIN_KV_DCP_ENV, main_kv_dcp

    # The QSA main-KV split over TP rank pairs is this model's own path (vLLM's
    # decode_context_parallel_size stays 1, rejected above): it shards the
    # unified pool's main-KV pages, so it needs that pool and pairs of ranks.
    dcp = main_kv_dcp()
    if dcp > 1 and (
        not use_unified_pool()
        or p.tensor_parallel_size < dcp * t.num_key_value_heads
        or p.tensor_parallel_size % (dcp * t.num_key_value_heads)
    ):
        raise ValueError(
            f"{MAIN_KV_DCP_ENV}={dcp} splits each replicated KV head across "
            f"{dcp}-rank TP pairs on the unified pool: requires the unified "
            f"pool and tensor_parallel_size divisible by "
            f"{dcp} * num_key_value_heads ({dcp * t.num_key_value_heads}), got "
            f"tensor_parallel_size={p.tensor_parallel_size} and "
            f"num_key_value_heads={t.num_key_value_heads}; set "
            f"tensor_parallel_size to a multiple of "
            f"{dcp * t.num_key_value_heads} or unset {MAIN_KV_DCP_ENV}"
        )
    raw_mk = str(envs.VLLM_TPU_MEGAKERNEL or "").strip().lower()
    if dcp > 1 and raw_mk and raw_mk not in ("0", "false", "off"):
        raise ValueError(
            f"{MAIN_KV_DCP_ENV}={dcp} and VLLM_TPU_MEGAKERNEL cannot be "
            "enabled together: the megakernel's select op and whole-step "
            "cache writer assume replicated main-KV pages; unset "
            f"VLLM_TPU_MEGAKERNEL or {MAIN_KV_DCP_ENV}"
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
    # Qwen4ExpCache.allocate. Active work remains bounded by the token budget;
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
        (c.enable_prefix_caching and not pooled)
        or s.disable_hybrid_kv_cache_manager
        or cfg.lora_config is not None
        or c.kv_cache_memory_bytes is not None
    ):
        raise ValueError(
            "Qwen4Exp TPU requires profiled hybrid caches, APC only with unified pool, "
            "and no LoRA"
        )
    if cfg.speculative_config is not None:
        validate_mtp_config(cfg)
    transfer = cfg.kv_transfer_config
    if host_offload_enabled(cfg):
        validate_host_offload_config(cfg)
    elif transfer is not None:
        connector_module = PD_CONNECTOR_MODULE
        if (
            (p.tensor_parallel_size, p.data_parallel_size) != (1, 8)
            or not p.enable_expert_parallel
            or transfer.kv_connector != PD_CONNECTOR
            or transfer.kv_connector_module_path not in (None, connector_module)
            or transfer.kv_role not in ("kv_producer", "kv_consumer")
            or not pooled
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
    if pooled and c.enable_prefix_caching and c.mamba_cache_mode != "align":
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


def validate_host_offload_config(cfg):
    """Admit host-DRAM prefix-cache offload (TPURaidenOffloadingConnector).

    The connector moves whole unified-pool pages, one per (cache group,
    block), as raw bytes: main KV, compressed keys (packed or not) and the
    GDN/PLE align checkpoints round-trip unchanged whatever their dtype or
    shape (see ``offload.offload_pool_geometry``). That needs the unified
    pool, prefix caching with align-mode Mamba checkpoints (a host hit, like
    an HBM hit, resumes from the state saved at a block boundary, where the
    per-request QSA raw ring holds no history yet) and both store and load
    roles on one engine.
    """
    c, p = cfg.cache_config, cfg.parallel_config
    transfer = cfg.kv_transfer_config
    extra = transfer.kv_connector_extra_config or {}
    if (
        transfer.kv_connector_module_path not in (None, OFFLOAD_CONNECTOR_MODULE)
        or transfer.kv_role != "kv_both"
        or not use_unified_pool()
        or not c.enable_prefix_caching
        or c.mamba_cache_mode != "align"
        or p.pipeline_parallel_size != 1
    ):
        raise ValueError(
            "Qwen4Exp host KV offload requires "
            f"{OFFLOAD_CONNECTOR} from {OFFLOAD_CONNECTOR_MODULE} with "
            "kv_role='kv_both', the unified pool, prefix caching, "
            "mamba_cache_mode='align' and PP1"
        )
    cpu_bytes = extra.get("cpu_bytes_to_use")
    try:
        cpu_bytes = int(cpu_bytes)
    except (TypeError, ValueError):
        cpu_bytes = 0
    if cpu_bytes <= 0:
        raise ValueError(
            "Qwen4Exp host KV offload requires a positive "
            "kv_connector_extra_config.cpu_bytes_to_use (host bytes summed "
            "over all workers of the engine)"
        )
    if extra.get("blocks_per_chunk") not in (None, 1) or "block_size" in extra:
        raise ValueError(
            "Qwen4Exp host KV offload stores one pool page per scheduler block; "
            "leave blocks_per_chunk/block_size unset"
        )


def validate_mtp_config(cfg):
    """Admit only the checkpoint's native MTP draft on the unified pool.

    GDN and PLE roll back rejected drafts by selecting a per-position state
    checkpoint block, which only the unified pool allocates; the compact
    (non-pooled) cache has no checkpoint slots. The draft shares the target's
    TP layout (embeddings, LM head and QSA head sharding).
    """
    spec = cfg.speculative_config
    t = cfg.model_config.hf_text_config
    p = cfg.parallel_config
    k = int(spec.num_speculative_tokens or 0)
    draft_tp = getattr(spec, "draft_tensor_parallel_size", None)
    mtp = getattr(t, "mtp", None) or {}
    mtp_types = list(mtp.get("layer_types", ["full_attention"]))
    if spec.method != "mtp":
        raise ValueError(
            f"Qwen4Exp TPU speculative decoding supports only method='mtp', "
            f"got {spec.method!r}"
        )
    if not use_unified_pool():
        raise ValueError("Qwen4Exp MTP requires the unified block pool")
    if int(getattr(t, "mtp_num_hidden_layers", 0) or 0) != 1 or mtp_types != [
        "full_attention"
    ]:
        raise ValueError(
            "Qwen4Exp TPU MTP requires exactly one full-attention MTP layer"
        )
    if getattr(t, "mtp_use_dedicated_embeddings", False):
        raise ValueError("Qwen4Exp TPU MTP shares the target embeddings and LM head")
    if not 1 <= k <= max_qsa_speculative_tokens(t):
        raise ValueError(
            "Qwen4Exp MTP num_speculative_tokens must be in "
            f"[1, {max_qsa_speculative_tokens(t)}] so rejected drafts cannot "
            "evict QSA raw-ring history"
        )
    if draft_tp not in (None, p.tensor_parallel_size):
        raise ValueError(
            "Qwen4Exp MTP requires draft_tensor_parallel_size equal to the "
            "target tensor_parallel_size"
        )
    if p.data_parallel_size != 1:
        raise ValueError("Qwen4Exp TPU MTP currently requires data_parallel_size=1")
    if pd_transfer_enabled(cfg):
        raise ValueError("Qwen4Exp PD requires MTP off")
    if getattr(cfg.cache_config, "mamba_cache_mode", "none") not in ("none", "align"):
        raise ValueError("Qwen4Exp MTP requires mamba_cache_mode 'none' or 'align'")

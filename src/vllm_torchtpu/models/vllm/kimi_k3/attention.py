# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi MLA and KDA layers for TPU."""

from __future__ import annotations

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.forward_context import (get_forward_context,
                                  is_forward_context_available)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator, MambaStateShapeCalculator,
    is_conv_state_dim_first)
from vllm.model_executor.layers.mla import (MLAModules,
                                            MultiHeadLatentAttentionWrapper)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, sharded_weight_loader)
from vllm.model_executor.utils import set_weight_attrs
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_torchtpu.layers.common.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.vllm.custom_ops.kda_attention_op import (
    build_kimi_kda_op, build_kimi_sconv_op)


def _load_a_log(parameter: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Load either the old ``[1, 1, H, 1]`` or current ``[H]`` layout."""

    if loaded_weight.ndim == 4:
        loaded_weight = loaded_weight.flatten()
    rank = get_tensor_model_parallel_rank()
    shard_size = parameter.shape[0]
    loaded_weight = loaded_weight.narrow(0, rank * shard_size, shard_size)
    default_weight_loader(parameter, loaded_weight)


class MultiHeadLatentAttention(nn.Module):
    """NoPE MLA dispatched through the TPU Pallas MLA custom op."""

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        if not config.mla_use_nope:
            raise ValueError("The TPU Kimi model supports NoPE MLA only")
        required = {
            "kv_lora_rank": config.kv_lora_rank,
            "qk_nope_head_dim": config.qk_nope_head_dim,
            "qk_rope_head_dim": config.qk_rope_head_dim,
            "v_head_dim": config.v_head_dim,
        }
        if any(value is None for value in required.values()):
            raise ValueError(f"Incomplete Kimi K3 MLA config: {required}")
        assert config.kv_lora_rank is not None
        assert config.qk_nope_head_dim is not None
        assert config.qk_rope_head_dim is not None
        assert config.v_head_dim is not None

        tp_size = get_tensor_model_parallel_world_size()
        if config.num_attention_heads % tp_size:
            raise ValueError(
                "num_attention_heads must be divisible by TP size")
        num_heads = config.num_attention_heads // tp_size
        qk_head_dim = int(config.qk_nope_head_dim) + int(
            config.qk_rope_head_dim)
        quant_config = vllm_config.quant_config
        super().__init__()

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.v_head_dim = int(config.v_head_dim)
        self.qk_head_dim = qk_head_dim

        if self.q_lora_rank is None:
            self.q_proj = ColumnParallelLinear(
                config.hidden_size,
                config.num_attention_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
            self.fused_qkv_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None
        else:
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                config.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                config.num_attention_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
            self.q_proj = None
            self.kv_a_proj_with_mqa = None
        if self.q_lora_rank is None:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                config.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            config.num_attention_heads *
            (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.g_proj = (ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * self.v_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj",
        ) if getattr(config, "mla_use_output_gate", False) else None)
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=None,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=self.q_proj,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
        )
        # The TPU MLA wrapper supports K3's optional output gate, while the
        # current upstream MLAModules dataclass does not expose that field.
        # Attach it as backend-specific metadata instead of passing a stale
        # constructor argument.
        mla_modules.g_proj = self.g_proj
        self.mla_attn = MultiHeadLatentAttentionWrapper(
            config.hidden_size,
            num_heads,
            qk_head_dim**-0.5,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            vllm_config.cache_config,
            quant_config,
            prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states)


class CausalDepthwiseConv1d(nn.Module):
    """Checkpoint-compatible depthwise-convolution weight container."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if channels % tp_size:
            raise ValueError(
                "Convolution channels must be divisible by TP size")
        self.channels = channels // tp_size
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(self.channels, 1, kernel_size))
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def weight_loader(
        self,
        parameter: nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        if loaded_weight.ndim == 2:
            loaded_weight = loaded_weight.unsqueeze(1)
        rank = get_tensor_model_parallel_rank()
        loaded_weight = loaded_weight.chunk(
            get_tensor_model_parallel_world_size(), dim=0)[rank]
        parameter.data.copy_(loaded_weight)


class KimiDeltaAttention(nn.Module, MambaBase):
    """KDA registered with vLLM and executed by TPU Pallas custom ops."""

    supports_dcp = False

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "TPU KDA does not support speculative decode")
        kda_config = config.linear_attn_config
        if kda_config is None:
            raise ValueError("KDA requires linear_attn_config")

        self.prefix = prefix
        self.vllm_config = vllm_config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.head_dim = int(kda_config["head_dim"])
        self.total_heads = int(kda_config["num_heads"])
        if self.total_heads % self.tp_size:
            raise ValueError("KDA num_heads must be divisible by TP size")
        self.num_heads = self.total_heads // self.tp_size
        self.projection_size = self.total_heads * self.head_dim
        self.local_projection_size = self.num_heads * self.head_dim
        self.conv_size = int(kda_config["short_conv_kernel_size"])
        self.gate_lower_bound = kda_config.get("gate_lower_bound")
        self.use_full_rank_gate = kda_config.get("use_full_rank_gate", False)
        quant_config = vllm_config.quant_config

        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            self.projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            self.projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            self.projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.q_conv1d = CausalDepthwiseConv1d(self.projection_size,
                                              self.conv_size)
        self.k_conv1d = CausalDepthwiseConv1d(self.projection_size,
                                              self.conv_size)
        self.v_conv1d = CausalDepthwiseConv1d(self.projection_size,
                                              self.conv_size)
        self.f_a_proj = ReplicatedLinear(
            config.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_a_proj",
        )
        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.b_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.b_proj",
        )
        if self.use_full_rank_gate:
            self.g_proj = ColumnParallelLinear(
                config.hidden_size,
                self.projection_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
        else:
            self.g_a_proj = ReplicatedLinear(
                config.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_a_proj",
            )
            self.g_b_proj = ColumnParallelLinear(
                self.head_dim,
                self.projection_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_b_proj",
            )

        self.A_log = nn.Parameter(
            torch.empty(self.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(
            torch.empty(self.local_projection_size, dtype=torch.float32))
        set_weight_attrs(self.A_log, {"weight_loader": _load_a_log})
        set_weight_attrs(self.dt_bias,
                         {"weight_loader": sharded_weight_loader(0)})
        self.o_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.o_proj = RowParallelLinear(
            self.projection_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.kv_cache = None
        self.sconv_op = build_kimi_sconv_op(
            prefix,
            kernel_size=self.conv_size,
            state_dim_first=is_conv_state_dim_first(),
        )
        self.kda_op = build_kimi_kda_op(
            prefix,
            lower_bound=self.gate_lower_bound,
            eps=config.rms_norm_eps,
        )

        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.LINEAR

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.vllm_config.model_config.dtype,
            self.vllm_config.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.total_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=0,
        )

    def _metadata(self) -> AttentionMetadata | None:
        if not is_forward_context_available():
            return None
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return None
        layer_metadata = metadata[self.prefix]
        if not isinstance(layer_metadata, AttentionMetadata):
            raise TypeError("KDA received incompatible attention metadata")
        return layer_metadata

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del positions
        query, _ = self.q_proj(hidden_states)
        key, _ = self.k_proj(hidden_states)
        value, _ = self.v_proj(hidden_states)
        mixed_qkv = torch.cat((query, key, value), dim=-1)
        f_a, _ = self.f_a_proj(hidden_states)
        raw_gate, _ = self.f_b_proj(f_a)
        beta, _ = self.b_proj(hidden_states)
        if self.use_full_rank_gate:
            output_gate, _ = self.g_proj(hidden_states)
        else:
            g_a, _ = self.g_a_proj(hidden_states)
            output_gate, _ = self.g_b_proj(g_a)

        metadata = self._metadata()
        if (metadata is None or self.kv_cache is None
                or self.kv_cache[0].numel() == 0):
            output = torch.zeros_like(query)
        else:
            sconv_cache, recurrent_cache = self.kv_cache
            mixed_qkv = self.sconv_op(
                mixed_qkv,
                sconv_cache,
                self.q_conv1d.weight,
                self.k_conv1d.weight,
                self.v_conv1d.weight,
                metadata.query_start_loc,
                metadata.mamba_state_indices,
                metadata.seq_lens,
            )
            output = self.kda_op(
                mixed_qkv,
                raw_gate,
                beta,
                output_gate,
                recurrent_cache,
                self.A_log,
                self.dt_bias,
                self.o_norm.weight,
                metadata.query_start_loc,
                metadata.mamba_state_indices,
                metadata.seq_lens,
            ).flatten(1)

        output, _ = self.o_proj(output)
        return output

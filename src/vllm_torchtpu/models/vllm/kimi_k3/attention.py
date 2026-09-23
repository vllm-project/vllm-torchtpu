# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi MLA and KDA layers for TPU."""

from __future__ import annotations

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              get_tp_group)
from vllm.forward_context import (get_forward_context,
                                  is_forward_context_available)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator, is_conv_state_dim_first)
from vllm.model_executor.layers.mla import (MLAModules,
                                            MultiHeadLatentAttentionWrapper)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, sharded_weight_loader)
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.custom_ops.kda_attention_op import (
    build_kimi_dispatched_kda_op, build_kimi_pooled_kda_op)
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.sequence_layout import \
    is_pcp_streaming_attention_metadata
from vllm_torchtpu.logger import init_logger

from .collective_ops import FusedPrefillCollectives

logger = init_logger(__name__)


def kda_state_dtype(
        vllm_config: VllmConfig) -> tuple[torch.dtype, torch.dtype]:
    """Storage dtypes for one KDA layer's conv and recurrent state.

    The conv cache is fp32 regardless of ``mamba_cache_dtype``: v3's conv1d
    needs fp32 compact layout, so anything narrower makes ``fused_conv1d_gdn``
    widen the whole slot pool once per layer per step. Past about a thousand
    slots that costs more than the fused kernel saves.
    ``VllmGatedDeltaNetAttention.get_state_dtype`` declares fp32 for the same
    reason.
    """
    _, recurrent_dtype = MambaStateDtypeCalculator.kda_state_dtype(
        vllm_config.model_config.dtype,
        vllm_config.cache_config.mamba_cache_dtype,
    )
    return torch.float32, recurrent_dtype


def _load_a_log(parameter: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Load either the old ``[1, 1, H, 1]`` or current ``[H]`` layout."""

    if loaded_weight.ndim == 4:
        loaded_weight = loaded_weight.flatten()
    rank = get_tensor_model_parallel_rank()
    shard_size = parameter.shape[0]
    loaded_weight = loaded_weight.narrow(0, rank * shard_size, shard_size)
    default_weight_loader(parameter, loaded_weight)


class MixedParallelMergedLinear(MergedColumnParallelLinear):
    """One local matmul containing replicated and TP-column weight slices.

    Every rank stores complete weights for replicated outputs and only its
    column-parallel slice for sharded outputs.  The physical output sizes are
    therefore local sizes, so the parent is intentionally constructed with TP
    disabled; this loader performs the selective checkpoint sharding instead.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        replicate_outputs: list[bool],
        *,
        bias: bool,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        if len(output_sizes) != len(replicate_outputs):
            raise ValueError("Every fused output needs a sharding mode")
        tp_size = get_tensor_model_parallel_world_size()
        if any(not replicate and size % tp_size
               for size, replicate in zip(output_sizes, replicate_outputs)):
            raise ValueError("Column-parallel fused outputs must divide TP")
        self.mixed_tp_size = tp_size
        self.mixed_tp_rank = get_tensor_model_parallel_rank()
        self.global_output_sizes = output_sizes
        self.replicate_outputs = replicate_outputs
        local_output_sizes = [
            size if replicate else size // tp_size
            for size, replicate in zip(output_sizes, replicate_outputs)
        ]
        super().__init__(
            input_size,
            local_output_sizes,
            bias=bias,
            quant_config=quant_config,
            prefix=prefix,
            disable_tp=True,
        )

    def _slice_loaded_weight(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> torch.Tensor:
        if (isinstance(loaded_shard_id, int)
                and not self.replicate_outputs[loaded_shard_id]):
            output_dim = getattr(param, "output_dim", None)
            is_sharded_weight = getattr(param, "is_sharded_weight", False)
            is_sharded_weight |= getattr(param, "use_bitsandbytes_4bit", False)
            if output_dim is not None and not is_sharded_weight:
                loaded_size = loaded_weight.shape[output_dim]
                if loaded_size % self.mixed_tp_size:
                    raise ValueError(
                        "Fused column-parallel checkpoint dimension does not "
                        f"divide TP: {loaded_size=} {self.mixed_tp_size=}")
                shard_size = loaded_size // self.mixed_tp_size
                loaded_weight = loaded_weight.narrow(
                    output_dim,
                    self.mixed_tp_rank * shard_size,
                    shard_size,
                )
        return loaded_weight

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        loaded_weight = self._slice_loaded_weight(
            param,
            loaded_weight,
            loaded_shard_id,
        )
        super().weight_loader(param, loaded_weight, loaded_shard_id)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        loaded_weight = self._slice_loaded_weight(
            param,
            loaded_weight,
            loaded_shard_id,
        )
        super().weight_loader_v2(param, loaded_weight, loaded_shard_id)


class MultiHeadLatentAttention(nn.Module):
    """NoPE MLA dispatched through the TPU Pallas MLA custom op."""

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str,
        *,
        use_rope: bool = False,
        non_causal_multi_token_decode: bool = False,
    ) -> None:
        if not use_rope and not config.mla_use_nope:
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

        self.mla_gate_is_fused = False
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
            qkv_a_output_sizes = [
                self.q_lora_rank,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ]
            if config.mla_use_output_gate:
                qkv_a_output_sizes.append(config.num_attention_heads *
                                          self.v_head_dim)
                self.fused_qkv_a_proj = MixedParallelMergedLinear(
                    config.hidden_size,
                    qkv_a_output_sizes,
                    [True, True, False],
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.fused_qkv_a_proj",
                )
                self.mla_gate_is_fused = True
            else:
                self.fused_qkv_a_proj = MergedColumnParallelLinear(
                    config.hidden_size,
                    qkv_a_output_sizes,
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
        ) if (config.mla_use_output_gate and not self.mla_gate_is_fused) else
                       None)
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            reduce_results=False,
        )
        rotary_emb = None
        if use_rope:
            rope_parameters = dict(
                getattr(config, "rope_parameters", None) or {
                    "rope_type": "default",
                    "rope_theta": getattr(config, "rope_theta", 10000.0),
                })
            if rope_parameters.get("rope_type", "default") != "default":
                rope_parameters["rope_type"] = ("deepseek_yarn"
                                                if rope_parameters.get(
                                                    "apply_yarn_scaling", True)
                                                else "deepseek_llama_scaling")
            rotary_emb = get_rope(
                self.qk_rope_head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=rope_parameters,
                is_neox_style=False,
                dtype=torch.float32,
            )

        self.prefill_group = (FusedPrefillCollectives(prefix)
                              if envs.TPU_K3_SP_PREFILL else None)
        mla_modules = MLAModules(
            g_proj=self.g_proj,
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=rotary_emb,
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
            gate_is_fused=self.mla_gate_is_fused,
            non_causal_multi_token_decode=non_causal_multi_token_decode,
        )

    def prepare_sp_mla_gate(self):
        """Refresh the TP2 MLA gate buffer after checkpoint loading or reload."""
        group = self.prefill_group
        projection = self.fused_qkv_a_proj
        weight = getattr(projection, 'weight', None)
        wrapper = self.mla_attn
        wrapper.kimi_preprocess = None
        wrapper.kimi_gate_weight = None
        if not (isinstance(group, FusedPrefillCollectives)
                and self.mla_gate_is_fused and weight is not None and
                weight.dtype == torch.bfloat16 and weight.shape == (7168, 2496)
                and getattr(projection, WEIGHT_FLIPPED_ATTR, False)):
            return
        eps = self.q_a_layernorm.variance_epsilon
        if eps != self.kv_a_layernorm.variance_epsilon:
            raise ValueError(
                'K3 MLA preprocessing requires matching norm epsilons')
        pack, wrapper.kimi_preprocess = group.mla_ops(eps)
        with torch.no_grad():
            wrapper.kimi_gate_weight = pack(weight)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        sequence_parallel: bool = False,
    ) -> torch.Tensor:
        group = self.prefill_group if sequence_parallel else get_tp_group()
        if (sequence_parallel
                and getattr(self.mla_attn, 'kimi_preprocess', None) is None):
            hidden_states = group.all_gather(hidden_states, dim=0)
        if sequence_parallel and isinstance(group, FusedPrefillCollectives):
            return self.mla_attn(positions, hidden_states, prefill_group=group)
        output = self.mla_attn(positions, hidden_states)
        return (group.reduce_scatter(output, dim=0)
                if sequence_parallel else group.all_reduce(output))


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
        self.num_spec_tokens = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config is not None else 0)
        self.gate_lower_bound = kda_config.get("gate_lower_bound")
        self.use_full_rank_gate = kda_config.get("use_full_rank_gate", False)
        quant_config = vllm_config.quant_config

        self.fused_qkvb_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [self.projection_size] * 3 + [self.total_heads],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkvb_proj",
        )
        if self.use_full_rank_gate:
            self.fused_fa_ga_proj = None
            self.f_a_proj = ReplicatedLinear(
                config.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.f_a_proj",
            )
            self.g_proj = ColumnParallelLinear(
                config.hidden_size,
                self.projection_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
        else:
            self.fused_fa_ga_proj = MergedColumnParallelLinear(
                config.hidden_size,
                [self.head_dim, self.head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_fa_ga_proj",
                disable_tp=True,
            )
            self.f_a_proj = None
            self.g_proj = None
        self.q_conv1d = CausalDepthwiseConv1d(self.projection_size,
                                              self.conv_size)
        self.k_conv1d = CausalDepthwiseConv1d(self.projection_size,
                                              self.conv_size)
        self.v_conv1d = CausalDepthwiseConv1d(self.projection_size,
                                              self.conv_size)
        # Both convolution consumers take one fused weight in the kernels'
        # [kernel_size, 3, heads, head_dim] layout;
        # `process_weights_after_loading` folds the checkpoint's three tensors
        # into it at load time. Non-persistent:
        # it is derived from q/k/v_conv1d.weight, not a checkpoint entry of its
        # own, and must not appear in a state dict.
        self.register_buffer(
            "conv_weight_fused",
            torch.empty(self.conv_size, 3, self.num_heads, self.head_dim),
            persistent=False,
        )
        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        if self.use_full_rank_gate:
            self.g_b_proj = None
        else:
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
            reduce_results=False,
        )
        self.prefill_group = (FusedPrefillCollectives(prefix)
                              if envs.TPU_K3_SP_PREFILL else None)
        self.register_buffer("packed_prefill_weight", None, persistent=False)
        self.kv_cache = None
        # The dispatched op owns the short convolution as well: the fused
        # kernel does the convolution itself, so it cannot be fed by a separate
        # convolution op.
        self.dispatched_kda_op = build_kimi_dispatched_kda_op(
            prefix,
            lower_bound=self.gate_lower_bound,
            eps=config.rms_norm_eps,
            state_dim_first=is_conv_state_dim_first(),
            num_spec_tokens=self.num_spec_tokens,
        )
        # Unified-pool variant: reads and writes both state regions through
        # the single attention-shaped buffer; used when the runner binds the
        # pool instead of the per-layer conv/recurrent caches. The manager
        # block size lets the op derive the manager->pool-block split for
        # its state-index remap. It is NOT final yet: the platform's
        # block-size derivation for the pool runs after model construction
        # (hybrid layers are built before the adjusted value exists), so the
        # op reads cache_config.block_size per call, not here.
        self.pooled_kda_op = build_kimi_pooled_kda_op(
            prefix,
            lower_bound=self.gate_lower_bound,
            eps=config.rms_norm_eps,
            vllm_config=vllm_config,
        )

        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.LINEAR

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return kda_state_dtype(self.vllm_config)

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        # Keep the KDA short-convolution cache in the Pallas DMA layout.
        return (
            (self.conv_size - 1, 3, self.num_heads, self.head_dim),
            (self.num_heads, self.head_dim, self.head_dim),
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

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Fold the three depthwise conv weights into the kernels' layout.

        The checkpoint keeps q/k/v as separate ``[dim, 1, kernel_size]``
        tensors, and the ops used to concatenate and transpose them on every
        step. Both convolution consumers want ``[kernel_size, 3, heads,
        head_dim]``, so build that once, here, at load time.
        """
        fused = torch.stack(
            tuple(w[:, 0, :].t().reshape(self.conv_size, self.num_heads,
                                         self.head_dim)
                  for w in (self.q_conv1d.weight, self.k_conv1d.weight,
                            self.v_conv1d.weight)),
            dim=1,
        )
        # `fused` is the lazy result of a stack of views, and torch_tpu
        # re-materializes that lineage as a per-step `tt_jit_as_strided`
        # program at the dispatched op's boundary (same failure mode and fix as
        # the w_scale case in quantization/fp8.py). A fresh buffer filled by
        # copy_ breaks the lazy chain so PJRT ships a plain row-major buffer.
        out = torch.empty_like(fused)
        out.copy_(fused)
        self.conv_weight_fused = out
        self._pack_prefill_projection()

    def _pack_prefill_projection(self):
        """Cache BF16 KDA input weights after the normal checkpoint loaders."""
        self.packed_prefill_weight = None
        if not (isinstance(getattr(self, 'prefill_group', None),
                           FusedPrefillCollectives) and self.use_full_rank_gate
                and self.num_heads == 3 and self.head_dim == 128):
            return
        projections = (self.fused_qkvb_proj, self.g_proj, self.f_a_proj)
        expected = ((7168, 1155), (7168, 384), (7168, 128))
        for projection, shape in zip(projections, expected):
            weight = getattr(projection, 'weight', None)
            if (weight is None or weight.dtype != torch.bfloat16
                    or weight.shape != shape
                    or not getattr(projection, WEIGHT_FLIPPED_ATTR, False)):
                return
        # q/k/v occupy 1152 columns; pad the 3 beta columns to an MXU tile.
        qkvb, gate, fa = (p.weight for p in projections)
        with torch.no_grad():
            packed = torch.cat((qkvb, qkvb.new_zeros((7168, 125)), gate, fa),
                               dim=-1)
            # Materialize the layout once, avoiding a lazy cat at every call.
            self.packed_prefill_weight = torch.empty_like(packed)
            self.packed_prefill_weight.copy_(packed)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        sequence_parallel: bool = False,
    ) -> torch.Tensor:
        del positions
        group = self.prefill_group if sequence_parallel else get_tp_group()
        if self.packed_prefill_weight is not None:
            # Every token bucket must capture the same parameter list: the
            # serving compiler hands executables between traces by input ABI.
            if sequence_parallel and isinstance(group,
                                                FusedPrefillCollectives):
                packed = group.gather_project(hidden_states,
                                              self.packed_prefill_weight)
            else:
                if sequence_parallel:
                    hidden_states = group.all_gather(hidden_states, dim=0)
                packed = hidden_states @ self.packed_prefill_weight
            mixed_qkv, beta_pad, output_gate, f_a = packed.split(
                [1152, 128, 384, 128], dim=-1)
            beta = beta_pad[:, :self.num_heads]
            query = mixed_qkv[:, :self.local_projection_size]
        else:
            if sequence_parallel:
                hidden_states = group.all_gather(hidden_states, dim=0)
            projected, _ = self.fused_qkvb_proj(hidden_states)
            query, key, value, beta = projected.split(
                [self.local_projection_size] * 3 + [self.num_heads], dim=-1)
            mixed_qkv = torch.cat((query, key, value), dim=-1)
            if self.use_full_rank_gate:
                assert self.f_a_proj is not None
                f_a, _ = self.f_a_proj(hidden_states)
                assert self.g_proj is not None
                output_gate, _ = self.g_proj(hidden_states)
            else:
                assert self.fused_fa_ga_proj is not None
                gate_inputs, _ = self.fused_fa_ga_proj(hidden_states)
                f_a, gate_input = gate_inputs.split(
                    [self.head_dim, self.head_dim], dim=-1)
        raw_gate, _ = self.f_b_proj(f_a)
        if not self.use_full_rank_gate:
            assert self.g_b_proj is not None
            output_gate, _ = self.g_b_proj(gate_input)

        metadata = self._metadata()
        if (metadata is None or self.kv_cache is None
                or self.kv_cache[0].numel() == 0):
            output = torch.zeros_like(query)
        elif len(self.kv_cache) == 1:
            # Unified pool: the single attention-shaped buffer carries both
            # KDA state regions.
            output = self._core_attention_pooled(
                mixed_qkv,
                raw_gate,
                beta,
                output_gate,
                self.kv_cache[0],
                metadata,
            ).flatten(1)
        else:
            sconv_cache, recurrent_cache = self.kv_cache
            window_distribution = metadata.mamba_request_distribution
            if window_distribution is None:
                window_distribution = metadata.request_distribution
            slot_read_offsets = metadata.mamba_slot_read_offsets
            if slot_read_offsets is None:
                # Non-speculative KDA keeps the legacy ABI without allocating
                # a pool-sized offset tensor.  The op's static
                # num_spec_tokens=0 path never indexes this dummy.
                slot_read_offsets = metadata.mamba_state_indices
                if slot_read_offsets is None:
                    slot_read_offsets = metadata.seq_lens
            output = self.dispatched_kda_op(
                mixed_qkv,
                raw_gate,
                beta,
                output_gate,
                sconv_cache,
                recurrent_cache,
                self.conv_weight_fused,
                self.A_log,
                self.dt_bias,
                self.o_norm.weight,
                metadata.query_start_loc,
                metadata.mamba_state_indices,
                metadata.seq_lens,
                metadata.request_distribution,
                window_distribution,
                slot_read_offsets,
            ).flatten(1)

        if sequence_parallel and isinstance(group, FusedPrefillCollectives):
            return group.project_reduce_scatter(output, self.o_proj)
        output, _ = self.o_proj(output)
        return (group.reduce_scatter(output, dim=0)
                if sequence_parallel else group.all_reduce(output))

    def _core_attention_pooled(
        self,
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        output_gate: torch.Tensor,
        pool: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """KDA over the unified pool: conv and recurrent state live in the
        pool's per-block byte regions and are gathered/scattered by the op."""
        if is_pcp_streaming_attention_metadata(metadata):
            raise NotImplementedError(
                "KDA does not support PCP streaming prefill with the unified "
                "KV pool")
        if metadata.mamba_slot_read_offsets is not None:
            # Speculative verify needs per-window state checkpoints; the
            # pooled gather/scatter path keeps a single state per block.
            raise NotImplementedError(
                "Speculative decoding is not supported with pooled KDA")
        state_indices = metadata.mamba_state_indices
        if state_indices is None:
            raise RuntimeError(
                "Pooled KDA requires mamba_state_indices in the attention "
                "metadata")
        return self.pooled_kda_op(
            mixed_qkv,
            raw_gate,
            beta,
            output_gate,
            pool,
            self.conv_weight_fused,
            self.A_log,
            self.dt_bias,
            self.o_norm.weight,
            metadata.query_start_loc,
            state_indices,
            metadata.seq_lens,
            metadata.request_distribution,
        )

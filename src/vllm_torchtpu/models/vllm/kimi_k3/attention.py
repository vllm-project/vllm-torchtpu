# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Kimi MLA and KDA layers for TPU."""

from __future__ import annotations

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.kda_attention import KdaAttention
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.logger import init_logger

from .collective_ops import FusedPrefillCollectives

logger = init_logger(__name__)


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
        if any(
            not replicate and size % tp_size
            for size, replicate in zip(output_sizes, replicate_outputs)
        ):
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
        if (
            isinstance(loaded_shard_id, int)
            and not self.replicate_outputs[loaded_shard_id]
        ):
            output_dim = getattr(param, "output_dim", None)
            is_sharded_weight = getattr(param, "is_sharded_weight", False)
            is_sharded_weight |= getattr(param, "use_bitsandbytes_4bit", False)
            if output_dim is not None and not is_sharded_weight:
                loaded_size = loaded_weight.shape[output_dim]
                if loaded_size % self.mixed_tp_size:
                    raise ValueError(
                        "Fused column-parallel checkpoint dimension does not "
                        f"divide TP: {loaded_size=} {self.mixed_tp_size=}"
                    )
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
            raise ValueError("num_attention_heads must be divisible by TP size")
        num_heads = config.num_attention_heads // tp_size
        qk_head_dim = int(config.qk_nope_head_dim) + int(config.qk_rope_head_dim)
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
                qkv_a_output_sizes.append(config.num_attention_heads * self.v_head_dim)
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
            config.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.g_proj = (
            ColumnParallelLinear(
                config.hidden_size,
                config.num_attention_heads * self.v_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
            if (config.mla_use_output_gate and not self.mla_gate_is_fused)
            else None
        )
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
                getattr(config, "rope_parameters", None)
                or {
                    "rope_type": "default",
                    "rope_theta": getattr(config, "rope_theta", 10000.0),
                }
            )
            if rope_parameters.get("rope_type", "default") != "default":
                rope_parameters["rope_type"] = (
                    "deepseek_yarn"
                    if rope_parameters.get("apply_yarn_scaling", True)
                    else "deepseek_llama_scaling"
                )
            rotary_emb = get_rope(
                self.qk_rope_head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=rope_parameters,
                is_neox_style=False,
                dtype=torch.float32,
            )

        self.prefill_group = (
            FusedPrefillCollectives(prefix) if envs.TPU_K3_SP_PREFILL else None
        )
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
        weight = getattr(projection, "weight", None)
        wrapper = self.mla_attn
        wrapper.kimi_preprocess = None
        wrapper.kimi_gate_weight = None
        if not (
            isinstance(group, FusedPrefillCollectives)
            and self.mla_gate_is_fused
            and weight is not None
            and weight.dtype == torch.bfloat16
            and weight.shape == (7168, 2496)
            and getattr(projection, WEIGHT_FLIPPED_ATTR, False)
        ):
            return
        assert self.q_a_layernorm is not None
        eps = self.q_a_layernorm.variance_epsilon
        if eps != self.kv_a_layernorm.variance_epsilon:
            raise ValueError("K3 MLA preprocessing requires matching norm epsilons")
        # torch's Module.__setattr__ stub rejects a function as the value.
        # pyrefly: ignore[bad-argument-type]
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
        assert group is not None
        if (
            sequence_parallel
            and getattr(self.mla_attn, "kimi_preprocess", None) is None
        ):
            hidden_states = group.all_gather(hidden_states, dim=0)
        if sequence_parallel and isinstance(group, FusedPrefillCollectives):
            return self.mla_attn(positions, hidden_states, prefill_group=group)
        output = self.mla_attn(positions, hidden_states)
        return (
            group.reduce_scatter(output, dim=0)
            if sequence_parallel
            else group.all_reduce(output)
        )


class KimiDeltaAttention(KdaAttention):
    """Kimi's KDA, with Kimi K3's fused sequence-parallel prefill."""

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        kda_config = config.linear_attn_config
        if kda_config is None:
            raise ValueError("KDA requires linear_attn_config")
        head_dim = int(kda_config["head_dim"])
        super().__init__(
            vllm_config,
            prefix,
            hidden_size=config.hidden_size,
            num_heads=int(kda_config["num_heads"]),
            head_k_dim=head_dim,
            head_v_dim=head_dim,
            conv_kernel_size=int(kda_config["short_conv_kernel_size"]),
            gate_rank=head_dim,
            norm_eps=config.rms_norm_eps,
            use_full_rank_gate=kda_config.get("use_full_rank_gate", False),
            gate_lower_bound=kda_config.get("gate_lower_bound"),
        )
        self.prefill_group = (
            FusedPrefillCollectives(prefix) if envs.TPU_K3_SP_PREFILL else None
        )
        self.register_buffer("packed_prefill_weight", None, persistent=False)

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        super().process_weights_after_loading(act_dtype)
        self._pack_prefill_projection()

    def _pack_prefill_projection(self):
        """Cache BF16 KDA input weights after the normal checkpoint loaders."""
        self.packed_prefill_weight = None
        if not (
            isinstance(getattr(self, "prefill_group", None), FusedPrefillCollectives)
            and self.use_full_rank_gate
            and self.num_heads == 3
            and self.head_k_dim == 128
        ):
            return
        projections = (self.fused_qkvb_proj, self.g_proj, self.f_a_proj)
        expected = ((7168, 1155), (7168, 384), (7168, 128))
        weights = []
        for projection, shape in zip(projections, expected):
            weight = getattr(projection, "weight", None)
            if (
                weight is None
                or weight.dtype != torch.bfloat16
                or weight.shape != shape
                or not getattr(projection, WEIGHT_FLIPPED_ATTR, False)
            ):
                return
            weights.append(weight)
        # q/k/v occupy 1152 columns; pad the 3 beta columns to an MXU tile.
        qkvb, gate, fa = weights
        with torch.no_grad():
            packed = torch.cat((qkvb, qkvb.new_zeros((7168, 125)), gate, fa), dim=-1)
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
        assert group is not None
        if self.packed_prefill_weight is not None:
            # Every token bucket must capture the same parameter list: the
            # serving compiler hands executables between traces by input ABI.
            if sequence_parallel and isinstance(group, FusedPrefillCollectives):
                packed = group.gather_project(hidden_states, self.packed_prefill_weight)
            else:
                if sequence_parallel:
                    hidden_states = group.all_gather(hidden_states, dim=0)
                packed = hidden_states @ self.packed_prefill_weight
            mixed_qkv, beta_pad, output_gate, f_a = packed.split(
                [1152, 128, 384, 128], dim=-1
            )
            beta = beta_pad[:, : self.num_heads]
            raw_gate, _ = self.f_b_proj(f_a)
        else:
            if sequence_parallel:
                hidden_states = group.all_gather(hidden_states, dim=0)
            mixed_qkv, raw_gate, beta, output_gate = self._project_inputs(hidden_states)
        output = self._core_attention(mixed_qkv, raw_gate, beta, output_gate)

        if sequence_parallel and isinstance(group, FusedPrefillCollectives):
            return group.project_reduce_scatter(output, self.o_proj)
        output, _ = self.o_proj(output)
        return (
            group.reduce_scatter(output, dim=0)
            if sequence_parallel
            else group.all_reduce(output)
        )

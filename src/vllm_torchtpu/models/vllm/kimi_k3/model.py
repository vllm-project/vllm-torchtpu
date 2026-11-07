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
"""Kimi K3 and Kimi-Linear models for TPU."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
)
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models import vision as vision_utils
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MultiModalEmbeddings,
    SupportsEncoderCudaGraph,
    SupportsMultiModal,
    SupportsQuant,
)
from vllm.model_executor.models.kimi_k25 import KimiK25MediaPixelInputs
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    mm_projector_forward,
    tpool_patch_merger_packed,
    vision_tower_forward,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _flatten_embeddings,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.model_executor.models.vision import is_vit_use_data_parallel
from vllm.models.kimi_k3.common.mm_preprocess import (
    KimiK3DummyInputsBuilder,
    KimiK3MultiModalProcessor,
    KimiK3ProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import fresh_buffer, synchronize_tensors

from .attention import KimiDeltaAttention, MultiHeadLatentAttention, kda_state_dtype
from .collective_ops import TokenShardCollectives
from .kimi_vit import KimiK3MoonViT3dPretrainedModel
from .layers import AttentionResidual, KimiMLP
from .moe import KimiMoE

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from vllm.config import VllmConfig
    from vllm.model_executor.layers.quantization import QuantizationConfig
    from vllm.multimodal.inputs import NestedTensors
    from vllm.sequence import IntermediateTensors
    from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

logger = init_logger(__name__)

ASSERT_SP_PREFILL_TP_SIZE = 32
ASSERT_GRID_THW_NDIM = 2
ASSERT_GRID_THW_COLS = 3
ASSERT_CONV_HEAD_GROUPS = 3
_SHARD_ID_ATTR = "shard_id"
_WEIGHT_LOADER_ATTR = "weight_loader"


class KimiDecoderLayer(nn.Module):
    """Single Kimi-K3 decoder layer combining KDA or MLA with dense MLP or MoE."""

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        """Initialize attention, feed-forward, norm, and block-residual modules."""
        super().__init__()
        self.layer_idx = int(prefix.rsplit(".", 1)[1])
        if config.is_kda_layer(self.layer_idx):
            self.self_attn = KimiDeltaAttention(
                config,
                vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = MultiHeadLatentAttention(
                config,
                vllm_config,
                prefix=f"{prefix}.self_attn",
            )

        is_moe = (
            config.num_experts is not None
            and self.layer_idx >= config.first_k_dense_replace
            and self.layer_idx % config.moe_layer_freq == 0
        )
        self.is_moe = is_moe
        if is_moe:
            self.block_sparse_moe = KimiMoE(
                config,
                vllm_config.quant_config,
                prefix=f"{prefix}.block_sparse_moe",
            )
        else:
            self.mlp = KimiMLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
                situ_beta=config.activation_situ_beta,
                situ_linear_beta=config.activation_situ_linear_beta,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.attn_res_block_size = config.attn_res_block_size
        if self.attn_res_block_size is not None:
            self.self_attention_res = AttentionResidual(
                config.hidden_size,
                config.rms_norm_eps,
                prefix=f"{prefix}.self_attention_res_proj",
            )
            self.mlp_res = AttentionResidual(
                config.hidden_size,
                config.rms_norm_eps,
                prefix=f"{prefix}.mlp_res_proj",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        block_residuals: torch.Tensor | None,
        *,
        sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply attention, block-residual routing, and MLP/MoE to hidden states."""
        if self.attn_res_block_size is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(
                positions, hidden_states, sequence_parallel=sequence_parallel
            )
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            if self.is_moe:
                hidden_states = self.block_sparse_moe(
                    hidden_states, sequence_parallel=sequence_parallel
                )
            else:
                hidden_states = self.mlp(
                    hidden_states, sequence_parallel=sequence_parallel
                )
            return residual + hidden_states, None

        if block_residuals is None:
            msg = "block_residuals is required when attn_res_block_size is set"
            raise ValueError(msg)
        prefix_sum = hidden_states
        if block_residuals.shape[-2] > 0:
            hidden_states = self.self_attention_res(prefix_sum, block_residuals)
        if self.layer_idx % self.attn_res_block_size == 0:
            block_residuals = torch.cat(
                (block_residuals, prefix_sum.unsqueeze(-2)), dim=-2
            )
            prefix_sum = None

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions, hidden_states, sequence_parallel=sequence_parallel
        )
        prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states
        hidden_states = self.mlp_res(prefix_sum, block_residuals)
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            hidden_states = self.block_sparse_moe(
                hidden_states, sequence_parallel=sequence_parallel
            )
        else:
            hidden_states = self.mlp(hidden_states, sequence_parallel=sequence_parallel)
        return prefix_sum + hidden_states, block_residuals


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
    }
)
class KimiModel(nn.Module):
    """Kimi-Linear text backbone with optional sequence-parallel prefill."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Initialize token embeddings, decoder stack, and output norm."""
        super().__init__()
        config = cast("KimiLinearConfig", vllm_config.model_config.hf_text_config)
        logger.info(
            "Kimi optional features: mla_use_output_gate=%s, hidden_act=%s, "
            "activation_situ_beta=%s, activation_situ_linear_beta=%s, "
            "attn_res_block_size=%s",
            config.mla_use_output_gate,
            config.hidden_act,
            config.activation_situ_beta,
            config.activation_situ_linear_beta,
            config.attn_res_block_size,
        )
        self.attn_res_block_size = config.attn_res_block_size
        self.sp_prefill = envs.TPU_K3_SP_PREFILL
        if self.sp_prefill:
            parallel = vllm_config.parallel_config
            if (
                parallel.tensor_parallel_size != ASSERT_SP_PREFILL_TP_SIZE
                or not parallel.enable_expert_parallel
                or parallel.pipeline_parallel_size != 1
                or parallel.data_parallel_size != 1
            ):
                msg = "K3 SP prefill requires TP32/EP32 and PP1/DP1"
                raise ValueError(msg)
            self.sp_group = TokenShardCollectives(get_tp_group())

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList(
            [
                KimiDecoderLayer(
                    config,
                    vllm_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.aux_hidden_state_layers: tuple[int, ...] = ()
        if self.attn_res_block_size is not None:
            self.output_attn_res = AttentionResidual(
                config.hidden_size,
                config.rms_norm_eps,
                prefix=f"{prefix}.output_attn_res_proj",
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings for `input_ids`."""
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Propagate token embeddings or hidden states through decoder layers."""
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            if input_ids is None:
                msg = "Either input_ids or inputs_embeds must be provided"
                raise ValueError(msg)
            hidden_states = self.embed_input_ids(input_ids)
        num_tokens = hidden_states.shape[0]
        sequence_parallel = (
            self.sp_prefill
            and num_tokens > 0
            and num_tokens % self.sp_group.world_size == 0
        )
        if sequence_parallel:
            hidden_states = hidden_states.reshape(
                self.sp_group.world_size, -1, hidden_states.shape[-1]
            )[self.sp_group.rank_in_group].contiguous()
        # A view avoids passing a symbolic local token count to TorchTPU's
        # concrete-size allocation kernel during the dynamic trace.
        block_residuals = (
            hidden_states.unsqueeze(1)[:, :0, :]
            if self.attn_res_block_size is not None
            else None
        )

        aux_hidden_states: list[torch.Tensor] = []
        if 0 in self.aux_hidden_state_layers:
            aux_hidden_states.append(hidden_states)

        for layer_idx, layer in enumerate(self.layers):
            hidden_states, block_residuals = layer(
                positions,
                hidden_states,
                block_residuals,
                sequence_parallel=sequence_parallel,
            )
            if layer_idx + 1 in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states)

        if block_residuals is not None:
            hidden_states = self.output_attn_res(hidden_states, block_residuals)
        hidden_states = self.norm(hidden_states)
        if sequence_parallel:
            hidden_states = self.sp_group.all_gather(hidden_states, dim=0)
            aux_hidden_states = [
                self.sp_group.all_gather(state, dim=0) for state in aux_hidden_states
            ]
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class KimiLinearForCausalLM(nn.Module, HasInnerState, IsHybrid):
    """Kimi text model for TPU."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.layers.": "model.layers.",
            "language_model.": "",
        },
        orig_to_new_substr={
            ".self_attention_res_norm.": ".self_attention_res.norm.",
            ".self_attention_res_proj.": ".self_attention_res.proj.",
            ".mlp_res_norm.": ".mlp_res.norm.",
            ".mlp_res_proj.": ".mlp_res.proj.",
            ".output_attn_res_norm.": ".output_attn_res.norm.",
            ".output_attn_res_proj.": ".output_attn_res.proj.",
        },
        orig_to_new_stacked={
            ".gate_proj.": (".gate_up_proj.", 0),
            ".up_proj.": (".gate_up_proj.", 1),
        },
    )
    fused_mla_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_a_proj.": (".fused_qkv_a_proj.", 0),
            ".kv_a_proj_with_mqa.": (".fused_qkv_a_proj.", 1),
        },
    )
    fused_attention_params_mapping = (
        (".self_attn.q_proj.", ".self_attn.fused_qkvb_proj.", 0),
        (".self_attn.k_proj.", ".self_attn.fused_qkvb_proj.", 1),
        (".self_attn.v_proj.", ".self_attn.fused_qkvb_proj.", 2),
        (".self_attn.b_proj.", ".self_attn.fused_qkvb_proj.", 3),
        (".self_attn.f_a_proj.", ".self_attn.fused_fa_ga_proj.", 0),
        (".self_attn.g_a_proj.", ".self_attn.fused_fa_ga_proj.", 1),
        (".self_attn.g_proj.", ".self_attn.fused_qkv_a_proj.", 2),
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Initialize the Kimi text model, LM head, and logits processor."""
        super().__init__()
        self.config = cast("KimiLinearConfig", vllm_config.model_config.hf_text_config)
        self.model = KimiModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings via the inner `KimiModel`."""
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """Configure auxiliary hidden-state capture layer indices."""
        invalid = [idx for idx in layers if idx < 0 or idx > len(self.model.layers)]
        if invalid:
            msg = f"Invalid Kimi aux hidden-state layers: {invalid}"
            raise ValueError(msg)
        self.model.aux_hidden_state_layers = tuple(layers)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors: object | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Compute language-model hidden states across local decoder layers."""
        if intermediate_tensors is not None:
            msg = "The TPU Kimi model does not support PP"
            raise ValueError(msg)
        return self.model(
            input_ids,
            positions,
            inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Project `hidden_states` to vocabulary logits."""
        return self.logits_processor(self.lm_head, hidden_states)

    def process_weights_after_loading(self) -> None:
        """Fold every AttentionResidual's norm and projection weights.

        The generic loader hook fires per module only for attention layers,
        so the residuals are folded from here instead.
        """
        for module in self.modules():
            if isinstance(module, AttentionResidual):
                module.process_weights_after_loading()
            elif isinstance(module, KimiMoE):
                module.initialize_dummy_router()
            elif isinstance(module, MultiHeadLatentAttention):
                module.prepare_sp_mla_gate()

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        """Return `(conv_state_dtype, ssm_state_dtype)` for KDA state caches."""
        # Same helper the layer uses. vLLM sizes the cache from here and
        # allocates from get_state_dtype, so the two cannot diverge.
        return kda_state_dtype(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """Return `(conv_state_shape, ssm_state_shape)` for KDA state caches."""
        config = cast("KimiLinearConfig", vllm_config.model_config.hf_text_config)
        kda_config = config.linear_attn_config
        if kda_config is None:
            msg = "KimiLinearConfig requires linear_attn_config for KDA state shapes"
            raise ValueError(msg)
        local_heads = (
            kda_config["num_heads"] // vllm_config.parallel_config.tensor_parallel_size
        )
        conv_shape = cast(
            "tuple[int, int]",
            (
                kda_config["short_conv_kernel_size"] - 1,
                1,
                ASSERT_CONV_HEAD_GROUPS * local_heads * kda_config["head_dim"],
            ),
        )
        return (
            conv_shape,
            (local_heads, kda_config["head_dim"], kda_config["head_dim"]),
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        """Return KDA state copy functions for prefix caching."""
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def _remap_fused_attention_weight(
        self,
        weight_key: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
    ) -> str:
        """Remap unfused attention projection names onto fused parameter shards."""
        for (
            weight_name,
            param_name,
            shard_id,
        ) in self.fused_attention_params_mapping:
            if weight_name not in weight_key:
                continue
            candidate = weight_key.replace(weight_name, param_name)
            if candidate not in params_dict:
                continue
            setattr(loaded_weight, _SHARD_ID_ATTR, shard_id)
            return candidate
        return weight_key

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream and load dense, attention, and routed MoE weights."""
        expert_params_mapping = (
            fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.num_experts,
            )
            if self.config.num_experts is not None
            else []
        )
        params_dict = dict(self.named_parameters())
        experts_use_weight = not any(
            name.endswith("w13_weight_packed") for name in params_dict
        )
        loaded_experts: set[str] = set()
        mapper = self.hf_to_vllm_mapper
        if self.config.q_lora_rank is not None:
            mapper |= self.fused_mla_mapper

        # Feed non-expert weights to AutoWeightsLoader as a GENERATOR, not a
        # list: with a streamed checkpoint the loop below drains the whole
        # stream, and a list would pin every dense weight's host tensor until
        # the loop ends (~115 GB per rank for Kimi-K3 -- enough to OOM a
        # 944 GB host at 8 ranks before loading finishes). Lazily interleaving
        # expert dispatch with AutoWeightsLoader consumption frees each host
        # tensor as soon as its parameter is loaded.
        def _ordinary_weights() -> Iterator[tuple[str, torch.Tensor]]:
            """Yield non-expert weights while dispatching routed experts eagerly."""
            for raw_name, loaded_weight in mapper.apply(weights):
                if raw_name.startswith(("vision_tower.", "mm_projector.")):
                    continue
                weight_key = (
                    raw_name.replace(".weight_packed", ".weight")
                    if experts_use_weight and raw_name.endswith(".weight_packed")
                    else raw_name
                )
                weight_key = self._remap_fused_attention_weight(
                    weight_key, loaded_weight, params_dict
                )

                for (
                    param_name,
                    weight_name,
                    expert_id,
                    expert_shard_id,
                ) in expert_params_mapping:
                    if weight_name not in weight_key:
                        continue
                    mapped_key = weight_key.replace(weight_name, param_name)
                    parameter = params_dict[mapped_key]
                    weight_loader = getattr(parameter, _WEIGHT_LOADER_ATTR)
                    weight_loader(
                        parameter,
                        loaded_weight,
                        mapped_key,
                        expert_id=expert_id,
                        shard_id=expert_shard_id,
                    )
                    loaded_experts.add(mapped_key)
                    break
                else:
                    yield weight_key, loaded_weight

        loader = AutoWeightsLoader(self)
        skip_mapper = (
            WeightsMapper(orig_to_new_prefix={"lm_head.": None})
            if self.config.tie_word_embeddings
            else None
        )
        # load_weights fully consumes the generator before returning, so
        # loaded_experts is complete when the union is taken.
        ordinary_loaded = loader.load_weights(_ordinary_weights(), mapper=skip_mapper)
        return loaded_experts | ordinary_loaded


def _tpu_tp_all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Gather a materialized encoder output without leaving the TPU."""
    materialized = fresh_buffer(input_)
    synchronize_tensors(materialized)
    gathered = tensor_model_parallel_all_gather(materialized, dim=dim)
    synchronize_tensors(gathered)
    return gathered


@MULTIMODAL_REGISTRY.register_processor(
    KimiK3MultiModalProcessor,
    info=KimiK3ProcessingInfo,
    dummy_inputs=KimiK3DummyInputsBuilder,
)
class KimiK3ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsEncoderCudaGraph,
    SupportsQuant,
    HasInnerState,
    IsHybrid,
):
    """Kimi-K3 with upstream preprocessing and TPU model execution."""

    supports_encoder_tp_data = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.layers.": "language_model.model.layers.",
            "mm_projector.proj.0": "mm_projector.linear_1",
            "mm_projector.proj.2": "mm_projector.linear_2",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """Return multimodal placeholder token string for `modality`."""
        if modality == "image" and i >= 0:
            return "<|kimi_image_placeholder|>"
        msg = f"Unsupported modality: {modality}"
        raise ValueError(msg)

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Initialize the vision tower, multimodal projector, and language model."""
        super().__init__()
        model_config = vllm_config.model_config
        config = cast("KimiK3Config", model_config.hf_config)
        if model_config.multimodal_config is None:
            msg = "KimiK3ForConditionalGeneration requires multimodal config"
            raise ValueError(msg)

        self.config = config
        self.hidden_size = config.text_config.hidden_size
        self.device = current_platform.current_device()
        self.use_data_parallel = is_vit_use_data_parallel(
            config.vision_config.num_attention_heads
        )

        vision_quant_config = self._maybe_ignore_quant_config(vllm_config.quant_config)
        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = KimiK3MoonViT3dPretrainedModel(
                config.vision_config,
                quant_config=vision_quant_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            if vision_quant_config is None:
                self.vision_tower = self.vision_tower.to(
                    device=self.device, dtype=cast("torch.dtype", model_config.dtype)
                )
            else:
                self.vision_tower = self.vision_tower.to(device=self.device)

            projector_cls = cast("Callable[..., nn.Module]", KimiK25MultiModalProjector)
            self.mm_projector = projector_cls(
                config=config.vision_config,
                out_hidden_size=config.text_config.hidden_size,
                use_data_parallel=self.use_data_parallel,
                quant_config=vision_quant_config,
                prefix=maybe_prefix(prefix, "mm_projector"),
            )
            self.mm_projector = self.mm_projector.to(
                device=self.device, dtype=cast("torch.dtype", model_config.dtype)
            )

        self.quant_config = vllm_config.quant_config
        with self._mark_language_model(vllm_config):
            self.language_model = cast(
                "KimiLinearForCausalLM",
                init_vllm_registered_model(
                    vllm_config=vllm_config,
                    hf_config=config.text_config,
                    prefix=maybe_prefix(prefix, "language_model"),
                    architectures=["KimiLinearForCausalLM"],
                ),
            )
        self.media_placeholder = config.media_placeholder_token_id

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """Forward auxiliary hidden-state layer indices to the language model."""
        if not hasattr(self.language_model, "set_aux_hidden_state_layers"):
            msg = "Kimi language model does not expose aux hidden states"
            raise RuntimeError(msg)
        self.language_model.set_aux_hidden_state_layers(layers)

    @staticmethod
    def _maybe_ignore_quant_config(
        quant_config: QuantizationConfig | None,
    ) -> QuantizationConfig | None:
        """Strip `CompressedTensorsConfig` for the unquantized vision tower."""
        if isinstance(quant_config, compressed_tensors.CompressedTensorsConfig):
            return None
        return quant_config

    def _parse_and_validate_media_input(
        self, **kwargs: object
    ) -> KimiK25MediaPixelInputs | None:
        """Validate and normalize `pixel_values` and `grid_thws` inputs."""
        pixel_values = kwargs.pop("pixel_values", None)
        grid_thws = kwargs.pop("grid_thws", None)
        if pixel_values is None:
            return None

        if isinstance(pixel_values, list):
            pixel_values = torch.cat(cast("list[torch.Tensor]", pixel_values), dim=0)
        if not isinstance(pixel_values, torch.Tensor):
            msg = (
                "pixel_values must be a tensor or list of tensors, "
                f"got {type(pixel_values)}"
            )
            raise TypeError(msg)

        if pixel_values.ndim in (3, 5):
            pixel_values = pixel_values.reshape(
                pixel_values.shape[0] * pixel_values.shape[1], *pixel_values.shape[2:]
            )
        pixel_values = pixel_values.to(dtype=next(self.vision_tower.parameters()).dtype)

        if not isinstance(grid_thws, torch.Tensor):
            msg = f"grid_thws must be a tensor, got {type(grid_thws)}"
            raise TypeError(msg)
        grid_thws = grid_thws.reshape(-1, grid_thws.shape[-1])
        if (
            grid_thws.ndim != ASSERT_GRID_THW_NDIM
            or grid_thws.shape[1] != ASSERT_GRID_THW_COLS
        ):
            msg = f"unexpected grid_thws shape: {grid_thws.shape}"
            raise ValueError(msg)

        return KimiK25MediaPixelInputs(
            type="pixel_values", pixel_values=pixel_values, grid_thws=grid_thws
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Compute vision tower and projector embeddings for multimodal inputs."""
        media_input = self._parse_and_validate_media_input(**kwargs)
        if media_input is None:
            return []
        if self.use_data_parallel:
            # Keep upstream's DP sharding/reassembly, but isolate its TPU
            # collective from the rank-local lazy encoder graphs.
            with patch.object(
                vision_utils,
                "tensor_model_parallel_all_gather",
                _tpu_tp_all_gather,
            ):
                return vision_tower_forward(
                    self.vision_tower,
                    media_input["pixel_values"],
                    media_input["grid_thws"],
                    mm_projector=self.mm_projector,
                    use_data_parallel=True,
                )
        return vision_tower_forward(
            self.vision_tower,
            media_input["pixel_values"],
            media_input["grid_thws"],
            mm_projector=self.mm_projector,
            use_data_parallel=False,
        )

    # -- SupportsEncoderCudaGraph protocol methods --

    supports_encoder_cudagraph = True

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        """Return encoder compilation configuration for image inputs."""
        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=[
                "pixel_values",
                "rope_freqs_cis",
                "cu_seqlens",
                "sequence_lengths",
                "max_seqlen",
                "pos_embeds",
                "merge_gather_idx",
            ],
            out_hidden_size=self.hidden_size,
        )

    def get_input_modality(self, mm_kwargs: dict[str, object]) -> str:
        """Return the active multimodal input modality."""
        modality = mm_kwargs.get("modality")
        return modality if isinstance(modality, str) else "image"

    def get_max_frames_per_video(self) -> int:
        """Return maximum frames per video item."""
        return 1

    def get_encoder_cudagraph_budget_range(
        self,
        vllm_config: VllmConfig,
    ) -> tuple[int, int]:
        """Return `(min_budget, max_budget)` in output vision tokens."""
        # Min: a 224x224 image -> 16x16 patches (patch_size=14) -> 2x2
        # merge -> 8x8 = 64 output tokens (mirrors Qwen2.5-VL's estimate).
        min_budget = 64
        max_budget = min(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.model_config.max_model_len,
        )
        return (min_budget, max_budget)

    @staticmethod
    def _grid_thw_list(grid_thws: object) -> list[list[int]]:
        """Convert `grid_thws` tensor or nested list into `list[list[int]]`."""
        if isinstance(grid_thws, torch.Tensor):
            grid_thws = cast("torch.Tensor", grid_thws).tolist()
        return cast("list[list[int]]", grid_thws)

    def get_encoder_cudagraph_item_specs(
        self, mm_kwargs: dict[str, object]
    ) -> list[EncoderItemSpec]:
        """Compute per-item patch counts and merged output token counts."""
        merge_h, merge_w = self.vision_tower.merge_kernel_size
        grid_thw_list = self._grid_thw_list(mm_kwargs["grid_thws"])
        return [
            EncoderItemSpec(
                input_size=t * h * w,
                output_tokens=t * (h // merge_h) * (w // merge_w),
            )
            for t, h, w in grid_thw_list
        ]

    def select_encoder_cudagraph_items(
        self,
        mm_kwargs: dict[str, object],
        indices: list[int],
    ) -> dict[str, object]:
        """Slice `pixel_values` and `grid_thws` for selected item `indices`."""
        pixel_values = cast("torch.Tensor", mm_kwargs["pixel_values"])
        grid_thws = cast("torch.Tensor", mm_kwargs["grid_thws"])
        grid_thw_list = self._grid_thw_list(grid_thws)

        if len(indices) == 0:
            return {
                "pixel_values": pixel_values[:0],
                "grid_thws": grid_thws[:0],
            }

        patches_per_item = [t * h * w for t, h, w in grid_thw_list]
        cum = [0]
        for patch_count in patches_per_item:
            cum.append(cum[-1] + patch_count)
        selected_pv = torch.cat([pixel_values[cum[i] : cum[i + 1]] for i in indices])
        return {
            "pixel_values": selected_pv,
            "grid_thws": grid_thws[indices],
        }

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        *args: object,
        **kwargs: object,
    ) -> EncoderCudaGraphCaptureInputs:
        """Create dummy vision inputs and metadata for encoder graph capture."""
        arg_names = (
            "token_budget",
            "max_batch_size",
            "max_items",
            "device",
            "dtype",
            "vllm_config",
            "axis_keys",
        )
        merged = {**kwargs, **dict(zip(arg_names, args, strict=False))}
        token_budget = int(cast("int", merged["token_budget"]))
        max_batch_size = int(cast("int", merged["max_batch_size"]))
        device = cast("torch.device", merged["device"])
        dtype = cast("torch.dtype", merged["dtype"])
        axis_keys = merged.get("axis_keys")
        if axis_keys:
            msg = "Kimi K3 does not define encoder capture axes"
            raise ValueError(msg)

        merge_h, merge_w = self.vision_tower.merge_kernel_size
        # Ceil so captured capacity is never smaller than token_budget when
        # it doesn't divide evenly by max_batch_size.
        per_item_output = (token_budget + max_batch_size - 1) // max_batch_size
        # Lay each dummy item out row by row inside the rope table, which only
        # spans rope_2d.max_height x max_width patches (512 x 512 for K3). A
        # single row overflows it as soon as few items share a large budget:
        # 8 items per batch at 8192 tokens is 1024 tokens, 2048 patches wide.
        # Filling rows first leaves every layout that already fit unchanged.
        rope = self.vision_tower.encoder.rope_2d
        out_w = min(per_item_output, rope.max_width // merge_w)
        out_h = (per_item_output + out_w - 1) // out_w
        if out_h * merge_h > rope.max_height:
            msg = (
                f"Encoder cudagraph budget {token_budget} over "
                f"{max_batch_size} items needs {per_item_output} tokens per "
                f"item, more than Kimi K3's "
                f"{rope.max_height}x{rope.max_width}-patch RoPE table holds; "
                "lower max_num_batched_tokens or raise "
                "encoder_cudagraph_max_vision_items_per_batch."
            )
            raise ValueError(msg)
        grid_config = [
            [1, out_h * merge_h, out_w * merge_w] for _ in range(max_batch_size)
        ]

        # vision_tower.patch_size is the raw (unnormalized, often a plain
        # int) HF config value; patch_embed.patch_size is always the
        # normalized (h, w) tuple.
        patch_h, patch_w = self.vision_tower.patch_embed.patch_size
        in_channels = self.vision_tower.patch_embed.proj.in_channels
        total_patches = sum(t * h * w for t, h, w in grid_config)
        dummy_pixel_values = torch.randn(
            total_patches, in_channels, patch_h, patch_w, device=device, dtype=dtype
        )

        # max_seqlen gets baked into the compiled graph as a constant (not
        # replayed -- see MMEncoderManager._pad_to_template's ndim==0
        # passthrough), so it must cover the worst single-item-consumes-
        # the-whole-budget replay case, and the dummy item itself, which
        # rounds up to whole rows.
        metadata = self.vision_tower.prepare_encoder_cudagraph_metadata(
            grid_config,
            max_batch_size=max_batch_size,
            max_seqlen_override=max(
                token_budget * merge_h * merge_w,
                out_h * merge_h * out_w * merge_w,
            ),
            device=device,
        )
        values = {key: value for key, value in metadata.items() if value is not None}
        values["pixel_values"] = dummy_pixel_values
        return EncoderCudaGraphCaptureInputs(values=values)

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, object],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        """Build per-step replay buffers for a captured encoder graph."""
        if max_frames_per_batch < 0 or not path:
            msg = "invalid encoder replay buffer configuration"
            raise ValueError(msg)
        pixel_values = cast("torch.Tensor", mm_kwargs["pixel_values"])
        grid_thw_list = self._grid_thw_list(mm_kwargs["grid_thws"])
        metadata = self.vision_tower.prepare_encoder_cudagraph_metadata(
            grid_thw_list,
            max_batch_size=max_batch_size,
            device=pixel_values.device,
        )
        values = dict(metadata)
        values["pixel_values"] = pixel_values
        return EncoderCudaGraphReplayBuffers(values=values)

    def encoder_cudagraph_forward(
        self,
        inputs: dict[str, torch.Tensor],
        path: str = "default",
    ) -> torch.Tensor:
        """Encode vision patches and project multimodal embeddings in graph mode."""
        if not path:
            msg = "encoder_cudagraph_forward path must be non-empty"
            raise ValueError(msg)
        pixel_values = inputs.pop("pixel_values")
        merge_gather_idx = inputs.pop("merge_gather_idx")
        pos_embeds = inputs.pop("pos_embeds")
        hidden_states = self.vision_tower.patch_embed(
            pixel_values, None, pos_embeds=pos_embeds
        )
        hidden_states = self.vision_tower.encoder(
            hidden_states, None, encoder_metadata=inputs
        )
        merged = tpool_patch_merger_packed(hidden_states, merge_gather_idx)
        return self.mm_projector(merged)

    def encoder_eager_forward(
        self,
        mm_kwargs: dict[str, object],
        path: str = "default",
    ) -> torch.Tensor:
        """Encode rank-local vision patches eagerly before cross-rank gather."""
        if not path:
            msg = "encoder_eager_forward path must be non-empty"
            raise ValueError(msg)
        # LOCAL-ONLY forward, mirroring Qwen3-VL's encoder_eager_forward.
        # Under mm-encoder-tp-mode=data the manager/runner has ALREADY
        # sharded the batch across TP ranks before calling this, so
        # embedding the full batch via vision_tower_forward's
        # run_dp_sharded_mrope_vision_model (Kimi's 12 vision heads < TP=8
        # force vision data-parallelism) would launch a cross-rank
        # all_gather that only the ranks hitting this eager fallback join --
        # ranks whose items fit a budget graph never issue it, so the
        # collective desyncs and the whole engine deadlocks (observed as
        # "No available shared memory broadcast block found" every 60s with
        # workers idle). Process only this rank's own subset: the runner
        # gathers the per-rank outputs at the DP level afterwards.
        pixel_values = cast("torch.Tensor", mm_kwargs["pixel_values"])
        grid_thw_list = self._grid_thw_list(mm_kwargs["grid_thws"])
        encoder_metadata = self.vision_tower.encoder.prepare_encoder_metadata(
            grid_thw_list, device=pixel_values.device
        )
        vt_outputs = self.vision_tower(
            pixel_values,
            grid_thw_list,
            encoder_metadata=encoder_metadata,
        )
        outputs = mm_projector_forward(self.mm_projector, list(vt_outputs))
        return torch.cat(list(outputs), dim=0)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed text tokens and merge multimodal embeddings at placeholders."""
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.get_language_model().embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if is_multimodal is None:
            msg = "is_multimodal is required when merging vision embeddings"
            raise ValueError(msg)

        mm_embeds = _flatten_embeddings(multimodal_embeddings).to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        mask_cpu = is_multimodal.detach().reshape(-1).to(device="cpu", dtype=torch.bool)
        expected = int(mask_cpu.sum().item())
        actual = mm_embeds.shape[0]
        if actual != expected:
            msg = (
                f"Attempted to assign {actual} multimodal tokens to "
                f"{expected} placeholders"
            )
            raise ValueError(msg)
        if actual == 0:
            return inputs_embeds

        media_indices_cpu = mask_cpu.to(torch.int64).cumsum(0).sub_(1)
        media_indices_cpu.clamp_(min=0)
        media_indices = media_indices_cpu.to(device=inputs_embeds.device)
        aligned_media = mm_embeds.index_select(0, media_indices)
        mask = is_multimodal.reshape(-1, 1).to(
            device=inputs_embeds.device, dtype=torch.bool
        )
        merged = torch.where(mask, aligned_media, inputs_embeds)
        if merged.device.type != "tpu":
            return merged

        materialized = fresh_buffer(merged)
        synchronize_tensors(materialized)
        return materialized

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass delegating to the underlying Kimi language model."""
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, **_kwargs: object
    ) -> torch.Tensor | None:
        """Compute next-token vocabulary logits from `hidden_states`."""
        return self.language_model.compute_logits(hidden_states)

    def process_weights_after_loading(self) -> None:
        """Finalize post-load attention-residual and MoE weights on the model."""
        # The language model walks its own AttentionResidual modules.
        self.language_model.process_weights_after_loading()

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        """Return `(conv_state_dtype, ssm_state_dtype)` from text config."""
        text_config = vllm_config.model_config.hf_config.text_config
        return KimiLinearForCausalLM.get_mamba_state_dtype_from_config(
            vllm_config.with_hf_config(text_config)
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """Return `(conv_state_shape, ssm_state_shape)` from text config."""
        text_config = vllm_config.model_config.hf_config.text_config
        return KimiLinearForCausalLM.get_mamba_state_shape_from_config(
            vllm_config.with_hf_config(text_config)
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        """Return KDA state copy functions for prefix caching."""
        return KimiLinearForCausalLM.get_mamba_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load vision, projector, and language-model checkpoint weights."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

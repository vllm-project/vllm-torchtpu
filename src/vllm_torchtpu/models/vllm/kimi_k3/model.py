# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 and Kimi-Linear models for TPU."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast
from unittest.mock import patch

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather
from vllm.model_executor.layers.fused_moe import \
    fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc, MambaStateCopyFuncCalculator)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors import \
    compressed_tensors
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models import vision as vision_utils
from vllm.model_executor.models.interfaces import (HasInnerState, IsHybrid,
                                                   SupportsMultiModal,
                                                   SupportsQuant)
from vllm.model_executor.models.kimi_k25 import KimiK25MediaPixelInputs
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector, vision_tower_forward)
from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper,
                                              _flatten_embeddings,
                                              init_vllm_registered_model,
                                              maybe_prefix)
from vllm.model_executor.models.vision import is_vit_use_data_parallel
from vllm.models.kimi_k3.common.mm_preprocess import (
    KimiK3DummyInputsBuilder, KimiK3MultiModalProcessor, KimiK3ProcessingInfo)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import NestedTensors
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

from .attention import (KimiDeltaAttention, MultiHeadLatentAttention,
                        kda_state_dtype)
from .collective_ops import TokenShardCollectives
from .kimi_vit import KimiK3MoonViT3dPretrainedModel
from .layers import AttentionResidual, KimiMLP
from .moe import KimiMoE

logger = init_logger(__name__)


class KimiDecoderLayer(nn.Module):

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
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

        is_moe = (config.num_experts is not None
                  and self.layer_idx >= config.first_k_dense_replace
                  and self.layer_idx % config.moe_layer_freq == 0)
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
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps)

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
        sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.attn_res_block_size is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(positions,
                                           hidden_states,
                                           sequence_parallel=sequence_parallel)
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            if self.is_moe:
                hidden_states = self.block_sparse_moe(
                    hidden_states, sequence_parallel=sequence_parallel)
            else:
                hidden_states = self.mlp(hidden_states,
                                         sequence_parallel=sequence_parallel)
            return residual + hidden_states, None

        assert block_residuals is not None
        prefix_sum = hidden_states
        if block_residuals.shape[-2] > 0:
            hidden_states = self.self_attention_res(prefix_sum,
                                                    block_residuals)
        if self.layer_idx % self.attn_res_block_size == 0:
            block_residuals = torch.cat(
                (block_residuals, prefix_sum.unsqueeze(-2)), dim=-2)
            prefix_sum = None

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions,
                                       hidden_states,
                                       sequence_parallel=sequence_parallel)
        prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states
        hidden_states = self.mlp_res(prefix_sum, block_residuals)
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            hidden_states = self.block_sparse_moe(
                hidden_states, sequence_parallel=sequence_parallel)
        else:
            hidden_states = self.mlp(hidden_states,
                                     sequence_parallel=sequence_parallel)
        return prefix_sum + hidden_states, block_residuals


@support_torch_compile(dynamic_arg_dims={
    "input_ids": 0,
    "positions": 0,
    "inputs_embeds": 0,
})
class KimiModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: KimiLinearConfig = vllm_config.model_config.hf_text_config
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
            if (parallel.tensor_parallel_size != 32
                    or not parallel.enable_expert_parallel
                    or parallel.pipeline_parallel_size != 1
                    or parallel.data_parallel_size != 1):
                raise ValueError(
                    "K3 SP prefill requires TP32/EP32 and PP1/DP1")
            self.sp_group = TokenShardCollectives(get_tp_group())

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList([
            KimiDecoderLayer(
                config,
                vllm_config,
                prefix=f"{prefix}.layers.{layer_idx}",
            ) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.aux_hidden_state_layers: tuple[int, ...] = ()
        if self.attn_res_block_size is not None:
            self.output_attn_res = AttentionResidual(
                config.hidden_size,
                config.rms_norm_eps,
                prefix=f"{prefix}.output_attn_res_proj",
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = (inputs_embeds if inputs_embeds is not None else
                         self.embed_input_ids(input_ids))
        num_tokens = hidden_states.shape[0]
        sequence_parallel = (self.sp_prefill and num_tokens > 0
                             and num_tokens % self.sp_group.world_size == 0)
        if sequence_parallel:
            hidden_states = hidden_states.reshape(
                self.sp_group.world_size, -1, hidden_states.shape[-1])[
                    self.sp_group.rank_in_group].contiguous()
        # A view avoids passing a symbolic local token count to TorchTPU's
        # concrete-size allocation kernel during the dynamic trace.
        block_residuals = (hidden_states.unsqueeze(1)[:, :0, :]
                           if self.attn_res_block_size is not None else None)
        assert hidden_states is not None

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
            hidden_states = self.output_attn_res(hidden_states,
                                                 block_residuals)
        hidden_states = self.norm(hidden_states)
        if sequence_parallel:
            hidden_states = self.sp_group.all_gather(hidden_states, dim=0)
            aux_hidden_states = [
                self.sp_group.all_gather(state, dim=0)
                for state in aux_hidden_states
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
    fused_mla_mapper = WeightsMapper(orig_to_new_stacked={
        ".q_a_proj.": (".fused_qkv_a_proj.", 0),
        ".kv_a_proj_with_mqa.": (".fused_qkv_a_proj.", 1),
    }, )
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
        super().__init__()
        self.config: KimiLinearConfig = vllm_config.model_config.hf_text_config
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
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        invalid = [
            idx for idx in layers if idx < 0 or idx > len(self.model.layers)
        ]
        if invalid:
            raise ValueError(
                f"Invalid Kimi aux hidden-state layers: {invalid}")
        self.model.aux_hidden_state_layers = tuple(layers)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors: object | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise ValueError("The TPU Kimi model does not support PP")
        return self.model(
            input_ids,
            positions,
            inputs_embeds,
        )

    def compute_logits(self,
                       hidden_states: torch.Tensor) -> torch.Tensor | None:
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
        # Same helper the layer uses. vLLM sizes the cache from here and
        # allocates from get_state_dtype, so the two cannot diverge.
        return kda_state_dtype(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], tuple[int, int, int]]:
        config: KimiLinearConfig = vllm_config.model_config.hf_text_config
        kda_config = config.linear_attn_config
        assert kda_config is not None
        local_heads = (kda_config["num_heads"] //
                       vllm_config.parallel_config.tensor_parallel_size)
        return (
            (kda_config["short_conv_kernel_size"] - 1, 3, local_heads,
             kda_config["head_dim"]),
            (local_heads, kda_config["head_dim"], kda_config["head_dim"]),
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls, ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        expert_params_mapping = (fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_experts,
        ) if self.config.num_experts is not None else [])
        params_dict = dict(self.named_parameters())
        experts_use_weight = not any(
            name.endswith("w13_weight_packed") for name in params_dict)
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
        def _ordinary_weights():
            for name, loaded_weight in mapper.apply(weights):
                if name.startswith(("vision_tower.", "mm_projector.")):
                    continue
                if experts_use_weight and name.endswith(".weight_packed"):
                    name = name.replace(".weight_packed", ".weight")

                for (weight_name, param_name,
                     shard_id) in self.fused_attention_params_mapping:
                    if weight_name not in name:
                        continue
                    candidate = name.replace(weight_name, param_name)
                    if candidate not in params_dict:
                        continue
                    name = candidate
                    loaded_weight.shard_id = shard_id
                    break

                for (param_name, weight_name, expert_id,
                     expert_shard_id) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    parameter = params_dict[name]
                    parameter.weight_loader(
                        parameter,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=expert_shard_id,
                    )
                    loaded_experts.add(name)
                    break
                else:
                    yield name, loaded_weight

        loader = AutoWeightsLoader(self)
        skip_mapper = (WeightsMapper(orig_to_new_prefix={"lm_head.": None})
                       if self.config.tie_word_embeddings else None)
        # load_weights fully consumes the generator before returning, so
        # loaded_experts is complete when the union is taken.
        ordinary_loaded = loader.load_weights(_ordinary_weights(),
                                              mapper=skip_mapper)
        return loaded_experts | ordinary_loaded


def _tpu_tp_all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Gather a materialized encoder output without leaving the TPU."""
    materialized = torch.empty_like(input_).copy_(input_)
    synchronize_tensors(materialized)
    gathered = tensor_model_parallel_all_gather(materialized, dim=dim)
    synchronize_tensors(gathered)
    return gathered


@MULTIMODAL_REGISTRY.register_processor(
    KimiK3MultiModalProcessor,
    info=KimiK3ProcessingInfo,
    dummy_inputs=KimiK3DummyInputsBuilder,
)
class KimiK3ForConditionalGeneration(nn.Module, SupportsMultiModal,
                                     SupportsQuant, HasInnerState, IsHybrid):
    """Kimi-K3 with upstream preprocessing and TPU model execution."""

    supports_encoder_tp_data = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.layers.": "language_model.model.layers.",
            "mm_projector.proj.0": "mm_projector.linear_1",
            "mm_projector.proj.2": "mm_projector.linear_2",
        })

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        del i
        if modality == "image":
            return "<|kimi_image_placeholder|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config: KimiK3Config = model_config.hf_config
        if model_config.multimodal_config is None:
            raise ValueError(
                "KimiK3ForConditionalGeneration requires multimodal config")

        self.config = config
        self.hidden_size = config.text_config.hidden_size
        self.device = current_platform.current_device()
        self.use_data_parallel = is_vit_use_data_parallel(
            config.vision_config.num_attention_heads)

        vision_quant_config = self._maybe_ignore_quant_config(
            vllm_config.quant_config)
        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = KimiK3MoonViT3dPretrainedModel(
                config.vision_config,
                quant_config=vision_quant_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            if vision_quant_config is None:
                self.vision_tower = self.vision_tower.to(
                    device=self.device, dtype=model_config.dtype)
            else:
                self.vision_tower = self.vision_tower.to(device=self.device)

            self.mm_projector = KimiK25MultiModalProjector(
                config=config.vision_config,
                use_data_parallel=self.use_data_parallel,
                quant_config=vision_quant_config,
                prefix=maybe_prefix(prefix, "mm_projector"),
            )
            self.mm_projector = self.mm_projector.to(device=self.device,
                                                     dtype=model_config.dtype)

        self.quant_config = vllm_config.quant_config
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["KimiLinearForCausalLM"],
            )
        self.media_placeholder = config.media_placeholder_token_id

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        if not hasattr(self.language_model, "set_aux_hidden_state_layers"):
            raise RuntimeError(
                "Kimi language model does not expose aux hidden states")
        self.language_model.set_aux_hidden_state_layers(layers)

    @staticmethod
    def _maybe_ignore_quant_config(
        quant_config: QuantizationConfig | None,
    ) -> QuantizationConfig | None:
        if isinstance(quant_config,
                      compressed_tensors.CompressedTensorsConfig):
            return None
        return quant_config

    def _parse_and_validate_media_input(
            self, **kwargs: object) -> KimiK25MediaPixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        grid_thws = kwargs.pop("grid_thws", None)
        if pixel_values is None:
            return None

        if isinstance(pixel_values, list):
            pixel_values = torch.cat(cast(list[torch.Tensor], pixel_values),
                                     dim=0)
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(
                "pixel_values must be a tensor or list of tensors, "
                f"got {type(pixel_values)}")

        if pixel_values.ndim in (3, 5):
            pixel_values = pixel_values.reshape(
                pixel_values.shape[0] * pixel_values.shape[1],
                *pixel_values.shape[2:])
        pixel_values = pixel_values.to(
            dtype=next(self.vision_tower.parameters()).dtype)

        if not isinstance(grid_thws, torch.Tensor):
            raise TypeError(
                f"grid_thws must be a tensor, got {type(grid_thws)}")
        grid_thws = grid_thws.reshape(-1, grid_thws.shape[-1])
        if grid_thws.ndim != 2 or grid_thws.shape[1] != 3:
            raise ValueError(f"unexpected grid_thws shape: {grid_thws.shape}")

        return KimiK25MediaPixelInputs(type="pixel_values",
                                       pixel_values=pixel_values,
                                       grid_thws=grid_thws)

    def embed_multimodal(self, **kwargs: object) -> NestedTensors | None:
        media_input = self._parse_and_validate_media_input(**kwargs)
        if media_input is None:
            return None
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

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.get_language_model().embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if is_multimodal is None:
            raise ValueError(
                "is_multimodal is required when merging vision embeddings")

        mm_embeds = _flatten_embeddings(multimodal_embeddings).to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        mask_cpu = is_multimodal.detach().reshape(-1).to(device="cpu",
                                                         dtype=torch.bool)
        expected = int(mask_cpu.sum().item())
        actual = mm_embeds.shape[0]
        if actual != expected:
            raise ValueError(
                f"Attempted to assign {actual} multimodal tokens to "
                f"{expected} placeholders")
        if actual == 0:
            return inputs_embeds

        media_indices_cpu = mask_cpu.to(torch.int64).cumsum(0).sub_(1)
        media_indices_cpu.clamp_(min=0)
        media_indices = media_indices_cpu.to(device=inputs_embeds.device)
        aligned_media = mm_embeds.index_select(0, media_indices)
        mask = is_multimodal.reshape(-1, 1).to(device=inputs_embeds.device,
                                               dtype=torch.bool)
        merged = torch.where(mask, aligned_media, inputs_embeds)
        if merged.device.type != "tpu":
            return merged

        materialized = torch.empty_like(merged).copy_(merged)
        synchronize_tensors(materialized)
        return materialized

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del kwargs
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor,
                       **kwargs: object) -> torch.Tensor | None:
        del kwargs
        return self.language_model.compute_logits(hidden_states)

    def process_weights_after_loading(self) -> None:
        # The language model walks its own AttentionResidual modules.
        self.language_model.process_weights_after_loading()

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        text_config = vllm_config.model_config.hf_config.text_config
        return KimiLinearForCausalLM.get_mamba_state_dtype_from_config(
            vllm_config.with_hf_config(text_config))

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        text_config = vllm_config.model_config.hf_config.text_config
        return KimiLinearForCausalLM.get_mamba_state_shape_from_config(
            vllm_config.with_hf_config(text_config))

    @classmethod
    def get_mamba_state_copy_func(cls):
        return KimiLinearForCausalLM.get_mamba_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

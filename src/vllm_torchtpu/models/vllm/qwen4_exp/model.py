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
"""Qwen4Exp (Qwen3.8 Flash Next) text model for TPU.

Follows the layer structure of upstream vllm/models/qwen4_exp/nvidia/model.py;
upstream enables Qwen4Exp only on CUDA and ROCm. Hyper-connections use the
upstream torch implementation in qwen4_exp.common; the sublayers reuse the GDN,
gated attention and sparse MoE layers that Qwen3.5 runs on TPU.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextSparseMoeBlock,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    maybe_prefix,
)
from vllm.models.qwen4_exp.common.hyperconnection import (
    GatedResidual,
    HyperConnectionConfig,
)
from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig


def _hc_config(vllm_config: VllmConfig) -> HyperConnectionConfig:
    config = cast(Qwen4ExpTextConfig, vllm_config.model_config.hf_text_config)
    # Qwen4Exp checkpoints store one hc_norm weight per element of the
    # hc_count * hidden_size stream, normalized per stream.
    return HyperConnectionConfig(
        hc_count=config.hc_count,
        hidden_size=config.hidden_size,
        params_dtype=torch.get_default_dtype(),
        hc_lowrank=config.hc_lowrank,
        rms_norm_eps=config.rms_norm_eps,
        hc_per_branch_norm=True,
    )


class Qwen4ExpDecoderLayer(nn.Module):
    """Attention and MoE sublayers, each wrapped by a gated hyper-connection.

    The residual is hc_count * hidden_size wide; hc_norm replaces the usual
    input and post-attention norms.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = cast(Qwen4ExpTextConfig, vllm_config.model_config.hf_text_config)
        layer_types = config.layer_types
        assert layer_types is not None
        self.layer_type = layer_types[extract_layer_index(prefix)]

        if self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
            )
        elif self.layer_type == "full_attention":
            # TODO(QSA): add the indexer and attend over the selected blocks.
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=vllm_config.model_config,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Unsupported Qwen4Exp layer type {self.layer_type!r}")

        self.mlp = Qwen3NextSparseMoeBlock(vllm_config, prefix=f"{prefix}.mlp")
        hc_config = _hc_config(vllm_config)
        self.attn_hyper_connection = GatedResidual(hc_config)
        self.mlp_hyper_connection = GatedResidual(hc_config)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        block_input, residuals = self.attn_hyper_connection.mix(hidden_states)
        if self.layer_type == "linear_attention":
            block_output = self.linear_attn(hidden_states=block_input)
        else:
            block_output = self.self_attn(
                positions=positions, hidden_states=block_input
            )
        hidden_states = self.attn_hyper_connection.combine(block_output, residuals)

        block_input, residuals = self.mlp_hyper_connection.mix(hidden_states)
        block_output = self.mlp(block_input)
        return self.mlp_hyper_connection.combine(block_output, residuals)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
    }
)
class Qwen4ExpModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = cast(Qwen4ExpTextConfig, vllm_config.model_config.hf_text_config)
        self.hc_count = config.hc_count
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList(
            Qwen4ExpDecoderLayer(vllm_config, prefix=f"{prefix}.layers.{idx}")
            for idx in range(config.num_hidden_layers)
        )
        # Reduces the hc_count streams to the single stream fed to lm_head.
        # It only mixes, so it has no block_inject_weight.
        self.hyper_connection_mixer = GatedResidual(
            _hc_config(vllm_config), use_combine=False
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds.repeat(1, self.hc_count)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        hidden_states, _ = self.hyper_connection_mixer.mix(hidden_states)
        return hidden_states


class Qwen4ExpForCausalLM(nn.Module, HasInnerState, IsHybrid):
    """Text-only Qwen4Exp on TPU."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
    # Qwen3.5's mapping fuses the split GDN, attention and shared-expert
    # projections that Qwen4Exp checkpoints use as well.
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | WeightsMapper(
        # TODO(QSA): load the indexer weights once the QSA indexer lands;
        # dense attention does not use them.
        orig_to_new_substr={".self_attn.indexer.": None},
        orig_to_new_prefix={"model.language_model.": "model.", "mtp.": None},
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = cast(Qwen4ExpTextConfig, vllm_config.model_config.hf_text_config)
        self.model = Qwen4ExpModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: object | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise NotImplementedError("Qwen4Exp on TPU does not support PP")
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        config = cast(Qwen4ExpTextConfig, vllm_config.model_config.hf_text_config)
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.linear_conv_kernel_dim,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

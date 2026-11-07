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
"""MiniMax M3 (text) on TPU."""

from collections.abc import Iterable

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)

from vllm_torchtpu.models.minimax_m3.attention import MiniMaxM3SparseAttention
from vllm_torchtpu.models.minimax_m3.moe import MiniMaxM3MLP, MiniMaxM3MoE
from vllm_torchtpu.models.minimax_m3.quant import CT_TO_MODELOPT_EXPERT_SUFFIX


class MiniMaxM3DecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, layer_id: int, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.self_attn = MiniMaxM3SparseAttention(
            config,
            quant_config,
            vllm_config.cache_config,
            prefix=f"{prefix}.self_attn",
        )
        # Dense layers name their FFN `mlp`, MoE layers `block_sparse_moe`.
        self.is_moe_layer = bool(config.moe_layer_freq[layer_id])
        if self.is_moe_layer:
            self.block_sparse_moe = MiniMaxM3MoE(
                config, quant_config, prefix=f"{prefix}.block_sparse_moe"
            )
        else:
            self.mlp = MiniMaxM3MLP(
                config,
                config.dense_intermediate_size,
                quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        return ffn(hidden_states), residual


@support_torch_compile(
    dynamic_arg_dims={"input_ids": 0, "positions": 0, "inputs_embeds": 0}
)
class MiniMaxM3Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        # Keys this model's compile-artifact cache directory.
        self.prefix = prefix
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        self.layers = nn.ModuleList(
            MiniMaxM3DecoderLayer(vllm_config, idx, prefix=f"{prefix}.layers.{idx}")
            for idx in range(config.num_hidden_layers)
        )
        self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        # Residual-stream entries of these layers (k = output of layer k-1,
        # residual added, before the next norm) are also returned; set by
        # the DSpark drafter before the first trace.
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states, residual = inputs_embeds, None
        aux_hidden_states = []
        if 0 in self.aux_hidden_state_layers:
            aux_hidden_states.append(inputs_embeds)
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)
            if idx + 1 in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # (param, checkpoint shard, shard id). Leading dots keep `.q_proj`
        # from matching `index_q_proj`.
        stacked_params_mapping: list[tuple[str, str, int | str]] = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".qkv_proj", ".index_q_proj", "index_q"),
            (".qkv_proj", ".index_k_proj", "index_k"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            name = name.replace(".weight_scale_inv", ".weight_scale")
            if ".experts." in name:
                for ct_suffix, modelopt_suffix in CT_TO_MODELOPT_EXPERT_SUFFIX.items():
                    name = name.replace(ct_suffix, modelopt_suffix)
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    raise ValueError(f"Unexpected expert weight {name}")
            else:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    param.weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class MiniMaxM3SparseForCausalLM(nn.Module):
    """Text-only MiniMax M3; also serves the vision-language architecture."""

    # compute_logits is the LM head alone (runner/greedy_top_tokens.py).
    logits_from_lm_head = True

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.": "",
            "vision_tower.": None,
            "multi_modal_projector.": None,
            "patch_merge_mlp.": None,
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.model = MiniMaxM3Model(
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

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        num_layers = len(self.model.layers)
        invalid = [idx for idx in layers if not 0 <= idx <= num_layers]
        if invalid:
            raise ValueError(f"Invalid MiniMax M3 aux hidden-state layers: {invalid}")
        self.model.aux_hidden_state_layers = tuple(layers)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: object | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if intermediate_tensors is not None:
            raise NotImplementedError("MiniMax M3 on TPU does not support PP")
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )

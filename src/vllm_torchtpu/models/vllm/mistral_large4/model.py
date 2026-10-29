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
"""Initial Mistral Large 4 text definition for TPU.

Uses the pinned vLLM MLP/MoE layers and local ML4 layout/LeRoPE adapters.
The backbone is compiled for TPU and prepares
queries and sparse indices for the shared TPU MLA core, including cross-layer
KV sharing and TPU FP8 weight methods. Pixtral integration lives in
vision.py; speculative decoding is not integrated yet.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import cast

import torch
from torch import nn
from transformers import DeepseekV3Config, PreTrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLP, DeepseekV2MoE
from vllm.model_executor.models.utils import (
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.model_executor.parameter import BasevLLMParameter

from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.models.vllm.mistral_large4.attention import MistralMLAAttention
from vllm_torchtpu.models.vllm.mistral_large4.checkpoint_config import (
    MistralLayerSpec,
    get_layer_specs,
)
from vllm_torchtpu.models.vllm.mistral_large4.config import TPUMistralLarge4Config
from vllm_torchtpu.models.vllm.mistral_large4.rope import MistralRotaryEmbedding


class MistralLarge4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: PreTrainedConfig,
        spec: MistralLayerSpec,
        prefix: str,
        quant_config: QuantizationConfig | None,
        topk_indices_buffer: torch.Tensor | None = None,
        zero_kv_inputs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MistralMLAAttention(
            vllm_config,
            config,
            spec,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
            zero_kv_inputs=zero_kv_inputs,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp: nn.Module
        if spec.is_moe:
            self.mlp = DeepseekV2MoE(
                cast(DeepseekV3Config, config),
                vllm_config.parallel_config,
                quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV2MLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
                quant_config,
                reduce_results=True,
                prefix=f"{prefix}.mlp",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        log_positions: torch.Tensor,
        rotary_emb: MistralRotaryEmbedding,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions, hidden_states, log_positions, rotary_emb
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


@support_torch_compile(
    dynamic_arg_dims={"input_ids": 0, "positions": -1, "inputs_embeds": 0}
)
class MistralLarge4Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.layer_specs = get_layer_specs(config)

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        # One RoPE table shared by all layers.
        self.rotary_emb = MistralRotaryEmbedding(config, config.qk_rope_head_dim)

        self.topk_indices_buffer = (
            torch.full(
                (
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.index_topk,
                ),
                -1,
                dtype=torch.int32,
            )
            if any(spec.use_sparse for spec in self.layer_specs)
            else None
        )

        zero_kv_inputs = None
        if any(spec.shares_kv for spec in self.layer_specs):
            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            zero_kv_inputs = (
                torch.zeros(max_tokens, config.kv_lora_rank),
                torch.zeros(max_tokens, 1, config.qk_rope_head_dim),
            )
        self.layers = nn.ModuleList(
            MistralLarge4DecoderLayer(
                vllm_config,
                config,
                spec,
                prefix=f"{prefix}.layers.{spec.layer_idx}",
                quant_config=vllm_config.quant_config,
                topk_indices_buffer=self.topk_indices_buffer,
                zero_kv_inputs=zero_kv_inputs,
            )
            for spec in self.layer_specs
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: object = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            assert input_ids is not None
            hidden_states = self.embed_input_ids(input_ids)
        log_positions = torch.log(positions.to(torch.float32) + 1)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, residual, log_positions, self.rotary_emb
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# Mistral consolidated checkpoint name -> vLLM name. Patterns match whole name
# segments, so the map renames both weights and the module names quantization
# configs list (e.g. `layers.1.attention.wq`).
_CHECKPOINT_NAME_MAP: list[tuple[str, str]] = [
    (r"^tok_embeddings(?=\.|$)", "model.embed_tokens"),
    (r"^norm(?=\.|$)", "model.norm"),
    (r"^output(?=\.|$)", "lm_head"),
    (r"^layers\.", "model.layers."),
    (r"\.attention_norm(?=\.|$)", ".input_layernorm"),
    (r"\.ffn_norm(?=\.|$)", ".post_attention_layernorm"),
    (r"\.attention\.indexer\.ssmax_scaling(?=\.|$)", ".self_attn.indexer.ssmax"),
    (r"\.attention\.ssmax_scaling(?=\.|$)", ".self_attn.ssmax"),
    (r"\.attention\.wq(?=\.|$)", ".self_attn.q_proj"),
    (r"\.attention\.wkv_a_with_mqa(?=\.|$)", ".self_attn.kv_a_proj_with_mqa"),
    (r"\.attention\.kv_a_norm(?=\.|$)", ".self_attn.kv_a_layernorm"),
    (r"\.attention\.wkv_b(?=\.|$)", ".self_attn.kv_b_proj"),
    (r"\.attention\.wo(?=\.|$)", ".self_attn.o_proj"),
    (r"\.attention\.indexer(?=\.|$)", ".self_attn.indexer"),
    (r"\.feed_forward(?=\.|$)", ".mlp"),
    (r"\.shared_experts(?=\.|$)", ".mlp.shared_experts"),
    (r"\.gate(?=\.|$)", ".mlp.gate"),
    (r"\.experts(?=\.|$)", ".mlp.experts"),
    (r"\.qscale_weight$", ".weight_scale_inv"),
]

# (fused param, checkpoint shard, shard id)
_STACKED_PARAMS: list[tuple[str, str, int]] = [
    (".gate_up_proj.", ".w1.", 0),
    (".gate_up_proj.", ".w3.", 1),
    (".down_proj.", ".w2.", -1),
    (".wk_weights_proj.", ".wk.", 0),
    (".wk_weights_proj.", ".weights_proj.", 1),
]


def _to_vllm_weight_name(name: str) -> str:
    for pattern, repl in _CHECKPOINT_NAME_MAP:
        name = re.sub(pattern, repl, name)
    return name


def _map_quant_config_layers(quant_config: QuantizationConfig, prefix: str) -> None:
    """Map consolidated checkpoint module names before constructing layers."""
    quant_config.apply_vllm_mapper(
        WeightsMapper(
            orig_to_new_regex={
                re.compile(pattern): repl for pattern, repl in _CHECKPOINT_NAME_MAP
            },
            orig_to_new_prefix={
                "model.": f"{maybe_prefix(prefix, 'model')}.",
                "lm_head": maybe_prefix(prefix, "lm_head"),
            },
        )
    )
    stacked: dict[str, list[str]] = {}
    for param_part, ckpt_part, _ in _STACKED_PARAMS:
        stacked.setdefault(param_part.strip("."), []).append(ckpt_part.strip("."))
    quant_config.packed_modules_mapping = {
        **quant_config.packed_modules_mapping,
        **stacked,
    }


def _resolve_scale_name(name: str, params: dict[str, nn.Parameter]) -> str:
    # Native FP8's canonical name is weight_scale_inv; its weight_scale
    # compatibility alias shares the same storage.
    if name.endswith("weight_scale") and f"{name}_inv" in params:
        return f"{name}_inv"
    if (
        name not in params
        and name.endswith("weight_scale_inv")
        and name.removesuffix("_inv") in params
    ):
        return name.removesuffix("_inv")
    return name


def _load_weights(
    model: nn.Module,
    mistral_model: MistralLarge4Model,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Load a Mistral consolidated checkpoint into ``model``.

    ``model`` holds ``mistral_model`` as ``model.model`` and the LM head as
    ``model.lm_head``.
    """
    config = mistral_model.config
    params = dict(model.named_parameters())
    expert_mapping = fused_moe_make_expert_params_mapping(
        model,
        ckpt_gate_proj_name="w1",
        ckpt_down_proj_name="w2",
        ckpt_up_proj_name="w3",
        num_experts=config.n_routed_experts or 0,
    )
    loaded: set[str] = set()
    lerope_alpha: torch.Tensor | None = None

    for ckpt_name, weight in weights:
        if ckpt_name == "lerope.alpha":
            lerope_alpha = weight
            continue
        if ckpt_name.startswith("mtp_"):
            # Speculative-decoding heads are loaded by their draft model.
            continue
        # Layers beyond the configured depth (e.g. a truncated debug model).
        layer = re.match(r"layers\.(\d+)\.", ckpt_name)
        if layer is not None and int(layer.group(1)) >= config.num_hidden_layers:
            continue
        name = _to_vllm_weight_name(ckpt_name)

        if ".mlp.experts." in name:
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in name:
                    continue
                mapped = _resolve_scale_name(
                    name.replace(weight_name, param_name), params
                )
                param = params[mapped]
                loader = cast(
                    Callable[..., bool], cast(BasevLLMParameter, param).weight_loader
                )
                if loader(
                    param,
                    weight,
                    mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                ):
                    loaded.add(mapped)
                # A recognized expert may belong to another EP rank. Its
                # loader returns False, including for unfiltered scale tensors.
                break
            else:
                raise ValueError(f"Unexpected expert weight: {ckpt_name}")
            continue

        for param_part, ckpt_part, stacked_shard_id in _STACKED_PARAMS:
            if ckpt_part not in name:
                continue
            name = name.replace(ckpt_part, param_part)
            name = _resolve_scale_name(name, params)
            param = params[name]
            loader = cast(BasevLLMParameter, param).weight_loader
            if stacked_shard_id < 0:
                loader(param, weight)
            else:
                loader(param, weight, stacked_shard_id)
            break
        else:
            name = _resolve_scale_name(name, params)
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, weight)
        loaded.add(name)

    # Record both names for FP8 scale aliases backed by the same storage.
    for module_name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if isinstance(method, VllmFp8LinearMethodTPU) and method.block_quant:
            canonical = f"{module_name}.weight_scale_inv"
            if canonical in loaded:
                loaded.add(f"{module_name}.weight_scale")
    if missing := sorted(set(params) - loaded):
        raise ValueError(f"Parameters not loaded from the checkpoint: {missing}")

    if config.lerope:
        assert lerope_alpha is not None, "Checkpoint is missing lerope.alpha"
        mistral_model.rotary_emb.set_lerope_alpha(lerope_alpha)
    return loaded


class MistralLarge4ForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        TPUMistralLarge4Config.verify_and_update_config(vllm_config)
        config = vllm_config.model_config.hf_text_config
        self.config = config
        if vllm_config.quant_config is not None:
            _map_quant_config_layers(vllm_config.quant_config, prefix)
        self.model = MistralLarge4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            config.vocab_size, soft_cap=config.softmax_tanh_gating
        )
        # Preserve the standard vLLM model interface; PP is rejected above.
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: object = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return _load_weights(self, self.model, weights)

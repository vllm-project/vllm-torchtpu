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

import contextlib
from collections.abc import Iterable, Iterator

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
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

from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.models.minimax_m3 import overlap
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
            layer_id=layer_id,
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
        # Two-batch overlap of large steps (`overlap`; off = overlap.OFF).
        self.tp_size = get_tensor_model_parallel_world_size()
        self.overlap = overlap.config_from_env() if self.tp_size > 1 else overlap.OFF

    @contextlib.contextmanager
    def _layer_owned_reductions(self) -> Iterator[None]:
        """While tracing an overlap step, o_proj and the FFN return per-rank
        partial sums and the layer all-reduces them; every other trace keeps
        their own all-reduces, so the other programs are unchanged."""
        o_proj = self.self_attn.o_proj
        if self.is_moe_layer:
            experts = self.block_sparse_moe.experts
            # A MoE op that all-reduces inside it (fused MoE decode / TP
            # decode kernel) would ignore skip_final_all_reduce.
            from vllm_torchtpu.layers.adapter.moe_tp_decode import (
                moe_tp_decode_supported,
            )

            if getattr(experts, "_fused_moe_decode_armed", False) or (
                moe_tp_decode_supported(getattr(experts, "_quant_method", None))
            ):
                raise ValueError(
                    "TPU_MINIMAX_M3_PREFILL_OVERLAP needs a MoE that leaves its "
                    "all-reduce to the model; this layer's MoE op reduces "
                    "inside it (fused MoE decode / TP decode kernel)."
                )
            cfg = experts.moe_config
            saved = (o_proj.reduce_results, cfg.skip_final_all_reduce)
            o_proj.reduce_results, cfg.skip_final_all_reduce = False, True
        else:
            down = self.mlp.down_proj
            saved = (o_proj.reduce_results, down.reduce_results)
            o_proj.reduce_results = down.reduce_results = False
        try:
            yield
        finally:
            if self.is_moe_layer:
                o_proj.reduce_results, cfg.skip_final_all_reduce = saved
            else:
                o_proj.reduce_results, down.reduce_results = saved

    def _attend(self, positions, hidden_states, index_reuse, qkv=None):
        if index_reuse is None:
            return self.self_attn.attend(positions, hidden_states, qkv=qkv)
        return self.self_attn.attend(positions, hidden_states, index_reuse, qkv=qkv)

    def forward_overlap(
        self,
        positions: torch.Tensor,
        hidden_states,
        residual,
        half,
        index_reuse: dict | None = None,
    ):
        """Overlap layer (`overlap`): attention on every token, then o_proj,
        the post-attention norm, the FFN and their all-reduces per token
        half.

        `hidden_states`: the full embedding (first layer, `residual` None)
        or the previous layer's pair of in-flight FFN all-reduces; `residual`:
        None or the pair of residual halves. `half`: rows per half (the
        step's token count symbol // 2). Returns the pair of in-flight FFN
        all-reduces and the pair of residual halves.
        """
        cfg = self.overlap
        route = cfg.ar
        with self._layer_owned_reductions():
            attn_mod = self.self_attn
            if residual is None:
                res_a, res_b = overlap.split(hidden_states, half)
                attn = self._attend(
                    positions, self.input_layernorm(hidden_states), index_reuse
                )
            else:
                ffn_a, ffn_b = hidden_states
                res_a, res_b = residual
                x_a, res_a = self.input_layernorm(overlap.wait(ffn_a), res_a)
                if cfg.qkv:
                    qkv_a, _ = attn_mod.qkv_proj(x_a)
                    x_b, res_b = self.input_layernorm(overlap.wait(ffn_b), res_b)
                    qkv_b, _ = attn_mod.qkv_proj(x_b)
                    attn = self._attend(
                        positions, None, index_reuse, torch.cat([qkv_a, qkv_b])
                    )
                else:
                    x_b, res_b = self.input_layernorm(overlap.wait(ffn_b), res_b)
                    attn = self._attend(positions, torch.cat([x_a, x_b]), index_reuse)
            attn_a, attn_b = overlap.split(attn, half)
            o_a, _ = attn_mod.o_proj(attn_a)
            o_b, _ = attn_mod.o_proj(attn_b)
            o_a = overlap.all_reduce(o_a, route)
            o_b = overlap.all_reduce(o_b, route)
            ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
            x_a, res_a = self.post_attention_layernorm(overlap.wait(o_a), res_a)
            ffn_a = overlap.all_reduce(ffn(x_a), route)
            x_b, res_b = self.post_attention_layernorm(overlap.wait(o_b), res_b)
            with token_padding.row_offset(half):
                ffn_b = ffn(x_b)
            ffn_b = overlap.all_reduce(ffn_b, route)
        return (ffn_a, ffn_b), (res_a, res_b)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        index_reuse: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`index_reuse` (MSA index reuse on) is the step's shared selection
        dict (`MiniMaxM3SparseAttention.forward`)."""
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if index_reuse is None:
            hidden_states = self.self_attn(positions, hidden_states)
        else:
            hidden_states = self.self_attn(positions, hidden_states, index_reuse)
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
        # MSA index reuse (`index_reuse`): the layers share one dict per step;
        # only when some layer's op runs it (else the calls are unchanged).
        self.index_reuse = any(
            layer.self_attn.index_reuse_full is not None for layer in self.layers
        )
        # Two-batch overlap of large steps (`overlap`,
        # TPU_MINIMAX_M3_PREFILL_OVERLAP).
        self.tp_size = get_tensor_model_parallel_world_size()
        self.overlap = overlap.config_from_env() if self.tp_size > 1 else overlap.OFF

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
        reuse = {"index_reuse": {}} if getattr(self, "index_reuse", False) else {}
        ovl = getattr(self, "overlap", overlap.OFF)
        # The token count as the input_ids symbol (lifted first in every
        # trace); read only with the overlap on.
        num_tokens = (
            (input_ids if input_ids is not None else inputs_embeds).shape[0]
            if ovl.on
            else 0
        )
        if ovl.on and overlap.step(ovl, num_tokens, self.tp_size):
            # Halves of the embedding's own row count (the same symbol as
            # input_ids' when the embedding is computed here).
            hidden_states = self._forward_overlap(
                positions,
                inputs_embeds,
                inputs_embeds.shape[0] // 2,
                aux_hidden_states,
                reuse,
            )
        else:
            for idx, layer in enumerate(self.layers):
                hidden_states, residual = layer(
                    positions, hidden_states, residual, **reuse
                )
                if idx + 1 in self.aux_hidden_state_layers:
                    aux_hidden_states.append(hidden_states + residual)
            hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def _forward_overlap(
        self,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        half,
        aux_hidden_states: list[torch.Tensor],
        reuse: dict,
    ) -> torch.Tensor:
        """The layers on token halves (`overlap`); joined after the final
        norm (and for each aux hidden state)."""
        hidden, residual = inputs_embeds, None
        for idx, layer in enumerate(self.layers):
            hidden, residual = layer.forward_overlap(
                positions, hidden, residual, half, **reuse
            )
            if idx + 1 in self.aux_hidden_state_layers:
                aux_hidden_states.append(
                    torch.cat(
                        [overlap.wait(h) + r for h, r in zip(hidden, residual)]
                    )
                )
        out = [self.norm(overlap.wait(h), r)[0] for h, r in zip(hidden, residual)]
        return torch.cat(out)

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

    def compute_logits_shard(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This rank's vocab shard of ``compute_logits`` (pad columns
        included), without the all-gather."""
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )

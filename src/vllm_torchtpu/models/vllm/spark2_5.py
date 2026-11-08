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
"""Spark2_5ForCausalLM (model_type spark2_5, e.g. XHToken/Spark-X2.6-4B).

vLLM has no model class for this architecture. It is a dense decoder built from
stock vLLM layers:
- attention: one merged q|k|v projection (q_k_v_proj in the checkpoint), GQA,
  a per-layer sliding window (layer_types) and per-layer-type RoPE
  (rope_parameters[layer_type]: theta and partial_rotary_factor, neox style
  over the first rotary_dim lanes), then a headwise output gate:
  out_proj(attn * act(g_proj(x))) with one gate logit per query head;
- MLP: down(gelu(gate(x)) * up(x)) with the exact (erf) GELU;
- RMSNorm (x * w), tied input / output embeddings.
"""

from collections.abc import Iterable

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    extract_layer_index,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors


class Spark2_5MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        bias: bool,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "gelu":
            # the checkpoint's modeling code accepts only "gelu" (ACT2FN: erf GELU)
            raise ValueError(
                f"Spark2_5 supports hidden_act='gelu' only, got {hidden_act!r}"
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = GeluAndMul(approximate="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(gate_up))
        return x


class Spark2_5Attention(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0
        # tp > num_kv_heads replicates each KV head on tp / num_kv_heads ranks
        assert (
            self.total_num_kv_heads % tp_size == 0
            or tp_size % self.total_num_kv_heads == 0
        )
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        bias = getattr(config, "attention_bias", False)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.gate_act = None
        if getattr(config, "headwise_attn_output_gate", False):
            mode = getattr(config, "gate_attn_act_mode", "sigmoid")
            if mode not in ("sigmoid", "silu"):
                raise ValueError(f"Unsupported gate_attn_act_mode: {mode!r}")
            self.gate_act = torch.sigmoid if mode == "sigmoid" else nn.functional.silu
            # one gate logit per query head, column-parallel like the heads
            self.g_proj = ColumnParallelLinear(
                config.hidden_size,
                self.total_num_heads,
                bias=bias,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
        self.out_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        layer_type = config.layer_types[extract_layer_index(prefix)]
        sliding_window = (
            config.sliding_window if layer_type == "sliding_attention" else None
        )
        # theta and partial_rotary_factor per layer type; get_rope derives rotary_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters[layer_type],
            is_neox_style=True,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        if self.gate_act is not None:
            gate, _ = self.g_proj(hidden_states)
            # the gate is computed in fp32 and cast once, as the checkpoint's code does
            gate = self.gate_act(gate.float()).to(attn_output.dtype)
            attn_output = (
                attn_output.unflatten(-1, (self.num_heads, self.head_dim))
                * gate.unsqueeze(-1)
            ).flatten(-2)
        output, _ = self.out_proj(attn_output)
        return output


class Spark2_5DecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.self_attn = Spark2_5Attention(
            config,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Spark2_5MLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            bias=getattr(config, "mlp_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
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
        return self.mlp(hidden_states), residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Spark2_5Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        # checkpoint name: model.embedding (not embed_tokens)
        self.embedding = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.embedding",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Spark2_5DecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_input_ids(input_ids)
        )
        residual = None
        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Spark2_5ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_k_v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = Spark2_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embedding)
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        cfg = self.config
        # the checkpoint's merged q_k_v_proj rows are q | k | v (all heads)
        qkv_split = [
            cfg.num_attention_heads * cfg.head_dim,
            cfg.num_key_value_heads * cfg.head_dim,
            cfg.num_key_value_heads * cfg.head_dim,
        ]
        stacked = [
            (".gate_up_proj.", ".gate_proj.", 0),
            (".gate_up_proj.", ".up_proj.", 1),
        ]
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            if name.startswith("lm_head.") and cfg.tie_word_embeddings:
                continue
            if ".q_k_v_proj." in name:
                name = name.replace(".q_k_v_proj.", ".qkv_proj.")
                param = params[name]
                for shard_id, part in zip(
                    ("q", "k", "v"), weight.split(qkv_split, dim=0)
                ):
                    param.weight_loader(param, part, shard_id)
                loaded.add(name)
                continue
            for param_part, weight_part, shard_id in stacked:
                if weight_part in name:
                    name = name.replace(weight_part, param_part)
                    param = params[name]
                    param.weight_loader(param, weight, shard_id)
                    break
            else:
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
            loaded.add(name)
        return loaded

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
"""Ripple (``RicursiveForCausalLM``, model_type ``ricursive``) on TPU.

vLLM has no Ricursive model. It is a dense GQA transformer built from vLLM's
standard layers, with four differences from Llama:

- Value residual: the scaled token embedding ``v_res`` is fed to every layer,
  and ``v = v_proj(lerp(v_res, x, sigmoid(value_res_lambda)))`` with ``x`` the
  layer's normed input. ``v_proj`` reads another input than q / k, so it is
  its own projection; q, k and the output gate share one.
- Hybrid attention: ``sliding_attention`` layers use a 1024-key window
  (including the query) and neox RoPE; ``full_attention`` layers use no
  positional encoding. Both apply per-head q / k RMSNorm.
- Gated attention output ``o * sigmoid(fp32 g_proj(x))``, and an MLP that
  clamps gate (above only) and up (both sides) at ``ffn_clamp_threshold``.
- Gemma-3 style sandwich norms, but plain ``w * x`` RMSNorm.
"""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
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
from vllm.model_executor.models.utils import extract_layer_index, maybe_prefix


class RicursiveMLP(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.clamp = config.ffn_clamp_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        if self.clamp:
            # The checkpoint's MLP bounds gate from above only, up on both sides.
            gate = gate.clamp(max=self.clamp)
            up = up.clamp(min=-self.clamp, max=self.clamp)
        x, _ = self.down_proj(F.silu(gate) * up)
        return x


class RicursiveAttention(nn.Module):
    def __init__(
        self,
        config,
        max_position: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.head_dim = config.head_dim
        if config.num_attention_heads % tp_size or config.num_key_value_heads % tp_size:
            # k_proj / v_proj are column-sharded by KV head (no replication).
            raise ValueError(
                f"Ricursive needs num_attention_heads ({config.num_attention_heads}) "
                f"and num_key_value_heads ({config.num_key_value_heads}) divisible "
                f"by the tensor-parallel size ({tp_size})"
            )
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = config.num_key_value_heads // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        q_total = config.num_attention_heads * self.head_dim
        kv_total = config.num_key_value_heads * self.head_dim

        # q | k | g all read x; loaded from q_proj / k_proj / g_proj (shards 0-2).
        self.qkg_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [q_total, kv_total, q_total],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkg_proj",
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            kv_total,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.o_proj = RowParallelLinear(
            q_total,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.value_res_lambda = nn.Parameter(torch.zeros(1), requires_grad=False)

        layer_type = config.layer_types[extract_layer_index(prefix)]
        self.is_sliding = layer_type == "sliding_attention"
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                max_position=max_position,
                is_neox_style=True,
                rope_parameters={
                    "rope_type": "default",
                    "rope_theta": config.rope_theta,
                },
            )
            if self.is_sliding
            else None
        )
        # vLLM's window counts the query (keys k >= q - 1023), as the checkpoint.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=config.sliding_window if self.is_sliding else None,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self, positions: torch.Tensor, x: torch.Tensor, v_res: torch.Tensor
    ) -> torch.Tensor:
        qkg, _ = self.qkg_proj(x)
        q, k, g = qkg.split([self.q_size, self.kv_size, self.q_size], dim=-1)
        # The blend weight is rounded to the activation dtype first, as the
        # checkpoint's code does (it is 1.0 in bf16 on some layers); the lerp
        # itself runs in fp32 and is rounded once.
        blend = torch.sigmoid(self.value_res_lambda.float()).to(x.dtype)
        v_in = torch.lerp(v_res.float(), x.float(), blend.float()).to(x.dtype)
        v, _ = self.v_proj(v_in)
        q = self.q_norm(q.unflatten(-1, (self.num_heads, self.head_dim))).flatten(-2)
        k = self.k_norm(k.unflatten(-1, (self.num_kv_heads, self.head_dim))).flatten(-2)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        o = o * torch.sigmoid(g.float()).to(o.dtype)
        out, _ = self.o_proj(o)
        return out


class RicursiveDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        max_position: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = RicursiveAttention(
            config,
            max_position,
            cache_config,
            quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = RicursiveMLP(config, quant_config, prefix=f"{prefix}.mlp")
        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        v_res: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # h += post_attn_norm(attn(input_norm(h)));
        # h += post_ffn_norm(mlp(pre_ffn_norm(h))),
        # with each residual add fused into the next norm.
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, v_res)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return hidden_states, residual


@support_torch_compile
class RicursiveModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        if not getattr(config, "use_valres", True) or not getattr(
            config, "use_gated_attention", True
        ):
            raise NotImplementedError(
                "Ricursive without the value residual or the attention gate"
            )
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        # RoPE positions never exceed max_model_len; the config's 2M-entry
        # max_position_embeddings would only size a larger cos/sin table.
        max_position = min(
            config.max_position_embeddings, vllm_config.model_config.max_model_len
        )
        self.layers = nn.ModuleList(
            RicursiveDecoderLayer(
                config,
                max_position,
                vllm_config.cache_config,
                quant_config,
                prefix=f"{prefix}.layers.{i}",
            )
            for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # None: the checkpoint folded the scale into embed_tokens.weight.
        self.embed_scale = config.embed_scale

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        if self.embed_scale is None:
            return h
        # Scale in fp32 and round once, as the checkpoint's GPU code computes
        # bf16 * python float (a bf16 scalar would apply 71.5, not 71.554).
        return (h.float() * float(self.embed_scale)).to(h.dtype)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        # The scaled embedding is both h0 and every layer's value residual.
        hidden_states, residual, v_res = inputs_embeds, None, inputs_embeds
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, v_res)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class RicursiveForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkg_proj": ["q_proj", "k_proj", "g_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = RicursiveModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # (packed param, checkpoint name, shard id): q / k / g all read x and
        # load into one merged projection; v_proj reads the value residual.
        stacked = [
            (".qkg_proj.", ".q_proj.", 0),
            (".qkg_proj.", ".k_proj.", 1),
            (".qkg_proj.", ".g_proj.", 2),
            (".gate_up_proj.", ".gate_proj.", 0),
            (".gate_up_proj.", ".up_proj.", 1),
        ]
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            if name.startswith("lm_head.") and self.config.tie_word_embeddings:
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

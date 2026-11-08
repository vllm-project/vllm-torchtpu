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
"""GraniteSwitchForCausalLM (model_type granite_switch, e.g. granite-switch-4.1-8b).

A dense Granite 4.1 decoder with activated-LoRA adapters embedded in the
checkpoint, selected per token by a small attention "switch" (the
granite-switch 0.1.1 package; vLLM and transformers have no class for it):
- switch: a counting head (keys at the position-0 anchor and at control
  tokens, value 1 at the anchor, so a query reads 1 / (1 + n)) and a memory
  head (key codebook[n] * gain at control tokens, value = adapter id), each a
  1-head paged attention with its own KV cache; round / clamp gives each
  token's adapter index (0 = base), then control ids are replaced by their
  substitute ids before the embedding;
- decoder: fused qkv, neox RoPE, GQA with scale attention_multiplier, SwiGLU
  shared MLP; each of qkv_proj / o_proj / input_linear / output_linear adds
  x @ A[a - 1].T @ B[a - 1].T for tokens with adapter index a > 0 (B is
  pre-scaled at compose time), computed with the TPU multi-LoRA ops;
- Granite scalars: embedding_multiplier, residual_multiplier (separate
  residual add, then RMSNorm), logits_scaling; tied embeddings.
config.num_hidden_layers counts the decoder layers plus the switch's 2 cache
slots; the switch's attention layers take the last 2 layer indices.
"""

from collections.abc import Iterable

import torch
from torch import nn
from transformers import GraniteMoeHybridConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import make_layers, maybe_prefix
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.lora.lora_ops import bgmv_expand, bgmv_shrink, one_hot_slots

SWITCH_CACHE_LAYERS = 2  # counting + memory heads (MultiSwitch.num_cache_layers)
_NEG = -1e9  # finite mask: the one-hot queries multiply it by 0


class GraniteSwitchConfig(GraniteMoeHybridConfig):
    """granite_switch config: GraniteMoeHybrid fields plus the switch and adapter
    fields, which PretrainedConfig keeps as attributes."""

    model_type = "granite_switch"


def _check_supported(config) -> None:
    unsupported = {
        "num_local_experts": getattr(config, "num_local_experts", 0) not in (0, None),
        "fused_add_norm": bool(getattr(config, "fused_add_norm", False)),
        "dual_stream": bool(getattr(config, "dual_stream", False)),
        "position_embedding_type": config.position_embedding_type != "rope",
        "ms_code_type": getattr(config, "ms_code_type", "kerdock") != "kerdock",
        "hidden_act": config.hidden_act != "silu",
    }
    bad = [k for k, v in unsupported.items() if v]
    if bad or config.num_adapters < 1 or config.shared_intermediate_size < 1:
        raise ValueError(
            "GraniteSwitch supports the dense LoRA/aLoRA layout with a MultiSwitch "
            f"(Kerdock codes) only; unsupported config fields: {bad}"
        )


class SwitchedLoRALinear(nn.Module):
    """A vLLM linear plus its stacked per-token adapters.

    Checkpoint tensors: base_layer.weight, then lora_A / lora_B (one slice) or
    lora_{A,B}_slices.<i> (one per output slice), A [L, 1, r, in] and
    B [L, 1, out, r]. Column-parallel layers keep A whole and shard B's rows like
    the base output; row-parallel layers shard A's input like the base and keep
    B whole, adding the partial delta before the all-reduce.
    """

    def __init__(
        self,
        base: nn.Module,
        num_adapters: int,
        rank: int,
        in_size: int,
        out_shards: list[tuple[int, int, int]],
        row_parallel: bool,
    ) -> None:
        """out_shards: per output slice (full rows, local rows, local shard index)."""
        super().__init__()
        self.base_layer = base
        self.row_parallel = row_parallel
        tp, tp_rank = (
            get_tensor_model_parallel_world_size(),
            get_tensor_model_parallel_rank(),
        )
        self.tp_size = tp
        in_local = in_size // tp if row_parallel else in_size

        def param(shape, shard: tuple[int, int, int] | None = None):
            """shard = (dim, rows, index): this rank's rows of the checkpoint tensor."""
            p = nn.Parameter(torch.empty(shape), requires_grad=False)

            def loader(param, w):
                if shard is not None:
                    dim, rows, index = shard
                    w = w.narrow(dim, index * rows, rows)
                assert param.shape == w.shape, (param.shape, w.shape)
                param.data.copy_(w)

            set_weight_attrs(p, {"weight_loader": loader})
            return p

        a = [
            param(
                (num_adapters, 1, rank, in_local),
                (3, in_local, tp_rank) if row_parallel else None,
            )
            for _ in out_shards
        ]
        b = [
            param(
                (num_adapters, 1, local, rank),
                (2, local, index) if local != full else None,
            )
            for full, local, index in out_shards
        ]
        # parameter names match the checkpoint's
        if len(out_shards) == 1:
            self.lora_A, self.lora_B = a[0], b[0]
        else:
            self.lora_A_slices = nn.ParameterList(a)
            self.lora_B_slices = nn.ParameterList(b)

    def forward(self, x: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
        y, _ = self.base_layer(x)
        pairs = (
            zip(self.lora_A_slices, self.lora_B_slices)
            if hasattr(self, "lora_A_slices")
            else [(self.lora_A, self.lora_B)]
        )
        # adapter slot a - 1 for each token, scale 1: B is pre-scaled
        delta = [
            bgmv_expand(bgmv_shrink(x, a, onehot, 1.0), b, onehot) for a, b in pairs
        ]
        y = y + (delta[0] if len(delta) == 1 else torch.cat(delta, dim=-1))
        if self.row_parallel and self.tp_size > 1:
            y = tensor_model_parallel_all_reduce(y)
        return y


class GraniteSwitchAttention(nn.Module):
    def __init__(self, config, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        tp, tp_rank = (
            get_tensor_model_parallel_world_size(),
            get_tensor_model_parallel_rank(),
        )
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp == 0
        assert self.total_num_kv_heads % tp == 0 or tp % self.total_num_kv_heads == 0
        self.num_heads = self.total_num_heads // tp
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp)
        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # tp > num_kv_heads replicates each KV head on tp / num_kv_heads ranks
        kv_index = tp_rank // max(1, tp // self.total_num_kv_heads)
        lora = (config.num_adapters, config.max_lora_rank)
        quant_config = vllm_config.quant_config
        self.qkv_proj = SwitchedLoRALinear(
            QKVParallelLinear(
                config.hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=config.attention_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            ),
            *lora,
            config.hidden_size,
            [
                (self.total_num_heads * self.head_dim, self.q_size, tp_rank),
                (self.total_num_kv_heads * self.head_dim, self.kv_size, kv_index),
                (self.total_num_kv_heads * self.head_dim, self.kv_size, kv_index),
            ],
            row_parallel=False,
        )
        self.o_proj = SwitchedLoRALinear(
            RowParallelLinear(
                self.total_num_heads * self.head_dim,
                config.hidden_size,
                bias=config.attention_bias,
                reduce_results=False,  # the LoRA partial joins the all-reduce
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj",
            ),
            *lora,
            self.total_num_heads * self.head_dim,
            [(config.hidden_size, config.hidden_size, 0)],
            row_parallel=True,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=True,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            config.attention_multiplier,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self, positions: torch.Tensor, x: torch.Tensor, onehot: torch.Tensor
    ) -> torch.Tensor:
        qkv = self.qkv_proj(x, onehot)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        return self.o_proj(self.attn(q, k, v), onehot)


class GraniteSwitchMLP(nn.Module):
    def __init__(self, config, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        tp, tp_rank = (
            get_tensor_model_parallel_world_size(),
            get_tensor_model_parallel_rank(),
        )
        inter = config.shared_intermediate_size
        lora = (config.num_adapters, config.max_lora_rank)
        self.input_linear = SwitchedLoRALinear(
            MergedColumnParallelLinear(
                config.hidden_size,
                [inter] * 2,
                bias=False,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.input_linear",
            ),
            *lora,
            config.hidden_size,
            [(inter, inter // tp, tp_rank)] * 2,
            row_parallel=False,
        )
        self.output_linear = SwitchedLoRALinear(
            RowParallelLinear(
                inter,
                config.hidden_size,
                bias=False,
                reduce_results=False,  # the LoRA partial joins the all-reduce
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.output_linear",
            ),
            *lora,
            inter,
            [(config.hidden_size, config.hidden_size, 0)],
            row_parallel=True,
        )
        self.act_fn = SiluAndMul()  # gate | up halves of input_linear

    def forward(self, x: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
        return self.output_linear(self.act_fn(self.input_linear(x, onehot)), onehot)


class GraniteSwitchDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.residual_multiplier = config.residual_multiplier
        self.self_attn = GraniteSwitchAttention(
            config, vllm_config, prefix=f"{prefix}.self_attn"
        )
        self.shared_mlp = GraniteSwitchMLP(
            config, vllm_config, prefix=f"{prefix}.shared_mlp"
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
        onehot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # fused_add_norm False: the residual add rounds to the model dtype before
        # the norm, as the checkpoint's own vLLM model does
        residual = hidden_states if residual is None else residual + hidden_states
        h = self.self_attn(positions, self.input_layernorm(residual), onehot)
        residual = residual + h * self.residual_multiplier
        h = self.shared_mlp(self.post_attention_layernorm(residual), onehot)
        return h * self.residual_multiplier, residual


class GraniteSwitchRouter(nn.Module):
    """MultiSwitch: per-token adapter index and control-token exchange.

    Both heads are 1-head paged attentions over the request's tokens (vLLM
    confines each to its request), so a decode step reads the anchor and the
    control tokens from the cache. They use the decoder's head size: the
    counting head needs only lane 0 and the codes 2**ms_code_m lanes, and one
    head size gives all layers one KV-cache spec.
    """

    def __init__(self, config, vllm_config: VllmConfig, prefix: str, first_index: int):
        super().__init__()
        self.num_adapters = config.num_adapters
        self.ctrl_ids = [int(i) for i in config.adapter_token_ids]
        # a leading base-reset token (num_adapters + 1 ids) writes adapter 0
        self.offset = 0 if len(self.ctrl_ids) == config.num_adapters + 1 else 1
        self.code_dim = 2**config.ms_code_m
        self.head_size = config.hidden_size // config.num_attention_heads
        assert self.head_size >= max(self.code_dim, config.ms_counting_head_dim)
        self.memory_gain = config.ms_memory_gain
        self.dtype = vllm_config.model_config.dtype
        # Kerdock m: 2**(2m - 1) codewords of 2**m lanes; load_weights requires both
        # buffers from the checkpoint (a zero codebook would make every read uniform)
        self.register_buffer(
            "codebook",
            torch.zeros(
                2 ** (2 * config.ms_code_m - 1), self.code_dim, dtype=torch.float32
            ),
        )
        # -1 = not a control token; built from config as the checkpoint stores it
        lut = torch.full((config.vocab_size,), -1, dtype=torch.int64)
        lut[self.ctrl_ids] = torch.tensor(config.adapter_substitute_token_ids)
        self.register_buffer("control_to_substitute_lut", lut)
        self.counting_attn, self.memory_attn = (
            Attention(
                1,
                self.head_size,
                1.0,
                num_kv_heads=1,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                # one integer per layer name; after the decoder layers
                prefix=f"{prefix}.layers.{first_index + i}.attn",
            )
            for i in range(SWITCH_CACHE_LAYERS)
        )

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t, dev, dt = input_ids.shape[0], input_ids.device, self.dtype
        match = torch.stack([input_ids == c for c in self.ctrl_ids], dim=1)
        is_ctrl = match.any(dim=1)
        expert = torch.where(is_ctrl, match.long().argmax(dim=1) + self.offset, 0)
        zero, one = (
            torch.zeros((), dtype=dt, device=dev),
            torch.ones((), dtype=dt, device=dev),
        )
        # counting head: one-hot query on lane 0; key 0 at the anchor (position 0)
        # and control tokens, -1e9 elsewhere; value 1 at the anchor -> 1 / (1 + n)
        q = torch.zeros(t, self.head_size, dtype=dt, device=dev)
        q[:, 0] = one
        k = torch.zeros(t, self.head_size, dtype=dt, device=dev)
        k[:, 0] = torch.where((positions == 0) | is_ctrl, zero, _NEG)
        v = torch.zeros(t, self.head_size, dtype=dt, device=dev)
        v[:, 0] = torch.where(positions == 0, one, zero)
        signal = self.counting_attn(q, k, v).view(t, self.head_size)[:, 0]
        count = torch.round(1.0 / signal.float() - 1.0).long()  # fp32 inversion
        code = self.codebook[count.clamp(0, self.codebook.shape[0] - 1)].to(dt)
        # memory head: key code(n) * gain at control tokens, value = adapter id;
        # every token queries code(n) and reads the id written at its address
        q = torch.zeros(t, self.head_size, dtype=dt, device=dev)
        q[:, : self.code_dim] = code
        k = torch.zeros(t, self.head_size, dtype=dt, device=dev)
        k[:, : self.code_dim] = code * self.memory_gain * is_ctrl[:, None].to(dt)
        v = torch.zeros(t, self.head_size, dtype=dt, device=dev)
        v[:, 0] = (expert * is_ctrl).to(dt)
        mem = self.memory_attn(q, k, v).view(t, self.head_size)[:, 0]
        idx = torch.round(mem.float()).long().clamp(0, self.num_adapters)
        sub = self.control_to_substitute_lut[input_ids]
        return idx, torch.where(sub >= 0, sub, input_ids)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class GraniteSwitchModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        _check_supported(config)
        self.config = config
        num_layers = config.num_hidden_layers - SWITCH_CACHE_LAYERS
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.switch = GraniteSwitchRouter(
            config, vllm_config, f"{prefix}.switch", first_index=num_layers
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            num_layers,
            lambda prefix: GraniteSwitchDecoderLayer(vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        sub = self.switch.control_to_substitute_lut[input_ids]
        return self.embed_tokens(torch.where(sub >= 0, sub, input_ids))

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("GraniteSwitch routes adapters from input_ids")
        idx, ids = self.switch(input_ids, positions)
        onehot = one_hot_slots(idx - 1, self.config.num_adapters)  # 0 -> no slot
        h = self.embed_tokens(ids) * self.config.embedding_multiplier
        residual = None
        for layer in self.layers[self.start_layer : self.end_layer]:
            h, residual = layer(positions, h, residual, onehot)
        return self.norm(residual + h)


class GraniteSwitchForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = GraniteSwitchModel(
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
        self.logits_processor = LogitsProcessor(
            config.vocab_size, scale=1.0 / config.logits_scaling
        )

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
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        # fused on disk: qkv rows q | k | v, input_linear rows gate | up
        split = {
            ".qkv_proj.base_layer.": (
                ("q", "k", "v"),
                [
                    cfg.num_attention_heads * head_dim,
                    cfg.num_key_value_heads * head_dim,
                    cfg.num_key_value_heads * head_dim,
                ],
            ),
            ".input_linear.base_layer.": ((0, 1), [cfg.shared_intermediate_size] * 2),
        }
        params = dict(self.named_parameters())
        # the switch's 2 checkpoint buffers (its Attention layers own scale buffers too)
        buffers = {
            n: b
            for n, b in self.named_buffers()
            if n in ("model.switch.codebook", "model.switch.control_to_substitute_lut")
        }
        loaded: set[str] = set()
        for name, weight in weights:
            if name.startswith("lm_head.") and cfg.tie_word_embeddings:
                continue
            if name in buffers:  # switch codebook and control -> substitute LUT
                buffers[name].copy_(weight)
                loaded.add(name)
                continue
            param = params[name]
            part = next((k for k in split if k in name), None)
            if part is not None:
                for shard_id, w in zip(split[part][0], weight.split(split[part][1])):
                    param.weight_loader(param, w, shard_id)
            else:
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
            loaded.add(name)
        missing = set(buffers) - loaded
        if missing:
            raise ValueError(f"checkpoint lacks the switch tensors {sorted(missing)}")
        return loaded - set(buffers)

# SPDX-License-Identifier: Apache-2.0
"""Qwen3MoeForCausalLM for vllm-torchtpu built from vLLM / vllm-torchtpu layers.

Embedding, RMSNorms, GQA attention, sparse MoE and lm_head all use the vLLM / vllm-torchtpu layers
(paged KV attention and the fused MoE kernel)."""

from collections.abc import Iterable
from itertools import islice
import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, make_layers, maybe_prefix
from vllm.sequence import IntermediateTensors

BLOCK_VARIANTS = {"embed": "engine", "input_norm": "engine", "attn": "engine", "post_attn_norm": "engine", "moe": "engine", "final_norm": "engine", "lm_head": "engine"}

class TPUQwen3MoeAttention(nn.Module):
    """Attention: GQA with qkv packing, per-head q/k RMSNorm, RoPE and the
    engine's paged-KV `Attention`. Heads split over TP (KV heads replicated when fewer than TP ranks)."""

    def __init__(self, config, cache_config, quant_config, prefix: str = "") -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0
        # KV heads are partitioned when >= TP ranks, else replicated (tp_size % kv == 0).
        assert (self.total_num_kv_heads % tp_size == 0) if self.total_num_kv_heads >= tp_size \
            else (tp_size % self.total_num_kv_heads == 0)
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(config.hidden_size, self.head_dim, self.total_num_heads,
                                          self.total_num_kv_heads, bias=getattr(config, "attention_bias", False),
                                          quant_config=quant_config, prefix=f"{prefix}.qkv_proj")
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, config.hidden_size, bias=False,
                                        quant_config=quant_config, prefix=f"{prefix}.o_proj")
        self.rotary_emb = get_rope(self.head_dim, max_position=getattr(config, "max_position_embeddings", 8192),
                                   rope_parameters=config.rope_parameters)
        self.attn = Attention(self.num_heads, self.head_dim, self.head_dim**-0.5, num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config, quant_config=quant_config, prefix=f"{prefix}.attn")
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(*q.shape[:-1], -1, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(*k.shape[:-1], -1, self.head_dim)).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        out, _ = self.o_proj(self.attn(q, k, v))
        return out

class TPUQwen3MoeSparseMoe(nn.Module):
    """Sparse MoE: replicated router (`GateLinear`) + `FusedMoEFactory` routed
    experts (softmax top-k, renormalise=config.norm_topk_prob); EP / TP split and the TPU kernel are the engine's."""

    def __init__(self, vllm_config, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        if get_tensor_model_parallel_world_size() > config.num_experts:
            raise ValueError("tensor parallel size exceeds the number of experts")
        if parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError("TPUQwen3MoeSparseMoe: sequence-parallel MoE is not supported")
        self.gate = GateLinear(config.hidden_size, config.num_experts, quant_config=vllm_config.quant_config,
                               prefix=f"{prefix}.gate")
        self.experts = FusedMoEFactory(
            gate=self.gate, num_experts=config.num_experts, top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size, intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob, quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.experts", enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=parallel_config.eplb_config.num_redundant_experts,
            is_sequence_parallel=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The runner applies `gate` itself (router_logits arg carries the hidden states).
        return self.experts(hidden_states=hidden_states, router_logits=hidden_states)


class TPUQwen3MoeDecoderLayer(nn.Module):
    """One `model.layers.{L}` layer: input_norm -> attn -> post_attn_norm -> moe."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        cache_config, quant_config = vllm_config.cache_config, vllm_config.quant_config
        if getattr(config, "mlp_only_layers", []) or getattr(config, "decoder_sparse_step", 1) != 1:
            raise NotImplementedError("TPUQwen3MoeDecoderLayer: every layer must be a MoE layer")
        if getattr(config, "dual_chunk_attention_config", None):
            raise NotImplementedError("TPUQwen3MoeAttention: dual-chunk attention is not supported")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = TPUQwen3MoeAttention(config, cache_config, quant_config, prefix=f"{prefix}.self_attn")
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = TPUQwen3MoeSparseMoe(vllm_config, prefix=f"{prefix}.mlp")

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor,
                residual: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        # input RMSNorm (fused with the residual add)
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # self-attention
        hidden_states = self.self_attn(positions, hidden_states)
        # post-attention RMSNorm
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # sparse MoE
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class TPUQwen3MoeModel(nn.Module):
    """Decoder: embed -> (norm, attention, norm, MoE) x num_hidden_layers -> final_norm."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, quant_config=quant_config, prefix=f"{prefix}.embed_tokens")
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: TPUQwen3MoeDecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers")
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        residual = None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)
        # the final norm also folds in the last residual add
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class TPUQwen3MoeForCausalLM(nn.Module):
    """Top-level `Qwen3MoeForCausalLM` (TP / EP only: no pipeline-parallel, LoRA, EAGLE or EPLB hooks)."""

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_stacked={'.q_proj': ('.qkv_proj', 'q'), '.k_proj': ('.qkv_proj', 'k'), '.v_proj': ('.qkv_proj', 'v')})
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    embedding_modules = {"embed_tokens": "input_embeddings", "lm_head": "output_embeddings"}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config, self.quant_config = config, quant_config
        self.model = TPUQwen3MoeModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=quant_config, prefix=maybe_prefix(prefix, 'lm_head'))
        if config.tie_word_embeddings:
            raise NotImplementedError("TPUQwen3MoeForCausalLM: tied word embeddings are not supported")
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Expert tensors are routed to the MoE module's own load_weights by AutoWeightsLoader.
        return AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)

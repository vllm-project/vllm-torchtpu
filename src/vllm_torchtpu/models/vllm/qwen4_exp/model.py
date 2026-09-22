# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-native text backbone with explicit TPU state/input ownership."""
import copy
from collections.abc import Iterable

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.v1.kv_cache_interface import MambaSpec

from .gdn import Qwen4ExpGatedDeltaNetAttention, gdn_state_shapes
from .hyperconnection import GatedResidual, HyperConnectionConfig
from .moe import Qwen4ExpSparseMoeBlock
from .ple import Qwen4ExpPLE
from .qsa import QSAMetadata, Qwen4ExpQSA


def hc_config(config):
    return HyperConnectionConfig(hc_count=config.hc_count,
                                 hidden_size=config.hidden_size,
                                 params_dtype=torch.bfloat16,
                                 hc_lowrank=config.hc_lowrank,
                                 rms_norm_eps=config.rms_norm_eps,
                                 hc_per_branch_norm=True)


class Qwen4ExpDecoderLayer(nn.Module):

    def __init__(self, cfg, index, prefix):
        super().__init__()
        t = cfg.model_config.hf_text_config
        rank = get_tensor_model_parallel_rank()
        self.ple = (Qwen4ExpPLE(
            tp_rank=rank, prefix=prefix +
            ".ple", eos_token_id=t.eos_token_id) if index == 1 else None)
        self.layer_type = t.layer_types[index]
        if self.layer_type == "linear_attention":
            dense_cfg = copy.copy(cfg)
            dense_cfg.quant_config = None
            self.linear_attn = Qwen4ExpGatedDeltaNetAttention(
                t,
                vllm_config=dense_cfg,
                prefix=prefix + ".linear_attn",
                gqa_interleaved_layout=False)
        else:
            self.self_attn = Qwen4ExpQSA(tp_rank=rank,
                                         device=cfg.device_config.device,
                                         prefix=prefix + ".self_attn")
            cfg.compilation_config.static_forward_context[
                prefix + ".self_attn"] = self.self_attn
        self.mlp = Qwen4ExpSparseMoeBlock(ep_rank=rank)
        self.attn_hyper_connection = GatedResidual(hc_config(t))
        self.mlp_hyper_connection = GatedResidual(hc_config(t))

    def forward(self, hidden, positions, input_ids, query_start_loc,
                ngram_context, state_slots, context_lens, valid_tokens,
                main_block_table, compressed_block_table):
        if self.ple is not None:
            hidden = hidden + self.ple(hidden, input_ids, query_start_loc,
                                       ngram_context, state_slots, context_lens
                                       > 0)
        block, residual = self.attn_hyper_connection.mix(hidden)
        if self.layer_type == "linear_attention":
            block = self.linear_attn(block)
        else:
            block = self.self_attn(
                block, positions,
                QSAMetadata(query_start_loc, context_lens, state_slots,
                            main_block_table, compressed_block_table))
        hidden = self.attn_hyper_connection.combine(block, residual)
        block, residual = self.mlp_hyper_connection.mix(hidden)
        return self.mlp_hyper_connection.combine(self.mlp(block, valid_tokens),
                                                 residual)


@support_torch_compile(dynamic_arg_dims={
    "input_ids": 0,
    "positions": 0,
    "inputs_embeds": 0,
    "valid_tokens": 0
})
class Qwen4ExpBackbone(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.prefix = "model"
        t = cfg.model_config.hf_text_config
        self.embed_tokens = VocabParallelEmbedding(t.vocab_size, t.hidden_size)
        self.layers = nn.ModuleList([
            Qwen4ExpDecoderLayer(cfg, i, f"model.layers.{i}")
            for i in range(t.num_hidden_layers)
        ])
        self.hyper_connection_mixer = GatedResidual(hc_config(t),
                                                    use_combine=False)

    def forward(self,
                input_ids,
                positions,
                inputs_embeds=None,
                *,
                query_start_loc,
                ngram_context,
                state_slots,
                context_lens,
                valid_tokens,
                main_block_table,
                compressed_block_table):
        hidden = self.embed_tokens(
            input_ids) if inputs_embeds is None else inputs_embeds
        hidden = hidden.repeat(1, 4)
        for layer in self.layers:
            hidden = layer(hidden, positions.to(torch.int64),
                           input_ids.to(torch.int64), query_start_loc,
                           ngram_context, state_slots, context_lens,
                           valid_tokens, main_block_table,
                           compressed_block_table)
        return self.hyper_connection_mixer.mix(hidden)[0]


class Qwen4ExpForCausalLM(nn.Module, HasInnerState, IsHybrid):
    requires_raw_input_tokens = True
    packed_modules_mapping = {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"]
    }

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.config = vllm_config.model_config.hf_text_config
        self.model = Qwen4ExpBackbone(vllm_config)
        self.lm_head = ParallelLMHead(self.config.vocab_size,
                                      self.config.hidden_size,
                                      prefix="lm_head")
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def forward(self, input_ids, positions, inputs_embeds=None, **kwargs):
        return self.model(input_ids, positions, inputs_embeds, **kwargs)

    def embed_input_ids(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, cfg):
        return torch.float32, torch.float32

    @classmethod
    def get_mamba_state_shape_from_config(cls, cfg):
        return gdn_state_shapes(cfg.parallel_config.tensor_parallel_size)

    @classmethod
    def get_ple_mamba_state_dtype_from_config(cls, cfg):
        return (torch.bfloat16, )

    @classmethod
    def get_ple_mamba_state_shape_from_config(cls, cfg):
        return ((9, 10240), )

    @classmethod
    def get_mamba_specs_from_config(cls, cfg):
        return (MambaSpec(block_size=-1,
                          shapes=cls.get_mamba_state_shape_from_config(cfg),
                          dtypes=cls.get_mamba_state_dtype_from_config(cfg)),
                MambaSpec(block_size=-1,
                          shapes=((9, 10240), ),
                          dtypes=(torch.bfloat16, ),
                          tp_replicated=True))

    @classmethod
    def get_mamba_state_copy_func(cls):
        raise NotImplementedError(
            "Qwen4Exp S1 does not support prefix-cache state copies")

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        parameters = dict(self.named_parameters())
        loaded, source = set(), set()
        finalized = set()
        # Exactly the Qwen3.5 checkpoint projection mapping. HC remains
        # separate checkpoint-native parameters (no NVIDIA packed-HC mapper).
        gdn_map = {
            "in_proj_qkv.weight": ("in_proj_qkvz.weight", (0, 1, 2)),
            "in_proj_z.weight": ("in_proj_qkvz.weight", 3),
            "in_proj_b.weight": ("in_proj_ba.weight", 0),
            "in_proj_a.weight": ("in_proj_ba.weight", 1)
        }
        for original, weight in weights:
            if original.startswith(("model.visual.", "mtp.")):
                continue
            name = original.replace("model.language_model.", "model.", 1)
            if name in source:
                raise ValueError(f"Duplicate checkpoint tensor {original}")
            source.add(name)
            if name.startswith("model.layers."):
                _, _, idx, family, relative = name.split(".", 4)
                layer = self.model.layers[int(idx)]
                if family in ("mlp", "ple", "self_attn"):
                    module = getattr(layer, family)
                    module.load_weights([(relative, weight)])
                    if family == "mlp" and len(
                            module._loaded) == 197 and not module._pending:
                        module.finalize_weights()
                        finalized.add(int(idx))
                    continue
                if family == "linear_attn" and relative in gdn_map:
                    mapped, shard = gdn_map[relative]
                    target = f"model.layers.{idx}.linear_attn.{mapped}"
                    param = parameters[target]
                    param.weight_loader(param, weight, shard)
                    loaded.add(target)
                    continue
            if name not in parameters:
                raise ValueError(
                    f"Unknown Qwen4Exp checkpoint tensor {original}")
            param = parameters[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, weight)
            loaded.add(name)
        for i, layer in enumerate(self.model.layers):
            prefix = f"model.layers.{i}.mlp."
            required = {
                f"{prefix}experts.{e}.{p}.{f}"
                for e in range(512)
                for p in ("gate_proj", "up_proj", "down_proj")
                for f in ("weight", "weight_scale_inv")
            }
            required.update(prefix + n
                            for n in ("gate.weight",
                                      "shared_expert_gate.weight",
                                      "shared_expert.gate_proj.weight",
                                      "shared_expert.up_proj.weight",
                                      "shared_expert.down_proj.weight"))
            if required - source:
                raise ValueError(
                    f"Incomplete layer{i} expert checkpoint: {len(required-source)} missing"
                )
            if i not in finalized:
                layer.mlp.finalize_weights()
            if layer.ple is not None:
                layer.ple.finalize_weights()
            if layer.layer_type == "full_attention":
                layer.self_attn.finalize_weights()
            else:
                base = f"model.layers.{i}.linear_attn."
                if {base + n for n in gdn_map} - source:
                    raise ValueError(
                        f"Incomplete layer{i} GDN projection checkpoint")
        ordinary = {
            n
            for n in parameters
            if not any(marker in n
                       for marker in (".mlp.", ".ple.", ".self_attn."))
        }
        if ordinary - loaded:
            raise ValueError(
                f"Incomplete Qwen4Exp checkpoint: {sorted(ordinary-loaded)}")
        return set(dict(self.named_parameters()))

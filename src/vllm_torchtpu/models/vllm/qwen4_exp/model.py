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
"""Checkpoint-native text backbone with explicit TPU state/input ownership."""

from collections.abc import Iterable, Sequence
from typing import cast

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncsByType
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
)
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_torchtpu import envs
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

from .config import gdn_cache_spec, require_unified_pool
from .gdn import Qwen4ExpGatedDeltaNetAttention
from .hyperconnection import GatedResidual, HyperConnectionConfig
from .parallel import PLEParallel
from .pd_identity import LoadedWeightIdentity
from .ple import PLEInputs
from .ple_geometry import ple_state_shape
from .ple_host import shared_ple_options
from .pooled_ple import Qwen4ExpPooledPLE
from .qsa import QSAMetadata, Qwen4ExpQSA


def _parameter_storage_key(parameter):
    return (
        parameter.untyped_storage()._cdata,
        parameter.storage_offset(),
        parameter.shape,
        parameter.stride(),
    )


def hc_config(config):
    return HyperConnectionConfig(
        hc_count=config.hc_count,
        hidden_size=config.hidden_size,
        params_dtype=torch.bfloat16,
        hc_lowrank=config.hc_lowrank,
        rms_norm_eps=config.rms_norm_eps,
        hc_per_branch_norm=True,
    )


class Qwen4ExpDecoderLayer(nn.Module):
    def __init__(self, cfg, index, prefix):
        super().__init__()
        require_unified_pool()
        t = cfg.model_config.hf_text_config
        tp = get_tp_group()
        engram = cfg.additional_config.get("qwen4_exp", {}).get("engram_config", {})
        shared = (
            shared_ple_options(
                cfg.model_config, prefix + ".ple", tp.rank_in_group, tp.world_size
            )
            if index + 1 in t.ple_layer_ids and engram.get("dp_shared_memory", False)
            else None
        )
        self.ple = (
            Qwen4ExpPooledPLE(
                config=t,
                ple_layer_index=sorted(set(t.ple_layer_ids)).index(index + 1),
                quant_config=cfg.quant_config,
                table_group=get_ep_group()
                if engram.get("embedding_across_dp", False)
                else tp,
                parallel=PLEParallel(
                    tp,
                    get_dp_group(),
                    embedding_across_dp=engram.get("embedding_across_dp", False),
                ),
                prefix=prefix + ".ple",
                cpu_offload=engram.get("cpu_offload", False),
                shared=shared,
            )
            if index + 1 in t.ple_layer_ids
            else None
        )
        if self.ple is not None and engram.get("cpu_offload", False):
            cfg.compilation_config.static_forward_context[prefix + ".ple"] = self.ple
        self.layer_type = t.layer_types[index]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen4ExpGatedDeltaNetAttention(
                t,
                vllm_config=cfg,
                prefix=prefix + ".linear_attn",
                gqa_interleaved_layout=False,
            )
        else:
            self.self_attn = Qwen4ExpQSA(
                config=t,
                max_model_len=cfg.model_config.max_model_len,
                tp_rank=tp.rank_in_group,
                tp_size=tp.world_size,
                device=cfg.device_config.device,
                cache_dtype=cfg.cache_config.cache_dtype,
                prefix=prefix + ".self_attn",
            )
            cfg.compilation_config.static_forward_context[prefix + ".self_attn"] = (
                self.self_attn
            )
        self.mlp = Qwen3NextSparseMoeBlock(vllm_config=cfg, prefix=prefix + ".mlp")
        self.attn_hyper_connection = GatedResidual(hc_config(t))
        self.mlp_hyper_connection = GatedResidual(hc_config(t))

    def forward(
        self,
        hidden,
        positions,
        input_ids,
        query_start_loc,
        ngram_context,
        state_slots,
        context_lens,
        valid_tokens,
        main_block_table,
        compressed_block_table,
        ple_raw_embeddings=None,
    ):
        if self.ple is not None:
            per_layer = cast(
                dict[str, AttentionMetadata], get_forward_context().attn_metadata
            )
            ple_slots = per_layer[self.ple.prefix].mamba_state_indices
            assert isinstance(ple_slots, torch.Tensor)
            hidden = hidden + self.ple.forward_with_history(
                hidden,
                PLEInputs(input_ids, query_start_loc, ngram_context),
                ple_slots,
                context_lens > 0,
                query_start_loc[-1:],
                ple_raw_embeddings,
            )
        block, residual = self.attn_hyper_connection.mix(hidden)
        if self.layer_type == "linear_attention":
            block = self.linear_attn(block)
        else:
            block = self.self_attn(
                block,
                positions,
                QSAMetadata(
                    query_start_loc,
                    context_lens,
                    state_slots,
                    main_block_table,
                    compressed_block_table,
                ),
            )
        hidden = self.attn_hyper_connection.combine(block, residual)
        block, residual = self.mlp_hyper_connection.mix(hidden)
        # GDN may leave padded rows unwritten; MoE dispatch requires finite inputs.
        block = torch.where(valid_tokens[:, None], block, 0)
        return self.mlp_hyper_connection.combine(self.mlp(block), residual)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
        "valid_tokens": 0,
    }
)
class Qwen4ExpBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.prefix = "model"
        t = cfg.model_config.hf_text_config
        self.hc_count = t.hc_count
        self.embed_tokens = VocabParallelEmbedding(t.vocab_size, t.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen4ExpDecoderLayer(cfg, i, f"model.layers.{i}")
                for i in range(t.num_hidden_layers)
            ]
        )
        self.host_ple_index = next(
            (
                i
                for i, layer in enumerate(
                    cast(Sequence[Qwen4ExpDecoderLayer], self.layers)
                )
                if layer.ple is not None and layer.ple.cpu_offload
            ),
            None,
        )
        # All QSA layers in this model use the same position/rotation config.
        # Share the immutable device buffer so XLA can reuse position gathers;
        # request-owned KV, ring and tags remain separate for every layer.
        attentions = [
            layer.self_attn
            for layer in self.layers
            if layer.layer_type == "indexed_attention"
        ]
        for attention in attentions[1:]:
            attention.rope_cache = attentions[0].rope_cache
        self.hyper_connection_mixer = GatedResidual(hc_config(t), use_combine=False)

    def forward(
        self,
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
        compressed_block_table,
    ):
        # Resolve host lookup before any decoder work so shape arithmetic
        # stays in the main graph and model preparation precedes the D2H wait.
        ple_raw_embeddings = None
        if self.host_ple_index is not None:
            ple = cast(Qwen4ExpDecoderLayer, self.layers[self.host_ple_index]).ple
            assert ple is not None
            ple_raw_embeddings = ple.lookup_for_model(
                PLEInputs(input_ids, query_start_loc, ngram_context),
                query_start_loc[-1:],
            )
        hidden = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        hidden = hidden.repeat(1, self.hc_count)
        for layer in self.layers:
            hidden = layer(
                hidden,
                positions.to(torch.int64),
                input_ids.to(torch.int64),
                query_start_loc,
                ngram_context,
                state_slots,
                context_lens,
                valid_tokens,
                main_block_table,
                compressed_block_table,
                ple_raw_embeddings,
            )
        return self.hyper_connection_mixer.mix(hidden)[0]


class Qwen4ExpForCausalLM(nn.Module, HasInnerState, IsHybrid):
    requires_raw_input_tokens = True
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model."}
    )
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self._pd_weight_identity_enabled = vllm_config.kv_transfer_config is not None
        self.pd_loaded_weight_identity: str | None = None
        self.pd_loaded_ple_identity: str | None = None
        self.pd_gdn_qk_pair_layout = envs.TPU_GDN_CONV_QK_PAIR_LAYOUT
        self.config = vllm_config.model_config.hf_text_config
        self.model = Qwen4ExpBackbone(vllm_config)
        self.lm_head = ParallelLMHead(
            self.config.vocab_size, self.config.hidden_size, prefix="lm_head"
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def forward(self, input_ids, positions, inputs_embeds=None, **kwargs):
        return self.model(input_ids, positions, inputs_embeds, **kwargs)

    def embed_input_ids(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, cfg):
        return gdn_cache_spec(cfg).dtypes

    @classmethod
    # TPU convolution state has an extra singleton axis absent from IsHybrid.
    # pyrefly: ignore[bad-override]
    def get_mamba_state_shape_from_config(
        cls, vllm_config
    ) -> tuple[tuple[int, ...], ...]:
        return gdn_cache_spec(vllm_config).shapes

    @classmethod
    def get_ple_mamba_state_dtype_from_config(cls, cfg):
        return (torch.bfloat16,)

    @classmethod
    def get_ple_mamba_state_shape_from_config(cls, cfg):
        return (ple_state_shape(cfg.model_config.hf_text_config),)

    @classmethod
    def get_mamba_specs_from_config(cls, cfg):
        return (
            MambaSpec(
                block_size=-1,
                shapes=cls.get_mamba_state_shape_from_config(cfg),
                dtypes=cls.get_mamba_state_dtype_from_config(cfg),
            ),
            MambaSpec(
                block_size=-1,
                shapes=cls.get_ple_mamba_state_shape_from_config(cfg),
                dtypes=(torch.bfloat16,),
                tp_replicated=True,
            ),
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateCopyFuncCalculator,
        )

        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(cls, mamba_types) -> MambaStateCopyFuncsByType:
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateCopyFuncCalculator,
        )
        from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

        functions: MambaStateCopyFuncsByType = {
            MambaAttentionBackendEnum.GDN_ATTN: cls.get_mamba_state_copy_func(),
            MambaAttentionBackendEnum.SHORT_CONV: (
                MambaStateCopyFuncCalculator.short_conv_state_copy_func()
            ),
        }
        return {kind: functions[kind] for kind in mamba_types}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self.pd_loaded_weight_identity = None
        self.pd_loaded_ple_identity = None
        identity = LoadedWeightIdentity() if self._pd_weight_identity_enabled else None
        if identity is not None:
            weights = identity.observe(weights)
        parameters = dict(self.named_parameters())
        loaded, source = set(), set()
        moe_mapper = WeightsMapper(
            orig_to_new_stacked={
                "shared_expert.gate_proj": ("shared_expert.gate_up_proj", 0),
                "shared_expert.up_proj": ("shared_expert.gate_up_proj", 1),
            }
        )
        layers = cast(Sequence[Qwen4ExpDecoderLayer], self.model.layers)
        moe_loaders = [AutoWeightsLoader(layer.mlp) for layer in layers]
        # Exactly the Qwen3.5 checkpoint projection mapping. HC remains
        # separate checkpoint-native parameters (no NVIDIA packed-HC mapper).
        gdn_map = {
            "in_proj_qkv.weight": ("in_proj_qkvz.weight", (0, 1, 2)),
            "in_proj_z.weight": ("in_proj_qkvz.weight", 3),
            "in_proj_b.weight": ("in_proj_ba.weight", 0),
            "in_proj_a.weight": ("in_proj_ba.weight", 1),
        }
        for original, weight in weights:
            if original.startswith(("model.visual.", "mtp.")):
                continue
            name = next(iter(self.hf_to_vllm_mapper.apply([(original, weight)])))[0]
            if name in source:
                raise ValueError(f"Duplicate checkpoint tensor {original}")
            source.add(name)
            if name.startswith("model.layers."):
                _, _, idx, family, relative = name.split(".", 4)
                layer = layers[int(idx)]
                if family == "mlp":
                    targets = moe_loaders[int(idx)].load_weights(
                        iter([(relative, weight)]), mapper=moe_mapper
                    )
                    loaded.update(
                        f"model.layers.{idx}.mlp.{target}" for target in targets
                    )
                    continue
                if family == "ple":
                    assert layer.ple is not None
                    layer.ple.load_checkpoint([(relative, weight)])
                    continue
                if family == "self_attn":
                    layer.self_attn.load_weights([(relative, weight)])
                    continue
                if family == "linear_attn" and relative in gdn_map:
                    mapped, shard = gdn_map[relative]
                    target = f"model.layers.{idx}.linear_attn.{mapped}"
                    param = parameters[target]
                    cast(BasevLLMParameter, param).weight_loader(param, weight, shard)
                    loaded.add(target)
                    continue
            if name not in parameters:
                raise ValueError(f"Unknown Qwen4Exp checkpoint tensor {original}")
            param = parameters[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, weight)
            loaded.add(name)
        for i, layer in enumerate(layers):
            prefix = f"model.layers.{i}.mlp."
            experts = layer.mlp.experts.routed_experts
            required = {
                prefix + weight_name + field
                for _, weight_name, expert_id, _ in experts.get_expert_mapping()
                if experts._map_global_expert_id_to_local_expert_id(expert_id) >= 0
                for field in ("weight", "weight_scale_inv")
            }
            required.update(
                prefix + name
                for name in (
                    "gate.weight",
                    "shared_expert_gate.weight",
                    "shared_expert.gate_proj.weight",
                    "shared_expert.up_proj.weight",
                    "shared_expert.down_proj.weight",
                )
            )
            if required - source:
                raise ValueError(
                    f"Incomplete layer{i} expert checkpoint: "
                    f"{len(required - source)} missing"
                )
            for checkpoint_projection, (
                projection,
                _,
            ) in moe_mapper.orig_to_new_stacked.items():
                required_shards = {}
                for field, parameter in layer.mlp.get_submodule(
                    projection
                ).named_parameters(recurse=False):
                    aliases = required_shards.setdefault(
                        _parameter_storage_key(parameter), set()
                    )
                    aliases.add(prefix + checkpoint_projection + "." + field)
                missing_shards = [
                    sorted(aliases)
                    for aliases in required_shards.values()
                    if aliases.isdisjoint(source)
                ]
                if missing_shards:
                    raise ValueError(
                        f"Incomplete layer{i} shared checkpoint: {missing_shards}"
                    )
            if layer.ple is not None:
                layer.ple.finalize_weights()
            if layer.layer_type == "indexed_attention":
                layer.self_attn.finalize_weights()
            else:
                base = f"model.layers.{i}.linear_attn."
                if {base + n for n in gdn_map} - source:
                    raise ValueError(f"Incomplete layer{i} GDN projection checkpoint")
        ordinary = {
            n
            for n in parameters
            if not any(
                marker in n for marker in (".mlp.experts.", ".ple.", ".self_attn.")
            )
        }
        # Quantization methods may register names for the same scale storage.
        loaded_storage = {
            _parameter_storage_key(parameters[name])
            for name in loaded & parameters.keys()
        }
        missing = {
            name
            for name in ordinary - loaded
            if _parameter_storage_key(parameters[name]) not in loaded_storage
        }
        if missing:
            raise ValueError(f"Incomplete Qwen4Exp checkpoint: {sorted(missing)}")
        if identity is not None:
            self.pd_loaded_weight_identity = identity.finish()
            self.pd_loaded_ple_identity = identity.finish(name_fragment=".ple.")
        return set(dict(self.named_parameters()))

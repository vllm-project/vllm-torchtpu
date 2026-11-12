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

import dataclasses
from collections.abc import Iterable
from typing import NamedTuple

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context
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

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed import pallas_all_reduce
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.minimax_m3.attention import MiniMaxM3SparseAttention
from vllm_torchtpu.models.minimax_m3.moe import MiniMaxM3MLP, MiniMaxM3MoE
from vllm_torchtpu.models.minimax_m3.quant import CT_TO_MODELOPT_EXPERT_SUFFIX
from vllm_torchtpu.models.minimax_m3.split import join_seq_lens, split_query_layout

logger = init_logger(__name__)

# The residual stream between layers: one tensor, or in a split step (see
# MiniMaxM3Model.forward) its two token halves.
Stream = torch.Tensor | tuple[torch.Tensor, ...]


# Rows of the earlier output an ordered all-reduce waits on: any slice of it
# carries the dependency.
_AFTER_ROWS = 8


def _all_reduce(x: torch.Tensor, after: torch.Tensor | None = None) -> torch.Tensor:
    """TP all-reduce of x; with `after` (an earlier all-reduce's output) it
    starts once `after` exists and is never combined with it."""
    if after is not None:
        out = pallas_all_reduce.all_reduce(
            get_tp_group().ranks, x, after.narrow(0, 0, _AFTER_ROWS)
        )
        if out is not None:
            return out
    return tensor_model_parallel_all_reduce(x)


def _after(x: torch.Tensor, dep: torch.Tensor) -> torch.Tensor:
    """x, but computed only once `dep` exists (adds an exact zero from it)."""
    zero = torch.nan_to_num(dep[:1, :1], nan=0.0, posinf=0.0, neginf=0.0) * 0
    return x + zero.to(x.dtype)


def _stream_sum(hidden_states: Stream, residual: Stream) -> torch.Tensor:
    if isinstance(hidden_states, tuple):
        return torch.cat([h + r for h, r in zip(hidden_states, residual)])
    return hidden_states + residual


class MiniMaxM3DecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, layer_id: int, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        # With the split on, forward does every residual all-reduce itself.
        self.split_min_tokens = envs.TPU_MINIMAX_M3_SPLIT_MIN_TOKENS
        reduce_results = not self.split_min_tokens
        self.self_attn = MiniMaxM3SparseAttention(
            config,
            quant_config,
            vllm_config.cache_config,
            prefix=f"{prefix}.self_attn",
            reduce_results=reduce_results,
        )
        # Dense layers name their FFN `mlp`, MoE layers `block_sparse_moe`.
        self.is_moe_layer = bool(config.moe_layer_freq[layer_id])
        if self.is_moe_layer:
            self.block_sparse_moe = MiniMaxM3MoE(
                config,
                quant_config,
                prefix=f"{prefix}.block_sparse_moe",
                reduce_results=reduce_results,
            )
            # Upstream honours reduce_results=False only on its late
            # all-reduce path; anywhere else the FFN output would be reduced
            # twice.
            moe_config = getattr(self.block_sparse_moe.experts, "moe_config", None)
            if not reduce_results and not getattr(
                moe_config, "skip_final_all_reduce", False
            ):
                raise ValueError(
                    "TPU_MINIMAX_M3_SPLIT_MIN_TOKENS needs a MoE that leaves its "
                    "all-reduce to the model (no all2all, sequence-parallel or "
                    "zero-expert MoE)."
                )
        else:
            self.mlp = MiniMaxM3MLP(
                config,
                config.dense_intermediate_size,
                quant_config,
                prefix=f"{prefix}.mlp",
                reduce_results=reduce_results,
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
        attn_metadata: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        attn = self.self_attn(positions, hidden_states, attn_metadata)
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        if not self.split_min_tokens:
            hidden_states, residual = self.post_attention_layernorm(attn, residual)
            return ffn(hidden_states), residual
        hidden_states, residual = self.post_attention_layernorm(
            _all_reduce(attn), residual
        )
        return _all_reduce(ffn(hidden_states)), residual

    def forward_split(
        self,
        halves: tuple["_Half", "_Half"],
        hidden_states: tuple[torch.Tensor, torch.Tensor],
        residual: tuple[torch.Tensor, torch.Tensor] | None,
        last: torch.Tensor | None,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor]:
        """The layer on the step's two token halves (see MiniMaxM3Model.forward).

        Each half's chain (norm, attention, all-reduce, norm, FFN, all-reduce)
        depends on the other half only through the KV cache the first half's
        attention writes, so one half's all-reduce on the SparseCore runs
        while the TensorCore computes the other half. The all-reduces are
        chained in the order they become ready (`last`), which also keeps XLA
        from combining a pair into one that both halves would wait for.
        """
        attn, res = [], []
        for i, h in enumerate(hidden_states):
            if attn and envs.TPU_MINIMAX_M3_SPLIT_ATTENTION_FIRST:
                # The second half waits for the first half's attention, which
                # then runs while the second half's previous all-reduce is
                # still on the SparseCore.
                h = _after(h, attn[0])
            if residual is None:
                n, r = self.input_layernorm(h), h
            else:
                n, r = self.input_layernorm(h, residual[i])
            attn.append(self.self_attn(halves[i].positions, n, halves[i].metadata))
            res.append(r)
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        post = []
        for i, a in enumerate(attn):
            last = _all_reduce(a, last)
            post.append(self.post_attention_layernorm(last, res[i]))
        out = []
        for h, _ in post:
            last = _all_reduce(ffn(h), last)
            out.append(last)
        return tuple(out), tuple(r for _, r in post), last


class _Half(NamedTuple):
    """One token half of a split step: its positions and the attention
    metadata (per KV-cache prefix) of its queries."""

    positions: torch.Tensor
    metadata: dict


def _split_metadata(md: AttentionMetadata, split: torch.Tensor) -> tuple:
    """`md` for the queries before token `split` (a 0-d tensor) and for the
    rest."""
    first_qsl, first_lens, second_qsl, second_lens = split_query_layout(
        md.query_start_loc, md.seq_lens, split
    )
    first = dataclasses.replace(md, query_start_loc=first_qsl, seq_lens=first_lens)
    second = dataclasses.replace(md, query_start_loc=second_qsl, seq_lens=second_lens)
    for m in (first, second):
        m.query_start_loc_cpu = None
        m.seq_lens_cpu = None
    return first, second


def _split_halves(
    positions: torch.Tensor, split: int, num_tokens: int
) -> tuple[_Half, _Half] | None:
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):
        return None
    first_positions = positions.narrow(0, 0, split)
    # The split point enters the metadata as a tensor computed in the graph.
    # As a python int it was frozen at the bucket the graph was traced at
    # (the token count is dynamic, but a scalar argument of clip is not), so
    # other buckets kept the traced split: at a smaller bucket the first half
    # took every query and the second half none, and the second half's K/V
    # were never written (TPU state check: every second-half page zero).
    split_at = torch.ones_like(first_positions, dtype=torch.int32).sum()
    derived: dict[int, tuple] = {}
    first, second = {}, {}
    for prefix, md in metadata.items():
        if id(md) not in derived:
            derived[id(md)] = _split_metadata(md, split_at)
        first[prefix], second[prefix] = derived[id(md)]
    return (
        _Half(first_positions, first),
        _Half(positions.narrow(0, split, num_tokens - split), second),
    )


def _whole_metadata(halves: tuple[_Half, _Half]) -> dict:
    """The step's metadata rejoined from its halves' (equal values)."""
    whole, joined = {}, {}
    for prefix, first in halves[0].metadata.items():
        if id(first) not in joined:
            second = halves[1].metadata[prefix]
            qsl = first.query_start_loc + second.query_start_loc
            md = dataclasses.replace(
                first,
                query_start_loc=qsl,
                seq_lens=join_seq_lens(
                    first.seq_lens, second.query_start_loc, second.seq_lens
                ),
            )
            md.query_start_loc_cpu = None
            md.seq_lens_cpu = None
            joined[id(first)] = md
        whole[prefix] = joined[id(first)]
    return whole


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
        self.split_min_tokens = envs.TPU_MINIMAX_M3_SPLIT_MIN_TOKENS
        if 0 < self.split_min_tokens < 2 * _AFTER_ROWS:
            raise ValueError(
                f"TPU_MINIMAX_M3_SPLIT_MIN_TOKENS must be 0 or at least "
                f"{2 * _AFTER_ROWS}, got {self.split_min_tokens}."
            )
        if self.split_min_tokens:
            logger.info_once(
                "TPU_MINIMAX_M3_SPLIT_MIN_TOKENS=%d: token-half split on, split "
                "point traced as a tensor",
                self.split_min_tokens,
            )
        if self.split_min_tokens and envs.TPU_TP_PALLAS_ALL_REDUCE_MAX_BYTES <= 0:
            logger.warning(
                "TPU_MINIMAX_M3_SPLIT_MIN_TOKENS without the Pallas all-reduce "
                "(TPU_TP_PALLAS_ALL_REDUCE_MAX_BYTES=0): the halves' all-reduces "
                "are unordered and XLA may combine them."
            )

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
        num_tokens = hidden_states.shape[0]
        halves = None
        if self.split_min_tokens:
            # One symbol for the token count: a separate one for positions'
            # length would be an extra input of the split trace only.
            torch._check(positions.shape[0] == num_tokens)
            # Every trace derives the halves' metadata (unsplit steps rejoin
            # them), so traces for different buckets take the same inputs in
            # the same order, as shape_variants needs to hand a bucket's
            # executable from one trace to another.
            halves = _split_halves(positions, num_tokens // 2, num_tokens)
        split = halves is not None and num_tokens >= self.split_min_tokens
        if not split:
            metadata = None
            if halves is not None:
                metadata = _whole_metadata(halves)
                positions = torch.cat([halves[0].positions, halves[1].positions])
            for idx, layer in enumerate(self.layers):
                hidden_states, residual = layer(
                    positions, hidden_states, residual, metadata
                )
                if idx + 1 in self.aux_hidden_state_layers:
                    aux_hidden_states.append(hidden_states + residual)
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            first = halves[0].positions.shape[0]
            stream = (
                hidden_states.narrow(0, 0, first),
                hidden_states.narrow(0, first, num_tokens - first),
            )
            last = None
            for idx, layer in enumerate(self.layers):
                stream, residual, last = layer.forward_split(
                    halves, stream, residual, last
                )
                if idx + 1 in self.aux_hidden_state_layers:
                    aux_hidden_states.append(_stream_sum(stream, residual))
            hidden_states = torch.cat(
                [self.norm(h, r)[0] for h, r in zip(stream, residual)]
            )
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

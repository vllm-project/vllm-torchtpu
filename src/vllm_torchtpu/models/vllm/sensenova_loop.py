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
"""SenseNova-Loop text model for TPU: a dense Qwen3.5 backbone with Loop and mHC.

Loop runs physical layers [start, end) `eval_count` times with shared weights;
every visit keeps its own KV / GDN state, so each repeat visit is a separate
decoder-layer module (own cache prefix) that borrows the physical layer's
weight-bearing submodules. mHC layers carry `streams` residual copies mixed by
two fp32 routers per layer (attention and MLP branch). The sublayers are the
Qwen3.5 ones the TPU backend already serves (GDN v3, gated attention on RPA v3,
SwiGLU MLP); the checkpoint's remote code (modeling_sensenova_loop.py) is the
reference.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm as Qwen3_5RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs

from vllm_torchtpu.models.vllm.sensenova_loop_config import (
    LAYER_TYPES,
    execution_plan,
    mhc_settings,
    physical_layer_types,
    visit_layer_physicals,
)

if TYPE_CHECKING:
    from transformers import Qwen3NextConfig


# ----------------------------------------------------------------------------
# mHC router math
# ----------------------------------------------------------------------------
def mhc_sinkhorn(logits: torch.Tensor, iterations: int) -> torch.Tensor:
    """Log-space Sinkhorn over the last two axes (rows, then columns), fp32;
    HF remote code `sinkhorn`."""
    logits = logits.float()
    for _ in range(iterations):
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
    return logits.exp()


def mhc_route(
    streams: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, iterations: int
):
    """mHC router (HF MHCRouter.forward): streams [T, S, H] bf16 -> branch [T, H]
    (streams dtype), post [T, S] and mix [T, S, S] fp32. The S-way contractions
    run as fp32 multiply-adds (= a bf16 matmul with fp32 accumulation, rounded
    once), not as a batched [1, S] x [S, H] matmul."""
    s = streams.shape[-2]
    flat = streams.flatten(-2).float()
    normed = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
    logits = torch.nn.functional.linear(normed, weight, bias)
    pre = logits[..., :s].sigmoid()
    post = 2 * logits[..., s : 2 * s].sigmoid()
    mix = mhc_sinkhorn(logits[..., 2 * s :].unflatten(-1, (s, s)), iterations)
    # HF rounds pre to the stream dtype before the matmul
    pre_q = pre.to(streams.dtype).float()
    branch = (pre_q.unsqueeze(-1) * streams.float()).sum(-2).to(streams.dtype)
    return branch, post, mix


def mhc_merge(
    streams: torch.Tensor, branch: torch.Tensor, post: torch.Tensor, mix: torch.Tensor
) -> torch.Tensor:
    """HF MHCRouter.merge: mix @ streams + post (x) branch, [T, S, H]; mix / post
    rounded to the stream dtype as HF does, the sum accumulated in fp32 and rounded
    once."""
    mix_q, post_q = mix.to(streams.dtype).float(), post.to(streams.dtype).float()
    mixed = (mix_q.unsqueeze(-1) * streams.float().unsqueeze(-3)).sum(-2)
    return (mixed + post_q.unsqueeze(-1) * branch.float().unsqueeze(-2)).to(
        streams.dtype
    )


class SenseNovaMHCRouter(nn.Module):
    """One mHC router: fp32 weight [2S + S*S, S*H] and bias [2S + S*S]."""

    def __init__(self, hidden_size: int, streams: int, iterations: int) -> None:
        super().__init__()
        out = 2 * streams + streams * streams
        self.iterations = iterations
        # fp32 like the checkpoint and HF (_keep_in_fp32_modules): Sinkhorn
        # over bf16 logits would move the mix.
        self.weight = nn.Parameter(
            torch.empty(out, streams * hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(out, dtype=torch.float32), requires_grad=False
        )
        for param in (self.weight, self.bias):
            set_weight_attrs(param, {"weight_loader": default_weight_loader})

    def forward(self, streams: torch.Tensor):
        return mhc_route(streams, self.weight, self.bias, self.iterations)


# ----------------------------------------------------------------------------
# Decoder layer
# ----------------------------------------------------------------------------
class SenseNovaLoopDecoderLayer(nn.Module):
    """Qwen3.5 dense layer (GDN or gated attention, then SwiGLU), with the two
    mHC routers when its physical index lies in the mHC range."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        physical: int,
        mhc: dict | None,
        prefix: str,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        if layer_type not in LAYER_TYPES:
            raise ValueError(f"Unsupported SenseNova-Loop layer type {layer_type!r}")
        self.layer_type = layer_type
        self.physical = physical
        if layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                cast("Qwen3NextConfig", config),
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
            )
        else:
            self.self_attn = Qwen3NextAttention(
                cast("Qwen3NextConfig", config),
                model_config=vllm_config.model_config,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.self_attn",
            )
        self.mlp = Qwen3NextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.has_mhc = mhc is not None and mhc["start"] <= physical < mhc["end"]
        if self.has_mhc:
            assert mhc is not None
            self.mhc_attn = SenseNovaMHCRouter(
                config.hidden_size, mhc["streams"], mhc["iterations"]
            )
            self.mhc_ffn = SenseNovaMHCRouter(
                config.hidden_size, mhc["streams"], mhc["iterations"]
            )

    def _mix(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.layer_type == "linear_attention":
            return self.linear_attn(hidden_states=hidden_states)
        return self.self_attn(positions=positions, hidden_states=hidden_states)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Upstream Qwen3_5DecoderLayer.forward for a dense layer (no layer
        scale, no sequence-parallel MoE): fused add + norm residual stream."""
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self._mix(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual

    def forward_streams(
        self, positions: torch.Tensor, streams: torch.Tensor
    ) -> torch.Tensor:
        """Remote code MHCDecoderLayer.forward on streams [T, S, H]."""
        branch, post, mix = self.mhc_attn(streams)
        branch = self._mix(positions, self.input_layernorm(branch))
        streams = mhc_merge(streams, branch, post, mix)
        branch, post, mix = self.mhc_ffn(streams)
        branch = self.mlp(self.post_attention_layernorm(branch))
        return mhc_merge(streams, branch, post, mix)


def share_weights(
    visit: SenseNovaLoopDecoderLayer, physical: SenseNovaLoopDecoderLayer
) -> None:
    """Point a repeat visit at the physical layer's weights.

    The visit keeps its own token-mixer module, whose prefix owns this visit's
    KV / GDN state, and its own Attention cache layer; every other child module
    and parameter is the physical layer's object, so loading and
    process_weights_after_loading (both deduplicated by module identity) run once.
    """
    linear = physical.layer_type == "linear_attention"
    mixer_name = "linear_attn" if linear else "self_attn"
    for name, child in physical.named_children():
        if name != mixer_name:
            setattr(visit, name, child)
    mixer, source = getattr(visit, mixer_name), getattr(physical, mixer_name)
    for name, child in source.named_children():
        if not isinstance(child, AttentionLayerBase):
            setattr(mixer, name, child)
    for name, param in source.named_parameters(recurse=False):
        setattr(mixer, name, param)
    # Any parameter outside the cache layers must now be the physical layer's.
    cache_params = {
        id(p)
        for m in visit.modules()
        if isinstance(m, AttentionLayerBase) and m is not mixer
        for p in m.parameters(recurse=False)
    }
    owned = {id(p) for p in physical.parameters()}
    stray = [
        n
        for n, p in visit.named_parameters()
        if id(p) not in owned and id(p) not in cache_params
    ]
    if stray:
        raise RuntimeError(f"Loop visit layer holds unshared weights: {stray}")


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
    }
)
class SenseNovaLoopModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        config = vllm_config.model_config.hf_text_config
        types = physical_layer_types(config)
        self.plan = execution_plan(hf_config, len(types))
        self.mhc = mhc_settings(hf_config, len(types))
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )
        # Modules 0..N-1 are the physical layers (checkpoint names); modules
        # N.. are the repeat visits, whose distinct prefixes give each visit
        # its own cache.
        repeats = visit_layer_physicals(self.plan)
        layers = [
            SenseNovaLoopDecoderLayer(
                vllm_config, kind, idx, self.mhc, prefix=f"{prefix}.layers.{idx}"
            )
            for idx, kind in enumerate(types)
        ]
        for physical in repeats:
            idx = len(layers)
            visit = SenseNovaLoopDecoderLayer(
                vllm_config,
                types[physical],
                physical,
                self.mhc,
                prefix=f"{prefix}.layers.{idx}",
            )
            share_weights(visit, layers[physical])
            layers.append(visit)
        self.layers = nn.ModuleList(layers)
        # (module index, physical index) per visit, in execution order.
        next_repeat = iter(range(len(types), len(layers)))
        seen: set[int] = set()
        self.schedule: list[tuple[int, int]] = []
        for physical in self.plan:
            self.schedule.append(
                (next(next_repeat) if physical in seen else physical, physical)
            )
            seen.add(physical)
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states, residual = inputs_embeds, None
        streams = hidden_states  # rebound at the first mHC layer
        for index, physical in self.schedule:
            layer = cast(SenseNovaLoopDecoderLayer, self.layers[index])
            if not layer.has_mhc:
                hidden_states, residual = layer(positions, hidden_states, residual)
                continue
            mhc = cast(dict, self.mhc)
            if physical == mhc["start"]:
                # Remote code `enter`: S copies of the full residual stream
                # (the pending fused add, rounded once as HF's bf16 add is).
                if residual is not None:
                    hidden_states, residual = hidden_states + residual, None
                streams = (
                    hidden_states.unsqueeze(-2)
                    .expand(-1, mhc["streams"], -1)
                    .contiguous()
                )
            streams = layer.forward_streams(positions, streams)
            if physical == mhc["end"] - 1:
                # Remote code `leave`: mean over streams (fp32, rounded once).
                hidden_states = streams.float().mean(-2).to(streams.dtype)
        if residual is None:
            return self.norm(hidden_states)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class SenseNovaLoopForConditionalGeneration(nn.Module, HasInnerState, IsHybrid):
    """Text-only SenseNova-Loop on TPU; the vision tower and MTP are not loaded."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
    # Qwen3.5's mapping fuses the split GDN, attention and MLP projections.
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "model.",
            "model.visual.": None,
            "mtp.": None,
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError("SenseNova-Loop on TPU does not support PP")
        self.model = SenseNovaLoopModel(
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

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: object | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise NotImplementedError("SenseNova-Loop on TPU does not support PP")
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        config = vllm_config.model_config.hf_text_config
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.linear_conv_kernel_dim,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )

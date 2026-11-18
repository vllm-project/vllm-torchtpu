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
"""OLMo 3.5 (Ai2's hybrid KDA / attention / latent-MoE model) for TPU.

Follows MaxText's `decoder_block: olmoe3` (`maxtext/models/olmoe3.py`), the
training reference for the checkpoints.
Per layer:

    h   = x + attn_out_norm(mixer(attn_in_norm(x)))
    out = h + ffn_out_norm(shared_ffn(y) + latent_up(moe(latent_down(y))))
          where y = ffn_in_norm(h)

The mixer is full attention on every `full_attention_interval`-th layer and
Kimi Delta Attention elsewhere; the first `first_k_dense_replace` layers have a
wide dense SwiGLU and no routed experts. The token embedding is scaled by
sqrt(hidden_size) and RMS-normalized before the first layer.

The KDA layers run the same fused conv1d + GDN v3 Pallas kernel as Kimi, with
OLMo's value heads twice as wide as its key heads and beta in [0, 2). The
routed experts go through vLLM's TPU fused-MoE backend with the latent
projections as its routed input / output transforms, as in Kimi K3.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Iterable
from typing import Any, cast

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    extract_layer_index,
    maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_torchtpu.layers.adapter.custom_ops.kda_attention_op import (
    build_kimi_dispatched_kda_op,
)
from vllm_torchtpu.layers.adapter.latent_proj_intra_chip import make_latent_projection
from vllm_torchtpu.layers.adapter.moe_hierarchical import (
    hierarchical_moe_parallel_config,
    hierarchical_split_or_none,
)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.kimi_k3.attention import (
    CausalDepthwiseConv1d,
    _load_a_log,
    kda_state_dtype,
)
from vllm_torchtpu.models.vllm.kimi_k3.layers import KimiMLP
from vllm_torchtpu.models.vllm.kimi_k3.moe import KimiRoutedOutputTransform
from vllm_torchtpu.utils import fresh_buffer

from .config import Olmo35Config

logger = init_logger(__name__)


def _config(vllm_config: VllmConfig) -> Olmo35Config:
    return cast(Olmo35Config, vllm_config.model_config.hf_text_config)


def _kda_local_dims(config: Olmo35Config, tp_size: int) -> tuple[int, int, int, int]:
    """Per-rank (heads, key head dim, value head dim, conv channels)."""
    if config.linear_num_heads % tp_size:
        raise ValueError(
            f"OLMo 3.5 KDA heads ({config.linear_num_heads}) must divide TP ({tp_size})"
        )
    heads = config.linear_num_heads // tp_size
    head_k = config.linear_key_head_dim
    head_v = config.linear_value_head_dim
    return heads, head_k, head_v, heads * (2 * head_k + head_v)


class Olmo35KimiDeltaAttention(nn.Module, MambaBase):
    """OLMo 3.5 KDA, executed by the fused conv1d + GDN v3 Pallas kernel.

    Differs from Kimi's KDA in the value head size (2x the key head size),
    the low-rank decay / output-gate projections (rank = value head size,
    and the gate's second projection has a bias) and beta's range ([0, 2)).
    """

    supports_dcp = False

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = _config(vllm_config)
        self.prefix = prefix
        self.vllm_config = vllm_config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_heads = config.linear_num_heads
        self.num_heads, self.head_k_dim, self.head_v_dim, self.local_conv_dim = (
            _kda_local_dims(config, self.tp_size)
        )
        self.conv_size = config.linear_conv_kernel_dim
        key_width = self.total_heads * self.head_k_dim
        value_width = self.total_heads * self.head_v_dim
        quant_config = vllm_config.quant_config
        if vllm_config.speculative_config is not None:
            raise NotImplementedError("OLMo 3.5 on TPU has no speculative decoding")

        # q | k | v | b in one matmul. Each rank's output is its own q, k, v
        # and b slices back to back, so the first `local_conv_dim` columns
        # are exactly the kernel's mixed_qkv layout.
        self.qkvb_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [key_width, key_width, value_width, self.total_heads],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkvb_proj",
        )
        # Low-rank decay (f) and output gate (g) down-projections, replicated.
        self.fg_a_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [self.head_v_dim, self.head_v_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fg_a_proj",
            disable_tp=True,
        )
        self.f_b_proj = ColumnParallelLinear(
            self.head_v_dim,
            key_width,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.g_b_proj = ColumnParallelLinear(
            self.head_v_dim,
            value_width,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.g_b_proj",
        )
        self.q_conv1d = CausalDepthwiseConv1d(key_width, self.conv_size)
        self.k_conv1d = CausalDepthwiseConv1d(key_width, self.conv_size)
        self.v_conv1d = CausalDepthwiseConv1d(value_width, self.conv_size)
        # [kernel_size, local q | k | v channels], built from the three
        # checkpoint convs in `process_weights_after_loading`.
        self.register_buffer(
            "conv_weight_fused",
            torch.empty(self.conv_size, self.local_conv_dim),
            persistent=False,
        )
        self.A_log = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(
            torch.empty(self.num_heads * self.head_k_dim, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": _load_a_log})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})
        self.o_norm = RMSNorm(self.head_v_dim, config.linear_norm_eps)
        self.o_proj = RowParallelLinear(
            value_width,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        # Bound by vLLM to (conv_state, recurrent_state); empty until then.
        self.kv_cache: tuple[torch.Tensor, ...] = ()
        self.kda_op = build_kimi_dispatched_kda_op(
            prefix,
            lower_bound=None,
            eps=config.linear_norm_eps,
            state_dim_first=is_conv_state_dim_first(),
            beta_scale=config.linear_beta_scale,
        )

        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.LINEAR

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return kda_state_dtype(self.vllm_config)

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (
            (self.conv_size - 1, 1, self.local_conv_dim),
            (self.num_heads, self.head_k_dim, self.head_v_dim),
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        del act_dtype
        fused = torch.cat(
            [
                w[:, 0, :].t()
                for w in (
                    self.q_conv1d.weight,
                    self.k_conv1d.weight,
                    self.v_conv1d.weight,
                )
            ],
            dim=1,
        )
        # Break the lazy view chain so the op sees a plain row-major buffer
        # (see KimiDeltaAttention.process_weights_after_loading).
        self.conv_weight_fused = fresh_buffer(fused)

    def _metadata(self) -> AttentionMetadata | None:
        if not is_forward_context_available():
            return None
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return None
        layer_metadata = metadata[self.prefix]
        if not isinstance(layer_metadata, AttentionMetadata):
            raise TypeError("KDA received incompatible attention metadata")
        return layer_metadata

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        del positions
        projected, _ = self.qkvb_proj(hidden_states)
        mixed_qkv, beta = projected.split([self.local_conv_dim, self.num_heads], dim=-1)
        fg, _ = self.fg_a_proj(hidden_states)
        f_a, g_a = fg.split([self.head_v_dim, self.head_v_dim], dim=-1)
        raw_gate, _ = self.f_b_proj(f_a)
        output_gate, _ = self.g_b_proj(g_a)

        metadata = self._metadata()
        if metadata is None or not self.kv_cache or self.kv_cache[0].numel() == 0:
            output = hidden_states.new_zeros(
                (hidden_states.shape[0], self.num_heads * self.head_v_dim)
            )
        elif len(self.kv_cache) == 1:
            raise NotImplementedError(
                "OLMo 3.5 KDA has no unified-pool path yet; run with "
                "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0"
            )
        else:
            conv_cache, recurrent_cache = self.kv_cache
            window_distribution = metadata.mamba_request_distribution
            if window_distribution is None:
                window_distribution = metadata.request_distribution
            # Unused without speculative decoding; any int32 [N] tensor keeps
            # the op's ABI (see KimiDeltaAttention.forward).
            slot_read_offsets = metadata.mamba_slot_read_offsets
            if slot_read_offsets is None:
                slot_read_offsets = metadata.mamba_state_indices
                if slot_read_offsets is None:
                    slot_read_offsets = metadata.seq_lens
            output = self.kda_op(
                mixed_qkv,
                raw_gate,
                beta,
                output_gate,
                conv_cache,
                recurrent_cache,
                self.conv_weight_fused,
                self.A_log,
                self.dt_bias,
                self.o_norm.weight,
                cast(torch.Tensor, metadata.query_start_loc),
                cast(torch.Tensor, metadata.mamba_state_indices),
                cast(torch.Tensor, metadata.seq_lens),
                cast(torch.Tensor, metadata.request_distribution),
                cast(torch.Tensor, window_distribution),
                cast(torch.Tensor, slot_read_offsets),
            ).flatten(1)
        output, _ = self.o_proj(output)
        return output


def _head_sharded_loader(total_heads: int):
    """Load a per-head [heads, ...] tensor onto this rank's heads.

    Replicates heads round-robin the way QKVParallelLinear does for KV heads
    when TP exceeds the head count.
    """

    def loader(parameter: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        local = parameter.shape[0]
        if total_heads >= tp_size:
            start = rank * local
        else:
            start = rank // (tp_size // total_heads)
        loaded_weight = loaded_weight.reshape(total_heads, *parameter.shape[1:])
        default_weight_loader(parameter, loaded_weight.narrow(0, start, local))

    return loader


class Olmo35Attention(nn.Module):
    """GQA full attention: NoPE, per-head QK RMSNorm, scalable softmax and a
    sigmoid output gate.

    MaxText scales the query by head_dim^-0.5 * log(position + 1) * gain[head]
    after the QK norm (scalable softmax) and runs the softmax unscaled, so the
    whole factor is folded into the query here and the attention scale is 1.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = _config(vllm_config)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_heads % tp_size:
            raise ValueError("num_attention_heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = config.rms_norm_eps
        self.scaling = self.head_dim**-0.5
        self.use_scalable_softmax = config.use_scalable_softmax
        quant_config = vllm_config.quant_config

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.g_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        # Independent RMSNorm gains per head (MaxText qk_norm_per_head).
        self.q_norm = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        self.k_norm = nn.Parameter(torch.ones(self.num_kv_heads, self.head_dim))
        set_weight_attrs(
            self.q_norm, {"weight_loader": _head_sharded_loader(self.total_num_heads)}
        )
        set_weight_attrs(
            self.k_norm,
            {"weight_loader": _head_sharded_loader(self.total_num_kv_heads)},
        )
        self.ssmax_scale = nn.Parameter(torch.ones(self.num_heads, dtype=torch.float32))
        set_weight_attrs(
            self.ssmax_scale,
            {"weight_loader": _head_sharded_loader(self.total_num_heads)},
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            1.0,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def _head_rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        x_f32 = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_f32 * weight.float()

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        num_tokens = hidden_states.shape[0]
        dtype = hidden_states.dtype
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        gate, _ = self.g_proj(hidden_states)

        q = self._head_rms_norm(
            q.reshape(num_tokens, self.num_heads, self.head_dim), self.q_norm
        )
        k = self._head_rms_norm(
            k.reshape(num_tokens, self.num_kv_heads, self.head_dim), self.k_norm
        ).to(dtype)
        q_scale = torch.full_like(positions, self.scaling, dtype=torch.float32)
        if self.use_scalable_softmax:
            q_scale = q_scale * torch.log(positions.float() + 1.0)
            q = q * (q_scale[:, None] * self.ssmax_scale[None, :])[..., None]
        else:
            q = q * q_scale[:, None, None]
        q = q.to(dtype)

        output = self.attn(
            q.reshape(num_tokens, self.q_size),
            k.reshape(num_tokens, self.kv_size),
            v.contiguous(),
        )
        output = output * torch.sigmoid(gate)
        output, _ = self.o_proj(output)
        return output


class Olmo35MoE(nn.Module):
    """Shared SwiGLU expert plus routed experts in a latent space.

    The router scores the full-width input; the experts see
    `latent_down(x)` and their weighted sum goes through `latent_up`.
    Routing is softmax over all experts, top-k, renormalized and scaled by
    `routed_scaling_factor` (top_k); the scale is folded into `latent_up`'s
    weight at load time, so the fused-MoE router runs unscaled.
    """

    def __init__(
        self,
        config: Olmo35Config,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        hidden_size = config.hidden_size
        latent_size = config.moe_latent_size
        tp_size = get_tensor_model_parallel_world_size()
        vllm_config = get_current_vllm_config_or_none()
        use_ep = (
            vllm_config is not None
            and vllm_config.parallel_config.enable_expert_parallel
        )

        self.gate = GateLinear(
            input_size=hidden_size,
            output_size=config.num_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.latent_down = make_latent_projection(
            hidden_size,
            latent_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.latent_down",
        )
        self.latent_up = make_latent_projection(
            latent_size,
            hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.latent_up",
        )
        self.shared_experts = KimiMLP(
            hidden_size,
            config.shared_expert_intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.shared_experts",
            reduce_results=False,
        )

        intermediate_size = config.moe_intermediate_size
        if use_ep:
            split = hierarchical_split_or_none()
            moe_tp_size = split[2] if split is not None else 1
        else:
            moe_tp_size = tp_size
        min_per_partition = 128
        if moe_tp_size > 1 and intermediate_size // moe_tp_size < min_per_partition:
            raise ValueError(
                "OLMo 3.5 expert width per TP shard is below the TPU GMM "
                "minimum; enable expert parallelism"
            )
        with hierarchical_moe_parallel_config():
            self.experts = FusedMoEFactory(
                shared_experts=self.shared_experts,
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=latent_size,
                intermediate_size=intermediate_size,
                activation=config.hidden_act,
                renormalize=True,
                quant_config=quant_config,
                prefix=f"{prefix}.experts",
                scoring_func="softmax",
                routed_scaling_factor=1.0,
                routed_input_transform=self.latent_down,
                routed_output_transform=KimiRoutedOutputTransform(None, self.latent_up),
            )

    def initialize_dummy_router(self) -> None:
        """Give dummy-loaded routers unit-variance logits.

        The dummy loader's +/-1e-3 weights make every router logit ~0, so top-k
        ties send every token to the same experts and EP benchmarks measure
        one hot rank. Fan-in scaled normal weights spread the load the way a
        trained router roughly does.
        """
        config = get_current_vllm_config_or_none()
        if config is None or str(config.load_config.load_format).lower() != "dummy":
            return
        generator = torch.Generator(device="cpu").manual_seed(
            1234 + zlib.crc32(self.prefix.encode())
        )
        weight_param = cast(torch.Tensor, self.gate.weight)
        weight = torch.randn(weight_param.shape, generator=generator)
        weight *= self.gate.input_size**-0.5
        with torch.no_grad():
            weight_param.copy_(weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.gate(hidden_states)
        return self.experts(hidden_states=hidden_states, router_logits=router_logits)


class Olmo35DecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = _config(vllm_config)
        self.layer_idx = extract_layer_index(prefix)
        self.is_full_attention = config.layer_types[self.layer_idx] == "full_attention"
        if self.is_full_attention:
            self.mixer: nn.Module = Olmo35Attention(vllm_config, f"{prefix}.mixer")
        else:
            self.mixer = Olmo35KimiDeltaAttention(vllm_config, f"{prefix}.mixer")

        self.is_moe = self.layer_idx >= config.first_k_dense_replace
        if self.is_moe:
            self.moe = Olmo35MoE(
                config, vllm_config.quant_config, prefix=f"{prefix}.moe"
            )
        else:
            self.mlp = KimiMLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
            )
        eps = config.rms_norm_eps
        self.attn_in_norm = RMSNorm(config.hidden_size, eps)
        self.attn_out_norm = RMSNorm(config.hidden_size, eps)
        self.ffn_in_norm = RMSNorm(config.hidden_size, eps)
        self.ffn_out_norm = RMSNorm(config.hidden_size, eps)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        mixer_out = self.mixer(positions, self.attn_in_norm(hidden_states))
        hidden_states = hidden_states + self.attn_out_norm(mixer_out)
        ffn_in = self.ffn_in_norm(hidden_states)
        ffn_out = self.moe(ffn_in) if self.is_moe else self.mlp(ffn_in)
        return hidden_states + self.ffn_out_norm(ffn_out)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
    }
)
class Olmo35Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = _config(vllm_config)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.embed_scale = math.sqrt(config.hidden_size)
        self.embedding_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [
                Olmo35DecoderLayer(vllm_config, prefix=f"{prefix}.layers.{i}")
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            assert input_ids is not None
            hidden_states = self.embed_input_ids(input_ids)
        hidden_states = self.embedding_norm(hidden_states * self.embed_scale)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)


class Olmo35ForCausalLM(nn.Module, HasInnerState, IsHybrid):
    """OLMo 3.5 text model for TPU."""

    # (checkpoint name, parameter name, shard id). A KDA layer and an
    # attention layer both have mixer.q_proj / k_proj / v_proj, so each entry
    # applies only where its target parameter exists.
    stacked_params_mapping = (
        (".mixer.q_proj.", ".mixer.qkvb_proj.", 0),
        (".mixer.k_proj.", ".mixer.qkvb_proj.", 1),
        (".mixer.v_proj.", ".mixer.qkvb_proj.", 2),
        (".mixer.b_proj.", ".mixer.qkvb_proj.", 3),
        (".mixer.f_a_proj.", ".mixer.fg_a_proj.", 0),
        (".mixer.g_a_proj.", ".mixer.fg_a_proj.", 1),
        (".mixer.q_proj.", ".mixer.qkv_proj.", "q"),
        (".mixer.k_proj.", ".mixer.qkv_proj.", "k"),
        (".mixer.v_proj.", ".mixer.qkv_proj.", "v"),
        (".gate_proj.", ".gate_up_proj.", 0),
        (".up_proj.", ".gate_up_proj.", 1),
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = _config(vllm_config)
        self.model = Olmo35Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors: object | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise ValueError("OLMo 3.5 on TPU does not support PP")
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def process_weights_after_loading(self) -> None:
        for module in self.modules():
            if isinstance(module, Olmo35MoE):
                module.initialize_dummy_router()

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return kda_state_dtype(vllm_config)

    @classmethod
    # The TPU KDA conv state is 3-D ([kernel - 1, 1, conv_dim], as for
    # KimiLinearForCausalLM); IsHybrid annotates a 2-D conv shape.
    # pyrefly: ignore[bad-override]
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...], tuple[int, int, int]]:
        config = _config(vllm_config)
        heads, head_k, head_v, conv_dim = _kda_local_dims(
            config, vllm_config.parallel_config.tensor_parallel_size
        )
        return (
            (config.linear_conv_kernel_dim - 1, 1, conv_dim),
            (heads, head_k, head_v),
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        routed_scale = self.config.routed_scaling_factor

        def _ordinary_weights():
            for name, loaded_weight in weights:
                if ".moe.latent_up." in name and routed_scale != 1.0:
                    # See Olmo35MoE: the top-k rescale is linear, so it is
                    # folded into the latent up-projection.
                    loaded_weight = loaded_weight * routed_scale
                if ".moe.experts." in name:
                    for (
                        param_name,
                        weight_name,
                        expert_id,
                        expert_shard_id,
                    ) in expert_params_mapping:
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        # FusedMoE attaches weight_loader to its parameters.
                        parameter = cast(Any, params_dict[name])
                        parameter.weight_loader(
                            parameter,
                            loaded_weight,
                            name,
                            expert_id=expert_id,
                            shard_id=expert_shard_id,
                        )
                        loaded.add(name)
                        break
                    continue
                for weight_name, param_name, shard_id in self.stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    candidate = name.replace(weight_name, param_name)
                    if candidate not in params_dict:
                        continue
                    # Merged linear layers attach weight_loader to their weights.
                    parameter = cast(Any, params_dict[candidate])
                    parameter.weight_loader(parameter, loaded_weight, shard_id)
                    loaded.add(candidate)
                    break
                else:
                    yield name, loaded_weight

        loader = AutoWeightsLoader(self)
        return loaded | loader.load_weights(_ordinary_weights())

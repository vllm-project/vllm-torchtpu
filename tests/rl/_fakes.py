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
"""Shared lightweight synthetic vLLM/TorchTPU layers and models for RL CPU tests."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from vllm_torchtpu.rl.manifest_generator import WEIGHT_FLIPPED_ATTR


class VllmUnquantizedLinearMethod:
    pass


class VllmUnquantizedFusedMoEMethod(nn.Module):
    def __init__(self, moe_config: _MockMoeConfig) -> None:
        super().__init__()
        self.moe = moe_config


class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        tp_size: int = 1,
        org_vocab_size: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size
        self.org_vocab_size = org_vocab_size or num_embeddings
        self.num_embeddings_padded = num_embeddings
        self.weight = nn.Parameter(
            torch.zeros(num_embeddings // tp_size, embedding_dim, dtype=dtype),
            requires_grad=False,
        )


class ParallelLMHead(VocabParallelEmbedding):
    def tie_weights(self, embed_tokens: VocabParallelEmbedding) -> None:
        self.weight = embed_tokens.weight


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=dtype), requires_grad=False
        )


class GemmaRMSNorm(RMSNorm):
    pass


class RMSNormGated(RMSNorm):
    pass


class ReplicatedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        flipped: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.quant_method = VllmUnquantizedLinearMethod()
        self.tp_size = 1
        shape = (in_features, out_features) if flipped else (out_features, in_features)
        self.weight = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
        setattr(self, WEIGHT_FLIPPED_ATTR, flipped)
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=dtype), requires_grad=False
            )


class GateLinear(ReplicatedLinear):
    pass


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tp_size: int = 1,
        bias: bool = False,
        flipped: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.quant_method = VllmUnquantizedLinearMethod()
        self.tp_size = tp_size
        local_out = out_features // tp_size
        shape = (in_features, local_out) if flipped else (local_out, in_features)
        self.weight = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
        setattr(self, WEIGHT_FLIPPED_ATTR, flipped)
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(local_out, dtype=dtype), requires_grad=False
            )


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        in_features: int,
        output_sizes: list[int],
        *,
        tp_size: int = 1,
        bias: bool = False,
        flipped: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.output_sizes = list(output_sizes)
        super().__init__(
            in_features,
            sum(output_sizes),
            tp_size=tp_size,
            bias=bias,
            flipped=flipped,
            dtype=dtype,
        )


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        *,
        tp_size: int = 1,
        bias: bool = False,
        flipped: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        nn.Module.__init__(self)
        self.quant_method = VllmUnquantizedLinearMethod()
        self.tp_size = tp_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.num_kv_head_replicas = max(1, tp_size // total_num_kv_heads)
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = max(1, total_num_kv_heads // tp_size)
        local_out = (self.num_heads + 2 * self.num_kv_heads) * head_size
        shape = (hidden_size, local_out) if flipped else (local_out, hidden_size)
        self.weight = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
        setattr(self, WEIGHT_FLIPPED_ATTR, flipped)
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(local_out, dtype=dtype), requires_grad=False
            )


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tp_size: int = 1,
        bias: bool = False,
        flipped: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.quant_method = VllmUnquantizedLinearMethod()
        self.tp_size = tp_size
        local_in = in_features // tp_size
        shape = (local_in, out_features) if flipped else (out_features, local_in)
        self.weight = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
        setattr(self, WEIGHT_FLIPPED_ATTR, flipped)
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=dtype), requires_grad=False
            )


class TpuRowParallelLinear(RowParallelLinear):
    pass


class GdnColumnParallelLinear(MergedColumnParallelLinear):
    def __init__(
        self,
        *,
        input_size: int,
        num_k_heads: int,
        num_v_heads: int,
        d_k: int,
        d_v: int,
        include_z: bool,
        tp_size: int = 1,
        is_conv1d: bool = False,
        flipped: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        nn.Module.__init__(self)
        self.tp_size = tp_size
        kq_rep = max(1, tp_size // num_k_heads)
        self.checkpoint_sizes = [num_k_heads * d_k] * 2 + [num_v_heads * d_v] * (
            1 + int(include_z)
        )
        self.replication_factors = [kq_rep] * 2 + [1] * (1 + int(include_z))
        local_out = sum(
            (sz * rep) // tp_size
            for sz, rep in zip(self.checkpoint_sizes, self.replication_factors)
        )
        if is_conv1d:
            self.weight = nn.Parameter(
                torch.zeros(local_out, 1, input_size, dtype=dtype), requires_grad=False
            )
        else:
            self.quant_method = VllmUnquantizedLinearMethod()
            shape = (input_size, local_out) if flipped else (local_out, input_size)
            self.weight = nn.Parameter(
                torch.zeros(shape, dtype=dtype), requires_grad=False
            )
            setattr(self, WEIGHT_FLIPPED_ATTR, flipped)


class GdnInterleavedColumnParallelLinear(MergedColumnParallelLinear):
    def __init__(
        self,
        *,
        input_size: int,
        num_kq_heads: int,
        num_v_heads: int,
        group_sizes: list[int],
        replicated_segments: list[bool],
        tp_size: int = 1,
        flipped: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        nn.Module.__init__(self)
        self.quant_method = VllmUnquantizedLinearMethod()
        self.tp_size = tp_size
        self.geometry = SimpleNamespace(
            num_kq_heads=num_kq_heads, num_v_heads=num_v_heads
        )
        self.group_sizes = list(group_sizes)
        self.replicated_segments = list(replicated_segments)
        local_out = (num_kq_heads * sum(group_sizes)) // tp_size
        shape = (input_size, local_out) if flipped else (local_out, input_size)
        self.weight = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
        setattr(self, WEIGHT_FLIPPED_ATTR, flipped)


@dataclass
class _MockMoeParallelConfig:
    use_ep: bool = False


@dataclass
class _MockMoeConfig:
    num_experts: int
    ep_size: int
    tp_size: int
    hidden_dim: int
    intermediate_size_per_partition: int
    moe_parallel_config: _MockMoeParallelConfig
    has_bias: bool = False


class RoutedExperts(nn.Module):
    def __init__(
        self,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        ep_size: int = 1,
        moe_tp_size: int = 1,
        has_bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        half = intermediate_size // moe_tp_size
        aligned_half = ((half + 127) // 128) * 128
        local_e = num_experts // ep_size
        self.moe_config = _MockMoeConfig(
            num_experts=num_experts,
            ep_size=ep_size,
            tp_size=moe_tp_size,
            hidden_dim=hidden_size,
            intermediate_size_per_partition=half,
            moe_parallel_config=_MockMoeParallelConfig(use_ep=ep_size > 1),
            has_bias=has_bias,
        )
        self.quant_method = VllmUnquantizedFusedMoEMethod(self.moe_config)
        self.w13_weight = nn.Parameter(
            torch.zeros(local_e, hidden_size, 2 * aligned_half, dtype=dtype),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.zeros(local_e, aligned_half, hidden_size, dtype=dtype),
            requires_grad=False,
        )


class Qwen2ForCausalLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 64,
        padded_vocab_size: int | None = None,
        hidden_size: int = 64,
        intermediate_size: int = 128,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        num_layers: int = 1,
        tp_size: int = 1,
        tie_word_embeddings: bool = True,
        flipped: bool = True,
    ) -> None:
        super().__init__()
        p_vocab = padded_vocab_size or vocab_size
        self.config = SimpleNamespace(architectures=["Qwen2ForCausalLM"])
        self.model = nn.Module()
        self.model.embed_tokens = VocabParallelEmbedding(
            p_vocab, hidden_size, tp_size=tp_size, org_vocab_size=vocab_size
        )
        layers = []
        for _ in range(num_layers):
            layer = nn.Module()
            layer.input_layernorm = RMSNorm(hidden_size)
            layer.self_attn = nn.Module()
            layer.self_attn.qkv_proj = QKVParallelLinear(
                hidden_size,
                hidden_size // num_heads,
                num_heads,
                num_kv_heads,
                tp_size=tp_size,
                bias=True,
                flipped=flipped,
            )
            layer.self_attn.o_proj = TpuRowParallelLinear(
                hidden_size, hidden_size, tp_size=tp_size, bias=False, flipped=flipped
            )
            layer.post_attention_layernorm = RMSNorm(hidden_size)
            layer.mlp = nn.Module()
            layer.mlp.gate_up_proj = MergedColumnParallelLinear(
                hidden_size,
                [intermediate_size, intermediate_size],
                tp_size=tp_size,
                flipped=flipped,
            )
            layer.mlp.down_proj = TpuRowParallelLinear(
                intermediate_size, hidden_size, tp_size=tp_size, flipped=flipped
            )
            layers.append(layer)
        self.model.layers = nn.ModuleList(layers)
        self.model.norm = RMSNorm(hidden_size)
        self.lm_head = ParallelLMHead(
            p_vocab, hidden_size, tp_size=tp_size, org_vocab_size=vocab_size
        )
        if tie_word_embeddings:
            self.lm_head.tie_weights(self.model.embed_tokens)


class Qwen3_5MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(
        self,
        *,
        vocab_size: int = 64,
        hidden_size: int = 64,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        head_dim: int = 8,
        num_k_heads: int = 2,
        num_v_heads: int = 4,
        d_k: int = 8,
        d_v: int = 8,
        num_experts: int = 4,
        moe_intermediate_size: int = 192,
        shared_intermediate_size: int = 64,
        tp_size: int = 1,
        ep_size: int = 1,
        moe_tp_size: int = 1,
        qkvz_flipped: bool = False,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"])
        self.model = nn.Module()
        self.model.embed_tokens = VocabParallelEmbedding(
            vocab_size, hidden_size, tp_size=tp_size
        )

        def _moe() -> nn.Module:
            m = nn.Module()
            m.gate = GateLinear(hidden_size, num_experts, flipped=True)
            m.shared_expert_gate = ReplicatedLinear(hidden_size, 1, flipped=True)
            m.shared_expert = nn.Module()
            m.shared_expert.gate_up_proj = MergedColumnParallelLinear(
                hidden_size,
                [shared_intermediate_size, shared_intermediate_size],
                tp_size=tp_size,
            )
            m.shared_expert.down_proj = TpuRowParallelLinear(
                shared_intermediate_size, hidden_size, tp_size=tp_size
            )
            m.experts = RoutedExperts(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=moe_intermediate_size,
                ep_size=ep_size,
                moe_tp_size=moe_tp_size,
            )
            return m

        l0 = nn.Module()
        l0.input_layernorm = GemmaRMSNorm(hidden_size)
        la = nn.Module()
        la.tp_size = tp_size
        la.conv1d = GdnColumnParallelLinear(
            input_size=4,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            d_k=d_k,
            d_v=d_v,
            include_z=False,
            tp_size=tp_size,
            is_conv1d=True,
        )
        la.in_proj_qkvz = GdnColumnParallelLinear(
            input_size=hidden_size,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            d_k=d_k,
            d_v=d_v,
            include_z=True,
            tp_size=tp_size,
            flipped=qkvz_flipped,
        )
        la.in_proj_ba = MergedColumnParallelLinear(
            hidden_size, [num_v_heads, num_v_heads], tp_size=tp_size, flipped=True
        )
        la.A_log = nn.Parameter(
            torch.zeros(num_v_heads // tp_size, dtype=torch.float32),
            requires_grad=False,
        )
        la.dt_bias = nn.Parameter(
            torch.ones(num_v_heads // tp_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        la.norm = RMSNormGated(d_v)
        la.out_proj = TpuRowParallelLinear(
            num_v_heads * d_v, hidden_size, tp_size=tp_size, flipped=True
        )
        l0.linear_attn = la
        l0.post_attention_layernorm = GemmaRMSNorm(hidden_size)
        l0.mlp = _moe()

        l1 = nn.Module()
        l1.input_layernorm = GemmaRMSNorm(hidden_size)
        attn = nn.Module()
        attn.qkv_proj = QKVParallelLinear(
            hidden_size,
            head_dim,
            num_heads * 2,
            num_kv_heads,
            tp_size=tp_size,
            flipped=True,
        )
        attn.o_proj = TpuRowParallelLinear(
            num_heads * head_dim, hidden_size, tp_size=tp_size, flipped=True
        )
        attn.q_norm = GemmaRMSNorm(head_dim)
        attn.k_norm = GemmaRMSNorm(head_dim)
        l1.self_attn = attn
        l1.post_attention_layernorm = GemmaRMSNorm(hidden_size)
        l1.mlp = _moe()

        self.model.layers = nn.ModuleList([l0, l1])
        self.model.norm = GemmaRMSNorm(hidden_size)
        self.lm_head = ParallelLMHead(vocab_size, hidden_size, tp_size=tp_size)


def slice_global_to_local(
    tensor: torch.Tensor,
    sharding: list[str | None],
    *,
    ranks: dict[str, int],
    world_sizes: dict[str, int],
) -> torch.Tensor:
    out = tensor
    for axis, ax_name in enumerate(sharding):
        if ax_name is not None and world_sizes[ax_name] > 1:
            chunk = out.shape[axis] // world_sizes[ax_name]
            out = out.narrow(axis, ranks[ax_name] * chunk, chunk)
    return out.contiguous()


def build_qwen2_hf_weights(
    *,
    vocab_size: int = 64,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_heads: int = 8,
    num_kv_heads: int = 2,
    tie_word_embeddings: bool = True,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    hd = hidden_size // num_heads
    q_d, kv_d = num_heads * hd, num_kv_heads * hd
    r = lambda *s: torch.randn(s, generator=g, dtype=torch.bfloat16)  # noqa: E731
    w = {
        "model.embed_tokens.weight": r(vocab_size, hidden_size),
        "model.layers.0.input_layernorm.weight": r(hidden_size),
        "model.layers.0.self_attn.q_proj.weight": r(q_d, hidden_size),
        "model.layers.0.self_attn.q_proj.bias": r(q_d),
        "model.layers.0.self_attn.k_proj.weight": r(kv_d, hidden_size),
        "model.layers.0.self_attn.k_proj.bias": r(kv_d),
        "model.layers.0.self_attn.v_proj.weight": r(kv_d, hidden_size),
        "model.layers.0.self_attn.v_proj.bias": r(kv_d),
        "model.layers.0.self_attn.o_proj.weight": r(hidden_size, hidden_size),
        "model.layers.0.post_attention_layernorm.weight": r(hidden_size),
        "model.layers.0.mlp.gate_proj.weight": r(intermediate_size, hidden_size),
        "model.layers.0.mlp.up_proj.weight": r(intermediate_size, hidden_size),
        "model.layers.0.mlp.down_proj.weight": r(hidden_size, intermediate_size),
        "model.norm.weight": r(hidden_size),
    }
    if not tie_word_embeddings:
        w["lm_head.weight"] = r(vocab_size, hidden_size)
    return w


def build_qwen35_hf_weights(
    *,
    vocab_size: int = 64,
    hidden_size: int = 64,
    num_heads: int = 8,
    num_kv_heads: int = 2,
    head_dim: int = 8,
    num_k_heads: int = 2,
    num_v_heads: int = 4,
    d_k: int = 8,
    d_v: int = 8,
    num_experts: int = 4,
    moe_intermediate_size: int = 192,
    shared_intermediate_size: int = 64,
    seed: int = 123,
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    key_d, val_d = num_k_heads * d_k, num_v_heads * d_v
    conv_d = 2 * key_d + val_d
    r = lambda *s, dt=torch.bfloat16: torch.randn(s, generator=g, dtype=dt)  # noqa: E731
    w: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": r(vocab_size, hidden_size),
        "model.norm.weight": r(hidden_size),
        "lm_head.weight": r(vocab_size, hidden_size),
        "model.layers.0.input_layernorm.weight": r(hidden_size),
        "model.layers.0.linear_attn.conv1d.weight": r(conv_d, 1, 4),
        "model.layers.0.linear_attn.in_proj_qkv.weight": r(conv_d, hidden_size),
        "model.layers.0.linear_attn.in_proj_z.weight": r(val_d, hidden_size),
        "model.layers.0.linear_attn.in_proj_b.weight": r(num_v_heads, hidden_size),
        "model.layers.0.linear_attn.in_proj_a.weight": r(num_v_heads, hidden_size),
        "model.layers.0.linear_attn.A_log": r(num_v_heads, dt=torch.float32),
        "model.layers.0.linear_attn.dt_bias": r(num_v_heads),
        "model.layers.0.linear_attn.norm.weight": r(d_v),
        "model.layers.0.linear_attn.out_proj.weight": r(hidden_size, val_d),
        "model.layers.0.post_attention_layernorm.weight": r(hidden_size),
        "model.layers.1.input_layernorm.weight": r(hidden_size),
        "model.layers.1.self_attn.q_proj.weight": r(
            2 * num_heads * head_dim, hidden_size
        ),
        "model.layers.1.self_attn.k_proj.weight": r(
            num_kv_heads * head_dim, hidden_size
        ),
        "model.layers.1.self_attn.v_proj.weight": r(
            num_kv_heads * head_dim, hidden_size
        ),
        "model.layers.1.self_attn.o_proj.weight": r(hidden_size, num_heads * head_dim),
        "model.layers.1.self_attn.q_norm.weight": r(head_dim),
        "model.layers.1.self_attn.k_norm.weight": r(head_dim),
        "model.layers.1.post_attention_layernorm.weight": r(hidden_size),
    }
    for li in (0, 1):
        lp = f"model.layers.{li}.mlp"
        w[f"{lp}.gate.weight"] = r(num_experts, hidden_size)
        w[f"{lp}.shared_expert_gate.weight"] = r(1, hidden_size)
        w[f"{lp}.shared_expert.gate_proj.weight"] = r(
            shared_intermediate_size, hidden_size
        )
        w[f"{lp}.shared_expert.up_proj.weight"] = r(
            shared_intermediate_size, hidden_size
        )
        w[f"{lp}.shared_expert.down_proj.weight"] = r(
            hidden_size, shared_intermediate_size
        )
        for e in range(num_experts):
            w[f"{lp}.experts.{e}.gate_proj.weight"] = r(
                moe_intermediate_size, hidden_size
            )
            w[f"{lp}.experts.{e}.up_proj.weight"] = r(
                moe_intermediate_size, hidden_size
            )
            w[f"{lp}.experts.{e}.down_proj.weight"] = r(
                hidden_size, moe_intermediate_size
            )
    return w

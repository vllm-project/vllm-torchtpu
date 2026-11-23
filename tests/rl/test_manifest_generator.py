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
"""CPU unit and numerical parity tests for the RL hybrid manifest generator."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from vllm_torchtpu.rl.manifest_generator import (
    WEIGHT_FLIPPED_ATTR,
    generate_manifest,
    generate_manifest_dict,
)
from vllm_torchtpu.rl.weight_conversion import (
    from_hf,
    invert_manifest,
    to_hf,
    validate_manifest,
)

# ---------------------------------------------------------------------------
# Lightweight synthetic vLLM / TorchTPU layer mocks (pure PyTorch, CPU-only)
# ---------------------------------------------------------------------------


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
        self.org_vocab_size = (
            org_vocab_size if org_vocab_size is not None else num_embeddings
        )
        assert num_embeddings % tp_size == 0
        self.weight = nn.Parameter(
            torch.zeros(num_embeddings // tp_size, embedding_dim, dtype=dtype),
            requires_grad=False,
        )


class ParallelLMHead(VocabParallelEmbedding):
    def tie_weights(self, embed_tokens: VocabParallelEmbedding) -> ParallelLMHead:
        self.weight = embed_tokens.weight
        return self


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
        input_size: int,
        output_size: int,
        *,
        bias: bool = False,
        flipped: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        shape = (input_size, output_size) if flipped else (output_size, input_size)
        self.weight = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
        setattr(self, WEIGHT_FLIPPED_ATTR, flipped)
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(output_size, dtype=dtype), requires_grad=False
            )


class GateLinear(ReplicatedLinear):
    pass


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        tp_size: int = 1,
        bias: bool = False,
        flipped: bool = True,
        num_kv_head_replicas: int = 1,
        head_size: int | None = None,
        total_num_kv_heads: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size
        self.num_kv_head_replicas = num_kv_head_replicas
        if head_size is not None:
            self.head_size = head_size
        if total_num_kv_heads is not None:
            self.total_num_kv_heads = total_num_kv_heads
        local_out = (output_size * num_kv_head_replicas) // tp_size
        shape = (input_size, local_out) if flipped else (local_out, input_size)
        self.weight = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
        setattr(self, WEIGHT_FLIPPED_ATTR, flipped)
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(local_out, dtype=dtype), requires_grad=False
            )


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        *,
        tp_size: int = 1,
        bias: bool = False,
        flipped: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.output_sizes = list(output_sizes)
        super().__init__(
            input_size,
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
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.num_kv_head_replicas = max(1, tp_size // total_num_kv_heads)
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = max(1, total_num_kv_heads // tp_size)
        local_out = (self.num_heads + 2 * self.num_kv_heads) * head_size
        nn.Module.__init__(self)
        self.tp_size = tp_size
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
        input_size: int,
        output_size: int,
        *,
        tp_size: int = 1,
        bias: bool = False,
        flipped: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size
        local_in = input_size // tp_size
        shape = (local_in, output_size) if flipped else (output_size, local_in)
        self.weight = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
        setattr(self, WEIGHT_FLIPPED_ATTR, flipped)
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(output_size, dtype=dtype), requires_grad=False
            )


class TpuRowParallelLinear(RowParallelLinear):
    pass


@dataclass(frozen=True)
class _MockGdnGeometry:
    num_kq_heads: int
    num_v_heads: int
    parallel_size: int
    kq_replication_factor: int
    local_num_kq_heads: int
    local_num_v_heads: int


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
        self.tp_size = tp_size
        kq_rep = max(1, tp_size // num_kq_heads)
        self.geometry = _MockGdnGeometry(
            num_kq_heads=num_kq_heads,
            num_v_heads=num_v_heads,
            parallel_size=tp_size,
            kq_replication_factor=kq_rep,
            local_num_kq_heads=num_kq_heads // min(num_kq_heads, tp_size),
            local_num_v_heads=num_v_heads // tp_size,
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
        self.ckpt_gate_proj_name = "gate_proj"
        self.ckpt_up_proj_name = "up_proj"
        self.ckpt_down_proj_name = "down_proj"
        # Post-VllmUnquantizedFusedMoEMethod.process_weights_after_loading layout
        self.w13_weight = nn.Parameter(
            torch.zeros(local_e, hidden_size, 2 * aligned_half, dtype=dtype),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.zeros(local_e, aligned_half, hidden_size, dtype=dtype),
            requires_grad=False,
        )
        if has_bias:
            self.w13_bias = nn.Parameter(
                torch.zeros(local_e, 1, 2 * aligned_half, dtype=torch.float32),
                requires_grad=False,
            )
            self.w2_bias = nn.Parameter(
                torch.zeros(local_e, 1, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )


# ---------------------------------------------------------------------------
# Synthetic Qwen2.5 Dense and Qwen3.5-MoE models
# ---------------------------------------------------------------------------


class _TinyQwen2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        head_dim = hidden_size // num_heads
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            tp_size=tp_size,
            bias=True,
            flipped=True,
        )
        self.o_proj = TpuRowParallelLinear(
            hidden_size, hidden_size, tp_size=tp_size, bias=False, flipped=True
        )


class _TinyQwen2MLP(nn.Module):
    def __init__(
        self, hidden_size: int, intermediate_size: int, *, tp_size: int = 1
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            tp_size=tp_size,
            bias=False,
            flipped=True,
        )
        self.down_proj = TpuRowParallelLinear(
            intermediate_size, hidden_size, tp_size=tp_size, bias=False, flipped=True
        )


class _TinyQwen2Layer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = _TinyQwen2Attention(
            hidden_size, num_heads, num_kv_heads, tp_size=tp_size
        )
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = _TinyQwen2MLP(hidden_size, intermediate_size, tp_size=tp_size)


class _TinyQwen2Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        padded_vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        num_layers: int,
        *,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            padded_vocab_size,
            hidden_size,
            tp_size=tp_size,
            org_vocab_size=vocab_size,
        )
        self.layers = nn.ModuleList(
            [
                _TinyQwen2Layer(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    num_kv_heads,
                    tp_size=tp_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size)


class Qwen2ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

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
    ) -> None:
        super().__init__()
        p_vocab = padded_vocab_size if padded_vocab_size is not None else vocab_size
        self.model = _TinyQwen2Model(
            vocab_size,
            p_vocab,
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            num_layers,
            tp_size=tp_size,
        )
        self.lm_head = ParallelLMHead(
            p_vocab, hidden_size, tp_size=tp_size, org_vocab_size=vocab_size
        )
        if tie_word_embeddings:
            self.lm_head.tie_weights(self.model.embed_tokens)


class VllmGatedDeltaNetAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        d_k: int,
        d_v: int,
        conv_kernel_size: int = 4,
        tp_size: int = 1,
        qkvz_flipped: bool = False,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size
        self.conv1d = GdnColumnParallelLinear(
            input_size=conv_kernel_size,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            d_k=d_k,
            d_v=d_v,
            include_z=False,
            tp_size=tp_size,
            is_conv1d=True,
        )
        self.in_proj_qkvz = GdnColumnParallelLinear(
            input_size=hidden_size,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            d_k=d_k,
            d_v=d_v,
            include_z=True,
            tp_size=tp_size,
            flipped=qkvz_flipped,
        )
        self.in_proj_ba = MergedColumnParallelLinear(
            hidden_size,
            [num_v_heads, num_v_heads],
            tp_size=tp_size,
            bias=False,
            flipped=True,
        )
        self.A_log = nn.Parameter(
            torch.zeros(num_v_heads // tp_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.dt_bias = nn.Parameter(
            torch.ones(num_v_heads // tp_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.norm = RMSNormGated(d_v)
        self.out_proj = TpuRowParallelLinear(
            num_v_heads * d_v, hidden_size, tp_size=tp_size, bias=False, flipped=True
        )


class _TinyQwen35FullAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        # attn_output_gate=True doubles Q heads in qkv_proj
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            head_dim,
            num_heads * 2,
            num_kv_heads,
            tp_size=tp_size,
            bias=False,
            flipped=True,
        )
        self.o_proj = TpuRowParallelLinear(
            num_heads * head_dim, hidden_size, tp_size=tp_size, bias=False, flipped=True
        )
        self.q_norm = GemmaRMSNorm(head_dim)
        self.k_norm = GemmaRMSNorm(head_dim)


class _TinyQwen35SparseMoeBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        moe_intermediate_size: int,
        shared_intermediate_size: int,
        tp_size: int = 1,
        ep_size: int = 1,
        moe_tp_size: int = 1,
    ) -> None:
        super().__init__()
        self.gate = GateLinear(hidden_size, num_experts, bias=False, flipped=True)
        self.shared_expert_gate = ReplicatedLinear(
            hidden_size, 1, bias=False, flipped=True
        )
        self.shared_expert = _TinyQwen2MLP(
            hidden_size, shared_intermediate_size, tp_size=tp_size
        )
        self.experts = RoutedExperts(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate_size,
            ep_size=ep_size,
            moe_tp_size=moe_tp_size,
        )


class _TinyQwen35Layer(nn.Module):
    def __init__(
        self,
        layer_type: str,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_k_heads: int,
        num_v_heads: int,
        d_k: int,
        d_v: int,
        num_experts: int,
        moe_intermediate_size: int,
        shared_intermediate_size: int,
        tp_size: int = 1,
        ep_size: int = 1,
        moe_tp_size: int = 1,
        qkvz_flipped: bool = False,
    ) -> None:
        super().__init__()
        self.input_layernorm = GemmaRMSNorm(hidden_size)
        if layer_type == "linear_attention":
            self.linear_attn = VllmGatedDeltaNetAttention(
                hidden_size=hidden_size,
                num_k_heads=num_k_heads,
                num_v_heads=num_v_heads,
                d_k=d_k,
                d_v=d_v,
                tp_size=tp_size,
                qkvz_flipped=qkvz_flipped,
            )
        else:
            self.self_attn = _TinyQwen35FullAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                tp_size=tp_size,
            )
        self.post_attention_layernorm = GemmaRMSNorm(hidden_size)
        self.mlp = _TinyQwen35SparseMoeBlock(
            hidden_size=hidden_size,
            num_experts=num_experts,
            moe_intermediate_size=moe_intermediate_size,
            shared_intermediate_size=shared_intermediate_size,
            tp_size=tp_size,
            ep_size=ep_size,
            moe_tp_size=moe_tp_size,
        )


class _TinyQwen35Model(nn.Module):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        vocab_size = kwargs["vocab_size"]
        hidden_size = kwargs["hidden_size"]
        tp_size = kwargs["tp_size"]
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size, hidden_size, tp_size=tp_size
        )
        self.layers = nn.ModuleList(
            [
                _TinyQwen35Layer(
                    "linear_attention",
                    hidden_size=hidden_size,
                    num_heads=kwargs["num_heads"],
                    num_kv_heads=kwargs["num_kv_heads"],
                    head_dim=kwargs["head_dim"],
                    num_k_heads=kwargs["num_k_heads"],
                    num_v_heads=kwargs["num_v_heads"],
                    d_k=kwargs["d_k"],
                    d_v=kwargs["d_v"],
                    num_experts=kwargs["num_experts"],
                    moe_intermediate_size=kwargs["moe_intermediate_size"],
                    shared_intermediate_size=kwargs["shared_intermediate_size"],
                    tp_size=tp_size,
                    ep_size=kwargs["ep_size"],
                    moe_tp_size=kwargs["moe_tp_size"],
                    qkvz_flipped=kwargs.get("qkvz_flipped", False),
                ),
                _TinyQwen35Layer(
                    "full_attention",
                    hidden_size=hidden_size,
                    num_heads=kwargs["num_heads"],
                    num_kv_heads=kwargs["num_kv_heads"],
                    head_dim=kwargs["head_dim"],
                    num_k_heads=kwargs["num_k_heads"],
                    num_v_heads=kwargs["num_v_heads"],
                    d_k=kwargs["d_k"],
                    d_v=kwargs["d_v"],
                    num_experts=kwargs["num_experts"],
                    moe_intermediate_size=kwargs["moe_intermediate_size"],
                    shared_intermediate_size=kwargs["shared_intermediate_size"],
                    tp_size=tp_size,
                    ep_size=kwargs["ep_size"],
                    moe_tp_size=kwargs["moe_tp_size"],
                    qkvz_flipped=kwargs.get("qkvz_flipped", False),
                ),
            ]
        )
        self.norm = GemmaRMSNorm(hidden_size)


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
        self.model = _TinyQwen35Model(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            d_k=d_k,
            d_v=d_v,
            num_experts=num_experts,
            moe_intermediate_size=moe_intermediate_size,
            shared_intermediate_size=shared_intermediate_size,
            tp_size=tp_size,
            ep_size=ep_size,
            moe_tp_size=moe_tp_size,
            qkvz_flipped=qkvz_flipped,
        )
        self.lm_head = ParallelLMHead(vocab_size, hidden_size, tp_size=tp_size)


# ---------------------------------------------------------------------------
# Reference sharding & vLLM+TorchTPU load_weights + process_weights_after_loading
# ---------------------------------------------------------------------------


def _slice_global_to_local(
    tensor: torch.Tensor,
    sharding: list[str | None],
    *,
    ranks: dict[str, int],
    world_sizes: dict[str, int],
) -> torch.Tensor:
    """Reference contiguous worker-side slicing matching TPUWorker."""
    out = tensor
    for axis, axis_name in enumerate(sharding):
        if axis_name is None:
            continue
        rank = ranks[axis_name]
        ws = world_sizes[axis_name]
        if ws == 1:
            continue
        chunk = out.shape[axis] // ws
        out = out.narrow(axis, rank * chunk, chunk)
    return out.contiguous()


def _build_qwen2_hf_weights(
    *,
    vocab_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_kv_heads: int,
    tie_word_embeddings: bool,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    head_dim = hidden_size // num_heads
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=g, dtype=torch.bfloat16)

    weights = {
        "model.embed_tokens.weight": rand(vocab_size, hidden_size),
        "model.layers.0.input_layernorm.weight": rand(hidden_size),
        "model.layers.0.self_attn.q_proj.weight": rand(q_dim, hidden_size),
        "model.layers.0.self_attn.q_proj.bias": rand(q_dim),
        "model.layers.0.self_attn.k_proj.weight": rand(kv_dim, hidden_size),
        "model.layers.0.self_attn.k_proj.bias": rand(kv_dim),
        "model.layers.0.self_attn.v_proj.weight": rand(kv_dim, hidden_size),
        "model.layers.0.self_attn.v_proj.bias": rand(kv_dim),
        "model.layers.0.self_attn.o_proj.weight": rand(hidden_size, q_dim),
        "model.layers.0.post_attention_layernorm.weight": rand(hidden_size),
        "model.layers.0.mlp.gate_proj.weight": rand(intermediate_size, hidden_size),
        "model.layers.0.mlp.up_proj.weight": rand(intermediate_size, hidden_size),
        "model.layers.0.mlp.down_proj.weight": rand(hidden_size, intermediate_size),
        "model.norm.weight": rand(hidden_size),
    }
    if not tie_word_embeddings:
        weights["lm_head.weight"] = rand(vocab_size, hidden_size)
    return weights


def _reference_qwen2_rank_weights(
    hf: dict[str, torch.Tensor],
    *,
    tp_rank: int,
    tp_size: int,
    padded_vocab_size: int,
    num_heads: int,
    num_kv_heads: int,
    tie_word_embeddings: bool,
) -> dict[str, torch.Tensor]:
    """Reference vLLM load_weights() + VllmUnquantizedLinearMethod PWAL."""
    embed = hf["model.embed_tokens.weight"]
    if padded_vocab_size > embed.shape[0]:
        pad = torch.zeros(
            padded_vocab_size - embed.shape[0], embed.shape[1], dtype=embed.dtype
        )
        embed = torch.cat([embed, pad], dim=0)
    local_vocab = padded_vocab_size // tp_size
    embed_local = embed[tp_rank * local_vocab : (tp_rank + 1) * local_vocab]

    kv_rep = max(1, tp_size // num_kv_heads)
    kv_unique = tp_size // kv_rep
    kv_rank = tp_rank // kv_rep

    def shard0(t: torch.Tensor, rank: int, ws: int) -> torch.Tensor:
        c = t.shape[0] // ws
        return t[rank * c : (rank + 1) * c]

    def shard1(t: torch.Tensor, rank: int, ws: int) -> torch.Tensor:
        c = t.shape[1] // ws
        return t[:, rank * c : (rank + 1) * c]

    q_w = shard0(hf["model.layers.0.self_attn.q_proj.weight"], tp_rank, tp_size)
    k_w = shard0(hf["model.layers.0.self_attn.k_proj.weight"], kv_rank, kv_unique)
    v_w = shard0(hf["model.layers.0.self_attn.v_proj.weight"], kv_rank, kv_unique)
    qkv_w = torch.cat([q_w, k_w, v_w], dim=0).transpose(0, 1).contiguous()

    q_b = shard0(hf["model.layers.0.self_attn.q_proj.bias"], tp_rank, tp_size)
    k_b = shard0(hf["model.layers.0.self_attn.k_proj.bias"], kv_rank, kv_unique)
    v_b = shard0(hf["model.layers.0.self_attn.v_proj.bias"], kv_rank, kv_unique)
    qkv_b = torch.cat([q_b, k_b, v_b], dim=0).contiguous()

    o_w = (
        shard1(hf["model.layers.0.self_attn.o_proj.weight"], tp_rank, tp_size)
        .transpose(0, 1)
        .contiguous()
    )
    gate_w = shard0(hf["model.layers.0.mlp.gate_proj.weight"], tp_rank, tp_size)
    up_w = shard0(hf["model.layers.0.mlp.up_proj.weight"], tp_rank, tp_size)
    gate_up_w = torch.cat([gate_w, up_w], dim=0).transpose(0, 1).contiguous()
    down_w = (
        shard1(hf["model.layers.0.mlp.down_proj.weight"], tp_rank, tp_size)
        .transpose(0, 1)
        .contiguous()
    )

    ref = {
        "model.embed_tokens.weight": embed_local,
        "model.layers.0.input_layernorm.weight": hf[
            "model.layers.0.input_layernorm.weight"
        ],
        "model.layers.0.self_attn.qkv_proj.weight": qkv_w,
        "model.layers.0.self_attn.qkv_proj.bias": qkv_b,
        "model.layers.0.self_attn.o_proj.weight": o_w,
        "model.layers.0.post_attention_layernorm.weight": hf[
            "model.layers.0.post_attention_layernorm.weight"
        ],
        "model.layers.0.mlp.gate_up_proj.weight": gate_up_w,
        "model.layers.0.mlp.down_proj.weight": down_w,
        "model.norm.weight": hf["model.norm.weight"],
    }
    if tie_word_embeddings:
        ref["lm_head.weight"] = embed_local
    else:
        lm = hf["lm_head.weight"]
        if padded_vocab_size > lm.shape[0]:
            pad = torch.zeros(
                padded_vocab_size - lm.shape[0], lm.shape[1], dtype=lm.dtype
            )
            lm = torch.cat([lm, pad], dim=0)
        ref["lm_head.weight"] = lm[tp_rank * local_vocab : (tp_rank + 1) * local_vocab]
    return ref


def _build_qwen35_hf_weights(
    *,
    vocab_size: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_k_heads: int,
    num_v_heads: int,
    d_k: int,
    d_v: int,
    num_experts: int,
    moe_intermediate_size: int,
    shared_intermediate_size: int,
    conv_kernel_size: int = 4,
    seed: int = 123,
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    key_dim = num_k_heads * d_k
    value_dim = num_v_heads * d_v
    conv_dim = 2 * key_dim + value_dim

    def rand(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.randn(shape, generator=g, dtype=dtype)

    w: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": rand(vocab_size, hidden_size),
        "model.norm.weight": rand(hidden_size),
        "lm_head.weight": rand(vocab_size, hidden_size),
        # Layer 0: GDN linear_attention
        "model.layers.0.input_layernorm.weight": rand(hidden_size),
        "model.layers.0.linear_attn.conv1d.weight": rand(conv_dim, 1, conv_kernel_size),
        "model.layers.0.linear_attn.in_proj_qkv.weight": rand(conv_dim, hidden_size),
        "model.layers.0.linear_attn.in_proj_z.weight": rand(value_dim, hidden_size),
        "model.layers.0.linear_attn.in_proj_b.weight": rand(num_v_heads, hidden_size),
        "model.layers.0.linear_attn.in_proj_a.weight": rand(num_v_heads, hidden_size),
        "model.layers.0.linear_attn.A_log": rand(num_v_heads, dtype=torch.float32),
        "model.layers.0.linear_attn.dt_bias": rand(num_v_heads),
        "model.layers.0.linear_attn.norm.weight": rand(d_v),
        "model.layers.0.linear_attn.out_proj.weight": rand(hidden_size, value_dim),
        "model.layers.0.post_attention_layernorm.weight": rand(hidden_size),
        # Layer 1: Full attention (with output gate -> 2*num_heads for Q)
        "model.layers.1.input_layernorm.weight": rand(hidden_size),
        "model.layers.1.self_attn.q_proj.weight": rand(
            num_heads * 2 * head_dim, hidden_size
        ),
        "model.layers.1.self_attn.k_proj.weight": rand(
            num_kv_heads * head_dim, hidden_size
        ),
        "model.layers.1.self_attn.v_proj.weight": rand(
            num_kv_heads * head_dim, hidden_size
        ),
        "model.layers.1.self_attn.o_proj.weight": rand(
            hidden_size, num_heads * head_dim
        ),
        "model.layers.1.self_attn.q_norm.weight": rand(head_dim),
        "model.layers.1.self_attn.k_norm.weight": rand(head_dim),
        "model.layers.1.post_attention_layernorm.weight": rand(hidden_size),
    }

    for layer_idx in (0, 1):
        p = f"model.layers.{layer_idx}.mlp"
        w[f"{p}.gate.weight"] = rand(num_experts, hidden_size)
        w[f"{p}.shared_expert_gate.weight"] = rand(1, hidden_size)
        w[f"{p}.shared_expert.gate_proj.weight"] = rand(
            shared_intermediate_size, hidden_size
        )
        w[f"{p}.shared_expert.up_proj.weight"] = rand(
            shared_intermediate_size, hidden_size
        )
        w[f"{p}.shared_expert.down_proj.weight"] = rand(
            hidden_size, shared_intermediate_size
        )
        for e in range(num_experts):
            w[f"{p}.experts.{e}.gate_proj.weight"] = rand(
                moe_intermediate_size, hidden_size
            )
            w[f"{p}.experts.{e}.up_proj.weight"] = rand(
                moe_intermediate_size, hidden_size
            )
            w[f"{p}.experts.{e}.down_proj.weight"] = rand(
                hidden_size, moe_intermediate_size
            )
    return w


def _reference_routed_experts_rank_weights(
    hf: dict[str, torch.Tensor],
    prefix: str,
    *,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
    moe_tp_rank: int,
    moe_tp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference RoutedExperts load_weights + VllmUnquantizedFusedMoEMethod PWAL."""
    local_e = num_experts // ep_size
    e_start = ep_rank * local_e
    e_end = e_start + local_e

    w13_list = []
    w2_list = []
    for e in range(e_start, e_end):
        gate = hf[f"{prefix}.{e}.gate_proj.weight"]
        up = hf[f"{prefix}.{e}.up_proj.weight"]
        down = hf[f"{prefix}.{e}.down_proj.weight"]
        half = gate.shape[0] // moe_tp_size
        g_slice = gate[moe_tp_rank * half : (moe_tp_rank + 1) * half]
        u_slice = up[moe_tp_rank * half : (moe_tp_rank + 1) * half]
        d_slice = down[:, moe_tp_rank * half : (moe_tp_rank + 1) * half]
        w13_list.append(torch.cat([g_slice, u_slice], dim=0))
        w2_list.append(d_slice)

    w13 = torch.stack(w13_list, dim=0).transpose(1, 2).contiguous()
    w2 = torch.stack(w2_list, dim=0).transpose(1, 2).contiguous()

    half = w13.shape[-1] // 2
    aligned_half = ((half + 127) // 128) * 128
    if aligned_half != half:
        pad1 = w13.new_zeros((*w13.shape[:-1], aligned_half - half))
        w13 = torch.cat([w13[..., :half], pad1, w13[..., half:], pad1], dim=-1)
        pad2 = w2.new_zeros((w2.shape[0], aligned_half - half, w2.shape[2]))
        w2 = torch.cat([w2, pad2], dim=1)
    return w13.contiguous(), w2.contiguous()


# ---------------------------------------------------------------------------
# 1. Qwen2.5 Dense Parity & Round-Trip Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tp_size,num_heads,num_kv_heads,tie_word_embeddings,padded_vocab",
    [
        (1, 8, 2, True, 64),
        (2, 8, 2, True, 64),
        (4, 8, 2, True, 64),  # GQA KV replication: num_kv_head_replicas=2
        (2, 4, 4, False, 72),  # Pure MHA + untied lm_head + vocab padding
    ],
)
def test_qwen2_dense_parity_and_roundtrip(
    tp_size: int,
    num_heads: int,
    num_kv_heads: int,
    tie_word_embeddings: bool,
    padded_vocab: int,
) -> None:
    vocab_size = 64
    hidden_size = 64
    intermediate_size = 128
    model = Qwen2ForCausalLM(
        vocab_size=vocab_size,
        padded_vocab_size=padded_vocab,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        tp_size=tp_size,
        tie_word_embeddings=tie_word_embeddings,
    )
    manifest_json = generate_manifest(model, tp_size=tp_size)
    manifest = json.loads(manifest_json)
    validate_manifest(manifest)
    assert manifest["stripes"] == {"tp_size": tp_size}

    if tp_size > num_kv_heads:
        # Verify repeat op is emitted for GQA KV replication
        all_ops = [next(iter(op)) for r in manifest["rules"] for op in r["ops"]]
        assert "repeat" in all_ops

    if tie_word_embeddings:
        assert manifest["tied_keys"] == [
            ["model.embed_tokens.weight", "lm_head.weight"]
        ]
    else:
        assert manifest["tied_keys"] == []

    hf_weights = _build_qwen2_hf_weights(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        tie_word_embeddings=tie_word_embeddings,
    )
    converted = from_hf(manifest_json, hf_weights)
    assert not any(k.startswith("__tmp__:") for k in converted)

    rule_by_target = {
        r["target_keys"][0]: r
        for r in manifest["rules"]
        if len(r["target_keys"]) == 1 and not r["target_keys"][0].startswith("__tmp__:")
    }

    # Check bit-exact parity across EVERY TP rank via pure contiguous slicing
    for tp_rank in range(tp_size):
        ref = _reference_qwen2_rank_weights(
            hf_weights,
            tp_rank=tp_rank,
            tp_size=tp_size,
            padded_vocab_size=padded_vocab,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            tie_word_embeddings=tie_word_embeddings,
        )
        for key, expected in ref.items():
            if key in rule_by_target:
                sharding = rule_by_target[key]["sharding"]
            else:
                # Tied key inherits sharding from its source key
                sharding = rule_by_target["model.embed_tokens.weight"]["sharding"]
            actual = _slice_global_to_local(
                converted[key],
                sharding,
                ranks={"tp": tp_rank},
                world_sizes={"tp": tp_size},
            )
            assert actual.shape == expected.shape, f"Shape mismatch for {key}"
            assert actual.dtype == expected.dtype, f"Dtype mismatch for {key}"
            assert torch.equal(actual, expected), (
                f"Value mismatch for {key} rank={tp_rank}"
            )

    # Verify invert_manifest() round-trip recovers original HF weights bit-for-bit
    inv_manifest = invert_manifest(manifest_json)
    validate_manifest(inv_manifest)
    recovered = to_hf(json.dumps(inv_manifest), converted)
    assert set(recovered.keys()) == set(hf_weights.keys())
    for k, orig in hf_weights.items():
        assert torch.equal(recovered[k], orig), f"Round-trip mismatch for {k}"


def test_standalone_kv_replication_uses_repeat_op() -> None:
    """Standalone ColumnParallelLinear KV projection uses repeat/unrepeat ops."""

    class _StandaloneKVModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.k_proj = ColumnParallelLinear(
                input_size=32,
                output_size=16,  # 2 KV heads * head_dim 8
                tp_size=4,
                bias=True,
                flipped=True,
                num_kv_head_replicas=2,
                head_size=8,
                total_num_kv_heads=2,
            )

    model = _StandaloneKVModel()
    manifest_json = generate_manifest(model, tp_size=4)
    manifest = json.loads(manifest_json)

    # Confirm repeat op is emitted for both weight and bias
    for rule in manifest["rules"]:
        op_names = [next(iter(op)) for op in rule["ops"]]
        assert "repeat" in op_names

    g = torch.Generator().manual_seed(7)
    hf = {
        "k_proj.weight": torch.randn(16, 32, generator=g, dtype=torch.bfloat16),
        "k_proj.bias": torch.randn(16, generator=g, dtype=torch.bfloat16),
    }
    converted = from_hf(manifest_json, hf)
    assert converted["k_proj.weight"].shape == (32, 32)
    assert converted["k_proj.bias"].shape == (32,)

    # Check each of 4 ranks gets KV head rank // 2
    for rank in range(4):
        kv_idx = rank // 2
        expected_w = hf["k_proj.weight"][kv_idx * 8 : (kv_idx + 1) * 8].T.contiguous()
        expected_b = hf["k_proj.bias"][kv_idx * 8 : (kv_idx + 1) * 8]
        actual_w = _slice_global_to_local(
            converted["k_proj.weight"],
            [None, "tp"],
            ranks={"tp": rank},
            world_sizes={"tp": 4},
        )
        actual_b = _slice_global_to_local(
            converted["k_proj.bias"],
            ["tp"],
            ranks={"tp": rank},
            world_sizes={"tp": 4},
        )
        assert torch.equal(actual_w, expected_w)
        assert torch.equal(actual_b, expected_b)

    # Round-trip via invert_manifest (unrepeat) recovers exact HF tensors
    inv = invert_manifest(manifest_json)
    recovered = to_hf(json.dumps(inv), converted)
    assert torch.equal(recovered["k_proj.weight"], hf["k_proj.weight"])
    assert torch.equal(recovered["k_proj.bias"], hf["k_proj.bias"])


# ---------------------------------------------------------------------------
# 2. Qwen3.5-MoE (FA + GDN + RoutedExperts + Shared Experts + Router) Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tp_size,dp_size,ep_size,moe_tp_size,qkvz_flipped",
    [
        (1, 1, 1, 1, False),  # TP=1, EP=1, padded 192 -> 256
        (2, 1, 1, 2, False),  # TP=2, MoE TP=2 shard-local interleave, padded 96 -> 128
        (4, 1, 1, 4, True),  # TP=4 (GDN Q/K rep=2, FA KV rep=2), flipped in_proj_qkvz
        (2, 1, 2, 1, False),  # EP=2, MoE TP=1
        (2, 2, 2, 2, False),  # Hybrid EP=2 + MoE TP=2 (DP=2, TP=2)
    ],
)
def test_qwen35_moe_parity_and_roundtrip(
    tp_size: int,
    dp_size: int,
    ep_size: int,
    moe_tp_size: int,
    qkvz_flipped: bool,
) -> None:
    cfg = dict(
        vocab_size=64,
        hidden_size=64,
        num_heads=8,
        num_kv_heads=2,
        head_dim=8,
        num_k_heads=2,
        num_v_heads=4,
        d_k=8,
        d_v=8,
        num_experts=4,
        moe_intermediate_size=192,
        shared_intermediate_size=64,
    )
    model = Qwen3_5MoeForCausalLM(
        **cfg,
        tp_size=tp_size,
        ep_size=ep_size,
        moe_tp_size=moe_tp_size,
        qkvz_flipped=qkvz_flipped,
    )
    manifest_json = generate_manifest(
        model, tp_size=tp_size, dp_size=dp_size, ep_size=ep_size
    )
    manifest = json.loads(manifest_json)
    validate_manifest(manifest)
    assert manifest["expert_axis_grouped"] is True
    assert manifest["stripes"] == {
        "tp_size": tp_size,
        "ep_size": ep_size,
        "moe_tp_size": moe_tp_size,
    }

    hf = _build_qwen35_hf_weights(**cfg)
    converted = from_hf(manifest_json, hf)
    assert not any(k.startswith("__tmp__:") for k in converted)
    rule_by_target = {
        r["target_keys"][0]: r
        for r in manifest["rules"]
        if len(r["target_keys"]) == 1 and not r["target_keys"][0].startswith("__tmp__:")
    }

    # Verify RoutedExperts parity across all (ep_rank, moe_tp_rank) combinations
    for layer_idx in (0, 1):
        prefix = f"model.layers.{layer_idx}.mlp.experts"
        w13_key = f"{prefix}.w13_weight"
        w2_key = f"{prefix}.w2_weight"
        for ep_rank in range(ep_size):
            for moe_tp_rank in range(moe_tp_size):
                ref_w13, ref_w2 = _reference_routed_experts_rank_weights(
                    hf,
                    prefix,
                    num_experts=cfg["num_experts"],
                    ep_rank=ep_rank,
                    ep_size=ep_size,
                    moe_tp_rank=moe_tp_rank,
                    moe_tp_size=moe_tp_size,
                )
                ranks = {"tp": 0, "ep": ep_rank, "moe_tp": moe_tp_rank}
                ws = {"tp": tp_size, "ep": ep_size, "moe_tp": moe_tp_size}
                act_w13 = _slice_global_to_local(
                    converted[w13_key],
                    rule_by_target[w13_key]["sharding"],
                    ranks=ranks,
                    world_sizes=ws,
                )
                act_w2 = _slice_global_to_local(
                    converted[w2_key],
                    rule_by_target[w2_key]["sharding"],
                    ranks=ranks,
                    world_sizes=ws,
                )
                assert torch.equal(act_w13, ref_w13)
                assert torch.equal(act_w2, ref_w2)

    # Verify GDN layer (in_proj_qkvz, conv1d, in_proj_ba, A_log, dt_bias, out_proj)
    key_dim = cfg["num_k_heads"] * cfg["d_k"]
    val_dim = cfg["num_v_heads"] * cfg["d_v"]
    kq_rep = max(1, tp_size // cfg["num_k_heads"])
    kq_unique = tp_size // kq_rep

    for tp_rank in range(tp_size):
        kq_rank = tp_rank // kq_rep
        ranks = {"tp": tp_rank, "ep": 0, "moe_tp": 0}
        ws = {"tp": tp_size, "ep": ep_size, "moe_tp": moe_tp_size}

        # in_proj_qkvz
        qkv = hf["model.layers.0.linear_attn.in_proj_qkv.weight"]
        z = hf["model.layers.0.linear_attn.in_proj_z.weight"]
        q_part, k_part, v_part = torch.split(qkv, [key_dim, key_dim, val_dim], dim=0)
        ref_qkvz = torch.cat(
            [
                q_part.chunk(kq_unique, dim=0)[kq_rank],
                k_part.chunk(kq_unique, dim=0)[kq_rank],
                v_part.chunk(tp_size, dim=0)[tp_rank],
                z.chunk(tp_size, dim=0)[tp_rank],
            ],
            dim=0,
        )
        if qkvz_flipped:
            ref_qkvz = ref_qkvz.transpose(0, 1).contiguous()
        qkvz_key = "model.layers.0.linear_attn.in_proj_qkvz.weight"
        act_qkvz = _slice_global_to_local(
            converted[qkvz_key],
            rule_by_target[qkvz_key]["sharding"],
            ranks=ranks,
            world_sizes=ws,
        )
        assert torch.equal(act_qkvz, ref_qkvz)

        # conv1d (3D)
        conv = hf["model.layers.0.linear_attn.conv1d.weight"]
        cq, ck, cv = torch.split(conv, [key_dim, key_dim, val_dim], dim=0)
        ref_conv = torch.cat(
            [
                cq.chunk(kq_unique, dim=0)[kq_rank],
                ck.chunk(kq_unique, dim=0)[kq_rank],
                cv.chunk(tp_size, dim=0)[tp_rank],
            ],
            dim=0,
        )
        conv_key = "model.layers.0.linear_attn.conv1d.weight"
        act_conv = _slice_global_to_local(
            converted[conv_key],
            rule_by_target[conv_key]["sharding"],
            ranks=ranks,
            world_sizes=ws,
        )
        assert torch.equal(act_conv, ref_conv)

        # in_proj_ba (homogeneous MergedColumnParallelLinear)
        b_w = hf["model.layers.0.linear_attn.in_proj_b.weight"]
        a_w = hf["model.layers.0.linear_attn.in_proj_a.weight"]
        ref_ba = (
            torch.cat(
                [
                    b_w.chunk(tp_size, dim=0)[tp_rank],
                    a_w.chunk(tp_size, dim=0)[tp_rank],
                ],
                dim=0,
            )
            .transpose(0, 1)
            .contiguous()
        )
        ba_key = "model.layers.0.linear_attn.in_proj_ba.weight"
        act_ba = _slice_global_to_local(
            converted[ba_key],
            rule_by_target[ba_key]["sharding"],
            ranks=ranks,
            world_sizes=ws,
        )
        assert torch.equal(act_ba, ref_ba)

        # A_log (fp32) & dt_bias
        for name in ("A_log", "dt_bias"):
            k = f"model.layers.0.linear_attn.{name}"
            ref_t = hf[k].chunk(tp_size, dim=0)[tp_rank]
            act_t = _slice_global_to_local(
                converted[k],
                rule_by_target[k]["sharding"],
                ranks=ranks,
                world_sizes=ws,
            )
            assert torch.equal(act_t, ref_t)

    # Full invert_manifest() round-trip parity
    inv = invert_manifest(manifest_json)
    validate_manifest(inv)
    recovered = to_hf(json.dumps(inv), converted)
    assert set(recovered.keys()) == set(hf.keys())
    for k, orig in hf.items():
        assert torch.equal(recovered[k], orig), f"Round-trip mismatch for {k}"


@pytest.mark.parametrize("tp_size", [1, 2])
def test_gdn_interleaved_ba_parity_and_roundtrip(tp_size: int) -> None:
    class _InterleavedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj_ba = GdnInterleavedColumnParallelLinear(
                input_size=32,
                num_kq_heads=2,
                num_v_heads=4,
                group_sizes=[2, 2],
                replicated_segments=[False, False],
                tp_size=tp_size,
                flipped=True,
            )

    model = _InterleavedModel()
    manifest_json = generate_manifest(model, tp_size=tp_size)
    manifest = json.loads(manifest_json)

    g = torch.Generator().manual_seed(99)
    hf_w = torch.randn(8, 32, generator=g, dtype=torch.bfloat16)
    converted = from_hf(manifest_json, {"in_proj_ba.weight": hf_w})

    # Reference GdnInterleavedColumnParallelLinear.weight_loader + transpose
    groups = hf_w.reshape(2, 4, 32)
    b_all, a_all = groups.split([2, 2], dim=1)
    b_flat = b_all.flatten(0, 1)  # [4, 32]
    a_flat = a_all.flatten(0, 1)  # [4, 32]
    for rank in range(tp_size):
        ref_local = (
            torch.cat(
                [
                    b_flat.chunk(tp_size, dim=0)[rank],
                    a_flat.chunk(tp_size, dim=0)[rank],
                ],
                dim=0,
            )
            .transpose(0, 1)
            .contiguous()
        )
        act_local = _slice_global_to_local(
            converted["in_proj_ba.weight"],
            manifest["rules"][0]["sharding"],
            ranks={"tp": rank},
            world_sizes={"tp": tp_size},
        )
        assert torch.equal(act_local, ref_local)

    inv = invert_manifest(manifest_json)
    recovered = to_hf(json.dumps(inv), converted)
    assert torch.equal(recovered["in_proj_ba.weight"], hf_w)


# ---------------------------------------------------------------------------
# 3. Fail-Closed Guardrail Tests
# ---------------------------------------------------------------------------


def test_fail_closed_quantized_dtypes() -> None:
    dtypes = [torch.int8, torch.uint8]
    if hasattr(torch, "float8_e4m3fn"):
        dtypes.append(torch.float8_e4m3fn)
    for dt in dtypes:
        m = nn.Module()
        m.proj = ReplicatedLinear(16, 16, dtype=dt)
        with pytest.raises(NotImplementedError, match="dtype"):
            generate_manifest(m)


@pytest.mark.parametrize("quant_name", ["fp8", "mxfp4", "nvfp4", "awq", "gptq"])
def test_fail_closed_quantization_config(quant_name: str) -> None:
    @dataclass
    class _MC:
        quantization: str

    m = Qwen2ForCausalLM()
    with pytest.raises(NotImplementedError, match="Quantization"):
        generate_manifest(m, model_config=_MC(quantization=quant_name))


def test_fail_closed_quantized_method_and_scale_param() -> None:
    class VllmFp8LinearMethodTPU:
        pass

    m = Qwen2ForCausalLM()
    m.model.layers[0].self_attn.o_proj.quant_method = VllmFp8LinearMethodTPU()
    with pytest.raises(NotImplementedError, match="Quantized method"):
        generate_manifest(m)

    m2 = nn.Module()
    layer = ReplicatedLinear(16, 16)
    layer.weight_scale = nn.Parameter(
        torch.ones(16, dtype=torch.float32), requires_grad=False
    )
    m2.proj = layer
    with pytest.raises(NotImplementedError, match="Quantization parameter"):
        generate_manifest(m2)


@pytest.mark.parametrize(
    "field,val,match",
    [
        ("pipeline_parallel_size", 2, "PP"),
        ("prefill_context_parallel_size", 2, "PCP"),
        ("decode_context_parallel_size", 2, "DCP"),
        ("enable_eplb", True, "EPLB"),
    ],
)
def test_fail_closed_unsupported_parallelism(field: str, val: Any, match: str) -> None:
    class _PC:
        tensor_parallel_size = 1
        data_parallel_size = 1
        pipeline_parallel_size = 1
        prefill_context_parallel_size = 1
        decode_context_parallel_size = 1
        enable_eplb = False

    pc = _PC()
    setattr(pc, field, val)
    m = Qwen2ForCausalLM()
    with pytest.raises(NotImplementedError, match=match):
        generate_manifest(m, parallel_config=pc)


def test_fail_closed_lora() -> None:
    @dataclass
    class _MC:
        enable_lora: bool = True

    m = Qwen2ForCausalLM()
    with pytest.raises(NotImplementedError, match="LoRA"):
        generate_manifest(m, model_config=_MC())

    class ColumnParallelLinearWithLoRA(ColumnParallelLinear):
        pass

    m2 = nn.Module()
    m2.proj = ColumnParallelLinearWithLoRA(16, 16)
    with pytest.raises(NotImplementedError, match="LoRA"):
        generate_manifest(m2)


def test_fail_closed_vision_and_mtp() -> None:
    m_vis = nn.Module()
    m_vis.visual = ReplicatedLinear(16, 16)
    with pytest.raises(NotImplementedError, match="vision/MTP"):
        generate_manifest(m_vis)

    m_mtp = nn.Module()
    m_mtp.mtp = ReplicatedLinear(16, 16)
    with pytest.raises(NotImplementedError, match="vision/MTP"):
        generate_manifest(m_mtp)

    @dataclass
    class _MC:
        enable_mtp: bool = True

    with pytest.raises(NotImplementedError, match="MTP"):
        generate_manifest(Qwen2ForCausalLM(), model_config=_MC())


def test_fail_closed_unrecognized_module_and_expert_bias() -> None:
    m = nn.Module()
    m.unknown = nn.Linear(16, 16, bias=False)
    with pytest.raises(NotImplementedError, match="Unrecognized module"):
        generate_manifest(m)

    m_bias = nn.Module()
    m_bias.experts = RoutedExperts(
        num_experts=2,
        hidden_size=32,
        intermediate_size=64,
        has_bias=True,
    )
    with pytest.raises(NotImplementedError, match="bias"):
        generate_manifest(m_bias)


def test_fail_closed_gqa_q_head_padding() -> None:
    m = Qwen2ForCausalLM(num_heads=8, num_kv_heads=2, tp_size=2)
    m.model.layers[0].self_attn.orig_total_num_heads = 7
    m.model.layers[0].self_attn.total_num_heads = 8
    with pytest.raises(NotImplementedError, match="zero-padding"):
        generate_manifest(m, tp_size=2)


def test_manifest_generator_has_zero_jax_imports() -> None:
    src_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "vllm_torchtpu"
        / "rl"
        / "manifest_generator.py"
    )
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("jax")
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            assert not node.module.startswith("jax")


def test_generate_manifest_dict_matches_json() -> None:
    m = Qwen2ForCausalLM()
    d = generate_manifest_dict(m)
    s = generate_manifest(m)
    assert json.loads(s) == d


def test_torch_tpu_dp_size_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """TORCH_TPU_DP_SIZE overrides data_parallel_size when dp_size is None."""

    class _PC:
        tensor_parallel_size = 2
        data_parallel_size = 1
        enable_expert_parallel = True

    monkeypatch.setenv("TORCH_TPU_DP_SIZE", "2")
    model = Qwen3_5MoeForCausalLM(
        num_experts=4,
        tp_size=2,
        ep_size=4,
        moe_tp_size=1,
    )
    manifest = generate_manifest_dict(model, parallel_config=_PC())
    assert manifest["stripes"] == {"tp_size": 2, "ep_size": 4, "moe_tp_size": 1}

    # Explicit dp_size kwarg takes precedence over TORCH_TPU_DP_SIZE
    model_dp1 = Qwen3_5MoeForCausalLM(
        num_experts=4,
        tp_size=2,
        ep_size=2,
        moe_tp_size=1,
    )
    manifest_explicit = generate_manifest_dict(
        model_dp1, parallel_config=_PC(), dp_size=1
    )
    assert manifest_explicit["stripes"] == {
        "tp_size": 2,
        "ep_size": 2,
        "moe_tp_size": 1,
    }


def test_square_routed_experts_disambiguation() -> None:
    """Square RoutedExperts dims disambiguate via quant_method or fail closed."""

    class VllmUnquantizedFusedMoEMethod:
        pass

    class UnquantizedFusedMoEMethod:
        pass

    def _make_square_model(
        quant_method: Any, *, hidden_size: int = 128, intermediate_size: int = 128
    ) -> nn.Module:
        m = nn.Module()
        # w2_weight has shape (2, 128, 128) -> square dims (1, 2) == hidden_size
        experts = RoutedExperts(
            num_experts=2,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            ep_size=1,
            moe_tp_size=1,
        )
        if quant_method is not None:
            experts.quant_method = quant_method
        m.experts = experts
        return m

    # 1. VllmUnquantizedFusedMoEMethod -> transposed=True (emits transpose [0, 2, 1])
    m_tpu = _make_square_model(VllmUnquantizedFusedMoEMethod())
    d_tpu = generate_manifest_dict(m_tpu)
    w2_rule_tpu = next(
        r for r in d_tpu["rules"] if r["target_keys"] == ["experts.w2_weight"]
    )
    assert {"transpose": {"axes": [0, 2, 1]}} in w2_rule_tpu["ops"]

    # 2. UnquantizedFusedMoEMethod -> transposed=False (no transpose [0, 2, 1])
    m_stock = _make_square_model(UnquantizedFusedMoEMethod())
    # Stock layout for w13_weight is [E, 2*128, 128], w2_weight is [E, 128, 128]
    m_stock.experts.w13_weight = nn.Parameter(
        torch.zeros(2, 256, 128, dtype=torch.bfloat16), requires_grad=False
    )
    d_stock = generate_manifest_dict(m_stock)
    w2_rule_stock = next(
        r for r in d_stock["rules"] if r["target_keys"] == ["experts.w2_weight"]
    )
    assert {"transpose": {"axes": [0, 2, 1]}} not in w2_rule_stock["ops"]

    g = torch.Generator().manual_seed(11)
    hf = {
        "experts.0.gate_proj.weight": torch.randn(
            128, 128, generator=g, dtype=torch.bfloat16
        ),
        "experts.1.gate_proj.weight": torch.randn(
            128, 128, generator=g, dtype=torch.bfloat16
        ),
        "experts.0.up_proj.weight": torch.randn(
            128, 128, generator=g, dtype=torch.bfloat16
        ),
        "experts.1.up_proj.weight": torch.randn(
            128, 128, generator=g, dtype=torch.bfloat16
        ),
        "experts.0.down_proj.weight": torch.randn(
            128, 128, generator=g, dtype=torch.bfloat16
        ),
        "experts.1.down_proj.weight": torch.randn(
            128, 128, generator=g, dtype=torch.bfloat16
        ),
    }
    out_stock = from_hf(json.dumps(d_stock), hf)
    expected_w2 = torch.stack(
        [hf["experts.0.down_proj.weight"], hf["experts.1.down_proj.weight"]], dim=0
    )
    assert torch.equal(out_stock["experts.w2_weight"], expected_w2)

    # 3. No WEIGHT_FLIPPED_ATTR and no quant_method -> fail closed NotImplementedError
    m_ambig = _make_square_model(None)
    with pytest.raises(NotImplementedError, match="Ambiguous square RoutedExperts"):
        generate_manifest(m_ambig)


def test_routed_experts_stripes_moe_tp_matches_rule_when_cfg_omits_tp_size() -> None:
    """stripes['moe_tp_size'] matches _emit_routed_experts_rule when tp_size omitted."""

    class _CfgWithoutTp:
        num_experts = 4
        ep_size = 2
        hidden_dim = 64
        intermediate_size_per_partition = 128
        moe_parallel_config = _MockMoeParallelConfig(use_ep=True)
        has_bias = False

    m = nn.Module()
    experts = RoutedExperts(
        num_experts=4,
        hidden_size=64,
        intermediate_size=128,
        ep_size=2,
        moe_tp_size=1,
    )
    experts.moe_config = _CfgWithoutTp()
    m.experts = experts

    manifest = generate_manifest_dict(m, tp_size=2, ep_size=2)
    assert manifest["stripes"] == {"tp_size": 2, "ep_size": 2, "moe_tp_size": 1}
    w13_rule = next(
        r for r in manifest["rules"] if r["target_keys"] == ["experts.w13_weight"]
    )
    assert w13_rule["sharding"] == ["ep", None, None]


def test_fail_closed_routed_experts_swigluoai_without_bias() -> None:
    """RoutedExperts with swigluoai activation fails closed even without bias."""

    class _ActEnum:
        value = "swigluoai"

    m1 = nn.Module()
    e1 = RoutedExperts(
        num_experts=2,
        hidden_size=64,
        intermediate_size=128,
        has_bias=False,
    )
    e1.moe_config.activation = "swigluoai"
    m1.experts = e1
    with pytest.raises(NotImplementedError, match="swigluoai"):
        generate_manifest(m1)

    m2 = nn.Module()
    e2 = RoutedExperts(
        num_experts=2,
        hidden_size=64,
        intermediate_size=128,
        has_bias=False,
    )
    e2.activation = _ActEnum()
    m2.experts = e2
    with pytest.raises(NotImplementedError, match="swigluoai"):
        generate_manifest(m2)


def test_non_divisible_tp_dimensions_raise_value_error() -> None:
    """Non-divisible TP / replication dimensions raise a clear ValueError."""
    m = nn.Module()
    m.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    # output_sizes=[10, 10] is not divisible by tp_size=4
    layer = MergedColumnParallelLinear(
        input_size=16,
        output_sizes=[10, 10],
        tp_size=1,
        flipped=True,
    )
    layer.tp_size = 4
    m.gate_up_proj = layer
    with pytest.raises(ValueError, match="evenly divisible"):
        generate_manifest(m, tp_size=4)

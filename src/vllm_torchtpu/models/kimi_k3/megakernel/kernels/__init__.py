# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# src/vllm_torchtpu/megakernel/kimi_k3/kernels/__init__.py
"""Kimi K3 checkpoint configuration and shared whole-array math helpers.

``Config`` reads the checkpoint's ``config.json``. The functions below are
plain JAX on whole arrays (no fused kernels): RMS norm, the attention
residual, and the tensor-parallel sum that the decode megakernel evaluates
outside its Pallas calls. All matrix weights use ``[input, output]`` layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import jax
import jax.numpy as jnp

from vllm_torchtpu.models.kimi_k3.megakernel.common.constants import (
    AXIS_TP,
    DEFAULT_VOCAB_SIZE,
)
from vllm_torchtpu.models.kimi_k3.megakernel.common.paths import require_local_path

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Static model architecture and attention-block geometry for Kimi K3."""

    dim: int = 7168
    latent: int = 3584
    expert_hidden: int = 3072
    shared_hidden: int = 6144
    dense_hidden: int = 33792
    experts: int = 896
    top_k: int = 16
    layers: int = 93
    heads: int = 96
    head_dim: int = 128
    q_rank: int = 1536
    kv_rank: int = 512
    qk_extra: int = 64
    vocab: int = DEFAULT_VOCAB_SIZE
    conv_size: int = 4
    residual_block: int = 12
    eps: float = 1e-5
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0
    gate_lower_bound: float = -5.0
    full_attention: tuple[int, ...] = (*range(3, 92, 4), 92)

    @classmethod
    def from_checkpoint(cls, directory: Path | str) -> Self:
        """Load and validate a Kimi K3 configuration from ``config.json``."""
        checkpoint_dir = require_local_path(directory, label="checkpoint directory")
        raw = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
        text_config = raw.get("text_config", raw)
        linear_attn_config = text_config["linear_attn_config"]
        required = {
            "mla_use_nope": True,
            "mla_use_output_gate": True,
            "latent_moe_use_norm": True,
            "moe_renormalize": True,
            "moe_router_activation_func": "sigmoid",
            "hidden_act": "situ",
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "num_expert_group": 1,
            "routed_scaling_factor": 1.0,
        }
        for name, expected in required.items():
            if text_config[name] != expected:
                msg = f"Unsupported K3 setting {name}={text_config[name]!r}"
                raise ValueError(msg)
        use_full_rank_gate = linear_attn_config["use_full_rank_gate"]
        if not use_full_rank_gate:
            msg = (
                "K3 requires the full-rank output gate "
                f"(use_full_rank_gate={use_full_rank_gate!r})"
            )
            raise ValueError(msg)
        return cls(
            dim=text_config["hidden_size"],
            latent=text_config["routed_expert_hidden_size"],
            expert_hidden=text_config["moe_intermediate_size"],
            shared_hidden=(
                text_config["moe_intermediate_size"] * text_config["num_shared_experts"]
            ),
            dense_hidden=text_config["intermediate_size"],
            experts=text_config["num_experts"],
            top_k=text_config["num_experts_per_token"],
            layers=text_config["num_hidden_layers"],
            heads=text_config["num_attention_heads"],
            head_dim=linear_attn_config["head_dim"],
            q_rank=text_config["q_lora_rank"],
            kv_rank=text_config["kv_lora_rank"],
            qk_extra=text_config["qk_rope_head_dim"],
            vocab=text_config["vocab_size"],
            conv_size=linear_attn_config["short_conv_kernel_size"],
            residual_block=text_config["attn_res_block_size"],
            eps=text_config["rms_norm_eps"],
            situ_beta=text_config["activation_situ_beta"],
            situ_linear_beta=text_config["activation_situ_linear_beta"],
            gate_lower_bound=linear_attn_config["gate_lower_bound"],
            full_attention=tuple(
                layer_id - 1 for layer_id in linear_attn_config["full_attn_layers"]
            ),
        )


def rms(
    x: jax.Array,
    weight: jax.Array,
    eps: float = 1e-5,
) -> jax.Array:
    """Apply root-mean-square normalization in FP32 and cast back to input dtype."""
    value = x.astype(jnp.float32)
    value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + eps)
    return (value * weight.astype(jnp.float32)).astype(x.dtype)


def attention_residual(
    prefix: jax.Array,
    blocks: jax.Array,
    folded_weight: jax.Array,
    eps: float = 1e-5,
) -> jax.Array:
    """Mix block-boundary residual states with the current prefix via softmax."""
    values = jnp.concatenate((blocks, prefix[None]), axis=0).astype(jnp.float32)
    scores = jnp.sum(values * folded_weight, axis=-1)
    scores *= jax.lax.rsqrt(jnp.mean(values * values, axis=-1) + eps)
    mixed = jnp.sum(jax.nn.softmax(scores)[:, None] * values, axis=0)
    return jax.lax.optimization_barrier(mixed).astype(prefix.dtype)


def tp_sum(value: jax.Array, axis_name: str = AXIS_TP) -> jax.Array:
    """Keep the reference collective in FP32 before any BF16 output cast."""
    return jax.lax.optimization_barrier(jax.lax.psum(value, axis_name))


__all__ = [
    "Config",
    "attention_residual",
    "rms",
    "tp_sum",
]

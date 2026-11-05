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
"""Kimi K3 checkpoint configuration and shared whole-array math helpers.

``Config`` reads the checkpoint's ``config.json``. The functions below are
plain JAX on whole arrays (no fused kernels): RMS norm, the attention
residual and the tensor-parallel sum that the decode megakernel evaluates
outside its Pallas calls.
All matrix weights use [input, output] layout.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Config:
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
    vocab: int = 163840
    conv_size: int = 4
    residual_block: int = 12
    eps: float = 1e-5
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0
    gate_lower_bound: float = -5.0
    full_attention: tuple[int, ...] = tuple(range(3, 92, 4)) + (92,)

    @classmethod
    def from_checkpoint(cls, directory):
        raw = json.loads((Path(directory) / "config.json").read_text())
        c = raw.get("text_config", raw)
        a = c["linear_attn_config"]
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
            if c[name] != expected:
                raise ValueError(f"Unsupported K3 setting {name}={c[name]!r}")
        if not a["use_full_rank_gate"]:
            raise ValueError("K3 requires the full-rank output gate")
        return cls(
            dim=c["hidden_size"],
            latent=c["routed_expert_hidden_size"],
            expert_hidden=c["moe_intermediate_size"],
            shared_hidden=c["moe_intermediate_size"] * c["num_shared_experts"],
            dense_hidden=c["intermediate_size"],
            experts=c["num_experts"],
            top_k=c["num_experts_per_token"],
            layers=c["num_hidden_layers"],
            heads=c["num_attention_heads"],
            head_dim=a["head_dim"],
            q_rank=c["q_lora_rank"],
            kv_rank=c["kv_lora_rank"],
            qk_extra=c["qk_rope_head_dim"],
            vocab=c["vocab_size"],
            conv_size=a["short_conv_kernel_size"],
            residual_block=c["attn_res_block_size"],
            eps=c["rms_norm_eps"],
            situ_beta=c["activation_situ_beta"],
            situ_linear_beta=c["activation_situ_linear_beta"],
            gate_lower_bound=a["gate_lower_bound"],
            full_attention=tuple(i - 1 for i in a["full_attn_layers"]),
        )


def rms(x, weight, eps=1e-5):
    value = x.astype(jnp.float32)
    value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + eps)
    return (value * weight.astype(jnp.float32)).astype(x.dtype)


def attention_residual(prefix, blocks, folded_weight, eps=1e-5):
    values = jnp.concatenate((blocks, prefix[None]), axis=0).astype(jnp.float32)
    scores = jnp.sum(values * folded_weight, axis=-1)
    scores *= jax.lax.rsqrt(jnp.mean(values * values, axis=-1) + eps)
    mixed = jnp.sum(jax.nn.softmax(scores)[:, None] * values, axis=0)
    return jax.lax.optimization_barrier(mixed).astype(prefix.dtype)


def tp_sum(value, axis_name="tp"):
    """Keep the reference collective in FP32 before any BF16 output cast."""
    return jax.lax.optimization_barrier(jax.lax.psum(value, axis_name))

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
"""Muse Spark Open 816B A42B configuration and constants for the Pallas decode
megakernel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

F32 = jnp.float32
HIGHEST = jax.lax.Precision.HIGHEST
TP_AXIS_NAME = "tp"

PAGE_TOKENS = 2048
SW_RING = 2048
SUBPAGE_TOKENS = 128
KV_GROUP = (
    PAGE_TOKENS // SUBPAGE_TOKENS
)  # 16 subpages of 128 tokens per 2048 token superpage
TOPK_OUT = 64

NVFP4_LUT = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float32,
)


def dequantize_nvfp4(packed, block_scale, global_scale):
    """Linear layout E2M1 * block 16 E4M3 * global scale to fp32 (numpy)."""
    import ml_dtypes

    packed = np.asarray(packed, dtype=np.uint8)
    if packed.shape[-1] != block_scale.shape[-1] * 8:
        raise ValueError("NVFP4 block scales must cover 16 unpacked values")
    if packed.shape[:-1] != block_scale.shape[:-1]:
        raise ValueError("NVFP4 scale and weight row shapes disagree")
    if block_scale.dtype == np.uint8:
        block_scale = block_scale.view(ml_dtypes.float8_e4m3fn)
    low = NVFP4_LUT[packed & 15]
    high = NVFP4_LUT[packed >> 4]
    values = np.stack((low, high), axis=-1).reshape(*packed.shape[:-1], -1)
    scales = block_scale.astype(np.float32) * np.asarray(global_scale, dtype=np.float32)
    return (values.reshape(*block_scale.shape, 16) * scales[..., None]).reshape(
        values.shape
    )


def residual_coefficients(gate, temperature: float):
    """Compute (alpha, beta) residual gate vectors matching MuseSparkResidualGate."""
    w = np.asarray(gate, dtype=np.float32) / temperature

    def sigmoid(v):
        return 1.0 / (1.0 + np.exp(-v))

    beta = sigmoid(w)
    alpha = np.sqrt(np.clip(sigmoid(-w) * (1.0 + beta), 1e-3, None))
    return alpha.astype(np.float32), beta.astype(np.float32)


@dataclass(frozen=True)
class MuseMegakernelConfig:
    dim: int = 8192
    layers: int = 62
    heads: int = 128
    kv_heads: int = 16
    head_dim: int = 64
    vocab: int = 202048
    eps: float = 1e-5
    post_eps: float = 1e-8
    moe_hidden: int = 4096
    moe_intermediate: int = 4096
    experts: int = 256
    top_k: int = 8
    qk_scale_factor: float = 5.4730064863875
    output_multiplier: float = 0.17677669529663687
    soft_cap: float = 20.0
    gate_temperature: float = 0.3
    sliding_window: int = 2048
    rope_theta: float = 500000.0
    router_eps: float = 1e-15
    full_attention: tuple[int, ...] = tuple(range(1, 62, 4))
    bf16_expert_layers: tuple[int, ...] = (0, 61)
    bos: int = 200000
    eos: tuple[int, ...] = (200001, 200008)
    pad: int = 200018

    @classmethod
    def from_checkpoint(cls, directory: str | Path) -> MuseMegakernelConfig:
        raw = json.loads((Path(directory) / "config.json").read_text())
        c = raw.get("text_config", raw)
        layers = int(c["num_hidden_layers"])
        full = tuple(
            i for i in range(layers) if c["layer_types"][i] == "full_attention"
        )
        # The kernel, the KV ring and the vLLM pool binding place a global layer at
        # every i % 4 == 1 and give the others a SW_RING-token sliding window.
        supported_full = tuple(i for i in range(layers) if i % 4 == 1)
        if full != supported_full:
            raise NotImplementedError(
                "Muse Spark megakernel supports full_attention layers at i % 4 == 1 "
                f"only, got layer_types with full_attention at {full}."
            )
        if int(c["sliding_window"]) != SW_RING:
            raise NotImplementedError(
                f"Muse Spark megakernel supports sliding_window={SW_RING} only, "
                f"got {c['sliding_window']}."
            )
        eos_raw = c.get("eos_token_id", raw.get("eos_token_id"))
        if eos_raw is None:
            eos = cls.eos
        elif isinstance(eos_raw, int):
            eos = (eos_raw,)
        else:
            eos = tuple(int(x) for x in eos_raw)
        bf16_layers = (0, layers - 1) if layers >= 2 else (0,)
        # Check quantization_config ignore list if present
        qcfg = c.get("quantization_config") or raw.get("quantization_config") or {}
        ignore = qcfg.get("ignore") or []
        detected_bf16 = []
        for i in range(layers):
            if any(f"layers.{i}.mlp.experts" in item for item in ignore):
                detected_bf16.append(i)
        if detected_bf16:
            bf16_layers = tuple(detected_bf16)
        return cls(
            dim=int(c["hidden_size"]),
            layers=layers,
            heads=int(c["num_attention_heads"]),
            kv_heads=int(c["num_key_value_heads"]),
            head_dim=int(c["head_dim"]),
            vocab=int(c["vocab_size"]),
            eps=float(c["rms_norm_eps"]),
            post_eps=float(c["post_norm_eps"]),
            moe_hidden=int(c["moe_hidden_size"]),
            moe_intermediate=int(c["moe_intermediate_size"]),
            experts=int(c["num_local_experts"]),
            top_k=int(c["num_experts_per_tok"]),
            qk_scale_factor=float(c["qk_scale_factor"]),
            output_multiplier=float(c["output_multiplier"]),
            soft_cap=float(c["final_logit_softcapping"]),
            gate_temperature=float(c["residual_gate_temperature"]),
            sliding_window=int(c["sliding_window"]),
            rope_theta=float(c["rope_parameters"]["rope_theta"]),
            full_attention=full,
            bf16_expert_layers=bf16_layers,
            bos=int(c.get("bos_token_id", 200000)),
            eos=eos,
            pad=int(c.get("pad_token_id", 200018)),
        )

    @property
    def num_sw_layers(self) -> int:
        return sum(1 for i in range(self.layers) if (i % 4) != 1)

    @property
    def num_gl_layers(self) -> int:
        return max(1, sum(1 for i in range(self.layers) if (i % 4) == 1))

    @property
    def scale(self) -> float:
        return self.head_dim**-0.5 * self.qk_scale_factor

    @property
    def attn_scale(self) -> float:
        return self.head_dim**-0.5

    def use_rope(self, index: int) -> bool:
        return index not in self.full_attention

    def use_window(self, index: int) -> bool:
        return index not in self.full_attention


Config = MuseMegakernelConfig

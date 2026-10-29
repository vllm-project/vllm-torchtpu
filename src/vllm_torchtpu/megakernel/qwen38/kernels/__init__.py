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
"""Architecture and checkpoint quantization for Qwen3.8-2.4T-A95B decode."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

FP8_MODEL_ID = "Qwen/Qwen3.8-2.4T-A95B-FP8"
FP8_MODEL_REVISION = "d2dc35658bcf77e66643428cb52e774cc3b5bd29"
MODEL_ID = "nvidia/Qwen3.8-2.4T-A95B-NVFP4"
MODEL_REVISION = "74eb734297318ecf7c48016a4ac8ce1b47cc74a9"


@dataclass(frozen=True)
class Config:
    # Preserve the original constructor for synthetic FP8 fixtures. Production
    # always derives the format from the selected checkpoint's configuration.
    quantization: str = "fp8"
    dim: int = 8192
    intermediate: int = 2048
    moe_intermediate: int = 2048
    experts: int = 512
    top_k: int = 10
    layers: int = 92
    heads: int = 64
    kv_heads: int = 4
    head_dim: int = 256
    rotary_dim: int = 64
    rope_theta: float = 10000000.0
    k_heads: int = 16
    k_dim: int = 128
    v_heads: int = 128
    v_dim: int = 128
    conv_size: int = 4
    vocab: int = 248320
    eps: float = 1e-6
    full_attention: tuple[int, ...] = tuple(range(3, 92, 4))

    def __post_init__(self):
        if self.quantization not in ("fp8", "nvfp4"):
            raise ValueError("quantization must be fp8 or nvfp4")
        for name in (
            "dim",
            "intermediate",
            "moe_intermediate",
            "experts",
            "top_k",
            "layers",
            "heads",
            "kv_heads",
            "head_dim",
            "k_heads",
            "k_dim",
            "v_heads",
            "v_dim",
            "conv_size",
            "vocab",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.heads % self.kv_heads or self.v_heads % self.k_heads:
            raise ValueError("Query/value heads must divide into complete key-head groups")
        if not 0 < self.rotary_dim <= self.head_dim or self.rotary_dim % 2:
            raise ValueError("rotary_dim must be positive, even, and at most head_dim")
        if self.top_k > self.experts:
            raise ValueError("top_k exceeds experts")
        if self.eps <= 0 or self.rope_theta <= 0:
            raise ValueError("eps and rope_theta must be positive")
        if tuple(sorted(set(self.full_attention))) != self.full_attention:
            raise ValueError("full_attention must contain unique increasing layer indices")
        if any(i < 0 or i >= self.layers for i in self.full_attention):
            raise ValueError("full_attention contains an invalid layer index")

    @classmethod
    def from_checkpoint(cls, directory):
        """Read the text-only official config; dimensions may be reduced for fixtures."""
        raw = (
            dict(directory)
            if isinstance(directory, Mapping)
            else json.loads((Path(directory) / "config.json").read_text())
        )
        if "text_config" in raw:
            raise ValueError("Expected a top-level Qwen3.8 text config, not text_config")
        required = {
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "model_type": "qwen3_5_moe_text",
            "dtype": "bfloat16",
            "attention_bias": False,
            "attn_output_gate": True,
            "hidden_act": "silu",
            "mamba_ssm_dtype": "float32",
            "tie_word_embeddings": False,
        }
        for key, expected in required.items():
            if raw.get(key) != expected:
                raise ValueError(f"Unsupported {key}={raw.get(key)!r}; expected {expected!r}")
        if raw.get("output_gate_type") not in ("swish", "silu"):
            raise ValueError("GDN output_gate_type must be swish/silu")
        if raw.get("layer_scale", False) or not raw.get("norm_topk_prob", True):
            raise ValueError("Layer scales and unnormalized routing are unsupported")
        quant = raw.get("quantization_config", {})
        if quant.get("quant_method") == "modelopt":
            validate_nvfp4_config(raw)
            quantization = "nvfp4"
        elif (
            quant.get("quant_method") == "fp8"
            and quant.get("activation_scheme") == "dynamic"
            and quant.get("weight_block_size") == [128, 128]
        ):
            quantization = "fp8"
        else:
            raise ValueError(
                "Expected native 128x128 block-scaled FP8 or ModelOpt mixed NVFP4 checkpoint"
            )
        rope = raw["rope_parameters"]
        if rope.get("rope_type") != "default":
            raise ValueError("Only native default RoPE is supported")
        layer_types = tuple(raw["layer_types"])
        if len(layer_types) != raw["num_hidden_layers"] or any(
            kind not in ("linear_attention", "full_attention") for kind in layer_types
        ):
            raise ValueError("layer_types must specify every linear/full attention layer")
        rotary = raw["head_dim"] * rope["partial_rotary_factor"]
        if int(rotary) != rotary:
            raise ValueError("RoPE dimensions must be integral")
        return cls(
            quantization=quantization,
            dim=raw["hidden_size"],
            intermediate=raw["shared_expert_intermediate_size"],
            moe_intermediate=raw["moe_intermediate_size"],
            experts=raw["num_experts"],
            top_k=raw["num_experts_per_tok"],
            layers=raw["num_hidden_layers"],
            heads=raw["num_attention_heads"],
            kv_heads=raw["num_key_value_heads"],
            head_dim=raw["head_dim"],
            rotary_dim=int(rotary),
            rope_theta=rope["rope_theta"],
            k_heads=raw["linear_num_key_heads"],
            k_dim=raw["linear_key_head_dim"],
            v_heads=raw["linear_num_value_heads"],
            v_dim=raw["linear_value_head_dim"],
            conv_size=raw["linear_conv_kernel_dim"],
            vocab=raw["vocab_size"],
            eps=raw["rms_norm_eps"],
            full_attention=tuple(
                i for i, kind in enumerate(layer_types) if kind == "full_attention"
            ),
        )

    @property
    def linear_layers(self):
        return tuple(i for i in range(self.layers) if i not in self.full_attention)

    @property
    def full_layers(self):
        return self.full_attention

    @property
    def key_dim(self):
        return self.k_heads * self.k_dim

    @property
    def value_dim(self):
        return self.v_heads * self.v_dim

    @property
    def conv_dim(self):
        return 2 * self.key_dim + self.value_dim


def validate_nvfp4_config(raw):
    """Reject unsupported mixed-precision maps before interpreting tensor bytes."""
    quant = raw.get("quantization_config", {})
    if quant.get("quant_method") != "modelopt" or quant.get("quant_algo") != "MIXED_PRECISION":
        raise ValueError("Expected ModelOpt MIXED_PRECISION NVFP4")
    expected = {}
    for layer, kind in enumerate(raw["layer_types"]):
        prefix = f"model.layers.{layer}."
        if kind == "linear_attention":
            names = ("conv1d", "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj")
            expected.update(
                {prefix + "linear_attn." + name: {"quant_algo": "FP8"} for name in names}
            )
        elif kind == "full_attention":
            expected.update(
                {
                    prefix + "self_attn." + name: {"quant_algo": "FP8"}
                    for name in ("q_proj", "k_proj", "v_proj", "o_proj")
                }
            )
        else:
            raise ValueError(f"Unsupported layer type {kind!r}")
        expected[prefix + "mlp.experts"] = {"quant_algo": "NVFP4", "group_size": 16}
    if quant.get("quantized_layers") != expected:
        raise ValueError("NVFP4 checkpoint has an unsupported per-layer quantization map")
    covered = set()
    for group in quant.get("config_groups", {}).values():
        targets = group.get("targets", [])
        if covered.intersection(targets) or len(set(targets)) != len(targets):
            raise ValueError("Duplicate quantization targets")
        covered.update(targets)
        for target in targets:
            if target not in expected:
                raise ValueError(f"Unknown quantized target {target}")
            bits = 4 if expected[target]["quant_algo"] == "NVFP4" else 8
            for name in ("weights", "input_activations"):
                scheme = group.get(name, {})
                if (
                    scheme.get("num_bits") != bits
                    or scheme.get("type") != "float"
                    or scheme.get("dynamic") is not False
                    or (bits == 4 and scheme.get("group_size") != 16)
                ):
                    raise ValueError(f"Unsupported {name} quantization for {target}")
    if covered != set(expected):
        raise ValueError("Quantization groups must cover the complete per-layer map")

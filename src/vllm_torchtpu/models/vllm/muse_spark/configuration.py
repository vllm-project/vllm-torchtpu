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
"""Muse Spark Open configuration classes.

Vendored from transformers main (``configuration_muse_spark.py``, Apache 2.0).
The serving image ships transformers 5.14, which has no ``muse_spark`` and no
``SubConfigSpec`` / ``@strict``, so these are plain ``PretrainedConfig``
subclasses with the same fields, defaults and post init logic.
"""

from __future__ import annotations

import math
from typing import Any, cast

from transformers.configuration_utils import PretrainedConfig


def _default_text_layer_types(num_hidden_layers: int) -> list[str]:
    # NoPE full attention every 4th layer counted backward from the last.
    return [
        "full_attention"
        if (num_hidden_layers - 1 - i) % 4 == 0
        else "sliding_attention"
        for i in range(num_hidden_layers)
    ]


def _default_vision_layer_types(num_hidden_layers: int) -> list[str]:
    return [
        "full_attention"
        if (i + 1) % 4 == 0 or i == num_hidden_layers - 1
        else "window_attention"
        for i in range(num_hidden_layers)
    ]


class MuseSparkVisionConfig(PretrainedConfig):
    model_type = "muse_spark_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        patch_size: int = 14,
        pos_emb_height: int = 32,
        pos_emb_width: int = 32,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 50,
        hidden_size: int = 1536,
        intermediate_size: int = 8960,
        hidden_act: str = "gelu",
        rope_parameters: dict | None = None,
        patch_temporal: int = 2,
        merge_size: int = 2,
        max_position_embeddings: int = 32 * 32,
        layer_norm_eps: float = 1e-5,
        layer_types: list[str] | None = None,
        **kwargs,
    ) -> None:
        # transformers 5.14 validates layer_types against a fixed list that
        # lacks "window_attention", so set it after the base init runs.
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.pos_emb_height = pos_emb_height
        self.pos_emb_width = pos_emb_width
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rope_parameters = (
            dict(rope_parameters)
            if rope_parameters is not None
            else {"rope_type": "axial", "rope_theta": 10000.0}
        )
        self.patch_temporal = patch_temporal
        self.merge_size = merge_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.layer_types = (
            list(layer_types)
            if layer_types is not None
            else _default_vision_layer_types(num_hidden_layers)
        )

        self._check_layer_types()

    def _check_layer_types(self) -> None:
        allowed = {"full_attention", "window_attention"}
        if any(t not in allowed for t in self.layer_types):
            raise ValueError(
                f"vision layer_types entries must be in {sorted(allowed)}, "
                f"got {self.layer_types}"
            )
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"vision layer_types has {len(self.layer_types)} entries, "
                f"expected {self.num_hidden_layers}"
            )


class MuseSparkTextConfig(PretrainedConfig):
    model_type = "muse_spark_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 202_048,
        hidden_size: int = 8192,
        intermediate_size: int = 16384,
        num_hidden_layers: int = 62,
        num_attention_heads: int = 128,
        num_key_value_heads: int = 16,
        head_dim: int = 64,
        max_position_embeddings: int = 1_048_576,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = 200_001,
        bos_token_id: int | None = 200_000,
        tie_word_embeddings: bool = False,
        rope_parameters: dict | None = None,
        rope_theta: float | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window: int | None = 2048,
        layer_types: list[str] | None = None,
        final_logit_softcapping: float | None = 20.0,
        qk_scale_factor: float = 5.4730064863875,
        output_multiplier: float = 0.1767766952966369,
        post_norm_eps: float = 1e-8,
        layer_rope_theta: list[float] | None = None,
        hidden_act: str = "silu",
        moe_intermediate_size: int = 4096,
        num_local_experts: int = 256,
        num_experts_per_tok: int = 8,
        routed_scaling_factor: float = 1.0,
        moe_hidden_size: int = 4096,
        mlp_layer_types: list[str] | None = None,
        residual_gate_temperature: float = 0.3,
        residual_dtype: str = "fp32",
        output_router_logits: bool = False,
        **kwargs,
    ) -> None:
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) is not a multiple of "
                f"num_attention_heads ({num_attention_heads})"
            )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.qk_scale_factor = qk_scale_factor
        self.output_multiplier = output_multiplier
        self.post_norm_eps = post_norm_eps
        self.hidden_act = hidden_act
        self.moe_intermediate_size = moe_intermediate_size
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.moe_hidden_size = moe_hidden_size
        self.residual_gate_temperature = residual_gate_temperature
        self.residual_dtype = residual_dtype
        self.output_router_logits = output_router_logits

        if rope_parameters is None:
            theta = rope_theta if rope_theta is not None else 500_000.0
            rope_parameters = {"rope_type": "default", "rope_theta": theta}
        self.rope_parameters = dict(rope_parameters)
        # Some vLLM code paths read a flat rope_theta.
        self.rope_theta = self.rope_parameters.get("rope_theta", 500_000.0)

        self.mlp_layer_types = (
            list(mlp_layer_types)
            if mlp_layer_types is not None
            else ["sparse"] * num_hidden_layers
        )
        self.layer_types = (
            list(layer_types)
            if layer_types is not None
            else _default_text_layer_types(num_hidden_layers)
        )
        if layer_rope_theta is None:
            layer_rope_theta = [
                0 if (num_hidden_layers - 1 - i) % 4 == 0 else self.rope_theta
                for i in range(num_hidden_layers)
            ]
        self.layer_rope_theta = list(layer_rope_theta)
        for name in ("layer_types", "mlp_layer_types", "layer_rope_theta"):
            if len(getattr(self, name)) != num_hidden_layers:
                raise ValueError(
                    f"text {name} has {len(getattr(self, name))} entries, "
                    f"expected {num_hidden_layers}"
                )

        kwargs.update(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
        )
        super().__init__(**kwargs)


class MuseSparkAudioConfig(PretrainedConfig):
    model_type = "muse_spark_audio"
    base_config_key = "audio_config"

    def __init__(
        self,
        patch_sizes: list[int] | tuple[int, ...] = (16, 6, 5, 4),
        hidden_sizes: list[int] | tuple[int, ...] = (256, 512, 1024, 2048),
        depths: list[int] | tuple[int, ...] = (2, 2, 2, 2),
        num_hidden_layers: int = 4,
        mlp_ratio: float = 4.0,
        conv_kernel_size: int = 7,
        hidden_act: str = "gelu",
        rms_norm_eps: float = 1e-6,
        out_hidden_size: int = 8192,
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None:
        self.patch_sizes = list(patch_sizes)
        self.hidden_sizes = list(hidden_sizes)
        self.depths = list(depths)
        self.num_hidden_layers = num_hidden_layers
        self.mlp_ratio = mlp_ratio
        self.conv_kernel_size = conv_kernel_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.out_hidden_size = out_hidden_size
        self.initializer_range = initializer_range
        if not (
            len(self.patch_sizes)
            == len(self.hidden_sizes)
            == len(self.depths)
            == self.num_hidden_layers
        ):
            raise ValueError(
                "patch_sizes, hidden_sizes and depths must have one entry per "
                f"layer, got {len(self.patch_sizes)}, {len(self.hidden_sizes)} "
                f"and {len(self.depths)} for num_hidden_layers="
                f"{self.num_hidden_layers}"
            )
        super().__init__(**kwargs)

    @property
    def samples_per_token(self) -> int:
        """Waveform samples collapsed into one output token (1920 at 24 kHz)."""
        return math.prod(self.patch_sizes)


def _build_sub_config[T: PretrainedConfig](cls: type[T], value: Any) -> T:
    if value is None:
        return cls()
    if isinstance(value, dict):
        value = {k: v for k, v in value.items() if k != "model_type"}
        return cls(**value)
    return value


class MuseSparkConfig(PretrainedConfig):
    model_type = "muse_spark"

    def __init__(
        self,
        text_config: dict | MuseSparkTextConfig | None = None,
        vision_config: dict | MuseSparkVisionConfig | None = None,
        audio_config: dict | MuseSparkAudioConfig | None = None,
        image_token_id: int = 200092,
        video_token_id: int = 200091,
        audio_token_id: int = 200038,
        out_hidden_size: int = 6144,
        projector_hidden_size: int = 4096,
        projector_hidden_act: str = "gelu",
        **kwargs,
    ) -> None:
        self.text_config = _build_sub_config(MuseSparkTextConfig, text_config)
        self.vision_config = _build_sub_config(MuseSparkVisionConfig, vision_config)
        self.audio_config = _build_sub_config(MuseSparkAudioConfig, audio_config)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        self.out_hidden_size = out_hidden_size
        self.projector_hidden_size = projector_hidden_size
        self.projector_hidden_act = projector_hidden_act
        if self.audio_config.out_hidden_size != self.text_config.hidden_size:
            raise ValueError(
                f"audio_config.out_hidden_size ({self.audio_config.out_hidden_size})"
                f" must match text_config.hidden_size "
                f"({self.text_config.hidden_size})"
            )
        merged = self.vision_config.hidden_size * self.vision_config.merge_size**2
        if merged != out_hidden_size:
            raise ValueError(
                f"out_hidden_size ({out_hidden_size}) must equal the pixel "
                f"shuffle width vision hidden_size * merge_size**2 ({merged})"
            )
        super().__init__(**kwargs)

    def get_text_config(self, *args, **kwargs) -> MuseSparkTextConfig:
        return self.text_config


cast(Any, MuseSparkConfig).sub_configs = {
    "text_config": MuseSparkTextConfig,
    "vision_config": MuseSparkVisionConfig,
    "audio_config": MuseSparkAudioConfig,
}


class MuseSparkDSparkConfig(PretrainedConfig):
    model_type = "muse_spark_dspark"
    base_config_key = "text_config"
    # Target layers (0-indexed) whose outputs feed the released DSpark draft.
    DEFAULT_TARGET_NUM_HIDDEN_LAYERS = 62
    DEFAULT_TARGET_LAYER_IDS = (13, 29, 45, 57, 61)

    def __init__(
        self,
        vocab_size: int = 202048,
        hidden_size: int = 8192,
        intermediate_size: int = 16384,
        intermediate_size_mlp: int = 16384,
        num_hidden_layers: int = 5,
        num_attention_heads: int = 128,
        num_key_value_heads: int = 16,
        head_dim: int = 64,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        rms_post_norm_eps: float = 1e-8,
        use_cache: bool = True,
        rope_theta: float = 500000.0,
        rope_parameters: dict | None = None,
        sliding_window: int = 2048,
        use_sliding_window: bool = True,
        layer_types: list[str] | None = None,
        target_hidden_size: int = 8192,
        num_target_layers: int = 5,
        target_num_hidden_layers: int = 62,
        target_layer_ids: list[int] | None = None,
        draft_vocab_size: int = 202048,
        mask_token_id: int = 200018,
        markov_rank: int = 512,
        markov_head_type: str = "vanilla",
        sample_from_anchor: bool = True,
        block_size: int = 7,
        output_soft_cap_temp: float = 20.0,
        use_attn_o_gate: bool = True,
        use_attn_o_norm: bool = True,
        use_ffn_learnable_scales: bool = True,
        use_qk_norm: bool = True,
        use_residual_gate: bool = True,
        norm_zero_centered_gamma: bool = True,
        output_norm_gain_center_type: str = "zero",
        post_norm_gain_center_type: str = "one",
        residual_dtype: str = "fp32",
        dflash_config: dict | None = None,
        tie_word_embeddings: bool = False,
        bos_token_id: int = 200000,
        eos_token_id: int | list[int] | None = None,
        pad_token_id: int = 200018,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.intermediate_size_mlp = intermediate_size_mlp
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rms_post_norm_eps = rms_post_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_parameters = rope_parameters or {"rope_theta": rope_theta}
        self.sliding_window = sliding_window
        self.use_sliding_window = use_sliding_window
        self.layer_types = layer_types or ["sliding_attention"] * num_hidden_layers
        self.target_hidden_size = target_hidden_size
        self.num_target_layers = num_target_layers
        self.target_num_hidden_layers = target_num_hidden_layers
        self.target_layer_ids = target_layer_ids or list(self.DEFAULT_TARGET_LAYER_IDS)
        self.draft_vocab_size = draft_vocab_size
        self.mask_token_id = mask_token_id
        self.markov_rank = markov_rank
        self.markov_head_type = markov_head_type
        self.sample_from_anchor = sample_from_anchor
        self.block_size = block_size
        self.output_soft_cap_temp = output_soft_cap_temp
        self.use_attn_o_gate = use_attn_o_gate
        self.use_attn_o_norm = use_attn_o_norm
        self.use_ffn_learnable_scales = use_ffn_learnable_scales
        self.use_qk_norm = use_qk_norm
        self.use_residual_gate = use_residual_gate
        self.residual_gate_temperature = float(
            kwargs.pop("residual_gate_temperature", 0.3)
        )
        self.norm_zero_centered_gamma = norm_zero_centered_gamma
        self.output_norm_gain_center_type = output_norm_gain_center_type
        self.post_norm_gain_center_type = post_norm_gain_center_type
        self.residual_dtype = residual_dtype
        self.dflash_config = dflash_config or {
            "causal": True,
            "mask_token_id": mask_token_id,
            "target_layer_ids": self.target_layer_ids,
            "use_aux_hidden_state": True,
        }
        self.tie_word_embeddings = tie_word_embeddings
        self.qk_scale_factor = float(kwargs.pop("qk_scale_factor", 5.4730064863875))
        self.final_logit_scale = float(
            kwargs.pop("final_logit_scale", 0.17677669529663687)
        )
        self.final_logit_softcapping = (
            float(output_soft_cap_temp) if output_soft_cap_temp else None
        )
        kwargs.update(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
        )
        super().__init__(**kwargs)

    def get_text_config(self, *args, **kwargs) -> MuseSparkDSparkConfig:
        return self


MUSE_SPARK_CONFIGS: dict[str, type[PretrainedConfig]] = {
    "muse_spark": MuseSparkConfig,
    "muse_spark_text": MuseSparkTextConfig,
    "muse_spark_vision": MuseSparkVisionConfig,
    "muse_spark_audio": MuseSparkAudioConfig,
    "muse_spark_dspark": MuseSparkDSparkConfig,
}


def register_muse_spark_configs() -> None:
    """Register the vendored configs with transformers AutoConfig and vLLM.

    vLLM's default arch convertor reads head_dim, KV heads, expert counts and
    the sliding window from ``get_text_config()``, so no custom convertor is
    needed.
    """
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    for model_type, cls in MUSE_SPARK_CONFIGS.items():
        CONFIG_MAPPING._extra_content.setdefault(model_type, cls)
        _CONFIG_REGISTRY[model_type] = cls


register_muse_spark_configs()

__all__ = [
    "MuseSparkAudioConfig",
    "MuseSparkConfig",
    "MuseSparkDSparkConfig",
    "MuseSparkTextConfig",
    "MuseSparkVisionConfig",
    "MUSE_SPARK_CONFIGS",
    "register_muse_spark_configs",
]

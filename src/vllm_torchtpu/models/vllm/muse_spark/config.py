# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from typing import Any

from transformers.configuration_utils import PretrainedConfig


class MuseSparkTextConfig(PretrainedConfig):
    model_type = "muse_spark_text"
    base_config_key = "text_config"
    has_no_defaults_at_init = True

    def __init__(self, use_cache: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.use_cache = use_cache
        if self.hidden_act != "silu" or self.attention_bias:
            raise ValueError("Muse Spark requires SiLU and bias-free attention")
        if self.rope_parameters["rope_type"] != "default":
            raise ValueError("Unsupported Muse Spark text RoPE type")
        if self.num_shared_experts:
            raise ValueError("Muse Spark shared experts are not supported")
        for field, choices in (
            ("layer_types", {"full_attention", "sliding_attention"}),
            ("mlp_layer_types", {"sparse", "dense"}),
        ):
            values = getattr(self, field)
            if len(values) != self.num_hidden_layers or not set(values) <= choices:
                raise ValueError(f"Invalid Muse Spark {field}")
        if len(self.layer_rope_theta) != self.num_hidden_layers:
            raise ValueError("Muse Spark requires one RoPE theta per layer")


class MuseSparkVisionConfig(PretrainedConfig):
    model_type = "muse_spark_vision"
    base_config_key = "vision_config"
    has_no_defaults_at_init = True

    def __init__(self, layer_types: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.layer_types = layer_types
        if self.hidden_act != "gelu":
            raise ValueError("Muse Spark requires GELU vision activations")
        if self.rope_parameters["rope_type"] != "axial":
            raise ValueError("Unsupported Muse Spark vision RoPE type")
        if len(self.layer_types) != self.num_hidden_layers or not set(
            self.layer_types
        ) <= {"full_attention", "window_attention"}:
            raise ValueError("Invalid Muse Spark vision attention types")


class MuseSparkAudioConfig(PretrainedConfig):
    model_type = "muse_spark_audio"
    base_config_key = "audio_config"
    has_no_defaults_at_init = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.hidden_act != "gelu":
            raise ValueError("Muse Spark requires GELU audio activations")
        if not (
            len(self.patch_sizes)
            == len(self.depths)
            == len(self.hidden_sizes)
            == self.num_hidden_layers
        ):
            raise ValueError("Invalid Muse Spark audio stages")


class MuseSparkConfig(PretrainedConfig):
    model_type = "muse_spark"
    has_no_defaults_at_init = True
    sub_configs = {
        "text_config": MuseSparkTextConfig,
        "vision_config": MuseSparkVisionConfig,
        "audio_config": MuseSparkAudioConfig,
    }

    def __init__(
        self,
        text_config: dict[str, Any] | MuseSparkTextConfig | None = None,
        vision_config: dict[str, Any] | MuseSparkVisionConfig | None = None,
        audio_config: dict[str, Any] | MuseSparkAudioConfig | None = None,
        projector_hidden_size: int | None = None,
        out_hidden_size: int | None = None,
        projector_hidden_act: str = "gelu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        for name, value in (
            ("text_config", text_config),
            ("vision_config", vision_config),
            ("audio_config", audio_config),
        ):
            config_cls = self.sub_configs[name]
            if isinstance(value, dict):
                setattr(self, name, config_cls(**deepcopy(value)))
            elif isinstance(value, config_cls):
                setattr(self, name, value)
            else:
                raise ValueError(f"Muse Spark requires an HF {name}")
        if projector_hidden_size is None or out_hidden_size is None:
            raise ValueError("Muse Spark requires vision projector dimensions")
        if projector_hidden_act != "gelu":
            raise ValueError("Muse Spark requires a GELU vision projector")
        self.projector_hidden_size = projector_hidden_size
        self.out_hidden_size = out_hidden_size
        self.projector_hidden_act = projector_hidden_act
        if (
            out_hidden_size
            != self.vision_config.hidden_size * self.vision_config.merge_size**2
        ):
            raise ValueError("Muse Spark vision output must match patch merging")
        if self.audio_config.out_hidden_size != self.text_config.hidden_size:
            raise ValueError("Muse Spark audio output must match the text hidden size")
        self.architectures = ["MuseSparkForConditionalGeneration"]

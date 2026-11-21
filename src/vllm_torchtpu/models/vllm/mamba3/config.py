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
"""Lightweight HF/vLLM configuration and `--config-format mamba3` registration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from transformers import AutoConfig, PretrainedConfig
from vllm.model_executor.models.config import MambaModelConfig
from vllm.transformers_utils import config as vllm_tf_config
from vllm.transformers_utils.config import (
    HFConfigParser,
    _maybe_remap_hf_config_attrs,
    register_config_parser,
)
from vllm.transformers_utils.utils import without_trust_remote_code

from vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils import (
    validate_tpu_mamba_config,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_MAMBA3_PARSER_REGISTERED = False


def _deep_merge_config_dict(
    base: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Recursively merge nested configuration dicts (e.g. ``ssm_cfg``)."""
    merged = dict(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_config_dict(existing, value)
        else:
            merged[key] = value
    return merged


class Mamba3Config(PretrainedConfig):
    """Normalized configuration class for Mamba-3 models (`state-spaces/mamba3-*`)."""

    model_type = "mamba3"

    def __init__(
        self,
        d_model: int | None = None,
        hidden_size: int = 2048,
        d_intermediate: int = 0,
        n_layer: int | None = None,
        num_hidden_layers: int = 24,
        vocab_size: int = 128256,
        ssm_cfg: dict[str, Any] | None = None,
        rms_norm: bool = True,
        norm_epsilon: float | None = None,
        layer_norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = True,
        tie_embeddings: bool | None = None,
        tie_word_embeddings: bool = True,
        pad_vocab_size_multiple: int = 16,
        **kwargs: Any,
    ) -> None:
        ssm_cfg_dict = dict(ssm_cfg) if isinstance(ssm_cfg, dict) else {}

        resolved_hidden = int(d_model if d_model is not None else hidden_size)
        resolved_layers = int(n_layer if n_layer is not None else num_hidden_layers)

        raw_vocab_size = int(vocab_size)
        org_vocab_arg = kwargs.pop("org_vocab_size", None)
        self.org_vocab_size = (
            int(org_vocab_arg) if org_vocab_arg is not None else raw_vocab_size
        )
        pad_mult = int(pad_vocab_size_multiple)
        padded_vocab_size = raw_vocab_size
        if pad_mult > 1 and padded_vocab_size % pad_mult != 0:
            padded_vocab_size += pad_mult - (padded_vocab_size % pad_mult)

        self.d_model = resolved_hidden
        self.hidden_size = resolved_hidden
        self.d_intermediate = int(d_intermediate)
        self.n_layer = resolved_layers
        self.num_hidden_layers = resolved_layers
        self.vocab_size = padded_vocab_size
        self.pad_vocab_size_multiple = pad_mult
        self.ssm_cfg = ssm_cfg_dict

        self.state_size = int(
            ssm_cfg_dict.get(
                "d_state",
                kwargs.pop("state_size", kwargs.pop("d_state", 128)),
            )
        )
        self.expand = int(ssm_cfg_dict.get("expand", kwargs.pop("expand", 2)))
        self.head_dim = int(
            ssm_cfg_dict.get(
                "headdim",
                ssm_cfg_dict.get("head_dim", kwargs.pop("head_dim", 64)),
            )
        )
        inter_arg = kwargs.pop("intermediate_size", None)
        self.intermediate_size = (
            int(inter_arg) if inter_arg else (self.expand * self.hidden_size)
        )
        heads_arg = kwargs.pop("num_heads", None)
        self.num_heads = (
            int(heads_arg) if heads_arg else (self.intermediate_size // self.head_dim)
        )
        self.n_groups = int(
            ssm_cfg_dict.get(
                "ngroups",
                ssm_cfg_dict.get("n_groups", kwargs.pop("n_groups", 1)),
            )
        )
        self.is_mimo = bool(ssm_cfg_dict.get("is_mimo", kwargs.pop("is_mimo", False)))
        raw_rank = int(
            ssm_cfg_dict.get(
                "mimo_rank",
                kwargs.pop("mimo_rank", 4 if self.is_mimo else 1),
            )
        )
        self.mimo_rank = raw_rank if self.is_mimo else 1
        self.rope_fraction = float(
            ssm_cfg_dict.get("rope_fraction", kwargs.pop("rope_fraction", 0.5))
        )
        self.A_floor = float(ssm_cfg_dict.get("A_floor", kwargs.pop("A_floor", 1e-4)))
        self.use_rms_norm = bool(
            ssm_cfg_dict.get("is_outproj_norm", kwargs.pop("use_rms_norm", False))
        )
        self.use_bias = bool(ssm_cfg_dict.get("bias", kwargs.pop("use_bias", False)))

        resolved_eps = float(
            kwargs.pop(
                "rms_norm_eps",
                norm_epsilon if norm_epsilon is not None else layer_norm_epsilon,
            )
        )
        self.rms_norm = bool(rms_norm)
        self.norm_epsilon = resolved_eps
        self.layer_norm_epsilon = resolved_eps
        self.rms_norm_eps = resolved_eps
        self.residual_in_fp32 = bool(residual_in_fp32)

        resolved_tie = bool(
            tie_embeddings if tie_embeddings is not None else tie_word_embeddings
        )
        self.tie_embeddings = resolved_tie

        if not kwargs.get("architectures"):
            kwargs["architectures"] = ["Mamba3ForCausalLM"]
        kwargs["tie_word_embeddings"] = resolved_tie
        super().__init__(**kwargs)

    def _sync_from_ssm_cfg(self, ssm_overrides: dict[str, Any]) -> None:
        """Re-compute derived Mamba-3 attributes for keys updated in ``ssm_cfg``."""
        merged_ssm = self.ssm_cfg
        if "d_state" in ssm_overrides:
            super().__setattr__("state_size", int(merged_ssm["d_state"]))
        if "expand" in ssm_overrides:
            new_expand = int(merged_ssm["expand"])
            super().__setattr__("expand", new_expand)
            super().__setattr__("intermediate_size", new_expand * self.hidden_size)
            super().__setattr__("num_heads", self.intermediate_size // self.head_dim)
        if "headdim" in ssm_overrides or "head_dim" in ssm_overrides:
            new_hd = int(
                merged_ssm.get("headdim", merged_ssm.get("head_dim", self.head_dim))
            )
            super().__setattr__("head_dim", new_hd)
            super().__setattr__("num_heads", self.intermediate_size // new_hd)
        if "ngroups" in ssm_overrides or "n_groups" in ssm_overrides:
            super().__setattr__(
                "n_groups",
                int(
                    merged_ssm.get("ngroups", merged_ssm.get("n_groups", self.n_groups))
                ),
            )
        if "is_mimo" in ssm_overrides or "mimo_rank" in ssm_overrides:
            new_is_mimo = bool(merged_ssm.get("is_mimo", self.is_mimo))
            raw_rank = int(merged_ssm.get("mimo_rank", 4 if new_is_mimo else 1))
            super().__setattr__("is_mimo", new_is_mimo)
            super().__setattr__("mimo_rank", raw_rank if new_is_mimo else 1)
        if "rope_fraction" in ssm_overrides:
            super().__setattr__("rope_fraction", float(merged_ssm["rope_fraction"]))
        if "A_floor" in ssm_overrides:
            super().__setattr__("A_floor", float(merged_ssm["A_floor"]))
        if "is_outproj_norm" in ssm_overrides:
            super().__setattr__("use_rms_norm", bool(merged_ssm["is_outproj_norm"]))
        if "bias" in ssm_overrides:
            super().__setattr__("use_bias", bool(merged_ssm["bias"]))

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "ssm_cfg" and isinstance(value, dict) and "ssm_cfg" in self.__dict__:
            existing_ssm = self.__dict__["ssm_cfg"]
            if isinstance(existing_ssm, dict):
                merged_ssm = _deep_merge_config_dict(existing_ssm, value)
                super().__setattr__("ssm_cfg", merged_ssm)
                self._sync_from_ssm_cfg(value)
                return
        if "_commit_hash" in self.__dict__ and value is not None:
            if key in ("d_model", "hidden_size"):
                resolved_hidden = int(value)
                super().__setattr__("d_model", resolved_hidden)
                super().__setattr__("hidden_size", resolved_hidden)
                new_inter = self.expand * resolved_hidden
                super().__setattr__("intermediate_size", new_inter)
                super().__setattr__("num_heads", new_inter // self.head_dim)
                return
            if key in ("n_layer", "num_hidden_layers"):
                resolved_layers = int(value)
                super().__setattr__("n_layer", resolved_layers)
                super().__setattr__("num_hidden_layers", resolved_layers)
                return
        super().__setattr__(key, value)

    def update(self, config_dict: dict[str, Any]) -> None:
        for key, value in config_dict.items():
            setattr(self, key, value)


def ensure_mamba3_config(config: PretrainedConfig) -> Mamba3Config:
    """Return ``config`` as a normalized ``Mamba3Config``."""
    if isinstance(config, Mamba3Config):
        return config
    return Mamba3Config.from_dict(config.to_dict())


class TPUMamba3Config(MambaModelConfig):
    """Mamba cache initialization and TPU support checks for Mamba3ForCausalLM."""

    @classmethod
    def verify_and_update_config(cls, vllm_config: VllmConfig) -> None:
        super().verify_and_update_config(vllm_config)
        from vllm_torchtpu.layers.adapter.quantization.unquantized import (
            is_unquantized_tpu_quant_config,
        )

        if (
            vllm_config.model_config.quantization is not None
            or not is_unquantized_tpu_quant_config(vllm_config.quant_config)
        ):
            raise NotImplementedError(
                "Quantized weights are not yet supported for Mamba3ForCausalLM on TPU."
            )
        validate_tpu_mamba_config(
            "Mamba3ForCausalLM",
            cache_config=vllm_config.cache_config,
            vllm_config=vllm_config,
        )


class _Mamba3ConfigParser(HFConfigParser):
    """Dedicated `--config-format mamba3` parser for raw `state-spaces/mamba3-*`."""

    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], PretrainedConfig]:
        del trust_remote_code
        local_kwargs = dict(kwargs)
        local_kwargs["local_files_only"] = (
            vllm_tf_config.huggingface_hub.constants.HF_HUB_OFFLINE
        )
        local_kwargs = without_trust_remote_code(local_kwargs)
        hf_overrides = local_kwargs.pop("hf_overrides", None)
        config_dict, _ = PretrainedConfig.get_config_dict(
            model,
            revision=revision,
            code_revision=code_revision,
            **local_kwargs,
        )

        merged = (
            _deep_merge_config_dict(config_dict, hf_overrides)
            if isinstance(hf_overrides, dict)
            else dict(config_dict)
        )
        merged.setdefault("model_type", "mamba3")
        merged.setdefault("architectures", ["Mamba3ForCausalLM"])
        local_kwargs.setdefault("name_or_path", str(model))
        config = Mamba3Config.from_dict(merged, **local_kwargs)
        config = _maybe_remap_hf_config_attrs(config)
        return merged, config


def register_mamba3_config() -> None:
    """Register `Mamba3Config` with `AutoConfig` and `--config-format mamba3`."""
    global _MAMBA3_PARSER_REGISTERED
    AutoConfig.register("mamba3", Mamba3Config, exist_ok=True)
    if _MAMBA3_PARSER_REGISTERED:
        return
    register_config_parser("mamba3")(_Mamba3ConfigParser)
    _MAMBA3_PARSER_REGISTERED = True

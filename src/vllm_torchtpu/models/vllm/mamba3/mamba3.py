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
"""Mamba-3 causal language model wrapper for vLLM-TorchTPU."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import islice
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsAttentionFree,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op import (  # pyrefly: ignore
    MambaMixer3,
)


class Mamba3Config(PretrainedConfig):
    """Configuration class for Mamba-3 models (`state-spaces/mamba3-*`)."""

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
        resolved_hidden = d_model if d_model is not None else hidden_size
        resolved_layers = n_layer if n_layer is not None else num_hidden_layers
        resolved_eps = norm_epsilon if norm_epsilon is not None else layer_norm_epsilon
        resolved_tie = (
            tie_embeddings if tie_embeddings is not None else tie_word_embeddings
        )

        ssm_cfg_dict = dict(ssm_cfg) if ssm_cfg is not None else {}

        if pad_vocab_size_multiple > 1 and vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (
                vocab_size % pad_vocab_size_multiple
            )

        self.d_model = int(resolved_hidden)
        self.hidden_size = int(resolved_hidden)
        self.d_intermediate = int(d_intermediate)
        self.n_layer = int(resolved_layers)
        self.num_hidden_layers = int(resolved_layers)
        self.vocab_size = int(vocab_size)
        self.ssm_cfg = ssm_cfg_dict
        self.rms_norm = bool(rms_norm)
        self.norm_epsilon = float(resolved_eps)
        self.layer_norm_epsilon = float(resolved_eps)
        self.rms_norm_eps = float(kwargs.pop("rms_norm_eps", resolved_eps))
        self.residual_in_fp32 = bool(residual_in_fp32)
        self.tie_embeddings = bool(resolved_tie)
        self.pad_vocab_size_multiple = int(pad_vocab_size_multiple)

        self.state_size = int(
            ssm_cfg_dict.get("d_state", kwargs.pop("state_size", 128))
        )
        self.expand = int(ssm_cfg_dict.get("expand", kwargs.pop("expand", 2)))
        self.head_dim = int(
            ssm_cfg_dict.get(
                "headdim",
                ssm_cfg_dict.get("head_dim", kwargs.pop("head_dim", 64)),
            )
        )
        raw_inter = kwargs.pop("intermediate_size", None)
        self.intermediate_size = (
            int(raw_inter) if raw_inter is not None else self.expand * self.hidden_size
        )
        raw_heads = kwargs.pop("num_heads", None)
        self.num_heads = (
            int(raw_heads)
            if raw_heads is not None
            else self.intermediate_size // self.head_dim
        )
        self.n_groups = int(
            ssm_cfg_dict.get(
                "ngroups",
                ssm_cfg_dict.get("n_groups", kwargs.pop("n_groups", 1)),
            )
        )
        self.is_mimo = bool(ssm_cfg_dict.get("is_mimo", kwargs.pop("is_mimo", False)))
        default_mimo_rank = 4 if self.is_mimo else 1
        raw_mimo_rank = int(
            ssm_cfg_dict.get("mimo_rank", kwargs.pop("mimo_rank", default_mimo_rank))
        )
        self.mimo_rank = raw_mimo_rank if self.is_mimo else 1
        self.rope_fraction = float(
            ssm_cfg_dict.get("rope_fraction", kwargs.pop("rope_fraction", 0.5))
        )
        default_chunk_size = 16 if self.is_mimo else 64
        self.chunk_size = int(
            ssm_cfg_dict.get("chunk_size", kwargs.pop("chunk_size", default_chunk_size))
        )
        self.A_floor = float(ssm_cfg_dict.get("A_floor", kwargs.pop("A_floor", 1e-4)))
        self.use_rms_norm = bool(
            ssm_cfg_dict.get("is_outproj_norm", kwargs.pop("use_rms_norm", False))
        )
        self.use_bias = bool(ssm_cfg_dict.get("bias", kwargs.pop("use_bias", False)))

        if not kwargs.get("architectures"):
            kwargs["architectures"] = ["Mamba3ForCausalLM"]

        kwargs["tie_word_embeddings"] = resolved_tie
        super().__init__(**kwargs)


def _patch_mamba3_transformers_config() -> None:
    """Register `mamba3` with AutoConfig and auto-detect Mamba-3 checkpoints."""
    try:
        AutoConfig.register("mamba3", Mamba3Config, exist_ok=True)
    except Exception:
        CONFIG_MAPPING._extra_content["mamba3"] = Mamba3Config

    orig_get_config_dict = PretrainedConfig.get_config_dict
    if getattr(orig_get_config_dict, "_mamba3_patched", False):
        return

    @classmethod  # type: ignore[misc]
    def _patched_get_config_dict(
        cls: type[PretrainedConfig],
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        config_dict, kwargs = orig_get_config_dict.__func__(
            cls, pretrained_model_name_or_path, **kwargs
        )
        if isinstance(config_dict, dict):
            ssm_cfg = config_dict.get("ssm_cfg")
            if isinstance(ssm_cfg, dict) and ssm_cfg.get("layer") == "Mamba3":
                config_dict.setdefault("model_type", "mamba3")
                config_dict.setdefault("architectures", ["Mamba3ForCausalLM"])
            elif config_dict.get("model_type") == "mamba3":
                config_dict.setdefault("architectures", ["Mamba3ForCausalLM"])
        return config_dict, kwargs

    _patched_get_config_dict.__func__._mamba3_patched = True  # type: ignore[attr-defined]
    PretrainedConfig.get_config_dict = _patched_get_config_dict  # type: ignore[method-assign]


def _extract_mamba3_params(config: Any) -> dict[str, Any]:
    """Normalize Mamba-3 hyperparameters from Mamba3Config or generic HF config."""
    ssm_cfg = getattr(config, "ssm_cfg", None) or {}
    if not isinstance(ssm_cfg, dict):
        ssm_cfg = {}

    hidden_size = int(
        getattr(config, "hidden_size", None) or getattr(config, "d_model", 2048)
    )
    num_hidden_layers = int(
        getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", 24)
    )
    vocab_size = int(getattr(config, "vocab_size", 128256))
    pad_mult = int(getattr(config, "pad_vocab_size_multiple", 1))
    if pad_mult > 1 and vocab_size % pad_mult != 0:
        vocab_size += pad_mult - (vocab_size % pad_mult)

    d_intermediate = int(getattr(config, "d_intermediate", 0))
    state_size = int(
        ssm_cfg.get(
            "d_state",
            getattr(config, "state_size", getattr(config, "d_state", 128)),
        )
    )
    expand = int(ssm_cfg.get("expand", getattr(config, "expand", 2)))
    head_dim = int(
        ssm_cfg.get(
            "headdim",
            ssm_cfg.get("head_dim", getattr(config, "head_dim", 64)),
        )
    )
    intermediate_size = int(
        getattr(config, "intermediate_size", None) or (expand * hidden_size)
    )
    num_heads = int(
        getattr(config, "num_heads", None) or (intermediate_size // head_dim)
    )
    n_groups = int(
        ssm_cfg.get(
            "ngroups",
            ssm_cfg.get("n_groups", getattr(config, "n_groups", 1)),
        )
    )
    is_mimo = bool(ssm_cfg.get("is_mimo", getattr(config, "is_mimo", False)))
    raw_rank = int(
        ssm_cfg.get(
            "mimo_rank",
            getattr(config, "mimo_rank", 4 if is_mimo else 1),
        )
    )
    mimo_rank = raw_rank if is_mimo else 1
    rope_fraction = float(
        ssm_cfg.get("rope_fraction", getattr(config, "rope_fraction", 0.5))
    )
    default_chunk = 16 if is_mimo else 64
    chunk_size = int(
        ssm_cfg.get("chunk_size", getattr(config, "chunk_size", default_chunk))
    )
    A_floor = float(ssm_cfg.get("A_floor", getattr(config, "A_floor", 1e-4)))
    use_rms_norm = bool(
        ssm_cfg.get("is_outproj_norm", getattr(config, "use_rms_norm", False))
    )
    use_bias = bool(ssm_cfg.get("bias", getattr(config, "use_bias", False)))
    rms_norm_eps = float(
        getattr(
            config,
            "rms_norm_eps",
            getattr(
                config,
                "layer_norm_epsilon",
                getattr(config, "norm_epsilon", 1e-5),
            ),
        )
    )
    residual_in_fp32 = bool(getattr(config, "residual_in_fp32", True))
    tie_word_embeddings = bool(
        getattr(
            config,
            "tie_word_embeddings",
            getattr(config, "tie_embeddings", False),
        )
    )

    return {
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "vocab_size": vocab_size,
        "d_intermediate": d_intermediate,
        "state_size": state_size,
        "expand": expand,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "num_heads": num_heads,
        "n_groups": n_groups,
        "is_mimo": is_mimo,
        "mimo_rank": mimo_rank,
        "rope_fraction": rope_fraction,
        "chunk_size": chunk_size,
        "A_floor": A_floor,
        "use_rms_norm": use_rms_norm,
        "use_bias": use_bias,
        "rms_norm_eps": rms_norm_eps,
        "residual_in_fp32": residual_in_fp32,
        "tie_word_embeddings": tie_word_embeddings,
    }


def _override_weight_loader(param: nn.Parameter, loader: Any) -> None:
    if isinstance(param, BasevLLMParameter):
        param.weight_loader = loader
    else:
        if hasattr(param, "weight_loader"):
            delattr(param, "weight_loader")
        set_weight_attrs(param, {"weight_loader": loader})


class Mamba3GatedMLP(nn.Module):
    """SwiGLU GatedMLP layer for Mamba-3 with automatic TPU alignment padding.

    Mamba-3 MIMO 1.5B specifies `d_intermediate=3824` in `config.json`, while
    the checkpoint stores `fc1.weight` with shape `(7680, 2048)` (`2 * 3840`)
    and `fc2.weight` with shape `(2048, 3840)` (padded to a multiple of 32).
    This module pads `intermediate_size` to a multiple of `max(32, tp_size * 16)`
    and supports loading either unpadded (`3824`) or pre-padded (`3840`) weights.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size = hidden_size
        self.orig_intermediate_size = intermediate_size

        align = max(32, self.tp_size * 16)
        self.intermediate_size = ((intermediate_size + align - 1) // align) * align

        self.fc1 = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[self.intermediate_size, self.intermediate_size],
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.act = SiluAndMul()
        self.fc2 = RowParallelLinear(
            input_size=self.intermediate_size,
            output_size=hidden_size,
            bias=bias,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

        fc1_loader = self._make_fc1_weight_loader()
        _override_weight_loader(cast(nn.Parameter, self.fc1.weight), fc1_loader)
        if getattr(self.fc1, "bias", None) is not None:
            _override_weight_loader(cast(nn.Parameter, self.fc1.bias), fc1_loader)

        fc2_loader = self._make_fc2_weight_loader()
        _override_weight_loader(cast(nn.Parameter, self.fc2.weight), fc2_loader)

    def _make_fc1_weight_loader(self):
        padded_inter = self.intermediate_size
        tp_size = self.tp_size
        tp_rank = self.tp_rank
        local_inter = padded_inter // tp_size

        def _loader(
            param: torch.Tensor,
            loaded_weight: torch.Tensor,
            loaded_shard_id: int | None = None,
        ) -> None:
            if loaded_shard_id is not None:
                part = loaded_weight
                if part.shape[0] < padded_inter and part.shape[0] != local_inter:
                    pad_rows = padded_inter - part.shape[0]
                    pad_shape = (pad_rows, *part.shape[1:])
                    part = torch.cat([part, part.new_zeros(pad_shape)], dim=0)
                if part.shape[0] == padded_inter and tp_size > 1:
                    part = part.narrow(0, tp_rank * local_inter, local_inter)
                dest = param.data.narrow(0, loaded_shard_id * local_inter, local_inter)
                dest.copy_(part)
                return

            if loaded_weight.shape[0] == param.shape[0] and tp_size == 1:
                param.data.copy_(loaded_weight)
                return

            half = loaded_weight.shape[0] // 2
            gate, up = torch.split(loaded_weight, [half, half], dim=0)
            if half < padded_inter and not (tp_size > 1 and half == local_inter):
                pad_rows = padded_inter - half
                pad_shape = (pad_rows, *gate.shape[1:])
                gate = torch.cat([gate, gate.new_zeros(pad_shape)], dim=0)
                up = torch.cat([up, up.new_zeros(pad_shape)], dim=0)

            if gate.shape[0] == padded_inter and tp_size > 1:
                gate = gate.narrow(0, tp_rank * local_inter, local_inter)
                up = up.narrow(0, tp_rank * local_inter, local_inter)

            param.data.copy_(torch.cat([gate, up], dim=0))

        return _loader

    def _make_fc2_weight_loader(self):
        padded_inter = self.intermediate_size
        tp_size = self.tp_size
        tp_rank = self.tp_rank
        local_inter = padded_inter // tp_size

        def _loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            if loaded_weight.shape == param.shape:
                param.data.copy_(loaded_weight)
                return
            in_cols = loaded_weight.shape[1]
            if in_cols < padded_inter and not (tp_size > 1 and in_cols == local_inter):
                pad_cols = padded_inter - in_cols
                loaded_weight = F.pad(loaded_weight, (0, pad_cols))
            if loaded_weight.shape[1] == padded_inter and tp_size > 1:
                loaded_weight = loaded_weight.narrow(
                    1, tp_rank * local_inter, local_inter
                )
            param.data.copy_(loaded_weight)

        return _loader

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.fc1(hidden_states)
        activated = self.act(gate_up)
        output, _ = self.fc2(activated)
        return output


class Mamba3DecoderLayer(nn.Module):
    """Mamba-3 decoder block with `MambaMixer3` and optional `Mamba3GatedMLP`."""

    def __init__(
        self,
        config: PretrainedConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        params = _extract_mamba3_params(config)
        self.residual_in_fp32 = params["residual_in_fp32"]

        self.norm = RMSNorm(params["hidden_size"], eps=params["rms_norm_eps"])
        self.mixer = MambaMixer3(
            hidden_size=params["hidden_size"],
            ssm_state_size=params["state_size"],
            intermediate_size=params["intermediate_size"],
            use_bias=params["use_bias"],
            n_groups=params["n_groups"],
            num_heads=params["num_heads"],
            head_dim=params["head_dim"],
            rms_norm_eps=params["rms_norm_eps"],
            use_rms_norm=params["use_rms_norm"],
            is_mimo=params["is_mimo"],
            mimo_rank=params["mimo_rank"],
            rope_fraction=params["rope_fraction"],
            A_floor=params["A_floor"],
            chunk_size=params["chunk_size"],
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer",
        )

        if params["d_intermediate"] > 0:
            self.norm2: RMSNorm | None = RMSNorm(
                params["hidden_size"], eps=params["rms_norm_eps"]
            )
            self.mlp: Mamba3GatedMLP | None = Mamba3GatedMLP(
                hidden_size=params["hidden_size"],
                intermediate_size=params["d_intermediate"],
                bias=params["use_bias"],
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.norm2 = None
            self.mlp = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        hidden_states = self.mixer(hidden_states)

        if self.mlp is not None and self.norm2 is not None:
            hidden_states, residual = self.norm2(hidden_states, residual)
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Mamba3Model(nn.Module):
    """Mamba-3 backbone (`backbone.embedding`, `backbone.layers`, `backbone.norm_f`)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        params = _extract_mamba3_params(config)
        self.config = config
        self.vocab_size = params["vocab_size"]
        self.hidden_size = params["hidden_size"]

        self.embedding = VocabParallelEmbedding(
            self.vocab_size,
            self.hidden_size,
            org_num_embeddings=self.vocab_size,
        )
        self.embeddings = self.embedding

        def _layer_fn(prefix: str) -> Mamba3DecoderLayer:
            return Mamba3DecoderLayer(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            params["num_hidden_layers"],
            _layer_fn,
            prefix=f"{prefix}.layers",
        )

        self.norm_f = RMSNorm(self.hidden_size, eps=params["rms_norm_eps"])
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        pp_group = get_pp_group()
        if pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                assert input_ids is not None
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = cast(torch.Tensor, intermediate_tensors["hidden_states"])
            residual = cast(torch.Tensor, intermediate_tensors["residual"])

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(hidden_states, residual)

        if not pp_group.is_last_rank:
            assert residual is not None
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm_f(hidden_states, residual)
        return hidden_states


class Mamba3ForCausalLM(
    nn.Module,
    HasInnerState,
    IsAttentionFree,
    SupportsPP,
):
    """vLLM causal LM wrapper for Mamba-3 (SISO and MIMO)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.config = config
        self.model_config = vllm_config.model_config
        params = _extract_mamba3_params(config)

        self.backbone = Mamba3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "backbone"),
        )
        self.model = self.backbone

        if params["tie_word_embeddings"]:
            self.lm_head = self.backbone.embedding
        else:
            self.lm_head = ParallelLMHead(
                params["vocab_size"],
                params["hidden_size"],
                org_num_embeddings=params["vocab_size"],
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                prefix=maybe_prefix(prefix, "lm_head"),
            )

        self.logits_processor = LogitsProcessor(params["vocab_size"])
        self.make_empty_intermediate_tensors = (
            self.backbone.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        return self.backbone(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        _, ssm_state_dtype = MambaStateDtypeCalculator.mamba2_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )
        return torch.float32, ssm_state_dtype

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        params = _extract_mamba3_params(vllm_config.model_config.hf_config)
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        tped_dt_size = params["num_heads"] // tp_size
        num_rope_angles = int((params["state_size"] // 2) * params["rope_fraction"])
        aux_dim = (
            params["mimo_rank"] * (params["head_dim"] + params["state_size"])
            + num_rope_angles
        )
        aux_shape = (tped_dt_size, aux_dim)
        ssm_shape = (
            tped_dt_size,
            params["head_dim"],
            params["state_size"],
        )
        return aux_shape, ssm_shape

    @staticmethod
    def _normalize_weight_name(name: str) -> str:
        if name.startswith("model."):
            name = "backbone." + name[len("model.") :]
        name = name.replace(".embeddings.", ".embedding.")
        name = name.replace(".embed_tokens.", ".embedding.")
        name = name.replace(".tok_embeddings.", ".embedding.")
        name = name.replace(".final_layernorm.", ".norm_f.")
        name = name.replace(".mlp_norm.", ".norm2.")
        name = name.replace(".B_norm.weight", ".B_norm_weight")
        name = name.replace(".C_norm.weight", ".C_norm_weight")
        name = name.replace(".mimo_x", ".Psi")
        name = name.replace(".mimo_o", ".Phi")
        name = name.replace(".mimo_z", ".Zeta")
        if name.endswith(".mixer.b_dt"):
            name = name[: -len(".b_dt")] + ".dt_bias"
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        tie_embeddings = _extract_mamba3_params(self.config)["tie_word_embeddings"]
        loaded_params: set[str] = set()

        for raw_name, loaded_weight in weights:
            name = self._normalize_weight_name(raw_name)

            if tie_embeddings and name.startswith("lm_head."):
                if "backbone.embedding.weight" not in loaded_params:
                    name = name.replace("lm_head.", "backbone.embedding.", 1)
                else:
                    continue

            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
            if name == "backbone.embedding.weight":
                loaded_params.add("backbone.embeddings.weight")
                if tie_embeddings:
                    loaded_params.add("lm_head.weight")

        return loaded_params

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
from transformers import PretrainedConfig
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
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
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
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.layers.adapter.mamba_mixer3 import (
    MambaMixer3,
    mamba3_state_dtype,
    mamba3_state_layout,
    set_param_weight_loader,
)
from vllm_torchtpu.models.vllm.mamba3.config import (
    Mamba3Config,
    ensure_mamba3_config,
)


class Mamba3GatedMLP(nn.Module):
    """SwiGLU GatedMLP layer matching `mamba_ssm.modules.mlp.GatedMLP`.

    Upstream `GatedMLP` pads `hidden_features` to `multiple_of=128` (for example
    `3824 -> 3840` on `mamba3-mimo-1.5b`) and computes
    `y, gate = fc1(x).chunk(2, dim=-1); fc2(y * silu(gate))` (gating with the
    second half), whereas vLLM `SiluAndMul` computes `silu(first) * second`.
    This module pads `intermediate_size` to `max(128, tp_size * 16)` and swaps
    the two checkpoint halves `[y_up, gate] -> [gate, y_up]` on load.
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

        align = max(128, self.tp_size * 16)
        self.intermediate_size = ((intermediate_size + align - 1) // align) * align

        self.fc1 = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[self.intermediate_size, self.intermediate_size],
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        if not isinstance(self.fc1.quant_method, UnquantizedLinearMethod):
            raise NotImplementedError(
                "Quantized weights are not yet supported for Mamba3GatedMLP."
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
        if not isinstance(self.fc2.quant_method, UnquantizedLinearMethod):
            raise NotImplementedError(
                "Quantized weights are not yet supported for Mamba3GatedMLP."
            )

        fc1_loader = self._make_fc1_weight_loader()
        set_param_weight_loader(cast(nn.Parameter, self.fc1.weight), fc1_loader)
        if self.fc1.bias is not None:
            set_param_weight_loader(cast(nn.Parameter, self.fc1.bias), fc1_loader)

        fc2_loader = self._make_fc2_weight_loader()
        set_param_weight_loader(cast(nn.Parameter, self.fc2.weight), fc2_loader)

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
                if loaded_shard_id not in (0, 1):
                    raise ValueError(
                        f"Expected loaded_shard_id in (0, 1), got {loaded_shard_id}."
                    )
                part = loaded_weight
                if part.shape[0] < padded_inter and not (
                    tp_size > 1 and part.shape[0] == local_inter
                ):
                    pad_rows = padded_inter - part.shape[0]
                    pad_shape = (pad_rows, *part.shape[1:])
                    part = torch.cat([part, part.new_zeros(pad_shape)], dim=0)
                if part.shape[0] == padded_inter and tp_size > 1:
                    part = part.narrow(0, tp_rank * local_inter, local_inter)
                # Shard 0 in checkpoint is `y_up` (-> 1), shard 1 is `gate` (-> 0).
                dest_shard = 1 - loaded_shard_id
                dest = param.data.narrow(0, dest_shard * local_inter, local_inter)
                dest.copy_(part)
                return

            half = loaded_weight.shape[0] // 2
            y_up, gate = torch.split(loaded_weight, [half, half], dim=0)
            if half < padded_inter and not (tp_size > 1 and half == local_inter):
                pad_rows = padded_inter - half
                pad_shape = (pad_rows, *gate.shape[1:])
                gate = torch.cat([gate, gate.new_zeros(pad_shape)], dim=0)
                y_up = torch.cat([y_up, y_up.new_zeros(pad_shape)], dim=0)

            if gate.shape[0] == padded_inter and tp_size > 1:
                gate = gate.narrow(0, tp_rank * local_inter, local_inter)
                y_up = y_up.narrow(0, tp_rank * local_inter, local_inter)

            param.data.copy_(torch.cat([gate, y_up], dim=0))

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
    """Mamba-3 decoder block with `MambaMixer3` and optional `Mamba3GatedMLP`.

    Matches upstream vLLM `Mamba2DecoderLayer` by keeping the residual stream in
    the model activation dtype (`bfloat16`) across fused
    `RMSNorm(hidden_states, residual)` calls rather than promoting the residual
    stream to `float32`.
    """

    def __init__(
        self,
        config: Mamba3Config | PretrainedConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        cfg = ensure_mamba3_config(config)

        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mixer = MambaMixer3(
            hidden_size=cfg.hidden_size,
            ssm_state_size=cfg.state_size,
            intermediate_size=cfg.intermediate_size,
            use_bias=cfg.use_bias,
            n_groups=cfg.n_groups,
            num_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            rms_norm_eps=cfg.rms_norm_eps,
            use_rms_norm=cfg.use_rms_norm,
            is_mimo=cfg.is_mimo,
            mimo_rank=cfg.mimo_rank,
            rope_fraction=cfg.rope_fraction,
            A_floor=cfg.A_floor,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer",
        )

        if cfg.d_intermediate > 0:
            self.norm2: RMSNorm | None = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.mlp: Mamba3GatedMLP | None = Mamba3GatedMLP(
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.d_intermediate,
                bias=cfg.use_bias,
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
        cfg = ensure_mamba3_config(vllm_config.model_config.hf_config)
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = cfg
        self.vocab_size = cfg.vocab_size
        self.hidden_size = cfg.hidden_size

        pp_group = get_pp_group()
        if pp_group.is_first_rank:
            self.embedding: VocabParallelEmbedding | PPMissingLayer = (
                VocabParallelEmbedding(
                    self.vocab_size,
                    self.hidden_size,
                    org_num_embeddings=self.vocab_size,
                )
            )
        else:
            self.embedding = PPMissingLayer()
        self.embeddings = self.embedding

        def _layer_fn(prefix: str) -> Mamba3DecoderLayer:
            return Mamba3DecoderLayer(
                cfg,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            cfg.num_hidden_layers,
            _layer_fn,
            prefix=f"{prefix}.layers",
        )

        if pp_group.is_last_rank:
            self.norm_f: RMSNorm | PPMissingLayer = RMSNorm(
                self.hidden_size, eps=cfg.rms_norm_eps
            )
        else:
            self.norm_f = PPMissingLayer()

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
        del positions
        pp_group = get_pp_group()
        if pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("Either input_ids or inputs_embeds is required.")
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            if intermediate_tensors is None:
                raise ValueError(
                    "intermediate_tensors is required on non-first PP ranks."
                )
            hidden_states = cast(torch.Tensor, intermediate_tensors["hidden_states"])
            residual = cast(torch.Tensor, intermediate_tensors["residual"])

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(hidden_states, residual)

        if not pp_group.is_last_rank:
            if residual is None:
                raise ValueError("Expected non-null residual across decoder layers.")
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        assert isinstance(self.norm_f, RMSNorm)
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
        cfg = ensure_mamba3_config(vllm_config.model_config.hf_config)
        self.vllm_config = vllm_config
        self.config = cfg
        self.model_config = vllm_config.model_config

        self.backbone = Mamba3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "backbone"),
        )
        self.model = self.backbone

        pp_group = get_pp_group()
        if pp_group.is_last_rank:
            if cfg.tie_word_embeddings and pp_group.is_first_rank:
                self.lm_head: (
                    VocabParallelEmbedding | ParallelLMHead | PPMissingLayer
                ) = self.backbone.embedding
            else:
                self.lm_head = ParallelLMHead(
                    cfg.vocab_size,
                    cfg.hidden_size,
                    org_num_embeddings=cfg.vocab_size,
                    padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        # Keep full padded `cfg.vocab_size` width for `TPUModelRunner` precompiled
        # sampler / logprobs / structured-decoding graphs and mask padded tail to -inf.
        self.logits_processor = LogitsProcessor(cfg.vocab_size)
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
        del kwargs
        return self.backbone(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if isinstance(self.lm_head, PPMissingLayer):
            return None
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is not None and self.config.org_vocab_size < self.config.vocab_size:
            col_idx = torch.arange(logits.shape[-1], device=logits.device)
            logits = torch.where(
                col_idx < self.config.org_vocab_size,
                logits,
                torch.full_like(logits, float("-inf")),
            )
        return logits

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return mamba3_state_dtype(vllm_config.model_config, vllm_config.cache_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        cfg = ensure_mamba3_config(vllm_config.model_config.hf_config)
        layout = mamba3_state_layout(
            num_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            ssm_state_size=cfg.state_size,
            rope_fraction=cfg.rope_fraction,
            is_mimo=cfg.is_mimo,
            mimo_rank=cfg.mimo_rank,
            tp_size=vllm_config.parallel_config.tensor_parallel_size,
        )
        return layout.aux_shape, layout.ssm_shape

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
        if name.endswith(".mixer.b_dt"):
            name = name[: -len(".b_dt")] + ".dt_bias"
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        tie_embeddings = self.config.tie_word_embeddings
        loaded_params: set[str] = set()
        unexpected_names: list[str] = []

        for raw_name, loaded_weight in weights:
            name = self._normalize_weight_name(raw_name)

            if tie_embeddings:
                if name.startswith("lm_head."):
                    emb_name = name.replace("lm_head.", "backbone.embedding.", 1)
                    if "lm_head.weight" in params_dict:
                        if name in loaded_params:
                            continue
                    elif emb_name in params_dict:
                        if emb_name in loaded_params:
                            continue
                        name = emb_name
                elif name.startswith("backbone.embedding."):
                    head_name = name.replace("backbone.embedding.", "lm_head.", 1)
                    if name not in params_dict and head_name in params_dict:
                        if head_name in loaded_params:
                            continue
                        name = head_name

            if is_pp_missing_parameter(name, self):
                continue

            if name not in params_dict:
                unexpected_names.append(raw_name)
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
            if name == "backbone.embedding.weight":
                loaded_params.add("backbone.embeddings.weight")
                if tie_embeddings and isinstance(self.lm_head, VocabParallelEmbedding):
                    loaded_params.add("lm_head.weight")

        if unexpected_names:
            raise ValueError(
                f"Unexpected checkpoint tensor(s) for Mamba3ForCausalLM: "
                f"{unexpected_names}"
            )

        return loaded_params

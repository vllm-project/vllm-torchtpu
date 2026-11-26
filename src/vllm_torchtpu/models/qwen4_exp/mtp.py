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
"""Qwen4Exp MTP (Multi-Token Predictor) draft model for TPU.

Follows upstream vllm/models/qwen4_exp/nvidia/mtp.py while reusing the TPU
Qwen4ExpDecoderLayer (with Qwen4ExpAttention and Qwen3NextSparseMoeBlock) and
common PyTorch GatedResidual hyper-connections.

The draft model:
  - keeps the target's `hc_count`-stream residual width (`hc_count * H`) as its
    backbone input and carry state across loop steps,
  - fuses the token embedding (`fc_embedding`) and multi-stream backbone hidden
    (`fc_hidden`) via broadcast addition in fp32 (matching upstream's
    `hc_combine` with `injection=None`),
  - disables PLE on draft layers and uses prefix `mtp.layers.0` so checkpoint
    FP8 exclusion lists and expert placement match verbatim,
  - returns `(sample_hidden, multi_hidden)` = `([T, H], [T, hc_count * H])`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch
from torch import nn
from transformers import Qwen4ExpTextConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.models.qwen4_exp.common.hyperconnection import GatedResidual

from vllm_torchtpu.models.qwen4_exp.model import (
    Qwen4ExpDecoderLayer,
    Qwen4ExpForCausalLM,
    _hc_config,
)


def _remap_mtp_weight_name(name: str) -> str | None:
    """Maps Qwen4Exp checkpoint weight paths onto `Qwen4ExpMTP` module names."""
    for checkpoint_prefix in ("model.language_model.", "language_model."):
        if name.startswith(checkpoint_prefix):
            name = name.removeprefix(checkpoint_prefix)
            break

    if name.startswith("embed_tokens."):
        name = f"model.{name}"
    if name.startswith("model.mtp."):
        name = name.removeprefix("model.")
    if name.startswith("model.lm_head."):
        return name.removeprefix("model.")
    if name.startswith("mtp.shared_head.head."):
        return None
    if name.startswith("mtp."):
        return name.replace("mtp.", "model.", 1)
    if name.startswith("model.embed_tokens.") or name.startswith("lm_head."):
        return name
    return None


class Qwen4ExpMultiTokenPredictor(nn.Module):
    """Multi-stream HC MTP predictor body for Qwen4Exp on TPU."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = cast(Qwen4ExpTextConfig, vllm_config.model_config.hf_text_config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        quant_config = vllm_config.quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            self.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.pre_fc_norm_embedding = GemmaRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            self.hidden_size * self.hc_count, eps=config.rms_norm_eps
        )
        # ReplicatedLinear avoids two TP all-gathers per draft step for a pair
        # of small [H, H] projections (26 MB total in BF16).
        self.fc_embedding = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "fc_embedding"),
        )
        self.fc_hidden = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "fc_hidden"),
        )
        # Single-element ModuleList keeps checkpoint parameter names at
        # `model.layers.0.*`; TPUQwen4ExpConfig rejects mtp_num_hidden_layers != 1.
        self.layers = nn.ModuleList(
            [
                Qwen4ExpDecoderLayer(
                    vllm_config,
                    prefix=maybe_prefix(prefix, "layers.0"),
                    layer_type="indexed_attention",
                    enable_ple=False,
                )
            ]
        )
        self.hyper_connection_mixer = GatedResidual(
            _hc_config(vllm_config), use_combine=False
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)
        emb = self.fc_embedding(self.pre_fc_norm_embedding(inputs_embeds))

        num_tokens = hidden_states.shape[0]
        h = self.pre_fc_norm_hidden(hidden_states).view(
            num_tokens, self.hc_count, self.hidden_size
        )
        h = self.fc_hidden(h)
        # Matches upstream hc_combine(..., injection=None): add in fp32 then
        # cast back to the activation dtype.
        h = (h.float() + emb.float().unsqueeze(-2)).to(h.dtype).flatten(-2)

        del spec_step_idx
        multi_hidden = self.layers[0](positions, h, input_ids=None)
        sample_hidden, _ = self.hyper_connection_mixer.mix(multi_hidden)
        return sample_hidden, multi_hidden


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "hidden_states": 0,
        "inputs_embeds": 0,
    }
)
class Qwen4ExpMTP(nn.Module):
    """Top-level Qwen4Exp MTP draft model for TPU.

    Intentionally defines no class-level `hf_to_vllm_mapper`: target and draft
    share one `quant_config` during draft loading, and a class-level `"mtp." ->
    "model."` rename would rewrite `mtp.*` entries in `ignored_layers` and turn
    BF16 draft projections into FP8.
    """

    packed_modules_mapping = Qwen4ExpForCausalLM.packed_modules_mapping
    has_own_embed_tokens = False
    has_own_lm_head = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.prefix = prefix or "mtp"
        config = cast(Qwen4ExpTextConfig, vllm_config.model_config.hf_text_config)
        self.config = config
        self.model = Qwen4ExpMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "mtp"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: object | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if intermediate_tensors is not None:
            raise NotImplementedError("Qwen4ExpMTP on TPU does not support PP.")
        return self.model(
            input_ids,
            positions,
            hidden_states,
            inputs_embeds=inputs_embeds,
            spec_step_idx=spec_step_idx,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        del spec_step_idx
        return self.logits_processor(self.lm_head, hidden_states)

    def is_unused_checkpoint_weight(self, name: str) -> bool:
        """Skip target backbone weights before loading shard bytes."""
        return _remap_mtp_weight_name(name) is None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap_weight_names() -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                remapped = _remap_mtp_weight_name(name)
                if remapped is not None:
                    yield remapped, weight

        mapper = Qwen3_5Model.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={
                "hyper_connection_mixer.block_inject_weight": None,
            }
        )
        has_indexer = any(
            layer.self_attn.indexer is not None for layer in self.model.layers
        )
        if not has_indexer:
            mapper = mapper | WeightsMapper(
                orig_to_new_substr={".self_attn.indexer.": None}
            )
        loaded = AutoWeightsLoader(self).load_weights(
            remap_weight_names(), mapper=mapper
        )
        if not any(name.startswith("model.layers.0.") for name in loaded):
            raise ValueError(
                "Qwen4ExpMTP checkpoint did not load any mtp.layers.0.* weights."
            )
        return loaded

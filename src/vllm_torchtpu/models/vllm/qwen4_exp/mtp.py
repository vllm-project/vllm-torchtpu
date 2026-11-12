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
"""TPU-native Qwen4Exp MTP built from the target decoder.

The fusion follows vLLM db9527a4's Qwen4Exp residual_linear_shared math.
The single MTP layer is reused across draft steps; carry width is HC * H.
"""

import copy
import re
from collections.abc import Iterable, Sequence
from typing import cast

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import set_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.model_executor.models.utils import maybe_prefix

from vllm_torchtpu.layers.adapter.quantization import get_tpu_quantization_config
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

from .hyperconnection import GatedResidual
from .model import (
    Qwen4ExpDecoderLayer,
    Qwen4ExpForCausalLM,
    hc_config,
    load_qwen4_exp_weights,
)
from .pool_layout import pooled_cache_specs


def remap_mtp_quant_name(name: str, start_layer: int) -> str:
    if not name.startswith("mtp."):
        return name
    return re.sub(
        r"(?<=\.layers\.)\d+",
        lambda match: str(start_layer + int(match.group(0))),
        name,
    )


def remap_mtp_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Select draft tensors and map their checkpoint prefix to the model."""
    for name, weight in weights:
        name = re.sub(r"^(?:model\.)?language_model\.", "", name)
        name = name.removeprefix("model.")
        if name.startswith("mtp."):
            yield "model." + name[4:], weight


def make_mtp_config(vllm_config):
    spec = vllm_config.speculative_config
    assert spec is not None and spec.draft_model_config is not None
    cfg = copy.copy(vllm_config)
    cfg.model_config = copy.deepcopy(spec.draft_model_config)
    text = cfg.model_config.hf_text_config
    # Preserve target layer numbering; only the MTP decoder's position config
    # changes. Never mutate the target's shared text config or quant config.
    text.ple_layer_ids = []
    target_model_cfg = getattr(vllm_config, "model_config", None)
    target_text = getattr(target_model_cfg, "hf_text_config", None)
    if getattr(target_text, "rope_parameters", None) is not None:
        text.rope_parameters = copy.deepcopy(target_text.rope_parameters)
    text.rope_parameters["rope_theta"] = text.mtp["rope_theta"]
    text.rope_theta = text.mtp["rope_theta"]
    cfg.quant_config = get_tpu_quantization_config(cfg)
    configure_quant_config(cfg.quant_config, Qwen4ExpMTP)
    assert isinstance(cfg.quant_config, VllmFp8Config)
    cfg.quant_config.ignored_layers = [
        remap_mtp_quant_name(name, text.num_hidden_layers)
        for name in cfg.quant_config.ignored_layers
    ]
    return cfg


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "hidden_states": 0,
        "inputs_embeds": 0,
    }
)
class Qwen4ExpMultiTokenPredictor(nn.Module):
    def __init__(self, cfg, *, prefix="mtp"):
        super().__init__()
        self.prefix = prefix
        text = cfg.model_config.hf_text_config
        self.hidden_size = text.hidden_size
        self.hc_count = text.hc_count
        self.embed_tokens = VocabParallelEmbedding(text.vocab_size, text.hidden_size)
        self.fc_embedding = ColumnParallelLinear(
            text.hidden_size,
            text.hidden_size,
            bias=False,
            gather_output=True,
            return_bias=False,
            quant_config=cfg.quant_config,
            prefix=f"{prefix}.fc_embedding",
        )
        self.fc_hidden = ColumnParallelLinear(
            text.hidden_size,
            text.hidden_size,
            bias=False,
            gather_output=True,
            return_bias=False,
            quant_config=cfg.quant_config,
            prefix=f"{prefix}.fc_hidden",
        )
        self.pre_fc_norm_embedding = GemmaRMSNorm(text.hidden_size, text.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            text.hidden_size * text.hc_count, text.rms_norm_eps
        )
        self.attention_prefix = f"{prefix}.layers.{text.num_hidden_layers}.self_attn"
        self.layers = nn.ModuleList(
            [
                Qwen4ExpDecoderLayer(
                    cfg,
                    text.num_hidden_layers,
                    f"{prefix}.layers.{text.num_hidden_layers}",
                    layer_type="indexed_attention",
                    # Drafts compile independently of the target's bucket
                    # retracing. Keep HC's Torch graph valid at every size;
                    # small decode buckets already use this math path.
                    hc_prefill_tile=0,
                )
            ]
        )
        self.hyper_connection_mixer = GatedResidual(
            hc_config(text), use_combine=False, prefill_full_tile=0
        )

    def forward(self, input_ids, positions, hidden_states, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        embedding = self.fc_embedding(self.pre_fc_norm_embedding(inputs_embeds))
        # Normalize over all HC streams, then apply one shared projection to
        # every branch, as in upstream residual_linear_shared.
        normalized = self.pre_fc_norm_hidden(hidden_states).unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        hidden = (self.fc_hidden(normalized) + embedding.unsqueeze(-2)).flatten(-2)
        metadata = cast(
            dict[str, AttentionMetadata], get_forward_context().attn_metadata
        )
        md = metadata[self.attention_prefix]
        starts = cast(torch.Tensor, md.query_start_loc)
        lengths = starts[1:] - starts[:-1]
        context = cast(torch.Tensor, md.seq_lens) - lengths
        valid = torch.arange(hidden.shape[0], device=hidden.device) < starts[-1]
        hidden = self.layers[0](
            hidden,
            positions.to(torch.int64),
            None,  # PLE is disabled for the MTP decoder.
            starts,
            None,
            context,
            valid,
        )
        return self.hyper_connection_mixer.mix(hidden)[0], hidden


class Qwen4ExpMTP(nn.Module, MixtureOfExperts):
    has_own_embed_tokens = False
    has_own_lm_head = False
    packed_modules_mapping = Qwen4ExpForCausalLM.packed_modules_mapping

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        cfg = make_mtp_config(vllm_config)
        self.vllm_config = cfg
        self.config = cfg.model_config.hf_text_config
        with set_current_vllm_config(cfg):
            self.model = Qwen4ExpMultiTokenPredictor(
                cfg, prefix=maybe_prefix(prefix, "mtp")
            )
        self.lm_head = (
            self.model.embed_tokens
            if self.config.tie_word_embeddings
            else ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        layers = cast(Sequence[Qwen4ExpDecoderLayer], self.model.layers)
        self.moe_layers = [layer.mlp.experts for layer in layers]
        example = layers[0].mlp
        self.expert_weights = []
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example.n_logical_experts
        self.num_physical_experts = example.n_physical_experts
        self.num_local_physical_experts = example.n_local_physical_experts
        self.num_routed_experts = example.n_routed_experts
        self.num_redundant_experts = example.n_redundant_experts

    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ) -> None:
        raise NotImplementedError("Qwen4Exp MTP does not support EPLB")

    def embed_input_ids(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def forward(
        self, input_ids, positions, hidden_states, inputs_embeds=None, spec_step_idx=0
    ):
        return self.model(input_ids, positions, hidden_states, inputs_embeds)

    def compute_logits(self, hidden_states, spec_step_idx=0):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Shares the target's strict HC/QSA/MoE/FP8 completeness checks. Actual
        # module names use local index 0; kernel/cache prefixes use global 48.
        parameters = dict(self.named_parameters())
        loaded = load_qwen4_exp_weights(
            self,
            remap_mtp_weights(weights),
            cast(Sequence[Qwen4ExpDecoderLayer], self.model.layers),
            {
                n
                for n in parameters
                if n.startswith(("model.embed_tokens.", "lm_head."))
            },
            block_size=pooled_cache_specs(self.vllm_config)[0].block_size,
        )
        return loaded

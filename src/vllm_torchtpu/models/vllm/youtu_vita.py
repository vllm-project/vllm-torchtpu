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
"""YoutuVITAForCausalLM (model_type youtu_vita, e.g. tencent/Youtu-Parsing-Omni).

vLLM has no model class for this architecture. Its language model (text_config,
model_type youtu_vita_text) is a DeepSeek-V2 style decoder: MLA attention
(q_lora + kv_lora, interleaved RoPE) and a dense SwiGLU MLP in every layer, an
untied lm_head. vLLM's DeepseekV2ForCausalLM serves exactly that when no layer
is MoE, so this class builds it on text_config.

Text only: the omni (vision / audio / video) encoder under model.omni_model is
not built and its checkpoint tensors are skipped.
"""

import copy
from collections.abc import Iterable

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)


class YoutuVITAModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """vLLM picks MLA from a fixed list of model types; the Youtu text model is
    MLA whenever kv_lora_rank is set (head size kv_lora_rank + rope dim)."""

    def is_deepseek_mla(self) -> bool:
        return getattr(self.hf_text_config, "kv_lora_rank", None) is not None


class YoutuVITAForCausalLM(DeepseekV2ForCausalLM):
    # The language model sits under model.language_model in the checkpoint.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model."}
    )
    # Not built by this text-only class (see the module docstring).
    skipped_prefixes = ("model.omni_model.",)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        text_config = copy.deepcopy(vllm_config.model_config.hf_config.text_config)
        # DeepseekV2 reads its MoE fields unconditionally; 0 routed experts and
        # the first dense layer count = all layers make every layer dense.
        text_config.n_routed_experts = 0
        text_config.first_k_dense_replace = text_config.num_hidden_layers
        # transformers v5 keeps tie_word_embeddings on the outer config: False,
        # the checkpoint ships its own lm_head.weight.
        text_config.tie_word_embeddings = (
            vllm_config.model_config.hf_config.tie_word_embeddings
        )
        super().__init__(
            vllm_config=vllm_config.with_hf_config(
                text_config, architectures=["DeepseekV2ForCausalLM"]
            ),
            prefix=prefix,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = (
            (n, w) for n, w in weights if not n.startswith(self.skipped_prefixes)
        )
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )

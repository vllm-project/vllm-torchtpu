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
"""GLM-5.3-Flash MTP draft head (checkpoint layer 45) for speculative decoding.

Adapted from the draft head on upstream branch
``stephenchen-glm5-3-flash-experimental`` (``82f82c089``) to the current
target: layer 45 is built with the *same* sparse-MLA class as the target's
DSA layers, including its own k-pool indexer (the checkpoint ships a complete
``layers.45.self_attn.indexer.*`` block), instead of the dense MLA that branch
used. What layer 45 looks like in the checkpoint, each point silent when
wrong:

* A full DSA attention block and a routed MoE (288 experts + shared), but **no
  ``hc_*`` tensors**: no mHC, a plain pre-norm residual.
* The usual nextn wrapper: ``enorm``, ``hnorm``, ``eh_proj`` (bf16, not
  quantized) and ``shared_head.norm``. ``eh_proj`` takes ``[embeds | hidden]``
  in that order -- its second half is ~1.78 * I, a residual pass-through of
  the previous hidden state.
* **No ``shared_head.head``**: the logits come from the target's ``lm_head``,
  shared by the proposer along with ``embed_tokens``.
* ``num_hidden_layers`` (45) is the MTP layer index; a config override that
  zeroed it would rename the module key and load nothing.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import maybe_prefix

from vllm_torchtpu.models.vllm.glm5_next.attention import Glm5NextMLA
from vllm_torchtpu.models.vllm.glm5_next.model import Glm5NextForCausalLM
from vllm_torchtpu.models.vllm.glm5_next.moe import Glm5NextMoE

#: Names directly under ``model.layers.<mtp>.`` that belong to the nextn
#: wrapper; everything else on that layer is the decoder block.
_MTP_WRAPPER_NAMES = ("enorm", "hnorm", "eh_proj", "shared_head")


def _topk_indices_buffer(vllm_config: VllmConfig, config) -> torch.Tensor:
    """The draft's own candidate-list buffer, sized like the target's."""
    width = config.index_topk + (
        config.index_kpool - 1
        if getattr(config, "index_kpool_always_select_tail", False)
        else 0
    )
    width = -(-width // 256) * 256
    return torch.empty(
        vllm_config.scheduler_config.max_num_batched_tokens,
        width,
        dtype=torch.int32,
        device=vllm_config.device_config.device,
    )


class Glm5NextMTPBlock(nn.Module):
    """Layer 45's decoder block: sparse MLA + routed MoE, plain residual."""

    def __init__(
        self, vllm_config: VllmConfig, prefix: str, topk_indices_buffer
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.self_attn = Glm5NextMLA(
            config,
            vllm_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
            # Traced once for all buckets: no token-count-specialized path.
            all_heads_replicas=False,
        )
        self.mlp = Glm5NextMoE(vllm_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(positions, self.input_layernorm(hidden_states))
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = tensor_model_parallel_all_reduce(
            self.mlp(self.post_attention_layernorm(hidden_states))
        )
        return residual + hidden_states


class Glm5NextMultiTokenPredictorLayer(nn.Module):
    """One nextn layer: the embed/hidden merge, the block, the head norm."""

    def __init__(
        self, vllm_config: VllmConfig, prefix: str, topk_indices_buffer
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        hidden_size = config.hidden_size
        self.enorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        # bf16 in the checkpoint and listed in `modules_to_not_convert`; a vLLM
        # linear built at the checkpoint prefix lets the quant config see that.
        self.eh_proj = ReplicatedLinear(
            hidden_size * 2,
            hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.eh_proj",
        )
        self.shared_head_norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        # Built at the layer's own prefix (not `.mtp_block`) so the FP8
        # ignore list, which names `model.layers.45.self_attn.*`, matches.
        self.mtp_block = Glm5NextMTPBlock(vllm_config, prefix, topk_indices_buffer)

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        # Position 0 has no predecessor; every GLM MTP zeroes its embedding.
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0.0, inputs_embeds)
        hidden_states = self.eh_proj(
            torch.cat(
                [self.enorm(inputs_embeds), self.hnorm(previous_hidden_states)],
                dim=-1,
            )
        )[0]
        return self.mtp_block(positions, hidden_states)


class Glm5NextMultiTokenPredictor(nn.Module):
    """The nextn layers, keyed by their absolute checkpoint layer index."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 0)
        if not self.num_mtp_layers:
            raise ValueError(
                "glm5_next MTP was requested but the config declares "
                "num_nextn_predict_layers=0: there is no draft layer to load."
            )
        self.topk_indices_buffer = _topk_indices_buffer(vllm_config, config)
        self.layers = nn.ModuleDict(
            {
                str(idx): Glm5NextMultiTokenPredictorLayer(
                    vllm_config,
                    f"{prefix}.layers.{idx}",
                    self.topk_indices_buffer,
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        # Replaced by the target's embedding at load (`maybe_share_embeddings`).
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

    def _layer(self, spec_step_idx: int) -> Glm5NextMultiTokenPredictorLayer:
        step = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + step)]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        layer = self._layer(spec_step_idx)
        hidden_states = layer(positions, previous_hidden_states, inputs_embeds)
        # Normed here, not in `compute_logits`: the proposer chains this output
        # into the next draft step as well, and the first step's input is the
        # target's final-normed hidden. Carrying the pre-norm hidden instead
        # (vLLM's DeepSeek MTP, and upstream's 82f82c089) measured worse on
        # GSM8K at K=3: per-position acceptance 0.90/0.71/0.43 vs 0.90/0.75/0.57.
        return layer.shared_head_norm(hidden_states)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "hidden_states": 0,
        "inputs_embeds": 0,
    }
)
class Glm5NextMTP(nn.Module):
    """GLM-5.3-Flash's draft head, as a standalone vLLM model."""

    packed_modules_mapping = Glm5NextForCausalLM.packed_modules_mapping
    hf_to_vllm_mapper = Glm5NextForCausalLM.hf_to_vllm_mapper
    stacked_params_mapping = Glm5NextForCausalLM.stacked_params_mapping
    #: The proposer shares the target's embedding (`maybe_share_embeddings`).
    has_own_embed_tokens = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.model = Glm5NextMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.num_mtp_layers = self.model.num_mtp_layers
        # Replaced by the target's `lm_head` at load (`maybe_share_lm_head`).
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
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
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return Glm5NextForCausalLM.get_expert_mapping(self)

    def _rewrite(self, name: str) -> str | None:
        """Checkpoint name (after the HF mapper) -> this module's name, or
        None for tensors the draft does not own (the target's 45 layers, the
        vision tower). ``embed_tokens`` and ``lm_head`` are loaded too, so the
        loader's every-parameter-loaded check holds and the draft is right
        even if the proposer does not share them; sharing then frees them."""
        if name.startswith(("model.embed_tokens.", "lm_head.")):
            return name
        prefix = f"model.layers.{self.model.mtp_start_layer_idx}."
        if not name.startswith(prefix):
            return None
        tail = name[len(prefix) :]
        if tail.startswith("shared_head.norm."):
            return f"{prefix}shared_head_norm.{tail[len('shared_head.norm.') :]}"
        if tail.split(".", 1)[0] in _MTP_WRAPPER_NAMES:
            return name
        return f"{prefix}mtp_block.{tail}"

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def draft_weights():
            for name, tensor in self.hf_to_vllm_mapper.apply(weights):
                new = self._rewrite(name)
                if new is not None:
                    yield new, tensor

        # Same stacked / expert / sequence-parallel-replica handling as the target.
        return Glm5NextForCausalLM.load_weights(self, draft_weights())


__all__ = ["Glm5NextMTP"]

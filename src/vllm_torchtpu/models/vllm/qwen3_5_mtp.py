# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 MTP drafts that one pipeline stage hosts whole."""

from typing import ClassVar, Literal

import torch
from vllm.distributed import get_pp_group, tensor_model_parallel_all_gather
from vllm.model_executor.models import qwen3_5_mtp as upstream
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors


def _predictor_forward(
    self: upstream.Qwen3_5MultiTokenPredictor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    spec_step_idx: int = 0,
) -> torch.Tensor:
    """Upstream's forward, branching on the inputs instead of the PP rank.

    The draft runs whole on the last pipeline stage, where
    ``get_pp_group().is_first_rank`` is False; the embedding branch is the
    one with no intermediate tensors (vllm-project/vllm#46994).
    """
    if intermediate_tensors is None:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)
        residual = None
    else:
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    current_step_idx = spec_step_idx % self.num_mtp_layers
    mtp_layer = self.layers[current_step_idx]
    if mtp_layer.use_attn_reduce_scatter_for_moe:
        assert hidden_states.shape[0] == positions.shape[-1]
        hidden_states = sequence_parallel_chunk(hidden_states)
        assert residual is None
    hidden_states, residual = mtp_layer(
        positions=positions,
        hidden_states=hidden_states,
        residual=residual,
    )

    if not get_pp_group().is_last_rank:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )

    hidden_states, _ = self.norm(hidden_states, residual)
    if mtp_layer.use_attn_reduce_scatter_for_moe:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        hidden_states = hidden_states[: positions.shape[-1]]
    return hidden_states


upstream.Qwen3_5MultiTokenPredictor.forward = _predictor_forward


class Qwen3_5MTP(upstream.Qwen3_5MTP, SupportsPP):
    """Upstream's draft, admitted under pipeline parallelism.

    vLLM's config check requires every model under a pipeline to declare
    `SupportsPP`. The pipeline never splits this draft: the last stage builds
    and runs it whole, and the predictor's forward above takes its embedding
    branch there.
    """

    supports_pp: ClassVar[Literal[True]] = True

    @property
    def make_empty_intermediate_tensors(self):
        return self.model.make_empty_intermediate_tensors


class Qwen3_5MoeMTP(upstream.Qwen3_5MoeMTP, SupportsPP):
    """The MoE draft, admitted under pipeline parallelism; see `Qwen3_5MTP`."""

    supports_pp: ClassVar[Literal[True]] = True

    @property
    def make_empty_intermediate_tensors(self):
        return self.model.make_empty_intermediate_tensors

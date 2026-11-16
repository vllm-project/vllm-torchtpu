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
"""Trace-stable sequence-parallel forwards for Qwen3.5 MoE.

vLLM's Qwen3.5 decoder layer and model inherit these forwards from the
Qwen3Next classes, so the patch replaces them there.

Copies of the upstream vLLM forwards with three changes, so torch.compile
serves every bucket under TPU_MOE_SEQUENCE_PARALLEL=1:
- torch._check(num_tokens % tp == 0) states that buckets divide by TP.
- narrow(0, 0, num_tokens) replaces [:num_tokens] after each all-gather to
  preserve exact symbolic output dimensions across TorchDynamo traces.
- The pad before the reduce-scatter is eliminated, removing redundant padding
  computations across layers.

Two opt-in changes on top (distributed/sequence_parallel_rows):
- TPU_SP_MIN_ROWS: backbone steps below the threshold keep every token on
  every rank, with an all-reduce after attention instead of the all-gather and
  reduce-scatter.
- TPU_SP_FP8_GATHER: the all-gather before attention carries fp8 rows and
  per-row scales instead of bf16 rows.
"""

from itertools import islice

import torch
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.models import qwen3_next
from vllm.sequence import IntermediateTensors

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.sequence_parallel_rows import replicated_rows_step
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

UPSTREAM_DECODER_LAYER_FORWARD = qwen3_next.Qwen3NextDecoderLayer.forward
UPSTREAM_MODEL_FORWARD = qwen3_next.Qwen3NextModel.forward


def _check_tokens_divide_by_tp(num_tokens: int) -> None:
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size > 1:
        torch._check(num_tokens % tp_size == 0)


def _all_gather_rows(hidden_states: torch.Tensor) -> torch.Tensor:
    """All-gather this rank's normed rows for attention, in fp8 if asked."""
    if envs.TPU_SP_FP8_GATHER:
        # Imported here: the module needs TPU-only packages at import time.
        from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
            quantized_all_gather,
        )

        return quantized_all_gather(get_tp_group(), hidden_states)
    return tensor_model_parallel_all_gather(hidden_states, 0)


def patched_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    replicated_rows: bool = False,
    **kwargs: object,
):
    # replicated_rows: this step holds every token on every rank
    # (TPU_SP_MIN_ROWS); only the backbone's model forward sets it.
    full_num_tokens = positions.shape[-1]
    sequence_parallel = self.use_attn_reduce_scatter_for_moe and not replicated_rows

    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    if sequence_parallel:
        hidden_states = _all_gather_rows(hidden_states)
        hidden_states = hidden_states.narrow(0, 0, full_num_tokens)

    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(hidden_states=hidden_states)
    elif self.layer_type == "full_attention":
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
        )
    else:
        raise ValueError("Invalid layer_type")

    if self.layer_scale:
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states * (
                self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
            )
        else:
            hidden_states = hidden_states * (
                self.attn_layer_scale.to(hidden_states.dtype) + 1
            )

    if sequence_parallel:
        _check_tokens_divide_by_tp(full_num_tokens)
        hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
    elif self.use_attn_reduce_scatter_for_moe:
        # The attention output projections skip their own all-reduce under
        # sequence-parallel MoE.
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    if self.use_attn_reduce_scatter_for_moe:
        # Either form: the block neither chunks nor gathers; the MoE
        # communicator gathers this rank's rows or, on a replicated step,
        # all-reduces the combine.
        hidden_states = self.mlp(
            hidden_states,
            already_sequence_parallel=True,
        )
    else:
        hidden_states = self.mlp(hidden_states)

    if self.layer_scale:
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states * (
                self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
            )
        else:
            if len(hidden_states.shape) != len(self.ffn_layer_scale.shape):
                raise RuntimeError(
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
            hidden_states = hidden_states * (
                self.ffn_layer_scale.to(hidden_states.dtype) + 1
            )

    return hidden_states, residual


def patched_model_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        if intermediate_tensors is None:
            raise RuntimeError(
                "intermediate_tensors must not be None when "
                "pipeline parallelism is active"
            )
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    assert isinstance(hidden_states, torch.Tensor)
    full_num_tokens = positions.shape[-1]
    replicated_rows = self.use_sequence_parallel and replicated_rows_step(
        full_num_tokens
    )
    sequence_parallel = self.use_sequence_parallel and not replicated_rows
    layer_kwargs = {"replicated_rows": True} if replicated_rows else {}
    if sequence_parallel:
        _check_tokens_divide_by_tp(full_num_tokens)
        hidden_states = qwen3_next.sequence_parallel_chunk(hidden_states)

    aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
    for layer_idx, layer in enumerate(
        islice(self.layers, self.start_layer, self.end_layer),
        start=self.start_layer,
    ):
        hidden_states, residual = layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
            **layer_kwargs,
        )
        self._maybe_add_hidden_state(
            aux_hidden_states, layer_idx + 1, hidden_states, residual
        )

    if not get_pp_group().is_last_rank:
        assert isinstance(residual, torch.Tensor)
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )
    hidden_states, _ = self.norm(hidden_states, residual)
    if sequence_parallel:
        if aux_hidden_states:
            hidden_size = hidden_states.shape[-1]
            hidden_states = torch.cat([hidden_states, *aux_hidden_states], dim=-1)
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states.narrow(0, 0, full_num_tokens)
            hidden_states, *aux_hidden_states = hidden_states.split(hidden_size, dim=-1)
        else:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states.narrow(0, 0, full_num_tokens)
    if aux_hidden_states:
        return hidden_states, aux_hidden_states
    return hidden_states


def maybe_patch_qwen3_next(*args: object, **kwargs: object) -> None:
    """Install trace-stable forwards when TPU_MOE_SEQUENCE_PARALLEL=1."""
    if not envs.TPU_MOE_SEQUENCE_PARALLEL:
        return None
    qwen3_next.Qwen3NextDecoderLayer.forward = patched_decoder_layer_forward
    qwen3_next.Qwen3NextModel.forward = patched_model_forward
    logger.info(
        "Applied TPU patch: Qwen3.5 MoE trace-stable sequence-parallel forward."
    )
    return None

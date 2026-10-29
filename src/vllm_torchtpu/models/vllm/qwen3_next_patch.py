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
"""

import threading
from itertools import islice

import torch
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.models import qwen3_next
from vllm.sequence import IntermediateTensors

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.dcp import get_dcp_group
from vllm_torchtpu.layers.adapter.cp_attention import (
    DCPGroupRowParallelLinear,
    is_dcp_duplicate_o_proj_enabled,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

UPSTREAM_DECODER_LAYER_FORWARD = qwen3_next.Qwen3NextDecoderLayer.forward
UPSTREAM_MODEL_FORWARD = qwen3_next.Qwen3NextModel.forward
UPSTREAM_ATTENTION_INIT = qwen3_next.Qwen3NextAttention.__init__
UPSTREAM_PROJECT_QKV_GATE = qwen3_next.Qwen3NextAttention._project_qkv_gate
UPSTREAM_ATTENTION_FORWARD = qwen3_next.Qwen3NextAttention.forward

_INIT_LOCK = threading.Lock()


def patched_attention_init(self, *args, **kwargs):
    use_dcp_o_dup = is_dcp_duplicate_o_proj_enabled()
    with _INIT_LOCK:
        orig_row_parallel = qwen3_next.RowParallelLinear
        if use_dcp_o_dup:
            qwen3_next.RowParallelLinear = DCPGroupRowParallelLinear
        try:
            UPSTREAM_ATTENTION_INIT(self, *args, **kwargs)
        finally:
            qwen3_next.RowParallelLinear = orig_row_parallel

    self.dcp_duplicate_o_proj = bool(
        use_dcp_o_dup and getattr(self.o_proj, "orep_active", False)
    )
    if self.dcp_duplicate_o_proj:
        assert getattr(self, "num_kv_heads", 1) == 1, (
            "DCP duplicate o_proj requires num_kv_heads == 1 per rank "
            f"(got {getattr(self, 'num_kv_heads', None)})."
        )
        self.attn.dcp_duplicate_o_proj = True
        impl = getattr(self.attn, "impl", None)
        if impl is not None:
            impl.dcp_duplicate_o_proj = True
        self.use_fused_qk_norm_rope_gate = False


def patched_project_qkv_gate(self, qkv, positions):
    q, k, v, gate = UPSTREAM_PROJECT_QKV_GATE(self, qkv, positions)
    if getattr(self, "dcp_duplicate_o_proj", False):
        dcp_group = get_dcp_group()
        if dcp_group is None:
            raise RuntimeError(
                "DCP group not initialized; cannot run dcp_duplicate_o_proj."
            )
        dcp_size = int(dcp_group.world_size)
        if dcp_size > 1 and q.shape[-1] == self.num_heads * self.head_dim:
            q_reshaped = q.view(-1, self.num_heads, self.head_dim)
            if gate is not None:
                orig_gate_shape = gate.shape[:-1]
                gate_reshaped = gate.view(-1, self.num_heads, self.head_dim)
                qg_local = torch.cat([q_reshaped, gate_reshaped], dim=0)
                qg_across_dcp = dcp_group.all_gather(qg_local, dim=1)
                q_across, gate_across = torch.chunk(qg_across_dcp, 2, dim=0)
                q = q_across.reshape(-1, self.num_heads * dcp_size * self.head_dim)
                gate = gate_across.reshape(
                    *orig_gate_shape, self.num_heads * dcp_size * self.head_dim
                )
            else:
                q_across = dcp_group.all_gather(q_reshaped.contiguous(), dim=1)
                q = q_across.reshape(-1, self.num_heads * dcp_size * self.head_dim)
    return q, k, v, gate


def patched_attention_forward(self, positions, hidden_states):
    if not getattr(self, "dcp_duplicate_o_proj", False):
        return UPSTREAM_ATTENTION_FORWARD(self, positions, hidden_states)
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v, gate = self._project_qkv_gate(qkv, positions)
    dcp_size = int(getattr(self.o_proj, "group_size", 1))
    output_shape = torch.Size((k.shape[0], self.num_heads * dcp_size * self.head_dim))
    attn_output = self.attn(q, k, v, output_shape=output_shape)
    if gate is not None:
        attn_output = attn_output * torch.sigmoid(gate)
    output, _ = self.o_proj(attn_output)
    return output


def _check_tokens_divide_by_tp(num_tokens: int) -> None:
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size > 1:
        torch._check(num_tokens % tp_size == 0)


def patched_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    **kwargs: object,
):
    full_num_tokens = positions.shape[-1]

    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    if self.use_attn_reduce_scatter_for_moe:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
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

    if self.use_attn_reduce_scatter_for_moe:
        _check_tokens_divide_by_tp(full_num_tokens)
        hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    if self.use_attn_reduce_scatter_for_moe:
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
    if self.use_sequence_parallel:
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
    if self.use_sequence_parallel:
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
    """Install attention patches and trace-stable sequence-parallel forwards."""
    qwen3_next.Qwen3NextAttention.__init__ = patched_attention_init
    qwen3_next.Qwen3NextAttention._project_qkv_gate = patched_project_qkv_gate
    qwen3_next.Qwen3NextAttention.forward = patched_attention_forward

    if not envs.TPU_MOE_SEQUENCE_PARALLEL:
        return None
    qwen3_next.Qwen3NextDecoderLayer.forward = patched_decoder_layer_forward
    qwen3_next.Qwen3NextModel.forward = patched_model_forward
    logger.info(
        "Applied TPU patch: Qwen3.5 MoE trace-stable sequence-parallel forward."
    )
    return None

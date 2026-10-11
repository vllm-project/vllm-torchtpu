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

from contextlib import contextmanager
from itertools import islice

import torch
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models import qwen3_next
from vllm.sequence import IntermediateTensors

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
    VllmGatedDeltaNetAttention as TpuQwen3NextGatedDeltaNet,
)
from vllm_torchtpu.layers.adapter.linear_common import (
    _get_prequant_matmul_op,
    _get_quantize_rows_op,
    project_prequant,
    quantize_rows,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

UPSTREAM_DECODER_LAYER_FORWARD = qwen3_next.Qwen3NextDecoderLayer.forward
UPSTREAM_MODEL_FORWARD = qwen3_next.Qwen3NextModel.forward
UPSTREAM_LINEAR_BASE_FORWARD = LinearBase.forward
UPSTREAM_GDN_FORWARD = TpuQwen3NextGatedDeltaNet.forward


def _check_tokens_divide_by_tp(num_tokens: int) -> None:
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size > 1:
        torch._check(num_tokens % tp_size == 0)


def _use_sequence_parallel_for_bucket(
    self,
    num_tokens: int,
    hidden_states: torch.Tensor | None = None,
) -> bool:
    if not getattr(
        self,
        "use_attn_reduce_scatter_for_moe",
        getattr(self, "use_sequence_parallel", False),
    ):
        return False
    # If the caller (such as Qwen3_5MultiTokenPredictor) has already chunked
    # hidden_states along the token axis, keep sequence parallelism active.
    if hidden_states is not None and hidden_states.shape[0] != num_tokens:
        return True
    min_tokens = envs.VLLM_TPU_SP_MIN_TOKENS
    return not (
        min_tokens > 0 and isinstance(num_tokens, int) and num_tokens < min_tokens
    )


def _run_non_sp_mlp(mlp: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Run `mlp(hidden_states)` as a non-sequence-parallel TP/EP MoE block."""
    saved: list[tuple[object, str, object]] = []
    if hasattr(mlp, "is_sequence_parallel"):
        saved.append((mlp, "is_sequence_parallel", mlp.is_sequence_parallel))
        mlp.is_sequence_parallel = False
    experts = getattr(mlp, "experts", None)
    if experts is not None:
        if hasattr(experts, "is_sequence_parallel"):
            saved.append(
                (experts, "is_sequence_parallel", experts.is_sequence_parallel)
            )
            experts.is_sequence_parallel = False
        moe_config = getattr(experts, "moe_config", None)
        if moe_config is not None and hasattr(moe_config, "is_sequence_parallel"):
            saved.append(
                (moe_config, "is_sequence_parallel", moe_config.is_sequence_parallel)
            )
            moe_config.is_sequence_parallel = False
    shared_expert = getattr(mlp, "shared_expert", None)
    if shared_expert is not None and hasattr(shared_expert, "is_sequence_parallel"):
        saved.append(
            (
                shared_expert,
                "is_sequence_parallel",
                shared_expert.is_sequence_parallel,
            )
        )
        shared_expert.is_sequence_parallel = False
        down_proj = getattr(shared_expert, "down_proj", None)
        if down_proj is not None and getattr(down_proj, "tp_size", 1) == 1:
            saved.append((mlp, "_scale_replicated_shared_expert", True))
    try:
        out = mlp(hidden_states)
        if any(attr == "_scale_replicated_shared_expert" for _, attr, _ in saved):
            tp_size = get_tensor_model_parallel_world_size()
            if tp_size > 1 and shared_expert is not None:
                shared_out = shared_expert(hidden_states)
                out = out - shared_out + (shared_out / tp_size)
                out = tensor_model_parallel_all_reduce(out)
        return out
    finally:
        for obj, attr, val in saved:
            if attr != "_scale_replicated_shared_expert":
                setattr(obj, attr, val)


def _is_per_channel_fp8_linear(layer: object) -> bool:
    if not isinstance(getattr(layer, "quant_method", None), VllmFp8LinearMethodTPU):
        return False
    weight = getattr(layer, "weight", None)
    scale = getattr(layer, "weight_scale", None)
    return (
        weight is not None
        and weight.dtype == torch.float8_e4m3fn
        and scale is not None
        and scale.dim() == 1
    )


def _can_use_sp_fp8_gather(self, num_tokens: int | None = None) -> bool:
    """True when the attention block's input projections are per-channel FP8."""
    if not envs.VLLM_TPU_SEQUENCE_PARALLEL_FP8_GATHER:
        return False
    min_tokens = envs.VLLM_TPU_SP_FP8_GATHER_MIN_TOKENS
    if min_tokens > 0 and isinstance(num_tokens, int) and num_tokens < min_tokens:
        return False
    if self.layer_type == "linear_attention":
        attn = self.linear_attn
        if not isinstance(attn, TpuQwen3NextGatedDeltaNet):
            return False
        if attn._has_active_pooled_pcp_state(attn.kv_cache):
            return False
        projs = (
            (attn.in_proj_qkv, attn.in_proj_ba, attn.in_proj_z)
            if hasattr(attn, "in_proj_qkv")
            else (attn.in_proj_qkvz, attn.in_proj_ba)
        )
        return all(_is_per_channel_fp8_linear(p) for p in projs)
    if self.layer_type == "full_attention":
        qkv_proj = getattr(self.self_attn, "qkv_proj", None)
        return _is_per_channel_fp8_linear(qkv_proj)
    return False


@contextmanager
def _with_prequant_qkv_proj(
    qkv_proj: LinearBase, prequant: tuple[torch.Tensor, torch.Tensor]
):
    """Temporarily feed pre-quantized `(x_q, x_s)` into `qkv_proj`."""
    qkv_proj._tpu_sp_prequant = prequant
    try:
        yield
    finally:
        qkv_proj._tpu_sp_prequant = None


def tpu_sp_fp8_gather_LinearBase(
    self: LinearBase,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.nn.Parameter | None]:
    prequant = getattr(self, "_tpu_sp_prequant", None)
    if prequant is not None:
        return project_prequant(self, x, prequant), None
    return UPSTREAM_LINEAR_BASE_FORWARD(self, x)


def tpu_sp_fp8_gather_Qwen3NextGatedDeltaNet(
    self: TpuQwen3NextGatedDeltaNet,
    hidden_states: torch.Tensor,
    prequant: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    if prequant is None:
        prequant = getattr(self, "_tpu_sp_prequant", None)
    return UPSTREAM_GDN_FORWARD(self, hidden_states, prequant=prequant)


def patched_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    **kwargs: object,
):
    full_num_tokens = positions.shape[-1]
    use_sp_bucket = _use_sequence_parallel_for_bucket(
        self, full_num_tokens, hidden_states
    )

    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    prequant: tuple[torch.Tensor, torch.Tensor] | None = None
    if use_sp_bucket:
        if _can_use_sp_fp8_gather(self, full_num_tokens):
            x_q, x_s = quantize_rows(hidden_states)
            x_q = tensor_model_parallel_all_gather(x_q, 0).narrow(0, 0, full_num_tokens)
            x_s = tensor_model_parallel_all_gather(x_s, 0).narrow(0, 0, full_num_tokens)
            prequant = (x_q, x_s)
            hidden_states = x_q
        else:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states.narrow(0, 0, full_num_tokens)

    if self.layer_type == "linear_attention":
        if prequant is not None:
            self.linear_attn._tpu_sp_prequant = prequant
            try:
                hidden_states = self.linear_attn(hidden_states=hidden_states)
            finally:
                self.linear_attn._tpu_sp_prequant = None
        else:
            hidden_states = self.linear_attn(hidden_states=hidden_states)
    elif self.layer_type == "full_attention":
        if prequant is not None:
            with _with_prequant_qkv_proj(self.self_attn.qkv_proj, prequant):
                hidden_states = self.self_attn(
                    hidden_states=hidden_states,
                    positions=positions,
                )
        else:
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

    if use_sp_bucket:
        _check_tokens_divide_by_tp(full_num_tokens)
        hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
    elif self.use_attn_reduce_scatter_for_moe:
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    if use_sp_bucket:
        hidden_states = self.mlp(
            hidden_states,
            already_sequence_parallel=True,
        )
    elif self.use_attn_reduce_scatter_for_moe:
        hidden_states = _run_non_sp_mlp(self.mlp, hidden_states)
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
    use_sp_bucket = self.use_sequence_parallel and _use_sequence_parallel_for_bucket(
        self, full_num_tokens
    )
    if use_sp_bucket:
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
    if use_sp_bucket:
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
    if envs.VLLM_TPU_SEQUENCE_PARALLEL_FP8_GATHER:
        _get_quantize_rows_op()
        _get_prequant_matmul_op()
        LinearBase.forward = tpu_sp_fp8_gather_LinearBase
        TpuQwen3NextGatedDeltaNet.forward = tpu_sp_fp8_gather_Qwen3NextGatedDeltaNet
    qwen3_next.Qwen3NextDecoderLayer.forward = patched_decoder_layer_forward
    qwen3_next.Qwen3NextModel.forward = patched_model_forward
    logger.info(
        "Applied TPU patch: Qwen3.5 MoE trace-stable sequence-parallel forward."
    )
    return None

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
"""Trace-stable sequence-parallel forwards for Qwen3-Next and Qwen3.5 MoE.

vLLM's Qwen3.5 decoder layer and model inherit these forwards from the
Qwen3Next classes, so the patch replaces them there.

Copies of the upstream vLLM forwards with three changes, so torch.compile
serves every bucket under TPU_MOE_SEQUENCE_PARALLEL=1 / VLLM_TPU_MOE_SEQUENCE_PARALLEL:
- torch._check(num_tokens % tp == 0) states that buckets divide by TP.
- narrow(0, 0, num_tokens) replaces [:num_tokens] after each all-gather to
  preserve exact symbolic output dimensions across TorchDynamo traces.
- The pad before the reduce-scatter is eliminated, removing redundant padding
  computations across layers.
"""

from itertools import islice
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.models import qwen3_next, qwen3_next_mtp
from vllm.sequence import IntermediateTensors

import vllm_torchtpu.envs as envs
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

linear_common: Any = None

UPSTREAM_ATTENTION_FORWARD = qwen3_next.Qwen3NextAttention.forward
UPSTREAM_DECODER_LAYER_FORWARD = qwen3_next.Qwen3NextDecoderLayer.forward
UPSTREAM_MODEL_FORWARD = qwen3_next.Qwen3NextModel.forward
UPSTREAM_MTP_FORWARD = qwen3_next_mtp.Qwen3NextMultiTokenPredictor.forward
UPSTREAM_MLP_INIT = qwen3_next.Qwen3NextMLP.__init__
_QWEN3_NEXT_MOE_MODEL_TYPES = frozenset(
    {"qwen3_next", "qwen3_5_moe", "qwen3_5_moe_text"}
)


def _get_linear_common() -> Any:
    global linear_common
    if linear_common is None:
        from vllm_torchtpu.layers.adapter import linear_common as _lc

        linear_common = _lc
    return linear_common


def _check_tokens_divide_by_tp(num_tokens: int) -> None:
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size > 1:
        torch._check(num_tokens % tp_size == 0)


def _should_use_sp(
    full_num_tokens: int | torch.SymInt, *, is_mtp: bool = False
) -> bool:
    """Whether this step should shard activations [T/TP, H] across TP ranks."""
    min_sp_tokens = envs.VLLM_TPU_SP_MIN_TOKENS
    if not min_sp_tokens and envs.USE_MOE_FUSED_TP_AR:
        from vllm_torchtpu.layers.adapter.fused_moe_tp_ar import MAX_DECODE_TOKENS

        min_sp_tokens = MAX_DECODE_TOKENS + 1
    if min_sp_tokens <= 2:
        return True
    from vllm_torchtpu.compilation import shape_variants

    # Submodules compiled outside shape_variants.warmup (such as the 1-layer
    # MTP drafter) keep a single token-independent graph without SP collectives.
    if is_mtp or (torch.compiler.is_compiling() and shape_variants._session is None):
        return False
    return bool(full_num_tokens >= min_sp_tokens)


def patched_mlp_init(
    self,
    *args: Any,
    **kwargs: Any,
) -> None:
    is_sequence_parallel = kwargs.get("is_sequence_parallel", False)
    prefix = kwargs.get("prefix", "")
    if (
        is_sequence_parallel
        and envs.USE_MOE_FUSED_TP_AR
        and prefix.endswith(".shared_expert")
    ):
        from vllm.config import get_current_vllm_config

        vllm_cfg = get_current_vllm_config()
        if not vllm_cfg.parallel_config.enable_expert_parallel:
            # Keep the shared expert TP-sharded so arm_fused_tp_ar can fuse its
            # [H, 2I/TP] and [I/TP, H] slices into the routed MoE kernel.
            kwargs["is_sequence_parallel"] = False
    UPSTREAM_MLP_INIT(self, *args, **kwargs)


def patched_attention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    prequant: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    if prequant is None:
        return UPSTREAM_ATTENTION_FORWARD(
            self, positions=positions, hidden_states=hidden_states
        )
    qkv = _get_linear_common().project_prequant(self.qkv_proj, hidden_states, prequant)
    q, k, v, gate = self._project_qkv_gate(qkv, positions)
    attn_output = self.attn(q, k, v)
    if gate is not None:
        attn_output = attn_output * torch.sigmoid(gate)
    output, _ = self.o_proj(attn_output)
    return output


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

    attn_kwargs: dict[str, object] = {}
    use_sp = self.use_attn_reduce_scatter_for_moe and _should_use_sp(
        full_num_tokens, is_mtp=getattr(self, "_is_mtp_layer", False)
    )
    if use_sp:
        from vllm_torchtpu.compilation import shape_variants

        min_fp8_tokens = envs.VLLM_TPU_SP_FP8_GATHER_MIN_TOKENS or (
            16 * get_tensor_model_parallel_world_size() ** 2 + 1
        )
        can_specialize_on_tokens = (
            min_fp8_tokens <= 2
            or not torch.compiler.is_compiling()
            or shape_variants._session is not None
        )
        if (
            envs.VLLM_TPU_SEQUENCE_PARALLEL_FP8_GATHER
            and can_specialize_on_tokens
            and full_num_tokens >= min_fp8_tokens
        ):
            x_q, x_s = _get_linear_common().quantize_rows(hidden_states)
            x_q = tensor_model_parallel_all_gather(x_q, 0).narrow(0, 0, full_num_tokens)
            x_s = (
                tensor_model_parallel_all_gather(x_s.squeeze(-1), 0)
                .narrow(0, 0, full_num_tokens)
                .unsqueeze(-1)
            )
            attn_kwargs["prequant"] = (x_q, x_s)
            hidden_states = x_q
        else:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states.narrow(0, 0, full_num_tokens)

    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(hidden_states=hidden_states, **attn_kwargs)
    elif self.layer_type == "full_attention":
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            **attn_kwargs,
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
        if use_sp:
            _check_tokens_divide_by_tp(full_num_tokens)
            hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
        else:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    if self.use_attn_reduce_scatter_for_moe:
        if envs.USE_MOE_FUSED_TP_AR:
            from vllm_torchtpu.layers.adapter import fused_moe_tp_ar

            prev_use_sp = fused_moe_tp_ar._CURRENT_LAYER_USE_SP
            fused_moe_tp_ar._CURRENT_LAYER_USE_SP = use_sp
            try:
                hidden_states = self.mlp(
                    hidden_states,
                    already_sequence_parallel=True,
                )
            finally:
                fused_moe_tp_ar._CURRENT_LAYER_USE_SP = prev_use_sp
        elif use_sp:
            hidden_states = self.mlp(
                hidden_states,
                already_sequence_parallel=True,
            )
        else:
            hidden_states = self.mlp(hidden_states)
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
    full_num_tokens = hidden_states.shape[0]
    if torch.compiler.is_compiling():
        positions = positions.reshape(*positions.shape[:-1], full_num_tokens)
    use_sp = self.use_sequence_parallel and _should_use_sp(full_num_tokens)
    if use_sp:
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
    if use_sp:
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


def patched_mtp_forward(
    self: qwen3_next_mtp.Qwen3NextMultiTokenPredictor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    spec_step_idx: int = 0,
) -> torch.Tensor:
    mtp_layer = self.layers[spec_step_idx % self.num_mtp_layers]
    if not mtp_layer.use_attn_reduce_scatter_for_moe:
        return UPSTREAM_MTP_FORWARD(
            self,
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx,
        )
    # Upstream enables use_attn_reduce_scatter_for_moe only when
    # pipeline_parallel_size == 1, so this rank is both first and last.
    if inputs_embeds is None:
        inputs_embeds = self.embed_input_ids(input_ids)
    if hidden_states.shape[-1] != inputs_embeds.shape[-1]:
        raise RuntimeError(
            f"hidden_states and inputs_embeds dimension mismatch: "
            f"{hidden_states.shape[-1]} vs {inputs_embeds.shape[-1]}"
        )
    hidden_states = self.fc(
        torch.cat(
            [
                self.pre_fc_norm_embedding(inputs_embeds),
                self.pre_fc_norm_hidden(hidden_states),
            ],
            dim=-1,
        )
    )
    full_num_tokens = hidden_states.shape[0]
    if torch.compiler.is_compiling():
        positions = positions.reshape(*positions.shape[:-1], full_num_tokens)
    mtp_layer._is_mtp_layer = True
    use_sp = _should_use_sp(full_num_tokens, is_mtp=True)
    if use_sp:
        _check_tokens_divide_by_tp(full_num_tokens)
        hidden_states = qwen3_next.sequence_parallel_chunk(hidden_states)
    hidden_states, residual = mtp_layer(
        positions=positions,
        hidden_states=hidden_states,
        residual=None,
    )
    hidden_states, _ = self.norm(hidden_states, residual)
    if use_sp:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0).narrow(
            0, 0, full_num_tokens
        )
    return hidden_states


def maybe_patch_qwen3_next(
    model_config: "ModelConfig | None" = None,
    *args: object,
    **kwargs: object,
) -> bool | None:
    """Install trace-stable forwards when sequence-parallel MoE is active."""
    mode = envs.VLLM_TPU_MOE_SEQUENCE_PARALLEL
    if mode == "off" and not envs.TPU_MOE_SEQUENCE_PARALLEL:
        return False if model_config is not None else None
    if model_config is not None:
        if (
            getattr(getattr(model_config, "hf_config", None), "model_type", None)
            not in _QWEN3_NEXT_MOE_MODEL_TYPES
        ):
            return False
    elif mode != "off" and not envs.TPU_MOE_SEQUENCE_PARALLEL:
        return False
    from vllm.model_executor.models import qwen3_5_mtp

    if envs.VLLM_TPU_SEQUENCE_PARALLEL_FP8_GATHER:
        lc = _get_linear_common()
        lc._get_quantize_rows_op()
        lc._get_prequant_matmul_op()
        qwen3_next.Qwen3NextAttention.forward = patched_attention_forward
    if envs.USE_MOE_FUSED_TP_AR:
        qwen3_next.Qwen3NextMLP.__init__ = patched_mlp_init
    qwen3_next.Qwen3NextDecoderLayer.forward = patched_decoder_layer_forward
    qwen3_next.Qwen3NextModel.forward = patched_model_forward
    qwen3_5_mtp.Qwen3_5MultiTokenPredictor.forward = patched_mtp_forward
    qwen3_next_mtp.Qwen3NextMultiTokenPredictor.forward = patched_mtp_forward
    logger.info(
        "Applied TPU patch: Qwen3.5 MoE trace-stable sequence-parallel forward "
        "(fp8_gather=%s, fp8_min_tokens=%d, sp_min_tokens=%d, fused_tp_ar=%s).",
        envs.VLLM_TPU_SEQUENCE_PARALLEL_FP8_GATHER,
        envs.VLLM_TPU_SP_FP8_GATHER_MIN_TOKENS,
        envs.VLLM_TPU_SP_MIN_TOKENS,
        envs.USE_MOE_FUSED_TP_AR,
    )
    return None

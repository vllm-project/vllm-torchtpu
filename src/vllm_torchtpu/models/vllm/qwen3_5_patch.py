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
"""Qwen3.5 MoE: overlap the attention TP all-reduce with MoE routing.

Upstream all-reduces the attention output, then every rank runs the
post-attention norm, the router gate, and the experts on all tokens. Here a
layer reduce-scatters ``attn_partial + residual / tp``, runs the norm and the
router gate on its token shard, all-gathers the normed tokens and the logits,
and adds its attention partial sum to its MoE partial sum so the MoE
all-reduce also completes the attention reduction. The layer returns the
pre-attention residual, so the next fused add+norm sees ``moe + attn +
residual``, the upstream stream.

When ``TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER`` is enabled, each layer replaces
the post-MoE all-reduce with a reduce-scatter so the residual stream stays
sharded across consecutive overlapped layers, all-gathers normed activations in
per-row ``float8_e4m3fn`` with ``float32`` scales before attention (layers 1+)
and MoE, and all-gathers the final output stream once at ``Qwen3_5Model.forward``
exit.
"""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING, Any

import torch
from torch import _check as torch_check
from vllm.distributed import (
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5Model

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.distributed.collectives import quantized_all_gather
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.qwen3_next_patch import all_gather_sharded_stream

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

_orig_init = Qwen3_5DecoderLayer.__init__
_orig_model_init = Qwen3_5Model.__init__
_orig_model_forward = Qwen3_5Model.forward

_MIN_PER_CHIP_SIZE_MB = 8


def _overlap_min_tokens(hidden_size: int, tp_size: int, element_size: int) -> int:
    return (_MIN_PER_CHIP_SIZE_MB * 1024 * 1024 * tp_size) // (
        hidden_size * element_size
    )


def _in_overlap_trace() -> bool:
    """Return True in eager mode or during a shape-variant backbone warmup trace."""
    # Only the backbone warmup gives each side of this branch its own trace.
    # The MTP drafter compiles outside warmup, so it keeps the all-reduce path.
    return (
        not torch.compiler.is_compiling() or shape_variants.aot_cache_tag() is not None
    )


def _narrow_all_gather(x: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """All-gather across TP ranks and trim padded rows to num_tokens."""
    return tensor_model_parallel_all_gather(x, 0).narrow(0, 0, num_tokens)


def _is_sharded_residual_active(
    layer: Qwen3_5DecoderLayer,
    num_tokens: int,
) -> bool:
    """Return True when the sharded-residual FP8 overlap path is active."""
    layer_any: Any = layer
    return (
        layer_any.uses_sharded_residual
        and _in_overlap_trace()
        and num_tokens >= layer_any.fp8_overlap_min_tokens
    )


def _configure_layer(layer: Qwen3_5DecoderLayer) -> None:
    mlp: Any = layer.mlp
    if mlp.tp_size <= 1:
        raise ValueError("TPU_ATTN_MOE_OVERLAP requires TP > 1")
    if layer.use_attn_reduce_scatter_for_moe:
        raise ValueError("TPU_ATTN_MOE_OVERLAP does not support sequence-parallel MoE")
    if mlp.replicate_shared_expert:
        # A replicated shared expert is added on every rank before the MoE
        # all-reduce; folding the attention rows into that all-reduce would
        # sum it tp times.
        raise ValueError(
            "TPU_ATTN_MOE_OVERLAP does not support replicate_shared_expert"
        )
    proj = (
        layer.linear_attn.out_proj
        if layer.layer_type == "linear_attention"
        else layer.self_attn.o_proj
    )
    proj.reduce_results = False
    # MoERunner recomputes router_logits from its own gate when one is set.
    experts: Any = mlp.experts
    experts.gate = None
    experts.moe_config.skip_final_all_reduce = True


def _patched_init(
    self: Qwen3_5DecoderLayer,
    vllm_config: Any,
    layer_type: str,
    prefix: str = "",
) -> None:
    _orig_init(self, vllm_config, layer_type, prefix=prefix)
    _configure_layer(self)
    # _patched_model_init enables this on backbone layers, whose
    # Qwen3_5Model.forward all-gathers the sharded output; MTP layers keep False.
    layer_any: Any = self
    layer_any.uses_sharded_residual = False
    layer_any.fp8_overlap_min_tokens = None


def _patched_model_init(
    self: Qwen3_5Model,
    *,
    vllm_config: VllmConfig,
    prefix: str = "",
) -> None:
    """Build the backbone and mark its layers for the sharded FP8 residual."""
    _orig_model_init(self, vllm_config=vllm_config, prefix=prefix)
    # Only backbone layers participate in Qwen3_5Model.forward's exit all-gather.
    layers: list[Any] = list(islice(self.layers, self.start_layer, self.end_layer))
    if layers:
        model_dtype = vllm_config.model_config.dtype
        if not isinstance(model_dtype, torch.dtype):
            msg = (
                "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER expects resolved torch.dtype "
                f"in vllm_config.model_config.dtype, got {model_dtype!r}"
            )
            raise TypeError(msg)
        tp_size: int = layers[0].mlp.tp_size
        hidden_size: int = self.config.hidden_size
        min_tokens = _overlap_min_tokens(hidden_size, tp_size, model_dtype.itemsize)
        for layer in layers:
            layer.uses_sharded_residual = True
            layer.fp8_overlap_min_tokens = min_tokens
        logger.info(
            "Qwen3.5 FP8 overlap all-gather is on for token buckets of at least "
            "%d MiB per TP rank (%d tokens at %s, TP%d).",
            _MIN_PER_CHIP_SIZE_MB,
            min_tokens,
            str(model_dtype).removeprefix("torch."),
            tp_size,
        )


def _overlap_forward(
    self: Qwen3_5DecoderLayer,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    num_tokens = positions.shape[-1]
    mlp: Any = self.mlp
    tp_size: int = mlp.tp_size
    fp8_overlap = _is_sharded_residual_active(self, num_tokens)
    is_first_layer = residual is None

    if is_first_layer:
        if fp8_overlap:
            # Unify input_ids.shape[0] and positions.shape[-1] in ShapeEnv so
            # layer 0's reduce-scatter matches subsequent narrows to num_tokens.
            torch_check(
                hidden_states.shape[0] == num_tokens,
                lambda: "hidden_states and positions token counts must match",
            )
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if fp8_overlap:
            hidden_states = quantized_all_gather(hidden_states, get_tp_group()).narrow(
                0, 0, num_tokens
            )

    if self.layer_type == "linear_attention":
        attn = self.linear_attn(hidden_states=hidden_states)
    else:
        attn = self.self_attn(hidden_states=hidden_states, positions=positions)
    # out_proj / o_proj run with reduce_results=False: per-rank partial sum.
    if self.layer_scale:
        attn = attn * (self.attn_layer_scale.to(attn.dtype)[0] + 1)

    layer_any: Any = self
    if layer_any.uses_sharded_residual:
        overlap = fp8_overlap
    else:
        min_tokens = _overlap_min_tokens(attn.shape[-1], tp_size, attn.element_size())
        overlap = _in_overlap_trace() and num_tokens >= min_tokens
    if overlap:
        if fp8_overlap and not is_first_layer:
            attn_loc = tensor_model_parallel_reduce_scatter(attn, 0)
            h_loc, residual = self.post_attention_layernorm(attn_loc, residual)
        else:
            x_loc = tensor_model_parallel_reduce_scatter(attn + residual / tp_size, 0)
            h_loc = self.post_attention_layernorm(x_loc)
            if fp8_overlap:
                residual = x_loc
        logits_loc, _ = mlp.gate(h_loc)
        logits = _narrow_all_gather(logits_loc, num_tokens)
        if fp8_overlap:
            h = quantized_all_gather(h_loc, get_tp_group()).narrow(0, 0, num_tokens)
        else:
            h = _narrow_all_gather(h_loc, num_tokens)
    else:
        h, residual = self.post_attention_layernorm(
            tensor_model_parallel_all_reduce(attn), residual
        )
        logits, _ = mlp.gate(h)
    out = mlp.experts(hidden_states=h, router_logits=logits)
    if self.layer_scale:
        out = out * (self.ffn_layer_scale.to(out.dtype)[0] + 1)
    if fp8_overlap:
        return tensor_model_parallel_reduce_scatter(out, 0), residual
    if overlap:
        out = out + attn
    return tensor_model_parallel_all_reduce(out), residual


def _overlap_model_forward(
    self: Qwen3_5Model,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
    """Run Qwen3.5 model forward and all-gather sharded FP8 overlap outputs at exit."""
    out: Any = _orig_model_forward(
        self,
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
    )
    num_tokens = positions.shape[-1]
    first_layer: Any = self.layers[self.start_layer]
    is_tuple = isinstance(out, tuple)
    hidden_states: torch.Tensor = out[0] if is_tuple else out
    if not _is_sharded_residual_active(first_layer, num_tokens):
        return out
    if not is_tuple:
        return all_gather_sharded_stream(hidden_states, [], num_tokens)[0]
    aux = list(out[1])
    # Layer id 0 is captured before layer 0 runs, so it still holds every row.
    full_rows = aux[:1] if 0 in self.aux_hidden_state_layers else []
    hidden_states, sharded = all_gather_sharded_stream(
        hidden_states, aux[len(full_rows) :], num_tokens
    )
    return hidden_states, [*full_rows, *sharded]


def maybe_patch_qwen3_5_attn_moe_overlap(
    model_config: ModelConfig | None,
) -> bool | None:
    """Returns False until a Qwen3.5 MoE model config is available."""
    if not envs.TPU_ATTN_MOE_OVERLAP:
        if envs.TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER:
            msg = (
                "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER requires TPU_ATTN_MOE_OVERLAP=True"
            )
            raise ValueError(msg)
        return None
    if (
        model_config is None
        or model_config.hf_text_config.model_type != "qwen3_5_moe_text"
    ):
        return False
    if envs.USE_MOE_FUSED_EP_KERNEL:
        raise ValueError("TPU_ATTN_MOE_OVERLAP requires USE_MOE_FUSED_EP_KERNEL=False")
    if envs.TPU_MOE_SEQUENCE_PARALLEL:
        raise ValueError(
            "TPU_ATTN_MOE_OVERLAP requires TPU_MOE_SEQUENCE_PARALLEL=False"
        )
    Qwen3_5DecoderLayer.__init__ = _patched_init
    Qwen3_5DecoderLayer.forward = _overlap_forward
    if envs.TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER:
        Qwen3_5Model.__init__ = _patched_model_init
        Qwen3_5Model.forward = _overlap_model_forward
    logger.info("Applied TPU patch: Qwen3.5 MoE attention-to-MoE collective overlap.")
    return None

# SPDX-License-Identifier: Apache-2.0
"""Row-gated sequence parallelism for the Qwen3-Next and Qwen3.5 decoders.

vLLM's `use_attn_reduce_scatter_for_moe` form is sequence parallel for every
step. Here a layer takes it only for a step of at least
`sequence_parallel.MIN_ROWS` rows and keeps the all-reduce form below that
(layers/adapter/sequence_parallel). The form is decided from the step's row
count, `positions.shape[-1]`: that leaves a shape guard, so each form is its
own trace; a Python flag would be frozen at the first trace.

The layer is still built for vLLM's form (projections without their
all-reduce, the MoE block and experts sequence parallel), then its
`use_attn_reduce_scatter_for_moe` is cleared so the model forward neither
chunks the input nor gathers the output; the first layer of a
sequence-parallel step slices its own rows after the reduce-scatter, and the
model forward gathers the rows back after the final norm.

Only the backbone takes it. The MTP draft ("eagle_head" graph) is compiled
for every bucket from one trace, which cannot carry both forms, so it is
built and run as vLLM's plain TP form.
"""

import functools

import torch

from vllm_torchtpu.layers.adapter import sequence_parallel
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Set on a layer built for the row-gated form.
SP_LAYER_ATTR = "_tpu_sp_by_rows"
# This rank's TP index as device data, so every rank compiles one program.
SP_RANK_ATTR = "_tpu_sp_rank"


def install() -> None:
    """Patch the decoder layers and the forwards around them, if selected."""
    sequence_parallel.check_flags()
    if not sequence_parallel.enabled():
        return
    from vllm.model_executor.models import qwen3_5, qwen3_next

    layer_cls = qwen3_next.Qwen3NextDecoderLayer
    if getattr(layer_cls, "_tpu_sp_by_rows_patch", False):
        return
    # Qwen3.5's layer has its own __init__ and skips Qwen3-Next's.
    for cls in (layer_cls, qwen3_5.Qwen3_5DecoderLayer):
        cls.__init__ = _build_row_gated(cls.__init__)
    fp8_gather = sequence_parallel.fp8_gather()
    layer_cls.forward = _row_gated_forward(layer_cls.forward, fp8_gather)
    if fp8_gather:
        attention_cls = qwen3_next.Qwen3NextAttention
        attention_cls.forward = _attention_forward(attention_cls.forward)
    model_cls = qwen3_next.Qwen3NextModel
    model_cls.forward = _gather_model_rows(model_cls.forward)
    layer_cls._tpu_sp_by_rows_patch = True
    logger.info(
        "Applied TPU patch: sequence parallelism for steps of at least %d rows.",
        sequence_parallel.MIN_ROWS,
    )


def _build_row_gated(original_init):
    @functools.wraps(original_init)
    def __init__(self, vllm_config, *args, **kwargs):
        # Read at call time: `set_model_tag` rebinds the module global.
        from vllm.compilation import backends
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        # Backbone only (module docstring). A replicated draft is also built
        # under a TP group collapsed to one rank.
        if (
            backends.model_tag == "backbone"
            and get_tensor_model_parallel_world_size() > 1
        ):
            with sequence_parallel.building_layer():
                original_init(self, vllm_config, *args, **kwargs)
        else:
            original_init(self, vllm_config, *args, **kwargs)
        row_gated = self.use_attn_reduce_scatter_for_moe
        setattr(self, SP_LAYER_ATTR, row_gated)
        if not row_gated:
            return

        tp_size = vllm_config.parallel_config.tensor_parallel_size
        sequence_parallel.check_buckets(
            vllm_config.compilation_config.compile_sizes or (), tp_size
        )
        self.use_attn_reduce_scatter_for_moe = False
        if sequence_parallel.fp8_gather():
            # Built here, before any trace: a build inside one is not allowed.
            from vllm_torchtpu.layers.adapter import linear_common

            linear_common._get_quantize_rows_op()
            linear_common._get_prequant_matmul_op()
        setattr(
            self,
            SP_RANK_ATTR,
            torch.tensor(
                [get_tensor_model_parallel_rank()],
                dtype=torch.int32,
                device=self.input_layernorm.weight.device,
            ),
        )
        sequence_parallel.record_layer()

    return __init__


def _row_gated_forward(original_forward, fp8_gather=False):
    from vllm.distributed import (
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
        tensor_model_parallel_reduce_scatter,
    )

    from vllm_torchtpu.layers.adapter import linear_common

    @functools.wraps(original_forward)
    def forward(self, hidden_states, residual, positions, **kwargs):
        if not getattr(self, SP_LAYER_ATTR, False):
            return original_forward(self, hidden_states, residual, positions, **kwargs)
        sp = positions.shape[-1] >= sequence_parallel.MIN_ROWS
        # The first layer takes the step's rows whole; a later layer of a
        # sequence-parallel step takes this rank's slice.
        first_layer = residual is None
        if first_layer:
            # Both forms lift the rank buffer, which only the sequence-parallel
            # one reads: a bucket is handed between traces only when their
            # inputs match (compilation/shape_variants).
            positions = positions + (getattr(self, SP_RANK_ATTR) * 0)[0]
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attn_kwargs = {}
        if sp and not first_layer and fp8_gather:
            # Each rank quantizes its own rows once; the in-projections take
            # the gathered fp8 rows and scales, and read only the row count
            # of `hidden_states`.
            x_q, x_s = linear_common.quantize_rows(hidden_states)
            x_q = tensor_model_parallel_all_gather(x_q, 0)
            attn_kwargs["prequant"] = (x_q, tensor_model_parallel_all_gather(x_s, 0))
            hidden_states = x_q
        elif sp and not first_layer:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states, **attn_kwargs)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states, positions=positions, **attn_kwargs
            )
        else:
            raise ValueError("Invalid layer_type")
        if self.layer_scale:
            hidden_states = _scale(hidden_states, self.attn_layer_scale)

        if sp:
            # Every sequence-parallel bucket divides by TP (check_buckets).
            hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
            if first_layer:
                tp_size = get_tensor_model_parallel_world_size()
                residual = (
                    residual.reshape(tp_size, -1, residual.shape[-1])
                    .index_select(0, getattr(self, SP_RANK_ATTR))
                    .squeeze(0)
                )
        else:
            # The projections were built without their all-reduce.
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # Own rows or replicated rows: the communicator exchanges accordingly
        # (sequence_parallel.set_active), and the block never chunks itself.
        hidden_states = self.mlp(hidden_states, already_sequence_parallel=True)
        if self.layer_scale:
            hidden_states = _scale(hidden_states, self.ffn_layer_scale)
        return hidden_states, residual

    return forward


def _attention_forward(original_forward):
    """Qwen3NextAttention.forward, with the in-projection from gathered fp8
    rows when the decoder passes them; otherwise vLLM's own."""

    @functools.wraps(original_forward)
    def forward(self, positions, hidden_states, prequant=None):
        if prequant is None:
            return original_forward(self, positions, hidden_states)
        qkv = sequence_parallel.project(self.qkv_proj, hidden_states, prequant)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        attn_output = self.attn(q, k, v)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        output, _ = self.o_proj(attn_output)
        return output

    return forward


def _scale(hidden_states, layer_scale):
    """vLLM's `x * (layer_scale + 1)`, for 2-D and 3-D `x`."""
    scale = layer_scale.to(hidden_states.dtype)
    if hidden_states.dim() == 2:
        scale = scale[0]
    return hidden_states * (scale + 1)


def _gather_rows(out, num_tokens):
    """Gather every output this rank holds a slice of back to the step's rows.

    Auxiliary hidden states taken before the first layer are whole already.
    """
    from vllm.distributed import tensor_model_parallel_all_gather

    if isinstance(out, tuple):
        hidden_states, aux_hidden_states = out
        return _gather_rows(hidden_states, num_tokens), [
            _gather_rows(aux, num_tokens) for aux in aux_hidden_states
        ]
    if out.shape[0] == num_tokens:
        return out
    return tensor_model_parallel_all_gather(out, 0)


def _gather_model_rows(original_forward):
    @functools.wraps(original_forward)
    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        out = original_forward(
            self, input_ids, positions, intermediate_tensors, inputs_embeds
        )
        first = self.layers[self.start_layer]
        if (
            getattr(first, SP_LAYER_ATTR, False)
            and positions.shape[-1] >= sequence_parallel.MIN_ROWS
        ):
            return _gather_rows(out, positions.shape[-1])
        return out

    return forward

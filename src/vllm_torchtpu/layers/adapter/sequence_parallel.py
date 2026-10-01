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
"""Sequence parallelism for the tensor-parallel prefill, gated by row count.

Selected with TPU_SEQUENCE_PARALLEL_PREFILL. A step of at least `MIN_ROWS`
rows reduce-scatters each attention output, so every rank runs the residual
add, the norms, the router and the shared expert on its own slice of the rows,
and the MoE dispatch all-gathers them for the experts. Smaller steps keep the
all-reduce form: they are latency-bound, and two collectives per layer beat
four.

The decoder layers (models/vllm/qwen3_next_sp_patch) take the form from the
row count, a shape, so each form is its own trace (compilation/shape_variants).
The communicator cannot see the row count, so the runner records the form
here before each backbone call (`set_active`); a trace reads it once. Only
the backbone takes it: the MTP draft runs the plain TP form.
"""

import contextlib
from collections.abc import Iterator, Sequence

import torch

from vllm_torchtpu import envs

# Rows from which a step is sequence parallel. PR #989's value, kept pending a
# probe: at TP8 its first sequence-parallel bucket (512) gives each rank 64
# rows, above the fused EP v3 step gate.
MIN_ROWS = 257

_active = False
# Set while a layer that implements the row-gated form is being built.
_building = False
_layers_built = 0


def enabled() -> bool:
    """Whether the operator selected sequence parallelism."""
    return envs.TPU_SEQUENCE_PARALLEL_PREFILL


def fp8_gather() -> bool:
    """Whether the rows are gathered as fp8 before the in-projections."""
    return enabled() and envs.TPU_SEQUENCE_PARALLEL_FP8_GATHER


def check_flags() -> None:
    """Raise on flags that cannot be combined with sequence parallelism."""
    if envs.TPU_SEQUENCE_PARALLEL_FP8_GATHER and not enabled():
        raise ValueError(
            "TPU_SEQUENCE_PARALLEL_FP8_GATHER needs TPU_SEQUENCE_PARALLEL_PREFILL=1."
        )
    if not enabled():
        return
    # Chunk pipelining owns the MoE exchange, over DP.
    if envs.TPU_MOE_COLLECTION_CHUNK_SIZE > 0:
        raise NotImplementedError(
            "TPU_SEQUENCE_PARALLEL_PREFILL exchanges MoE rows over the EP group, "
            "which TPU_MOE_COLLECTION_CHUNK_SIZE (DP chunk pipelining) does not "
            "serve. Unset one of them."
        )


def use_sequence_parallel_moe(parallel_config) -> bool:
    """`ParallelConfig.use_sequence_parallel_moe` on TPU.

    True only while a layer that implements the row-gated form is built, so
    no other model takes vLLM's always-on form, which the communicator does
    not serve.
    """
    return (
        _building
        and enabled()
        and parallel_config.tensor_parallel_size > 1
        and parallel_config.enable_expert_parallel
        and parallel_config.data_parallel_size == 1
        and parallel_config.prefill_context_parallel_size == 1
        and parallel_config.pipeline_parallel_size == 1
    )


@contextlib.contextmanager
def building_layer() -> Iterator[None]:
    """Build a decoder layer that implements the row-gated form."""
    global _building
    previous, _building = _building, True
    try:
        yield
    finally:
        _building = previous


def record_layer() -> None:
    global _layers_built
    _layers_built += 1


def check_buckets(compile_sizes: Sequence[int], tp_size: int) -> None:
    """Raise unless every sequence-parallel bucket splits evenly over TP."""
    uneven = [n for n in compile_sizes if n >= MIN_ROWS and n % tp_size]
    if uneven:
        raise NotImplementedError(
            f"TPU_SEQUENCE_PARALLEL_PREFILL splits every token bucket of at "
            f"least {MIN_ROWS} rows over tensor_parallel_size={tp_size}, and "
            f"buckets {uneven} do not divide. Choose --max-num-batched-tokens "
            f"and TPU_TOKEN_BUCKET_EXTRA as multiples of {tp_size}."
        )


def check_installed(tp_size: int) -> None:
    """Raise at boot when the operator selected sequence parallelism and no
    layer of the loaded model implements it."""
    if enabled() and _layers_built == 0:
        raise NotImplementedError(
            "TPU_SEQUENCE_PARALLEL_PREFILL serves Qwen3-Next and Qwen3.5 MoE "
            "decoders under --tensor-parallel-size>1 --enable-expert-parallel "
            "with data, prefill-context and pipeline parallelism 1; this "
            f"model or deployment (tensor_parallel_size={tp_size}) has no such "
            "layer. Unset TPU_SEQUENCE_PARALLEL_PREFILL."
        )


def step_is_sequence_parallel(num_tokens: int) -> bool:
    return enabled() and num_tokens >= MIN_ROWS


def set_active(active: bool) -> None:
    """Record the form of the compiled-model call about to run.

    Called outside the compiled region: a global written inside a traced
    forward becomes a graph output.
    """
    global _active
    _active = bool(active)


def is_active() -> bool:
    return _active


def rows_are_sliced(is_sequence_parallel: bool) -> bool:
    """Whether a MoE layer's rows are this rank's slice of the step."""
    return is_sequence_parallel and _active


def rows_are_replicated(is_sequence_parallel: bool) -> bool:
    """Whether a sequence-parallel MoE layer sees the step's rows replicated
    on every rank (a step below `MIN_ROWS`)."""
    return is_sequence_parallel and not _active


def project(
    layer: torch.nn.Module,
    x: torch.Tensor,
    prequant: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    """A column-parallel projection of the step's rows.

    `prequant` is the rows gathered as fp8 with their per-row scales
    (TPU_SEQUENCE_PARALLEL_FP8_GATHER), or None to project `x`. A per-channel
    fp8 projection takes them as they are; any other is given the rows
    dequantized.
    """
    if prequant is None:
        out, _ = layer(x)
        return out
    from vllm_torchtpu.layers.adapter.linear_common import (
        KEEP_VLLM_LAYOUT_ATTR,
        quantized_matmul_prequant,
    )
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

    x_q, x_s = prequant
    scale = getattr(layer, "weight_scale", None)
    if (
        isinstance(layer.quant_method, VllmFp8LinearMethodTPU)
        and layer.weight.dtype == torch.float8_e4m3fn
        and scale is not None
        and scale.dim() == 1
    ):
        weight = layer.weight
        if getattr(layer, KEEP_VLLM_LAYOUT_ATTR, False):
            weight = weight.transpose(0, 1)
        out = quantized_matmul_prequant(x_q, x_s, weight, scale)
        bias = None if layer.skip_bias_add else layer.bias
        return out if bias is None else out + bias
    out, _ = layer((x_q.to(torch.float32) * x_s).to(torch.bfloat16))
    return out

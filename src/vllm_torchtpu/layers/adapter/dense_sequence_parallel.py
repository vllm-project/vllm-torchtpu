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
"""Model-independent sequence-parallel helpers for dense tensor-parallel layers.

Instead of all-reducing each row-parallel linear's partial sum and repeating
RMSNorms, residual adds, and (on FP8 models) row quantization across all TP
ranks, sequence parallelism reduce-scatters the partial sums along the token
axis, runs the per-token ops on `num_tokens // tp_size` tokens per rank, and
all-gathers the next column-parallel linear's inputs (1-byte FP8 rows plus f32
row scales for per-channel FP8 linears, or bf16 rows for unquantized linears).
"""

import torch
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.layers.adapter.custom_ops import fp8_linear_op
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

# Measured on tpu7x at TP=8 with Gemma 4 31B (2-round ABBA pure-prefill medians):
# XLA sequence parallelism regresses at 16 rows/rank (`128` tokens, +1.3%),
# saves sub-millisecond at 32 rows/rank (`256` tokens, -5.2%), and wins clearly
# at 64–128+ rows/rank (`512` tokens -13.2%, `1024` tokens -24.0%). 128 rows/rank
# (`1024` at TP=8) matches the 128-row TPU tile / streaming-RS block size while
# keeping smaller decode/chunk steps on all-reduce (or single-shot VMEM RS+AG).
MIN_TOKENS_PER_RANK = 128


def validate_sequence_parallel_env(enabled: bool) -> None:
    """Rejects sub-flags that require `tpu_dense_sequence_parallel`."""
    if envs.TPU_DENSE_SP_HIER_RS != "off" and not enabled:
        raise ValueError(
            "TPU_DENSE_SP_HIER_RS requires "
            "additional_config['tpu_dense_sequence_parallel'] = True"
        )


def is_unquantized_linear(linear: LinearBase) -> bool:
    """Whether `linear` uses `UnquantizedLinearMethod`."""
    return isinstance(linear.quant_method, UnquantizedLinearMethod)


def validate_fp8_linear_for_sp(linear: LinearBase) -> VllmFp8LinearMethodTPU:
    """Validates at layer init that `linear` uses bias-free per-channel FP8."""
    quant_method = linear.quant_method
    if not isinstance(quant_method, VllmFp8LinearMethodTPU):
        raise NotImplementedError(
            f"{linear.prefix}: tpu_dense_sequence_parallel requires "
            f"VllmFp8LinearMethodTPU, got {type(quant_method).__name__}"
        )
    quant_method.validate_prequantized_config(linear)
    return quant_method


def validate_linear_for_sp(linear: LinearBase) -> None:
    """Validates at layer init that `linear` supports sequence-parallel gather."""
    if is_unquantized_linear(linear):
        return
    if not isinstance(linear.quant_method, VllmFp8LinearMethodTPU):
        raise NotImplementedError(
            f"{linear.prefix}: tpu_dense_sequence_parallel requires "
            "VllmFp8LinearMethodTPU or UnquantizedLinearMethod, got "
            f"{type(linear.quant_method).__name__}"
        )
    validate_fp8_linear_for_sp(linear)


def use_sequence_parallel(num_tokens: int | torch.SymInt, tp_size: int) -> bool:
    """Whether a padded `num_tokens` bucket runs on the sequence-parallel path."""
    return (
        shape_variants.can_branch_on_token_count()
        and num_tokens % tp_size == 0
        and num_tokens // tp_size >= MIN_TOKENS_PER_RANK
    )


def slice_token_shard(
    x: torch.Tensor,
    num_tokens: int | torch.SymInt,
    tp_rank: torch.Tensor,
    tp_size: int,
) -> torch.Tensor:
    """Slices this rank's contiguous token shard from replicated `[num_tokens, ...]`.

    `tp_rank` is a 1-element int32 per-rank buffer rather than a Python `int` so
    every TP rank traces the same FX graph and fingerprint for peer replay.
    """
    shard_tokens = num_tokens // tp_size
    sharded = x.reshape(tp_size, shard_tokens, *x.shape[1:])
    return torch.index_select(sharded, 0, tp_rank).squeeze(0)


def replicated_tokens(
    x: torch.Tensor,
    num_tokens: int | torch.SymInt,
    tp_rank: torch.Tensor,
) -> torch.Tensor:
    """Returns replicated `[num_tokens, ...]` while lifting the same symbolic
    token count and rank buffer placeholder as `slice_token_shard`."""
    narrowed = x.narrow(0, 0, num_tokens)
    return torch.index_select(narrowed.unsqueeze(0), 0, tp_rank * 0).squeeze(0)


def all_gather_tokens(
    x_shard: torch.Tensor, num_tokens: int | torch.SymInt
) -> torch.Tensor:
    """All-gathers token shards across TP ranks, preserving symbolic `num_tokens`."""
    return tensor_model_parallel_all_gather(x_shard, 0).narrow(0, 0, num_tokens)


def apply_prequantized_fp8_linear(
    linear: LinearBase, x_q: torch.Tensor, x_scale: torch.Tensor
) -> torch.Tensor:
    """Applies `linear`'s pre-quantized FP8 matmul to `(x_q, x_scale)`."""
    quant_method = linear.quant_method
    if not isinstance(quant_method, VllmFp8LinearMethodTPU):
        raise NotImplementedError(
            f"{linear.prefix}: tpu_dense_sequence_parallel requires "
            f"VllmFp8LinearMethodTPU, got {type(quant_method).__name__}"
        )
    return quant_method.apply_prequantized(linear, x_q, x_scale)


def all_gather_prequantized_fp8_linear(
    linear: LinearBase,
    x_q_shard: torch.Tensor,
    x_scale_shard: torch.Tensor,
    num_tokens: int | torch.SymInt,
) -> torch.Tensor:
    """All-gathers pre-quantized FP8 shards and applies `linear` over all tokens."""
    return apply_prequantized_fp8_linear(
        linear,
        all_gather_tokens(x_q_shard, num_tokens),
        all_gather_tokens(x_scale_shard, num_tokens),
    )


def quantize_all_gather_linear(
    linear: LinearBase, x_shard: torch.Tensor, num_tokens: int | torch.SymInt
) -> torch.Tensor:
    """Quantizes `x_shard` to FP8, all-gathers across TP, and applies `linear`."""
    x_q_shard, x_scale_shard = fp8_linear_op.quantize_fp8_activations(x_shard)
    return all_gather_prequantized_fp8_linear(
        linear, x_q_shard, x_scale_shard, num_tokens
    )


def all_gather_linear(
    linear: LinearBase, x_shard: torch.Tensor, num_tokens: int | torch.SymInt
) -> torch.Tensor:
    """All-gathers `x_shard` (quantizing first for FP8 linears) and runs `linear`."""
    if is_unquantized_linear(linear):
        out, bias = linear(all_gather_tokens(x_shard, num_tokens))
        return out if bias is None else out + bias
    return quantize_all_gather_linear(linear, x_shard, num_tokens)

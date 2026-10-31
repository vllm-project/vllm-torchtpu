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
"""Torch bridge for the streamed LM-head kernel in ``kernels/streamed_logits.py``."""

import torch
from torch_tpu._internal import pallas
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

from vllm_torchtpu.kernels.streamed_logits import block_rows, streamed_logits


def _fake_streamed_logits(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)


# Registered at import: compute_logits runs inside a compiled graph.
_op = pallas.jax_op("pallas::streamed_logits", streamed_logits)
_op.register_fake(_fake_streamed_logits)


def _is_plain_matmul(
    processor: LogitsProcessor, head: VocabParallelEmbedding, hidden: torch.Tensor
) -> bool:
    """Whether ``processor(head, hidden)`` is exactly ``hidden @ head.weight.T``."""
    w = head.weight
    return (
        hidden.device.type == "tpu"
        and hidden.dim() == 2
        and hidden.dtype == w.dtype == torch.bfloat16
        and isinstance(head.quant_method, UnquantizedEmbeddingMethod)
        and head.tp_size == 1
        and getattr(head, "bias", None) is None
        and not processor.logits_as_input
        and processor.soft_cap is None
        and processor.scale == 1.0
        and getattr(processor, "head_dtype", None) in (None, w.dtype)
        and processor.org_vocab_size == w.shape[0]
        and block_rows(w.shape[0]) > 0
    )


def compute_logits(
    processor: LogitsProcessor, head: VocabParallelEmbedding, hidden: torch.Tensor
) -> torch.Tensor | None:
    """``processor(head, hidden)``, through the streamed kernel when it is a
    plain bf16 head matmul."""
    if _is_plain_matmul(processor, head, hidden):
        return _op(hidden, head.weight)
    return processor(head, hidden)

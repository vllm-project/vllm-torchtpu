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
"""An LM head method on the e4m3 head kernel (``kernels/fp8_lm_head.py``):
the weight is quantized after loading, in place of the bf16 copy."""

import torch
from torch.nn import Parameter
from torch_tpu._internal import pallas
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

from vllm_torchtpu.kernels import fp8_lm_head as kernel
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import fresh_buffer, synchronize_tensors

logger = init_logger(__name__)


def _fake(x, weight, scale):
    return torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)


# Registered at import: Dynamo cannot trace a registration inside a forward.
_op = pallas.jax_op("pallas::fp8_lm_head", kernel.fp8_lm_head)
_op.register_fake(_fake)


def apply_head(
    rows: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """``[rows, vocab]`` logits of bf16 ``rows`` against a `quantize`d head."""
    return _op(rows, weight, scale)


def quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[vocab, hidden]`` -> (e4m3 ``[vocab, hidden]``, f32 ``[1, vocab]``
    per-row scale): each row's amax maps to e4m3's largest finite value.
    The scale gets a buffer of its own shape: as a view of the ``[vocab, 1]``
    amax, torch_tpu re-materialized it before every head call."""
    w = weight.float()
    amax = w.abs().amax(dim=1, keepdim=True)
    scale = torch.clamp(amax / kernel.E4M3_MAX, min=torch.finfo(torch.float32).tiny)
    return (w / scale).to(torch.float8_e4m3fn), fresh_buffer(scale.view(1, -1))


class Fp8LMHeadMethod(UnquantizedEmbeddingMethod):
    """`UnquantizedEmbeddingMethod` for a ParallelLMHead whose weight becomes
    e4m3 ``[vocab, hidden]`` with an f32 ``weight_scale [1, vocab]`` after
    loading. The head is only ever applied, never looked up."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        weight = layer.weight
        if weight.dtype == torch.float8_e4m3fn:
            return  # reload paths re-enter this hook
        q, scale = quantize(weight.data)
        layer.weight = Parameter(q, requires_grad=False)
        layer.weight_scale = Parameter(scale, requires_grad=False)
        if q.device.type == "tpu":
            synchronize_tensors([layer.weight, layer.weight_scale])
        logger.info_once(
            "LM head on an e4m3 weight with per-vocabulary-row scales %s.",
            tuple(q.shape),
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is not None:
            raise NotImplementedError("The e4m3 LM head takes no bias.")
        rows = x.reshape(-1, x.shape[-1])
        logits = apply_head(rows, layer.weight, layer.weight_scale)
        return logits.view(*x.shape[:-1], logits.shape[-1])

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("The e4m3 LM head is not an embedding table.")

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
"""Gemma4's GeGLU and its down projection's FP8 input in one pass
(TPU_GEMMA4_FUSED_GELU_QUANT).

Upstream `Gemma4MLP.forward` runs `GeluAndMul("tanh")` and hands its bf16
output to the W8A8 down projection, which quantizes it per token: XLA runs two
passes over the `[tokens, intermediate]` activation. With the flag,
`down_proj_from_gate_up` runs `linear_common.gelu_tanh_and_mul_quantize` (one
Pallas pass, kernels/gelu_quant.py) and the down projection's matmul on its
FP8 rows. It is a model patch because upstream exposes nothing between the
activation and the down projection's quantization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.models.gemma4 import Gemma4MLP

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gelu_quant import COLUMN_ALIGNMENT
from vllm_torchtpu.layers.adapter.linear_common import gelu_tanh_and_mul_quantize
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_orig_mlp_init = Gemma4MLP.__init__


def down_proj_from_gate_up(mlp: Any, gate_up: torch.Tensor) -> torch.Tensor:
    """`mlp.down_proj(mlp.act_fn(gate_up))[0]`, fused when
    TPU_GEMMA4_FUSED_GELU_QUANT is set; either way the result is reduced over
    the TP ranks only if the down projection reduces its results."""
    down = mlp.down_proj
    method = down.quant_method
    rows = gate_up.shape[0]
    if not envs.TPU_GEMMA4_FUSED_GELU_QUANT or rows <= method.w8a16_max_rows:
        # Buckets under TPU_FP8_LINEAR_W8A16_MAX_ROWS feed bf16 rows to the
        # down projection, so there is no quantization to fuse. Its forward
        # reduces by itself.
        return down(mlp.act_fn(gate_up))[0]
    if not method.supports_quantized_input(down, rows):
        raise NotImplementedError(
            "TPU_GEMMA4_FUSED_GELU_QUANT needs a per-channel W8A8 down "
            f"projection quantizing {gate_up.shape[0]} rows per token"
        )
    x_q, x_scale = gelu_tanh_and_mul_quantize(gate_up, down.weight)
    out = method.apply_quantized_input(down, x_q, x_scale)
    if down.reduce_results and down.tp_size > 1:
        out = tensor_model_parallel_all_reduce(out)
    return out


def _patched_mlp_init(self: Gemma4MLP, *args: Any, **kwargs: Any) -> None:
    """Upstream's init, then a refusal of what the fused kernel cannot run,
    so it surfaces at model build rather than at the first trace."""
    _orig_mlp_init(self, *args, **kwargs)
    act = self.act_fn
    if not isinstance(act, GeluAndMul) or act.approximate != "tanh":
        raise NotImplementedError(
            "TPU_GEMMA4_FUSED_GELU_QUANT needs GeluAndMul(approximate='tanh'), "
            f"got {type(act).__name__}(approximate="
            f"{getattr(act, 'approximate', None)!r})"
        )
    if not isinstance(self.down_proj.quant_method, VllmFp8LinearMethodTPU):
        raise NotImplementedError(
            "TPU_GEMMA4_FUSED_GELU_QUANT needs an FP8 down projection, got "
            f"{type(self.down_proj.quant_method).__name__}"
        )
    if self.down_proj.input_size_per_partition % COLUMN_ALIGNMENT:
        raise NotImplementedError(
            "TPU_GEMMA4_FUSED_GELU_QUANT needs an intermediate size per TP rank "
            f"that is a multiple of {COLUMN_ALIGNMENT}, got "
            f"{self.down_proj.input_size_per_partition}"
        )


def _patched_mlp_forward(self: Gemma4MLP, x: torch.Tensor) -> torch.Tensor:
    gate_up, _ = self.gate_up_proj(x)
    return down_proj_from_gate_up(self, gate_up)


def maybe_patch_gemma4_fused_gelu_quant(
    model_config: ModelConfig | None,
) -> bool | None:
    """Returns False until a Gemma4 text model config is available."""
    if not envs.TPU_GEMMA4_FUSED_GELU_QUANT:
        return None
    if model_config is None or model_config.hf_text_config.model_type != "gemma4_text":
        return False
    Gemma4MLP.__init__ = _patched_mlp_init
    Gemma4MLP.forward = _patched_mlp_forward
    logger.info(
        "Applied TPU patch: Gemma4 GeGLU and its down projection's FP8 input "
        "quantization run as one Pallas pass."
    )
    return None

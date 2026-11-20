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
"""GDN output projection with the gated norm and fp8 quantization fused.

With ``TPU_GDN_FUSED_NORM_QUANT`` set, :func:`install` checks every
gated-delta-net layer after loading and marks it, and the layer's output
projection runs :func:`output_projection`: one kernel produces the out-proj's
fp8 rows and scales from the GDN output and the gate, and the fp8 matmul runs
on them, instead of ``RMSNormGated`` followed by the fp8 linear.
"""

from collections.abc import Callable

import jax
import torch
from torch_tpu._internal import pallas
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)

from vllm_torchtpu.kernels.experimental import gated_norm_quant as gnq
from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_norm_ops: dict[float, Callable] = {}


def _build_ops(eps: float) -> None:
    if eps not in _norm_ops:

        def norm(
            o: jax.Array, z: jax.Array, w: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            return gnq.gated_norm_quant(o, z, w, eps)

        op = pallas.jax_op(f"pallas::gdn_gated_norm_quant_{len(_norm_ops)}", norm)

        def fake_norm(o, z, w):
            return (
                o.new_empty(o.shape, dtype=torch.float8_e4m3fn),
                o.new_empty((o.shape[0], 1), dtype=torch.float32),
            )

        op.register_fake(fake_norm)
        _norm_ops[eps] = op
    linear_common._get_prequant_matmul_op()


def unsupported_reason(layer: QwenGatedDeltaNetAttention) -> str | None:
    """Why the fused path cannot serve ``layer``, or None."""
    norm, out_proj = layer.norm, layer.out_proj
    if not isinstance(norm, RMSNormGated):
        return f"the output norm is {type(norm).__name__}, not RMSNormGated"
    if not norm.norm_before_gate:
        return "the output norm gates before normalizing"
    if norm.activation not in ("swish", "silu"):
        return f"the output norm gates with {norm.activation}, not silu"
    if norm.group_size not in (None, norm.weight.shape[0]):
        return f"the output norm groups by {norm.group_size}, not per head"
    if out_proj.weight.dtype != torch.float8_e4m3fn:
        return f"out_proj weight is {out_proj.weight.dtype}, not float8_e4m3fn"
    if out_proj.weight_scale.shape != (out_proj.weight.shape[-1],):
        return (
            f"out_proj scale is {tuple(out_proj.weight_scale.shape)}, not "
            "per-channel (set no REQUANTIZE_BLOCK_SIZE)"
        )
    if not getattr(out_proj, WEIGHT_FLIPPED_ATTR, False):
        return "out_proj weight is not in the (k, n) layout"
    if out_proj.bias is not None and not out_proj.skip_bias_add:
        return "out_proj has a bias"
    if not out_proj.input_is_parallel:
        return "out_proj input is not parallel"
    return None


def install(model: torch.nn.Module) -> int:
    """Route every GDN layer's output projection through the fused path.

    Call after the weights are processed. Refuses a model it cannot serve.
    """
    layers = [
        module
        for module in model.modules()
        if isinstance(module, QwenGatedDeltaNetAttention)
    ]
    if not layers:
        raise NotImplementedError(
            "TPU_GDN_FUSED_NORM_QUANT=1 but the model has no gated-delta-net "
            "layer; unset TPU_GDN_FUSED_NORM_QUANT."
        )
    for layer in layers:
        reason = unsupported_reason(layer)
        if reason is not None:
            raise NotImplementedError(
                f"TPU_GDN_FUSED_NORM_QUANT=1 cannot serve {layer.prefix}: "
                f"{reason}. Unset TPU_GDN_FUSED_NORM_QUANT."
            )
    for layer in layers:
        _build_ops(float(layer.norm.eps))
        layer.fused_norm_quant = True
    logger.info(
        "[GDN] TPU_GDN_FUSED_NORM_QUANT: %d layers run the gated norm and "
        "out_proj fp8 quantization as one kernel",
        len(layers),
    )
    return len(layers)


def output_projection(
    layer: QwenGatedDeltaNetAttention, core_attn_out: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """core_attn_out, z: [tokens, local v heads, head_v_dim]."""
    tokens = core_attn_out.shape[0]
    out_proj = layer.out_proj
    q, s = _norm_ops[float(layer.norm.eps)](
        core_attn_out, z.reshape(core_attn_out.shape), layer.norm.weight
    )
    output = linear_common.quantized_matmul_prequant(
        q.reshape(tokens, -1), s, out_proj.weight, out_proj.weight_scale
    )
    if out_proj.reduce_results and out_proj.tp_size > 1:
        output = tensor_model_parallel_all_reduce(output)
    return output

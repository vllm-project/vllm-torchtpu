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
"""Fused GDN output projection (TPU_GDN_FUSED_NORM_QUANT).

Replaces ``RMSNormGated`` plus the per-channel FP8 ``out_proj`` of a GDN
layer with two torch ops: ``kernels/gated_norm_quant.py`` turns the GDN
output and gate into the out-projection's FP8 rows and per-token scales in
one pass, and a matmul on those rows does the rest of the out-projection
without quantizing the activation again. Layers the fused path does not
cover keep the unfused path.
"""

import functools
from collections.abc import Callable

import jax
import torch
from torch_tpu._internal import pallas
from vllm.distributed import tensor_model_parallel_all_reduce

from vllm_torchtpu import envs
from vllm_torchtpu.kernels import gated_norm_quant
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_op_cache: dict[str, Callable] = {}


def _op_name_float(value: float) -> str:
    """A float rendered so it is safe inside a torch op name."""
    return f"{value:.6g}".replace(".", "p").replace("-", "m").replace("+", "")


def _gated_norm_quant_jax(
    attn_out: jax.Array,
    gate: jax.Array,
    weight: jax.Array,
    *,
    eps: float,
    max_tile: int,
) -> tuple[jax.Array, jax.Array]:
    return gated_norm_quant.gated_norm_quant(
        attn_out, gate, weight, eps, max_tile=max_tile
    )


def _fp8_matmul_on_quantized_rows_jax(
    x_q: jax.Array, x_scale: jax.Array, w_q: jax.Array, w_scale: jax.Array
) -> jax.Array:
    return gated_norm_quant.fp8_matmul_on_quantized_rows(x_q, x_scale, w_q, w_scale)


def _gated_norm_quant_op_name(eps: float, max_tile: int) -> str:
    # eps and the tile are baked into the traced op, so they name it.
    return f"pallas::gdn_gated_norm_quant_e{_op_name_float(eps)}_t{max_tile}"


_FP8_MATMUL_OP_NAME = "pallas::gdn_fp8_matmul_on_quantized_rows"


def _build_gated_norm_quant_op(eps: float, max_tile: int) -> Callable:
    op = pallas.jax_op(
        _gated_norm_quant_op_name(eps, max_tile),
        functools.partial(_gated_norm_quant_jax, eps=eps, max_tile=max_tile),
    )

    def _fake(attn_out, gate, weight):
        del gate, weight
        return (
            torch.empty(
                attn_out.shape, dtype=torch.float8_e4m3fn, device=attn_out.device
            ),
            torch.empty(
                (attn_out.shape[0], 1), dtype=torch.float32, device=attn_out.device
            ),
        )

    op.register_fake(_fake)
    return op


def _build_fp8_matmul_op() -> Callable:
    op = pallas.jax_op(_FP8_MATMUL_OP_NAME, _fp8_matmul_on_quantized_rows_jax)

    def _fake(x_q, x_scale, w_q, w_scale):
        del x_scale, w_scale
        return torch.empty(
            (x_q.shape[0], w_q.shape[1]), dtype=torch.bfloat16, device=x_q.device
        )

    op.register_fake(_fake)
    return op


def prebuild_ops(eps: float) -> None:
    """Build the fused ops for a gated norm with this ``eps``.

    ``pallas.jax_op`` inspects the wrapped function's signature, which Dynamo
    cannot trace, so the ops are built at load time and the forward only looks
    them up. One op serves every token bucket.
    """
    if not envs.TPU_GDN_FUSED_NORM_QUANT:
        return
    max_tile = int(envs.TPU_GATED_NORM_TILE)
    norm_quant_op_name = _gated_norm_quant_op_name(eps, max_tile)
    if norm_quant_op_name not in _op_cache:
        _op_cache[norm_quant_op_name] = _build_gated_norm_quant_op(eps, max_tile)
    if _FP8_MATMUL_OP_NAME not in _op_cache:
        _op_cache[_FP8_MATMUL_OP_NAME] = _build_fp8_matmul_op()


def _prebuilt_op(op_name: str) -> Callable:
    op = _op_cache.get(op_name)
    if op is None:
        raise RuntimeError(
            f"TPU_GDN_FUSED_NORM_QUANT: {op_name} was not built before the "
            "forward; call prebuild_ops at weight loading."
        )
    return op


def _unsupported_reason(
    norm: torch.nn.Module, out_proj: torch.nn.Module, num_tokens: int
) -> str | None:
    """Why the fused path cannot reproduce this layer's unfused output."""
    if not getattr(norm, "norm_before_gate", False):
        return "the gated norm gates before normalizing"
    if getattr(norm, "activation", None) not in ("swish", "silu"):
        return f"the gate activation is {getattr(norm, 'activation', None)}"
    if getattr(norm, "group_size", None) not in (None, norm.weight.shape[0]):
        return f"the norm groups by {norm.group_size}, not per head"
    weight = out_proj.weight
    weight_scale = getattr(out_proj, "weight_scale", None)
    if weight.dtype != torch.float8_e4m3fn:
        return f"out_proj weight dtype is {weight.dtype}"
    if weight_scale is None or weight_scale.ndim != 1:
        return "out_proj is not per-channel quantized"
    if getattr(out_proj, KEEP_VLLM_LAYOUT_ATTR, False) or not getattr(
        out_proj, WEIGHT_FLIPPED_ATTR, False
    ):
        return "out_proj weight is not in the [in, out] layout"
    if getattr(out_proj, "bias", None) is not None:
        return "out_proj has a bias"
    if not getattr(out_proj, "input_is_parallel", True):
        return "out_proj input is not parallel"
    w8a16_max_rows = getattr(out_proj.quant_method, "w8a16_max_rows", 0)
    if 0 < num_tokens <= w8a16_max_rows:
        # The unfused path keeps bf16 activations (W8A16) for this bucket.
        return "TPU_FP8_LINEAR_W8A16_MAX_ROWS covers this token bucket"
    return None


def can_fuse(norm: torch.nn.Module, out_proj: torch.nn.Module, num_tokens: int) -> bool:
    """Whether to run the fused output projection for this layer and bucket.

    Decided at trace time from static attributes and shapes.
    """
    if not envs.TPU_GDN_FUSED_NORM_QUANT:
        return False
    reason = _unsupported_reason(norm, out_proj, num_tokens)
    if reason is not None:
        logger.info_once(
            "TPU_GDN_FUSED_NORM_QUANT: unfused GDN output projection because %s.",
            reason,
        )
        return False
    return True


def fused_output_projection(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    norm: torch.nn.Module,
    out_proj: torch.nn.Module,
) -> torch.Tensor:
    """``out_proj(norm(core_attn_out, z))`` with the norm and quantization fused.

    ``core_attn_out`` is ``[tokens, heads, head_dim]``. Call only when
    ``can_fuse`` is true, after ``prebuild_ops``.
    """
    num_tokens = core_attn_out.shape[0]
    norm_quant_op = _prebuilt_op(
        _gated_norm_quant_op_name(float(norm.eps), int(envs.TPU_GATED_NORM_TILE))
    )
    quantized_rows, row_scales = norm_quant_op(
        core_attn_out, z.reshape(core_attn_out.shape), norm.weight
    )
    output = _prebuilt_op(_FP8_MATMUL_OP_NAME)(
        quantized_rows.reshape(num_tokens, -1),
        row_scales,
        out_proj.weight,
        out_proj.weight_scale,
    )
    # The all-reduce RowParallelLinear.forward would have done.
    if out_proj.reduce_results and out_proj.tp_size > 1:
        output = tensor_model_parallel_all_reduce(output)
    return output

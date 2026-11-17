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

Where the kernel was measured (the Qwen3.5 rule,
``measured.maybe_install_fused_norm_kernels``), :func:`install_after_loading`
has vLLM's loader call :func:`route_measured_model` on every model it
finishes, which calls :func:`install_if_supported`. :func:`install` checks
every gated-delta-net layer and marks it, and the layer's output
projection runs :func:`output_projection`: one kernel produces the out-proj's
fp8 rows and scales from the GDN output and the gate, and the fp8 matmul runs
on them, instead of ``RMSNormGated`` followed by the fp8 linear.

Custom-op registration and shape guards for fused post-GDN RMSNormGated +
FP8 quantization.
"""

import functools
from collections.abc import Callable, Collection
from typing import cast

import jax
import torch
from torch_tpu._internal import pallas
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)

from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.qwen3_5.kernels import gated_norm_quant as gnq

logger = init_logger(__name__)

# Steps of fewer rows run the JAX reference inside the same custom op.
MIN_ROWS = 16

# Marks the wrapper install_after_loading puts on the loader's post-processing.
_INSTALLED_ATTR = "_tpu_gdn_fused_norm_quant"

_out_proj_ops: dict[float, Callable] = {}


def _build_ops(eps: float) -> None:
    """Register the fused output projection op for one norm epsilon.

    The op computes the gated norm and the out_proj's per-token FP8 rows and
    scales in one kernel and runs the FP8 matmul on them; registering it
    inside a traced forward is not allowed, so :func:`install` builds it.
    """
    if eps in _out_proj_ops:
        return

    def fused_out_proj(
        o: jax.Array,
        z: jax.Array,
        w_norm: jax.Array,
        w_proj: jax.Array,
        w_scale: jax.Array,
    ) -> jax.Array:
        if MIN_ROWS > 0 and o.shape[0] < MIN_ROWS:
            q, s = gnq.reference(o, z, w_norm, eps)
        else:
            q, s = gnq.gated_norm_quant(o, z, w_norm, eps)
        return linear_common._prequant_matmul_jax(
            q.reshape(o.shape[0], -1), s, w_proj, w_scale
        )

    proj_op = pallas.jax_op(
        f"pallas::gdn_gated_norm_out_proj_{len(_out_proj_ops)}",
        fused_out_proj,
    )

    def fake_out_proj(o, z, w_norm, w_proj, w_scale):
        return o.new_empty((o.shape[0], w_proj.shape[-1]), dtype=torch.bfloat16)

    proj_op.register_fake(fake_out_proj)
    _out_proj_ops[eps] = proj_op


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
    weight_scale = cast(torch.Tensor, out_proj.weight_scale)
    if weight_scale.shape != (cast(torch.Tensor, out_proj.weight).shape[-1],):
        return (
            f"out_proj scale is {tuple(weight_scale.shape)}, not "
            "per-channel (set no REQUANTIZE_BLOCK_SIZE)"
        )
    if not getattr(out_proj, WEIGHT_FLIPPED_ATTR, False):
        return "out_proj weight is not in the (k, n) layout"
    if out_proj.bias is not None and not out_proj.skip_bias_add:
        return "out_proj has a bias"
    if not out_proj.input_is_parallel:
        return "out_proj input is not parallel"
    if getattr(out_proj.quant_method, "w8a16_max_rows", 0):
        # The fused path always quantizes the rows (W8A8); keep the linear's
        # own small-step W8A16 when TPU_FP8_LINEAR_W8A16_MAX_ROWS opts into it.
        return "out_proj runs small steps W8A16 (TPU_FP8_LINEAR_W8A16_MAX_ROWS)"
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
            "The fused GDN gated norm + FP8 quantization needs gated-delta-net "
            "layers, and the model has no gated-delta-net layer."
        )
    for layer in layers:
        reason = unsupported_reason(layer)
        if reason is not None:
            raise NotImplementedError(
                f"The fused GDN gated norm + FP8 quantization cannot serve "
                f"{layer.prefix}: {reason}."
            )
    for layer in layers:
        _build_ops(float(layer.norm.eps))
        # torch's Module.__setattr__ stub only accepts modules and tensors.
        # pyrefly: ignore[bad-argument-type]
        layer.fused_norm_quant = True
    logger.info(
        "[GDN] %d layers run the gated norm and out_proj fp8 quantization "
        "as one kernel",
        len(layers),
    )
    return len(layers)


def install_if_supported(
    model: torch.nn.Module, value_heads: Collection[int] | None = None
) -> int:
    """:func:`install` when every gated-delta-net layer can take the fused
    path, else nothing; the number of layers routed.

    The Qwen3.5 rule's entry point. A model without GDN layers (a pipeline
    stage or a draft without them), with a layer the kernel cannot serve
    (for example a BF16 or block-scaled out_proj), or, when ``value_heads``
    is given, with a layer whose value heads per device are not in it keeps
    the ``RMSNormGated`` + linear path.
    """
    layers = [
        module
        for module in model.modules()
        if isinstance(module, QwenGatedDeltaNetAttention)
    ]
    if not layers:
        return 0
    for layer in layers:
        reason = unsupported_reason(layer)
        if reason is None and value_heads is not None:
            heads = layer.num_v_heads // layer.tp_size
            if heads not in value_heads:
                reason = f"{heads} value heads per device were not measured"
        if reason is not None:
            logger.info(
                "[GDN] Fused gated norm + FP8 quantization not used: %s: %s.",
                layer.prefix,
                reason,
            )
            return 0
    return install(model)


def route_measured_model(model: torch.nn.Module, model_config) -> int:
    """Qwen3.5 rule: route the GDN output projections of a loaded model
    through the fused kernel where it was measured
    (``measured.is_measured_model``: Qwen3.5-397B-A17B on TPU7x, at the GDN
    value heads per device of its legs) and every GDN layer allows it.
    Other models, other Qwen3.5 checkpoints (the MTP draft among them) and
    other devices are untouched. Returns the number of layers routed."""
    # Imported here so that importing the GDN layer, which imports this
    # module, does not load the Qwen3.5 config hook the rule reads.
    from vllm_torchtpu.models.qwen3_5.kernels import measured

    if not measured.is_measured_model(model_config):
        return 0
    return install_if_supported(model, value_heads=measured.GDN_VALUE_HEADS)


def install_after_loading() -> None:
    """Call :func:`route_measured_model` on every model vLLM's loader builds,
    right after its weights are processed into their runtime layout (the
    out_proj's FP8 weight, scales and (k, n) layout the fused path checks).

    Idempotent. Every model a loader builds, pipeline stages and the draft
    included, passes through ``process_weights_after_loading``, so the
    routing needs no call of its own in the runner.
    """
    from vllm.model_executor.model_loader import base_loader

    original = base_loader.process_weights_after_loading
    if getattr(original, _INSTALLED_ATTR, False):
        return

    @functools.wraps(original)
    def process_weights_after_loading(model, model_config, target_device):
        result = original(model, model_config, target_device)
        route_measured_model(model, model_config)
        return result

    setattr(process_weights_after_loading, _INSTALLED_ATTR, True)
    base_loader.process_weights_after_loading = process_weights_after_loading


def output_projection(
    layer: QwenGatedDeltaNetAttention, core_attn_out: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """core_attn_out: [tokens, local v heads, head_v_dim]; z: 2D or 3D."""
    out_proj = layer.out_proj
    if z.ndim == 3:
        z = z.reshape(core_attn_out.shape[0], -1)
    output = _out_proj_ops[float(layer.norm.eps)](
        core_attn_out,
        z,
        layer.norm.weight,
        out_proj.weight,
        out_proj.weight_scale,
    )
    if out_proj.reduce_results and out_proj.tp_size > 1:
        output = tensor_model_parallel_all_reduce(output)
    return output

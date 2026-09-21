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
"""The torch ops for the fused gated norm and quantisation of the GDN output (TPU_GDN_FUSED_NORM_QUANT).

Two ops, built once at load (never inside a traced region): ``gated_norm_quant`` (kernels/gated_norm_quant.py)
turns the GDN kernel's bf16 output and the gate into the out-projection's fp8 rows and per-token scales in one
pass; ``prequant_matmul_in_out`` runs the [in, out] fp8 matmul on those rows, the same arithmetic as the linear
layer's own path minus the quantisation it would redo. The GDN op uses them in place of RMSNormGated plus the
fp8 linear when the flag is on and the out-projection stores its weight [in, out]."""
import jax
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels import gated_norm_quant as gnq
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)
_NORM_OP = None
_MATMUL_OP = None


def _matmul_fn(x_q: jax.Array, x_scale: jax.Array, w_q: jax.Array,
               w_s: jax.Array) -> jax.Array:
    # The op bridge builds the torch signature from these annotations and
    # refuses an unannotated argument, so the kernel's `out_dtype` keyword
    # stays out of the op's operand list.
    return gnq.prequant_matmul_in_out(x_q, x_scale, w_q, w_s)


def _find_eps(model) -> float:
    """The GDN layers' gated norm epsilon: one value for the model, every
    gated norm checked for the semantics the kernel assumes."""
    from vllm.model_executor.layers.layernorm import RMSNormGated
    found = set()
    for module in model.modules():
        if not isinstance(module, RMSNormGated):
            continue
        eps = getattr(module, "eps", None)
        if eps is None:
            eps = getattr(module, "variance_epsilon")
        if not getattr(module, "norm_before_gate", True):
            raise ValueError(
                "TPU_GDN_FUSED_NORM_QUANT: the kernel normalises before the gate; this norm gates first"
            )
        if getattr(module, "activation", "swish") not in ("swish", "silu"):
            raise ValueError(
                "TPU_GDN_FUSED_NORM_QUANT: the kernel gates with silu; this norm uses "
                f"{module.activation}")
        if getattr(module, "group_size",
                   None) not in (None, module.weight.shape[0]):
            raise ValueError(
                "TPU_GDN_FUSED_NORM_QUANT: the kernel normalises per head; this norm groups by "
                f"{module.group_size}")
        found.add(float(eps))
    if not found:
        raise ValueError(
            "TPU_GDN_FUSED_NORM_QUANT: no RMSNormGated on the model")
    if len(found) > 1:
        raise ValueError(
            "TPU_GDN_FUSED_NORM_QUANT: the gated norms use different epsilons "
            f"{sorted(found)}; the fused op takes one value for the model")
    return found.pop()


def install(model) -> bool:
    """Register the fused norm-and-quantise ops with the model's eps."""
    global _NORM_OP, _MATMUL_OP
    from torch_tpu._internal.pallas import pallas  # the torch<->jax op bridge
    eps = _find_eps(model)

    def _norm_fn(o: jax.Array, z: jax.Array,
                 w: jax.Array) -> tuple[jax.Array, jax.Array]:
        return gnq.gated_norm_quant(o, z, w, eps)

    norm = pallas.jax_op("pallas::gdn_gated_norm_quant", _norm_fn)

    def _fake_norm(o, z, w):
        return o.new_empty(o.shape, dtype=torch.float8_e4m3fn), o.new_empty(
            (o.shape[0], 1), dtype=torch.float32)

    norm.register_fake(_fake_norm)
    matmul = pallas.jax_op("pallas::gdn_prequant_matmul_in_out", _matmul_fn)

    def _fake_matmul(x_q, x_scale, w_q, w_s):
        return x_q.new_empty((x_q.shape[0], w_q.shape[1]),
                             dtype=torch.bfloat16)

    matmul.register_fake(_fake_matmul)
    _NORM_OP, _MATMUL_OP = norm, matmul
    logger.warning(
        "TPU_GDN_FUSED_NORM_QUANT: the GDN output's gated norm and fp8 quantisation run as one pass (eps %g)",
        eps)
    return True


def install_if_enabled(model) -> bool:
    """Install the fused ops when TPU_GDN_FUSED_NORM_QUANT is set."""
    if not envs.TPU_GDN_FUSED_NORM_QUANT:
        return False
    return install(model)


def enabled() -> bool:
    """Whether the fused ops are on and installed."""
    return bool(envs.TPU_GDN_FUSED_NORM_QUANT) and _NORM_OP is not None


def output_projection(core_attn_out: torch.Tensor, z: torch.Tensor, norm,
                      out_proj) -> torch.Tensor:
    """core_attn_out, z: [tokens, heads, head_dim]; norm: the RMSNormGated; out_proj: the fp8 [in, out] linear."""
    tokens = core_attn_out.shape[0]
    q, s = _NORM_OP(core_attn_out, z.reshape(core_attn_out.shape), norm.weight)
    return _MATMUL_OP(q.reshape(tokens, -1), s, out_proj.weight,
                      out_proj.weight_scale)

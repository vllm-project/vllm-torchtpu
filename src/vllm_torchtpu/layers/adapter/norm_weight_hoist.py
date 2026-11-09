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
"""Precompute each GemmaRMSNorm's ``1 + weight`` once (TPU_NORM_WEIGHT_HOIST).

vLLM's ``GemmaRMSNorm.forward_native`` computes ``weight.float() + 1.0`` in
every forward, so the compiled step converts and adds every norm's weight on
every call. After the weights are loaded that sum is a constant, so it is
computed once here, kept as a non-persistent buffer on the module, and the
forward reads the buffer. The numerics are unchanged: the same float32 sum,
done once instead of every step.
"""

import types

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The buffer name is shared with any op that wants the precomputed weight.
WEIGHT_PLUS_ONE_BUFFER = "_tpu_weight_plus_one"


def _forward_native_with_precomputed_weight(
    self, x: torch.Tensor, residual: torch.Tensor | None = None
):
    from vllm import ir

    weight = getattr(self, WEIGHT_PLUS_ONE_BUFFER, None)
    if weight is None:
        weight = self.weight.float() + 1.0
    if residual is None:
        return ir.ops.rms_norm(x, weight, self.variance_epsilon)
    return ir.ops.fused_add_rms_norm(x, residual, weight, self.variance_epsilon)


def install(model: torch.nn.Module) -> int:
    """Give every GemmaRMSNorm in ``model`` its float32 ``1 + weight`` buffer.

    Call after the weights are loaded. Returns the number of norms changed.
    """
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    original_forward_native = GemmaRMSNorm.forward_native
    num_norms = 0
    for module in model.modules():
        if not isinstance(module, GemmaRMSNorm):
            continue
        weight_plus_one = (module.weight.detach().float() + 1.0).contiguous()
        if hasattr(module, WEIGHT_PLUS_ONE_BUFFER):
            getattr(module, WEIGHT_PLUS_ONE_BUFFER).copy_(weight_plus_one)
        else:
            module.register_buffer(
                WEIGHT_PLUS_ONE_BUFFER, weight_plus_one, persistent=False
            )
        # When the custom op is disabled, CustomOp binds the original
        # forward_native at construction; rebind that instance to the new one.
        bound_forward = getattr(module, "_forward_method", None)
        if getattr(bound_forward, "__func__", None) is original_forward_native:
            module._forward_method = types.MethodType(
                _forward_native_with_precomputed_weight, module
            )
        num_norms += 1
    if num_norms:
        # When the custom op is enabled, forward_tpu / forward_oot call
        # self.forward_native at call time, so the class patch reaches them.
        GemmaRMSNorm.forward_native = _forward_native_with_precomputed_weight
    return num_norms


def install_if_enabled(model: torch.nn.Module) -> int:
    """Precompute the norm weights when TPU_NORM_WEIGHT_HOIST is set."""
    if not envs.TPU_NORM_WEIGHT_HOIST:
        return 0
    num_norms = install(model)
    logger.info(
        "TPU_NORM_WEIGHT_HOIST: %d GemmaRMSNorm layers use a precomputed "
        "float32 (1 + weight) buffer",
        num_norms,
    )
    return num_norms

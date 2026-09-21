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
"""The norm weights hoisted out of the step (TPU_NORM_WEIGHT_HOIST).

vLLM's GemmaRMSNorm computes ``weight.float() + 1.0`` inside every forward, so
the traced step converts and adds every norm's weight on every call. After the
weights are loaded the sum is a constant, so it is computed once here, kept
as a buffer on the module, and the forward uses the buffer. Numerics are
unchanged (the same f32 sum, done once)."""
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)
_BUFFER = "_tpu_weight_plus_one"


def _hoisted_forward_native(self,
                            x: torch.Tensor,
                            residual: torch.Tensor | None = None):
    from vllm import ir
    weight = getattr(self, _BUFFER, None)
    if weight is None:
        weight = self.weight.float() + 1.0
    if residual is None:
        return ir.ops.rms_norm(x, weight, self.variance_epsilon)
    return ir.ops.fused_add_rms_norm(x, residual, weight,
                                     self.variance_epsilon)


def install(model: torch.nn.Module) -> int:
    """Gives every GemmaRMSNorm in ``model`` its (1 + w) f32 buffer and routes the class's native forward
    through it. Returns the number of norms hoisted. Call after the weights are loaded."""
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    n = 0
    for module in model.modules():
        if isinstance(module, GemmaRMSNorm):
            value = (module.weight.detach().float() + 1.0).contiguous()
            if hasattr(module, _BUFFER):
                getattr(module, _BUFFER).copy_(value)
            else:
                module.register_buffer(_BUFFER, value, persistent=False)
            n += 1
    if n and not getattr(GemmaRMSNorm, "_tpu_norm_weight_hoisted", False):
        # CustomOp binds forward_tpu at construction and forward_tpu calls
        # self.forward_native at call time, so patching the class's
        # forward_native reaches every instance, built before or after. (With
        # the op disabled in vLLM's custom-op config the bound native forward
        # is captured at construction and this patch does not reach it.)
        GemmaRMSNorm.forward_native = _hoisted_forward_native
        GemmaRMSNorm._tpu_norm_weight_hoisted = True
    return n


def install_if_enabled(model: torch.nn.Module) -> int:
    """Hoist the norm weights when TPU_NORM_WEIGHT_HOIST is set; the count."""
    if not envs.TPU_NORM_WEIGHT_HOIST:
        return 0
    n = install(model)
    logger.warning(
        "TPU_NORM_WEIGHT_HOIST: %d norms carry (1 + w) in f32 as a buffer; the per-step convert and add are gone",
        n)
    return n

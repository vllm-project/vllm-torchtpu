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
"""RMSNorm that can read a float32 copy of its weight, made once at load.

The ``tpu`` RMSNorm providers in ``ir_ops.py`` multiply by the weight in
float32 (``weight.to(torch.float32)``). The checkpoint stores the weight in
bfloat16, so every forward step runs one convert op per norm. In a GLM-5.2
decode step (TP8, 16 tokens) each convert is a separate XLA op of about
0.6 us, and a decoder layer has four norms (input, post-attention, q_a and
kv_a), so the converts cost about 2.4 us per layer.

``arm_fp32_weights`` stores a float32 copy of each RMSNorm weight once, after
the weights are loaded, as the non-persistent buffer ``weight_fp32``.
``TpuRMSNorm.forward_native`` then passes that buffer to the IR op instead of
the bfloat16 weight. A bfloat16 to float32 conversion is exact, and
``weight.to(torch.float32)`` on a float32 tensor returns it unchanged, so the
``tpu`` providers compute bit-identical results.

The ``native`` provider rounds the normalized activation to ``weight.dtype``
before the weight multiply, so a float32 weight would change its result.
``arm_fp32_weights`` therefore arms only when the ``tpu`` provider is the
first choice of both IR ops (the platform default, see
``TpuPlatform.get_default_ir_op_priority``).

The copy is not refreshed if the bfloat16 weight is overwritten after load.
The feature is off by default (``TPU_RMSNORM_FP32_WEIGHT``) for that reason.
"""

import torch
from vllm import ir
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_torchtpu.layers.adapter.ir_ops import TPU_IR_PROVIDER
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

FP32_WEIGHT_BUFFER = "weight_fp32"


@CustomOp.register_oot(name="RMSNorm")
class TpuRMSNorm(RMSNorm):
    """``RMSNorm`` that uses ``weight_fp32`` when ``arm_fp32_weights`` set it.

    Without the buffer, the forward is the upstream ``forward_native``.
    """

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        weight_fp32 = getattr(self, FP32_WEIGHT_BUFFER, None)
        if weight_fp32 is None:
            return super().forward_native(x, residual)
        if residual is None:
            return ir.ops.rms_norm(
                x,
                weight_fp32 if self.pass_weight else None,
                self.variance_epsilon,
                self.variance_size_override,
            )
        return ir.ops.fused_add_rms_norm.maybe_inplace(
            x,
            residual,
            weight_fp32 if self.pass_weight_add else None,
            self.variance_epsilon,
            self.variance_size_override,
        )


def _tpu_provider_is_first() -> bool:
    """True if the ``tpu`` provider is the first choice of both RMSNorm ops."""
    return all(
        op.get_priority()[:1] == [TPU_IR_PROVIDER]
        for op in (ir.ops.rms_norm, ir.ops.fused_add_rms_norm)
    )


def arm_fp32_weights(*models: torch.nn.Module | None) -> int:
    """Give every eligible ``TpuRMSNorm`` in ``models`` a float32 weight copy.

    A module is eligible if it has a learned weight (``has_weight``) in a
    16-bit float dtype. A module that is reachable from several models, or
    that is already armed, gets one copy.

    Args:
      models: The loaded models, for example the target model and the draft
        model. ``None`` entries are skipped.

    Returns:
      The number of modules that received a float32 copy in this call. It is 0
      if the ``tpu`` provider is not the first choice of both RMSNorm IR ops.
    """
    if not _tpu_provider_is_first():
        logger.warning(
            "TPU_RMSNORM_FP32_WEIGHT is set, but the RMSNorm IR op priority is "
            "rms_norm=%s, fused_add_rms_norm=%s; the float32 weight copies are "
            "made only when %r is first, so RMSNorm keeps its bfloat16 weights.",
            ir.ops.rms_norm.get_priority(),
            ir.ops.fused_add_rms_norm.get_priority(),
            TPU_IR_PROVIDER,
        )
        return 0

    armed: list[torch.Tensor] = []
    for model in models:
        if model is None:
            continue
        for module in model.modules():
            if not isinstance(module, TpuRMSNorm) or not module.has_weight:
                continue
            if getattr(module, FP32_WEIGHT_BUFFER, None) is not None:
                continue
            weight = module.weight.data
            if weight.dtype not in (torch.bfloat16, torch.float16):
                continue
            weight_fp32 = weight.to(torch.float32)
            module.register_buffer(FP32_WEIGHT_BUFFER, weight_fp32, persistent=False)
            armed.append(weight_fp32)

    if armed and armed[0].device.type == "tpu":
        # Materialize the converts now, so that no forward step runs them.
        from vllm_torchtpu.utils import synchronize_tensors

        synchronize_tensors(armed)
    logger.info("Stored float32 weight copies for %d RMSNorm modules.", len(armed))
    return len(armed)

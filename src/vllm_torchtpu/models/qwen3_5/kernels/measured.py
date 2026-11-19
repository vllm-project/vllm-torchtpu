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
"""Where the fused q/k norm + rope and the fused GDN gated norm + FP8
quantization were measured, and so where they run.

Both kernels ran on every leg of the validation entries: Qwen3.5-397B-A17B on TPU7x,
prefill legs at TP1 and decode legs at TP2 and TP8. The q/k norm + rope
changes the last bits of the forward pass (no bf16 rounding between the norm
and the rope). The gated norm + FP8 kernel rounds the normed rows to bf16, as
RMSNormGated returns them, and quantizes them as the fp8 linear does: on the
host its FP8 rows and scales equal the fp8 linear's bit for bit. On TPU7x,
19 of 20 measured cases gave bit-identical scales; in the other, one row's
scale was one bf16 step off, which moves that row's FP8 values by at most one
float8_e4m3fn step. Both run only there. Every other Qwen3.5 checkpoint,
main's CI model Qwen3.5-35B-A3B among them, every other device and every
other per-device head count keeps main's eager q/k norm and rope and main's
``RMSNormGated`` + FP8 linear.
"""

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.qwen3_5 import hooks, settings

logger = init_logger(__name__)

# (query heads, kv heads) per device of the 397B attention layers at TP1, TP2
# and TP8 (32 query and 2 kv heads in all).
ATTENTION_HEADS = frozenset({(32, 2), (16, 1), (4, 1)})

# Value heads per device of the 397B GDN layers at TP1, TP2 and TP8 (64 in all).
GDN_VALUE_HEADS = frozenset({64, 32, 8})


def is_measured_checkpoint(model_config) -> bool:
    """Qwen3.5-397B-A17B's checkpoint shape on a TPU7x device: the calibrated
    shape of ``hooks._is_calibrated`` (the KV budget's rule), which reads the
    device only once the shape matches."""
    return hooks._is_calibrated(model_config)


def is_measured_model(model_config) -> bool:
    """Whether the engine serves the model and device the kernels were
    measured on: a Qwen3.5 architecture (``settings.is_qwen3_5``) with
    :func:`is_measured_checkpoint`. A config without ``architectures`` (the
    stand-ins callers hand the patch registry) is not."""
    if getattr(model_config, "architectures", None) is None:
        return False
    if not settings.is_qwen3_5(model_config):
        return False
    if is_measured_checkpoint(model_config):
        return True
    logger.info_once(
        "Qwen3.5 fused q/k norm + rope and fused GDN gated norm + FP8 "
        "quantization not used: they were measured for Qwen3.5-397B-A17B on "
        "TPU7x only; this engine keeps the eager q/k norm and rope and the "
        "RMSNormGated + FP8 linear."
    )
    return False


def maybe_install_fused_norm_kernels(model_config) -> bool | None:
    """Qwen3.5 rule: the engine the kernels were measured on
    (:func:`is_measured_model`) runs the fused q/k norm + rope
    (``fused_qk_norm_rope.install_qk_norm_rope_patch``) and routes its loaded
    GDN layers through the fused gated norm + FP8 quantization
    (``gated_norm_quant_op.install_after_loading``).

    Registered as a model-specific patch (``patch_registry``). Returns False
    until such a model config is available, so other models, including
    Qwen3-Next, which shares ``Qwen3NextAttention`` and the GDN layer class,
    and every other Qwen3.5 checkpoint or device, never install either and
    keep vLLM's eager q/k norm and rope and ``RMSNormGated`` + FP8 linear.
    """
    if not is_measured_model(model_config):
        return False
    # Imported here: both modules import this one for the rule.
    from vllm_torchtpu.models.qwen3_5.kernels import (
        fused_qk_norm_rope,
        gated_norm_quant_op,
    )

    fused_qk_norm_rope.install_qk_norm_rope_patch()
    gated_norm_quant_op.install_after_loading()
    logger.info(
        "Applied TPU patch: Qwen3.5 fused q/k norm + rope and fused GDN gated "
        "norm + FP8 quantization."
    )
    return None

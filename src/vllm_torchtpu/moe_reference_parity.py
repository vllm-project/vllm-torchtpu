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
"""MoE reference-parity settings, read from `--additional-config`.

RL and distillation workloads that compare TPU rollouts with an unfused
bfloat16 MoE (the GPU vLLM path) turn these on. Each changes one rounding
point and both are off by default. Remove a setting once full-model evals
make its rounding the default.
"""

import dataclasses
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Round the gmm_v2-fused SwiGLU gate, up and silu(gate) to bfloat16, as
# upstream SiluAndMul does ("silu" layers only).
SILU_BF16_KEY = "tpu_moe_silu_bf16"
# Keep the top-k routing weights and the expert combine in float32 instead of
# the activation dtype.
TOPK_WEIGHTS_FLOAT32_KEY = "tpu_moe_topk_weights_float32"


@dataclasses.dataclass(frozen=True)
class MoEReferenceParity:
    silu_bf16: bool = False
    topk_weights_float32: bool = False


def get_moe_reference_parity(vllm_config: "VllmConfig | None") -> MoEReferenceParity:
    """Parse both settings; a non-bool value raises."""
    if vllm_config is None:
        return MoEReferenceParity()
    additional_config = cast("dict[str, Any]", vllm_config.additional_config)
    values = {}
    for field, key in (
        ("silu_bf16", SILU_BF16_KEY),
        ("topk_weights_float32", TOPK_WEIGHTS_FLOAT32_KEY),
    ):
        value = additional_config.get(key, False)
        if not isinstance(value, bool):
            raise ValueError(
                f"additional_config[{key!r}] must be a bool, got {value!r}"
            )
        values[field] = value
    return MoEReferenceParity(**values)

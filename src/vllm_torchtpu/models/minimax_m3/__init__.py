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

"""MiniMax M3 on TPU.

Kept import-free: `register` runs at plugin load, before vLLM selects a
model, so the model modules load only when this architecture is served.
"""

_ARCHITECTURES = (
    "MiniMaxM3SparseForCausalLM",
    # The released checkpoint declares the vision-language arch; serve it
    # with the text-only model.
    "MiniMaxM3SparseForConditionalGeneration",
)
_MODEL = "vllm_torchtpu.models.minimax_m3.model:MiniMaxM3SparseForCausalLM"


def register() -> None:
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from vllm_torchtpu.layers.adapter.quantization import register_tpu_quant_config
    from vllm_torchtpu.models.minimax_m3.hooks import TPUMiniMaxM3Config
    from vllm_torchtpu.models.minimax_m3.quant import (
        QUANT_METHOD,
        MiniMaxM3QuantConfig,
    )
    from vllm_torchtpu.offload.multi_shapes_kv import (
        register_multi_shapes_kv_architectures,
    )

    for arch in _ARCHITECTURES:
        ModelRegistry.register_model(arch, _MODEL)
        MODELS_CONFIG_MAP[arch] = TPUMiniMaxM3Config
    register_tpu_quant_config(QUANT_METHOD, MiniMaxM3QuantConfig)
    # The [K | V] and index caches differ in row width, so KV offload
    # registers them array by array.
    register_multi_shapes_kv_architectures(*_ARCHITECTURES)

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
"""Model communication requirements needed before TPU worker initialization.

Keep this query free of model/torch imports: process binding runs before the
accelerator runtime is initialized. EP and DP-sharded embeddings both need a
unified DP world and matching busy/idle forwards; expert ownership is separate.
"""


def requires_dp_collectives(vllm_config) -> bool:
    if vllm_config.parallel_config.enable_expert_parallel:
        return True
    model = vllm_config.model_config
    return (
        model is not None
        and model.architecture
        in ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration")
        and vllm_config.additional_config.get("qwen4_exp", {})
        .get("engram_config", {})
        .get("embedding_across_dp", False)
    )

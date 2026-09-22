# SPDX-License-Identifier: Apache-2.0
"""Model communication requirements needed before TPU worker initialization.

Keep this query free of model/torch imports: process binding runs before the
accelerator runtime is initialized. EP and DP-sharded embeddings both need a
unified DP world and matching busy/idle forwards; expert ownership is separate.
"""


def requires_dp_collectives(vllm_config) -> bool:
    if vllm_config.parallel_config.enable_expert_parallel:
        return True
    model = vllm_config.model_config
    return (model is not None and model.architecture
            in ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration")
            and vllm_config.additional_config.get("qwen4_exp", {}).get(
                "engram_config", {}).get("embedding_across_dp", False))

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
"""Static checks for pipelined prefill across DP engines (TPU_DP_PIPELINE_PREFILL)."""

from typing import Any

from vllm_torchtpu import envs


def validate_dp_pipeline_config(vllm_config: Any) -> None:
    """Refuses configurations the pipelined prefill does not serve; a no-op
    with the flag off."""
    if not envs.TPU_DP_PIPELINE_PREFILL:
        return
    from vllm_torchtpu import utils

    parallel = vllm_config.parallel_config
    dp_size = utils.get_dp_size(parallel)
    unsupported = [
        name
        for name, present in (
            ("expert parallelism", parallel.enable_expert_parallel),
            ("PP", parallel.pipeline_parallel_size > 1),
            ("PCP", parallel.prefill_context_parallel_size > 1),
            ("DCP", parallel.decode_context_parallel_size > 1),
            # A helper's KV pages only hold its own layers.
            ("prefix caching", vllm_config.cache_config.enable_prefix_caching),
            ("speculative decoding", vllm_config.speculative_config is not None),
            ("KV transfer connectors", vllm_config.kv_transfer_config is not None),
        )
        if present
    ]
    if unsupported:
        raise ValueError(
            f"TPU_DP_PIPELINE_PREFILL does not support {', '.join(unsupported)}"
        )
    scheduler = vllm_config.scheduler_config
    if scheduler.max_num_scheduled_tokens not in (
        None,
        scheduler.max_num_batched_tokens,
    ):
        # Pipelined chunks are full chunks of the token budget (the runner
        # checks at startup that the largest token bucket is that budget).
        raise ValueError(
            "TPU_DP_PIPELINE_PREFILL needs --max-num-scheduled-tokens unset or "
            f"equal to --max-num-batched-tokens ({scheduler.max_num_batched_tokens})"
        )
    stages = envs.TPU_DP_PIPELINE_STAGES
    if not 2 <= stages <= dp_size:
        raise ValueError(
            f"TPU_DP_PIPELINE_STAGES={stages} must be between 2 and the "
            f"data-parallel size {dp_size}"
        )
    from vllm_torchtpu.distributed.dp_pipeline.plan import (
        parse_splits,
        parse_stage_tokens,
    )

    # Refused at startup rather than at the first long prompt.
    parse_splits(envs.TPU_DP_PIPELINE_SPLITS)
    parse_stage_tokens(envs.TPU_DP_PIPELINE_STAGE_TOKENS)

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
"""Direct token-window micro-batching instrumentation with XLA optimization barriers."""

from vllm_torchtpu.models.microbatch.fence import (
    FenceMode,
    fence_windows,
    microbatch_fence,
    microbatch_fence_pair,
    parse_fence_mode,
)
from vllm_torchtpu.models.microbatch.metadata import (
    bind_step_inputs_for_trace_alignment,
    slice_attention_metadata,
    slice_forward_metadata,
    slice_padding_state,
)
from vllm_torchtpu.models.microbatch.qwen import (
    MicroBatchConfig,
    ScheduleMode,
    can_split_moe_stages,
    install_microbatch,
    maybe_install_microbatch,
    parse_schedule_mode,
    patch_qwen_microbatch,
    uninstall_microbatch,
    unsupported_config_reasons,
)

__all__ = [
    "FenceMode",
    "MicroBatchConfig",
    "ScheduleMode",
    "bind_step_inputs_for_trace_alignment",
    "can_split_moe_stages",
    "fence_windows",
    "install_microbatch",
    "maybe_install_microbatch",
    "microbatch_fence",
    "microbatch_fence_pair",
    "parse_fence_mode",
    "parse_schedule_mode",
    "patch_qwen_microbatch",
    "slice_attention_metadata",
    "slice_forward_metadata",
    "slice_padding_state",
    "uninstall_microbatch",
    "unsupported_config_reasons",
]

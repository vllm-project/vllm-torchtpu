# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
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
"""Host-side agentic serving telemetry for vllm-torchtpu."""

from vllm_torchtpu.telemetry.eviction_tracker import (
    BlockEvictionTracker,
    TpuPaddingCounterTracker,
    compute_request_padding_tokens,
    compute_tpu_bucket_padding_tokens,
    extract_hardware_span_attributes,
)
from vllm_torchtpu.telemetry.padding_metrics import (
    finish_tpu_padding_warmup,
    record_tpu_round_padding,
    register_tpu_padding_collector,
)
from vllm_torchtpu.telemetry.token_breakdown import (
    CATEGORIES,
    BoundedLRU,
    ChatTurnMarkers,
    SessionTurnTracker,
    TokenBreakdown,
    compute_rolling_block_hashes,
    resolve_turn_markers_from_tokenizer,
)
from vllm_torchtpu.telemetry.vllm_hooks import (
    AGENT_TRACE_HEADERS,
    SPAN_ATTRIBUTE_KEYS,
    AgenticTelemetryService,
    extract_agent_headers,
    get_telemetry_service,
    install_agentic_telemetry_patches,
    patch_block_pool,
    patch_output_processor,
    patch_trace_headers,
    reset_telemetry_service,
    sync_block_size_from_config,
)

__all__ = [
    "AGENT_TRACE_HEADERS",
    "AgenticTelemetryService",
    "BlockEvictionTracker",
    "BoundedLRU",
    "CATEGORIES",
    "ChatTurnMarkers",
    "SPAN_ATTRIBUTE_KEYS",
    "SessionTurnTracker",
    "TokenBreakdown",
    "TpuPaddingCounterTracker",
    "compute_request_padding_tokens",
    "compute_rolling_block_hashes",
    "compute_tpu_bucket_padding_tokens",
    "extract_agent_headers",
    "extract_hardware_span_attributes",
    "finish_tpu_padding_warmup",
    "get_telemetry_service",
    "install_agentic_telemetry_patches",
    "patch_block_pool",
    "patch_output_processor",
    "patch_trace_headers",
    "record_tpu_round_padding",
    "register_tpu_padding_collector",
    "reset_telemetry_service",
    "resolve_turn_markers_from_tokenizer",
    "sync_block_size_from_config",
]

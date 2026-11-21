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
from vllm_torchtpu.telemetry.token_breakdown import (
    CATEGORIES,
    BoundedLRU,
    ChatTurnMarkers,
    SessionTurnTracker,
    TokenBreakdown,
    compute_rolling_block_hashes,
    resolve_turn_markers_from_tokenizer,
)

__all__ = [
    "BlockEvictionTracker",
    "BoundedLRU",
    "CATEGORIES",
    "ChatTurnMarkers",
    "SessionTurnTracker",
    "TokenBreakdown",
    "TpuPaddingCounterTracker",
    "compute_request_padding_tokens",
    "compute_rolling_block_hashes",
    "compute_tpu_bucket_padding_tokens",
    "extract_hardware_span_attributes",
    "resolve_turn_markers_from_tokenizer",
]

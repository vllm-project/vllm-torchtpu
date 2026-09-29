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
"""Compatibility shim redirecting to structured_spec_decoding."""

from vllm_torchtpu.structured_spec_decoding import (
    DRAFT_FLAG_EMPTY,
    DRAFT_FLAG_GRAMMAR,
    DRAFT_FLAG_MODEL,
    CompositeDraftPlan,
    CompositeDraftPlanner,
    JumpStats,
    propose_jump_topup,
    propose_jump_topup_with_flags,
    validate_draft_row_invariant,
)
from vllm_torchtpu.structured_spec_decoding import (
    patch_vllm_grammar_jump_decoding as _patch_vllm_grammar_jump_decoding,
)
from vllm_torchtpu.structured_spec_decoding import (
    topup_running_requests as _topup_running_requests,
)
from vllm_torchtpu.structured_spec_decoding import (
    topup_scheduled_spec_decode_tokens as _topup_scheduled_spec_decode_tokens,
)
from vllm_torchtpu.structured_spec_decoding.patches.scheduler_patch import (
    _STATS,
    get_stats,
)

__all__ = [
    "DRAFT_FLAG_EMPTY",
    "DRAFT_FLAG_MODEL",
    "DRAFT_FLAG_GRAMMAR",
    "validate_draft_row_invariant",
    "JumpStats",
    "propose_jump_topup",
    "propose_jump_topup_with_flags",
    "CompositeDraftPlan",
    "CompositeDraftPlanner",
    "_topup_running_requests",
    "_topup_scheduled_spec_decode_tokens",
    "_patch_vllm_grammar_jump_decoding",
    "_STATS",
    "get_stats",
]

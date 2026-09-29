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
"""Public API and exports for the structured speculative decoding subsystem."""

from vllm_torchtpu.structured_spec_decoding.constants import (
    DRAFT_FLAG_EMPTY,
    DRAFT_FLAG_GRAMMAR,
    DRAFT_FLAG_MODEL,
)
from vllm_torchtpu.structured_spec_decoding.core.draft_planner import (
    CompositeDraftPlan,
    CompositeDraftPlanner,
)
from vllm_torchtpu.structured_spec_decoding.core.fsm_bridge import (
    JumpStats,
    propose_jump_topup,
    propose_jump_topup_with_flags,
)
from vllm_torchtpu.structured_spec_decoding.core.invariants import (
    validate_draft_row_invariant,
)
from vllm_torchtpu.structured_spec_decoding.execution.metadata_builder import (
    SpecDecodeMetadataBuilder,
)
from vllm_torchtpu.structured_spec_decoding.patches.scheduler_patch import (
    patch_vllm_grammar_jump_decoding,
    topup_running_requests,
    topup_scheduled_spec_decode_tokens,
)
from vllm_torchtpu.structured_spec_decoding.types import SpecDecodeMetadata

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
    "SpecDecodeMetadata",
    "SpecDecodeMetadataBuilder",
    "topup_running_requests",
    "topup_scheduled_spec_decode_tokens",
    "patch_vllm_grammar_jump_decoding",
]

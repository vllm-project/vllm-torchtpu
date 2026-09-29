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
"""Invariant validators for structured speculative decoding drafts."""

from vllm_torchtpu.structured_spec_decoding.constants import (
    DRAFT_FLAG_EMPTY,
    DRAFT_FLAG_GRAMMAR,
    DRAFT_FLAG_MODEL,
)


def validate_draft_row_invariant(flags: list[int]) -> bool:
    """Verifies that a proposed draft slot flag row satisfies the kernel contract.

    Contract:
    1. Every entry must be a valid flag (0=EMPTY, 1=MODEL, 2=GRAMMAR).
    2. EMPTY (0) entries must form a strict suffix: once an EMPTY slot is seen,
       all subsequent slots in the row must also be EMPTY (no holes).
    3. Any interleaving of MODEL (1) and GRAMMAR (2) before the EMPTY suffix
       is permitted.
    """
    seen_empty = False
    for f in flags:
        if f not in (DRAFT_FLAG_EMPTY, DRAFT_FLAG_MODEL, DRAFT_FLAG_GRAMMAR):
            return False
        if f == DRAFT_FLAG_EMPTY:
            seen_empty = True
        elif seen_empty:
            return False
    return True

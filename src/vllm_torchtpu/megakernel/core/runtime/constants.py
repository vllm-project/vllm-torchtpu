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
"""Runtime constants for bucket policy and step eligibility."""

from __future__ import annotations

from .contract import PAR_DP_LOCKSTEP

DEFAULT_BUCKETS: tuple[int, ...] = (8, 16, 32)
DEFAULT_VERIFY_BUCKETS: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64)
MTP_VERIFY_TOKENS: int = 4
MTP_SPECULATIVE_TOKENS: int = MTP_VERIFY_TOKENS - 1
MAX_BUCKET_ROWS: int = 64
MIN_VERIFY_TOKENS: int = 2
FALSE_STRS: tuple[str, ...] = ("0", "false", "no", "off", "none", "")

STARTUP_FLAG_RULES: tuple[tuple[str, str], ...] = (
    ("unaligned_cache_layout", "requires unified KV and recurrent cache layout"),
    (
        "multimodal_inputs",
        "requires text-only token IDs (multimodal inputs unsupported)",
    ),
    ("lora_enabled", "does not support LoRA adapters"),
    ("pipeline_parallel", "requires single-stage pipeline execution"),
    ("context_parallel", "does not support context parallelism"),
    (PAR_DP_LOCKSTEP, "does not support data-parallel lockstep"),
    ("aux_hidden_states", "does not support auxiliary hidden-state capture"),
)

__all__ = [
    "DEFAULT_BUCKETS",
    "DEFAULT_VERIFY_BUCKETS",
    "FALSE_STRS",
    "MAX_BUCKET_ROWS",
    "MIN_VERIFY_TOKENS",
    "MTP_SPECULATIVE_TOKENS",
    "MTP_VERIFY_TOKENS",
    "STARTUP_FLAG_RULES",
]

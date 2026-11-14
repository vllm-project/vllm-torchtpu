# Copyright 2025 Google LLC
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
"""Sequence-parallel linear and normalization adapter helpers for TPU."""

import sys

from vllm_torchtpu.layers.vllm import sequence_parallel as _sp
from vllm_torchtpu.layers.vllm.sequence_parallel import (
    _ACTIVE, check_buckets, enabled, fp8_gather, is_active, min_rows,
    rows_are_replicated, rows_are_sliced, set_active,
    step_is_sequence_parallel)

__all__ = [
    "_ACTIVE",
    "check_buckets",
    "enabled",
    "fp8_gather",
    "is_active",
    "min_rows",
    "rows_are_replicated",
    "rows_are_sliced",
    "set_active",
    "step_is_sequence_parallel",
]

sys.modules[__name__] = _sp

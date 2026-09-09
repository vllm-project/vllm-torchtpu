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
"""DCP (Decode Context Parallelism) process-group accessors.

Mirrors the role of distributed/pcp.py for PCP: a narrow, safe accessor
layer every call site shares, rather than each re-deriving group state
independently.

"""

from __future__ import annotations

from typing import Any


def get_dcp_group() -> Any | None:
    """Return vLLM's DCP GroupCoordinator, or None if not initialized."""
    try:
        from vllm.distributed.parallel_state import get_dcp_group as _get_group
        return _get_group()
    except (AssertionError, ImportError):
        return None


def get_dcp_rank() -> int:
    group = get_dcp_group()
    return 0 if group is None else int(group.rank_in_group)


def get_dcp_world_size() -> int:
    group = get_dcp_group()
    return 1 if group is None else int(group.world_size)

# Copyright 2026 The vLLM Authors.
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

import logging
import os
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any, Optional, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


def parallel_precompile(
    items: Sequence[T],
    compile_fn: Callable[[T], Any],
    *,
    poll_cache_fn: Optional[Callable[[T], Optional[str]]] = None,
    poll_timeout_s: float = 300.0,
    poll_interval_s: float = 0.5,
) -> None:
    """Generic 2-Phase Disjoint Parallel Precompilation across multi-DP workers.

    Phase 1 (Strict Disjoint Cold Compile):
      Rank r exclusively cold-compiles items where: index % world_size == r.
      100% eliminates duplicate C++ XLA / PyTorch Dynamo compiles across workers.

    Phase 2 (Ring-Rotated Peer Cache Fast-Load):
      Rank r iterates through remaining items rotated by (-local_rank).
      Optionally polls for peer disk artifacts before fast-loading into memory.

    Args:
      items: Sequence of items (e.g. RangeEntry objects, shape tuples, int buckets)
      compile_fn: Single-item compilation callable.
      poll_cache_fn: Optional callable returning expected on-disk artifact path.
      poll_timeout_s: Maximum seconds to wait for peer artifact in Phase 2.
      poll_interval_s: Sleep interval between polls in seconds.
    """
    total_len = len(items)
    if total_len == 0:
        return

    parallel_enabled = bool(
        os.environ.get("TPU_PARALLEL_PRECOMPILE", "0") in ("1", "true",
                                                           "True"))
    world_size = int(
        os.environ.get("WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "1")))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not parallel_enabled or world_size <= 1:
        for item in items:
            compile_fn(item)
        return

    # Phase 1: Compile own disjoint partition exclusively
    my_assigned_indices = [
        i for i in range(total_len)
        if i % world_size == (local_rank % world_size)
    ]
    for idx in my_assigned_indices:
        compile_fn(items[idx])

    # Phase 2: Stagger remaining indices in rotated order to prevent lock contention
    remaining_indices = [
        i for i in range(total_len) if i not in my_assigned_indices
    ]
    d = deque(remaining_indices)
    d.rotate(-local_rank)

    for idx in d:
        item = items[idx]
        if poll_cache_fn:
            try:
                expected_file = poll_cache_fn(item)
                if expected_file and not os.path.exists(expected_file):
                    max_polls = max(1, int(poll_timeout_s / poll_interval_s))
                    for _ in range(max_polls):
                        if os.path.exists(expected_file):
                            break
                        time.sleep(poll_interval_s)
            except Exception as e:
                logger.debug("Failed polling peer cache for item %s: %s", item,
                             e)

        compile_fn(item)

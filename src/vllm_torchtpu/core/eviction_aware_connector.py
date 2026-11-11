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
"""KV connectors that act on the scheduler's eviction order.

A host-offload connector that stores chunks only when HBM is about to drop
them (write-back) needs to see which cached blocks the allocator hands out
next. The scheduler gives such a connector its block pool once at
construction; the connector only reads the free queue and never frees or
allocates.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EvictionAwareConnector(Protocol):
    def attach_block_pool(self, block_pool: Any) -> None:
        """Called once by the scheduler with its `BlockPool`."""
        ...

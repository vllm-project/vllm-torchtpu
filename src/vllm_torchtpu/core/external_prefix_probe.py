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
"""Side-effect-free probe of the prompt prefix a KV connector holds.

Admission policies (`PrefixAwareScheduler`) rank waiting requests by how much
of their prompt is already cached. A KV connector with its own tier (host
offload) can serve part of that prompt too, but its
`get_num_new_matched_tokens` is a commitment step: it records per-request
lookup state, may pin entries and starts async lookups. A connector that
implements this protocol answers the ranking question without any of that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vllm.v1.request import Request


@runtime_checkable
class ExternalPrefixProbe(Protocol):
    """Connector-side estimate of loadable prompt tokens past a local hit."""

    def peek_external_prefix_tokens(
        self, request: Request, num_local_tokens: int
    ) -> int:
        """Tokens past `num_local_tokens` the connector could load for `request`.

        Must not pin, touch (reorder eviction), start transfers or change any
        per-request state. An estimate is fine: callers use it for ordering
        only. Return 0 when unknown.
        """
        ...
